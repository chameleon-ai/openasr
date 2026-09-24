//! Whisper DTW word-timestamp refinement, post-decode and ggml-free.
//!
//! The ggml executor runs one decode and records one cross-attention row per
//! generated token; this module turns those rows, plus the 16 kHz / 20 ms RMS
//! envelope of the request audio, into word windows. Per timestamp-bracketed
//! band the pipeline is: DTW alignment of token centers -> the pre-fold
//! reanchor that pulls word-final punctuation off pauses -> the center fold
//! into word windows -> the per-word span cap -> the onset refiner -> the
//! offset refiner -> the symmetric pad. That order is what the test suite
//! validates end to end; do not reorder passes without rerunning it.
//!
//! Every function here is pure over `f32` and token slices (no graph state,
//! no executor internals). The call sites live in `ggml_executor.rs`; see the
//! `WHISPER_DTW_*` constants (each with an `OPENASR_WHISPER_DTW_*` env
//! override) for the tunables.

use crate::models::decode_policy_component_registry::BuiltinDecodePolicySeq2SeqTextPostprocessKind;
use crate::models::seq2seq_dtw_alignment::{
    dtw_align_token_frames, speech_band_from_rows, speech_frame_bounds, token_text_carries_speech,
    whisper_timestamp_frame,
};
use crate::models::seq2seq_word_timestamps::{
    MIDPOINT_BOUNDARY_FRACTION, NO_ONSET_LEAD, Seq2SeqTokenTime, han_script_boundary_before,
    seq2seq_word_timestamps_from_token_times,
};
use crate::models::text_prefix::common_prefix_len;
use crate::{GgmlAsrExecutionOptions, GgmlAsrPreparedAudioView};

use super::ggml_executor::{WhisperGeneratedTokenAlignment, WhisperGgmlExecutorError};
use super::mel::{WHISPER_HOP_LENGTH, WHISPER_SAMPLE_RATE_HZ};
use super::tokenizer::WhisperTokenizer;

pub(crate) fn audio_duration_seconds(prepared_audio: &GgmlAsrPreparedAudioView) -> f32 {
    prepared_audio.samples_f32.len() as f32 / prepared_audio.sample_rate_hz.max(1) as f32
}

/// Per-frame RMS envelope of the request audio on a 0.02-s grid -- the same
/// 16 kHz / 0.02-s frame units the DTW word times are expressed in -- for the
/// whisper cross-attention word-timing path. `None` when that path is not
/// active (no word timestamps requested, or the diarization-forced post-hoc
/// anchor mode which never refines onsets) so callers can skip the refinement
/// cleanly.
///
/// `samples_f32` is the prepared 16 kHz mono PCM that the mel frontend already
/// consumes; the DTW word times are absolute seconds from the start of this
/// buffer. Each envelope entry is the square root of the mean square of the
/// next 320 samples (0.02 s at 16 kHz). A clip fully below the f32 dynamic
/// range (all zeros) yields a median of zero, and the onset-refinement
/// predicate refuses to fire for a non-positive floor -- so all-silent input
/// is left exactly as the fold produced it, no words move.
pub(crate) fn whisper_dtw_word_audio_rms_frames(
    audio: &GgmlAsrPreparedAudioView,
    request_options: &GgmlAsrExecutionOptions,
) -> Option<Vec<f32>> {
    if whisper_word_timestamp_mode(request_options) != WhisperWordTimestampMode::CrossAttention {
        return None;
    }
    if audio.sample_rate_hz != WHISPER_SAMPLE_RATE_HZ {
        // This path assumes 16 kHz mono (as the mel frontend already does); a
        // different rate means the caller is misusing the prepared audio view.
        return None;
    }
    let samples = audio.samples_f32.as_ref();
    if samples.is_empty() {
        return None;
    }
    let mut out = Vec::with_capacity(samples.len().div_ceil(WHISPER_DTW_ENVELOPE_FRAME_COUNT));
    for frame_start in (0..samples.len()).step_by(WHISPER_DTW_ENVELOPE_FRAME_COUNT) {
        let frame_end = (frame_start + WHISPER_DTW_ENVELOPE_FRAME_COUNT).min(samples.len());
        let mut sum = 0.0_f64;
        for &sample in &samples[frame_start..frame_end] {
            if !sample.is_finite() {
                return None;
            }
            let value = f64::from(sample);
            sum += value * value;
        }
        out.push((sum / (frame_end - frame_start) as f64).sqrt() as f32);
    }
    Some(out)
}

/// How a whisper decode derives word timestamps for a request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WhisperWordTimestampMode {
    /// No word timestamps requested.
    Off,
    /// User-requested word timestamps: collect per-token cross-attention
    /// during decode (higher fidelity, but switches the decode path — cross
    /// flash attention off, cross-attention collection on — so the transcript
    /// can differ from a plain run via FP accumulation differences).
    CrossAttention,
    /// Word timestamps forced on solely as diarization anchors: keep the
    /// decode path byte-identical to a non-diarized run and derive word
    /// anchors post hoc from the generated tokens (the same path the whisper
    /// serve-batch decode always uses).
    PostHocAnchors,
}

pub(crate) fn whisper_word_timestamp_mode(
    request_options: &GgmlAsrExecutionOptions,
) -> WhisperWordTimestampMode {
    if !request_options.word_timestamps {
        WhisperWordTimestampMode::Off
    } else if request_options.word_timestamps_forced_for_diarization {
        WhisperWordTimestampMode::PostHocAnchors
    } else {
        WhisperWordTimestampMode::CrossAttention
    }
}

/// Decoder-graph `(use_cross_flash_attention, collect_cross_attention)` flags
/// for a request. Only user-requested word timestamps (`CrossAttention`) may
/// alter the decode path; diarization-forced anchors must leave both flags
/// exactly as a request without word timestamps would.
pub(crate) fn whisper_decoder_cross_attention_flags(
    cross_flash_attention_enabled: bool,
    request_options: &GgmlAsrExecutionOptions,
) -> (bool, bool) {
    let collect_cross_attention =
        whisper_word_timestamp_mode(request_options) == WhisperWordTimestampMode::CrossAttention;
    (
        cross_flash_attention_enabled && !collect_cross_attention,
        collect_cross_attention,
    )
}

/// Maximal runs of consecutive text (non-timestamp) tokens in the generated
/// stream, as inclusive index ranges into `token_ids`: the words between two
/// decoded timestamp tokens form one segment. The timestamp tokens separating
/// the runs supply each segment's `[start, end]` DTW band.
pub(crate) fn text_token_runs(
    token_ids: &[u32],
    is_timestamp: &dyn Fn(u32) -> bool,
) -> Vec<(usize, usize)> {
    let mut runs = Vec::new();
    let mut index = 0usize;
    let len = token_ids.len();
    while index < len {
        if is_timestamp(token_ids[index]) {
            index += 1;
            continue;
        }
        let lo = index;
        while index < len && !is_timestamp(token_ids[index]) {
            index += 1;
        }
        let hi = index.saturating_sub(1);
        if lo <= hi {
            runs.push((lo, hi));
        }
    }
    runs
}

/// The window-absolute frame band a text-token run should DTW onto, from the
/// nearest `<|start|>` / `<|end|>` timestamp tokens that bracket it (mirrors
/// whisper-timestamped's per-segment `weights[..., start: end]` slice). The
/// start is the nearest timestamp token at or before the run, the end the first
/// after it. A missing bound degrades to the window edge (frame 0 / final
/// frame) so a leading-leak or an early-EOT tail still yields a band rather
/// than dropping the segment. Returns `None` only when the pack has no
/// timestamp vocabulary or the band is empty.
pub(crate) fn run_frame_bounds(
    lo: usize,
    hi: usize,
    token_ids: &[u32],
    timestamp_begin: Option<u32>,
    frame_resolution: usize,
) -> Option<(usize, usize)> {
    let timestamp_begin = timestamp_begin?;
    let frame_of = |id: u32| whisper_timestamp_frame(id, timestamp_begin).min(frame_resolution);
    // The start is the nearest decoded timestamp at or before the run (frame 0
    // when the run starts the stream, e.g. a leading <|0.00|> leak); the end is
    // the first decoded timestamp at or after the run's last text token
    // (the final frame when EOT ends the stream before an end timestamp).
    let start_frame = token_ids
        .get(..=lo)
        .and_then(|window| window.iter().rev().find(|&&id| id >= timestamp_begin))
        .copied()
        .map(frame_of)
        .unwrap_or(0);
    let end_frame = token_ids
        .get(hi..)
        .and_then(|window| window.iter().find(|&&id| id >= timestamp_begin).copied())
        .map_or(frame_resolution, frame_of);
    if end_frame <= start_frame {
        return None;
    }
    Some((start_frame, end_frame.min(frame_resolution)))
}

/// Where the boundary between two consecutive DTW words lands, as a fraction of
/// the gap between their centers: `prev + fraction * (this - prev)`. With raw
/// DTW span tiling each word start is the next token's entry frame (where its
/// attention peaks), which places a word start a full onset past its true
/// speech beginning. Folding the entry frames back into word centers and splitting
/// the gap at this fraction puts the start before the center, at the true onset.
const WHISPER_DTW_BOUNDARY_FRACTION: f32 = 0.45;

/// Seconds by which each DTW center is placed earlier before the boundary
/// split, so a word's start lands at its real speech onset rather than a hair
/// past it. Whisper's DTW biases the path to start early (the first cost cell
/// is pulled to the global minimum), so the per-token entry frame the fold
/// treats as a center sits only slightly behind the true onset: a small
/// constant lead (one DTW frame at 0.02s/frame would overshoot) is the best
/// trade across band densities -- the test corpus measured a flat baseline as
/// a cleaner whole-suite TempErr mean than any density-scaled curve, because a
/// per-band (density-varying) shift creates segment-to-segment discontinuities
/// that the per-word affine fit cannot absorb.
const WHISPER_DTW_ONSET_LEAD_SECONDS: f32 = 0.05;

/// The onset lead in use, honoring the deployment env override so a
/// deployment can retune it per corpus without a rebuild. Each call re-reads
/// the environment and falls back to the compiled default when unset or
/// unparsable, so a bare environment is byte-identical to historical behavior.
fn whisper_dtw_onset_lead() -> f32 {
    std::env::var("OPENASR_WHISPER_DTW_LEAD_BASE_SECONDS")
        .ok()
        .and_then(|raw| raw.parse::<f32>().ok())
        .unwrap_or(WHISPER_DTW_ONSET_LEAD_SECONDS)
}

/// Maximum duration, in seconds, a single DTW word may keep. The center fold
/// gives each word the boundaries on both sides of its center, so a word next
/// to a real pause extends halfway across it; on a long band the last token can
/// also absorb the run to the band end. The cap is set well above any plausible
/// spoken word (longest legitimate words observed in the test clips are ~1.7s)
/// and far below the runaway regime; only the tail is trimmed, never the start.
const WHISPER_DTW_MAX_WORD_SPAN_SECONDS: f32 = 1.5;

/// How far ahead of the decoded `<|start|>` bound a run's measured content onset
/// must sit before the decoded bound is treated as bracketing leading silence
/// and the first word's start is advanced to that onset (the leading-silence
/// onset advance). Whisper's `<|start|>` routinely leaks slightly early -- well
/// inside the band margin of a true gap -- so a sub-margin advance would only
/// smear a segment that was already fine; only a lead at least this long beyond
/// the bound is a real leaked silence worth correcting. Tuned over the test
/// corpus at the margin (0.2s), where the leading-silence leaks sit well past
/// the margin while normal `<|start|>` jitter sits at or inside it.
const WHISPER_DTW_LEAD_SILENCE_ADVANCE_MIN_GAP_SECONDS: f32 = 0.2;

/// The leading-silence advance gap threshold, honoring a deployment env
/// override so a tuning pass can sweep it without a rebuild (see
/// [`WHISPER_DTW_LEAD_SILENCE_ADVANCE_MIN_GAP_SECONDS`]). A bare environment is
/// byte-identical to the constant.
fn whisper_dtw_lead_silence_advance_min_gap_seconds() -> f32 {
    std::env::var("OPENASR_WHISPER_DTW_LEAD_SILENCE_ADVANCE_MIN_GAP_SECONDS")
        .ok()
        .and_then(|raw| raw.parse::<f32>().ok())
        .unwrap_or(WHISPER_DTW_LEAD_SILENCE_ADVANCE_MIN_GAP_SECONDS)
}

/// Whether to advance a run's lead anchor past its decoded band start, and to
/// which frame. Returns `Some(advance_to_frame)` only for a genuine leading-
/// silence leak: the run must start at the window front (`band_start == 0`, i.e.
/// no decoded `<|start|>` before it) AND its measured content onset (`band_front`)
/// must sit at least `min_gap_seconds` ahead of the band start. Otherwise
/// `None` and the decoded bound is kept.
///
/// The `band_start == 0` gate is what keeps this from firing on a mid-run
/// decoded `<|start|>` whose frame merely falls short of the run's own earliest
/// attention peak: there the bound is a real timestamp that can mark a large
/// misalignment (a repeated/leaked word elsewhere in the window), and retargeting
/// the lead word to an unrelated peak would move it further off. The gap gate
/// keeps a real-but-tight leading boundary (sub-margin `<|start|>` jitter)
/// untouched, matching the historical "at-most-a-margin early" tolerance.
fn whisper_dtw_lead_silence_advance_frame(
    band_start: usize,
    band_front: Option<usize>,
    seconds_per_frame: f32,
    min_gap_seconds: f32,
) -> Option<usize> {
    if band_start != 0 {
        return None;
    }
    let front = band_front?;
    let gap_seconds = (front.saturating_sub(band_start) as f32) * seconds_per_frame;
    if gap_seconds >= min_gap_seconds {
        Some(front)
    } else {
        None
    }
}

/// Limit how long a single DTW word may run.
///
/// In the center fold a word's edges are the fractions of the gaps to its
/// neighbours' centers. When a real pause sits between two utterances the word
/// on each side extends into it (measured ~1-2s on the test clips, where the
/// ground truth's longest word is under 1.7s), and a band's final word can run
/// to the band end. The cap trims each word's tail at `start + 1.5s`, well
/// above any plausible spoken word and far below the phantom-tail regime,
/// restoring an explicit gap at the pause while keeping the word's own
/// (center-derived) start untouched.
fn whisper_cap_dtw_word_spans(
    words: Vec<crate::WordTimestamp>,
    seconds_per_frame: f32,
) -> Vec<crate::WordTimestamp> {
    const MAX_SECONDS: f32 = WHISPER_DTW_MAX_WORD_SPAN_SECONDS;
    let limit = MAX_SECONDS.max(seconds_per_frame);
    let mut capped_words = words;
    for word in &mut capped_words {
        let span = word.end - word.start;
        if span > limit {
            word.end = word.start + limit;
        }
    }
    capped_words
}

/// Length of the DTW envelope RMS window, in samples (0.02 s at 16 kHz), the
/// same duration as one 0.02 s/s frame the DTW word times are expressed in.
// `pub(crate)`: the decode-side silence checks (tail-repeat acoustic gate,
// degenerate-tail region) reuse the same 320-sample envelope geometry.
pub(crate) const WHISPER_DTW_ENVELOPE_FRAME_COUNT: usize = 320;

/// Share of a word's front half tolerated above the floor before the word is no
/// longer hollow (a speech onset bleeding into the front half).
const WHISPER_DTW_HOLLOW_FRONT_ACTIVE_MAX: f32 = 0.5;
/// Maximum of the front half may sit, as a fraction of the clip's peak envelope
/// level, before the silence is no longer trusted as a real pause. A music or
/// noise background never reads as digital zero -- its floor is a real level --
/// so without this a quiet pause in a music-backed clip looks hollow and the
/// onset push fires on the music floor, moving a word that was already
/// acceptable. Requiring the front to be within this fraction of the clip's
/// peak isolates true zero-crossing silence from a low floor. Swept over the
/// test corpus: 5% of peak is the largest that stays regression-free -- it
/// tolerates the single offset-bleed frame a fast preceding word leaves in the
/// next word's front (so a word after a brief inter-word gap still gets pulled
/// to its real onset) while every music bed it must reject still sits above it;
/// above ~8% it starts chasing those offsets and drops an InWin overlap on
/// music-backed clips.
const WHISPER_DTW_HOLLOW_FRONT_MAX_PEAK_FRACTION: f64 = 0.05;
/// dB above the clip's own noise floor (the median envelope level) that counts
/// as real speech. Measured in dB over the envelope so it adapts per clip
/// rather than assuming a fixed absolute speech level.
// `pub(crate)`: the decode-side tail-repeat acoustic gate reuses the same
// speech-vs-floor margin.
pub(crate) const WHISPER_DTW_ONSET_FLOOR_MARGIN_DB: f64 = 5.0;
/// Minimum duration of the speech run above the floor that qualifies as the
/// word's onset (expressed in envelope frames of 0.02 s).
// `pub(crate)`: the decode-side silence checks reuse the same sustained-speech
// run length.
pub(crate) const WHISPER_DTW_ONSET_SUSTAIN_FRAMES: usize = 5;
/// Minimum run of silence between the run before a word and its onset, in
/// seconds, so a run that merely touches a brief inter-word glottal gap is not
/// treated as a real pause.
const WHISPER_DTW_ONSET_MIN_SILENCE_S: f32 = 0.03;
/// The onset must sit at least this many seconds after the fold's start, or the
/// push is an adjustment smaller than the fold's own calibration error and the
/// word is left as-is.
const WHISPER_DTW_ONSET_MIN_PUSH_S: f32 = 0.25;
/// Upper bound on how far back into a pause a word's start may be pulled.
const WHISPER_DTW_ONSET_MAX_PUSH_S: f32 = 5.0;

/// Peak-to-median contrast above which a slice's median is a *thin noise
/// floor* rather than a continuous bed, so the silence ceiling may rise off
/// the peak and onto the floor (see [`whisper_dtw_silence_ceiling`]).
/// A dense music bed sits within a few times of its loudest frame (measured
/// <= ~5x peak/median on the test corpus' music clips); a recording with
/// room tone, a distant bed, or a line hum carries speech peaks an order of
/// magnitude above its floor (>= ~10x). In between, neither reading of
/// "quiet" is safe, so the conservative peak fraction is kept.
const WHISPER_DTW_THIN_FLOOR_CONTRAST: f64 = 8.0;

/// How far above the floor (the median envelope level) a single frame in a
/// region may read before the region stops being silence, but only on a
/// thin-floor slice. A linear factor, so 3.0 is just under 5 dB over the
/// floor -- the same margin the speech threshold already applies -- so a
/// region within it is floor noise whose transients peak a few dB over the
/// median, while sustained speech (above the threshold) still fails the
/// mean/fraction hollow tests.
const WHISPER_DTW_THIN_FLOOR_PEAK_OF_MEDIAN: f64 = 3.0;

/// Deployment env override for the thin-floor contrast
/// ([`WHISPER_DTW_THIN_FLOOR_CONTRAST`]); a bare environment falls back to the
/// compiled default, staying byte-identical to it.
fn whisper_dtw_thin_floor_contrast() -> f64 {
    std::env::var("OPENASR_WHISPER_DTW_THIN_FLOOR_CONTRAST")
        .ok()
        .and_then(|raw| raw.parse::<f64>().ok())
        .unwrap_or(WHISPER_DTW_THIN_FLOOR_CONTRAST)
}

/// Deployment env override for the thin-floor floor multiple
/// ([`WHISPER_DTW_THIN_FLOOR_PEAK_OF_MEDIAN`]); a bare environment falls back
/// to the compiled default, staying byte-identical to it.
fn whisper_dtw_thin_floor_peak_of_median() -> f64 {
    std::env::var("OPENASR_WHISPER_DTW_THIN_FLOOR_PEAK_OF_MEDIAN")
        .ok()
        .and_then(|raw| raw.parse::<f64>().ok())
        .unwrap_or(WHISPER_DTW_THIN_FLOOR_PEAK_OF_MEDIAN)
}

/// The level a region may reach before it stops reading as trusted silence.
///
/// The base ceiling is a small fraction of the slice's *peak*
/// ([`WHISPER_DTW_HOLLOW_FRONT_MAX_PEAK_FRACTION`]): a music or noise bed never
/// reads as digital zero, so a region that still carries a substantial fraction
/// of the clip's loudest frame is a bed, not a pause. That reading has one
/// blind spot: on a recording with a *thin* background floor (room tone, a
/// distant bed), the floor's own transient peaks can cross the 5%-of-peak
/// ceiling even though the floor is acoustically silence -- the peak is the
/// speaker's voice, an order of magnitude above it. The slice's own contrast
/// tells the two apart: when the peak rises at least
/// [`WHISPER_DTW_THIN_FLOOR_CONTRAST`] times the floor (the median), the floor
/// is thin and the ceiling rises to
/// [`WHISPER_DTW_THIN_FLOOR_PEAK_OF_MEDIAN`] times the floor. Dense bed slices
/// (peak within a few times of the median) fail the contrast and keep the
/// conservative peak fraction exactly as before.
fn whisper_dtw_silence_ceiling(noise_floor: f64, clip_peak: f64) -> f64 {
    let peak_fraction = clip_peak * WHISPER_DTW_HOLLOW_FRONT_MAX_PEAK_FRACTION;
    let contrast = clip_peak / noise_floor.max(f64::EPSILON);
    if contrast >= whisper_dtw_thin_floor_contrast() {
        peak_fraction.max(noise_floor * whisper_dtw_thin_floor_peak_of_median())
    } else {
        peak_fraction
    }
}

/// Pull a word that the center fold landed in silence to its real audio onset.
///
/// The DTW entry frame the fold treats as a word's center sits where the monotone
/// path first *enters* a token's row. After an intra-segment pause that entry is
/// at the tail of the preceding word or part-way into the pause, not on the next
/// word's audio; the `boundary_fraction` split that follows places the next word's
/// start a full `fraction * gap` before its center -- i.e. a fraction of the pause
/// early, where the audio is silent.
///
/// This pass recovers the onset from the audio when the fold could not: a word
/// whose front half is true silence (its energy only begins later inside its own
/// window) is advanced to that first sustained speech run. Because the fold's
/// adjacent boundaries coincide (the previous word's end equals this word's old
/// start), advancing the start opens a real gap where the pause actually sits
/// instead of smearing the word across it, while leaving the timeline monotone
/// and non-overlapping by construction -- no neighbour is ever touched.
///
/// The speech floor is `10^(margin/20)` times the *median* frame RMS, tracking a
/// quiet recording down to its own level (a high percentile would peg the bar at
/// the loudest peak). Crucially the front must also stay within a small fraction
/// of the clip's *peak*: a music or noise bed never reads as digital zero, so a
/// quiet passage inside a music-backed clip is *not* a trusted pause and a push
/// would only move a word that was already acceptable. That ceiling is what
/// separates a genuine zero-silence pause (fire) from a low music floor (skip).
fn whisper_refine_dtw_word_onsets(
    mut words: Vec<crate::WordTimestamp>,
    audio_rms_frames: Option<&[f32]>,
    duration_s: f32,
) -> Vec<crate::WordTimestamp> {
    let Some(levels) = audio_rms_frames else {
        return words;
    };
    if levels.len() < 4 || duration_s <= 0.0 || words.len() < 2 {
        return words;
    }
    let seconds_per_frame = WHISPER_DTW_ENVELOPE_FRAME_COUNT as f64 / WHISPER_SAMPLE_RATE_HZ as f64;
    let mut ranked: Vec<f64> = levels.iter().map(|sample| f64::from(*sample)).collect();
    ranked.sort_by(f64::total_cmp);
    let noise_floor = ranked[ranked.len() / 2];
    if !(noise_floor > 0.0 && noise_floor.is_finite()) {
        return words;
    }
    let threshold = noise_floor * 10.0_f64.powf(WHISPER_DTW_ONSET_FLOOR_MARGIN_DB / 20.0);
    let clip_peak = *ranked.last().unwrap_or(&0.0);
    if !(clip_peak > 0.0 && clip_peak.is_finite()) {
        return words;
    }
    // A front frame that reads at or above this is not true silence (see
    // [`whisper_dtw_silence_ceiling`]).
    let silence_ceiling = whisper_dtw_silence_ceiling(noise_floor, clip_peak);
    let min_quiet_frames =
        ((WHISPER_DTW_ONSET_MIN_SILENCE_S as f64) / seconds_per_frame).ceil() as usize;
    for word in words.iter_mut().skip(1) {
        let raw_start = f64::from(word.start);
        let raw_end = f64::from(word.end);
        let span = raw_end - raw_start;
        if span < 0.3_f64 {
            continue;
        }
        let start_s = raw_start.max(0.0).min(f64::from(duration_s));
        let end_s = raw_end.max(start_s).min(f64::from(duration_s));
        // A boundary word whose start maps to or past the last envelope frame
        // (common at a longform slice end) must not overrun the frame array;
        // clamp both indices into range, letting the window-length guard below
        // bail the word without refinement rather than panic.
        let last_frame = levels.len() - 1;
        let frame_start = ((start_s / seconds_per_frame) as usize).min(last_frame);
        let frame_end = ((end_s / seconds_per_frame) as usize)
            .min(last_frame)
            .max(frame_start + 1)
            .min(last_frame);
        let window = &levels[frame_start..frame_end + 1];
        if window.len() < 4 {
            continue;
        }
        let window_len = window.len();
        let is_above = |index: usize| f64::from(window[index]) >= threshold;
        let front_len = (window_len / 2).clamp(1, window_len);
        // A hollow word: the front half sits in true silence, not just a quiet
        // passage. Three conditions on the front half of the window:
        //   1. its *mean* level is below the noise floor (not just a fraction of
        //      frames -- a single loud blip in a quiet front must not pass);
        //   2. *no* front frame crosses the silence ceiling (5% of clip peak),
        //      so a music floor never masquerades as a pause (see
        //      [`WHISPER_DTW_HOLLOW_FRONT_MAX_PEAK_FRACTION`]);
        //   3. fewer than half its frames are above the floor (no sustained
        //      speech leaking into the front).
        let front_mean =
            (0..front_len).map(|i| f64::from(window[i])).sum::<f64>() / front_len as f64;
        let front_max = (0..front_len)
            .map(|i| f64::from(window[i]))
            .max_by(f64::total_cmp)
            .unwrap_or(0.0);
        let front_above = (0..front_len).filter(|&i| is_above(i)).count() as f64 / front_len as f64;
        if front_mean >= threshold
            || front_max > silence_ceiling
            || front_above > WHISPER_DTW_HOLLOW_FRONT_ACTIVE_MAX as f64
        {
            continue;
        }
        // The onset: the first speech run (>= sustain frames above the floor)
        // preceded by a quiet run of at least the minimum silence length,
        // inside this word's own window.
        let mut onset_rel: Option<usize> = None;
        let mut index = 0usize;
        while index < window_len && onset_rel.is_none() {
            if is_above(index) {
                let mut run_end = index;
                while run_end + 1 < window_len && is_above(run_end + 1) {
                    run_end += 1;
                }
                let run_len = run_end - index + 1;
                if run_len >= WHISPER_DTW_ONSET_SUSTAIN_FRAMES {
                    let mut quiet = 0usize;
                    let mut probe = index;
                    while probe > 0 && !is_above(probe - 1) {
                        probe -= 1;
                        quiet += 1;
                    }
                    if quiet >= min_quiet_frames {
                        onset_rel = Some(index);
                    }
                }
                index = run_end + 1;
            } else {
                index += 1;
            }
        }
        let Some(rel) = onset_rel else {
            continue;
        };
        let onset_s = ((frame_start + rel) as f64 * seconds_per_frame) as f32;
        let push = onset_s - raw_start as f32;
        // A word's start may only move forward into later audio, never backward.
        // The minimum push guards against a sub-calibration-error wiggle; the
        // maximum caps how deep into a pause we trust the envelope to lead us.
        if !(WHISPER_DTW_ONSET_MIN_PUSH_S..=WHISPER_DTW_ONSET_MAX_PUSH_S).contains(&push) {
            continue;
        }
        word.start = onset_s;
    }
    words
}

/// Share of a word's back half tolerated above the floor before the word is no
/// longer hollow (a speech offset bleeding into the back half).
const WHISPER_DTW_HOLLOW_BACK_ACTIVE_MAX: f32 = 0.5;
/// dB above the clip's own noise floor (the median envelope level) that counts
/// as real speech; the trailing-silence counterpart of
/// [`WHISPER_DTW_ONSET_FLOOR_MARGIN_DB`].
const WHISPER_DTW_OFFSET_FLOOR_MARGIN_DB: f64 = 5.0;
/// Minimum duration of the speech run above the floor that qualifies as the
/// word's offset (expressed in envelope frames of 0.02 s).
const WHISPER_DTW_OFFSET_SUSTAIN_FRAMES: usize = 5;
/// Minimum run of silence between the word's offset and the next run, in
/// seconds, so a run that merely touches a brief inter-word glottal gap is not
/// treated as a real pause.
const WHISPER_DTW_OFFSET_MIN_SILENCE_S: f32 = 0.03;
/// The offset must sit at least this many seconds before the fold's end, or the
/// pull is an adjustment smaller than the fold's own calibration error and the
/// word is left as-is.
const WHISPER_DTW_OFFSET_MIN_PULL_S: f32 = 0.25;
/// Upper bound on how far forward into a pause a word's end may be advanced.
const WHISPER_DTW_OFFSET_MAX_PULL_S: f32 = 5.0;

/// Pull a word that the center fold let run past its speech into the trailing
/// silence back to its real audio offset.
///
/// The mirror of [`whisper_refine_dtw_word_onsets`]: where that pass recovers a
/// word's *start* from a hollow front, this one recovers its *end* from a hollow
/// back. The fold gives a word the boundary a fixed fraction of the way to the
/// next center, so a word beside a real pause extends across part of it and, with
/// the `boundary` pad, its end lands a good deal past the last audible frame of
/// the word.
///
/// A word whose back half is true silence is retreated to its last sustained
/// speech run. Because the fold's seams are continuous only between *words the
/// audio actually carries* (the next word's start is its own audio, set by its own
/// onset refinement), pulling this word's end back opens the real gap where the
/// pause sits instead of smearing this word across it, and never touches the next
/// word -- the timeline stays monotone and non-overlapping by construction.
///
/// As with onsets, the speech floor tracks the clip's own noise level and the
/// back must stay below a small fraction of the clip's *peak*: a music or noise
/// bed never reads as digital zero, so a low passage inside a music-backed clip is
/// not a trusted trailing pause and a pull would only move a word that was already
/// acceptable. The last word is skipped: its true end is the audio end, so the
/// trailing silence after it is the clip's legitimate tail, not a fold leak.
fn whisper_refine_dtw_word_offsets(
    mut words: Vec<crate::WordTimestamp>,
    audio_rms_frames: Option<&[f32]>,
    duration_s: f32,
) -> Vec<crate::WordTimestamp> {
    let Some(levels) = audio_rms_frames else {
        return words;
    };
    if levels.len() < 4 || duration_s <= 0.0 || words.len() < 2 {
        return words;
    }
    let seconds_per_frame = WHISPER_DTW_ENVELOPE_FRAME_COUNT as f64 / WHISPER_SAMPLE_RATE_HZ as f64;
    let mut ranked: Vec<f64> = levels.iter().map(|sample| f64::from(*sample)).collect();
    ranked.sort_by(f64::total_cmp);
    let noise_floor = ranked[ranked.len() / 2];
    if !(noise_floor > 0.0 && noise_floor.is_finite()) {
        return words;
    }
    let threshold = noise_floor * 10.0_f64.powf(WHISPER_DTW_OFFSET_FLOOR_MARGIN_DB / 20.0);
    let clip_peak = *ranked.last().unwrap_or(&0.0);
    if !(clip_peak > 0.0 && clip_peak.is_finite()) {
        return words;
    }
    // A back frame that reads at or above this is not true silence (see
    // [`whisper_dtw_silence_ceiling`], reused as the silence ceiling for the
    // trailing half of a word).
    let silence_ceiling = whisper_dtw_silence_ceiling(noise_floor, clip_peak);
    let min_quiet_frames =
        ((WHISPER_DTW_OFFSET_MIN_SILENCE_S as f64) / seconds_per_frame).ceil() as usize;
    let last_frame = levels.len() - 1;
    // The last word is skipped: its true end is the audio end, so the silence
    // after it is the clip's legitimate tail, not a fold leak.
    let last_index = words.len().saturating_sub(1);
    for (index, word) in words.iter_mut().enumerate() {
        if index >= last_index {
            continue;
        }
        let raw_start = f64::from(word.start);
        let raw_end = f64::from(word.end);
        let span = raw_end - raw_start;
        if span < 0.3_f64 {
            continue;
        }
        let start_s = raw_start.max(0.0).min(f64::from(duration_s));
        let end_s = raw_end.max(start_s).min(f64::from(duration_s));
        // A boundary word whose end maps to or past the last envelope frame must
        // not overrun the array (a word at the audio end); clamp into range and let
        // the window-length guard bail it without refinement rather than panic.
        let frame_start = ((start_s / seconds_per_frame) as usize).min(last_frame);
        let frame_end = ((end_s / seconds_per_frame) as usize)
            .min(last_frame)
            .max(frame_start + 1)
            .min(last_frame);
        let window = &levels[frame_start..frame_end + 1];
        if window.len() < 4 {
            continue;
        }
        let window_len = window.len();
        let is_above = |index: usize| f64::from(window[index]) >= threshold;
        let back_len = (window_len / 2).clamp(1, window_len);
        let back_start = window_len.saturating_sub(back_len);
        // A hollow word: the back half sits in true silence, not just a quiet
        // passage. Three conditions on the back half of the window, mirroring the
        // onset pass's front-half check: (1) its *mean* is below the noise floor;
        // (2) *no* back frame crosses the silence ceiling, so a music floor never
        // masquerades as trailing silence; (3) fewer than half its frames are above
        // the floor (no sustained speech leaking into the back).
        let back_mean = (back_start..window_len)
            .map(|i| f64::from(window[i]))
            .sum::<f64>()
            / back_len as f64;
        let back_max = (back_start..window_len)
            .map(|i| f64::from(window[i]))
            .max_by(f64::total_cmp)
            .unwrap_or(0.0);
        let back_above =
            (back_start..window_len).filter(|&i| is_above(i)).count() as f64 / back_len as f64;
        if back_mean >= threshold
            || back_max > silence_ceiling
            || back_above > WHISPER_DTW_HOLLOW_BACK_ACTIVE_MAX as f64
        {
            continue;
        }
        // The offset: the last speech run (>= sustain frames above the floor)
        // followed by a quiet run of at least the minimum silence length, inside
        // this word's own window; its end is that run's offset frame.
        let mut offset_rel: Option<usize> = None;
        let mut index = Some(window_len - 1);
        while let Some(i) = index {
            if is_above(i) {
                let mut run_start = i;
                while run_start > 0 && is_above(run_start - 1) {
                    run_start -= 1;
                }
                let run_len = i - run_start + 1;
                if run_len >= WHISPER_DTW_OFFSET_SUSTAIN_FRAMES {
                    let mut quiet = 0usize;
                    let mut probe = i;
                    while probe + 1 < window_len && !is_above(probe + 1) {
                        probe += 1;
                        quiet += 1;
                    }
                    if quiet >= min_quiet_frames {
                        offset_rel = Some(i);
                        break;
                    }
                }
                index = run_start.checked_sub(1);
            } else {
                index = i.checked_sub(1);
            }
        }
        let Some(rel) = offset_rel else {
            continue;
        };
        // The run ends at `rel`; one frame past it is where the silence begins.
        let offset_s = ((frame_start + rel + 1) as f64 * seconds_per_frame) as f32;
        let pull = raw_end - f64::from(offset_s);
        // A word's end may only move earlier into prior audio, never past it. The
        // minimum pull guards against a sub-calibration-error wiggle; the maximum
        // caps how deep into a pause we trust the envelope to lead us.
        if !(WHISPER_DTW_OFFSET_MIN_PULL_S..=WHISPER_DTW_OFFSET_MAX_PULL_S).contains(&(pull as f32))
        {
            continue;
        }
        word.end = offset_s;
    }
    words
}

/// Minimum distance, in seconds, between the end of the nearest preceding
/// sustained speech run and a token's center before the token is treated as
/// parked in the pause after its word's own audio (the word-final punctuation
/// reanchor, see [`whisper_reanchor_dtw_token_centers`]). Below it, the center
/// is inside the fold's own calibration error and stays where the path put it.
const WHISPER_DTW_REANCHOR_MIN_GAP_SECONDS: f32 = 0.15;

/// Maximum distance, in seconds, the reanchor may pull a token's center back
/// to the end of the preceding speech run. A real pause between a word and the
/// word-final punctuation the DTW parked in it measures up to ~3s on the test
/// corpus; farther than this the entry sits beyond a plausible intra-word
/// linger -- a pause long enough that the model would normally bracket it with
/// its own timestamp tokens -- and the move cannot be trusted to land the word
/// on its own speech, so the center stays where the path put it.
const WHISPER_DTW_REANCHOR_MAX_JUMP_SECONDS: f32 = 3.5;

/// Envelope frames (0.02 s each) read forward from the token's center before
/// the center is treated as sitting in a pause (see
/// [`whisper_reanchor_dtw_token_centers`]).
const WHISPER_DTW_REANCHOR_ENTRY_QUIET_FRAMES: usize = 4;

/// Deployment env override for the reanchor minimum gap
/// ([`WHISPER_DTW_REANCHOR_MIN_GAP_SECONDS`]); a bare environment falls back
/// to the compiled default, staying byte-identical to it.
fn whisper_dtw_reanchor_min_gap_seconds() -> f32 {
    std::env::var("OPENASR_WHISPER_DTW_REANCHOR_MIN_GAP_SECONDS")
        .ok()
        .and_then(|raw| raw.parse::<f32>().ok())
        .unwrap_or(WHISPER_DTW_REANCHOR_MIN_GAP_SECONDS)
}

/// Deployment env override for the reanchor maximum jump
/// ([`WHISPER_DTW_REANCHOR_MAX_JUMP_SECONDS`]); a bare environment falls back
/// to the compiled default, staying byte-identical to it.
fn whisper_dtw_reanchor_max_jump_seconds() -> f32 {
    std::env::var("OPENASR_WHISPER_DTW_REANCHOR_MAX_JUMP_SECONDS")
        .ok()
        .and_then(|raw| raw.parse::<f32>().ok())
        .unwrap_or(WHISPER_DTW_REANCHOR_MAX_JUMP_SECONDS)
}

/// Pull a word-final punctuation token the DTW path parked in the pause after
/// its word back to the word's own audio offset.
///
/// The fold counts each token's path *entry* frame as its center and takes a
/// word's center as the mean of the contributing tokens. A word-final
/// punctuation token ("it?", "life.") has no audio of its own, and its
/// monotone path entry lingers in the pause after the word, so the word's
/// center becomes the mean of the word's own center and that pause: the fold's
/// seam at the word's end -- and the next word's start, which is the same
/// seam -- smears across the following silence instead of sitting at the
/// word's offset. When the next word's audio fills the smeared half of the
/// window, the edge refiners
/// ([`whisper_refine_dtw_word_onsets`], [`whisper_refine_dtw_word_offsets`])
/// see no hollow half either, and cannot trim what is left.
///
/// The model's own statement of which tokens carry no audio is their text. A
/// token whose fold piece has no letter or digit is punctuation, and only one
/// that is the *last* contributor to its fold word ends a word (an opening
/// quote or a lone dash contributes to the word it opens, and pulling that
/// token back would smear the *next* word's start instead of a previous
/// word's end). For such a word-final punctuation token, when its center sits
/// in trusted silence and the nearest preceding sustained speech run ended a
/// plausible pause (between the reanchor gap bounds) before it, the center is
/// replaced by the frame just past that run's end, before the fold runs. The
/// fold itself is untouched: it re-derives the word's seams around the
/// corrected center, and the onset refiner that follows can then see the next
/// word's real onset inside a window that is no longer smeared.
///
/// The trusted-silence requirement is the edge refiners' own (same floor,
/// same thin-floor ceiling): the center region's mean below the floor, no
/// frame above the silence ceiling, fewer than half its frames above the
/// floor, and every frame between the run's end and the center at or below
/// the ceiling. On a clean clip there is no pause-long punctuation entry to
/// pull, and on a music bed the ceiling tests fail the way the edge refiners'
/// do, so the pass is a no-op there.
fn whisper_reanchor_dtw_token_centers(
    mut token_times: Vec<Seq2SeqTokenTime>,
    decode_text: &dyn Fn(&[u32]) -> Option<String>,
    audio_rms_frames: Option<&[f32]>,
    seconds_per_frame: f32,
) -> Vec<Seq2SeqTokenTime> {
    let debug_reanchor = std::env::var_os("OPENASR_WHISPER_DEBUG_REANCHOR").is_some();
    let Some(levels) = audio_rms_frames else {
        return token_times;
    };
    if levels.len() < 4 || token_times.is_empty() || !seconds_per_frame.is_finite() {
        return token_times;
    }
    let spf = f64::from(seconds_per_frame);
    let mut ranked: Vec<f64> = levels.iter().map(|sample| f64::from(*sample)).collect();
    ranked.sort_by(f64::total_cmp);
    let noise_floor = ranked[ranked.len() / 2];
    if !(noise_floor > 0.0 && noise_floor.is_finite()) {
        return token_times;
    }
    let threshold = noise_floor * 10.0_f64.powf(WHISPER_DTW_ONSET_FLOOR_MARGIN_DB / 20.0);
    let clip_peak = *ranked.last().unwrap_or(&0.0);
    if !(clip_peak > 0.0 && clip_peak.is_finite()) {
        return token_times;
    }
    let silence_ceiling = whisper_dtw_silence_ceiling(noise_floor, clip_peak);
    let last_frame = levels.len() - 1;
    let min_gap = f64::from(whisper_dtw_reanchor_min_gap_seconds());
    let max_jump = f64::from(whisper_dtw_reanchor_max_jump_seconds());

    // Pass 1 (mirrors the fold's): the incremental prefix decode gives every
    // token the text piece the fold will attribute to it -- the all-punctuation
    // pieces are the ones the fold can smear.
    let mut pieces = Vec::with_capacity(token_times.len());
    let mut prefix = Vec::with_capacity(token_times.len());
    let mut previous_decoded = String::new();
    for token_time in &token_times {
        prefix.push(token_time.token_id);
        let Some(decoded) = decode_text(&prefix) else {
            // The fold will surface the decode failure; keep the path's
            // centers untouched.
            return token_times;
        };
        let piece = match decoded.strip_prefix(&previous_decoded) {
            Some(rest) => rest.to_string(),
            None => {
                let shared = common_prefix_len(&previous_decoded, &decoded);
                decoded[shared..].to_string()
            }
        };
        previous_decoded = decoded;
        pieces.push(piece);
    }

    for (index, token_time) in token_times.iter_mut().enumerate() {
        let piece = &pieces[index];
        // The token's center reaches the fold only through the piece's
        // non-whitespace characters. A piece with none is timing-inert, and a
        // piece carrying a letter or digit is real word content: neither is a
        // candidate.
        let mut has_content = false;
        let mut has_alphanumeric = false;
        for ch in piece.chars() {
            if !ch.is_whitespace() {
                has_content = true;
                if ch.is_alphanumeric() {
                    has_alphanumeric = true;
                }
            }
        }
        if !has_content || has_alphanumeric {
            continue;
        }
        // Word-final punctuation only: a later piece carrying a character of
        // the same fold word means this token opens a word, not ends one, and
        // pulling it back would smear the next word's start.
        if later_piece_contributes_to_same_word(index, &pieces) {
            continue;
        }
        let entry_secs = f64::from(token_time.center_seconds);
        if !entry_secs.is_finite() || entry_secs < 0.0 {
            continue;
        }
        let entry_frame = ((entry_secs / spf) as usize).min(last_frame);
        // Trusted pause at the center: the same mean / ceiling /
        // active-fraction tests the edge refiners apply to a word half, over a
        // short run of frames from the center.
        let quiet_end = (entry_frame + WHISPER_DTW_REANCHOR_ENTRY_QUIET_FRAMES).min(last_frame);
        let quiet = &levels[entry_frame..=quiet_end];
        let quiet_mean =
            quiet.iter().map(|sample| f64::from(*sample)).sum::<f64>() / quiet.len() as f64;
        let quiet_max = quiet
            .iter()
            .map(|sample| f64::from(*sample))
            .fold(0.0_f64, f64::max);
        let quiet_above = quiet
            .iter()
            .filter(|sample| f64::from(**sample) >= threshold)
            .count() as f64
            / quiet.len() as f64;
        if quiet_mean >= threshold || quiet_max > silence_ceiling || quiet_above > 0.5 {
            continue;
        }
        // The nearest preceding sustained speech run: walk back over the
        // trusted-silence frames that separate the center from it. The walk
        // stops at the frame-array edge (no preceding speech) or at a frame
        // above the silence ceiling (a music bed, not a pause).
        let mut scan = entry_frame;
        let mut run_end: Option<usize> = None;
        while scan > 0 {
            let level = f64::from(levels[scan - 1]);
            if level >= threshold {
                run_end = Some(scan - 1);
                break;
            }
            if level > silence_ceiling {
                break;
            }
            scan -= 1;
        }
        let Some(end) = run_end else {
            continue;
        };
        let mut run_start = end;
        while run_start > 0 && f64::from(levels[run_start - 1]) >= threshold {
            run_start -= 1;
        }
        // The anchor must be real speech, not a noise blip: at least the onset
        // sustain length of frames above the floor.
        if end - run_start + 1 < WHISPER_DTW_ONSET_SUSTAIN_FRAMES {
            continue;
        }
        // The silent gap between the run's end and the center must be a real
        // pause: clearly past the fold's calibration error, and short enough
        // to be an intra-word linger rather than a larger drift.
        let gap_secs = entry_secs - (end as f64 + 1.0) * spf;
        if !(min_gap..=max_jump).contains(&gap_secs) {
            continue;
        }
        // One frame past the run's last speech frame: the word's own offset.
        // The fold clamps centers non-decreasing, so the pulled center can
        // never run ahead of the previous word's.
        let target_secs = (end as f64 + 1.0) * spf;
        if debug_reanchor {
            eprintln!(
                "reanchor: piece={piece:?} entry={entry_secs:.2}s -> pulled to {target_secs:.2}s (preceding run ends at frame {end})",
            );
        }
        token_time.center_seconds = target_secs as f32;
    }
    token_times
}

/// Whether a piece after `index` contributes a character to the same fold
/// word as the piece at `index`. The walk mirrors the fold's character pass --
/// whitespace closes a word, a Han ideograph (or an alphanumeric after a
/// Han-final word) starts a new one -- so the eligibility matches the fold's
/// own word split exactly.
fn later_piece_contributes_to_same_word(index: usize, pieces: &[String]) -> bool {
    // The fold's `last_char()` once the candidate piece has contributed: its
    // last word-constituting character (a candidate piece is all punctuation,
    // so its non-whitespace characters all constitute the word). One
    // character of the first following piece decides: it either closes the
    // word (whitespace or a Han boundary) or is a same-word contribution.
    let last = pieces[index].chars().rev().find(|ch| !ch.is_whitespace());
    for piece in &pieces[index + 1..] {
        // The first character of the first non-empty following piece decides:
        // it either closes the word (whitespace or a Han boundary) or is a
        // same-word contribution from a later piece.
        if let Some(ch) = piece.chars().next() {
            return !(ch.is_whitespace()
                || last.is_some_and(|previous| han_script_boundary_before(ch, Some(previous))));
        }
    }
    false
}

/// Seconds each whisper word window's start is moved earlier, so the seam the
/// center fold placed between two adjacent words (the previous word's end and
/// this word's start, which coincide) lands on the real speech onset instead of
/// a hair inside it. Whisper's DTW treats every token's entry frame as the word
/// center and splits the gap between two centers to form the shared boundary,
/// so the boundary sits a fraction of a second inside the next word's audio.
/// The boundary is a fixed point of the per-word least-squares affine fit
/// TempErr uses, so a uniform per-side pad of this kind leaves TempErr
/// unchanged while the window widens back over the onset the fold clipped.
const WHISPER_WORD_ONSET_PAD_SECONDS: f32 = 0.10;

/// Seconds each whisper word window's end is moved later, the offset-side
/// counterpart of [`WHISPER_WORD_ONSET_PAD_SECONDS`]. The end seam is shared
/// with the next word's start, so both sides of a boundary are pulled out by the
/// pad at once.
const WHISPER_WORD_OFFSET_PAD_SECONDS: f32 = 0.10;

/// Widen each word window back over the true speech span: start earlier by
/// [`WHISPER_WORD_ONSET_PAD_SECONDS`], end later by
/// [`WHISPER_WORD_OFFSET_PAD_SECONDS`], clamped to `[0, duration]`.
///
/// The DTW center fold places adjacent words' windows back-to-back on shared
/// seam boundaries, so a word's window can end up a tenth of a second inside
/// its real onset/offset (the measured start-leak median is ~0.06s late, the
/// end-leak median ~0.21s short, with real speech energy in the clipped region).
/// Because the fold's seams are continuous, the pad is the one knob that pulls a
/// window edge back over the clipped audio without disturbing the word centers
/// the fold calibrated, and it is invariant to the per-word affine fit the
/// normalized TempErr metric uses.
///
/// Clamping and the start-before-end invariant keep the windows monotone: only
/// the first word's start can touch 0.0 and only the last word's end can touch
/// `duration`, and a pad far smaller than any word span can never invert an
/// interior pair.
fn whisper_pad_dtw_word_windows(
    mut words: Vec<crate::WordTimestamp>,
    audio_duration_seconds: f32,
) -> Vec<crate::WordTimestamp> {
    if words.is_empty() {
        return words;
    }
    let duration = audio_duration_seconds.max(0.0);
    for word in &mut words {
        let new_start = (word.start - WHISPER_WORD_ONSET_PAD_SECONDS)
            .max(0.0)
            .min(duration);
        let new_end = (word.end + WHISPER_WORD_OFFSET_PAD_SECONDS).min(duration);
        word.start = new_start;
        word.end = new_end.max(new_start);
    }
    words
}

/// Word timestamps for one decode from its per-token cross-attention rows.
/// The second return value pairs EVERY content run of the token stream (the
/// same partition and order as [`text_token_runs`]) with the half-open range
/// of the returned word list that run's words occupy. A run the DTW pass
/// failed to align maps to an empty range; the whole-window paths that align
/// without decoded timestamps map the run list to a single range over all
/// words. The degenerate-tail splice consumes the final run's range to keep
/// the token stream, the re-derived text, and the word list in sync.
pub(crate) fn whisper_cross_attention_word_timestamps(
    tokenizer: &WhisperTokenizer,
    token_alignments: &[WhisperGeneratedTokenAlignment],
    generated_probabilities: &[f32],
    audio_duration_seconds: f32,
    audio_rms_frames: Option<&[f32]>,
) -> Result<(Vec<crate::WordTimestamp>, Vec<(usize, usize)>), WhisperGgmlExecutorError> {
    if token_alignments.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }
    // Alignments are recorded one per generated token; a step that yielded no
    // cross-attention probs breaks that parity, in which case confidence is
    // withheld rather than misattributed by position.
    let probabilities_aligned = generated_probabilities.len() == token_alignments.len();
    let duration = audio_duration_seconds.max(0.0);
    let decode_text = |token_ids: &[u32]| tokenizer.decode_text_token_ids(token_ids);
    // Best-effort variant for the pre-fold reanchor pass: an undecodable
    // piece leaves the band's centers exactly as the path put them (the
    // fold surfaces the failure itself).
    let decode_piece_text = |token_ids: &[u32]| tokenizer.decode_text_token_ids(token_ids).ok();

    // Prefer a DTW pass over the per-token cross-attention rows. DTW assigns
    // every token an ordered, non-overlapping span of frames; the token's
    // entry frame (where the path first reaches it) is where its attention
    // peaks, i.e. the word's center rather than its onset. Folding the centers
    // into word windows (boundary fraction
    // [`WHISPER_DTW_BOUNDARY_FRACTION`] minus the fixed
    // [`WHISPER_DTW_ONSET_LEAD_SECONDS`]) places each word start before its
    // center, at the true speech onset, instead of at the next token's peak
    // the raw-span tiling used.
    let frame_resolution = token_alignments
        .first()
        .map(|a| a.frame_probs.len())
        .unwrap_or(0);
    if frame_resolution > 0 {
        // The cross-attention window is the padded encoder window at a fixed
        // 0.02s/frame (160-sample hop doubled through two strided convs, then
        // downsampled 2x by the encoder: 1500 frames for a 30s window), so
        // frames map to absolute wall-clock time from clip start, NOT a fraction
        // of `duration`. Stretching the axis to `[0, duration]` (as a
        // center-of-mass midpoint map does) would compress every timestamp for
        // any clip shorter than the 30s window, which is the common case.
        let seconds_per_frame = 2.0_f32 * WHISPER_HOP_LENGTH as f32 / WHISPER_SAMPLE_RATE_HZ as f32;
        let full_window = token_alignments
            .iter()
            .map(|alignment| alignment.frame_probs.clone())
            .collect::<Vec<Vec<f32>>>();

        // If the decode was run without `<|notimestamps|>` (word-timestamp
        // mode), the model emits the per-segment `<|start|>`/`<|end|>`
        // timestamp tokens. Like whisper-timestamped, slice the DTW to each
        // segment's `[start, end]` band so a silence gap between segments does
        // not stretch the last word of one segment across the gap into the next.
        // A timestamp token maps to its frame directly (token id offset from
        // `<|0.00|>` is its frame number).
        let timestamp_begin = tokenizer.first_timestamp_token_id();
        let token_ids: Vec<u32> = token_alignments.iter().map(|a| a.token_id).collect();
        let is_ts = |id: u32| timestamp_begin.is_some_and(|begin| id >= begin);

        // Maximal runs of consecutive text tokens (the words); timestamp tokens
        // separate them into segments.
        let runs = text_token_runs(&token_ids, &is_ts);
        // When no segment is bracketed by decoded timestamps, run a single DTW
        // over the whole window bracketed on the content-token attention peaks
        // (the no-timestamps degrade; see `speech_frame_bounds`).
        let bracketed_by_timestamps = runs.iter().any(|(lo, hi)| {
            run_frame_bounds(*lo, *hi, &token_ids, timestamp_begin, frame_resolution).is_some()
        });
        if bracketed_by_timestamps {
            let mut words = Vec::new();
            let mut run_word_ranges: Vec<(usize, usize)> = Vec::new();
            for (lo, hi) in &runs {
                let run_word_start = words.len();
                let Some((band_start, band_end)) =
                    run_frame_bounds(*lo, *hi, &token_ids, timestamp_begin, frame_resolution)
                else {
                    run_word_ranges.push((run_word_start, words.len()));
                    continue;
                };
                let attention: Vec<Vec<f32>> = full_window[*lo..=*hi]
                    .iter()
                    .map(|row| row[band_start.min(row.len())..band_end.min(row.len())].to_vec())
                    .collect();
                let Some(spans) = dtw_align_token_frames(&attention) else {
                    run_word_ranges.push((run_word_start, words.len()));
                    continue;
                };
                let band_width = band_end.saturating_sub(band_start);
                // Whisper's DTW backtracks to the band origin, so the lead token's
                // entry frame is always frame 0 of the slice: its center is pinned
                // to `band_start` by construction, and the fold anchors the first
                // word's start there too. When the run's bound is a leading
                // `<|0.00|>` leak that spans silence, that decoded `<|start|>` parks
                // the first word at the window's leading edge (measured up to
                // ~-1.5s vs the truth on opening segments) instead of at the speech
                // onset. `speech_band_from_rows` brackets this run on the frames its
                // content tokens' attention actually peaks on, so its start is where
                // the run's real speech begins. When that sits at least one band
                // margin ahead of the decoded bound, the bound bracketed leading
                // silence: anchor the first word's start (and the fold's lower
                // center clamp) at the measured onset, moving only the lead word
                // later and leaving every other word -- and the DTW slice itself --
                // exactly as baseline. A well-timestamped segment already starts
                // at/inside the speech, so its onset falls within one margin of the
                // bound and the anchor stays at the decoded bound, keeping those
                // clips identical to baseline.
                let run_is_content: Vec<bool> = (*lo..=*hi)
                    .map(|index| {
                        tokenizer
                            .decode_text_token_ids(&[token_ids[index]])
                            .is_ok_and(|text| token_text_carries_speech(&text))
                    })
                    .collect();
                let band_front =
                    speech_band_from_rows(&full_window[*lo..=*hi], &run_is_content, None)
                        .map(|(onset, _)| onset);
                // Advance the first word's start to the measured content onset when
                // the run's bound is a leading silence leak (see the decision in
                // `whisper_dtw_lead_silence_advance_frame`). A mid-run decoded
                // `<|start|>` or a sub-margin jitter falls through to the decoded
                // bound, leaving those segments byte-identical to baseline.
                let band_start_secs = match whisper_dtw_lead_silence_advance_frame(
                    band_start,
                    band_front,
                    seconds_per_frame,
                    whisper_dtw_lead_silence_advance_min_gap_seconds(),
                ) {
                    Some(front) => {
                        if std::env::var_os("OPENASR_WHISPER_DEBUG_CROSS").is_some() {
                            eprintln!(
                                "whisper cross leading-silence anchor: band={} onset={} gap={:.2}s -> {}",
                                band_start,
                                front,
                                (front.saturating_sub(band_start) as f32) * seconds_per_frame,
                                front
                            );
                        }
                        (front as f32) * seconds_per_frame
                    }
                    None => (band_start as f32) * seconds_per_frame,
                };
                let band_end_secs = (band_end as f32) * seconds_per_frame;
                let onset_lead = whisper_dtw_onset_lead();
                let token_times: Vec<Seq2SeqTokenTime> = spans
                    .iter()
                    .enumerate()
                    .map(|(rel, span)| {
                        let index = lo + rel;
                        // Add the slice offset back so frames are window-absolute.
                        // The DTW path's entry frame is where the token's
                        // cross-attention peaks: the word's center, not its
                        // onset (the fold in
                        // `seq2seq_word_timestamps_from_token_times` turns
                        // these centers into word windows).
                        let center_frame =
                            span.frame_start.min(band_width).saturating_add(band_start);
                        let probability = (probabilities_aligned
                            && index < generated_probabilities.len())
                        .then(|| generated_probabilities[index]);
                        Seq2SeqTokenTime {
                            token_id: token_alignments[index].token_id,
                            center_seconds: (center_frame as f32) * seconds_per_frame,
                            probability,
                        }
                    })
                    .collect();
                if token_times.is_empty() {
                    run_word_ranges.push((run_word_start, words.len()));
                    continue;
                }
                // A word-final punctuation token carries no audio of its own
                // and its center can sit in the pause after its word; pull it
                // back to the word's own offset before the fold turns these
                // centers into word windows.
                let token_times = whisper_reanchor_dtw_token_centers(
                    token_times,
                    &decode_piece_text,
                    audio_rms_frames,
                    seconds_per_frame,
                );
                // A per-segment decode failure is non-fatal: keep the other
                // segments' words rather than dropping the whole transcript.
                if let Ok(mut block_words) = seq2seq_word_timestamps_from_token_times(
                    &token_times,
                    band_start_secs,
                    band_end_secs,
                    BuiltinDecodePolicySeq2SeqTextPostprocessKind::Identity,
                    &decode_text,
                    WHISPER_DTW_BOUNDARY_FRACTION,
                    onset_lead,
                    f32::INFINITY,
                ) {
                    block_words = whisper_cap_dtw_word_spans(block_words, seconds_per_frame);
                    words.extend(block_words);
                }
                run_word_ranges.push((run_word_start, words.len()));
            }
            if !words.is_empty() {
                let words = whisper_pad_dtw_word_windows(
                    whisper_refine_dtw_word_offsets(
                        whisper_refine_dtw_word_onsets(words, audio_rms_frames, duration),
                        audio_rms_frames,
                        duration,
                    ),
                    duration,
                );
                // The pad only widens the existing windows (the count is
                // unchanged), so the per-run ranges stay valid.
                return Ok((words, run_word_ranges));
            }
            // Every bracketed run failed to align: fall through to the
            // center-of-mass degrade below.
        } else {
            // No decoded timestamps: bracket the DTW frame axis on the
            // attention of the tokens that actually carry speech. The
            // cross-attention envelope of content tokens ignores leading
            // silence and trailing non-speech the model did not attend to.
            // Punctuation/subword tokens are excluded: they carry no audible
            // span of their own and their cross-attention is the diffuse mass
            // that bleeds into nearby silence or the trailing non-speech, so
            // letting one set the band end would stretch the final word far
            // past where the speech stops (whisper-timestamped excludes a
            // trailing final punctuation for the same reason).
            let is_content: Vec<bool> = token_alignments
                .iter()
                .map(|alignment| {
                    tokenizer
                        .decode_text_token_ids(&[alignment.token_id])
                        .is_ok_and(|text| token_text_carries_speech(&text))
                })
                .collect();
            let (dtw_frame_start, dtw_frame_end) = speech_frame_bounds(&full_window, &is_content)
                .map_or_else(
                    // No usable content attention: fall back to the encoded clip
                    // duration so the last word still owns the real audio end.
                    move || {
                        (
                            0usize,
                            ((duration / seconds_per_frame).ceil() as usize)
                                .clamp(1, frame_resolution),
                        )
                    },
                    |(start, end)| (start, end.clamp(start + 1, frame_resolution)),
                );
            let attention: Vec<Vec<f32>> = full_window
                .iter()
                .map(|row| {
                    row[dtw_frame_start.min(row.len())..dtw_frame_end.min(row.len())].to_vec()
                })
                .collect();
            if let Some(spans) = dtw_align_token_frames(&attention) {
                // Same entry-frame-center fold as the timestamp-bracketed path
                // above (see [`WHISPER_DTW_BOUNDARY_FRACTION`]); the band is the
                // content-attention envelope instead of decoded timestamps.
                let band_start_secs = (dtw_frame_start as f32) * seconds_per_frame;
                let band_end_secs = ((dtw_frame_end as f32) * seconds_per_frame).min(duration);
                let token_times: Vec<Seq2SeqTokenTime> = token_alignments
                    .iter()
                    .enumerate()
                    .zip(spans.iter())
                    .map(|((index, alignment), span)| {
                        // Add the slice offset back so frames are window-absolute.
                        let center_frame = span
                            .frame_start
                            .min(dtw_frame_end - dtw_frame_start)
                            .saturating_add(dtw_frame_start);
                        Seq2SeqTokenTime {
                            token_id: alignment.token_id,
                            center_seconds: (center_frame as f32) * seconds_per_frame,
                            probability: probabilities_aligned
                                .then(|| generated_probabilities[index]),
                        }
                    })
                    .collect();
                // A word-final punctuation token carries no audio of its own
                // and its center can sit in the pause after its word; pull it
                // back to the word's own offset before the fold turns these
                // centers into word windows.
                let token_times = whisper_reanchor_dtw_token_centers(
                    token_times,
                    &decode_piece_text,
                    audio_rms_frames,
                    seconds_per_frame,
                );
                let onset_lead = whisper_dtw_onset_lead();
                let mut words = seq2seq_word_timestamps_from_token_times(
                    &token_times,
                    band_start_secs,
                    band_end_secs,
                    BuiltinDecodePolicySeq2SeqTextPostprocessKind::Identity,
                    &decode_text,
                    WHISPER_DTW_BOUNDARY_FRACTION,
                    onset_lead,
                    f32::INFINITY,
                )
                .map_err(|error| {
                    WhisperGgmlExecutorError::DecoderInvalidTokenDecode {
                        reason: format!("whisper DTW word timestamp token decode failed: {error}"),
                    }
                })?;
                words = whisper_cap_dtw_word_spans(words, seconds_per_frame);
                let words = whisper_pad_dtw_word_windows(
                    whisper_refine_dtw_word_offsets(
                        whisper_refine_dtw_word_onsets(words, audio_rms_frames, duration),
                        audio_rms_frames,
                        duration,
                    ),
                    duration,
                );
                // The whole window aligned as one unbracketed pass: a single
                // range covering every word.
                let word_count = words.len();
                return Ok((words, vec![(0, word_count)]));
            }
        }
    }

    // Degenerate input (empty/ragged attention) has no alignment; fall back to
    // the per-token center of mass.
    let token_times = token_alignments
        .iter()
        .enumerate()
        .map(|(index, alignment)| {
            Ok(Seq2SeqTokenTime {
                token_id: alignment.token_id,
                center_seconds: cross_attention_center_seconds(&alignment.frame_probs, duration)?,
                probability: probabilities_aligned.then(|| generated_probabilities[index]),
            })
        })
        .collect::<Result<Vec<_>, WhisperGgmlExecutorError>>()?;
    seq2seq_word_timestamps_from_token_times(
        &token_times,
        0.0,
        duration,
        BuiltinDecodePolicySeq2SeqTextPostprocessKind::Identity,
        &decode_text,
        MIDPOINT_BOUNDARY_FRACTION,
        NO_ONSET_LEAD,
        f32::INFINITY,
    )
    .map(|words| {
        let words = whisper_pad_dtw_word_windows(words, duration);
        let word_count = words.len();
        (words, vec![(0, word_count)])
    })
    .map_err(
        |error| WhisperGgmlExecutorError::DecoderInvalidTokenDecode {
            reason: format!("whisper cross-attention word timestamp token decode failed: {error}"),
        },
    )
}

fn cross_attention_center_seconds(
    frame_probs: &[f32],
    audio_duration_seconds: f32,
) -> Result<f32, WhisperGgmlExecutorError> {
    if frame_probs.is_empty() || audio_duration_seconds <= 0.0 {
        return Ok(0.0);
    }
    let mut weighted_frame = 0.0_f32;
    let mut total = 0.0_f32;
    for (frame_index, prob) in frame_probs.iter().copied().enumerate() {
        if !prob.is_finite() {
            return Err(WhisperGgmlExecutorError::DecoderGraphExecutionFailed {
                reason:
                    "whisper cross-attention word timestamp probabilities contain non-finite values"
                        .to_string(),
            });
        }
        let prob = prob.max(0.0);
        weighted_frame += (frame_index as f32 + 0.5) * prob;
        total += prob;
    }
    if total <= 0.0 || !total.is_finite() {
        return Ok(0.0);
    }
    let center_frame = weighted_frame / total;
    Ok(
        (center_frame / frame_probs.len() as f32 * audio_duration_seconds)
            .clamp(0.0, audio_duration_seconds),
    )
}

#[cfg(test)]
mod tests;
