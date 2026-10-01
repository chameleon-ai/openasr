//! Cohere DTW word-timestamp refinement, post-decode and ggml-free.
//!
//! The ggml decoder runs one decode and records one last-layer cross-attention
//! frame row per generated token; this module turns those rows into word
//! windows. Per chunk the pipeline is: the peak-order gate (strict, the
//! dominant-early-sink strip, and a tolerant tier for long bands) -> the
//! speech band derived from the unmasked rows with the stripped sinks skipped
//! -> the audio-onset anchors that repair a band start displaced by the sink
//! substitution or bracketing leading silence -> the monotone DTW alignment of
//! token frames -> the pre-fold reanchor that pulls word-final punctuation off
//! a pause -> the center fold into word windows -> the per-word span cap -> the
//! onset refiner -> the offset refiner -> the caller's symmetric window pad.
//! The reanchor and the two refiners read the chunk's 0.02 s RMS envelope and
//! are no-ops when it is absent, so the fold's own windows stand unchanged.
//! Each gate failure has its own degrade: the per-word peak placement for long
//! windows, and the caller's uniform post-hoc baseline when the attention
//! cannot be trusted at all. That order is what the test suite validates end
//! to end; do not reorder passes without rerunning it.
//!
//! Every function here is pure over `f32`/token/alignment slices plus the
//! execution metadata (no graph state, no executor internals). The call sites
//! live in `decoder_graph.rs` (the DTW pass and the word-window pad) and
//! `ggml_executor.rs` (the measured audio onset); see the `COHERE_DTW_*`
//! constants (and the `OPENASR_COHERE_DTW_LEAD_*` env overrides) for the
//! tunables.

use crate::api::backend::WordTimestamp;
use crate::models::decode_policy_component_registry::BuiltinDecodePolicySeq2SeqTextPostprocessKind;
use crate::models::seq2seq_dtw_alignment::{
    DTW_SPEECH_BAND_MARGIN_FRAMES, dtw_align_token_frames, speech_band_from_rows,
    token_text_carries_speech,
};
use crate::models::seq2seq_word_timestamps::{
    MIDPOINT_BOUNDARY_FRACTION, NO_ONSET_LEAD, Seq2SeqTokenTime, han_script_boundary_before,
    seq2seq_word_timestamps_from_token_times,
};
use crate::models::text_prefix::common_prefix_len;

use super::runtime_contract::CohereTranscribeExecutionMetadata;

/// Window (seconds) and relative-to-peak level drop (dB) used to detect where
/// real audio content begins inside a chunk. 0.1s isolates a word from
/// surrounding silence at 16kHz, and a 16 dB drop below the chunk's loudest
/// window clears genuine room tone while catching even a quiet opener.
const COHERE_ONSET_WINDOW_SECONDS: f32 = 0.1;
const COHERE_ONSET_RELATIVE_DROP_DB: f32 = 16.0;

/// Sample rate the audio envelope runs at, in Hz.
const COHERE_DTW_ENVELOPE_SAMPLE_RATE_HZ: u32 = 16_000;

/// Length of one audio-envelope RMS window, in samples (0.02 s at 16 kHz). The
/// envelope is expressed in the same absolute 0.02 s time base the whisper
/// refiners use, independent of cohere's coarser 0.08 s cross-attention frame,
/// so the sustain/min-silence frame counts below keep their whisper meaning
/// (5 frames = 0.1 s of speech, 2 frames = 0.04 s of pause).
const COHERE_DTW_ENVELOPE_FRAME_COUNT: usize = 320;

/// Seconds covered by one envelope frame.
const fn cohere_dtw_envelope_seconds_per_frame() -> f64 {
    COHERE_DTW_ENVELOPE_FRAME_COUNT as f64 / COHERE_DTW_ENVELOPE_SAMPLE_RATE_HZ as f64
}

/// How far a punctuation-only token piece may sit from its word's content mean
/// and still count toward the word's center. Mirrors whisper's fold: a comma
/// parked close to its word is the DTW path's best statement of the word's
/// offset, while one the path lingered seconds into a following pause drags the
/// center off the speech. Honored with whisper's env-override convention so a
/// deployment can retune it without a rebuild.
const COHERE_DTW_PUNCTUATION_TRUST_RADIUS_SECONDS: f32 = 1.75;

/// How far an interior word's window may extend on either side of its own
/// center before the fold leaves the rest of an adjacent pause as real silence.
/// Mirrors whisper's clamp. Continuous speech never binds it: with cohere's
/// 0.35 boundary fraction only inter-center gaps well past a second reach it.
const COHERE_DTW_MAX_INTERIOR_HALF_SPAN_SECONDS: f32 = 1.0;

/// The punctuation trust radius in use, honoring the deployment env override so
/// a tuning pass can sweep it without a rebuild (see
/// [`COHERE_DTW_PUNCTUATION_TRUST_RADIUS_SECONDS`]). A bare environment is
/// byte-identical to the constant.
fn cohere_dtw_punctuation_trust_radius_seconds() -> f32 {
    std::env::var("OPENASR_COHERE_DTW_PUNCTUATION_TRUST_RADIUS_SECONDS")
        .ok()
        .and_then(|raw| raw.parse::<f32>().ok())
        .unwrap_or(COHERE_DTW_PUNCTUATION_TRUST_RADIUS_SECONDS)
}

/// The interior half-span in use, honoring the deployment env override so a
/// tuning pass can sweep it without a rebuild (see
/// [`COHERE_DTW_MAX_INTERIOR_HALF_SPAN_SECONDS`]).
fn cohere_dtw_max_interior_half_span_seconds() -> f32 {
    std::env::var("OPENASR_COHERE_DTW_MAX_INTERIOR_HALF_SPAN_SECONDS")
        .ok()
        .and_then(|raw| raw.parse::<f32>().ok())
        .unwrap_or(COHERE_DTW_MAX_INTERIOR_HALF_SPAN_SECONDS)
}

/// Share of a word's front (resp. back) half tolerated above the floor before the
/// word is no longer hollow (speech bleeding into that half).
const COHERE_DTW_HOLLOW_FRONT_ACTIVE_MAX: f32 = 0.5;
const COHERE_DTW_HOLLOW_BACK_ACTIVE_MAX: f32 = 0.5;

/// Maximum of the front half may sit, as a fraction of the chunk's peak envelope
/// level, before the silence is no longer trusted as a real pause. A music or
/// noise background never reads as digital zero -- its floor is a real level --
/// so without this a quiet pause in a music-backed chunk looks hollow and the
/// onset push fires on the music floor. Mirrors whisper's swept value.
const COHERE_DTW_HOLLOW_FRONT_MAX_PEAK_FRACTION: f64 = 0.05;

/// Absolute RMS level below which a word half reads as trusted digital silence,
/// regardless of the chunk's own noise floor, scaled by this multiple of the
/// median. On a dense chunk (music bed, voice FX) a *relative* dip passes every
/// relative hollow check while still carrying sustained audio, so bed-level
/// passages must also sit well below the chunk's own median to read as a pause.
const COHERE_DTW_HOLLOW_ABSOLUTE_MEDIAN_MULTIPLE: f64 = 0.3;

/// Multiple of the chunk median (the noise floor) the edge-refiner silence
/// ceiling never drops below. Without the floor, 5%-of-peak can sit *below* the
/// median on a dense chunk, so every ordinary floor frame crosses the ceiling and
/// no genuine pause ever reads as hollow, muting the refiners exactly on the
/// noisy chunks that stretch words across pauses.
const COHERE_DTW_EDGE_SILENCE_FLOOR_MEDIAN_MULTIPLE: f64 = 2.0;

/// Peak-to-median contrast above which a chunk's median is a *thin noise floor*
/// rather than a continuous bed, so the silence ceiling may rise off the peak
/// and onto the floor (see [`cohere_dtw_silence_ceiling`]). A dense music bed
/// sits within a few times of its loudest frame; a recording with room tone, a
/// distant bed, or a line hum carries speech peaks an order of magnitude above
/// its floor. In between, neither reading of "quiet" is safe, so the conservative
/// peak fraction is kept.
const COHERE_DTW_THIN_FLOOR_CONTRAST: f64 = 8.0;

/// How far above the floor a single frame in a region may read before the region
/// stops being silence, on a thin-floor chunk only. 3.0 is just under 5 dB over
/// the floor -- the same margin the speech threshold already applies.
const COHERE_DTW_THIN_FLOOR_PEAK_OF_MEDIAN: f64 = 3.0;

/// Deployment env override for the thin-floor contrast
/// ([`COHERE_DTW_THIN_FLOOR_CONTRAST`]).
fn cohere_dtw_thin_floor_contrast() -> f64 {
    std::env::var("OPENASR_COHERE_DTW_THIN_FLOOR_CONTRAST")
        .ok()
        .and_then(|raw| raw.parse::<f64>().ok())
        .unwrap_or(COHERE_DTW_THIN_FLOOR_CONTRAST)
}

/// Deployment env override for the thin-floor floor multiple
/// ([`COHERE_DTW_THIN_FLOOR_PEAK_OF_MEDIAN`]).
fn cohere_dtw_thin_floor_peak_of_median() -> f64 {
    std::env::var("OPENASR_COHERE_DTW_THIN_FLOOR_PEAK_OF_MEDIAN")
        .ok()
        .and_then(|raw| raw.parse::<f64>().ok())
        .unwrap_or(COHERE_DTW_THIN_FLOOR_PEAK_OF_MEDIAN)
}

/// dB above the chunk's own noise floor (the median envelope level) that counts
/// as real speech, for the onset pass.
const COHERE_DTW_ONSET_FLOOR_MARGIN_DB: f64 = 5.0;
/// dB below the search region's own peak that still counts as that word's audio
/// for the offset pass, so a quiet word over a music bed is judged against its
/// own level rather than the chunk-wide floor.
const COHERE_DTW_OFFSET_PEAK_FLOOR_MARGIN_DB: f64 = 12.0;

/// Minimum duration of the speech run above the floor that qualifies as the
/// word's onset / offset, in envelope frames of 0.02 s (0.1 s of speech).
const COHERE_DTW_ONSET_SUSTAIN_FRAMES: usize = 5;
const COHERE_DTW_OFFSET_SUSTAIN_FRAMES: usize = 5;

/// Minimum run of silence between two speech runs, in seconds, so a run that
/// merely touches a brief inter-word glottal gap is not treated as a real pause.
const COHERE_DTW_ONSET_MIN_SILENCE_S: f32 = 0.03;
const COHERE_DTW_OFFSET_MIN_SILENCE_S: f32 = 0.03;

/// The onset/offset must differ from the fold's own edge by at least this much,
/// or the adjustment is smaller than the fold's calibration error and the word is
/// left as-is. Mirrors whisper's floor.
const COHERE_DTW_ONSET_MIN_PUSH_S: f32 = 0.25;
const COHERE_DTW_OFFSET_MIN_PULL_S: f32 = 0.25;

/// Upper bound on how far into a pause the envelope is trusted to lead/pull a
/// word.
const COHERE_DTW_ONSET_MAX_PUSH_S: f32 = 5.0;
const COHERE_DTW_OFFSET_MAX_PULL_S: f32 = 5.0;

/// Shortest word span, in seconds, the onset/offset refiners examine at all --
/// below it the window is too short to split into halves meaningfully.
const COHERE_DTW_REFINE_MIN_SPAN_S: f64 = 0.3;

/// How far before the word's window the offset run search reaches. The fold's
/// entry frame can land late -- in the gap *after* a word's own audio -- so the
/// word's above-floor run sits partly before its window and an in-window search
/// would miss it.
const COHERE_DTW_OFFSET_EDGE_LEAD_S: f32 = 0.5;

/// Below-floor frames tolerated inside a speech run before it splits. A word's
/// offset tail decays through coarticulatory micro-silences shorter than this;
/// without the tolerance the run shreds and no qualifying offset is found.
const COHERE_DTW_OFFSET_RUN_GAP_TOLERANCE_FRAMES: usize = 3;

/// Consecutive frames above the silence ceiling before a hollow region stops
/// reading as silence. A single bed-crackle frame is not a music floor (that
/// would void a legitimate refinement on bed-backed chunks), while a sustained
/// floor crosses and still bails.
const COHERE_DTW_HOLLOW_CEILING_SUSTAIN_FRAMES: usize = 3;

/// Minimum distance, in seconds, between the end of the nearest preceding
/// sustained speech run and a token's center before the center is treated as
/// parked in the pause after its word's own audio (the word-final punctuation
/// reanchor). Below it the center is inside the fold's own calibration error.
const COHERE_DTW_REANCHOR_MIN_GAP_SECONDS: f32 = 0.15;

/// Maximum distance the reanchor may pull a token's center back. Farther than a
/// plausible intra-word linger the move cannot be trusted to land the word on its
/// own speech, so the center stays where the path put it.
const COHERE_DTW_REANCHOR_MAX_JUMP_SECONDS: f32 = 3.5;

/// Envelope frames read forward from the token's center before the center is
/// treated as sitting in a pause.
const COHERE_DTW_REANCHOR_ENTRY_QUIET_FRAMES: usize = 4;

/// Deployment env override for the reanchor minimum gap
/// ([`COHERE_DTW_REANCHOR_MIN_GAP_SECONDS`]).
fn cohere_dtw_reanchor_min_gap_seconds() -> f32 {
    std::env::var("OPENASR_COHERE_DTW_REANCHOR_MIN_GAP_SECONDS")
        .ok()
        .and_then(|raw| raw.parse::<f32>().ok())
        .unwrap_or(COHERE_DTW_REANCHOR_MIN_GAP_SECONDS)
}

/// Deployment env override for the reanchor maximum jump
/// ([`COHERE_DTW_REANCHOR_MAX_JUMP_SECONDS`]).
fn cohere_dtw_reanchor_max_jump_seconds() -> f32 {
    std::env::var("OPENASR_COHERE_DTW_REANCHOR_MAX_JUMP_SECONDS")
        .ok()
        .and_then(|raw| raw.parse::<f32>().ok())
        .unwrap_or(COHERE_DTW_REANCHOR_MAX_JUMP_SECONDS)
}

/// The level a region may reach before it stops reading as trusted silence.
///
/// The base ceiling is a small fraction of the chunk's *peak*: a music or noise
/// bed never reads as digital zero, so a region that still carries a substantial
/// fraction of the loudest frame is a bed, not a pause. On a chunk with a *thin*
/// background floor (room tone, a distant bed) the floor's own transients can
/// cross the peak fraction even though it is acoustically silence; the chunk's
/// own peak/median contrast tells the two apart, raising the ceiling onto the
/// floor for thin-floor chunks only. Dense-bed chunks keep the conservative peak
/// fraction exactly as before.
fn cohere_dtw_silence_ceiling(noise_floor: f64, clip_peak: f64) -> f64 {
    let peak_fraction = clip_peak * COHERE_DTW_HOLLOW_FRONT_MAX_PEAK_FRACTION;
    let contrast = clip_peak / noise_floor.max(f64::EPSILON);
    if contrast >= cohere_dtw_thin_floor_contrast() {
        peak_fraction.max(noise_floor * cohere_dtw_thin_floor_peak_of_median())
    } else {
        peak_fraction
    }
}

/// The trusted-silence ceiling for the onset/offset edge refiners: the shared
/// [`cohere_dtw_silence_ceiling`], floored at twice the chunk median so it can
/// never sit below the floor it judges against.
fn cohere_dtw_edge_silence_ceiling(noise_floor: f64, clip_peak: f64) -> f64 {
    cohere_dtw_silence_ceiling(noise_floor, clip_peak)
        .max(noise_floor * COHERE_DTW_EDGE_SILENCE_FLOOR_MEDIAN_MULTIPLE)
}

/// Seconds from the start of the chunk at which measurable audio energy first
/// appears, or `0.0` when none does. Used to anchor the DTW speech band when the
/// sink strip has displaced the first word's attention peak (see
/// `cohere_dtw_word_timestamps`): a chunk that opens with silence but whose
/// first token's peak was substituted away must not have its alignment band
/// start past the real audio onset. Measured on the raw chunk samples -- the
/// same source `audio_duration_seconds` reads.
pub(crate) fn audio_onset_seconds(samples: &[f32], sample_rate_hz: u32) -> f32 {
    let rate = sample_rate_hz.max(1) as f32;
    let window = (COHERE_ONSET_WINDOW_SECONDS * rate) as usize;
    if samples.is_empty() || window == 0 {
        return 0.0;
    }
    let mut rms = Vec::new();
    let mut peak: f32 = 0.0;
    let mut cursor = 0usize;
    while cursor + window <= samples.len() {
        let mut energy = 0.0f64;
        for &s in &samples[cursor..cursor + window] {
            let f = s as f64;
            energy += f * f;
        }
        let value = (energy / window as f64).sqrt() as f32;
        peak = peak.max(value);
        rms.push(value);
        cursor += window;
    }
    let Some(first) = rms.first().copied() else {
        return 0.0;
    };
    let threshold = peak * 10f32.powf(-COHERE_ONSET_RELATIVE_DROP_DB / 20.0);
    if peak <= 0.0 || first > threshold {
        return 0.0;
    }
    for (index, &value) in rms.iter().enumerate() {
        if value > threshold {
            return index as f32 * COHERE_ONSET_WINDOW_SECONDS;
        }
    }
    0.0
}

/// Per-frame RMS envelope of the request audio on a 0.02 s grid -- the time base
/// the word-timing refiners are expressed in, and finer than cohere's 0.08 s
/// cross-attention frame. `None` when the audio is not the 16 kHz mono PCM this
/// path assumes, or carries a non-finite sample, so every refinement below
/// degrades to a clean no-op instead of acting on garbage.
///
/// A clip fully below the f32 dynamic range (all zeros) yields a median of zero
/// and the hollow predicate refuses to fire for a non-positive floor, so
/// all-silent input is left exactly as the fold produced it -- no words move.
pub(crate) fn cohere_dtw_word_audio_rms_frames(
    samples: &[f32],
    sample_rate_hz: u32,
) -> Option<Vec<f32>> {
    if sample_rate_hz != COHERE_DTW_ENVELOPE_SAMPLE_RATE_HZ {
        // The refiners index the envelope on a fixed 0.02 s grid; a different
        // rate means the caller is misusing this helper.
        return None;
    }
    if samples.is_empty() {
        return None;
    }
    let mut out = Vec::with_capacity(samples.len().div_ceil(COHERE_DTW_ENVELOPE_FRAME_COUNT));
    for frame_start in (0..samples.len()).step_by(COHERE_DTW_ENVELOPE_FRAME_COUNT) {
        let frame_end = (frame_start + COHERE_DTW_ENVELOPE_FRAME_COUNT).min(samples.len());
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

/// The chunk's noise floor (median envelope level) and peak, or `None` when
/// either is non-positive/non-finite -- which covers a fully silent chunk and any
/// degenerate envelope. Shared by the reanchor and both edge refiners so they
/// all judge against one reading of "speech" and "silence".
fn cohere_dtw_envelope_floor_and_peak(levels: &[f32]) -> Option<(f64, f64)> {
    if levels.len() < 4 {
        return None;
    }
    let mut ranked: Vec<f64> = levels.iter().map(|sample| f64::from(*sample)).collect();
    ranked.sort_by(f64::total_cmp);
    let noise_floor = ranked[ranked.len() / 2];
    if !(noise_floor > 0.0 && noise_floor.is_finite()) {
        return None;
    }
    let clip_peak = *ranked.last().unwrap_or(&0.0);
    if !(clip_peak > 0.0 && clip_peak.is_finite()) {
        return None;
    }
    Some((noise_floor, clip_peak))
}

/// Pull a word-final punctuation token the DTW path parked in the pause after its
/// word back to the word's own audio offset.
///
/// The fold counts each token's path *entry* frame as its center and takes a
/// word's center as the mean of the contributing tokens. A word-final punctuation
/// token ("it?", "life.") has no audio of its own, and its monotone path entry
/// lingers in the pause after the word, so the word's center becomes the mean of
/// the word's own center and that pause. The fold's seam at the word's end -- and
/// the next word's start, which is the same seam -- smears across the following
/// silence instead of sitting at the word's offset. When the next word's audio
/// fills the smeared half of the window the edge refiners see no hollow half
/// either, and cannot trim what is left.
///
/// The model's own statement of which tokens carry no audio is their text. A
/// token whose fold piece has no letter or digit is punctuation, and only one
/// that is the *last* contributor to its fold word ends a word (an opening quote
/// or a lone dash contributes to the word it opens, and pulling that token back
/// would smear the *next* word's start instead). For such a word-final
/// punctuation token, when its center sits in trusted silence and the nearest
/// preceding sustained speech run ended a plausible pause before it, the center
/// is replaced by the frame just past that run's end, before the fold runs. The
/// fold itself is untouched: it re-derives the word's seams around the corrected
/// center.
///
/// The trusted-silence requirement is the edge refiners' own (same floor, same
/// thin-floor ceiling): the center region's mean below the floor, no frame above
/// the silence ceiling, fewer than half its frames above the floor, and every
/// frame between the run's end and the center at or below the ceiling. On a
/// clean chunk there is no pause-long punctuation entry to pull, and on a music
/// bed the ceiling tests fail the way the edge refiners' do, so the pass is a
/// no-op there.
fn cohere_reanchor_dtw_token_centers<E>(
    mut token_times: Vec<Seq2SeqTokenTime>,
    decode_text: &dyn Fn(&[u32]) -> Result<String, E>,
    audio_rms_frames: Option<&[f32]>,
) -> Vec<Seq2SeqTokenTime> {
    let Some(levels) = audio_rms_frames else {
        return token_times;
    };
    if token_times.is_empty() {
        return token_times;
    }
    let Some((noise_floor, clip_peak)) = cohere_dtw_envelope_floor_and_peak(levels) else {
        return token_times;
    };
    let threshold = noise_floor * 10.0_f64.powf(COHERE_DTW_ONSET_FLOOR_MARGIN_DB / 20.0);
    // The reanchor walks *backward* through ceiling-quiet frames to reach a
    // distant speech run, so it keeps the conservative (unfloored) ceiling: a
    // music bed must never be walkable silence here, or the pass would attach
    // punctuation to unrelated audio.
    let silence_ceiling = cohere_dtw_silence_ceiling(noise_floor, clip_peak);
    let envelope_spf = cohere_dtw_envelope_seconds_per_frame();
    let last_frame = levels.len() - 1;
    let min_gap = f64::from(cohere_dtw_reanchor_min_gap_seconds());
    let max_jump = f64::from(cohere_dtw_reanchor_max_jump_seconds());

    // Pass 1 (mirrors the fold's): the incremental prefix decode gives every
    // token the text piece the fold will attribute to it -- the all-punctuation
    // pieces are the ones the fold can smear.
    let mut pieces = Vec::with_capacity(token_times.len());
    let mut prefix = Vec::with_capacity(token_times.len());
    let mut previous_decoded = String::new();
    for token_time in &token_times {
        prefix.push(token_time.token_id);
        let Ok(decoded) = decode_text(&prefix) else {
            // The fold will surface the decode failure; keep the path's centers
            // untouched.
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
        // Word-final punctuation only: a later piece carrying a character of the
        // same fold word means this token opens a word, not ends one, and pulling
        // it back would smear the next word's start.
        if later_piece_contributes_to_same_word(index, &pieces) {
            continue;
        }
        let entry_secs = f64::from(token_time.center_seconds);
        if !entry_secs.is_finite() || entry_secs < 0.0 {
            continue;
        }
        let entry_frame = ((entry_secs / envelope_spf) as usize).min(last_frame);
        // Trusted pause at the center: the same mean / ceiling / active-fraction
        // tests the edge refiners apply to a word half, over a short run of
        // frames from the center.
        let quiet_end = (entry_frame + COHERE_DTW_REANCHOR_ENTRY_QUIET_FRAMES).min(last_frame);
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
        // stops at the frame-array edge (no preceding speech) or at a frame above
        // the silence ceiling (a music bed, not a pause).
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
        if end - run_start + 1 < COHERE_DTW_ONSET_SUSTAIN_FRAMES {
            continue;
        }
        // The silent gap between the run's end and the center must be a real
        // pause: clearly past the fold's calibration error, and short enough to be
        // an intra-word linger rather than a larger drift.
        let gap_secs = entry_secs - (end as f64 + 1.0) * envelope_spf;
        if !(min_gap..=max_jump).contains(&gap_secs) {
            continue;
        }
        // One frame past the run's last speech frame: the word's own offset. The
        // fold clamps centers non-decreasing, so the pulled center can never run
        // ahead of the previous word's.
        let target_secs = (end as f64 + 1.0) * envelope_spf;
        if std::env::var_os("OPENASR_COHERE_DEBUG_REANCHOR").is_some() {
            eprintln!(
                "cohere reanchor: piece={piece:?} entry={entry_secs:.2}s -> pulled to {target_secs:.2}s (preceding run ends at frame {end})"
            );
        }
        token_time.center_seconds = target_secs as f32;
    }
    token_times
}

/// Whether a piece after `index` contributes a character to the same fold word as
/// the piece at `index`. The walk mirrors the fold's character pass -- whitespace
/// closes a word, a Han ideograph (or an alphanumeric after a Han-final word)
/// starts a new one -- so the eligibility matches the fold's own word split.
fn later_piece_contributes_to_same_word(index: usize, pieces: &[String]) -> bool {
    // The fold's `last_char()` once the candidate piece has contributed: its last
    // word-constituting character (a candidate piece is all punctuation, so its
    // non-whitespace characters all constitute the word). One character of the
    // first following piece decides: it either closes the word (whitespace or a
    // Han boundary) or is a same-word contribution.
    let last = pieces[index].chars().rev().find(|ch| !ch.is_whitespace());
    for piece in &pieces[index + 1..] {
        // The first character of the first non-empty following piece decides: it
        // either closes the word (whitespace or a Han boundary) or is a same-word
        // contribution from a later piece.
        if let Some(ch) = piece.chars().next() {
            return !(ch.is_whitespace()
                || last.is_some_and(|previous| han_script_boundary_before(ch, Some(previous))));
        }
    }
    false
}

/// Pull a word the center fold landed in silence to its real audio onset.
///
/// The DTW entry frame the fold treats as a word's center sits where the monotone
/// path first *enters* a token's row. After an intra-segment pause that entry is
/// at the tail of the preceding word or part-way into the pause, not on the next
/// word's audio; the `boundary_fraction` split that follows places the next
/// word's start early across the gap -- into audio that is silent.
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
/// quiet recording down to its own level. Crucially the front must also stay
/// within a small fraction of the chunk's *peak*: a music or noise bed never
/// reads as digital zero, so a quiet passage inside a music-backed chunk is *not*
/// a trusted pause and a push would only move a word that was already acceptable.
/// That ceiling is what separates a genuine zero-silence pause (fire) from a low
/// music floor (skip).
fn cohere_refine_dtw_word_onsets(
    mut words: Vec<WordTimestamp>,
    audio_rms_frames: Option<&[f32]>,
    duration_s: f32,
) -> Vec<WordTimestamp> {
    let Some(levels) = audio_rms_frames else {
        return words;
    };
    if duration_s <= 0.0 || words.len() < 2 {
        return words;
    }
    let Some((noise_floor, clip_peak)) = cohere_dtw_envelope_floor_and_peak(levels) else {
        return words;
    };
    let threshold = noise_floor * 10.0_f64.powf(COHERE_DTW_ONSET_FLOOR_MARGIN_DB / 20.0);
    // A front frame that reads at or above this is not true silence (see
    // `cohere_dtw_edge_silence_ceiling`): the edge floor keeps the ceiling off
    // the sub-median regime where ordinary floor frames would void every genuine
    // pause on a dense chunk.
    let silence_ceiling = cohere_dtw_edge_silence_ceiling(noise_floor, clip_peak);
    let envelope_spf = cohere_dtw_envelope_seconds_per_frame();
    let min_quiet_frames = (COHERE_DTW_ONSET_MIN_SILENCE_S as f64 / envelope_spf).ceil() as usize;
    // The absolute quiet check only applies on dense chunks (contrast < 8). On
    // thin floors the relative threshold is already tight and the ceiling is
    // raised to 3x median; the absolute check would only block genuine-silence
    // refiners on quiet chunks.
    let is_dense_chunk =
        clip_peak / noise_floor.max(f64::EPSILON) < cohere_dtw_thin_floor_contrast();
    let absolute_quiet_threshold = noise_floor * COHERE_DTW_HOLLOW_ABSOLUTE_MEDIAN_MULTIPLE;
    for word in words.iter_mut().skip(1) {
        let raw_start = f64::from(word.start);
        let raw_end = f64::from(word.end);
        if raw_end - raw_start < COHERE_DTW_REFINE_MIN_SPAN_S {
            continue;
        }
        let start_s = raw_start.max(0.0).min(f64::from(duration_s));
        let end_s = raw_end.max(start_s).min(f64::from(duration_s));
        // A boundary word whose start maps to or past the last envelope frame
        // (common at a longform chunk end) must not overrun the frame array;
        // clamp both indices into range, letting the window-length guard below
        // bail the word without refinement rather than panic.
        let last_frame = levels.len() - 1;
        let frame_start = ((start_s / envelope_spf) as usize).min(last_frame);
        let frame_end = ((end_s / envelope_spf) as usize)
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
        // passage. Four conditions on the front half of the window:
        //   1. its *mean* level is below the noise floor (not just a fraction of
        //      frames -- a single loud blip in a quiet front must not pass);
        //   2. on a dense chunk its mean is also below a fraction of the noise
        //      floor, so a bed-level dip never reads as a pause;
        //   3. no *sustained* run of front frames crosses the silence ceiling, so
        //      a music floor never masquerades as a pause while a single
        //      bed-crackle frame does not void a legitimate pause;
        //   4. fewer than half its frames are above the floor (no sustained
        //      speech leaking into the front).
        let front_mean =
            (0..front_len).map(|i| f64::from(window[i])).sum::<f64>() / front_len as f64;
        let mut ceiling_sustained = false;
        let mut ceiling_run = 0usize;
        for &sample in window.iter().take(front_len) {
            ceiling_run = if f64::from(sample) > silence_ceiling {
                ceiling_run + 1
            } else {
                0
            };
            if ceiling_run >= COHERE_DTW_HOLLOW_CEILING_SUSTAIN_FRAMES {
                ceiling_sustained = true;
                break;
            }
        }
        let front_above = (0..front_len).filter(|&i| is_above(i)).count() as f64 / front_len as f64;
        if front_mean >= threshold
            || (is_dense_chunk && front_mean >= absolute_quiet_threshold)
            || ceiling_sustained
            || front_above > COHERE_DTW_HOLLOW_FRONT_ACTIVE_MAX as f64
        {
            continue;
        }
        // The onset: the first speech run (>= sustain frames above the floor)
        // preceded by a quiet run of at least the minimum silence length, inside
        // this word's own window.
        let mut onset_rel: Option<usize> = None;
        let mut index = 0usize;
        while index < window_len && onset_rel.is_none() {
            if is_above(index) {
                let mut run_end = index;
                while run_end + 1 < window_len && is_above(run_end + 1) {
                    run_end += 1;
                }
                if run_end - index + 1 >= COHERE_DTW_ONSET_SUSTAIN_FRAMES {
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
        let onset_s = ((frame_start + rel) as f64 * envelope_spf) as f32;
        let push = onset_s - raw_start as f32;
        // A word's start may only move forward into later audio, never backward.
        // The minimum push guards against a sub-calibration-error wiggle; the
        // maximum caps how deep into a pause we trust the envelope to lead us.
        if !(COHERE_DTW_ONSET_MIN_PUSH_S..=COHERE_DTW_ONSET_MAX_PUSH_S).contains(&push) {
            continue;
        }
        word.start = onset_s;
    }
    words
}

/// Pull a word the center fold let run past its speech into the trailing silence
/// back to its real audio offset.
///
/// The mirror of [`cohere_refine_dtw_word_onsets`]: where that pass recovers a
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
/// Two bed-backed realities the run search must survive. First, the fold's late
/// entry can park a word's window *behind* its own audio (the window starts in the
/// gap after the word), so the search reaches the
/// [`COHERE_DTW_OFFSET_EDGE_LEAD_S`] lead before the window; an offset that would
/// land before the word's own start is refused rather than inverting the window.
/// Second, a quiet word over a music bed sits far below the chunk-floor threshold,
/// so the search floor is the lower of the absolute floor and the region's own
/// peak minus [`COHERE_DTW_OFFSET_PEAK_FLOOR_MARGIN_DB`].
///
/// As with onsets, the back must stay below a small fraction of the chunk's
/// *peak*: a music or noise bed never reads as digital zero, so a low passage
/// inside a music-backed chunk is not a trusted trailing pause and a pull would
/// only move a word that was already acceptable. A single bed-crackle frame
/// crossing that ceiling does not void the pause; the crossing must be sustained
/// for [`COHERE_DTW_HOLLOW_CEILING_SUSTAIN_FRAMES`] frames.
fn cohere_refine_dtw_word_offsets(
    mut words: Vec<WordTimestamp>,
    audio_rms_frames: Option<&[f32]>,
    duration_s: f32,
) -> Vec<WordTimestamp> {
    let Some(levels) = audio_rms_frames else {
        return words;
    };
    if duration_s <= 0.0 || words.len() < 2 {
        return words;
    }
    let Some((noise_floor, clip_peak)) = cohere_dtw_envelope_floor_and_peak(levels) else {
        return words;
    };
    let threshold = noise_floor * 10.0_f64.powf(COHERE_DTW_ONSET_FLOOR_MARGIN_DB / 20.0);
    let silence_ceiling = cohere_dtw_edge_silence_ceiling(noise_floor, clip_peak);
    let envelope_spf = cohere_dtw_envelope_seconds_per_frame();
    let min_quiet_frames = (COHERE_DTW_OFFSET_MIN_SILENCE_S as f64 / envelope_spf).ceil() as usize;
    let is_dense_chunk =
        clip_peak / noise_floor.max(f64::EPSILON) < cohere_dtw_thin_floor_contrast();
    let absolute_quiet_threshold = noise_floor * COHERE_DTW_HOLLOW_ABSOLUTE_MEDIAN_MULTIPLE;
    let last_frame = levels.len() - 1;
    for word in words.iter_mut() {
        let raw_start = f64::from(word.start);
        let raw_end = f64::from(word.end);
        if raw_end - raw_start < COHERE_DTW_REFINE_MIN_SPAN_S {
            continue;
        }
        let start_s = raw_start.max(0.0).min(f64::from(duration_s));
        let end_s = raw_end.max(start_s).min(f64::from(duration_s));
        let frame_start = ((start_s / envelope_spf) as usize).min(last_frame);
        let frame_end = ((end_s / envelope_spf) as usize)
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
        // The same four hollow conditions as the onset pass, over the back half.
        let back_mean = (back_start..window_len)
            .map(|i| f64::from(window[i]))
            .sum::<f64>()
            / back_len as f64;
        let mut ceiling_sustained = false;
        let mut ceiling_run = 0usize;
        for &sample in window.iter().skip(back_start) {
            ceiling_run = if f64::from(sample) > silence_ceiling {
                ceiling_run + 1
            } else {
                0
            };
            if ceiling_run >= COHERE_DTW_HOLLOW_CEILING_SUSTAIN_FRAMES {
                ceiling_sustained = true;
                break;
            }
        }
        let back_above =
            (back_start..window_len).filter(|&i| is_above(i)).count() as f64 / back_len as f64;
        if back_mean >= threshold
            || (is_dense_chunk && back_mean >= absolute_quiet_threshold)
            || ceiling_sustained
            || back_above > COHERE_DTW_HOLLOW_BACK_ACTIVE_MAX as f64
        {
            continue;
        }
        // The offset: the last speech run followed by a quiet run of at least the
        // minimum silence length. The search covers the window plus the pre-window
        // lead (the fold's late entry can park the window behind the word's own
        // audio); run frames are counted above the search floor, and below-floor
        // gaps up to the tolerance do not split a run (coarticulatory
        // micro-silence in a decaying word tail).
        let region_start_s = (start_s - COHERE_DTW_OFFSET_EDGE_LEAD_S as f64).max(0.0);
        let region_frame_start = ((region_start_s / envelope_spf) as usize)
            .min(last_frame)
            .min(frame_start);
        let region = &levels[region_frame_start..frame_end + 1];
        let region_len = region.len();
        // The search floor: the lower of the absolute speech floor and the
        // region's own peak below the relative margin, so a quiet word over a
        // music bed is judged against its own level, not the chunk floor.
        let region_peak = region
            .iter()
            .fold(0.0f64, |peak, sample| peak.max(f64::from(*sample)));
        let floor = threshold
            .min(region_peak * 10.0_f64.powf(-COHERE_DTW_OFFSET_PEAK_FLOOR_MARGIN_DB / 20.0));
        // Gap-tolerant runs above the floor, gathered in order as (above-frame
        // count, last above frame).
        let mut runs: Vec<(usize, usize)> = Vec::new();
        let mut run_above = 0usize;
        let mut run_last = 0usize;
        let mut in_run = false;
        let mut below_gap = 0usize;
        for (i, &sample) in region.iter().enumerate() {
            if f64::from(sample) >= floor {
                if !in_run {
                    in_run = true;
                    run_above = 0;
                }
                run_above += 1;
                run_last = i;
                below_gap = 0;
            } else if in_run {
                below_gap += 1;
                if below_gap > COHERE_DTW_OFFSET_RUN_GAP_TOLERANCE_FRAMES {
                    runs.push((run_above, run_last));
                    in_run = false;
                    below_gap = 0;
                }
            }
        }
        if in_run {
            runs.push((run_above, run_last));
        }
        let mut offset_rel: Option<usize> = None;
        for &(run_count, run_end_index) in runs.iter().rev() {
            if run_count >= COHERE_DTW_OFFSET_SUSTAIN_FRAMES {
                let mut quiet = 0usize;
                let mut probe = run_end_index;
                while probe + 1 < region_len && f64::from(region[probe + 1]) < floor {
                    probe += 1;
                    quiet += 1;
                }
                if quiet >= min_quiet_frames {
                    offset_rel = Some(run_end_index);
                    break;
                }
            }
        }
        let Some(rel) = offset_rel else {
            continue;
        };
        // The run ends at `rel`; one frame past it is where the silence begins. An
        // offset before the word's own start would invert the window (its audio
        // sits entirely earlier than the window -- the onset pass's domain), so
        // such a word is refused.
        let offset_s = ((region_frame_start + rel + 1) as f64 * envelope_spf) as f32;
        if f64::from(offset_s) < raw_start {
            continue;
        }
        let pull = raw_end - f64::from(offset_s);
        // A word's end may only move earlier into prior audio, never past it. The
        // minimum pull guards against a sub-calibration-error wiggle; the maximum
        // caps how deep into a pause we trust the envelope to lead us.
        if !(COHERE_DTW_OFFSET_MIN_PULL_S..=COHERE_DTW_OFFSET_MAX_PULL_S).contains(&(pull as f32)) {
            continue;
        }
        word.end = offset_s;
    }
    words
}

/// Align per-token cross-attention frame rows to the audio timeline with a
/// monotone DTW pass and fold them into word timestamps, mirroring whisper's
/// no-timestamp DTW degrade. `token_alignments` pairs each generated (non-EOT)
/// token with its decoder's last-layer cross-attention frame row. Cohere
/// decodes `<|notimestamps|>` so there are no timestamp tokens: the DTW window
/// is bracketed on the content tokens' own attention peaks (leading/trailing
/// silence the model ignored is never bracketed by a peak, so it stays off the
/// timeline), and each content-token peak still owns its real audio span.
///
/// Once the band is aligned the window goes through the same refinement chain as
/// whisper's DTW path: the pre-fold reanchor that pulls word-final punctuation
/// off a trailing pause -> the center fold (punctuation trust radius, interior
/// half-span) -> the per-word span cap -> the onset refiner -> the offset
/// refiner. The audio-driven passes read `audio_rms_frames`, the chunk's 0.02 s
/// RMS envelope (see [`cohere_dtw_word_audio_rms_frames`]); when it is absent
/// every one of them is a no-op and the fold's own windows stand. The caller's
/// symmetric window pad runs after this function returns, so it also covers the
/// degrade tiers.
pub(crate) fn cohere_dtw_word_timestamps<E>(
    token_alignments: &[(u32, Vec<f32>)],
    metadata: CohereTranscribeExecutionMetadata,
    generated_probabilities: &[f32],
    duration: f32,
    audio_onset_seconds: f32,
    audio_rms_frames: Option<&[f32]>,
    decode_text: &dyn Fn(&[u32]) -> Result<String, E>,
) -> Result<Vec<WordTimestamp>, E> {
    let frame_count = token_alignments
        .first()
        .map(|alignment| alignment.1.len())
        .unwrap_or(0);
    if frame_count == 0 {
        return Ok(Vec::new());
    }
    // The three strided convs (k3,s2,p1) sub-sample the mel axis 8x, so one
    // encoder frame is 8 mel hops of audio; frames map to absolute wall-clock
    // time from clip start at that rate, not a fraction of `duration`.
    let hop = metadata.hop_length;
    let sample_rate = metadata.sample_rate_hz;
    if hop == 0 || sample_rate == 0 {
        return Ok(Vec::new());
    }
    let seconds_per_frame = 8.0 * hop as f32 / sample_rate as f32;
    if !seconds_per_frame.is_finite() || seconds_per_frame <= 0.0 {
        return Ok(Vec::new());
    }
    let duration = duration.max(0.0);
    let full_window = token_alignments
        .iter()
        .map(|alignment| alignment.1.clone())
        .collect::<Vec<Vec<f32>>>();
    if std::env::var_os("OPENASR_COHERE_DEBUG_CROSS").is_some() {
        for (row_index, alignment) in token_alignments.iter().enumerate() {
            let text = decode_text(&[alignment.0]).unwrap_or_default();
            let (peak_frame, &peak_value) = alignment
                .1
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap_or((0, &0.0));
            let mut top_vec = alignment
                .1
                .iter()
                .enumerate()
                .map(|(frame, value)| (frame, *value))
                .collect::<Vec<_>>();
            top_vec.sort_by(|a, b| b.1.total_cmp(&a.1));
            top_vec.truncate(4);
            let top = top_vec
                .iter()
                .map(|(frame, value)| format!("{frame}@{value:.4}"))
                .collect::<Vec<_>>()
                .join(",");
            let peak_secs = peak_frame as f32 * seconds_per_frame;
            eprintln!(
                "cohere cross row {row_index} token={} text={text:?} peak@{peak_frame}({peak_secs:.2}s)={peak_value:.4} top=[{top}]",
                alignment.0
            );
        }
    }
    let is_content: Vec<bool> = token_alignments
        .iter()
        .map(|alignment| {
            decode_text(&[alignment.0]).is_ok_and(|text| token_text_carries_speech(&text))
        })
        .collect();
    // Cohere's last-layer cross-attention is diffuse and front-loaded on real
    // audio: several unrelated tokens share one early "priming" frame peak, so
    // the per-token peaks are not a clean monotone order and the DTW pass
    // over-spreads the first words (measured TempErr worse than the uniform
    // baseline on every clip available). Only trust the DTW word spans when the
    // content-token attention peaks are order-aligned; otherwise return empty
    // and let the caller keep the proven uniform post-hoc timestamps.
    //
    // `dtw_window` starts as the raw attention. When the raw peak order
    // zig-zags, one more chance is given after masking the dominant early
    // "sinks": an early frame that is the global max for a dominant share of
    // the rows is a shared diffuse-attention artifact, not evidence for where
    // any one token is spoken. Stripping it lets each masked row's
    // next-strongest frame (its real region) surface, which restores a
    // monotone peak order on clips where every non-artifact row already
    // pointed the right way. A zigzag with no detectable sink, or a stripped
    // signal the tolerant tier rejects, goes to the fallback tier instead
    // (long window -> per-word peak placement, short -> the caller's uniform
    // baseline), never the DTW pass.
    // `stripped_sinks` holds the early frames the strip removed, so the DTW
    // band below can skip them when it brackets the speech. `None` when the
    // raw window is aligned as-is.
    let mut stripped_sinks: Option<Vec<u32>> = None;
    let dtw_window: Option<Vec<Vec<f32>>> =
        if cross_attention_peaks_order_aligned(&full_window, &is_content) {
            None
        } else {
            let detected =
                detect_dominant_early_sinks(&full_window, &is_content).unwrap_or_default();
            if detected.is_empty() {
                // No early artifact explains the zigzag: the per-token peaks
                // genuinely jump back and forth and no masking can expose a clean
                // signal, so the window takes the fallback tier (long -> peak
                // fallback, short -> uniform), never the DTW pass.
                if band_duration_seconds(&full_window, seconds_per_frame)
                    >= COHERE_DTW_PEAK_FALLBACK_MIN_SECONDS
                {
                    if std::env::var_os("OPENASR_COHERE_DEBUG_CROSS").is_some() {
                        eprintln!("cohere cross gate: peaks not aligned, using peak fallback");
                    }
                    return cohere_peak_fallback_word_timestamps(
                        &full_window,
                        &is_content,
                        token_alignments,
                        generated_probabilities,
                        seconds_per_frame,
                        audio_rms_frames,
                        duration,
                        decode_text,
                    );
                }
                if std::env::var_os("OPENASR_COHERE_DEBUG_CROSS").is_some() {
                    eprintln!("cohere cross gate: peaks not order-aligned, using uniform");
                }
                return Ok(Vec::new());
            }
            stripped_sinks = Some(detected.clone());
            let window = mask_frames_early(&full_window, &detected);
            if cross_attention_peaks_order_aligned(&window, &is_content) {
                if std::env::var_os("OPENASR_COHERE_DEBUG_CROSS").is_some() {
                    eprintln!("cohere cross gate: order restored after sink strip, using DTW");
                }
                Some(window)
            } else if content_backward_fraction(&window, &is_content)
                <= COHERE_DTW_MAX_BACKWARD_PAIR_FRACTION
                && band_duration_seconds(&full_window, seconds_per_frame)
                    >= COHERE_DTW_TOLERANT_MIN_BAND_SECONDS
            {
                // Not perfectly monotone, but the backward jumps are a tiny
                // minority of content pairs: the strip exposed a mostly-clean
                // left-to-right signal the DTW can be trusted for. Scoped to long
                // windows (cohere's 30s long-form chunks) where the pauses are
                // actually measurable.
                if std::env::var_os("OPENASR_COHERE_DEBUG_CROSS").is_some() {
                    eprintln!(
                        "cohere cross gate: order restored after sink strip (tolerant), using DTW"
                    );
                }
                Some(window)
            } else if band_duration_seconds(&full_window, seconds_per_frame)
                >= COHERE_DTW_PEAK_FALLBACK_MIN_SECONDS
            {
                // The strip was detected and the gate rejected the stripped
                // signal, but the window is long enough that per-word peak
                // placement beats the uniform baseline.
                if std::env::var_os("OPENASR_COHERE_DEBUG_CROSS").is_some() {
                    eprintln!("cohere cross gate: peaks not aligned, using peak fallback");
                }
                return cohere_peak_fallback_word_timestamps(
                    &full_window,
                    &is_content,
                    token_alignments,
                    generated_probabilities,
                    seconds_per_frame,
                    audio_rms_frames,
                    duration,
                    decode_text,
                );
            } else {
                if std::env::var_os("OPENASR_COHERE_DEBUG_CROSS").is_some() {
                    eprintln!("cohere cross gate: peaks not order-aligned, using uniform");
                }
                return Ok(Vec::new());
            }
        };
    // The band is derived from the unmasked raw attention except for the strip
    // artifact frames, which are skipped: on a stripped window the raw rows
    // still carry the sink at the window start, and bracketing the band on it
    // would begin the DTW in the leading silence (see `speech_band_from_rows`).
    // Masking the frames to zero instead would corrupt the earliest-peak
    // bound for the same reason the strip is only ever applied to the DTW
    // window, not the band.
    let (band_start, band_end) =
        speech_band_from_rows(&full_window, &is_content, stripped_sinks.as_deref()).map_or_else(
            move || {
                (
                    0usize,
                    ((duration / seconds_per_frame).ceil() as usize).clamp(1, frame_count),
                )
            },
            |(start, end)| (start, end.clamp(start + 1, frame_count)),
        );
    if std::env::var_os("OPENASR_COHERE_DEBUG_CROSS").is_some() {
        eprintln!(
            "cohere cross band=({band_start},{band_end}) frames={frame_count} spf={seconds_per_frame}"
        );
    }
    // `speech_band_from_rows` brackets the band on the post-sink-substitution
    // per-token peaks. When the leading content token's own dominant frame is
    // the shared sink, the substitution moves that token's peak to its
    // next-strongest frame, and the band's earliest bound collapses to a later
    // word: the band start then sits far past the first word, and the DTW's
    // entry frame (which anchors the first word's center) inherits that lateness
    // (measured up to ~4.4s on rye's lead). The tell-tale of that displacement
    // is the first content token's *raw* (unmasked) peak landing far short of
    // the band start -- a genuine leading word whose peak the strip hid. When
    // that holds, the band start is pulled back to the chunk's measured audio
    // onset (`audio_onset_seconds`), which for a stripped lead sits at or before
    // the hidden peak. When the first token's raw peak is already inside the
    // band, the band reflects real attention and is left untouched (the onset
    // would only push the first word earlier through lead-in room tone). A
    // non-finite onset or a window with no sink strip likewise leaves the band
    // as computed.
    let first_content_raw_peak = is_content.iter().position(|&flag| flag).and_then(|index| {
        let row = &full_window[index];
        row.iter()
            .enumerate()
            .filter(|&(_, value)| value.is_finite() && *value > 0.0)
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(frame, _)| frame)
    });
    let onset_frame = (audio_onset_seconds / seconds_per_frame).floor() as usize;
    let first_peak_displaced = first_content_raw_peak
        .is_some_and(|first_peak| first_peak + DTW_SPEECH_BAND_MARGIN_FRAMES < band_start);
    // The displacement anchor is only meant to correct a sink substitution that
    // moved the first word's peak to a *neighbouring* word: the true onset then
    // sits a fraction of a second before the post-substitution band. When the
    // measured energy onset instead sits many seconds ahead of the band, the
    // onset is a different region than the band brackets, and anchoring to it
    // would slide a correct band across that non-speech gap and re-park the
    // first word in the music. Only anchor when the onset is within this gap
    // of the corrected band.
    let onset_disp_gap_ok = band_start.saturating_sub(onset_frame) as f32 * seconds_per_frame
        <= ONSET_DISP_MAX_ANCHOR_GAP_SECONDS;
    let band_start = if stripped_sinks.is_some()
        && audio_onset_seconds.is_finite()
        && first_peak_displaced
        && onset_frame < band_start
        && onset_disp_gap_ok
    {
        if std::env::var_os("OPENASR_COHERE_DEBUG_CROSS").is_some() {
            eprintln!(
                "cohere cross first-token disp: band={band_start} onset={onset_frame} -> anchor {onset_frame}"
            );
        }
        onset_frame
    } else if stripped_sinks.is_some()
        && audio_onset_seconds.is_finite()
        && onset_frame > band_start
        && onset_frame.saturating_sub(band_start) >= DTW_SPEECH_BAND_MARGIN_FRAMES
    {
        // Counterpart of the displacement case above, for a chunk that opens
        // with real silence but whose sink-masked first content token still
        // leaves the band bracketing the chunk front (a residual early-attention
        // frame, not the stripped sink). The DTW's start-early bias then walks
        // that leading silence and parks the first word at the silent chunk
        // start instead of where speech begins. Advancing the band start to the
        // measured audio onset makes the DTW begin at the first real energy.
        //
        // Two guards keep this inert where it would be wrong. `onset_frame >
        // band_start` requires the measured onset to sit *ahead* of the band:
        // a chunk that genuinely opens with speech has onset 0 at/inside the
        // band, so the advance never touches it (nor a band that legitimately
        // starts before the onset, e.g. on a stripped leading word). The margin
        // guard ignores a sub-margin gap that is not a meaningful leading
        // silence.
        if std::env::var_os("OPENASR_COHERE_DEBUG_CROSS").is_some() {
            eprintln!(
                "cohere cross leading-silence advance: band={band_start} onset={onset_frame} -> {onset_frame}"
            );
        }
        onset_frame
    } else {
        band_start
    };
    if std::env::var_os("OPENASR_COHERE_DEBUG_CROSS").is_some() {
        eprintln!("cohere cross band after onset-anchor = ({band_start},{band_end})");
    }
    let attention: Vec<Vec<f32>> = dtw_window
        .as_ref()
        .unwrap_or(&full_window)
        .iter()
        .map(|row| row[band_start.min(row.len())..band_end.min(row.len())].to_vec())
        .collect();
    let Some(spans) = dtw_align_token_frames(&attention) else {
        return Ok(Vec::new());
    };
    // The monotone DTW path *enters* each token's row at the frame where the
    // path first reaches it, which for a token whose attention is a peak is
    // that peak -- i.e. roughly the MIDDLE of the spoken word, not its start.
    // Treating that entry frame as the word's start (the span-tiling `frame_end`
    // of the previous token) therefore places every word start about a half
    // word later than the word is actually spoken (measured +0.5s start offset
    // on every long-form chunk, with the word landing past the truth window).
    //
    // The entry frame is therefore the word's *center*, not its start. Each
    // token is folded into a word at its entry-frame center and the boundary
    // between two words falls part-way across the gap between their centers
    // (`COHERE_DTW_BOUNDARY_FRACTION`, pulled earlier by
    // `COHERE_DTW_ONSET_LEAD_SECONDS`) -- the same center-fold the
    // center-of-mass / peak-fallback paths use -- which keeps the timeline
    // monotone and non-overlapping while placing a word's *start* before its
    // center (i.e. before the peak), at the true speech onset. The band-anchored
    // first/last bounds below replace the tiling's "run to the window edge"
    // behaviour, so a mid-chunk band does not stretch its first/last word
    // across the band's surrounding silence.
    let probabilities_aligned = generated_probabilities.len() == token_alignments.len();
    let band_start_secs = band_start as f32 * seconds_per_frame;
    let band_end_secs = band_end as f32 * seconds_per_frame;
    let token_times: Vec<Seq2SeqTokenTime> = token_alignments
        .iter()
        .enumerate()
        .zip(spans.iter())
        .map(|((index, alignment), span)| Seq2SeqTokenTime {
            token_id: alignment.0,
            center_seconds: (span.frame_start.saturating_add(band_start)) as f32
                * seconds_per_frame,
            probability: probabilities_aligned.then(|| generated_probabilities[index]),
        })
        .collect();
    // A word-final punctuation token carries no audio of its own and its center
    // can sit in the pause after its word; pull it back to the word's own offset
    // before the fold turns these centers into word windows.
    let token_times = cohere_reanchor_dtw_token_centers(token_times, decode_text, audio_rms_frames);
    let onset_lead = cohere_dtw_onset_lead(
        band_end_secs - band_start_secs,
        is_content.iter().filter(|flag| **flag).count(),
    );
    // `max_edge_word_span` stays infinite so the fold anchors the first word's
    // start at the band start and the last word's end at the band end, exactly as
    // this fold has always been calibrated; what keeps those edges off the
    // surrounding silence is now the audible refiners below rather than a
    // blind positional clamp. The punctuation trust radius stops a word-final
    // comma the path lingered seconds into a following pause from dragging the
    // word's center off its speech, and the interior half-span bounds how much of
    // an adjacent pause a word may own on either side of its own center.
    let words = seq2seq_word_timestamps_from_token_times(
        &token_times,
        band_start_secs,
        band_end_secs,
        BuiltinDecodePolicySeq2SeqTextPostprocessKind::Identity,
        decode_text,
        COHERE_DTW_BOUNDARY_FRACTION,
        onset_lead,
        f32::INFINITY,
        cohere_dtw_punctuation_trust_radius_seconds(),
        cohere_dtw_max_interior_half_span_seconds(),
    )?;
    // `word_centers_to_timestamps` anchors the first word's start to
    // `segment_start` (the band start) and the last word's end to
    // `segment_end` (the band end), so a mid-chunk band does not stretch its
    // first/last word across the band's surrounding silence; the boundary
    // between two centers (`COHERE_DTW_BOUNDARY_FRACTION`, pulled earlier by
    // `COHERE_DTW_ONSET_LEAD_SECONDS`) lands where the following word's speech
    // begins. The timeline is monotone and non-overlapping by construction, so
    // the remaining corrections are all audible: cap any word that swallowed a
    // real pause, then pull the hollow-front and hollow-back words onto the
    // audio they actually sit on.
    let words = cohere_cap_dtw_word_spans(words, seconds_per_frame);
    let words = cohere_refine_dtw_word_offsets(
        cohere_refine_dtw_word_onsets(words, audio_rms_frames, duration),
        audio_rms_frames,
        duration,
    );
    if std::env::var_os("OPENASR_COHERE_DEBUG_CROSS").is_some() {
        eprintln!(
            "cohere cross dtw midpoint fold: {} words over band {}..{}s",
            words.len(),
            band_start_secs,
            band_end_secs
        );
    }
    Ok(words)
}

/// Order-gate fallback for long, pause-heavy windows: place each token at its
/// own strongest cross-attention frame and fold those into word timestamps.
///
/// Called only when the DTW order gate (strict and tolerant) has rejected the
/// window but the window is long enough that the stretch would hurt (see
/// [`COHERE_DTW_PEAK_FALLBACK_MIN_SECONDS`]). Unlike the DTW tiling or the
/// uniform baseline -- both of which tile the window contiguously and cannot
/// express a gap -- each word lands where its attention is strongest, so the
/// midpoints between word centers fall where the attention falls and a short
/// utterance inside a long chunk stays short. The timeline is bounded at the
/// earliest frame (the clip start) and the latest *content-token* peak: a
/// trailing punctuation token's diffuse peak (attending into the ignored
/// padding) must not stretch the final word's end toward the far edge of the
/// chunk. Returns `Ok(Vec::new())` when no content-token peak exists, so the
/// caller keeps the uniform baseline rather than emitting a single degenerate
/// word.
///
/// This tier gets the same per-word span cap and the same audible edge refiners
/// as the DTW path. The cap is load-bearing here, not merely tidying: this fold's
/// boundary fraction is the midpoint and its first word is anchored to the
/// window front, so a first token whose peak sits seconds into the chunk leaves
/// that word owning the whole lead-in. On a 30s longform chunk that is a
/// ~15s-wide head word, and because the longform assembler decides its seam trims
/// and phantom/reread drops from word windows, one such window changes which
/// words survive at the seam -- so the emitted *text* shifts, not just its
/// timing.
fn cohere_peak_fallback_word_timestamps<E>(
    full_window: &[Vec<f32>],
    is_content: &[bool],
    token_alignments: &[(u32, Vec<f32>)],
    generated_probabilities: &[f32],
    seconds_per_frame: f32,
    audio_rms_frames: Option<&[f32]>,
    duration: f32,
    decode_text: &dyn Fn(&[u32]) -> Result<String, E>,
) -> Result<Vec<WordTimestamp>, E> {
    let token_count = token_alignments.len();
    if token_count == 0 {
        return Ok(Vec::new());
    }
    let mut token_times = Vec::with_capacity(token_count);
    let mut last_content_peak_center: Option<f32> = None;
    for (index, alignment) in token_alignments.iter().enumerate() {
        let row = &full_window[index];
        let Some((peak_frame, &peak_value)) = row
            .iter()
            .enumerate()
            .filter(|&(_, &value)| value.is_finite())
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
        else {
            return Ok(Vec::new());
        };
        let center_seconds = (peak_frame as f32) * seconds_per_frame;
        if is_content.get(index).copied().unwrap_or(false) && peak_value > 0.0 {
            last_content_peak_center = Some(match last_content_peak_center {
                Some(existing) => existing.max(center_seconds),
                None => center_seconds,
            });
        }
        token_times.push(Seq2SeqTokenTime {
            token_id: alignment.0,
            center_seconds,
            probability: generated_probabilities.get(index).copied(),
        });
    }
    let Some(segment_end) = last_content_peak_center else {
        return Ok(Vec::new());
    };
    // `max_edge_word_span` stays infinite and the interior half-span with it, so
    // this tier's seams stay exactly the equidistant split it was tuned with; the
    // span cap below is what bounds the resulting windows. The punctuation trust
    // radius does apply: a word-final punctuation token's peak can land in the
    // silence the window brackets past the speech, and a radius keeps it from
    // dragging the word's center back into it.
    let words = seq2seq_word_timestamps_from_token_times(
        &token_times,
        0.0,
        segment_end,
        BuiltinDecodePolicySeq2SeqTextPostprocessKind::Identity,
        decode_text,
        MIDPOINT_BOUNDARY_FRACTION,
        NO_ONSET_LEAD,
        f32::INFINITY,
        cohere_dtw_punctuation_trust_radius_seconds(),
        f32::INFINITY,
    )?;
    Ok(cohere_refine_dtw_word_offsets(
        cohere_refine_dtw_word_onsets(
            cohere_cap_dtw_word_spans(words, seconds_per_frame),
            audio_rms_frames,
            duration,
        ),
        audio_rms_frames,
        duration,
    ))
}

/// Whether the per-token cross-attention peaks, read in decode order, form a
/// monotone (non-decreasing) frame sequence over the content tokens, allowing
/// ties and single-frame jitter.
///
/// A clean alignment signal has each content token's strongest frame at or
/// after the previous content token's strongest frame (the speech is left to
/// right, so the attention follows it). Cohere's last-layer cross-attention is
/// instead diffuse and front-loaded: several unrelated tokens share one early
/// "priming" frame as their global max, so the peak sequence zig-zags and the
/// DTW pass over-spreads the first words past where they are spoken (measured
/// worse than the uniform post-hoc baseline on every available clip). Only a
/// non-zig-zag peak sequence is a trustworthy DTW input; anything else should
/// fall back to the uniform timestamps. Returns `true` (vacuously aligned) when
/// fewer than two content peaks can be formed, leaving the decision to the DTW
/// pass itself.
fn cross_attention_peaks_order_aligned(attention: &[Vec<f32>], is_content: &[bool]) -> bool {
    const TOLERANCE_FRAMES: usize = 1;
    let mut previous_peak: Option<usize> = None;
    for (index, row) in attention.iter().enumerate() {
        if !is_content.get(index).copied().unwrap_or(false) || row.is_empty() {
            continue;
        }
        let peak = row
            .iter()
            .enumerate()
            .filter(|&(_, &value)| value.is_finite())
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(frame, _)| frame);
        let Some(peak) = peak else {
            continue;
        };
        if let Some(previous) = previous_peak {
            // A backward jump of two or more frames is a zig-zag (ties and a
            // single-frame jitter are tolerated).
            if peak
                .checked_add(TOLERANCE_FRAMES)
                .is_some_and(|shifted| shifted < previous)
            {
                return false;
            }
        }
        previous_peak = Some(peak);
    }
    true
}

/// Search horizon, in frames, for the dominant-early-sink mask. ~0.8s at the
/// family's 0.08s/frame; the shared "priming" attention sits at the start of
/// the window, so only early frames are candidates. Late peaks are never
/// masked because they may carry the one token's real speech location.
const SINK_STRIP_SEARCH_FRAMES: usize = 10;

/// Fraction of forward-vs-backward content-token pairs tolerated in the
/// post-sink-strip "mostly-monotone" tier of the order gate. The strict
/// re-test rejects any backward jump of 2+ frames. The tolerant tier admits
/// windows where such jumps affect at most this fraction of content pairs.
/// Windows that fall short of the tolerant tier fall back to the uniform baseline.
/// 0.10 was tuned against the old span-tiling DTW; the current entry-frame
/// center fold is robust to a modest residual zigzag, and raising this admits
/// the short clips whose DTW entry centers still land well inside the truth
/// windows. Measured in-window coverage only rises.
const COHERE_DTW_MAX_BACKWARD_PAIR_FRACTION: f32 = 0.25;

/// Minimum DTW band duration, in seconds, before the post-sink-strip
/// "mostly-monotone" tier of the order gate is allowed to admit a window.
/// The tolerant tier exists to catch windows whose raw peak order zig-zags from
/// the shared early sink but whose post-strip signal is still mostly
/// left-to-right. A band this short has too few frames for a meaningful gap, so
/// bands below the floor still fall back to the uniform baseline. The original
/// 20s floor predates the current fold: with the old span-tiling DTW, a 15-25s
/// window accumulated enough drift to time worse than uniform, so short windows
/// were deliberately kept uniform. The current entry-frame center fold lands
/// well inside the truth windows on those shorter bands), so the floor drops to
/// 10s to admit them; anything under 10s stays on the uniform baseline.
const COHERE_DTW_TOLERANT_MIN_BAND_SECONDS: f32 = 10.0;

/// Minimum window duration, in seconds, before the order-gate fallback switches
/// from "return empty -> uniform" to "place each word at its strongest
/// attention peak". On cohere's 30s long-form chunks the attention-peak order
/// on a pause-heavy decode is too zig-zaggy for the DTW pass, and neither the
/// DTW tiling nor the uniform fallback can leave a real gap: both stretch the
/// few seconds of speech across the whole 30s window (measured start-end span
/// error of up to ~14s on the worst windows). Placing each word at its own
/// strongest frame keeps the words where the attention is and lets the midpoints
/// between them fall naturally, so a 1s utterance in a 30s chunk stays 1s wide
/// instead of stretching to 30s. The threshold matches the tolerant tier's
/// band-length guard: below it the stretch is small enough that the plain
/// uniform baseline (which the caller emits on empty return) remains the safer
/// choice, keeping short-clip behavior unchanged.
const COHERE_DTW_PEAK_FALLBACK_MIN_SECONDS: f32 = 20.0;

/// Maximum duration, in seconds, a single DTW-tiled word span may keep.
/// The monotone tiling gives whatever frames lie between a token's entry and
/// the next token's entry to the earlier token, so the word preceding a real
/// pause swallows the whole pause as its span (measured up to ~18s on
/// pause-heavy 30s chunks, where the ground truth's longest word is under
/// 3s). The cap is set well above any plausible spoken word (longest
/// legitimate words observed in the test clips are ~1.7s) and far below the
/// runaway regime; only the span's tail is trimmed, never its start.
const COHERE_DTW_MAX_WORD_SPAN_SECONDS: f32 = 1.5;

/// Where the boundary between two consecutive DTW words lands, as a fraction of
/// the gap between their centers: `prev + fraction * (this - prev)`. The
/// equidistant midpoint (0.5, the plain fold) assumes a word's center -- the
/// DTW path's entry frame, where its cross-attention peaks -- is the moment the
/// word is *mid-utterance*. Cohere's cross-attention concentrates a bit after
/// the word is spoken, so the midpoint puts every word start a little late,
/// which is why whole runs of words drift past the truth window and appear "off
/// by one". Moving the boundary closer to the word's own center (0.35) pulls
/// each start toward the true speech onset.
const COHERE_DTW_BOUNDARY_FRACTION: f32 = 0.35;

/// Baseline seconds by which each DTW center is placed earlier before the
/// boundary split, for a band speaking at or below the density knee
/// ([`COHERE_DTW_LEAD_DENSITY_KNEE_PER_SEC`]). Cohere's cross-attention centers
/// a token slightly after its speech onset, so the boundary split
/// ([`COHERE_DTW_BOUNDARY_FRACTION`]) lands a little late and the fold pulls
/// each center earlier by this baseline. Denser bands add lead on top of this
/// via [`cohere_dtw_onset_lead`]: the DTW entry frame -- the word's center --
/// lands a growing fixed amount past the true onset as the speaking rate
/// climbs, which a fixed lead cannot correct for both sparse and dense bands.
/// The band start remains the floor: the first word still starts at the band
/// start, never before the spoken audio. The onset-lead and the boundary
/// fraction work together: more lead lets the fraction sit closer to the plain
/// midpoint, so the two effects are not double-counted. A flat 0.20 (the old
/// behavior) was tuned to mid-density bands and left dense bands -- e.g. a
/// rapid aside inside a long pause-heavy window -- with every word start a
/// third of a second late, just past its truth window.
const COHERE_DTW_ONSET_LEAD_SECONDS: f32 = 0.20;

/// Maximum seconds between the measured audio onset and the (post sink-strip)
/// band start for the first-token-displacement anchor to trust the onset as
/// the band floor. A larger gap means the onset is a separate region from the
/// one the cross-attention brackets (a music/ambient bed in a long-form chunk
/// whose real speech opens much later), in which case anchoring the first
/// word to the onset slides a correct band across that non-speech gap. Set
/// generously above the genuine sink-displacement case (a fraction of a
/// second) and far below the music-bed regime.
const ONSET_DISP_MAX_ANCHOR_GAP_SECONDS: f32 = 3.0;

/// Words per second of band audio above which the measured late-onset bias
/// grows and a larger onset lead is warranted.
const COHERE_DTW_LEAD_DENSITY_KNEE_PER_SEC: f32 = 2.4;

/// Rate (in seconds of added lead per extra word/second above the knee) by
/// which the onset lead grows with band density. A high slope (0.10) added so
/// much early-shift on densely-spoken bands (rapid conversation, ~3-4 words/s)
/// that whole runs of words landed past their truth window.
const COHERE_DTW_LEAD_DENSITY_SLOPE: f32 = 0.01;

/// Upper bound on the density-scaled onset lead. Above this the added lead
/// outruns the true onset on dense bands as quickly as it helps.
const COHERE_DTW_ONSET_LEAD_MAX_SECONDS: f32 = 0.42;

/// The onset lead for one DTW band, scaled by how densely its words pack.
///
/// Cohere's measured late-onset bias is density-dependent: on a slow band
/// (<= [`COHERE_DTW_LEAD_DENSITY_KNEE_PER_SEC`] words/s) the baseline
/// [`COHERE_DTW_ONSET_LEAD_SECONDS`] suffices, but as the speaking rate climbs
/// the DTW centers land a growing fixed amount past the true onset, so the
/// lead grows at [`COHERE_DTW_LEAD_DENSITY_SLOPE`] per extra word/second up to
/// [`COHERE_DTW_ONSET_LEAD_MAX_SECONDS`]. Whisper's DTW fold keeps its own,
/// separate, flat lead (see `whisper_dtw_onset_lead`); cohere's per-band
/// density term is an independent addition on this path (the reason dense,
/// rapid-aside windows such as the `sleepy` clip recovered in-window coverage
/// while their sparse neighbours held) and is not shared with whisper.
///
/// The density is in *content-token* count per second of band audio, matching
/// the window's own decoded output (cohere decodes `<|notimestamps|>` so the
/// content-token count of the band carries the same signal as a decoded word
/// count, without re-decoding the whole window).
/// Tuning of the onset-lead curve, read once per call so a deployment can
/// retune it without a rebuild. Each element falls back to its compiled
/// default when unset or unparsable, so a bare environment is byte-identical
/// to the historical behavior. The lead shifts every word start earlier
/// (DTW entry frames are near-onset already), so the four points dial how far
/// the fold reaches before the true speech onset.
///
/// Kept as a plain struct with a `Default` that mirrors the compiled constants
/// so the lead curve is a pure, env-free function (and the no-override path
/// stays byte-identical to the historical constants); see
/// `cohere_dtw_onset_lead_for`. The run-time overrides exist so a deployment
/// can retune the curve per corpus without a rebuild.
#[derive(Debug, Clone, Copy)]
struct CohereDtwLeadTuning {
    baseline: f32,
    knee: f32,
    slope: f32,
    maximum: f32,
}

impl Default for CohereDtwLeadTuning {
    fn default() -> Self {
        Self {
            baseline: COHERE_DTW_ONSET_LEAD_SECONDS,
            knee: COHERE_DTW_LEAD_DENSITY_KNEE_PER_SEC,
            slope: COHERE_DTW_LEAD_DENSITY_SLOPE,
            maximum: COHERE_DTW_ONSET_LEAD_MAX_SECONDS,
        }
    }
}

/// The onset lead for a tuning curve and one band: a flat baseline up to the
/// knee, then a linear growth with band density capped at the maximum. Pure
/// and env-free so the curve's shape is unit-testable.
fn cohere_dtw_onset_lead_for(
    tuning: &CohereDtwLeadTuning,
    band_seconds: f32,
    word_count: usize,
) -> f32 {
    let band_seconds = band_seconds.max(0.05);
    let density = word_count as f32 / band_seconds;
    let excess = (density - tuning.knee).max(0.0);
    (tuning.baseline + tuning.slope * excess).min(tuning.maximum)
}

/// The tuning curve to apply now, honoring the deployment env overrides. Each
/// element falls back to its compiled default (see
/// `CohereDtwLeadTuning::default`) when unset or unparsable, so a bare
/// environment is byte-identical to the historical behavior.
fn cohere_dtw_lead_tuning() -> CohereDtwLeadTuning {
    let default = CohereDtwLeadTuning::default();
    let read = |name: &str, fallback: f32| {
        std::env::var(name)
            .ok()
            .and_then(|raw| raw.parse::<f32>().ok())
            .unwrap_or(fallback)
    };
    CohereDtwLeadTuning {
        baseline: read("OPENASR_COHERE_DTW_LEAD_BASE_SECONDS", default.baseline),
        knee: read("OPENASR_COHERE_DTW_LEAD_KNEE_PER_SEC", default.knee),
        slope: read("OPENASR_COHERE_DTW_LEAD_SLOPE", default.slope),
        maximum: read("OPENASR_COHERE_DTW_LEAD_MAX_SECONDS", default.maximum),
    }
}

fn cohere_dtw_onset_lead(band_seconds: f32, word_count: usize) -> f32 {
    cohere_dtw_onset_lead_for(&cohere_dtw_lead_tuning(), band_seconds, word_count)
}

/// Limit how long a single DTW word may run.
///
/// In the midpoint fold a word's edge is the midpoint between its own center
/// (its token's DTW entry) and its neighbor's center. When a real pause sits
/// between two utterances, the word on each side of the pause extends halfway
/// across it (measured ~6-7s on pause-heavy 30s chunks, where the ground
/// truth's longest word is under 3s). The cap trims each word's tail at
/// `start + 1.5s`, well above any plausible spoken word and far below the
/// phantom-tail regime, restoring an explicit gap at the pause while keeping
/// the word's own (peak-derived) start untouched.
fn cohere_cap_dtw_word_spans(
    words: Vec<WordTimestamp>,
    seconds_per_frame: f32,
) -> Vec<WordTimestamp> {
    const MAX_SECONDS: f32 = COHERE_DTW_MAX_WORD_SPAN_SECONDS;
    let limit = MAX_SECONDS.max(seconds_per_frame);
    let mut capped = 0usize;
    let mut largest = f32::NAN;
    let mut capped_words = words;
    for word in &mut capped_words {
        let span = word.end - word.start;
        largest = largest.max(span);
        if span > limit {
            word.end = word.start + limit;
            capped += 1;
        }
    }
    if std::env::var_os("OPENASR_COHERE_DEBUG_CROSS").is_some() {
        eprintln!(
            "cohere cross dtw span cap: {capped} of {} words capped at {MAX_SECONDS}s (largest pre-cap {largest:?}s)",
            capped_words.len()
        );
    }
    capped_words
}

/// Seconds by which every cohere word window's start is moved earlier.
///
/// The fold places adjacent words back-to-back on a shared seam boundary, so a
/// word's window can end up a tenth of a second inside its real onset. Because
/// the seam is a fixed point of the per-word least-squares affine fit the
/// normalized TempErr metric uses, a uniform per-side pad of this kind leaves
/// TempErr unchanged while the window widens back over the clipped onset.
const COHERE_WORD_ONSET_PAD_SECONDS: f32 = 0.10;

/// Seconds by which every cohere word window's end is moved later, the
/// offset-side counterpart of [`COHERE_WORD_ONSET_PAD_SECONDS`]. Symmetric with
/// the start pad: the end seam is shared with the next word's start, so both
/// sides of a boundary are pulled out at once.
const COHERE_WORD_OFFSET_PAD_SECONDS: f32 = 0.10;

/// Move every word window's start earlier and its end later, clamped to the
/// audio's time range, so the window covers the speech's acoustic onset/offset
/// rather than the DTW seam frames. The timeline stays ordered because a start
/// only moves earlier and an end only moves later, so an already-monotone,
/// non-overlapping sequence stays so (adjacent windows may gain a little overlap,
/// which the downstream VTT de-overlap handles).
pub(crate) fn cohere_pad_word_windows(
    words: &[WordTimestamp],
    audio_duration_seconds: f32,
) -> Vec<WordTimestamp> {
    words
        .iter()
        .map(|word| {
            let end = (word.end + COHERE_WORD_OFFSET_PAD_SECONDS).min(audio_duration_seconds);
            WordTimestamp {
                word: word.word.clone(),
                start: (0.0f32.max(word.start - COHERE_WORD_ONSET_PAD_SECONDS)).min(end),
                end,
                confidence: word.confidence,
            }
        })
        .collect()
}

/// The wall-clock length of the window, given the per-row frame count and the
/// family's seconds-per-frame.
fn band_duration_seconds(window: &[Vec<f32>], seconds_per_frame: f32) -> f32 {
    let frame_count = window.first().map_or(0, |row| row.len());
    frame_count as f32 * seconds_per_frame
}

/// Returns the fraction of adjacent content-token peak pairs that backward-jump
/// by 2+ frames, in `[0.0, 1.0]`; `0.0` when there are no such pairs (fully
/// monotone after sinks). Non-content tokens are skipped.
fn content_backward_fraction(attention: &[Vec<f32>], is_content: &[bool]) -> f32 {
    let mut previous_peak: Option<usize> = None;
    let mut total_pairs = 0usize;
    let mut backward_pairs = 0usize;
    for (index, row) in attention.iter().enumerate() {
        if !is_content.get(index).copied().unwrap_or(false) || row.is_empty() {
            continue;
        }
        let peak = row
            .iter()
            .enumerate()
            .filter(|&(_, &value)| value.is_finite())
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(frame, _)| frame);
        let Some(peak) = peak else {
            continue;
        };
        if let Some(previous) = previous_peak {
            total_pairs += 1;
            if peak
                .checked_add(1)
                .is_some_and(|shifted| shifted < previous)
            {
                backward_pairs += 1;
            }
        }
        previous_peak = Some(peak);
    }
    if total_pairs == 0 {
        return 0.0;
    }
    backward_pairs as f32 / total_pairs as f32
}

/// Returns a copy of `window` with each dominant early sink frame zeroed in
/// every row, or `None` when no frame qualifies.
///
/// A dominant early sink is a frame within [`SINK_STRIP_SEARCH_FRAMES`] that
/// is the strict-majority global max of at least one of:
/// - all rows in the window (the strict, conservative reading), or
/// - the content rows only (the rows whose decodings actually carry speech;
///   the gate's order test ignores the non-content rows, so the sink check
///   matches the rows the gate cares about).
///
/// The two readings are a superset of each other in practice: a frame that is
/// a majority of all rows is almost always a majority of the (smaller) content
/// row set too, but the content reading can detect a shared early sink that
/// only dominates the meaningful rows. The union of the two is what this
/// function reports.
///
/// On diffuse, front-loaded decodes one such shared priming frame steals the
/// argmax from most tokens, and its removal is what exposes each row's
/// next-strongest frame (the token's real region) so the order gate can
/// re-test. The caller must still re-run `cross_attention_peaks_order_aligned`
/// on the result: this function only removes the artifact, the gate decides
/// whether the cleaned signal is trustworthy. A row whose only finite mass sat
/// on a masked frame has no valid peak after the strip, which the order gate
/// reads as a missing peak and ignores; when enough rows lose their peak the
/// order collapses and the gate rejects (the safe outcome).
///
/// Splits detection from application so the DTW band can be derived from the
/// raw window with exactly these frames skipped (see
/// [`speech_band_from_rows`]) instead of from a zero-masked copy. The caller
/// still re-runs the order gate on the masked window; this pair only finds and
/// removes the artifact.
fn detect_dominant_early_sinks(window: &[Vec<f32>], is_content: &[bool]) -> Option<Vec<u32>> {
    let row_count = window.len();
    let frame_count = window.first()?.len();
    if frame_count == 0 {
        return None;
    }
    let search = SINK_STRIP_SEARCH_FRAMES.min(frame_count);
    let mut all_peak_counts = vec![0usize; search];
    let mut content_peak_counts = vec![0usize; search];
    let mut content_row_count = 0usize;
    for (index, row) in window.iter().enumerate() {
        let (peak, &value) = row
            .iter()
            .enumerate()
            .filter(|&(_, &value)| value.is_finite())
            .max_by(|(_, a), (_, b)| a.total_cmp(b))?;
        if value <= 0.0 || peak >= search {
            continue;
        }
        all_peak_counts[peak] += 1;
        if is_content.get(index).copied().unwrap_or(false) {
            content_peak_counts[peak] += 1;
            content_row_count += 1;
        }
    }
    let mut sinks = Vec::new();
    for frame in 0..search {
        let is_all_majority = all_peak_counts[frame].saturating_mul(2) > row_count;
        let is_content_majority = content_row_count > 0
            && content_peak_counts[frame].saturating_mul(2) > content_row_count;
        if is_all_majority || is_content_majority {
            sinks.push(frame as u32);
        }
    }
    if sinks.is_empty() {
        return None;
    }
    if std::env::var_os("OPENASR_COHERE_DEBUG_CROSS").is_some() {
        eprintln!("cohere cross sink strip: masking frames {sinks:?}");
    }
    Some(sinks)
}

/// Returns a copy of `window` with each frame in `masked_frames` zeroed in
/// every row. The DTW only ever runs on this masked window, so zeroing is
/// safe there; the band must skip the frames instead of reading a masked copy
/// (see [`detect_dominant_early_sinks`]).
fn mask_frames_early(window: &[Vec<f32>], masked_frames: &[u32]) -> Vec<Vec<f32>> {
    let search = SINK_STRIP_SEARCH_FRAMES;
    window
        .iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .map(|(frame, &value)| {
                    if frame < search && masked_frames.iter().any(|m| *m as usize == frame) {
                        0.0
                    } else {
                        value
                    }
                })
                .collect()
        })
        .collect()
}

/// Combined detect-and-mask for callers that only need the masked window.
#[cfg(test)]
fn mask_dominant_early_sinks(window: &[Vec<f32>], is_content: &[bool]) -> Option<Vec<Vec<f32>>> {
    detect_dominant_early_sinks(window, is_content).map(|sinks| mask_frames_early(window, &sinks))
}

#[cfg(test)]
mod tests;
