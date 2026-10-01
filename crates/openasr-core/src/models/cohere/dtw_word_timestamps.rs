//! Cohere DTW word-timestamp refinement, post-decode and ggml-free.
//!
//! The ggml decoder runs one decode and records one last-layer cross-attention
//! frame row per generated token; this module turns those rows into word
//! windows. Per chunk the pipeline is: the peak-order gate (strict, the
//! dominant-early-sink strip, and a tolerant tier for long bands) -> the
//! speech band derived from the unmasked rows with the stripped sinks skipped
//! -> the audio-onset anchors that repair a band start displaced by the sink
//! substitution or bracketing leading silence -> the monotone DTW alignment of
//! token frames -> the center fold into word windows -> the per-word span cap.
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
    MIDPOINT_BOUNDARY_FRACTION, NO_ONSET_LEAD, Seq2SeqTokenTime,
    seq2seq_word_timestamps_from_token_times,
};

use super::runtime_contract::CohereTranscribeExecutionMetadata;

/// Window (seconds) and relative-to-peak level drop (dB) used to detect where
/// real audio content begins inside a chunk. 0.1s isolates a word from
/// surrounding silence at 16kHz, and a 16 dB drop below the chunk's loudest
/// window clears genuine room tone while catching even a quiet opener.
const COHERE_ONSET_WINDOW_SECONDS: f32 = 0.1;
const COHERE_ONSET_RELATIVE_DROP_DB: f32 = 16.0;

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

/// Align per-token cross-attention frame rows to the audio timeline with a
/// monotone DTW pass and fold them into word timestamps, mirroring whisper's
/// no-timestamp DTW degrade. `token_alignments` pairs each generated (non-EOT)
/// token with its decoder's last-layer cross-attention frame row. Cohere
/// decodes `<|notimestamps|>` so there are no timestamp tokens: the DTW window
/// is bracketed on the content tokens' own attention peaks (leading/trailing
/// silence the model ignored is never bracketed by a peak, so it stays off the
/// timeline), and each content-token peak still owns its real audio span.
pub(crate) fn cohere_dtw_word_timestamps<E>(
    token_alignments: &[(u32, Vec<f32>)],
    metadata: CohereTranscribeExecutionMetadata,
    generated_probabilities: &[f32],
    duration: f32,
    audio_onset_seconds: f32,
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
    let onset_lead = cohere_dtw_onset_lead(
        band_end_secs - band_start_secs,
        is_content.iter().filter(|flag| **flag).count(),
    );
    let words = seq2seq_word_timestamps_from_token_times(
        &token_times,
        band_start_secs,
        band_end_secs,
        BuiltinDecodePolicySeq2SeqTextPostprocessKind::Identity,
        decode_text,
        COHERE_DTW_BOUNDARY_FRACTION,
        onset_lead,
        COHERE_DTW_MAX_WORD_SPAN_SECONDS,
        f32::INFINITY,
        f32::INFINITY,
    )?;
    // `word_centers_to_timestamps` anchors the first word's start to
    // `segment_start` (the band start) and the last word's end to
    // `segment_end` (the band end), so a mid-chunk band does not stretch its
    // first/last word across the band's surrounding silence; the boundary
    // between two centers (`COHERE_DTW_BOUNDARY_FRACTION`, pulled earlier by
    // `COHERE_DTW_ONSET_LEAD_SECONDS`) lands where the following word's speech
    // begins. The timeline is monotone and non-overlapping by construction, so
    // the only remaining correction is capping any word that swallowed a real
    // pause (its tail would otherwise run across the following gap).
    let words = cohere_cap_dtw_word_spans(words, seconds_per_frame);
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
fn cohere_peak_fallback_word_timestamps<E>(
    full_window: &[Vec<f32>],
    is_content: &[bool],
    token_alignments: &[(u32, Vec<f32>)],
    generated_probabilities: &[f32],
    seconds_per_frame: f32,
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
    seq2seq_word_timestamps_from_token_times(
        &token_times,
        0.0,
        segment_end,
        BuiltinDecodePolicySeq2SeqTextPostprocessKind::Identity,
        decode_text,
        MIDPOINT_BOUNDARY_FRACTION,
        NO_ONSET_LEAD,
        COHERE_DTW_MAX_WORD_SPAN_SECONDS,
        f32::INFINITY,
        f32::INFINITY,
    )
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
/// Cohere's DTW tiles each word to a seam between its own token's entry frame
/// and the next token's; the seam is where the *next* token's attention
/// arrives, not where this word's speech starts. Measured across the long-form
/// test suite, a word's window start therefore sits a fixed small amount
/// (~0.1s median) past its true acoustic onset, which a wide window hides but
/// the tight truth windows of short function words expose as full window
/// misses. Pulling the start earlier by this amount re-covers the onset.
const COHERE_WORD_ONSET_PAD_SECONDS: f32 = 0.10;

/// Seconds by which every cohere word window's end is moved later.
///
/// The tile seam is likewise slightly short of the word's true acoustic
/// offset on a minority of words (measured ~5% of matched words miss their
/// truth window entirely on the early side). A small end pad covers that
/// offset without widening the common case meaningfully, and is much smaller
/// than the start pad because the end-side miss rate is lower.
const COHERE_WORD_OFFSET_PAD_SECONDS: f32 = 0.05;

/// Move every word window's start earlier and its end later, clamped to the
/// audio's time range, so the window covers the speech's acoustic
/// onset/offset rather than the DTW seam frames. The timeline stays ordered
/// because a start only moves earlier and an end only moves later, so an
/// already-monotone, non-overlapping sequence stays so (adjacent windows may
/// gain a little overlap, which the downstream VTT de-overlap handles).
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
