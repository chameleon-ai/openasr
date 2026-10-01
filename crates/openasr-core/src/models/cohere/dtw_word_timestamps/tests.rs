use tempfile::NamedTempFile;

use super::*;
use crate::models::cohere::runtime_contract::parse_cohere_transcribe_execution_metadata;
use crate::testing::{TinyGgufFixtureSpec, write_tiny_gguf_runtime_source};
use crate::{read_gguf_metadata_from_runtime_source, validate_ggml_runtime_source_path};

/// The metadata the DTW unit tests run against: the same tiny runtime-ready
/// fixture the graph tests write, parsed through the runtime contract. The DTW
/// passes only read the mel hop and sample rate (their 0.08 s frame grid), so
/// the temp file is dropped as soon as the parse is done.
fn cohere_transcribe_metadata_fixture() -> CohereTranscribeExecutionMetadata {
    let file = NamedTempFile::new().expect("temp file");
    let persisted = file.into_temp_path();
    let spec = TinyGgufFixtureSpec::cohere_oasr_v1_runtime_ready("cohere-dtw-fixture");
    write_tiny_gguf_runtime_source(&persisted, &spec).expect("write fixture");
    let runtime_source =
        validate_ggml_runtime_source_path(&persisted).expect("valid runtime source path");
    let metadata =
        read_gguf_metadata_from_runtime_source(&runtime_source).expect("read gguf metadata");
    parse_cohere_transcribe_execution_metadata(&metadata).expect("parse metadata")
}

#[test]
fn audio_onset_seconds_tracks_the_first_non_silent_window() {
    let rate = 16_000;
    let silent = 16_000; // 1.0s
    let speech = 32_000; // 2.0s
    let mut samples = vec![0.001_f32; silent];
    // A gentle low-frequency tone so every window after onset is clearly
    // above the relative drop threshold.
    for i in 0..speech {
        samples.push(0.5 * ((i as f32 * 0.02).sin()));
    }
    let onset = audio_onset_seconds(&samples, rate);
    assert!(
        (0.9..=1.15).contains(&onset),
        "onset should sit at the 1.0s silence-to-speech boundary, got {onset}"
    );
}

#[test]
fn audio_onset_seconds_at_chunk_edge_is_zero() {
    // Speech begins immediately: the first window already clears the
    // relative threshold, so the onset collapses to the chunk start.
    let rate = 16_000;
    let samples: Vec<f32> = (0..48_000u32)
        .map(|i| 0.5 * ((i as f32 * 0.02).sin()))
        .collect();
    assert_eq!(audio_onset_seconds(&samples, rate), 0.0);
}

#[test]
fn audio_onset_seconds_all_silent_is_zero() {
    let rate = 16_000;
    let samples = vec![0.0_f32; 16_000 * 5];
    assert_eq!(audio_onset_seconds(&samples, rate), 0.0);
}

#[test]
fn cohere_dtw_onset_lead_is_flat_below_the_density_knee() {
    // A band speaking at or below the knee gets the baseline lead verbatim;
    // no density term is added.
    let slow = cohere_dtw_onset_lead(10.0, 10); // 1.0 word/s < knee 2.4
    assert!((slow - COHERE_DTW_ONSET_LEAD_SECONDS).abs() < 1e-6);
    let at_knee = cohere_dtw_onset_lead(10.0, 24); // exactly 2.4 word/s
    assert!((at_knee - COHERE_DTW_ONSET_LEAD_SECONDS).abs() < 1e-6);
}

#[test]
fn cohere_dtw_onset_lead_grows_with_density_above_the_knee() {
    // Above the knee the lead rises at the configured slope per extra
    // word/second. At 4.0 word/s the excess over the 2.4 knee is 1.6, so
    // lead = baseline + slope * 1.6 (0.216 at the compiled slope 0.01) --
    // comfortably below the cap.
    let dense = cohere_dtw_onset_lead(10.0, 40);
    let expected = COHERE_DTW_ONSET_LEAD_SECONDS + COHERE_DTW_LEAD_DENSITY_SLOPE * 1.6_f32.max(0.0);
    assert!((dense - expected).abs() < 1e-6);
}

fn word(word: &str, start: f32, end: f32) -> WordTimestamp {
    WordTimestamp {
        word: word.to_string(),
        start,
        end,
        confidence: None,
    }
}

#[test]
fn cohere_pad_word_windows_widens_toward_the_edges_and_clamps_to_the_audio() {
    let words = vec![
        word("a", 0.0, 0.4),
        word("b", 0.4, 0.9),
        word("c", 9.95, 10.0),
    ];
    let padded = cohere_pad_word_windows(&words, 10.0);
    assert_eq!(padded[0].start, 0.0, "start clamps at 0");
    assert!((padded[0].end - (0.4 + COHERE_WORD_OFFSET_PAD_SECONDS)).abs() < f32::EPSILON);
    assert!((padded[1].start - (0.4 - COHERE_WORD_ONSET_PAD_SECONDS)).abs() < f32::EPSILON);
    assert!((padded[1].end - (0.9 + COHERE_WORD_OFFSET_PAD_SECONDS)).abs() < f32::EPSILON);
    assert!(padded[2].start < 9.95, "start pulled earlier");
    assert_eq!(padded[2].end, 10.0, "end clamps at the audio duration");
    assert_eq!(
        padded.iter().map(|w| w.word.as_str()).collect::<Vec<_>>(),
        ["a", "b", "c"],
        "word text and confidence are preserved"
    );
}

#[test]
fn cohere_dtw_onset_lead_caps_at_max_seconds() {
    // A very dense band (30 word/s) adds more lead than the cap allows; the
    // result is exactly the cap, not the unbounded density term. The density
    // is comfortably above the knee regardless of the (small) configured
    // slope, so the cap holds on any reasonable tuning.
    let runaway = cohere_dtw_onset_lead(10.0, 300);
    assert_eq!(runaway, COHERE_DTW_ONSET_LEAD_MAX_SECONDS);
}

#[test]
fn cohere_dtw_word_timestamps_returns_empty_without_alignments() {
    let metadata = cohere_transcribe_metadata_fixture();
    let decode_text = |_token_ids: &[u32]| Ok(String::new());
    let words = cohere_dtw_word_timestamps::<()>(&[], metadata, &[], 1.0, 0.0, &decode_text)
        .expect("dtw words");
    assert!(words.is_empty(), "no alignments -> no words");
}

#[test]
fn cohere_dtw_word_timestamps_places_word_at_earlier_attention() {
    let metadata = cohere_transcribe_metadata_fixture();
    let seconds_per_frame = 8.0 * metadata.hop_length as f32 / metadata.sample_rate_hz as f32;
    let frames = 12;
    // Two "words": token 0 peaks early, token 1 peaks late. Each frame row
    // is a unit-spike (its attention concentrated on one frame).
    let mut row0 = vec![0.01f32; frames];
    row0[2] = 0.97;
    let mut row1 = vec![0.01f32; frames];
    row1[9] = 0.97;
    let alignments = vec![(5u32, row0), (6u32, row1)];
    let decode_text = |token_ids: &[u32]| {
        let mut decoded = String::new();
        for &token_id in token_ids {
            match token_id {
                5 => decoded.push_str("hi"),
                6 => decoded.push_str(" there"),
                _ => {}
            }
        }
        Ok(decoded)
    };
    let duration = (frames as f32) * seconds_per_frame;
    let words = cohere_dtw_word_timestamps::<()>(
        &alignments,
        metadata,
        &[0.99, 0.99],
        duration,
        0.0,
        &decode_text,
    )
    .expect("dtw words");
    assert_eq!(words.len(), 2, "expected two words, got {words:?}");
    assert_eq!(words[0].word, "hi");
    assert_eq!(words[1].word, "there");
    // The DTW center fold anchors the first word's start to the band start
    // (frame 0 here). In this synthetic setup both centers collapse to the
    // band start after onset-lead, so the first word's window degenerates
    // to a point at 0.0.
    assert!(
        (words[0].start - 0.0).abs() < 1e-3,
        "first word must start at the band start, got {words:?}"
    );
    // The last word's end is no longer pinned to the segment end; the edge
    // clamp bounds it to `last_center + COHERE_DTW_MAX_WORD_SPAN_SECONDS/2`.
    // Here both centers are 0.0 so the last word ends at 0.75s instead of
    // stretching the full band end at 0.96s.
    let expected_last_end = COHERE_DTW_MAX_WORD_SPAN_SECONDS / 2.0;
    assert!(
        (words[1].end - expected_last_end).abs() < 1e-3,
        "last word must end within COHERE_DTW_MAX_WORD_SPAN_SECONDS/2 of its center, got {words:?}"
    );
    // The timeline stays monotone and non-overlapping.
    assert!(words[1].start >= words[0].end - 1e-6);
}

#[test]
fn cross_attention_peaks_order_aligned_gate() {
    let spike = |frames: usize, peak: usize| {
        let mut row = vec![0.01f32; frames];
        row[peak] = 0.97;
        row
    };
    // Monotone non-decreasing peaks form a clean left-to-right order.
    assert!(cross_attention_peaks_order_aligned(
        &[spike(12, 2), spike(12, 2), spike(12, 5), spike(12, 9)],
        &[true; 4]
    ));
    // A backward jump of two or more frames is the diffuse front-loaded
    // zig-zag the gate must reject.
    assert!(!cross_attention_peaks_order_aligned(
        &[spike(12, 9), spike(12, 2)],
        &[true; 2]
    ));
    // Ties and a single frame of jitter are tolerated (not a zig-zag).
    assert!(cross_attention_peaks_order_aligned(
        &[spike(12, 5), spike(12, 5), spike(12, 4), spike(12, 5)],
        &[true; 4]
    ));
    // Non-content (punctuation) rows are skipped, so a diffuse peak in
    // them cannot break the order of the surrounding content peaks.
    assert!(cross_attention_peaks_order_aligned(
        &[spike(12, 9), spike(12, 0), spike(12, 10)],
        &[true, false, true]
    ));
    // Fewer than two content peaks -> vacuously aligned (left to the DTW).
    assert!(cross_attention_peaks_order_aligned(
        &[spike(12, 7)],
        &[true]
    ));
}

// Builds a "sink-dominated" row: an early sink frame at frame `sink` holds
// the global max, with the token's real region (a lower value) at `real`.
// After stripping `sink` this row's argmax becomes `real`.
fn sink_row(frames: usize, sink: usize, real: usize) -> Vec<f32> {
    let mut row = vec![0.01f32; frames];
    row[sink] = 0.5;
    row[real] = 0.4;
    row
}

// A "right-pointing" row whose own speech location is already its global
// max (it escaped the priming artifact).
fn right_row(frames: usize, real: usize) -> Vec<f32> {
    let mut row = vec![0.01f32; frames];
    row[real] = 0.5;
    row
}

#[test]
fn mask_dominant_early_sinks_detects_and_strips_the_shared_peak() {
    let frames = 12;
    // A dominant sink at frame 2 is the *global* max for a strict majority
    // (4 of 5) of the rows; the fifth row peaks elsewhere.
    let raw = vec![
        sink_row(frames, 2, 4),
        sink_row(frames, 2, 5),
        sink_row(frames, 2, 6),
        sink_row(frames, 2, 7),
        right_row(frames, 9),
    ];
    let all_content = vec![true; raw.len()];
    let striped = mask_dominant_early_sinks(&raw, &all_content).expect("a sink must be found");
    assert_eq!(striped.len(), raw.len());
    for (raw_row, striped_row) in raw.iter().zip(&striped) {
        assert_eq!(
            striped_row[2], 0.0,
            "sink frame must be zeroed in every row"
        );
        for (frame, (a, b)) in raw_row.iter().zip(striped_row).enumerate() {
            if frame != 2 {
                assert_eq!(a, b, "non-sink frames stay untouched");
            }
        }
    }
    // Frame 2 is the argmax of only 2 of 4 rows: not a strict majority
    // (2*2 > 4 is false), so no sink qualifies.
    assert!(
        mask_dominant_early_sinks(
            &[
                sink_row(frames, 2, 4),
                sink_row(frames, 2, 5),
                right_row(frames, 8),
                right_row(frames, 9),
            ],
            &[true; 4],
        )
        .is_none()
    );
    // A shared peak beyond the 10-frame search horizon is out of scope
    // (late peaks carry a token's real region and must never be masked).
    assert!(
        mask_dominant_early_sinks(&[right_row(frames, 11), right_row(frames, 11),], &[true; 2])
            .is_none()
    );
    // Empty first row -> no frame count -> nothing to strip.
    assert!(mask_dominant_early_sinks(&[Vec::new(), vec![0.01; frames]], &[true; 2]).is_none());
}

#[test]
fn sink_strip_restores_order_for_diffuse_front_loaded_decode() {
    let frames = 44;
    // The measured cohere artifact: one early sink (frame 3) steals the
    // argmax from most rows while each row's real region walks left to
    // right. One row sits *right* of another sink row, so the raw peak
    // order zig-zags (rejected); after the sink is stripped the reals are
    // monotone left to right (accepted).
    let rows = vec![
        sink_row(frames, 3, 10),
        sink_row(frames, 3, 13),
        right_row(frames, 16),
        sink_row(frames, 3, 19),
        sink_row(frames, 3, 22),
        sink_row(frames, 3, 25),
        sink_row(frames, 3, 28),
        right_row(frames, 40),
    ];
    let is_content = vec![true; rows.len()];
    assert!(
        !cross_attention_peaks_order_aligned(&rows, &is_content),
        "raw zig-zag must be rejected before stripping"
    );
    let striped =
        mask_dominant_early_sinks(&rows, &is_content).expect("sink frame 3 must be found");
    assert!(
        cross_attention_peaks_order_aligned(&striped, &is_content),
        "stripping the dominant sink must restore a monotone peak order"
    );
}

#[test]
fn content_backward_fraction_counts_zigzag_pairs() {
    let frames = 20;
    // Fully monotone: no backward pairs -> fraction 0.0.
    let monotone = vec![
        right_row(frames, 4),
        right_row(frames, 8),
        right_row(frames, 12),
        right_row(frames, 16),
    ];
    assert_eq!(content_backward_fraction(&monotone, &[true; 4]), 0.0);
    // One backward jump of 8+ frames out of 3 pairs: 1/3.
    let zigzag = vec![
        right_row(frames, 4),
        right_row(frames, 12),
        right_row(frames, 5),
        right_row(frames, 16),
    ];
    assert!((content_backward_fraction(&zigzag, &[true; 4]) - 1.0 / 3.0).abs() < 1e-6);
    // Non-content rows are skipped entirely.
    assert_eq!(
        content_backward_fraction(
            &[
                right_row(frames, 4),
                right_row(frames, 0),
                right_row(frames, 16)
            ],
            &[true, false, true],
        ),
        0.0
    );
    // No content pairs at all -> vacuously 0.0 (the caller still has the
    // strict re-test to fall through to).
    assert_eq!(
        content_backward_fraction(&[right_row(frames, 4)], &[true]),
        0.0
    );
}

#[test]
fn band_duration_seconds_scales_with_row_length() {
    let window = vec![vec![0.0; 375], vec![0.0; 375]];
    assert!((band_duration_seconds(&window, 0.08) - 30.0).abs() < 1e-5);
    let short = vec![vec![0.0; 208], vec![0.0; 208]];
    assert!((band_duration_seconds(&short, 0.08) - 16.64).abs() < 1e-5);
    assert_eq!(band_duration_seconds(&[], 0.08), 0.0);
}

#[test]
fn mask_dominant_early_sinks_strips_frame_dominating_content_rows_only() {
    let frames = 12;
    // 51 content rows peak at the early sink (frame 2); 105 non-content
    // rows peak at later distinct frames. Frame 2 is not a majority of
    // all rows (51*2=102 is not greater than 156) but is a strict
    // majority of the content rows (51*2=102 > 51). The gate only tests
    // content rows, so the sink must still be stripped.
    let mut rows: Vec<Vec<f32>> = Vec::with_capacity(156);
    let mut is_content: Vec<bool> = Vec::with_capacity(156);
    for _ in 0..51 {
        rows.push(sink_row(frames, 2, 9));
        is_content.push(true);
    }
    for i in 0..105 {
        let mut row = vec![0.01f32; frames];
        row[(i + 4) % (frames - 4) + 4] = 0.3;
        rows.push(row);
        is_content.push(false);
    }
    let stripped =
        mask_dominant_early_sinks(&rows, &is_content).expect("sink must be found vs content rows");
    assert_eq!(
        stripped[0][2], 0.0,
        "sink frame must be zeroed in every row"
    );
    assert_eq!(
        stripped[100][2], 0.0,
        "sink strip applies to all rows, not just content"
    );
}

/// 20 content rows with the diffuse front-loaded artifact and a single
/// residual dip: 18 rows share the early sink (frame 3) with their real
/// region walking left to right around a gap at frame 20 (a right-pointing
/// row), so the raw peak order zig-zags. After the sink is stripped the
/// effective peaks are `10..15, 20, 15'..` with exactly one backward pair
/// (~5% of content pairs): just under the tolerant 10% threshold but above
/// the strict re-test (which allows zero).
fn zigzag_after_strip_rows(frames: usize) -> (Vec<Vec<f32>>, Vec<bool>) {
    let reals: [usize; 20] = [
        10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
    ];
    let mut rows = Vec::with_capacity(20);
    // Row 5 is the right-pointing escapee at frame 20; everything before it
    // climbs 10..14, then row 5 jumps to 20, then the sink rows resume at
    // 15 and climb to 29. The single 20 -> 15 step is the lone dip.
    for (index, &real) in reals.iter().enumerate() {
        if index == 5 {
            rows.push(right_row(frames, 20));
        } else {
            rows.push(sink_row(frames, 3, real));
        }
    }
    let is_content = vec![true; rows.len()];
    (rows, is_content)
}

#[test]
fn cohere_dtw_word_timestamps_uses_dtw_when_sink_strip_restores_order() {
    let metadata = cohere_transcribe_metadata_fixture();
    let seconds_per_frame = 8.0 * metadata.hop_length as f32 / metadata.sample_rate_hz as f32;
    let frames = 44;
    // Eight content words with the diffuse front-loaded artifact (same
    // shape as the unit test above): the raw peak order zig-zags, so only
    // the sink-stripped re-test can admit the DTW pass and emit spans.
    let rows = [
        sink_row(frames, 3, 10),
        sink_row(frames, 3, 13),
        right_row(frames, 16),
        sink_row(frames, 3, 19),
        sink_row(frames, 3, 22),
        sink_row(frames, 3, 25),
        sink_row(frames, 3, 28),
        right_row(frames, 40),
    ];
    let token_ids = [10u32, 11, 12, 13, 14, 15, 16, 17];
    let alignments: Vec<(u32, Vec<f32>)> = token_ids
        .iter()
        .zip(rows.iter())
        .map(|(&token_id, row)| (token_id, row.clone()))
        .collect();
    let decode_text = |token_ids: &[u32]| {
        let mut decoded = String::new();
        for &token_id in token_ids {
            match token_id {
                10 => decoded.push_str("And"),
                11 => decoded.push_str(" so"),
                12 => decoded.push_str(" my"),
                13 => decoded.push_str(" fellow"),
                14 => decoded.push_str(" country"),
                15 => decoded.push_str(" can"),
                16 => decoded.push_str(" for"),
                17 => decoded.push_str(" you"),
                _ => {}
            }
        }
        Ok(decoded)
    };
    let duration = (frames as f32) * seconds_per_frame;
    let words = cohere_dtw_word_timestamps::<()>(
        &alignments,
        metadata,
        &vec![0.99; token_ids.len()],
        duration,
        0.0,
        &decode_text,
    )
    .expect("dtw words");
    assert!(
        !words.is_empty(),
        "sink-stripped monotone peaks must produce DTW word spans, got empty"
    );
    let actual_words: Vec<&str> = words.iter().map(|word| word.word.as_str()).collect();
    assert_eq!(
        actual_words,
        ["And", "so", "my", "fellow", "country", "can", "for", "you"],
        "word stream must match the transcript"
    );
    // DTW spans tile the band monotonically: non-decreasing, no overlap,
    // and within the clip (the uniform path would spread them evenly).
    for pair in words.windows(2) {
        assert!(pair[0].start <= pair[1].start);
        assert!(pair[0].end - 1e-6 <= pair[1].start);
    }
    assert!(words.last().is_some_and(|last| last.end <= duration + 1e-3));
}

#[test]
fn cohere_dtw_word_timestamps_falls_back_when_sink_strip_cannot_save_order() {
    let metadata = cohere_transcribe_metadata_fixture();
    let seconds_per_frame = 8.0 * metadata.hop_length as f32 / metadata.sample_rate_hz as f32;
    let frames = 44;
    // A dominant sink at frame 3 is present (3 of 4 rows peak there), but
    // the reals still zig-zag after it is stripped (a middle word's real
    // sits *after* the last one), so no amount of stripping makes the
    // signal trustworthy and the caller keeps the uniform timestamps.
    let rows = [
        sink_row(frames, 3, 12),
        right_row(frames, 16),
        sink_row(frames, 3, 30),
        sink_row(frames, 3, 20),
    ];
    let token_ids = [10u32, 11, 12, 13];
    let alignments: Vec<(u32, Vec<f32>)> = token_ids
        .iter()
        .zip(rows.iter())
        .map(|(&token_id, row)| (token_id, row.clone()))
        .collect();
    let decode_text = |token_ids: &[u32]| {
        let mut decoded = String::new();
        for &token_id in token_ids {
            match token_id {
                10 => decoded.push_str("And"),
                11 => decoded.push_str(" so"),
                12 => decoded.push_str(" my"),
                13 => decoded.push_str(" fellow"),
                _ => {}
            }
        }
        Ok(decoded)
    };
    let duration = (frames as f32) * seconds_per_frame;
    let words = cohere_dtw_word_timestamps::<()>(
        &alignments,
        metadata,
        &vec![0.99; token_ids.len()],
        duration,
        0.0,
        &decode_text,
    )
    .expect("dtw words");
    assert!(
        words.is_empty(),
        "zig-zag surviving the sink strip must fall back (empty), got {words:?}"
    );
}

fn tolerant_tier_window(frames: usize) -> (Vec<(u32, Vec<f32>)>, Vec<Vec<f32>>, Vec<bool>, f32) {
    let (rows, is_content) = zigzag_after_strip_rows(frames);
    let duration = (frames as f32) * (8.0 * 320.0 / 16000.0);
    let token_ids: Vec<u32> = (20..20 + rows.len() as u32).collect();
    let alignments = rows
        .iter()
        .zip(&token_ids)
        .map(|(row, &token_id)| (token_id, row.clone()))
        .collect();
    (alignments, rows, is_content, duration)
}

#[test]
fn cohere_dtw_word_timestamps_uses_tolerant_dtw_when_strip_leaves_minor_zigzag_on_long_window() {
    let metadata = cohere_transcribe_metadata_fixture();
    // 260 frames * 0.08s = 20.8s, above the 20s threshold.
    let (alignments, rows, is_content, duration) = tolerant_tier_window(260);
    assert!(duration >= COHERE_DTW_TOLERANT_MIN_BAND_SECONDS);
    // Sanity-check the helper's contract: raw zig-zags (strict re-test fails),
    // but the fractional backward count is exactly 1/19 which is below the
    // 10% tolerant threshold.
    assert!(
        !cross_attention_peaks_order_aligned(&rows, &is_content),
        "the raw peak order must fail the strict monotone re-test"
    );
    let fraction = content_backward_fraction(&rows, &is_content);
    assert!(
        (fraction - 1.0 / 19.0).abs() < 1e-6,
        "expected exactly one backward pair out of 19, got {fraction}"
    );
    let token_ids: Vec<u32> = (20..20 + alignments.len() as u32).collect();
    let decode_text = |token_ids: &[u32]| {
        let mut s = String::new();
        for &token_id in token_ids {
            use std::fmt::Write;
            let _ = write!(s, "w{token_id} ");
        }
        Ok(s.trim_end().to_string())
    };
    let words = cohere_dtw_word_timestamps::<()>(
        &alignments,
        metadata,
        &vec![0.99; token_ids.len()],
        duration,
        0.0,
        &decode_text,
    )
    .expect("dtw words");
    assert!(
        !words.is_empty(),
        "tolerant-tier long window must emit DTW word spans, got empty"
    );
    for (index, word) in words.iter().enumerate() {
        assert_eq!(word.word, format!("w{}", token_ids[index]));
    }
    for pair in words.windows(2) {
        assert!(pair[0].start <= pair[1].start, "timeline must be monotone");
        assert!(pair[0].end - 1e-6 <= pair[1].start, "no overlaps");
    }
    assert!(words.last().is_some_and(|last| last.end <= duration + 1e-3));
}

#[test]
fn cohere_dtw_word_timestamps_falls_back_when_strip_zigzag_fits_short_window() {
    let metadata = cohere_transcribe_metadata_fixture();
    // 44 frames * 0.08s = 3.52s, below the 20s threshold. Even though the
    // post-strip backward fraction would clear the tolerant tier, the
    // band-length guard must reject this short window so it falls back to
    // uniform baseline.
    let (alignments, rows, is_content, duration) = tolerant_tier_window(44);
    assert!(
        duration < COHERE_DTW_TOLERANT_MIN_BAND_SECONDS,
        "short-window test must stay below the tolerant band threshold"
    );
    assert!(
        !cross_attention_peaks_order_aligned(&rows, &is_content),
        "raw zig-zag must fail the strict re-test"
    );
    let fraction = content_backward_fraction(&rows, &is_content);
    assert!(
        fraction <= COHERE_DTW_MAX_BACKWARD_PAIR_FRACTION,
        "fraction must be under the tolerant threshold so the only failing check is band length"
    );
    let token_ids: Vec<u32> = (20..20 + alignments.len() as u32).collect();
    let decode_text = |token_ids: &[u32]| {
        let mut s = String::new();
        for &token_id in token_ids {
            use std::fmt::Write;
            let _ = write!(s, "w{token_id} ");
        }
        Ok(s.trim_end().to_string())
    };
    let words = cohere_dtw_word_timestamps::<()>(
        &alignments,
        metadata,
        &vec![0.99; token_ids.len()],
        duration,
        0.0,
        &decode_text,
    )
    .expect("dtw words");
    assert!(
        words.is_empty(),
        "short window below the tolerant band threshold must still fall back, got {words:?}"
    );
}

#[test]
fn cohere_dtw_word_timestamps_falls_back_when_peaks_not_aligned() {
    let metadata = cohere_transcribe_metadata_fixture();
    let seconds_per_frame = 8.0 * metadata.hop_length as f32 / metadata.sample_rate_hz as f32;
    let frames = 12;
    // Two content words, but the second attends *earlier* than the first
    // (the diffuse front-loaded artifact). The gate rejects the DTW pass
    // and returns empty so the caller keeps the uniform post-hoc
    // timestamps instead of the over-spread spans the DTW would emit.
    let mut row0 = vec![0.01f32; frames];
    row0[9] = 0.97;
    let mut row1 = vec![0.01f32; frames];
    row1[2] = 0.97;
    let alignments = vec![(5u32, row0), (6u32, row1)];
    let decode_text = |token_ids: &[u32]| {
        let mut decoded = String::new();
        for &token_id in token_ids {
            match token_id {
                5 => decoded.push_str("hi"),
                6 => decoded.push_str(" there"),
                _ => {}
            }
        }
        Ok(decoded)
    };
    let duration = (frames as f32) * seconds_per_frame;
    let words = cohere_dtw_word_timestamps::<()>(
        &alignments,
        metadata,
        &[0.99, 0.99],
        duration,
        0.0,
        &decode_text,
    )
    .expect("dtw words");
    assert!(
        words.is_empty(),
        "non-order-aligned peaks must fall back (empty), got {words:?}",
    );
}

/// 30 content rows on `frames` frames arranged so the raw peak order
/// zig-zags immediately (row 0 peaks at ~100, row 1 at ~5) and ~50% of
/// adjacent pairs are backward pairs (well above the 10% tolerant
/// threshold), so BOTH the strict and the tolerant DTW tiers are
/// rejected. No frame in [0, 10) is a dominant early sink (only one row
/// peaks in that range), so `mask_dominant_early_sinks` returns `None`
/// and the gate falls through to the catch-all branch.
fn zigzag_no_sink_rows(frames: usize) -> Vec<Vec<f32>> {
    assert!(
        frames >= 110,
        "need at least 110 frames for the test pattern"
    );
    let n = 30usize;
    (0..n)
        .map(|i| {
            let step = i / 2;
            let frame: usize = if i % 2 == 0 {
                (frames / 2).saturating_sub(step.saturating_mul(3))
            } else {
                (5usize + step.saturating_mul(3)).min(10)
            };
            let mut row = vec![0.01_f32; frames];
            row[frame] = 0.5;
            row
        })
        .collect()
}

#[test]
fn cohere_dtw_word_timestamps_caps_a_word_span_swallowed_by_a_pause() {
    let metadata = cohere_transcribe_metadata_fixture();
    let spf = 8.0 * metadata.hop_length as f32 / metadata.sample_rate_hz as f32;
    let frames = 250; // 20.0s window
    let duration = (frames as f32) * spf;
    // Two content words with a real pause between them: token 0 peaks at
    // frame 10 (0.8s), token 1 at frame 200 (16.0s). The monotone DTW
    // path must spend the whole gap on token 0's row, so without the cap
    // word 0 would run 0.0->16.0s.
    let peak_frames = [10, 200];
    let rows: Vec<Vec<f32>> = (0..2)
        .map(|i| {
            let mut row = vec![0.01_f32; frames];
            row[peak_frames[i]] = 0.5;
            row
        })
        .collect();
    let token_ids = [40u32, 41];
    let alignments: Vec<(u32, Vec<f32>)> = token_ids
        .iter()
        .zip(rows.iter())
        .map(|(&id, row)| (id, row.to_vec()))
        .collect();
    let decode_text = |token_ids: &[u32]| {
        let mut s = String::new();
        for &id in token_ids {
            use std::fmt::Write;
            let _ = write!(s, "w{id} ");
        }
        Ok(s.trim_end().to_string())
    };
    // The raw DTW spans (pre-cap) must include a span wider than the cap:
    // the monotone path spends the 0.8s->16.0s pause on one of the two
    // rows regardless of which token "owns" the second peak frame.
    let band_rows: Vec<Vec<f32>> = rows.to_vec();
    let (band_start, band_end) =
        crate::models::seq2seq_dtw_alignment::speech_frame_bounds(&band_rows, &[true; 2])
            .expect("band");
    let sliced: Vec<Vec<f32>> = rows
        .iter()
        .map(|row| row[band_start..band_end].to_vec())
        .collect();
    let spans = dtw_align_token_frames(&sliced).expect("spans");
    assert!(
        spans
            .iter()
            .any(|span| (span.frame_end - span.frame_start) as f32 * spf
                > COHERE_DTW_MAX_WORD_SPAN_SECONDS),
        "the test pattern must produce a pre-cap span wider than the cap: {spans:?}"
    );
    let words = cohere_dtw_word_timestamps::<()>(
        &alignments,
        metadata,
        &[0.99, 0.99],
        duration,
        0.0,
        &decode_text,
    )
    .expect("dtw words");
    assert_eq!(words.len(), 2, "two words must be emitted, got {words:?}");
    // Every emitted word must be within the span cap: whatever token (or
    // word) the DTW path let the pause run through, its end cannot follow
    // the pause to the next token's entry frame.
    for word in &words {
        assert!(
            word.end - word.start <= COHERE_DTW_MAX_WORD_SPAN_SECONDS + 1e-6,
            "swallowed pause must be capped, got {word:?}"
        );
    }
    assert!(
        words[0].start <= words[1].start + 1e-6
            && words[0].end - 1e-6 <= words[1].start
            && words.iter().all(|w| w.end <= duration + 1e-3),
        "timeline must stay monotone, non-overlapping, and within the clip: {words:?}"
    );
}

#[test]
fn cohere_dtw_word_timestamps_band_skips_stripped_sink_on_long_window() {
    let metadata = cohere_transcribe_metadata_fixture();
    let spf = 8.0 * metadata.hop_length as f32 / metadata.sample_rate_hz as f32;
    let frames = 250; // 20.0s window
    let duration = (frames as f32) * spf;
    // Five content words. Row 0 already escaped the sink (peaks at its
    // real frame 30); the others peak on the shared sink at frame 3 with
    // their real frames further in. The raw order zigzags (30 then 3), so
    // only the sink-stripped path can admit the DTW, and the earliest
    // *real* peak (frame 30, 2.4s) must bound the band -- not the sink.
    let rows: Vec<Vec<f32>> = (0..5)
        .map(|i| {
            let mut row = vec![0.01_f32; frames];
            if i > 0 {
                row[3] = 0.5;
            }
            row[30 + i * 25] = 0.4 + (i == 0) as u8 as f32 * 0.1;
            row
        })
        .collect();
    assert!(
        !cross_attention_peaks_order_aligned(&rows, &vec![true; rows.len()]),
        "the shared early sink must zigzag the raw peak order"
    );
    let token_ids: Vec<u32> = (10..15).collect();
    let alignments: Vec<(u32, Vec<f32>)> = rows
        .iter()
        .zip(&token_ids)
        .map(|(row, &id)| (id, row.to_vec()))
        .collect();
    let decode_text = |token_ids: &[u32]| {
        let mut s = String::new();
        for &id in token_ids {
            use std::fmt::Write;
            let _ = write!(s, "w{id} ");
        }
        Ok(s.trim_end().to_string())
    };
    let words = cohere_dtw_word_timestamps::<()>(
        &alignments,
        metadata,
        &vec![0.99; token_ids.len()],
        duration,
        0.0,
        &decode_text,
    )
    .expect("dtw words");
    assert!(
        !words.is_empty(),
        "stripped monotone window must emit DTW words"
    );
    // The stripped sink (frame 3) must not drag the first word back into
    // the leading silence: the band starts at the earliest real peak
    // (frame 30) minus the margin (frame 20, 1.6s), and the DTW cannot
    // place anything before the band start.
    let first_start = words[0].start;
    assert!(
        first_start >= (20.0_f32 * spf) - 1e-6,
        "first word start ({first_start}s) must not precede the sink-skipped band start ({:?}s)",
        20.0 * spf
    );
    // Timeline still tiles the band monotonone to the window end.
    for pair in words.windows(2) {
        assert!(
            pair[0].start <= pair[1].start + 1e-6,
            "timeline must be monotone"
        );
        assert!(pair[0].end - 1e-6 <= pair[1].start, "no overlaps");
    }
    assert!(words.last().is_some_and(|last| last.end <= duration + 1e-3));
}

#[test]
fn cohere_dtw_word_timestamps_advances_band_on_measured_leading_silence() {
    let metadata = cohere_transcribe_metadata_fixture();
    let spf = 8.0 * metadata.hop_length as f32 / metadata.sample_rate_hz as f32;
    let frames = 250; // 20.0s window, same shape as the sink-skip test.
    let duration = (frames as f32) * spf;
    // Same shared-sink layout as `band_skips_stripped_sink_on_long_window`:
    // content word 0 peaks at its real frame 30, the rest peak on the
    // shared sink at frame 3. The band therefore starts at frame 20
    // (real peak 30 minus the 10-frame margin) and covers the whole
    // window tail.
    let rows: Vec<Vec<f32>> = (0..5)
        .map(|i| {
            let mut row = vec![0.01_f32; frames];
            if i > 0 {
                row[3] = 0.5;
            }
            row[30 + i * 25] = 0.4 + (i == 0) as u8 as f32 * 0.1;
            row
        })
        .collect();
    let token_ids: Vec<u32> = (10..15).collect();
    let alignments: Vec<(u32, Vec<f32>)> = rows
        .iter()
        .zip(&token_ids)
        .map(|(row, &id)| (id, row.to_vec()))
        .collect();
    let decode_text = |token_ids: &[u32]| {
        let mut s = String::new();
        for &id in token_ids {
            use std::fmt::Write;
            let _ = write!(s, "w{id} ");
        }
        Ok(s.trim_end().to_string())
    };
    let probs: Vec<f32> = vec![0.99; token_ids.len()];

    // Case 1: the chunk opens with speech (onset 0.0, the measured value
    // for a chunk whose first window already carries energy). No leading
    // silence, so the branch must NOT fire and the band keeps the
    // sink-skipping start at frame 20. The word fold clamps every start to
    // the band start, so the first word starts no earlier than frame 20.
    let at_chunk = cohere_dtw_word_timestamps::<()>(
        &alignments,
        metadata,
        &probs,
        duration,
        0.0,
        &decode_text,
    )
    .expect("dtw words");
    let at_chunk_start = at_chunk[0].start;
    assert!(
        at_chunk_start >= (20.0 * spf) - 1e-6,
        "onset-0 (no leading silence) keeps the sink-skipped band start; first word at {at_chunk_start}s must be >= frame 20 ({:?}s)",
        20.0 * spf
    );

    // Case 2: real leading silence. The chunk's audio onset is measured at
    // frame 40 (3.2s) -- well past the band's frame 20 start and ahead of
    // the margin, so the band brackets leading silence. The leading-silence
    // branch must advance the band start to the onset, pushing the first
    // word to the real speech instead of the silent chunk front.
    let onset = 40.0 * spf;
    let advanced = cohere_dtw_word_timestamps::<()>(
        &alignments,
        metadata,
        &probs,
        duration,
        onset,
        &decode_text,
    )
    .expect("dtw words");
    let advanced_start = advanced[0].start;
    assert!(
        advanced_start >= (40.0 * spf) - 1e-6,
        "leading silence must advance the first word to the measured onset; got {advanced_start}s, want >= frame 40 ({:?}s)",
        40.0 * spf
    );
    assert!(
        advanced_start > at_chunk_start,
        "the advanced band start must place the first word later (advanced {advanced_start}s > onset-0 {at_chunk_start}s)"
    );
}

#[test]
fn cohere_dtw_word_timestamps_uses_peak_fallback_on_long_zigzag_window() {
    let metadata = cohere_transcribe_metadata_fixture();
    let spf = 8.0 * metadata.hop_length as f32 / metadata.sample_rate_hz as f32;
    let frames = 250; // 20.0s >= PEAK_FALLBACK_MIN_SECONDS
    let duration = (frames as f32) * spf;
    assert!(duration >= COHERE_DTW_PEAK_FALLBACK_MIN_SECONDS);
    let rows = zigzag_no_sink_rows(frames);
    let token_ids: Vec<u32> = (50..50 + rows.len() as u32).collect();
    let alignments: Vec<(u32, Vec<f32>)> = rows
        .iter()
        .zip(&token_ids)
        .map(|(row, &id)| (id, row.to_vec()))
        .collect();
    let decode_text = |token_ids: &[u32]| {
        let mut s = String::new();
        for &id in token_ids {
            use std::fmt::Write;
            let _ = write!(s, "w{id} ");
        }
        Ok(s.trim_end().to_string())
    };
    // Pre-check: raw peaks zigzag (strict fails), and backward fraction is
    // > 10% (tolerant also fails). No early sink is detected.
    let is_content = vec![true; rows.len()];
    assert!(
        !cross_attention_peaks_order_aligned(&rows, &is_content),
        "raw peaks must zigzag for the strict test to fail"
    );
    assert!(
        mask_dominant_early_sinks(&rows, &is_content).is_none(),
        "no dominant early sink is expected in this pattern"
    );
    assert!(
        content_backward_fraction(&rows, &is_content) > COHERE_DTW_MAX_BACKWARD_PAIR_FRACTION,
        "backward fraction must exceed the tolerant threshold"
    );
    let words = cohere_dtw_word_timestamps::<()>(
        &alignments,
        metadata,
        &vec![0.99; token_ids.len()],
        duration,
        0.0,
        &decode_text,
    )
    .expect("peak fallback words");
    assert!(
        !words.is_empty(),
        "long zigzag window must emit peak-fallback words, got empty"
    );
    // Each word must be at a distinct position (not spread uniformly).
    // The first word starts near the clip start; the last ends at or below
    // the last content-peak center (NOT at the full duration).
    assert!(
        words[0].start >= 0.0,
        "first word must not start before clip start"
    );
    let last_end = words.last().map(|w| w.end).unwrap();
    assert!(
        last_end < duration,
        "last word end ({last_end}) must be bounded by the last content-peak center, not the full duration ({duration})"
    );
    // Monotone and non-overlapping.
    for pair in words.windows(2) {
        assert!(
            pair[0].start <= pair[1].start + 1e-6,
            "timeline must be monotone"
        );
        assert!(pair[0].end - 1e-6 <= pair[1].start, "no overlaps");
    }
}

#[test]
fn cohere_dtw_word_timestamps_falls_back_to_uniform_on_short_zigzag_window() {
    let metadata = cohere_transcribe_metadata_fixture();
    let spf = 8.0 * metadata.hop_length as f32 / metadata.sample_rate_hz as f32;
    let frames = 110; // 8.8s < PEAK_FALLBACK_MIN_SECONDS
    let duration = (frames as f32) * spf;
    assert!(duration < COHERE_DTW_PEAK_FALLBACK_MIN_SECONDS);
    let rows = zigzag_no_sink_rows(frames);
    let token_ids: Vec<u32> = (50..50 + rows.len() as u32).collect();
    let alignments: Vec<(u32, Vec<f32>)> = rows
        .iter()
        .zip(&token_ids)
        .map(|(row, &id)| (id, row.to_vec()))
        .collect();
    let decode_text = |token_ids: &[u32]| {
        let mut s = String::new();
        for &id in token_ids {
            use std::fmt::Write;
            let _ = write!(s, "w{id} ");
        }
        Ok(s.trim_end().to_string())
    };
    let words = cohere_dtw_word_timestamps::<()>(
        &alignments,
        metadata,
        &vec![0.99; token_ids.len()],
        duration,
        0.0,
        &decode_text,
    )
    .expect("dtw words");
    assert!(
        words.is_empty(),
        "short window below the peak-fallback threshold must fall back to uniform (empty), got {words:?}"
    );
}
