use super::*;

#[test]
fn whisper_dtw_onset_lead_is_the_flat_baseline() {
    // The onset lead is the single flat constant (no density-scaled curve); the
    // runtime reads it via whisper_dtw_onset_lead, whose env-override fallback is
    // the compiled default. Pin the constant here rather than mutating process
    // env, which is unsafe in this edition and races under parallel nextest.
    assert!((WHISPER_DTW_ONSET_LEAD_SECONDS - 0.05).abs() < 1e-6);
}

#[test]
fn whisper_dtw_lead_silence_advance_fires_only_on_a_leading_leak() {
    let spf = 0.02_f32; // 1500 frames over a 30s window.
    let min_gap = WHISPER_DTW_LEAD_SILENCE_ADVANCE_MIN_GAP_SECONDS; // 0.2s -> 10 frames.

    // A run at the window front whose content onset sits well past the bound
    // (a leading silence leak) is advanced to that onset.
    let advance = whisper_dtw_lead_silence_advance_frame(0, Some(55), spf, min_gap); // 1.1s gap.
    assert_eq!(advance, Some(55));

    // The same onset but on a mid-run decoded `<|start|>` bound is NOT advanced:
    // the band_start == 0 gate is what keeps a real timestamp (which can mark a
    // large misalignment) from retargeting the lead word to an unrelated peak.
    let mid_run = whisper_dtw_lead_silence_advance_frame(300, Some(355), spf, min_gap); // 1.1s gap.
    assert_eq!(mid_run, None);

    // A window-front onset just over the minimum gap fires; a gap just under it
    // is normal `<|start|>` jitter, not a leak, so the sub-margin gate keeps it
    // untouched. (Values are kept clearly off the margin since the exact 0.2s
    // boundary is an arbitrary f32 knife-edge, not a meaningful threshold.)
    let over = whisper_dtw_lead_silence_advance_frame(0, Some(11), spf, min_gap); // 0.22s gap.
    assert_eq!(over, Some(11));
    let under = whisper_dtw_lead_silence_advance_frame(0, Some(9), spf, min_gap); // 0.18s gap.
    assert_eq!(under, None);

    // No usable content peak: nothing to advance to.
    let no_front = whisper_dtw_lead_silence_advance_frame(0, None, spf, min_gap);
    assert_eq!(no_front, None);
}

// ---------------------------------------------------------------------------
// whisper_refine_dtw_word_onsets
// ---------------------------------------------------------------------------

fn word_ts(word: &str, start: f32, end: f32) -> crate::WordTimestamp {
    crate::WordTimestamp {
        word: word.to_string(),
        start,
        end,
        confidence: None,
    }
}

/// A 15 s, 0.02 s/frame envelope (750 frames) at a 0.001 noise floor with a
/// single 0.5 peak at 8 s that sets the clip peak (and so the 5% silence
/// ceiling). The [2.0, 4.0) word-b window is filled from 0.001 up to 3.8 s and
/// a 0.25 speech onset occupies [3.8, 4.0).
fn refine_fixture_envelope() -> Vec<f32> {
    let mut env = vec![0.001f32; 750];
    env[400] = 0.5;
    for s in env[190..200].iter_mut() {
        *s = 0.25;
    }
    env
}

/// `whisper_refine_dtw_word_onsets` advances a word the fold parked in true
/// zero-silence to its real onset at 3.8 s: the previous word's boundary is
/// left untouched, so a real gap is opened where the pause sits.
#[test]
fn refine_dtw_onsets_pushes_true_silence_word_to_its_onset() {
    let words = vec![word_ts("a", 0.5, 0.6), word_ts("b", 2.0, 4.0)];
    let env = refine_fixture_envelope();
    let out = whisper_refine_dtw_word_onsets(words, Some(&env), 15.0);
    assert!((out[1].start - 3.8).abs() < 0.05, "start={}", out[1].start);
    // The first word is never modified.
    assert!((out[0].start - 0.5).abs() < 1e-4 && (out[0].end - 0.6).abs() < 1e-4);
}

/// The same window with a low music floor filling the front half (a sustained
/// level, so the front's mean sits above the floor) is *not* trusted as a
/// pause: a quiet passage over a music bed is ambiguous, so no push fires and
/// the word keeps its fold position. This is the gate that stops the refinement
/// from regressing continuous-speech / music-backed clips.
#[test]
fn refine_dtw_onsets_refuses_a_music_floor_front() {
    let mut env = refine_fixture_envelope();
    for s in env[100..190].iter_mut() {
        *s = 0.021;
    }
    let words = vec![word_ts("a", 0.5, 0.6), word_ts("b", 2.0, 4.0)];
    let out = whisper_refine_dtw_word_onsets(words, Some(&env), 15.0);
    assert!((out[1].start - 2.0).abs() < 1e-4, "start={}", out[1].start);
    assert!((out[1].end - 4.0).abs() < 1e-4);
}

/// A boundary word whose start maps to or past the last envelope frame -- common
/// at a longform slice end, where the frame array is shorter than
/// `duration_s / seconds_per_frame` -- must not overrun the slice. Pre-fix this
/// indexed out of bounds and panicked (`range end index ... out of range`);
/// post-fix the word is clamped into range and, finding no usable window, is
/// left unrefined rather than aborting the run.
#[test]
fn refine_dtw_onsets_clamps_a_word_at_or_past_the_end() {
    let env = refine_fixture_envelope(); // 750 frames
    let words = vec![
        word_ts("a", 0.5, 0.6),
        word_ts("b", 15.5, 16.0), // start past the 750-frame end, span 0.5 >= 0.3
    ];
    // duration_s larger than the envelope implies so `start_s / spf` overshoots.
    let out = whisper_refine_dtw_word_onsets(words, Some(&env), 16.0);
    assert_eq!(out[1].start, 15.5, "unrefined; must not panic");
    assert_eq!(out[1].end, 16.0, "unrefined; must not panic");
}

/// No envelope (a run without cross-attention word timestamps) is a byte-exact
/// no-op.
#[test]
fn refine_dtw_onsets_noop_without_envelope() {
    let words = vec![word_ts("a", 0.5, 0.6), word_ts("b", 2.0, 4.0)];
    let out = whisper_refine_dtw_word_onsets(words, None, 15.0);
    assert_eq!(out[0].start, 0.5);
    assert_eq!(out[1].start, 2.0);
    assert_eq!(out[1].end, 4.0);
}

// ---------------------------------------------------------------------------
// whisper_refine_dtw_word_offsets
// ---------------------------------------------------------------------------

/// A 15 s, 0.02 s/frame envelope (750 frames) at a 0.001 noise floor with a
/// single 0.5 peak at 8 s that sets the clip peak (and so the 5% silence
/// ceiling). The [2.0, 4.0) word window has a 0.25 speech run in [2.0, 2.6)
/// followed by digital-zero silence to 4.0 s -- the trailing-silence (hollow
/// back) shape the offset refinement retreats.
fn offset_fixture_envelope() -> Vec<f32> {
    let mut env = vec![0.001f32; 750];
    env[400] = 0.5;
    for s in env[100..130].iter_mut() {
        *s = 0.25;
    }
    env
}

/// `whisper_refine_dtw_word_offsets` retreats a word the fold let run past its
/// speech into the trailing silence back to its real offset at 2.6 s: the next
/// word's start is left untouched, so a real gap is opened where the pause sits.
#[test]
fn refine_dtw_offsets_pulls_true_silence_word_to_its_offset() {
    let words = vec![
        word_ts("a", 0.5, 0.6),
        word_ts("b", 2.0, 4.0),
        word_ts("c", 4.0, 4.5),
    ];
    let env = offset_fixture_envelope();
    let out = whisper_refine_dtw_word_offsets(words, Some(&env), 15.0);
    assert!((out[1].end - 2.6).abs() < 0.05, "end={}", out[1].end);
    // start is untouched, as is the next word.
    assert!((out[1].start - 2.0).abs() < 1e-4 && (out[2].end - 4.5).abs() < 1e-4);
}

/// The same window but with a low music floor filling the back half (a
/// sustained level, so the back's mean sits above the floor) is *not* trusted
/// as trailing silence: a quiet passage over a music bed is ambiguous, so no
/// pull fires and the word keeps its fold position.
#[test]
fn refine_dtw_offsets_refuses_a_music_floor_back() {
    let mut env = offset_fixture_envelope();
    for s in env[150..200].iter_mut() {
        *s = 0.021;
    }
    let words = vec![
        word_ts("a", 0.5, 0.6),
        word_ts("b", 2.0, 4.0),
        word_ts("c", 4.0, 4.5),
    ];
    let out = whisper_refine_dtw_word_offsets(words, Some(&env), 15.0);
    assert!((out[1].end - 4.0).abs() < 1e-4, "end={}", out[1].end);
    assert!((out[1].start - 2.0).abs() < 1e-4);
}

/// A middle word whose end maps to or past the last envelope frame -- common at
/// a longform slice end where the frame array is shorter than
/// `duration_s / seconds_per_frame` -- must not overrun the slice. The word is
/// clamped into range and, finding no usable window, is left unrefined rather
/// than aborting the run.
#[test]
fn refine_dtw_offsets_clamps_a_word_at_or_past_the_end() {
    let env = offset_fixture_envelope(); // 750 frames
    let words = vec![
        word_ts("a", 0.5, 0.6),
        word_ts("b", 15.4, 16.0), // end past the 750-frame end, span 0.6 >= 0.3
        word_ts("c", 16.0, 16.2), // the last word is skipped regardless
    ];
    // duration_s larger than the envelope implies so `end_s / spf` overshoots.
    let out = whisper_refine_dtw_word_offsets(words, Some(&env), 16.0);
    assert_eq!(out[1].end, 16.0, "unrefined; must not panic");
}

/// The last word is never retreated: its true end is the audio end, so the
/// silence after it is the clip's legitimate tail, not a fold leak.
#[test]
fn refine_dtw_offsets_skips_the_last_word() {
    let mut env = offset_fixture_envelope();
    for s in env[0..600].iter_mut() {
        *s = 0.25; // trailing word's window [11.5, 13.5) sits in speech to its end
    }
    let words = vec![
        word_ts("a", 0.5, 0.6),
        word_ts("b", 11.5, 13.5), // the last word
    ];
    let out = whisper_refine_dtw_word_offsets(words, Some(&env), 15.0);
    assert_eq!(out[1].end, 13.5, "last word untouched");
}

/// No envelope (a run without cross-attention word timestamps) is a byte-exact
/// no-op.
#[test]
fn refine_dtw_offsets_noop_without_envelope() {
    let words = vec![
        word_ts("a", 0.5, 0.6),
        word_ts("b", 2.0, 4.0),
        word_ts("c", 4.0, 4.5),
    ];
    let out = whisper_refine_dtw_word_offsets(words, None, 15.0);
    assert_eq!(out[1].end, 4.0);
}

// ---------------------------------------------------------------------------
// whisper_pad_dtw_word_windows
// ---------------------------------------------------------------------------

/// An interior word is widened on both sides by exactly the pads, while the
/// first word's start is clamped to 0.0 and the last word's end is clamped to
/// the audio duration; interior order is preserved.
#[test]
fn pad_dtw_word_windows_widens_toward_the_edges_and_clamps_to_the_audio() {
    let words = vec![
        word_ts("a", 0.02, 0.70),
        word_ts("b", 0.70, 1.30),
        word_ts("c", 1.30, 14.98),
    ];
    let out = whisper_pad_dtw_word_windows(words, 15.0);
    assert!(
        (out[0].start - 0.0).abs() < 1e-4,
        "a.start={}",
        out[0].start
    );
    assert!((out[0].end - 0.80).abs() < 1e-4, "a.end={}", out[0].end);
    assert!(
        (out[1].start - 0.60).abs() < 1e-4,
        "b.start={}",
        out[1].start
    );
    assert!((out[1].end - 1.40).abs() < 1e-4, "b.end={}", out[1].end);
    assert!(
        (out[2].start - 1.20).abs() < 1e-4,
        "c.start={}",
        out[2].start
    );
    assert!((out[2].end - 15.0).abs() < 1e-4, "c.end={}", out[2].end);
    for (index, word) in out.iter().enumerate() {
        assert!(word.start <= word.end, "word[{index}] inverted");
        if index + 1 < out.len() {
            assert!(word.end <= out[index + 1].end);
        }
    }
}

/// Zero-duration audio clamps every window edge to 0.0 and keeps each window
/// non-negative; an empty input is a byte-exact no-op.
#[test]
fn pad_dtw_word_windows_is_a_noop_when_empty_and_clamps_zero_duration() {
    let empty = whisper_pad_dtw_word_windows(Vec::new(), 15.0);
    assert!(empty.is_empty());
    let zero = whisper_pad_dtw_word_windows(vec![word_ts("a", 0.30, 0.90)], 0.0);
    assert_eq!(zero[0].start, 0.0);
    assert_eq!(zero[0].end, 0.0);
}

// ---------------------------------------------------------------------------
// whisper_dtw_silence_ceiling
// ---------------------------------------------------------------------------

/// A dense bed (peak within the contrast gate of the median) keeps the
/// conservative 5%-of-peak ceiling; a thin floor raises the ceiling to 3x the
/// median when that is higher, and keeps the peak fraction when it dominates.
#[test]
fn dtw_silence_ceiling_rises_off_the_floor_only_on_thin_floors() {
    // Dense bed: 4x contrast, ceiling stays at 5% of peak.
    let dense = whisper_dtw_silence_ceiling(0.10, 0.40);
    assert!((dense - 0.02).abs() < 1e-9, "dense={dense}");
    // Thin floor (20x contrast) where the floor multiple (3x) beats the
    // peak fraction (5%).
    let thin = whisper_dtw_silence_ceiling(0.01, 0.20);
    assert!((thin - 0.03).abs() < 1e-9, "thin={thin}");
    // Thin floor where the peak fraction still dominates.
    let thin_peak_wins = whisper_dtw_silence_ceiling(0.005, 0.20);
    assert!(
        (thin_peak_wins - 0.015).abs() < 1e-9,
        "thin_peak_wins={thin_peak_wins}"
    );
}

// ---------------------------------------------------------------------------
// whisper_reanchor_dtw_token_centers
// ---------------------------------------------------------------------------

fn reanchor_token(token_id: u32, center_seconds: f32) -> Seq2SeqTokenTime {
    Seq2SeqTokenTime {
        token_id,
        center_seconds,
        probability: None,
    }
}

/// 15 s of 0.02 s frames at a 0.01 thin noise floor with a sustained 0.20
/// speech run over [1.0, 1.8) (frames 50..89). Peak/median contrast is 20x,
/// so the silence ceiling is 3x the floor (0.03), not 5% of the peak (0.01).
/// The speech threshold is ~5 dB over the floor (0.0316).
fn reanchor_fixture_envelope() -> Vec<f32> {
    let mut env = vec![0.01f32; 750];
    for sample in env[50..90].iter_mut() {
        *sample = 0.20;
    }
    env
}

/// A word-final punctuation token the DTW path parked 1.0 s past its word's
/// audio in a thin-floor pause is pulled back to the frame just past the run's
/// end (one frame past [1.0, 1.8) -> 1.8). The word-content token before it
/// keeps its path-derived center.
#[test]
fn reanchor_dtw_token_centers_pulls_word_final_punctuation_off_a_pause() {
    let decode = |ids: &[u32]| -> Option<String> {
        match ids {
            [100] => Some("it".to_string()),
            [100, 101] => Some("it?".to_string()),
            _ => None,
        }
    };
    let out = whisper_reanchor_dtw_token_centers(
        vec![reanchor_token(100, 1.2), reanchor_token(101, 2.8)], // "?" 1.0 s past its word's run
        &decode,
        Some(&reanchor_fixture_envelope()),
        0.02,
    );
    assert!(
        (out[0].center_seconds - 1.2).abs() < 1e-3,
        "word content keeps its center"
    );
    assert!(
        (out[1].center_seconds - 1.8).abs() < 1e-3,
        "center should land one frame past the run's end, got {}",
        out[1].center_seconds
    );
}

/// The same pull fires across a ~3 s pause (the longest intra-word pause
/// measured on the test corpus, under the 3.5 s max jump).
#[test]
fn reanchor_dtw_token_centers_pulls_punctuation_across_a_three_second_pause() {
    let decode = |ids: &[u32]| -> Option<String> {
        match ids {
            [100] => Some("life".to_string()),
            [100, 101] => Some("life.".to_string()),
            _ => None,
        }
    };
    let out = whisper_reanchor_dtw_token_centers(
        vec![reanchor_token(100, 1.3), reanchor_token(101, 4.9)], // "." 3.1 s past its word's run
        &decode,
        Some(&reanchor_fixture_envelope()),
        0.02,
    );
    assert!(
        (out[1].center_seconds - 1.8).abs() < 1e-3,
        "center should land one frame past the run's end, got {}",
        out[1].center_seconds
    );
}

/// A token whose center sits on sustained speech (not in a pause) is refused:
/// its quiet-region test fails and the center stays where the path put it.
#[test]
fn reanchor_dtw_token_centers_refuses_an_entry_on_speech() {
    let decode = |ids: &[u32]| -> Option<String> {
        match ids {
            [100] => Some("it".to_string()),
            [100, 101] => Some("it?".to_string()),
            _ => None,
        }
    };
    let out = whisper_reanchor_dtw_token_centers(
        vec![reanchor_token(100, 0.8), reanchor_token(101, 1.4)], // "?" entry inside the run
        &decode,
        Some(&reanchor_fixture_envelope()),
        0.02,
    );
    assert_eq!(
        out[1].center_seconds, 1.4,
        "entry on speech keeps its center"
    );
}

/// A center with no preceding speech run has nothing to anchor to: even with
/// a later speech run, nothing before the center is trusted audio and the
/// center stays where the path put it.
#[test]
fn reanchor_dtw_token_centers_refuses_without_a_preceding_speech_run() {
    let mut env = vec![0.01f32; 750];
    for sample in env[250..300].iter_mut() {
        *sample = 0.20; // run [5.0, 6.0) sits AFTER the center
    }
    let decode = |ids: &[u32]| -> Option<String> {
        match ids {
            [100] => Some("it".to_string()),
            [100, 101] => Some("it?".to_string()),
            _ => None,
        }
    };
    let out = whisper_reanchor_dtw_token_centers(
        vec![reanchor_token(100, 0.0), reanchor_token(101, 10.0)],
        &decode,
        Some(&env),
        0.02,
    );
    assert_eq!(
        out[1].center_seconds, 10.0,
        "no preceding run keeps the center"
    );
}

/// A gap past the reanchor max jump (3.5 s) is a misalignment too large to
/// trust as an intra-word linger: the center is kept.
#[test]
fn reanchor_dtw_token_centers_refuses_a_gap_past_the_max() {
    let decode = |ids: &[u32]| -> Option<String> {
        match ids {
            [100] => Some("it".to_string()),
            [100, 101] => Some("it?".to_string()),
            _ => None,
        }
    };
    let out = whisper_reanchor_dtw_token_centers(
        vec![reanchor_token(100, 1.2), reanchor_token(101, 5.8)], // 4.0 s past the run
        &decode,
        Some(&reanchor_fixture_envelope()),
        0.02,
    );
    assert_eq!(
        out[1].center_seconds, 5.8,
        "gap past the max keeps its center"
    );
}

/// A dense bed (peak within 8x of the median) keeps the 5%-of-peak silence
/// ceiling, so a quiet passage over a music bed is never trusted as a pause
/// and the pass is a no-op: continuous music-backed clips keep their fold
/// output byte-for-byte.
#[test]
fn reanchor_dtw_token_centers_noop_on_a_dense_music_bed() {
    // Bed at 0.04 with a 0.30 speech run: contrast is 7.5 < 8, so the ceiling
    // stays 5% of the peak (0.015) and the bed's 0.04 fails the quiet test.
    let mut env = vec![0.04f32; 750];
    for sample in env[50..80].iter_mut() {
        *sample = 0.30;
    }
    let decode = |ids: &[u32]| -> Option<String> {
        match ids {
            [100] => Some("it".to_string()),
            [100, 101] => Some("it?".to_string()),
            _ => None,
        }
    };
    let out = whisper_reanchor_dtw_token_centers(
        vec![reanchor_token(100, 0.6), reanchor_token(101, 2.2)],
        &decode,
        Some(&env),
        0.02,
    );
    assert_eq!(out[1].center_seconds, 2.2, "music bed keeps its center");
}

/// A token whose piece carries letters or digits is real word content (a
/// subword the path may genuinely place in a quiet passage): the piece gate
/// refuses it before any audio test, even when every audio gate would fire.
#[test]
fn reanchor_dtw_token_centers_refuses_a_piece_with_letters() {
    let decode = |ids: &[u32]| -> Option<String> {
        match ids {
            [100] => Some("it".to_string()),
            [100, 101] => Some("ithi".to_string()), // piece "hi": content, not punctuation
            _ => None,
        }
    };
    let out = whisper_reanchor_dtw_token_centers(
        vec![reanchor_token(100, 1.2), reanchor_token(101, 2.8)],
        &decode,
        Some(&reanchor_fixture_envelope()),
        0.02,
    );
    assert_eq!(
        out[1].center_seconds, 2.8,
        "a content piece never reanchors"
    );
}

/// Word-initial punctuation contributes to the word it *opens*: the `"` in
/// `"hello` is the first, not the last, contributor to its word, so pulling
/// it back would smear the next word's start. The walk that mirrors the fold's
/// word split keeps it where the path put it, even though its own audio gates
/// (thin-floor pause 0.4 s after the previous run) would all fire.
#[test]
fn reanchor_dtw_token_centers_refuses_word_initial_punctuation() {
    let mut env = reanchor_fixture_envelope();
    for sample in env[150..190].iter_mut() {
        *sample = 0.20; // the opened word's own run [3.0, 3.8)
    }
    let decode = |ids: &[u32]| -> Option<String> {
        match ids {
            [300] => Some("\"".to_string()),
            [300, 301] => Some("\"hello".to_string()),
            _ => None,
        }
    };
    let out = whisper_reanchor_dtw_token_centers(
        vec![reanchor_token(300, 2.2), reanchor_token(301, 3.2)],
        &decode,
        Some(&env),
        0.02,
    );
    assert_eq!(
        out[0].center_seconds, 2.2,
        "word-initial punctuation keeps its center"
    );
}

/// A frame of bed level (above the thin-floor silence ceiling, below the
/// speech threshold) in the gap between the run and the center is an untrusted
/// frame: the walk finds no anchored run and the center stays.
#[test]
fn reanchor_dtw_token_centers_refuses_a_gap_with_an_untrusted_frame() {
    let mut env = reanchor_fixture_envelope();
    env[120] = 0.0305; // above the 0.03 ceiling, below the 0.0316 floor margin
    let decode = |ids: &[u32]| -> Option<String> {
        match ids {
            [100] => Some("it".to_string()),
            [100, 101] => Some("it?".to_string()),
            _ => None,
        }
    };
    let out = whisper_reanchor_dtw_token_centers(
        vec![reanchor_token(100, 1.2), reanchor_token(101, 2.8)],
        &decode,
        Some(&env),
        0.02,
    );
    assert_eq!(
        out[1].center_seconds, 2.8,
        "untrusted gap frame keeps the center"
    );
}

/// No envelope (a run without usable 16 k audio) is a byte-exact no-op.
#[test]
fn reanchor_dtw_token_centers_noop_without_envelope() {
    let decode = |ids: &[u32]| -> Option<String> {
        match ids {
            [100] => Some("it?".to_string()),
            _ => None,
        }
    };
    let out =
        whisper_reanchor_dtw_token_centers(vec![reanchor_token(100, 1.4)], &decode, None, 0.02);
    assert_eq!(out[0].center_seconds, 1.4);
}
