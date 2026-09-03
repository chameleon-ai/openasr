use crate::api::backend::WordTimestamp;
use crate::models::decode_policy_component_registry::{
    BuiltinDecodePolicySeq2SeqTextPostprocessKind, seq2seq_transcript_byte_start,
};
use crate::models::text_prefix::common_prefix_len;

/// Boundary at the equidistant midpoint between two word centers
/// (`prev + 0.5 * (this - prev)`): each word owns the audio up to halfway into
/// its neighbour.
pub(crate) const MIDPOINT_BOUNDARY_FRACTION: f32 = 0.5;
/// No correction applied between a decoded center and the spoken onset.
pub(crate) const NO_ONSET_LEAD: f32 = 0.0;

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Seq2SeqTokenTime {
    pub token_id: u32,
    pub center_seconds: f32,
    /// Softmax probability of this token at its decode step, when captured.
    /// A word's confidence is the mean over its contributing tokens.
    pub probability: Option<f32>,
}

#[derive(Debug, Clone, PartialEq)]
struct WordCenter {
    word: String,
    center_seconds: f32,
    confidence: Option<f32>,
}

#[derive(Debug, Clone, Default)]
struct WordAccumulator {
    text: String,
    center_sum: f32,
    center_count: usize,
    probability_sum: f32,
    probability_count: usize,
}

impl WordAccumulator {
    fn new() -> Self {
        Self::default()
    }

    fn push_char(
        &mut self,
        ch: char,
        center_seconds: f32,
        probability: Option<f32>,
        contributed: &mut bool,
    ) {
        self.text.push(ch);
        if !*contributed {
            self.center_sum += center_seconds;
            self.center_count = self.center_count.saturating_add(1);
            if let Some(probability) = probability {
                self.probability_sum += probability;
                self.probability_count = self.probability_count.saturating_add(1);
            }
            *contributed = true;
        }
    }

    fn finish(&mut self, segment_start: f32, segment_end: f32) -> Option<WordCenter> {
        let word = self.text.trim().to_string();
        if word.is_empty() || self.center_count == 0 {
            *self = Self::default();
            return None;
        }
        let center_seconds = (self.center_sum / self.center_count as f32)
            .clamp(segment_start, segment_end)
            .max(segment_start);
        let confidence = (self.probability_count > 0)
            .then(|| (self.probability_sum / self.probability_count as f32).clamp(0.0, 1.0));
        *self = Self::default();
        Some(WordCenter {
            word,
            center_seconds,
            confidence,
        })
    }

    fn last_char(&self) -> Option<char> {
        self.text.chars().next_back()
    }
}

pub(crate) fn seq2seq_word_timestamps_from_generated_tokens<E>(
    generated_tokens: &[u32],
    token_probabilities: &[f32],
    segment_start: f32,
    segment_end: f32,
    postprocess_kind: BuiltinDecodePolicySeq2SeqTextPostprocessKind,
    decode_text_token_ids: &dyn Fn(&[u32]) -> Result<String, E>,
) -> Result<Vec<WordTimestamp>, E> {
    let span = sanitize_segment_span(segment_start, segment_end);
    if generated_tokens.is_empty() {
        return Ok(Vec::new());
    }
    let duration = (span.1 - span.0).max(0.0);
    let denominator = generated_tokens.len() as f32;
    let token_times = generated_tokens
        .iter()
        .enumerate()
        .map(|(index, token_id)| Seq2SeqTokenTime {
            token_id: *token_id,
            center_seconds: span.0 + duration * ((index as f32 + 0.5) / denominator),
            probability: token_probabilities.get(index).copied(),
        })
        .collect::<Vec<_>>();
    seq2seq_word_timestamps_from_token_times(
        &token_times,
        span.0,
        span.1,
        postprocess_kind,
        decode_text_token_ids,
        MIDPOINT_BOUNDARY_FRACTION,
        NO_ONSET_LEAD,
        f32::INFINITY,
    )
}

/// Fold per-token center times into word timestamps.
///
/// `boundary_fraction` and `onset_lead` tune how each word's window is cut
/// around its center (see [`word_centers_to_timestamps`]). The plain
/// midpoint/no-lead fold (`0.5`, `0.0`) is the default behavior: a word owns
/// the audio up to halfway into each neighbour, with no correction between
/// where the model's attention centers the token and when it is spoken.
///
/// `max_edge_word_span` clamps how far the *first* word's start may precede its
/// own center and the *last* word's end may follow its own center (each bound is
/// capped at `max_edge_word_span / 2` from the edge word's center). The fold
/// otherwise anchors the first word's start to `segment_start` and the last
/// word's end to `segment_end`; when those edges sit in leading/trailing
/// silence far from the edge word's center, that anchor smears the word across
/// the whole gap. `f32::INFINITY` disables the clamp and reproduces the
/// historical anchor exactly.
pub(crate) fn seq2seq_word_timestamps_from_token_times<E>(
    token_times: &[Seq2SeqTokenTime],
    segment_start: f32,
    segment_end: f32,
    postprocess_kind: BuiltinDecodePolicySeq2SeqTextPostprocessKind,
    decode_text_token_ids: &dyn Fn(&[u32]) -> Result<String, E>,
    boundary_fraction: f32,
    onset_lead: f32,
    max_edge_word_span: f32,
) -> Result<Vec<WordTimestamp>, E> {
    let (segment_start, segment_end) = sanitize_segment_span(segment_start, segment_end);

    // Pass 1: incremental decode into one text piece per token. Incremental
    // decode is normally monotonic (previous_decoded is a prefix of decoded).
    // If a BPE/byte-fallback recombination breaks that, fall back to the
    // longest-common-prefix delta rather than attributing the entire
    // transcript to this single token's center time.
    struct TokenPiece {
        text: String,
        center_seconds: f32,
        probability: Option<f32>,
    }
    let mut pieces = Vec::with_capacity(token_times.len());
    let mut full_decoded = String::new();
    let mut prefix_tokens = Vec::with_capacity(token_times.len());
    let mut previous_decoded = String::new();
    for token_time in token_times {
        let center_seconds = if token_time.center_seconds.is_finite() {
            token_time.center_seconds.clamp(segment_start, segment_end)
        } else {
            segment_start
        };
        prefix_tokens.push(token_time.token_id);
        let decoded = decode_text_token_ids(&prefix_tokens)?;
        let piece = match decoded.strip_prefix(&previous_decoded) {
            Some(rest) => rest.to_string(),
            None => {
                let shared = common_prefix_len(&previous_decoded, &decoded);
                decoded[shared..].to_string()
            }
        };
        previous_decoded = decoded;
        if piece.is_empty() {
            continue;
        }
        full_decoded.push_str(&piece);
        pieces.push(TokenPiece {
            text: piece,
            center_seconds,
            probability: token_time.probability,
        });
    }

    // The word stream must match the postprocessed transcript: characters
    // before the transcript start (e.g. qwen's control prefix) never form words.
    let transcript_byte_start = seq2seq_transcript_byte_start(postprocess_kind, &full_decoded);

    // Pass 2: fold transcript characters into words. Whitespace closes the
    // current word; a Han ideograph is a word of its own (CJK transcripts
    // carry no spaces), so it both closes the current word and bounds against
    // a following alphanumeric run.
    let mut words = Vec::new();
    let mut current = WordAccumulator::new();
    let mut byte_offset = 0usize;
    for piece in &pieces {
        let mut contributed_to_current = false;
        for ch in piece.text.chars() {
            let ch_offset = byte_offset;
            byte_offset += ch.len_utf8();
            if ch_offset < transcript_byte_start {
                continue;
            }
            if ch.is_whitespace() {
                if let Some(word) = current.finish(segment_start, segment_end) {
                    words.push(word);
                }
                contributed_to_current = false;
                continue;
            }
            if han_script_boundary_before(ch, current.last_char()) {
                if let Some(word) = current.finish(segment_start, segment_end) {
                    words.push(word);
                }
                contributed_to_current = false;
            }
            current.push_char(
                ch,
                piece.center_seconds,
                piece.probability,
                &mut contributed_to_current,
            );
        }
    }
    if let Some(word) = current.finish(segment_start, segment_end) {
        words.push(word);
    }
    Ok(word_centers_to_timestamps(
        words,
        segment_start,
        segment_end,
        boundary_fraction,
        onset_lead,
        max_edge_word_span,
    ))
}

/// Word boundary in unspaced CJK text: a Han ideograph always starts a new
/// word, and an alphanumeric character after a Han-final word starts a new
/// word ("用Rust写" -> 用 / Rust / 写). Trailing CJK punctuation stays attached
/// to the ideograph it follows ("好，" is one word).
fn han_script_boundary_before(ch: char, previous_char: Option<char>) -> bool {
    let Some(last) = previous_char else {
        return false;
    };
    if is_han_ideograph(ch) {
        return true;
    }
    ch.is_alphanumeric() && is_han_ideograph(last)
}

fn is_han_ideograph(ch: char) -> bool {
    matches!(
        u32::from(ch),
        0x3400..=0x4DBF      // CJK Unified Ideographs Extension A
        | 0x4E00..=0x9FFF    // CJK Unified Ideographs
        | 0xF900..=0xFAFF    // CJK Compatibility Ideographs
        | 0x20000..=0x2FA1F  // Extensions B..F + Compatibility Supplement
        | 0x30000..=0x3134F  // Extensions G..H
    )
}

/// Split one word's timeline from the centers of itself and its neighbours.
///
/// `boundary_fraction` places the boundary between two words at
/// `prev + fraction * (this - prev)` of the gap. `0.5` is the equidistant
/// midpoint (a word owns the audio up to halfway into its neighbour); a
/// smaller fraction moves the boundary closer to this word's own center,
/// which -- for a center that lags the spoken onset -- pulls every word's start
/// earlier, toward when its speech actually begins.
fn boundary(prev_center: f32, this_center: f32, fraction: f32) -> f32 {
    prev_center + fraction * (this_center - prev_center)
}

fn word_centers_to_timestamps(
    mut words: Vec<WordCenter>,
    segment_start: f32,
    segment_end: f32,
    boundary_fraction: f32,
    onset_lead: f32,
    max_edge_word_span: f32,
) -> Vec<WordTimestamp> {
    if words.is_empty() {
        return Vec::new();
    }
    // A center that lags the spoken onset (the model's attention concentrates a
    // token slightly after it starts) leaves every boundary behind the true
    // word start. `onset_lead` shifts each center earlier by a fixed amount,
    // the measured late-onset bias, so the boundaries fall where the speech
    // actually begins. `segment_start` is the floor: the first word's window
    // still covers from the band start, never from a time before the spoken
    // audio.
    let lead = onset_lead.max(0.0);
    let mut last_center = segment_start;
    for word in &mut words {
        if !word.center_seconds.is_finite() {
            word.center_seconds = last_center;
        } else {
            word.center_seconds -= lead;
        }
        word.center_seconds = word.center_seconds.clamp(last_center, segment_end);
        last_center = word.center_seconds;
    }

    // The first word's start and the last word's end are anchored to the
    // segment edges by construction. When those edges fall inside
    // leading/trailing silence (common on chunks whose middle speech was
    // collapsed, or where the first word's speech is a few seconds in), the
    // anchor smears the word across the whole gap. `max_edge_word_span / 2`
    // bounds how far an edge word's window may extend away from its own
    // center; a word whose center sits at the edge is unaffected.
    let half_span = if max_edge_word_span.is_finite() {
        (max_edge_word_span / 2.0).max(0.0)
    } else {
        f32::INFINITY
    };

    let mut timestamps = Vec::with_capacity(words.len());
    let first_center = words
        .first()
        .map(|w| w.center_seconds)
        .unwrap_or(segment_start);
    let last_center = words
        .last()
        .map(|w| w.center_seconds)
        .unwrap_or(segment_end);
    let first_min_start = if half_span.is_finite() {
        (first_center - half_span).max(segment_start)
    } else {
        segment_start
    };
    let last_max_end = if half_span.is_finite() {
        (last_center + half_span).min(segment_end)
    } else {
        segment_end
    };
    for (index, word) in words.iter().enumerate() {
        let start = if index == 0 {
            first_min_start
        } else {
            boundary(
                words[index - 1].center_seconds,
                word.center_seconds,
                boundary_fraction,
            )
        };
        let end = if index + 1 == words.len() {
            last_max_end
        } else {
            boundary(
                word.center_seconds,
                words[index + 1].center_seconds,
                boundary_fraction,
            )
        };
        let start = start.clamp(segment_start, segment_end);
        let end = end.clamp(start, segment_end);
        timestamps.push(WordTimestamp {
            word: word.word.clone(),
            start,
            end,
            confidence: word.confidence,
        });
    }
    timestamps
}

fn sanitize_segment_span(start: f32, end: f32) -> (f32, f32) {
    let start = if start.is_finite() {
        start.max(0.0)
    } else {
        0.0
    };
    let end = if end.is_finite() {
        end.max(start)
    } else {
        start
    };
    (start, end)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_position_map_builds_monotonic_word_timestamps() {
        let pieces = |ids: &[u32]| {
            Ok::<_, std::convert::Infallible>(
                ids.iter()
                    .map(|id| match id {
                        1 => "hello",
                        2 => " world",
                        other => panic!("unexpected token {other}"),
                    })
                    .collect::<String>(),
            )
        };

        let words = seq2seq_word_timestamps_from_generated_tokens(
            &[1, 2],
            &[0.9, 0.7],
            2.0,
            4.0,
            BuiltinDecodePolicySeq2SeqTextPostprocessKind::Identity,
            &pieces,
        )
        .unwrap();

        assert_eq!(words.len(), 2);
        assert_eq!(words[0].word, "hello");
        assert_eq!(words[1].word, "world");
        assert_eq!(words[0].start, 2.0);
        assert!(words[0].end <= words[1].start);
        assert_eq!(words[1].end, 4.0);
    }

    #[test]
    fn explicit_token_times_drive_word_boundaries() {
        let pieces = |ids: &[u32]| {
            Ok::<_, std::convert::Infallible>(
                ids.iter()
                    .map(|id| match id {
                        1 => "open",
                        2 => "asr",
                        3 => " rocks",
                        other => panic!("unexpected token {other}"),
                    })
                    .collect::<String>(),
            )
        };
        let token_times = vec![
            Seq2SeqTokenTime {
                token_id: 1,
                center_seconds: 0.2,
                probability: Some(0.9),
            },
            Seq2SeqTokenTime {
                token_id: 2,
                center_seconds: 0.4,
                probability: Some(0.5),
            },
            Seq2SeqTokenTime {
                token_id: 3,
                center_seconds: 1.6,
                probability: None,
            },
        ];

        let words = seq2seq_word_timestamps_from_token_times(
            &token_times,
            0.0,
            2.0,
            BuiltinDecodePolicySeq2SeqTextPostprocessKind::Identity,
            &pieces,
            MIDPOINT_BOUNDARY_FRACTION,
            NO_ONSET_LEAD,
            f32::INFINITY,
        )
        .unwrap();

        assert_eq!(words.len(), 2);
        assert_eq!(words[0].word, "openasr");
        assert_eq!(words[1].word, "rocks");
        assert!(words[0].end > 0.8);
        assert!(words[1].start >= words[0].end);
        assert_eq!(words[1].end, 2.0);
        // "openasr" was contributed by tokens with p=0.9 and p=0.5 -> mean
        // 0.7; " rocks" came from a token with no captured probability.
        assert!((words[0].confidence.unwrap() - 0.7).abs() < 1e-6);
        assert_eq!(words[1].confidence, None);
    }

    #[test]
    fn edge_word_clamp_binds_words_to_their_center_when_they_sit_far_from_the_edge() {
        // Segment [0, 2s] with two words whose centers are near the far ends
        // and a big gap between them. Without the clamp the first word's
        // start stays anchored at 0.0 and the last word's end stays anchored
        // at 2.0, so each spans a full second of leading/trailing silence.
        // With max_edge_word_span = 0.2 each edge word is bounded to 0.1s on
        // either side of its own center.
        let pieces = |ids: &[u32]| {
            Ok::<_, std::convert::Infallible>(
                ids.iter()
                    .map(|id| match id {
                        1 => "you",
                        2 => " deserve",
                        other => panic!("unexpected token {other}"),
                    })
                    .collect::<String>(),
            )
        };
        let token_times = vec![
            Seq2SeqTokenTime {
                token_id: 1,
                center_seconds: 0.3,
                probability: None,
            },
            Seq2SeqTokenTime {
                token_id: 2,
                center_seconds: 1.7,
                probability: None,
            },
        ];

        let clamped = seq2seq_word_timestamps_from_token_times(
            &token_times,
            0.0,
            2.0,
            BuiltinDecodePolicySeq2SeqTextPostprocessKind::Identity,
            &pieces,
            0.5,
            0.0,
            0.2,
        )
        .unwrap();
        assert_eq!(clamped.len(), 2);
        // First word: start is max(0.0, 0.3 - 0.1) = 0.2, not 0.0.
        assert!((clamped[0].start - 0.2).abs() < 1e-6);
        // Last word: end is min(2.0, 1.7 + 0.1) = 1.8, not 2.0.
        assert!((clamped[1].end - 1.8).abs() < 1e-6);
        // Timeline stays monotone and non-overlapping.
        assert!(clamped[0].end <= clamped[1].start);

        let unclamped = seq2seq_word_timestamps_from_token_times(
            &token_times,
            0.0,
            2.0,
            BuiltinDecodePolicySeq2SeqTextPostprocessKind::Identity,
            &pieces,
            0.5,
            0.0,
            f32::INFINITY,
        )
        .unwrap();
        assert!((unclamped[0].start - 0.0).abs() < 1e-6);
        assert!((unclamped[1].end - 2.0).abs() < 1e-6);
    }

    #[test]
    fn edge_word_clamp_is_noop_when_the_edge_word_center_is_at_the_edge() {
        // When the first word's center coincides with the segment start the
        // clamp has nothing to trim: the word still starts at the band edge.
        let pieces = |ids: &[u32]| {
            Ok::<_, std::convert::Infallible>(
                ids.iter()
                    .map(|id| match id {
                        1 => "hi",
                        2 => " there",
                        other => panic!("unexpected token {other}"),
                    })
                    .collect::<String>(),
            )
        };
        let token_times = vec![
            Seq2SeqTokenTime {
                token_id: 1,
                center_seconds: 0.0,
                probability: None,
            },
            Seq2SeqTokenTime {
                token_id: 2,
                center_seconds: 1.0,
                probability: None,
            },
        ];
        let words = seq2seq_word_timestamps_from_token_times(
            &token_times,
            0.0,
            2.0,
            BuiltinDecodePolicySeq2SeqTextPostprocessKind::Identity,
            &pieces,
            0.5,
            0.0,
            0.2,
        )
        .unwrap();
        assert_eq!(words.len(), 2);
        // First word's center is at 0.0, so max(0.0, 0.0 - 0.1) = 0.0.
        assert!((words[0].start - 0.0).abs() < 1e-6);
        // Last word's center is at 1.0, so min(2.0, 1.0 + 0.1) = 1.1.
        assert!((words[1].end - 1.1).abs() < 1e-6);
    }

    #[test]
    fn qwen_control_prefix_never_forms_words() {
        // Mirrors the text path's Qwen3AsrStripControlPrefixV0: everything up
        // to and including "<asr_text>" is decode scaffolding, not transcript.
        let pieces = |ids: &[u32]| {
            Ok::<_, std::convert::Infallible>(
                ids.iter()
                    .map(|id| match id {
                        1 => "language English",
                        2 => "<asr_text>",
                        3 => "hello",
                        4 => " there",
                        other => panic!("unexpected token {other}"),
                    })
                    .collect::<String>(),
            )
        };

        let words = seq2seq_word_timestamps_from_generated_tokens(
            &[1, 2, 3, 4],
            &[0.9, 0.9, 0.8, 0.6],
            0.0,
            4.0,
            BuiltinDecodePolicySeq2SeqTextPostprocessKind::Qwen3AsrStripControlPrefixV0,
            &pieces,
        )
        .unwrap();

        let texts = words.iter().map(|w| w.word.as_str()).collect::<Vec<_>>();
        assert_eq!(texts, ["hello", "there"]);
        assert!((words[0].confidence.unwrap() - 0.8).abs() < 1e-6);
        assert!((words[1].confidence.unwrap() - 0.6).abs() < 1e-6);
    }

    #[test]
    fn han_ideographs_split_into_single_character_words() {
        // Unspaced CJK: each ideograph is its own word; trailing CJK
        // punctuation stays attached; latin runs bound against ideographs.
        let pieces = |ids: &[u32]| {
            Ok::<_, std::convert::Infallible>(
                ids.iter()
                    .map(|id| match id {
                        1 => "你好，",
                        2 => "用Rust",
                        3 => "写代码",
                        other => panic!("unexpected token {other}"),
                    })
                    .collect::<String>(),
            )
        };

        let words = seq2seq_word_timestamps_from_generated_tokens(
            &[1, 2, 3],
            &[0.9, 0.8, 0.7],
            0.0,
            3.0,
            BuiltinDecodePolicySeq2SeqTextPostprocessKind::Identity,
            &pieces,
        )
        .unwrap();

        let texts = words.iter().map(|w| w.word.as_str()).collect::<Vec<_>>();
        assert_eq!(texts, ["你", "好，", "用", "Rust", "写", "代", "码"]);
        assert!(words.windows(2).all(|pair| pair[0].end <= pair[1].start));
        assert!((words[3].confidence.unwrap() - 0.8).abs() < 1e-6);
    }
}
