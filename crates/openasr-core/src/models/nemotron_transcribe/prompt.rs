//! Prompt planning for the nemotron-transcribe SALM-style decoder.
//!
//! The served prompts are chat-template documents with the `<|audio|>`
//! locator where the encoder's projected rows are spliced in at decode time.
//! The tokenizer has no `<|audio|>` id (the locator is *not* a vocabulary
//! entry - see `nemotron-reference-notes.md` section 6.2), so the prompt is
//! built as two token spans, one before and one after the audio rows:
//!
//! ```text
//! prefix = tokenize(PREFIX + text_before_locator)
//! suffix = tokenize(text_after_locator + SUFFIX)
//! full   = prefix + [A AUDIO ROWS] + suffix
//! ```
//!
//! No BOS is added (the chat template is the prompt) and special-token
//! sequences in the instruction are parsed atomically by the shared BPE
//! `parse_special` path.

use thiserror::Error;

use super::tokenizer::NemotronTranscribeTokenizer;

pub(crate) const NEMOTRON_TRANSCRIBE_CHAT_PREFIX: &str =
    "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n";
pub(crate) const NEMOTRON_TRANSCRIBE_CHAT_SUFFIX: &str =
    "<|im_end|>\n<|im_start|>assistant\n<think></think>";
/// The audio-row locator. Never a vocabulary token; the pack splits at the
/// single occurrence and splices the encoder projection in its place.
pub(crate) const NEMOTRON_TRANSCRIBE_AUDIO_LOCATOR: &str = "<|audio|>";

/// Served default transcription prompt (verbatim; keep the same language).
pub(crate) const NEMOTRON_TRANSCRIBE_TRANSCRIBE_PROMPT: &str = "Produce a verbatim transcript of the audio. Preserve named entities, abbreviations, numbers, dates, measurements, acronyms, and technical terms as clearly as possible. Keep the same language as spoken and do not translate. <|audio|>";
/// Diarize prompt (in-decoder speaker tags). Served only when the request
/// asks for speaker diarization.
pub(crate) const NEMOTRON_TRANSCRIBE_DIARIZE_PROMPT: &str = "Transcribe this audio. Include all spoken words and verbal fillers. Write numbers as spoken. Use no punctuation and no capitalization. Start every speaker turn with a speaker tag such as <spk:0>, <spk:1>, <spk:2>, etc.; assign one stable tag per speaker and keep speaker identities consistent. Output only the speaker-tagged transcript. <|audio|>";
/// Empty-transcript fallback (reference `kFallbackPrompt`, verbatim): when the
/// served transcription prompt decodes to nothing (the model takes its stop
/// token at step 0), the same encoder output is re-decoded once with this
/// prompt, which is tuned to pull speech out of music and background noise.
pub(crate) const NEMOTRON_TRANSCRIBE_FALLBACK_PROMPT: &str = "Transcribe the following. Write all numbers in words. Ignore any music or background sounds: <|audio|>";

pub(crate) const NEMOTRON_TRANSCRIBE_MAX_OUTPUT_TOKENS: usize = 21664;
pub(crate) const NEMOTRON_TRANSCRIBE_MIN_OUTPUT_TOKENS: usize = 24;
pub(crate) const NEMOTRON_TRANSCRIBE_OUTPUT_TOKENS_PER_SECOND: f32 = 6.0;

/// Language tag -> English name for the served language hint
/// (` The speech is in <Name>.` inserted before the locator). 43 entries;
/// unknown or `und`/`auto` tags produce no hint (never an error).
const NEMOTRON_TRANSCRIBE_LANGUAGE_NAMES: &[(&str, &str)] = &[
    ("ar", "Arabic"),
    ("bg", "Bulgarian"),
    ("bn", "Bengali"),
    ("cs", "Czech"),
    ("da", "Danish"),
    ("de", "German"),
    ("el", "Greek"),
    ("en", "English"),
    ("es", "Spanish"),
    ("et", "Estonian"),
    ("fi", "Finnish"),
    ("fr", "French"),
    ("he", "Hebrew"),
    ("hi", "Hindi"),
    ("hr", "Croatian"),
    ("hu", "Hungarian"),
    ("id", "Indonesian"),
    ("it", "Italian"),
    ("ja", "Japanese"),
    ("ko", "Korean"),
    ("lt", "Lithuanian"),
    ("lv", "Latvian"),
    ("mr", "Marathi"),
    ("mt", "Maltese"),
    ("nb", "Norwegian Bokmal"),
    ("nl", "Dutch"),
    ("nn", "Norwegian Nynorsk"),
    ("no", "Norwegian"),
    ("pl", "Polish"),
    ("pt", "Portuguese"),
    ("ro", "Romanian"),
    ("ru", "Russian"),
    ("sk", "Slovak"),
    ("sl", "Slovenian"),
    ("sv", "Swedish"),
    ("ta", "Tamil"),
    ("te", "Telugu"),
    ("th", "Thai"),
    ("tl", "Tagalog"),
    ("tr", "Turkish"),
    ("uk", "Ukrainian"),
    ("vi", "Vietnamese"),
    ("zh", "Chinese"),
];

/// Normalize a BCP-47-ish tag the way the reference does: whitespace-stripped,
/// lowercased, `_` -> `-`, then (max 2 subtags) primary-tag lookup. `und`,
/// `auto` and unknown primaries yield `None` (no hint), never an error.
fn normalize_language_tag(language: &str) -> Option<&'static str> {
    let normalized = language.trim().to_ascii_lowercase().replace('_', "-");
    let mut subtags = normalized.split('-').filter(|s| !s.is_empty());
    let primary = subtags.next()?;
    let _ = subtags.next();
    let base = match primary {
        "fil" => "tl",
        other => other,
    };
    NEMOTRON_TRANSCRIBE_LANGUAGE_NAMES
        .iter()
        .find(|(tag, _)| *tag == base)
        .map(|(_, name)| *name)
}

/// The served-instruction language hint for a served prompt. Returns the
/// sentence (with trailing space) to splice in before the locator, or `None`
/// when the tag is unknown/und/auto (or a custom instruction is used, which
/// the caller handles by not consulting this).
pub(crate) fn language_hint_sentence(language: &str) -> Option<String> {
    normalize_language_tag(language).map(|name| format!("The speech is in {name}."))
}

/// Build the served instruction text with the locator (without the chat
/// wrapper): `language_hint`, when set, is inserted BEFORE the locator with
/// exactly one space around it. The chat wrapper is applied once by
/// [`build_prompt_plan`].
pub(crate) fn build_served_instruction(
    instruction: &str,
    language_hint: Option<&str>,
) -> Result<String, NemotronTranscribePromptError> {
    if instruction
        .matches(NEMOTRON_TRANSCRIBE_AUDIO_LOCATOR)
        .count()
        != 1
    {
        return Err(NemotronTranscribePromptError::Locator);
    }
    let locator_at = instruction
        .find(NEMOTRON_TRANSCRIBE_AUDIO_LOCATOR)
        .expect("checked");
    let before = &instruction[..locator_at];
    let after = &instruction[locator_at..];
    if !before.ends_with(' ') {
        return Err(NemotronTranscribePromptError::Locator);
    }
    let hint = language_hint
        .and_then(language_hint_sentence)
        .map(|hint| format!("{hint} "))
        .unwrap_or_default();
    Ok(format!("{before}{hint}{after}"))
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub(crate) enum NemotronTranscribePromptError {
    #[error(
        "nemotron-transcribe prompt must contain exactly one final <|audio|> locator \
             preceded by exactly one space with a non-empty prefix"
    )]
    Locator,
    #[error("nemotron-transcribe prompt tokenizer failed: {reason}")]
    Tokenize { reason: String },
}

/// The two token spans around the audio rows, ready for the decode prompt:
/// `full = prefix + [audio_rows] + suffix`. The audio rows splice position is
/// `prefix_ids.len()`; the token at `prefix_ids.len() + audio_rows` is the
/// first suffix token and carries the sampling logits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct NemotronTranscribePromptPlan {
    pub prefix_ids: Vec<u32>,
    pub suffix_ids: Vec<u32>,
    /// The served instruction, with the language hint applied (when any).
    pub text: String,
}

impl NemotronTranscribePromptPlan {
    pub(crate) fn prompt_token_ids(&self, audio_rows: usize) -> Vec<u32> {
        let mut full =
            Vec::with_capacity(self.prefix_ids.len() + audio_rows + self.suffix_ids.len());
        full.extend_from_slice(&self.prefix_ids);
        full.extend(std::iter::repeat_n(0, audio_rows));
        full.extend_from_slice(&self.suffix_ids);
        full
    }
}

/// Build the served transcription instruction (verbatim prompt) with an
/// optional language hint. `instruction` is the served prompt text with the
/// locator already in place.
pub(crate) fn build_served_prompt(
    tokenizer: &NemotronTranscribeTokenizer,
    instruction: &str,
    language: Option<&str>,
) -> Result<NemotronTranscribePromptPlan, NemotronTranscribePromptError> {
    let text = build_served_instruction(instruction, language)?;
    build_prompt_plan(tokenizer, &text)
}

/// Splice a raw instruction (custom instruction path): the caller supplies
/// the full instruction text (with locator) and this validates and tokenizes
/// it. No language hint is applied.
pub(crate) fn build_prompt_plan(
    tokenizer: &NemotronTranscribeTokenizer,
    text: &str,
) -> Result<NemotronTranscribePromptPlan, NemotronTranscribePromptError> {
    if text.matches(NEMOTRON_TRANSCRIBE_AUDIO_LOCATOR).count() != 1 {
        return Err(NemotronTranscribePromptError::Locator);
    }
    let locator_at = text
        .find(NEMOTRON_TRANSCRIBE_AUDIO_LOCATOR)
        .expect("checked");
    let before = &text[..locator_at];
    let after = &text[locator_at..];
    if !before.ends_with(' ') || before.trim_end_matches(' ').is_empty() {
        return Err(NemotronTranscribePromptError::Locator);
    }
    let prefix = format!("{NEMOTRON_TRANSCRIBE_CHAT_PREFIX}{before}");
    let suffix = format!("{after}{NEMOTRON_TRANSCRIBE_CHAT_SUFFIX}");
    let prefix_ids = tokenizer.encode_prompt_text(&prefix).map_err(|error| {
        NemotronTranscribePromptError::Tokenize {
            reason: error.to_string(),
        }
    })?;
    let suffix_ids = tokenizer.encode_prompt_text(&suffix).map_err(|error| {
        NemotronTranscribePromptError::Tokenize {
            reason: error.to_string(),
        }
    })?;
    Ok(NemotronTranscribePromptPlan {
        prefix_ids,
        suffix_ids,
        text: text.to_string(),
    })
}

/// Output token budget for a clip of `seconds` (the reference's
/// `max(24, ceil(6.0 * seconds) + 24)`, clamped at the API ceiling).
pub(crate) fn output_budget_for_seconds(seconds: f32) -> usize {
    let budget = (seconds * NEMOTRON_TRANSCRIBE_OUTPUT_TOKENS_PER_SECOND).ceil() + 24.0;
    (budget as usize).clamp(
        NEMOTRON_TRANSCRIBE_MIN_OUTPUT_TOKENS,
        NEMOTRON_TRANSCRIBE_MAX_OUTPUT_TOKENS,
    )
}

/// Near-digital silence gate (peak |x| < 1e-3 AND rms < 3e-5, or empty input):
/// the request resolves to an empty transcript without running the model.
pub(crate) fn is_near_digital_silence(samples: &[f32]) -> bool {
    if samples.is_empty() {
        return true;
    }
    let mut peak = 0.0f32;
    let mut sum_squares = 0.0f64;
    for &sample in samples {
        let magnitude = sample.abs();
        if magnitude > peak {
            peak = magnitude;
        }
        sum_squares += (sample as f64) * (sample as f64);
    }
    let rms = (sum_squares / samples.len() as f64).sqrt();
    peak < 1e-3 && (rms as f32) < 3e-5
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn language_hint_normalizes_and_matches() {
        assert_eq!(
            language_hint_sentence("fr"),
            Some("The speech is in French.".to_string())
        );
        assert_eq!(
            language_hint_sentence("EN"),
            Some("The speech is in English.".to_string())
        );
        assert_eq!(
            language_hint_sentence("en_US"),
            Some("The speech is in English.".to_string())
        );
        assert_eq!(
            language_hint_sentence("pt-BR"),
            Some("The speech is in Portuguese.".to_string())
        );
        assert_eq!(language_hint_sentence("und"), None);
        assert_eq!(language_hint_sentence("auto"), None);
        assert_eq!(language_hint_sentence("xx"), None);
    }

    #[test]
    fn served_prompt_hint_splices_before_locator() {
        let text = build_served_instruction(NEMOTRON_TRANSCRIBE_TRANSCRIBE_PROMPT, Some("fr"))
            .expect("served instruction");
        assert!(text.contains("The speech is in French. <|audio|>"));
        assert!(text.ends_with(NEMOTRON_TRANSCRIBE_AUDIO_LOCATOR));
        // The chat wrapper is applied by build_prompt_plan, not here.
        assert!(!text.contains(NEMOTRON_TRANSCRIBE_CHAT_PREFIX));
        assert!(!text.contains(NEMOTRON_TRANSCRIBE_CHAT_SUFFIX));
    }

    /// The fallback prompt is a reference byte contract (reference
    /// `kFallbackPrompt`): it fires from the same request only when the
    /// served prompt decoded to nothing, so its text must not drift.
    #[test]
    fn fallback_prompt_is_the_reference_prompt_with_a_valid_locator() {
        let text = build_served_instruction(NEMOTRON_TRANSCRIBE_FALLBACK_PROMPT, None)
            .expect("fallback prompt locator layout");
        assert_eq!(
            text,
            "Transcribe the following. Write all numbers in words. \
             Ignore any music or background sounds: <|audio|>"
        );
    }

    #[test]
    fn near_digital_silence_gate_matches_reference() {
        assert!(is_near_digital_silence(&[]));
        // Peak and rms below the digital-silence floor (peak < 1e-3,
        // rms < 3e-5): the gate fires.
        assert!(is_near_digital_silence(&vec![0.00001; 1600]));
        // Quiet but audible: peak is below 1e-3, but the rms of a
        // full-scale 5e-4 tone exceeds the 3e-5 floor, so the gate must
        // NOT fire.
        assert!(!is_near_digital_silence(&vec![0.0005; 1600]));
        assert!(!is_near_digital_silence(&vec![0.5f32; 1600]));
    }

    #[test]
    fn output_budget_scales_and_clamps() {
        assert_eq!(
            output_budget_for_seconds(0.0),
            NEMOTRON_TRANSCRIBE_MIN_OUTPUT_TOKENS
        );
        assert_eq!(output_budget_for_seconds(11.0), 90);
        // An hour at the 6.0 tokens/s serve rate is 21600 + 24 = 21624, which
        // fits under the 21664 API ceiling.
        assert_eq!(output_budget_for_seconds(3600.0), 21624);
        // Above the ceiling the clamp binds (e.g. 4000 s -> 24024 -> ceiling).
        assert_eq!(
            output_budget_for_seconds(4000.0),
            NEMOTRON_TRANSCRIBE_MAX_OUTPUT_TOKENS
        );
    }
}
