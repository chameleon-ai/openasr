//! Nemotron-transcribe decode-step executors: wire the recurrent
//! `decoder_graph::NemotronTranscribeDecoderRuntime` (Mamba-2/MoE/attention
//! trunk with a resident KV span) into the shared greedy-decode driver.
//! This module never picks a token itself; `decode_step_logits` only returns
//! a logits row, and the shared driver owns argmax, suppression, stop-token,
//! and the degenerate-loop guard.
//!
//! Prompt layout: `[prefix tokens][audio rows][suffix tokens]`. The prefix
//! and suffix are text rows gathered from the mmap'd embedding table; the
//! audio rows are the PE encoder's `fusion.proj` output (2688-dim), spliced
//! in at the `<|audio|>` locator position. The token-id sequence passed to
//! the shared driver uses placeholder ids in the audio range (never read:
//! the executor embeds from `initial_audio_embeddings` directly).

use crate::ggml_runtime::GgmlSelectionEvidenceRef;
use crate::models::mapped_token_embedding::{MappedTokenEmbeddingError, MappedTokenEmbeddingTable};
use crate::models::seq2seq_greedy_decode::{
    Seq2SeqGreedyDecodeError, Seq2SeqGreedyDecodeStepExecutor, Seq2SeqGreedyDecodeStepInput,
    Seq2SeqGreedyDecodeStepLogitsOutput,
};

/// Deterministic splitmix64 stream for the temperature ladder's sampled rounds.
struct NemotronLadderRng {
    state: u64,
}

impl NemotronLadderRng {
    fn next_unit_f32(&mut self) -> f32 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^= value >> 31;
        let unit = (value >> 11) as f64 * (1.0f64 / (1u64 << 53) as f64);
        (unit as f32 + 1.0e-6).min(1.0f32 - 1.0e-7)
    }
    fn gumbel(&mut self) -> f32 {
        -(-self.next_unit_f32().ln()).ln()
    }
}

/// Gumbel-max sampling reduces to argmax over `logit / T + Gumbel(0,1)`, so the
/// shared driver's argmax samples from softmax(logits/T) with no driver change.
fn temper_logits(rng: &mut NemotronLadderRng, temperature: f32, logits: Vec<f32>) -> Vec<f32> {
    if temperature <= 0.0 {
        return logits;
    }
    let inv_temperature = 1.0f32 / temperature;
    logits
        .into_iter()
        .map(|logit| logit * inv_temperature + rng.gumbel())
        .collect()
}

use super::decoder_graph::{NemotronTranscribeDecoderError, NemotronTranscribeDecoderRuntime};

fn map_step_error(
    step_index: usize,
    label: &'static str,
) -> impl Fn(NemotronTranscribeDecoderError) -> Seq2SeqGreedyDecodeError {
    move |error| Seq2SeqGreedyDecodeError::DecoderStepFailed {
        reason: format!("nemotron-transcribe {label} decoder step {step_index}: {error}"),
    }
}

fn map_embedding_error(
    step_index: usize,
) -> impl Fn(MappedTokenEmbeddingError) -> Seq2SeqGreedyDecodeError {
    move |error| Seq2SeqGreedyDecodeError::DecoderStepFailed {
        reason: format!(
            "nemotron-transcribe decoder step {step_index} token embedding gather failed: {error}"
        ),
    }
}

/// Resident decode session. Owns the runtime once the executor checks it
/// out; `prefill` embeds the audio-spliced prompt and seeds both the KV
/// span and the mamba recurrent state (zeroed first).
pub(crate) struct NemotronTranscribeDecodeSession {
    runtime: NemotronTranscribeDecoderRuntime,
    last_evidence: Option<GgmlSelectionEvidenceRef>,
}

impl NemotronTranscribeDecodeSession {
    pub(crate) fn new(runtime: NemotronTranscribeDecoderRuntime) -> Self {
        Self {
            runtime,
            last_evidence: None,
        }
    }

    /// Prefill `t` already-materialized embedding rows (2688 floats each,
    /// row-major) and return the last position's logits.
    pub(crate) fn prefill(
        &mut self,
        embeddings: &[f32],
        t: usize,
    ) -> Result<Vec<f32>, NemotronTranscribeDecoderError> {
        let (logits, evidence) = self.runtime.prefill(embeddings, t)?;
        self.last_evidence = evidence;
        Ok(logits)
    }

    pub(crate) fn decode_step(
        &mut self,
        embedding: &[f32],
    ) -> Result<Vec<f32>, NemotronTranscribeDecoderError> {
        let (logits, evidence) = self.runtime.decode_step(embedding)?;
        self.last_evidence = evidence;
        Ok(logits)
    }

    pub(crate) fn take_compute_evidence(&mut self) -> Option<GgmlSelectionEvidenceRef> {
        self.last_evidence.take()
    }

    pub(crate) fn cached_positions(&self) -> usize {
        self.runtime.cached_positions()
    }
}

/// Assert the shared greedy driver calls us in the strict order the
/// incremental session assumes: step 0 with no generated tokens (prefill
/// covers the whole prompt), then exactly one new token per step.
fn incremental_new_token(
    session: &NemotronTranscribeDecodeSession,
    prompt_len: usize,
    input: &Seq2SeqGreedyDecodeStepInput<'_>,
) -> Result<u32, Seq2SeqGreedyDecodeError> {
    let expected_cached = prompt_len + input.generated_tokens.len().saturating_sub(1);
    let new_token = input.generated_tokens.last().copied();
    match new_token {
        Some(token) if session.cached_positions() == expected_cached => Ok(token),
        _ => Err(Seq2SeqGreedyDecodeError::DecoderStepFailed {
            reason: format!(
                "nemotron-transcribe decoder step {} out of incremental order: cached {} positions, \
                 {} generated tokens, prompt {prompt_len}",
                input.step_index,
                session.cached_positions(),
                input.generated_tokens.len(),
            ),
        }),
    }
}

/// Expand the `[prefix][audio rows][suffix]` prompt into embedding rows:
/// text rows gathered from the embedding table, audio rows taken from the
/// PE encoder projection (frame-major, 2688 floats per 80 ms frame).
pub(crate) fn materialize_audio_prompt_embeddings(
    embedding_table: &MappedTokenEmbeddingTable,
    prefix_ids: &[u32],
    audio_embeddings: &[f32],
    suffix_ids: &[u32],
    hidden: usize,
) -> Result<Vec<f32>, Seq2SeqGreedyDecodeError> {
    let mut out = Vec::with_capacity(
        (prefix_ids.len() + audio_embeddings.len() / hidden + suffix_ids.len()) * hidden,
    );
    let prefix = embedding_table
        .gather_rows(prefix_ids)
        .map_err(map_embedding_error(0))?;
    out.extend_from_slice(&prefix);
    out.extend_from_slice(audio_embeddings);
    let suffix = embedding_table
        .gather_rows(suffix_ids)
        .map_err(map_embedding_error(0))?;
    out.extend_from_slice(&suffix);
    Ok(out)
}

/// Keep-quantized audio-prompt step executor. The session's runner, weight
/// context, and resident arenas are already built; this only materializes
/// the per-request audio prompt and drives the decode.
pub(crate) struct NemotronTranscribeResidentAudioDecodeStepExecutor<'s> {
    session: &'s mut NemotronTranscribeDecodeSession,
    embedding_table: &'s MappedTokenEmbeddingTable,
    prefix_ids: Vec<u32>,
    suffix_ids: Vec<u32>,
    initial_audio_embeddings: Vec<f32>,
    hidden: usize,
    prompt_len: usize,
    prefilled: bool,
    temperature: f32,
    rng: NemotronLadderRng,
}

impl<'s> NemotronTranscribeResidentAudioDecodeStepExecutor<'s> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        session: &'s mut NemotronTranscribeDecodeSession,
        embedding_table: &'s MappedTokenEmbeddingTable,
        prefix_ids: Vec<u32>,
        suffix_ids: Vec<u32>,
        initial_audio_embeddings: Vec<f32>,
        hidden: usize,
        prompt_len: usize,
        temperature: f32,
        sample_seed: u64,
    ) -> Self {
        Self {
            session,
            embedding_table,
            prefix_ids,
            suffix_ids,
            initial_audio_embeddings,
            hidden,
            prompt_len,
            prefilled: false,
            temperature,
            rng: NemotronLadderRng { state: sample_seed },
        }
    }
}

impl Seq2SeqGreedyDecodeStepExecutor for NemotronTranscribeResidentAudioDecodeStepExecutor<'_> {
    fn decode_step_logits(
        &mut self,
        input: Seq2SeqGreedyDecodeStepInput<'_>,
    ) -> Result<Seq2SeqGreedyDecodeStepLogitsOutput, Seq2SeqGreedyDecodeError> {
        if self.prefilled {
            let new_token = incremental_new_token(self.session, self.prompt_len, &input)?;
            let embedding = self
                .embedding_table
                .gather_rows(&[new_token])
                .map_err(map_embedding_error(input.step_index))?;
            let logits = self
                .session
                .decode_step(&embedding)
                .map_err(map_step_error(input.step_index, "audio"))?;
            return Ok(Seq2SeqGreedyDecodeStepLogitsOutput {
                logits: temper_logits(&mut self.rng, self.temperature, logits),
                greedy_token_hint: None,
            });
        }

        if input.initial_prompt_tokens.len() != self.prompt_len {
            return Err(Seq2SeqGreedyDecodeError::DecoderStepFailed {
                reason: format!(
                    "nemotron-transcribe initial prompt length changed before prefill: {} != {}",
                    input.initial_prompt_tokens.len(),
                    self.prompt_len
                ),
            });
        }
        let prompt_embeddings = materialize_audio_prompt_embeddings(
            self.embedding_table,
            &self.prefix_ids,
            &self.initial_audio_embeddings,
            &self.suffix_ids,
            self.hidden,
        )?;
        let logits = self
            .session
            .prefill(&prompt_embeddings, input.initial_prompt_tokens.len())
            .map_err(map_step_error(input.step_index, "prefill"))?;
        self.prefilled = true;
        Ok(Seq2SeqGreedyDecodeStepLogitsOutput {
            logits: temper_logits(&mut self.rng, self.temperature, logits),
            greedy_token_hint: None,
        })
    }

    fn take_compute_evidence(&mut self) -> Option<GgmlSelectionEvidenceRef> {
        self.session.take_compute_evidence()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_temperature_leaves_logits_untouched() {
        let mut rng = NemotronLadderRng { state: 7 };
        let logits = vec![1.0, 3.0, -2.0];
        assert_eq!(temper_logits(&mut rng, 0.0, logits.clone()), logits);
    }

    #[test]
    fn tempering_is_deterministic_for_a_seed() {
        let row = vec![1.0, 1.05, 0.4];
        let mut a = NemotronLadderRng { state: 42 };
        let mut b = NemotronLadderRng { state: 42 };
        assert_eq!(
            temper_logits(&mut a, 0.7, row.clone()),
            temper_logits(&mut b, 0.7, row)
        );
    }

    #[test]
    fn a_ladder_seed_can_break_an_argmax_tie() {
        let row = vec![1.0, 1.01];
        let mut moved = false;
        for seed in 0..64u64 {
            let mut rng = NemotronLadderRng { state: seed };
            let tempered = temper_logits(&mut rng, 1.0, row.clone());
            if tempered[0] > tempered[1] {
                moved = true;
            }
        }
        assert!(moved, "no sampled round moved off the greedy token");
    }
}
