//! Persistent decoder-state topology for the nemotron-transcribe hybrid.
//!
//! The only token-scaled persistent state is the attention blocks' KV
//! (7 blocks: 5, 12, 19, 26, 33, 42, plus the blk.52 MTP head which plain
//! greedy decode never enters). The 23 Mamba-2 blocks carry fixed-size
//! conv (3 x 6144 f32) and SSM (128 x 4096 f32) state that does not scale
//! with sequence length, so the shared token-scaled planner does not model
//! it; the executor quotes those bytes in the prepared runtime instead.

use crate::capacity::topology::{
    DecoderStateDemandScope, DecoderStateTopology, InvocationEnvelope, InvocationShapeInput,
    PositionBoundProof, StateBytes, StateDemand, StateKind, TopologyError,
    causal_prefix_positions_with_context_cap,
};

use super::prompt::output_budget_for_seconds;
use super::runtime_contract::parse_nemotron_transcribe_execution_metadata;
use crate::models::ggml_asr_executor::{
    GgmlAsrDecoderStatePlanningError, GgmlAsrDecoderStatePlanningInput,
};
use crate::nn::decoder::LlmKvCacheSpec;

const SELF_KV_STATE_ID: &str = "nemotron-transcribe.decoder.self_kv";

pub(crate) const NEMOTRON_TRANSCRIBE_SELF_KV_STATE_ID: &str = SELF_KV_STATE_ID;

/// Number of attention blocks in the trunk (5, 12, 19, 26, 33, 42) plus the
/// blk.52 MTP head, which plain greedy decode never executes.
pub(crate) const NEMOTRON_TRANSCRIBE_TRUNK_ATTENTION_LAYERS: usize = 6;

/// `kv_heads` and `head_dim` for the attention blocks (GQA 32:2).
pub(crate) const NEMOTRON_TRANSCRIBE_ATTENTION_KV_HEADS: usize = 2;
pub(crate) const NEMOTRON_TRANSCRIBE_ATTENTION_HEAD_DIM: usize = 128;

/// Engine-side sequence cap (~one hour at the 80 ms frame grid). The raw
/// GGUF `context_length` is the HF value; the runtime caps at this.
pub(crate) const NEMOTRON_TRANSCRIBE_ENGINE_MAX_CONTEXT: usize = 73_728;

pub(crate) const NEMOTRON_TRANSCRIBE_DECODER_STATE_STREAMS:
    &[crate::models::ggml_asr_executor::GgmlAsrDecoderStateStreamContract] = &[
    crate::models::ggml_asr_executor::GgmlAsrDecoderStateStreamContract::new(
        SELF_KV_STATE_ID,
        StateKind::SelfAttentionKv,
    ),
];

fn resident_kv_bytes(positions: usize, sequences: usize) -> Result<StateBytes, TopologyError> {
    let geometry = crate::capacity::KvGeometry {
        n_layers: NEMOTRON_TRANSCRIBE_TRUNK_ATTENTION_LAYERS,
        kv_heads: NEMOTRON_TRANSCRIBE_ATTENTION_KV_HEADS,
        head_dim: NEMOTRON_TRANSCRIBE_ATTENTION_HEAD_DIM,
    };
    crate::capacity::topology::StateDemand::from_llm_kv_geometry(
        SELF_KV_STATE_ID,
        StateKind::SelfAttentionKv,
        positions,
        NEMOTRON_TRANSCRIBE_ENGINE_MAX_CONTEXT,
        geometry,
        LlmKvCacheSpec::DEFAULT,
        sequences,
        PositionBoundProof::Exact,
    )
    .map(|demand| demand.bytes)
}

pub(crate) fn packed_prompt_position_estimate(
    sample_rate_hz: u32,
    samples: usize,
    prompt_prefix_positions: usize,
) -> Result<usize, TopologyError> {
    let seconds = samples as f32 / sample_rate_hz as f32;
    if !seconds.is_finite() {
        return Err(TopologyError::ArithmeticOverflow {
            operation: "nemotron audio seconds",
        });
    }
    let audio_rows = (seconds * 12.5).ceil().max(1.0) as usize;
    prompt_prefix_positions
        .checked_add(audio_rows)
        .ok_or(TopologyError::ArithmeticOverflow {
            operation: "nemotron prompt positions",
        })
}

pub(crate) fn plan_nemotron_transcribe_decoder_state(
    input: &GgmlAsrDecoderStatePlanningInput<'_>,
) -> Result<crate::capacity::topology::DecoderStatePlan, GgmlAsrDecoderStatePlanningError> {
    let family = "nemotron-transcribe";
    let metadata = parse_nemotron_transcribe_execution_metadata(input.preflight.metadata.as_ref())
        .map_err(
            |error| GgmlAsrDecoderStatePlanningError::MetadataUnavailable {
                family,
                reason: error.to_string(),
            },
        )?;
    let tokenizer = super::tokenizer::NemotronTranscribeTokenizer::from_gguf_metadata(
        input.preflight.metadata.as_ref(),
    )
    .map_err(
        |error| GgmlAsrDecoderStatePlanningError::PromptTokenCountUnavailable {
            family,
            reason: error.to_string(),
        },
    )?;
    let prompt = super::prompt::build_served_prompt(
        &tokenizer,
        super::prompt::NEMOTRON_TRANSCRIBE_TRANSCRIBE_PROMPT,
        input.request_options.language.as_deref(),
    )
    .map_err(
        |error| GgmlAsrDecoderStatePlanningError::PromptTokenCountUnavailable {
            family,
            reason: error.to_string(),
        },
    )?;
    plan_nemotron_transcribe_decoder_state_with_components(
        input,
        metadata,
        prompt.prefix_ids.len(),
        prompt.suffix_ids.len(),
    )
}

fn plan_nemotron_transcribe_decoder_state_with_components(
    input: &GgmlAsrDecoderStatePlanningInput<'_>,
    metadata: super::runtime_contract::NemotronTranscribeExecutionMetadata,
    prefix_positions: usize,
    _suffix_positions: usize,
) -> Result<crate::capacity::topology::DecoderStatePlan, GgmlAsrDecoderStatePlanningError> {
    let family = "nemotron-transcribe";
    let _ = metadata;
    let seconds =
        input.invocation.samples() as f32 / input.invocation.sample_rate_hz().get() as f32;
    let budget = output_budget_for_seconds(seconds);
    let prompt_positions = packed_prompt_position_estimate(
        input.invocation.sample_rate_hz().get(),
        input.invocation.samples(),
        prefix_positions,
    )
    .map_err(|source| GgmlAsrDecoderStatePlanningError::Topology { family, source })?;
    let topology = NemotronTranscribeDecoderStateTopology::new(prompt_positions, budget);
    crate::capacity::topology::DecoderStatePlan::build(&topology, input.invocation, input.envelope)
        .map_err(|source| GgmlAsrDecoderStatePlanningError::Topology { family, source })
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct NemotronTranscribeDecoderStateTopology {
    logical_prompt_positions: usize,
    max_generated_positions: usize,
}

impl NemotronTranscribeDecoderStateTopology {
    pub(crate) const fn new(
        logical_prompt_positions: usize,
        max_generated_positions: usize,
    ) -> Self {
        Self {
            logical_prompt_positions,
            max_generated_positions,
        }
    }
}

impl DecoderStateTopology for NemotronTranscribeDecoderStateTopology {
    fn demands(
        &self,
        scope: DecoderStateDemandScope<InvocationShapeInput, InvocationEnvelope>,
    ) -> Result<Vec<StateDemand>, TopologyError> {
        let invocation = match scope {
            DecoderStateDemandScope::ExactInvocation(invocation) => invocation,
            DecoderStateDemandScope::StableEnvelope(envelope) => envelope.maximum_invocation(),
        };
        let sequences = invocation.sequences().get() as usize;
        let positions = causal_prefix_positions_with_context_cap(
            SELF_KV_STATE_ID,
            self.logical_prompt_positions,
            self.max_generated_positions,
            NEMOTRON_TRANSCRIBE_ENGINE_MAX_CONTEXT,
        )?;
        Ok(vec![StateDemand::new(
            SELF_KV_STATE_ID,
            StateKind::SelfAttentionKv,
            positions,
            NEMOTRON_TRANSCRIBE_ENGINE_MAX_CONTEXT,
            resident_kv_bytes(positions, sequences)?,
            PositionBoundProof::Conservative {
                basis: "prompt rows are the frontend's integer frame ceiling plus the served \
                        prompt token span; generation budget is the served 6 tokens/s rule",
            },
        )?])
    }
}
