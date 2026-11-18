//! Runtime contract for the nemotron-transcribe `.oasr` pack.
//!
//! The Prime directive of this validator is fail-closed admission of the one
//! published pack generation (`nemotron-3.5-transcribe-30b-a4b`, all three
//! quant tiers share the same tensor table and hparams; only the quantized
//! dtypes differ, plus the `quantize.imatrix.*` provenance keys present in
//! the two imatrix-calibrated AD tiers). It checks:
//!
//! - the `openasr.*` routing envelope byte-for-byte,
//! - the decoder (`nemotron_h_moe.*`), encoder (`pee.asr.*`), speaker branch
//!   (`pee.diar.*`), CTC head (`ctc.*`), fusion and preprocessor hparams,
//! - the tensor index: exactly 1303 tensors, with the mandatory scopes and
//!   the small set of load-bearing shapes pinned.
//!
//! It deliberately does NOT claim anything about execution: the family is
//! staged as not-yet-implemented and no graph is built from this contract
//! yet. When the real executor lands, this file evolves into the executor's
//! own typed metadata instead of being loosened.

use thiserror::Error;

use crate::arch::hparams::{
    NEMOTRON_TRANSCRIBE_ARCHITECTURE_ID, NEMOTRON_TRANSCRIBE_ASR_D_MODEL_KEY,
    NEMOTRON_TRANSCRIBE_ASR_FEAT_IN_KEY, NEMOTRON_TRANSCRIBE_ASR_FF_HIDDEN_KEY,
    NEMOTRON_TRANSCRIBE_ASR_N_HEADS_KEY, NEMOTRON_TRANSCRIBE_ASR_N_LAYERS_KEY,
    NEMOTRON_TRANSCRIBE_ASR_ROTARY_DIMS_KEY, NEMOTRON_TRANSCRIBE_ASR_SUBSAMPLING_FACTOR_KEY,
    NEMOTRON_TRANSCRIBE_ATTENTION_HEAD_COUNT_KEY, NEMOTRON_TRANSCRIBE_ATTENTION_KEY_LENGTH_KEY,
    NEMOTRON_TRANSCRIBE_ATTENTION_VALUE_LENGTH_KEY, NEMOTRON_TRANSCRIBE_BLOCK_COUNT_KEY,
    NEMOTRON_TRANSCRIBE_CONTEXT_LENGTH_KEY, NEMOTRON_TRANSCRIBE_CTC_D_MODEL_KEY,
    NEMOTRON_TRANSCRIBE_CTC_N_LAYERS_KEY, NEMOTRON_TRANSCRIBE_CTC_VOCAB_SIZE_KEY,
    NEMOTRON_TRANSCRIBE_DIAR_D_MODEL_KEY, NEMOTRON_TRANSCRIBE_DIAR_N_LAYERS_KEY,
    NEMOTRON_TRANSCRIBE_DIAR_NUM_SPEAKERS_KEY, NEMOTRON_TRANSCRIBE_DIAR_TF_D_MODEL_KEY,
    NEMOTRON_TRANSCRIBE_EMBEDDING_LENGTH_KEY, NEMOTRON_TRANSCRIBE_EXPERT_COUNT_KEY,
    NEMOTRON_TRANSCRIBE_EXPERT_FEED_FORWARD_LENGTH_KEY,
    NEMOTRON_TRANSCRIBE_EXPERT_SHARED_COUNT_KEY,
    NEMOTRON_TRANSCRIBE_EXPERT_SHARED_FEED_FORWARD_LENGTH_KEY,
    NEMOTRON_TRANSCRIBE_EXPERT_USED_COUNT_KEY, NEMOTRON_TRANSCRIBE_FUSION_PROJ_DIM_KEY,
    NEMOTRON_TRANSCRIBE_NEXTN_PREDICT_LAYERS_KEY, NEMOTRON_TRANSCRIBE_PREPROCESSOR_FEATURES_KEY,
    NEMOTRON_TRANSCRIBE_PREPROCESSOR_N_FFT_KEY, NEMOTRON_TRANSCRIBE_PREPROCESSOR_SAMPLE_RATE_KEY,
    NEMOTRON_TRANSCRIBE_ROPE_DIMENSION_COUNT_KEY, NEMOTRON_TRANSCRIBE_SSM_CONV_KERNEL_KEY,
    NEMOTRON_TRANSCRIBE_SSM_GROUP_COUNT_KEY, NEMOTRON_TRANSCRIBE_SSM_INNER_SIZE_KEY,
    NEMOTRON_TRANSCRIBE_SSM_STATE_SIZE_KEY, NEMOTRON_TRANSCRIBE_SSM_TIME_STEP_RANK_KEY,
    NEMOTRON_TRANSCRIBE_VOCAB_SIZE_KEY,
};
use crate::arch::{
    GENERAL_ARCHITECTURE_KEY, NEMOTRON_TRANSCRIBE_AUDIO_FRONTEND_ID,
    NEMOTRON_TRANSCRIBE_DECODE_POLICY_ID, NEMOTRON_TRANSCRIBE_GGML_ARCHITECTURE_ID,
    NEMOTRON_TRANSCRIBE_MODEL_FAMILY, NEMOTRON_TRANSCRIBE_TOKENIZER_ID,
};
use crate::models::ggml_family_adapter::GGML_TOKENIZER_ID_KEY;
use crate::models::oasr_metadata::{
    OASR_METADATA_KEY_AUDIO_FRONTEND, OASR_METADATA_KEY_DECODE_POLICY,
    OASR_METADATA_KEY_MODEL_ARCHITECTURE, OASR_METADATA_KEY_MODEL_FAMILY,
    OASR_METADATA_KEY_PACKAGE_VERSION, OASR_PACKAGE_VERSION_V1,
};
use crate::models::runtime_contract::{
    MetadataContractError, ScalarMetadataView, required_string_scalar, required_u64_scalar,
};

/// No shared constant exists yet for this key (the pack merge script is the
/// only writer); define it once here rather than drifting a second literal.
const OASR_METADATA_KEY_SOURCE_REVISION: &str = "openasr.source.revision";

/// Metadata-only parse of one pack's identity and hparams. Everything here is
/// pinned to the published `nemotron-3.5-transcribe-30b-a4b` generation; a
/// pack failing any check is refused, not re-interpreted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct NemotronTranscribeExecutionMetadata {
    pub decoder_blocks: usize,
    pub decoder_embedding_length: usize,
    pub decoder_vocab_size: usize,
    pub decoder_context_length: usize,
    pub decoder_expert_count: usize,
    pub decoder_expert_used_count: usize,
    pub nextn_predict_layers: usize,
    pub encoder_layers: usize,
    pub encoder_d_model: usize,
    pub encoder_n_heads: usize,
    pub encoder_feat_in: usize,
    pub encoder_ff_hidden: usize,
    pub diar_layers: usize,
    pub diar_d_model: usize,
    pub diar_num_speakers: usize,
    pub diar_tf_d_model: usize,
    pub fusion_proj_dim: usize,
    pub ctc_vocab_size: usize,
    pub ctc_d_model: usize,
    pub ctc_layers: usize,
    pub preprocessor_features: usize,
    pub preprocessor_sample_rate: u32,
    pub preprocessor_n_fft: usize,
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub(crate) enum NemotronTranscribeRuntimeContractError {
    #[error("nemotron-transcribe missing required GGUF metadata key '{key}'")]
    MissingRequiredMetadata { key: &'static str },
    #[error("nemotron-transcribe GGUF metadata '{key}' is invalid: {reason}")]
    InvalidMetadataValue { key: &'static str, reason: String },
    #[error("nemotron-transcribe expected general.architecture='{expected}', got '{found}'")]
    UnexpectedArchitecture {
        expected: &'static str,
        found: String,
    },
    #[error("nemotron-transcribe expected '{key}' = '{expected}', got '{found}'")]
    UnexpectedEnvelopeValue {
        key: &'static str,
        expected: &'static str,
        found: String,
    },
    #[error("nemotron-transcribe missing required GGUF tensor '{name}'")]
    MissingRequiredTensor { name: &'static str },
    #[error("nemotron-transcribe GGUF tensor '{name}' has invalid shape {shape}: {reason}")]
    InvalidTensorShape {
        name: &'static str,
        shape: String,
        reason: String,
    },
    #[error("nemotron-transcribe pack is expected to carry exactly 1303 tensors, got {found}")]
    UnexpectedTensorCount { found: usize },
}

fn map_metadata_contract_error(
    error: MetadataContractError,
) -> NemotronTranscribeRuntimeContractError {
    match error {
        MetadataContractError::MissingRequiredKey { key } => {
            NemotronTranscribeRuntimeContractError::MissingRequiredMetadata { key }
        }
        MetadataContractError::InvalidValue { key, reason } => {
            NemotronTranscribeRuntimeContractError::InvalidMetadataValue { key, reason }
        }
    }
}

fn required_u64<M: ScalarMetadataView>(
    metadata: &M,
    key: &'static str,
) -> Result<u64, NemotronTranscribeRuntimeContractError> {
    required_u64_scalar(metadata, key).map_err(map_metadata_contract_error)
}

fn required_usize<M: ScalarMetadataView>(
    metadata: &M,
    key: &'static str,
) -> Result<usize, NemotronTranscribeRuntimeContractError> {
    let value = required_u64(metadata, key)?;
    usize::try_from(value).map_err(|_| {
        NemotronTranscribeRuntimeContractError::InvalidMetadataValue {
            key,
            reason: format!("value {value} does not fit usize"),
        }
    })
}

fn required_u32<M: ScalarMetadataView>(
    metadata: &M,
    key: &'static str,
) -> Result<u32, NemotronTranscribeRuntimeContractError> {
    let value = required_u64(metadata, key)?;
    u32::try_from(value).map_err(
        |_| NemotronTranscribeRuntimeContractError::InvalidMetadataValue {
            key,
            reason: format!("value {value} does not fit u32"),
        },
    )
}

fn required_exact_string<M: ScalarMetadataView>(
    metadata: &M,
    key: &'static str,
    expected: &'static str,
) -> Result<(), NemotronTranscribeRuntimeContractError> {
    let found = required_string_scalar(metadata, key).map_err(map_metadata_contract_error)?;
    if found != expected {
        return Err(
            NemotronTranscribeRuntimeContractError::UnexpectedEnvelopeValue {
                key,
                expected,
                found: found.to_string(),
            },
        );
    }
    Ok(())
}

pub(crate) fn parse_nemotron_transcribe_execution_metadata<M: ScalarMetadataView>(
    metadata: &M,
) -> Result<NemotronTranscribeExecutionMetadata, NemotronTranscribeRuntimeContractError> {
    let architecture = required_string_scalar(metadata, GENERAL_ARCHITECTURE_KEY)
        .map_err(map_metadata_contract_error)?;
    if architecture != NEMOTRON_TRANSCRIBE_ARCHITECTURE_ID {
        return Err(
            NemotronTranscribeRuntimeContractError::UnexpectedArchitecture {
                expected: NEMOTRON_TRANSCRIBE_ARCHITECTURE_ID,
                found: architecture.to_string(),
            },
        );
    }
    required_exact_string(
        metadata,
        OASR_METADATA_KEY_PACKAGE_VERSION,
        OASR_PACKAGE_VERSION_V1,
    )?;
    required_exact_string(
        metadata,
        OASR_METADATA_KEY_MODEL_FAMILY,
        NEMOTRON_TRANSCRIBE_MODEL_FAMILY,
    )?;
    required_exact_string(
        metadata,
        OASR_METADATA_KEY_MODEL_ARCHITECTURE,
        NEMOTRON_TRANSCRIBE_GGML_ARCHITECTURE_ID,
    )?;
    required_exact_string(
        metadata,
        OASR_METADATA_KEY_AUDIO_FRONTEND,
        NEMOTRON_TRANSCRIBE_AUDIO_FRONTEND_ID,
    )?;
    required_exact_string(
        metadata,
        OASR_METADATA_KEY_DECODE_POLICY,
        NEMOTRON_TRANSCRIBE_DECODE_POLICY_ID,
    )?;
    required_exact_string(
        metadata,
        GGML_TOKENIZER_ID_KEY,
        NEMOTRON_TRANSCRIBE_TOKENIZER_ID,
    )?;
    let source_revision = required_string_scalar(metadata, OASR_METADATA_KEY_SOURCE_REVISION)
        .map_err(map_metadata_contract_error)?;
    let revision_hex = source_revision.bytes().all(|byte| byte.is_ascii_hexdigit());
    if !matches!(source_revision.len(), 32 | 40) || !revision_hex {
        return Err(
            NemotronTranscribeRuntimeContractError::InvalidMetadataValue {
                key: OASR_METADATA_KEY_SOURCE_REVISION,
                reason: format!(
                    "expected a 32- or 40-hex upstream checkpoint id, got '{source_revision}'"
                ),
            },
        );
    }

    let decoder_blocks = required_usize(metadata, NEMOTRON_TRANSCRIBE_BLOCK_COUNT_KEY)?;
    if decoder_blocks != 53 {
        return Err(
            NemotronTranscribeRuntimeContractError::InvalidMetadataValue {
                key: NEMOTRON_TRANSCRIBE_BLOCK_COUNT_KEY,
                reason: format!("expected the published 53-block hybrid, got {decoder_blocks}"),
            },
        );
    }
    let decoder_embedding_length =
        required_usize(metadata, NEMOTRON_TRANSCRIBE_EMBEDDING_LENGTH_KEY)?;
    let decoder_vocab_size = required_usize(metadata, NEMOTRON_TRANSCRIBE_VOCAB_SIZE_KEY)?;
    let decoder_context_length = required_usize(metadata, NEMOTRON_TRANSCRIBE_CONTEXT_LENGTH_KEY)?;
    let decoder_expert_count = required_usize(metadata, NEMOTRON_TRANSCRIBE_EXPERT_COUNT_KEY)?;
    let decoder_expert_used_count =
        required_usize(metadata, NEMOTRON_TRANSCRIBE_EXPERT_USED_COUNT_KEY)?;
    let _ = required_usize(metadata, NEMOTRON_TRANSCRIBE_EXPERT_FEED_FORWARD_LENGTH_KEY)?;
    let _ = required_usize(metadata, NEMOTRON_TRANSCRIBE_EXPERT_SHARED_COUNT_KEY)?;
    let _ = required_usize(
        metadata,
        NEMOTRON_TRANSCRIBE_EXPERT_SHARED_FEED_FORWARD_LENGTH_KEY,
    )?;
    let nextn_predict_layers =
        required_usize(metadata, NEMOTRON_TRANSCRIBE_NEXTN_PREDICT_LAYERS_KEY)?;
    // Decoder geometry that the executor will need once it lands; parsed now
    // so a malformed pack fails closed instead of at first inference.
    let _ = required_usize(metadata, NEMOTRON_TRANSCRIBE_ROPE_DIMENSION_COUNT_KEY)?;
    let _ = required_usize(metadata, NEMOTRON_TRANSCRIBE_SSM_CONV_KERNEL_KEY)?;
    let _ = required_usize(metadata, NEMOTRON_TRANSCRIBE_SSM_GROUP_COUNT_KEY)?;
    let _ = required_usize(metadata, NEMOTRON_TRANSCRIBE_SSM_INNER_SIZE_KEY)?;
    let _ = required_usize(metadata, NEMOTRON_TRANSCRIBE_SSM_STATE_SIZE_KEY)?;
    let _ = required_usize(metadata, NEMOTRON_TRANSCRIBE_SSM_TIME_STEP_RANK_KEY)?;
    let _ = required_usize(metadata, NEMOTRON_TRANSCRIBE_ATTENTION_HEAD_COUNT_KEY)?;
    let _ = required_usize(metadata, NEMOTRON_TRANSCRIBE_ATTENTION_KEY_LENGTH_KEY)?;
    let _ = required_usize(metadata, NEMOTRON_TRANSCRIBE_ATTENTION_VALUE_LENGTH_KEY)?;

    let encoder_layers = required_usize(metadata, NEMOTRON_TRANSCRIBE_ASR_N_LAYERS_KEY)?;
    let encoder_d_model = required_usize(metadata, NEMOTRON_TRANSCRIBE_ASR_D_MODEL_KEY)?;
    let encoder_n_heads = required_usize(metadata, NEMOTRON_TRANSCRIBE_ASR_N_HEADS_KEY)?;
    let encoder_feat_in = required_usize(metadata, NEMOTRON_TRANSCRIBE_ASR_FEAT_IN_KEY)?;
    let encoder_ff_hidden = required_usize(metadata, NEMOTRON_TRANSCRIBE_ASR_FF_HIDDEN_KEY)?;
    let _ = required_usize(metadata, NEMOTRON_TRANSCRIBE_ASR_ROTARY_DIMS_KEY)?;
    let _ = required_usize(metadata, NEMOTRON_TRANSCRIBE_ASR_SUBSAMPLING_FACTOR_KEY)?;

    let diar_layers = required_usize(metadata, NEMOTRON_TRANSCRIBE_DIAR_N_LAYERS_KEY)?;
    let diar_d_model = required_usize(metadata, NEMOTRON_TRANSCRIBE_DIAR_D_MODEL_KEY)?;
    let diar_num_speakers = required_usize(metadata, NEMOTRON_TRANSCRIBE_DIAR_NUM_SPEAKERS_KEY)?;
    let diar_tf_d_model = required_usize(metadata, NEMOTRON_TRANSCRIBE_DIAR_TF_D_MODEL_KEY)?;

    let fusion_proj_dim = required_usize(metadata, NEMOTRON_TRANSCRIBE_FUSION_PROJ_DIM_KEY)?;

    let ctc_vocab_size = required_usize(metadata, NEMOTRON_TRANSCRIBE_CTC_VOCAB_SIZE_KEY)?;
    let ctc_d_model = required_usize(metadata, NEMOTRON_TRANSCRIBE_CTC_D_MODEL_KEY)?;
    let ctc_layers = required_usize(metadata, NEMOTRON_TRANSCRIBE_CTC_N_LAYERS_KEY)?;

    let preprocessor_features =
        required_usize(metadata, NEMOTRON_TRANSCRIBE_PREPROCESSOR_FEATURES_KEY)?;
    let preprocessor_sample_rate =
        required_u32(metadata, NEMOTRON_TRANSCRIBE_PREPROCESSOR_SAMPLE_RATE_KEY)?;
    let preprocessor_n_fft = required_usize(metadata, NEMOTRON_TRANSCRIBE_PREPROCESSOR_N_FFT_KEY)?;

    if decoder_embedding_length != 2688
        || decoder_vocab_size != 131072
        || decoder_context_length != 1_048_576
        || decoder_expert_count != 128
        || decoder_expert_used_count != 6
        || nextn_predict_layers != 1
        || encoder_layers != 32
        || encoder_d_model != 1280
        || encoder_n_heads != 10
        || encoder_feat_in != 128
        || encoder_ff_hidden != 5120
        || diar_layers != 31
        || diar_d_model != 512
        || diar_num_speakers != 8
        || diar_tf_d_model != 192
        || fusion_proj_dim != 2688
        || ctc_vocab_size != 23863
        || ctc_d_model != 1280
        || ctc_layers != 2
        || preprocessor_features != 128
        || preprocessor_sample_rate != 16000
        || preprocessor_n_fft != 512
    {
        return Err(
            NemotronTranscribeRuntimeContractError::InvalidMetadataValue {
                key: "nemotron-transcribe.published-generation",
                reason: "pack hparams do not match the published nemotron-3.5-transcribe-30b-a4b generation"
                    .to_string(),
            },
        );
    }

    Ok(NemotronTranscribeExecutionMetadata {
        decoder_blocks,
        decoder_embedding_length,
        decoder_vocab_size,
        decoder_context_length,
        decoder_expert_count,
        decoder_expert_used_count,
        nextn_predict_layers,
        encoder_layers,
        encoder_d_model,
        encoder_n_heads,
        encoder_feat_in,
        encoder_ff_hidden,
        diar_layers,
        diar_d_model,
        diar_num_speakers,
        diar_tf_d_model,
        fusion_proj_dim,
        ctc_vocab_size,
        ctc_d_model,
        ctc_layers,
        preprocessor_features,
        preprocessor_sample_rate,
        preprocessor_n_fft,
    })
}

const PINNED_TENSOR_COUNT: usize = 1303;
/// (tensor name, exact dims as stored, must-match) for the load-bearing
/// tensors. Every published tier shares this table; only dtypes differ.
const PINNED_TENSORS: &[(&str, &[u64])] = &[
    ("token_embd.weight", &[2688, 131072]),
    ("output.weight", &[2688, 131072]),
    ("output_norm.weight", &[2688]),
    ("preprocessor.fb", &[257, 128]),
    ("asr.pre_encode.proj.weight", &[1024, 1280]),
    ("diar.pre_encode.proj.weight", &[1024, 512]),
    ("fusion.proj.weight", &[1280, 2688]),
    ("fusion.diar_kernel", &[1280, 8]),
    ("ctc.classifier.weight", &[1280, 23863]),
];
const PINNED_LAYER_TENSORS: &[&str] = &[
    "asr.layers.0.attn.w_qkv.weight",
    "asr.layers.31.attn.w_qkv.weight",
    "diar.layers.0.attn.w_qkv.weight",
    "diar.layers.30.attn.w_qkv.weight",
    "blk.0.ssm_in.weight",
    "blk.52.nextn.eh_proj.weight",
];

fn validate_nemotron_transcribe_tensors(
    index: &crate::GgufTensorIndex,
) -> Result<(), NemotronTranscribeRuntimeContractError> {
    if index.tensors().len() != PINNED_TENSOR_COUNT {
        return Err(
            NemotronTranscribeRuntimeContractError::UnexpectedTensorCount {
                found: index.tensors().len(),
            },
        );
    }
    for (name, expected) in PINNED_TENSORS {
        let Some(tensor) = index.get(name) else {
            return Err(NemotronTranscribeRuntimeContractError::MissingRequiredTensor { name });
        };
        if tensor.dims.as_slice() != *expected {
            return Err(NemotronTranscribeRuntimeContractError::InvalidTensorShape {
                name,
                shape: render_shape(&tensor.dims),
                reason: format!("expected {expected:?}"),
            });
        }
    }
    for name in PINNED_LAYER_TENSORS {
        if index.get(name).is_none() {
            return Err(NemotronTranscribeRuntimeContractError::MissingRequiredTensor { name });
        }
    }
    Ok(())
}

fn render_shape(dims: &[u64]) -> String {
    dims.iter()
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join("x")
}

/// `PackVerifier` contract hook: metadata + tensor index validation for the
/// staged family. Fails closed; carries no execution capability.
pub(crate) fn validate_runtime_pack_contract(
    preflight: &crate::GgufRuntimeSourcePreflight,
) -> Result<(), String> {
    let metadata = parse_nemotron_transcribe_execution_metadata(preflight.metadata())
        .map_err(|error| error.to_string())?;
    let _ = metadata;
    validate_nemotron_transcribe_tensors(preflight.tensor_index())
        .map_err(|error| error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn base_metadata() -> BTreeMap<String, String> {
        [
            (GENERAL_ARCHITECTURE_KEY, "nemotron_h_moe"),
            (OASR_METADATA_KEY_PACKAGE_VERSION, "1"),
            (OASR_METADATA_KEY_MODEL_FAMILY, "nemotron-transcribe"),
            (
                OASR_METADATA_KEY_MODEL_ARCHITECTURE,
                "nemotron-transcribe-hybrid-moe",
            ),
            (
                OASR_METADATA_KEY_AUDIO_FRONTEND,
                "nemotron-transcribe.mel128.16khz.mono.v0",
            ),
            (
                OASR_METADATA_KEY_DECODE_POLICY,
                "nemotron-transcribe.greedy.seq2seq.v1",
            ),
            (GGML_TOKENIZER_ID_KEY, "nemotron-transcribe.bpe-131k.v1"),
            (
                OASR_METADATA_KEY_SOURCE_REVISION,
                "0b5820fad1103208f65895b586bd65d2",
            ),
            ("nemotron_h_moe.block_count", "53"),
            ("nemotron_h_moe.embedding_length", "2688"),
            ("nemotron_h_moe.vocab_size", "131072"),
            ("nemotron_h_moe.context_length", "1048576"),
            ("nemotron_h_moe.expert_count", "128"),
            ("nemotron_h_moe.expert_used_count", "6"),
            ("nemotron_h_moe.expert_feed_forward_length", "1856"),
            ("nemotron_h_moe.expert_shared_count", "1"),
            ("nemotron_h_moe.expert_shared_feed_forward_length", "3712"),
            ("nemotron_h_moe.nextn_predict_layers", "1"),
            ("nemotron_h_moe.rope.dimension_count", "84"),
            ("nemotron_h_moe.ssm.conv_kernel", "4"),
            ("nemotron_h_moe.ssm.group_count", "8"),
            ("nemotron_h_moe.ssm.inner_size", "4096"),
            ("nemotron_h_moe.ssm.state_size", "128"),
            ("nemotron_h_moe.ssm.time_step_rank", "64"),
            ("nemotron_h_moe.attention.head_count", "32"),
            ("nemotron_h_moe.attention.key_length", "128"),
            ("nemotron_h_moe.attention.value_length", "128"),
            ("pee.asr.n_layers", "32"),
            ("pee.asr.d_model", "1280"),
            ("pee.asr.n_heads", "10"),
            ("pee.asr.feat_in", "128"),
            ("pee.asr.ff_hidden", "5120"),
            ("pee.asr.rotary_dims", "64"),
            ("pee.asr.subsampling_factor", "8"),
            ("pee.diar.n_layers", "31"),
            ("pee.diar.d_model", "512"),
            ("pee.diar.num_speakers", "8"),
            ("pee.diar.tf_d_model", "192"),
            ("pee.fusion.proj_dim", "2688"),
            ("ctc.vocab_size", "23863"),
            ("ctc.d_model", "1280"),
            ("ctc.n_layers", "2"),
            ("pee.preprocessor.features", "128"),
            ("pee.preprocessor.sample_rate", "16000"),
            ("pee.preprocessor.n_fft", "512"),
        ]
        .into_iter()
        .map(|(key, value)| (key.to_string(), value.to_string()))
        .collect()
    }

    #[test]
    fn accepts_published_generation_metadata() {
        let parsed = parse_nemotron_transcribe_execution_metadata(&base_metadata())
            .expect("published generation must parse");
        assert_eq!(parsed.decoder_blocks, 53);
        assert_eq!(parsed.encoder_layers, 32);
        assert_eq!(parsed.ctc_vocab_size, 23863);
    }

    #[test]
    fn rejects_wrong_family_envelope() {
        let mut metadata = base_metadata();
        metadata.insert(
            OASR_METADATA_KEY_MODEL_FAMILY.to_string(),
            "whisper".to_string(),
        );
        assert!(matches!(
            parse_nemotron_transcribe_execution_metadata(&metadata),
            Err(NemotronTranscribeRuntimeContractError::UnexpectedEnvelopeValue { .. })
        ));
    }

    #[test]
    fn rejects_wrong_architecture() {
        let mut metadata = base_metadata();
        metadata.insert(GENERAL_ARCHITECTURE_KEY.to_string(), "pee".to_string());
        assert!(matches!(
            parse_nemotron_transcribe_execution_metadata(&metadata),
            Err(NemotronTranscribeRuntimeContractError::UnexpectedArchitecture { .. })
        ));
    }

    #[test]
    fn rejects_tampered_hparams() {
        let mut metadata = base_metadata();
        metadata.insert("nemotron_h_moe.block_count".to_string(), "54".to_string());
        assert!(matches!(
            parse_nemotron_transcribe_execution_metadata(&metadata),
            Err(NemotronTranscribeRuntimeContractError::InvalidMetadataValue { .. })
        ));
    }
}
