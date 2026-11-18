//! Pack-side tensor-quantization policy for the nemotron-transcribe family.
//!
//! The published packs were produced externally (`llama-quantize` on the
//! converter's BF16 output plus the decoder/sidecar merge script in this
//! repo), so no Rust importer exists yet. This module carries the shared
//! `SemanticRolesV1` classifier the install-time Q8 floor and the
//! post-build quant audit consult, keyed on the merged pack's tensor names:
//!
//! - `asr.*` / `diar.*` / `ctc.*` / `fusion.*` / `preprocessor.*` are the PE
//!   encoder sidecar contents, always F16/F32 in every published pack; the
//!   Q8 floor trivially holds for them.
//! - `blk.*` are the decoder's Mamba-2/Transformer/MoE blocks: matrices, so
//!   they carry the decoder rung (Q8_0 / IQ4_NL / Q2_0 depending on tier).
//! - `token_embd.*` / `output*.weight` are the embedding and logits
//!   projection, always consumed through `get_rows`/f32-sensitive paths.
//! - everything else (norm vectors, biases, filterbanks, 1-D tables) is
//!   `NonQuantizable`.

use crate::arch::NEMOTRON_TRANSCRIBE_GGML_ARCHITECTURE_ID;
use crate::models::pack_quant::{QuantizedAxis, TensorQuantizationContract, TensorRole};

pub(crate) const TENSOR_QUANTIZATION_CONTRACT: TensorQuantizationContract =
    TensorQuantizationContract::SemanticRolesV1 {
        model_architecture: NEMOTRON_TRANSCRIBE_GGML_ARCHITECTURE_ID,
        classify: classify_nemotron_transcribe_tensor_role,
        quantized_axis: QuantizedAxis::First,
    };

fn classify_nemotron_transcribe_tensor_role(name: &str) -> TensorRole {
    const ENCODER_PREFIXES: &[&str] = &["asr.", "diar.", "ctc.", "fusion.", "preprocessor."];
    if name.starts_with("token_embd.") {
        // Embedding tables are the only rank-2 get_rows consumer here.
        return TensorRole::EmbeddingTable;
    }
    if !name.ends_with(".weight") {
        return TensorRole::NonQuantizable;
    }
    if name.starts_with("output_norm.") {
        return TensorRole::NonQuantizable;
    }
    if name == "output.weight" {
        return TensorRole::OutputProjection;
    }
    if ENCODER_PREFIXES
        .iter()
        .any(|prefix| name.starts_with(prefix))
    {
        return TensorRole::AcousticEncoderMatrix;
    }
    if name.starts_with("blk.") {
        return TensorRole::TextDecoderMatrix;
    }
    TensorRole::NonQuantizable
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_encoder_and_decoder_scopes() {
        assert_eq!(
            classify_nemotron_transcribe_tensor_role("asr.layers.0.attn.w_qkv.weight"),
            TensorRole::AcousticEncoderMatrix
        );
        assert_eq!(
            classify_nemotron_transcribe_tensor_role("diar.layers.30.ffn.net.0.weight"),
            TensorRole::AcousticEncoderMatrix
        );
        assert_eq!(
            classify_nemotron_transcribe_tensor_role(
                "ctc.transformer.layers.0.attn.qkv_proj.weight"
            ),
            TensorRole::AcousticEncoderMatrix
        );
        assert_eq!(
            classify_nemotron_transcribe_tensor_role("preprocessor.fb"),
            TensorRole::NonQuantizable
        );
        assert_eq!(
            classify_nemotron_transcribe_tensor_role("blk.52.attn_q.weight"),
            TensorRole::TextDecoderMatrix
        );
        assert_eq!(
            classify_nemotron_transcribe_tensor_role("blk.52.ffn_down_exps.weight"),
            TensorRole::TextDecoderMatrix
        );
        assert_eq!(
            classify_nemotron_transcribe_tensor_role("blk.52.nextn.eh_proj.weight"),
            TensorRole::TextDecoderMatrix
        );
        assert_eq!(
            classify_nemotron_transcribe_tensor_role("token_embd.weight"),
            TensorRole::EmbeddingTable
        );
        assert_eq!(
            classify_nemotron_transcribe_tensor_role("output.weight"),
            TensorRole::OutputProjection
        );
        assert_eq!(
            classify_nemotron_transcribe_tensor_role("output_norm.weight"),
            TensorRole::NonQuantizable
        );
    }

    #[test]
    fn quantization_contract_targets_this_architecture() {
        assert_eq!(
            TENSOR_QUANTIZATION_CONTRACT.model_architecture(),
            NEMOTRON_TRANSCRIBE_GGML_ARCHITECTURE_ID
        );
    }
}
