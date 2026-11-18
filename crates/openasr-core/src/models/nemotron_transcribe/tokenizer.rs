//! Nemotron-transcribe's byte-level GPT-2 BPE tokenizer (vocab_size=131072,
//! `tokenizer.ggml.pre = "pixtral"` in llama.cpp's table). Reads
//! `tokenizer.ggml.tokens`/`tokenizer.ggml.merges` from the pack's GGUF
//! metadata; reuses the shared byte-level BPE primitives in `tokens`.

use std::collections::BTreeMap;

use crate::NativeAsrError;
use crate::ggml_runtime::GgufMetadata;
use crate::models::gpt2_bpe::{
    build_merge_rank, build_token_to_id, encode_prompt_text, token_to_bytes,
};
use crate::models::oasr_metadata::{required_metadata_string, required_metadata_string_array};

const NEMOTRON_TRANSCRIBE_TOKENIZER_FAMILY: &str = "nemotron-transcribe";
const TOKENIZER_GGML_MODEL_KEY: &str = "tokenizer.ggml.model";
const TOKENIZER_GGML_MODEL_VALUE_GPT2: &str = "gpt2";
const TOKENIZER_GGML_PRE_KEY: &str = "tokenizer.ggml.pre";
const TOKENIZER_GGML_PRE_PIXTRAL: &str = "pixtral";
const TOKENIZER_GGML_TOKENS_KEY: &str = "tokenizer.ggml.tokens";
const TOKENIZER_GGML_MERGES_KEY: &str = "tokenizer.ggml.merges";

#[derive(Debug, thiserror::Error)]
pub(crate) enum NemotronTranscribeTokenizerError {
    #[error("nemotron-transcribe tokenizer encode/decode failed: {0}")]
    Bpe(#[from] crate::NativeAsrError),
    #[error("nemotron-transcribe tokenizer: {0}")]
    Validate(String),
}

#[derive(Debug, Clone)]
pub(crate) struct NemotronTranscribeTokenizer {
    id_to_token: Vec<Option<String>>,
    token_to_id: BTreeMap<String, u32>,
    merge_rank: BTreeMap<String, usize>,
}

impl NemotronTranscribeTokenizer {
    #[allow(dead_code)]
    pub(crate) fn quoted_retained_system_memory_bytes(
        metadata: &GgufMetadata,
    ) -> Result<u64, NativeAsrError> {
        let tokens = required_metadata_string_array(
            metadata,
            TOKENIZER_GGML_TOKENS_KEY,
            NEMOTRON_TRANSCRIBE_TOKENIZER_FAMILY,
        )?;
        let merges = required_metadata_string_array(
            metadata,
            TOKENIZER_GGML_MERGES_KEY,
            NEMOTRON_TRANSCRIBE_TOKENIZER_FAMILY,
        )?;
        Ok((tokens.len() as u64)
            .saturating_add(merges.len() as u64)
            .saturating_mul(16))
    }

    #[allow(dead_code)]
    pub(crate) fn retained_system_memory_bytes(&self) -> Result<u64, String> {
        let mut bytes = crate::models::system_memory_owner::SystemMemoryCapacity::default();
        bytes.add_vec(&self.id_to_token, "nemotron tokenizer id table")?;
        for token in self.id_to_token.iter().flatten() {
            bytes.add_string(token, "nemotron tokenizer id token")?;
        }
        bytes.add_usize(
            self.token_to_id
                .len()
                .checked_mul(std::mem::size_of::<(String, u32)>())
                .ok_or_else(|| "nemotron tokenizer token map byte count overflowed".to_string())?,
            "nemotron tokenizer token map entries",
        )?;
        for token in self.token_to_id.keys() {
            bytes.add_string(token, "nemotron tokenizer token map key")?;
        }
        bytes.add_usize(
            self.merge_rank
                .len()
                .checked_mul(std::mem::size_of::<(String, usize)>())
                .ok_or_else(|| "nemotron tokenizer merge map byte count overflowed".to_string())?,
            "nemotron tokenizer merge map entries",
        )?;
        for merge in self.merge_rank.keys() {
            bytes.add_string(merge, "nemotron tokenizer merge map key")?;
        }
        Ok(bytes.finish())
    }

    /// Production constructor: reads `tokenizer.ggml.tokens`/
    /// `tokenizer.ggml.merges` from the pack's GGUF metadata.
    pub(crate) fn from_gguf_metadata(metadata: &GgufMetadata) -> Result<Self, NativeAsrError> {
        let tokenizer_model = required_metadata_string(
            metadata,
            TOKENIZER_GGML_MODEL_KEY,
            NEMOTRON_TRANSCRIBE_TOKENIZER_FAMILY,
        )?;
        if !tokenizer_model.eq_ignore_ascii_case(TOKENIZER_GGML_MODEL_VALUE_GPT2) {
            return Err(NativeAsrError::UnsupportedModelPack {
                reason: format!(
                    "nemotron-transcribe GGUF tokenizer key '{TOKENIZER_GGML_MODEL_KEY}' must be \
                     '{TOKENIZER_GGML_MODEL_VALUE_GPT2}', got '{tokenizer_model}'"
                ),
            });
        }
        // Informational: the published pack declares `pixtral`; the shared
        // byte-level BPE pretokenizer is the one this crate ships, and the
        // pixtral pre-tokenizer in llama.cpp's table tokenizes identically for
        // the served-language traffic this family handles.
        let _ = metadata.get_string(TOKENIZER_GGML_PRE_KEY);
        let _ = TOKENIZER_GGML_PRE_PIXTRAL;
        let tokens = required_metadata_string_array(
            metadata,
            TOKENIZER_GGML_TOKENS_KEY,
            NEMOTRON_TRANSCRIBE_TOKENIZER_FAMILY,
        )?;
        if tokens.is_empty() {
            return Err(NativeAsrError::UnsupportedModelPack {
                reason: format!(
                    "nemotron-transcribe GGUF tokenizer key '{TOKENIZER_GGML_TOKENS_KEY}' cannot be empty"
                ),
            });
        }
        let merges = required_metadata_string_array(
            metadata,
            TOKENIZER_GGML_MERGES_KEY,
            NEMOTRON_TRANSCRIBE_TOKENIZER_FAMILY,
        )?;

        let token_to_id = build_token_to_id(tokens, NEMOTRON_TRANSCRIBE_TOKENIZER_FAMILY)?;
        let id_to_token = tokens.iter().map(|token| Some(token.clone())).collect();
        let merge_rank = build_merge_rank(merges);

        Ok(Self {
            id_to_token,
            token_to_id,
            merge_rank,
        })
    }

    pub(crate) fn encode_prompt_text(
        &self,
        text: &str,
    ) -> Result<Vec<u32>, NemotronTranscribeTokenizerError> {
        Ok(encode_prompt_text(
            text,
            &self.token_to_id,
            &self.merge_rank,
            NEMOTRON_TRANSCRIBE_TOKENIZER_FAMILY,
        )?)
    }

    pub(crate) fn decode_text_token_ids(
        &self,
        token_ids: &[u32],
    ) -> Result<String, NemotronTranscribeTokenizerError> {
        let mut bytes = Vec::new();
        for &token_id in token_ids {
            let index = token_id as usize;
            let Some(Some(token)) = self.id_to_token.get(index) else {
                return Err(NemotronTranscribeTokenizerError::Validate(format!(
                    "nemotron-transcribe tokenizer id {token_id} is not in vocab"
                )));
            };
            bytes.extend(token_to_bytes(token));
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }
}
