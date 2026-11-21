//! Nemotron 3.5 Transcribe 30B-A4B (`nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B`)
//! model family: `nemotron-transcribe`.
//!
//! NVIDIA's open-weight offline speech LLM (40 languages, recordings up to
//! one hour, speaker attribution, speech translation). The pack is a single
//! merged GGUF stream (`docs/format/OASR_PACKAGE_CONTRACT_V1.md`) carrying the
//! 53-block Mamba-2/Transformer/MoE decoder (`nemotron_h_moe`) plus the
//! shared ~1.67 GB F16 sidecar contents as `asr.*`/`diar.*`/`ctc.*`/
//! `fusion.*`/`preprocessor.*` tensors: a 32-layer ASR encoder, a 31-layer
//! Sortformer speaker branch, a CTC timestamp head, and the mel filterbank.
//! Packs are built externally (decoder + sidecar merge; see
//! `tooling/publish-model/scripts/nemotron_transcribe_gguf_to_oasr.py`) and
//! already carry the full `openasr.*` envelope.
//!
//! Bring-up state: the offline execution path (frontend -> PE encoder ->
//! nemotron_h_moe trunk -> shared greedy driver -> verbatim transcript) is
//! implemented; streaming is the shared snapshot driver (registration
//! completeness, not a streaming-tuned latency claim). In-decoder speaker
//! ids and the translation task remain unimplemented; the CTC timestamp head
//! and word timestamps are implemented (verified against the oracle); the
//! audit form stays honest about the rest.

pub(crate) mod capacity;
pub(crate) mod ctc_align;
pub(crate) mod ctc_head;
pub(crate) mod ctc_spm;
pub(crate) mod ctc_unit_tables;
pub(crate) mod decode_executor;
pub(crate) mod decoder_graph;
pub(crate) mod encoder_graph;
pub(crate) mod executor;
pub(crate) mod frontend;
pub(crate) mod offload;
pub(crate) mod package_import;
pub(crate) mod prompt;
pub(crate) mod runtime_contract;
pub(crate) mod slice;
pub(crate) mod tokenizer;

pub(crate) use executor::NemotronTranscribeGgmlExecutor;
