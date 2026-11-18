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
//! STAGED, fail-closed: this module exists so the family's packs can pass the
//! production `PackVerifier` and so the release pipeline can proceed. The
//! decoder/encoder/speaker-branch/CTC runtime itself has NOT landed: the
//! executor fails closed with a "not yet implemented" error, and the family
//! must not be treated as runnable until the executor, the streaming session
//! driver, and the golden-parity gate are real. See
//! `docs/model-audits/nemotron-transcribe.md`.

pub(crate) mod executor;
pub(crate) mod package_import;
pub(crate) mod runtime_contract;

pub(crate) use executor::NemotronTranscribeGgmlExecutor;
