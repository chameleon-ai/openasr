//! `GgmlAsrViewExecutor` implementation for nemotron-transcribe: wires the
//! pipeline (`frontend` -> `encoder_graph` -> `prompt` -> `decoder_graph`
//! resident session -> shared greedy-decode driver -> `tokenizer`) against a
//! real `.oasr` pack through typed, already-open runtime views.
//!
//! Scope of this bring-up pass:
//! - file-transcribe (offline) path: near-digital silence gate -> frontend ->
//!   PE encoder (ASR expert + Sortformer speaker branch + fusion) -> NoPE
//!   nemotron_h_moe trunk (Mamba-2/MoE/attention) -> shared greedy driver ->
//!   verbatim transcript. The speaker branch runs as part of the fused PE
//!   encoder (single-window path); the decode-mode diarize prompt is wired
//!   but the transcript keeps the raw `<spk:N>` tags.
//! - word timestamps: the ASR expert's post-`final_norm` states are tapped,
//!   the CTC head (`ctc_head`) runs them on the 40 ms grid, and the ported
//!   host aligner (`ctc_align` + `ctc_spm`) produces `Segment.words` when
//!   `request_options.word_timestamps` is set.
//! - in-decoder speaker ids remain unverified.
//! - streaming: the shared snapshot streaming driver re-runs this same
//!   offline pass over the growing buffer (registration completeness; no
//!   streaming-tuned latency claim).

use std::borrow::Cow;
use std::sync::Arc;

use thiserror::Error;

use super::ctc_align::{
    AlignmentConfig, AlignmentInputs, align_record, alignment_targets, clean_sot_transcript,
    public_words,
};
use super::ctc_head::{CTC_BLANK_ID, CTC_UPSAMPLING, CTC_VOCAB_SIZE, CtcHeadRuntime, bf16_round};
use super::ctc_spm::CtcSpm;
use super::decode_executor::NemotronTranscribeDecodeSession;
use super::decode_executor::NemotronTranscribeResidentAudioDecodeStepExecutor;
use super::decoder_graph::NemotronTranscribeDecoderRuntime;
use super::encoder_graph::NemotronTranscribeEncoderRuntime;
use super::frontend::{NemotronMelFeatures, NemotronMelFrontend, NemotronMelFrontendError};
use super::prompt::{self, NemotronTranscribePromptError};
use super::runtime_contract::NemotronTranscribeRuntimeContractError;
use super::tokenizer::NemotronTranscribeTokenizer;

use crate::api::backend::{Segment, Transcription};
use crate::ggml_runtime::{GgmlCpuGraphBackend, GgufRuntimeSourcePreflight};
use crate::models::admitted_pinned_runtime_actor_pool::{
    AdmittedPinnedRuntimeActorCheckoutPool, AdmittedPinnedRuntimeActorCheckoutPoolLimits,
    PinnedRuntimeActorCheckout, PinnedRuntimeActorError,
};
use crate::models::decode_policy_component_registry::{
    BuiltinDecodePolicyComponentRegistryError, BuiltinSeq2SeqDecodePolicyConfigInput,
    run_builtin_seq2seq_decode_policy,
};
use crate::models::ggml_asr_executor::{
    GgmlAsrExecutionError, GgmlAsrExecutionResult, GgmlAsrExecutionViewRequest,
    GgmlAsrPreparedAudioView, GgmlAsrViewExecutor,
};
use crate::models::mapped_token_embedding::{
    MappedTokenEmbeddingTable, load_mapped_token_embedding_table_from_reader,
};
use crate::models::native_execution_services::{ExecutionLaneKey, current_execution_lane_key};
use crate::models::runtime_cache_coordinator::PackContentKey;
use crate::models::runtime_preflight::build_runtime_tensor_reader_from_preflight;
use crate::models::seq2seq_greedy_decode::Seq2SeqGreedyDecodeError;
use crate::models::system_memory_owner::SystemMemoryOwner;

use crate::arch::NEMOTRON_TRANSCRIBE_GGML_ADAPTER_ID;

pub(crate) const NEMOTRON_TRANSCRIBE_EXECUTOR_ID: &str =
    crate::arch::NEMOTRON_TRANSCRIBE_EXECUTOR_COMPONENT_ID;
/// `<|im_end|>` is the greedy stop token for this family.
pub(crate) const NEMOTRON_TRANSCRIBE_EOT_TOKEN_ID: u32 = 11;
/// No-voice gate: utterance whose speaker probabilities never exceed this is
/// treated as silence (the LLM never runs).
pub(crate) const NEMOTRON_TRANSCRIBE_SPEAKER_ACTIVITY_THRESHOLD: f32 = 0.5;

const NEMOTRON_TRANSCRIBE_MIN_KV_POSITIONS: usize = 640;

/// Recovery ladder for a temperature-0 decode cut short by the repeat guard
/// or stripped to empty. One rung at 0.4 is the default: it is the only
/// configuration that recovered real speech on both bonnie and mikan without
/// the GPU faults seen with wider ladders. Override with a comma-separated
/// list in `OPENASR_NEMOTRON_TEMPERATURE_LADDER`, or set it to `0`/`off` to
/// disable. Nemotron decoder rounds cost far more than whisper's.
const NEMOTRON_DEFAULT_TEMPERATURE_LADDER: [f32; 1] = [0.4];
const NEMOTRON_LADDER_BASE_SEED: u64 = 0x9E37_79B9_7F4A_7C15;
/// Any recovery round (ladder or fallback prompt) must clear this mean
/// per-frame CTC score before its text is trusted. Scores below it are
/// force-alignments of hallucinated words onto music / noise: the bad
/// adoptions measured on lobster-uvr (1.86, "It's gonna be over there" x4)
/// and tequila-uvr (0.74, "Heeft die ja!") sit below it, while mikan's genuine
/// recoveries (3.34 and 4.66) clear it.
const NEMOTRON_MIN_ADOPT_CTC_SCORE: f32 = 3.0;
/// A sampled round must beat the incumbent's mean per-label CTC score by this
/// much to be adopted (scores are log-probs; the elided full-vocab term is a
/// per-frame constant, so comparisons across rounds on the same audio hold).
const NEMOTRON_LADDER_SCORE_MARGIN: f32 = 0.05;

fn nemotron_temperature_ladder() -> Vec<f32> {
    match std::env::var("OPENASR_NEMOTRON_TEMPERATURE_LADDER") {
        Ok(raw) => {
            let normalized = raw.trim().to_ascii_lowercase();
            if matches!(normalized.as_str(), "0" | "off" | "false" | "no") {
                Vec::new()
            } else {
                raw.split(',')
                    .filter_map(|part| part.trim().parse::<f32>().ok())
                    .filter(|temperature| *temperature > 0.0 && *temperature <= 1.0)
                    .collect()
            }
        }
        Err(_) => NEMOTRON_DEFAULT_TEMPERATURE_LADDER.to_vec(),
    }
}

/// Extra KV rows reserved at the family runtime beyond the planner's minimal
/// `P + G - 1` span. When the shared greedy driver's degenerate-ngram guard
/// drops tokens, its decode can advance one step past the nominal generation
/// budget and write one more KV row; without margin that step fails closed
/// (`decode position N exhausts the resident KV span N`). A small margin keeps
/// the admission and the actor pool key tied to the planner's stable span.
const NEMOTRON_TRANSCRIBE_KV_SPAN_MARGIN: usize = 16;
/// Resident unified-runtime pool limits: a few idle (pack, lane, span)
/// buckets; two actors per bucket. Native ownership is bound at build time,
/// so the pool tracks no extra SystemMemory bytes.
const NEMOTRON_TRANSCRIBE_RUNTIME_ACTOR_MAX_IDLE_ENTRIES: usize = 4;
const NEMOTRON_TRANSCRIBE_RUNTIME_ACTOR_MAX_INSTANCES_PER_KEY: usize = 2;

/// Resident-runtime identity. The span comes from the stable invocation
/// envelope, not the exact slice: longform slices of different lengths must
/// share one resident runtime, otherwise every distinct span builds a new
/// actor with its own pack residency and graph buffers. The envelope is
/// sized from the longform window ceiling, so it always covers a slice.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct NemotronTranscribeUnifiedRuntimeCacheKey {
    content: PackContentKey,
    lane: ExecutionLaneKey,
    required_span: usize,
}

/// Resident device/host runtime: the already-uploaded encoder and decoder
/// graphs plus the launch-side tokenizer and mmap'd embedding table. Owned
/// by a lane-keyed actor so it stays on the thread that built it and is
/// reused across requests.
struct NemotronTranscribeUnifiedRuntime {
    prepared: NemotronTranscribePreparedRuntime,
}

type NemotronTranscribeUnifiedRuntimePool = AdmittedPinnedRuntimeActorCheckoutPool<
    NemotronTranscribeUnifiedRuntimeCacheKey,
    NemotronTranscribeUnifiedRuntime,
>;
type NemotronTranscribeUnifiedRuntimeActor = PinnedRuntimeActorCheckout<
    NemotronTranscribeUnifiedRuntimeCacheKey,
    NemotronTranscribeUnifiedRuntime,
>;

#[derive(Debug, Error)]
#[allow(dead_code)]
pub(crate) enum NemotronTranscribeGgmlExecutorError {
    #[error("nemotron-transcribe requires adapter '{expected}', got '{found}'")]
    AdapterMismatch {
        expected: &'static str,
        found: String,
    },
    #[error("nemotron-transcribe audio duration {seconds:.1}s exceeds the supported envelope")]
    AudioTooLong { seconds: f32 },
    #[error("nemotron-transcribe runtime contract failed: {reason}")]
    RuntimeContractFailed { reason: String },
    #[error("nemotron-transcribe frontend failed: {reason}")]
    FrontendFailed { reason: String },
    #[error("nemotron-transcribe mel normalization failed: {reason}")]
    NormalizationFailed { reason: String },
    #[error("nemotron-transcribe encoder failed: {reason}")]
    EncoderFailed { reason: String },
    #[error("nemotron-transcribe tokenizer failed: {reason}")]
    TokenizerFailed { reason: String },
    #[error("nemotron-transcribe prompt assembly failed: {reason}")]
    PromptFailed { reason: String },
    #[error("nemotron-transcribe decode failed: {reason}")]
    DecodeFailed { reason: String },
    #[error("nemotron-transcribe decoder-state capacity failed: {reason}")]
    CapacityFailed { reason: String },
    #[error("nemotron-transcribe prepared runtime owner failed: {reason}")]
    RuntimeOwnershipFailed { reason: String },
    #[error("nemotron-transcribe CTC word timestamps failed: {reason}")]
    WordTimestampsFailed { reason: String },
}

fn map_registry_error(
    error: BuiltinDecodePolicyComponentRegistryError,
) -> Seq2SeqGreedyDecodeError {
    Seq2SeqGreedyDecodeError::DecoderStepFailed {
        reason: error.to_string(),
    }
}

impl From<NemotronMelFrontendError> for NemotronTranscribeGgmlExecutorError {
    fn from(error: NemotronMelFrontendError) -> Self {
        Self::FrontendFailed {
            reason: error.to_string(),
        }
    }
}

impl From<NemotronTranscribeRuntimeContractError> for NemotronTranscribeGgmlExecutorError {
    fn from(error: NemotronTranscribeRuntimeContractError) -> Self {
        Self::RuntimeContractFailed {
            reason: error.to_string(),
        }
    }
}

impl From<NemotronTranscribePromptError> for NemotronTranscribeGgmlExecutorError {
    fn from(error: NemotronTranscribePromptError) -> Self {
        Self::PromptFailed {
            reason: error.to_string(),
        }
    }
}

impl From<super::encoder_graph::NemotronTranscribeEncoderError>
    for NemotronTranscribeGgmlExecutorError
{
    fn from(error: super::encoder_graph::NemotronTranscribeEncoderError) -> Self {
        Self::EncoderFailed {
            reason: error.to_string(),
        }
    }
}

impl From<super::ctc_head::CtcHeadError> for NemotronTranscribeGgmlExecutorError {
    fn from(error: super::ctc_head::CtcHeadError) -> Self {
        Self::WordTimestampsFailed {
            reason: error.to_string(),
        }
    }
}

impl From<super::decoder_graph::NemotronTranscribeDecoderError>
    for NemotronTranscribeGgmlExecutorError
{
    fn from(error: super::decoder_graph::NemotronTranscribeDecoderError) -> Self {
        Self::DecodeFailed {
            reason: error.to_string(),
        }
    }
}

#[derive(Clone)]
pub(crate) struct NemotronTranscribeGgmlExecutor {
    unified_runtimes: Arc<NemotronTranscribeUnifiedRuntimePool>,
}

impl Default for NemotronTranscribeGgmlExecutor {
    fn default() -> Self {
        let max_committed_requested_bytes =
            crate::host::host_available_memory_bytes().unwrap_or(u64::MAX);
        Self {
            unified_runtimes: Arc::new(AdmittedPinnedRuntimeActorCheckoutPool::new(
                "openasr-nemotron-transcribe-unified-owner",
                AdmittedPinnedRuntimeActorCheckoutPoolLimits::new(
                    NEMOTRON_TRANSCRIBE_RUNTIME_ACTOR_MAX_IDLE_ENTRIES,
                    max_committed_requested_bytes,
                    NEMOTRON_TRANSCRIBE_RUNTIME_ACTOR_MAX_INSTANCES_PER_KEY,
                ),
            )),
        }
    }
}

impl std::fmt::Debug for NemotronTranscribeGgmlExecutor {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("NemotronTranscribeGgmlExecutor")
            .finish_non_exhaustive()
    }
}

/// Everything materialized from the pack for one resident runtime: tokenizer,
/// the mmap'd embedding table, the resident encoder runtime, and the
/// decoder session (KV span + mamba state). Owned by the lane-keyed actor
/// pool; the pack is re-mmap'd from the request's already-open source per
/// build, never re-read from disk.
struct NemotronTranscribePreparedRuntime {
    tokenizer: NemotronTranscribeTokenizer,
    embed_table: MappedTokenEmbeddingTable,
    encoder: NemotronTranscribeEncoderRuntime,
    decoder: NemotronTranscribeDecodeSession,
    /// CTC timestamp head (runs on the ASR expert's tapped states).
    ctc_head: CtcHeadRuntime,
    /// CTC aligner's SentencePiece tokenizer.
    ctc_spm: CtcSpm,
}

impl NemotronTranscribePreparedRuntime {
    fn build(
        preflight: &GgufRuntimeSourcePreflight,
        backend: GgmlCpuGraphBackend,
        decoder_span: usize,
    ) -> Result<Self, NemotronTranscribeGgmlExecutorError> {
        let metadata = &preflight.metadata;
        let tokenizer =
            NemotronTranscribeTokenizer::from_gguf_metadata(metadata).map_err(|error| {
                NemotronTranscribeGgmlExecutorError::TokenizerFailed {
                    reason: error.to_string(),
                }
            })?;
        let reader = build_runtime_tensor_reader_from_preflight(preflight).map_err(|error| {
            NemotronTranscribeGgmlExecutorError::RuntimeContractFailed {
                reason: error.to_string(),
            }
        })?;
        let embed_table = load_mapped_token_embedding_table_from_reader(
            &reader,
            "token_embd.weight",
            2688,
            131_072,
        )
        .map_err(
            |error| NemotronTranscribeGgmlExecutorError::RuntimeContractFailed {
                reason: error.to_string(),
            },
        )?;
        // Only the scheduler-backed Hybrid placement can split weight
        // residency (its CPU fallback hosts the offloaded tensors). Ask the
        // VRAM-derived policy which MoE experts must leave the device and
        // hand the same spec to every runner: encoder, decoder, and CTC head
        // coalesce onto one pack-wide weight owner, so three distinct specs
        // would alias or double the residency.
        let host_offload =
            match crate::models::native_execution_services::current_execution_placement() {
                Some(crate::device::execution_policy::ExecutionPlacement::Hybrid) => {
                    super::offload::compute_host_offload_spec(backend, &preflight.tensor_index)
                }
                _ => None,
            };
        let encoder = NemotronTranscribeEncoderRuntime::new_from_preflight(
            preflight,
            backend,
            host_offload.clone(),
        )?;
        let decoder = NemotronTranscribeDecoderRuntime::new_from_preflight(
            preflight,
            decoder_span,
            backend,
            host_offload.clone(),
        )?;
        // The CTC head and its aligner tokenizer ship in every pack. Fail
        // closed if the SPM metadata or its vocabulary is missing/invalid.
        const CTC_SPM_KEY: &str = "ctc.tokenizer.spm_model";
        let spm_b64 = metadata.get_string(CTC_SPM_KEY).ok_or_else(|| {
            NemotronTranscribeGgmlExecutorError::WordTimestampsFailed {
                reason: format!("pack metadata '{CTC_SPM_KEY}' is missing"),
            }
        })?;
        let spm_bytes = super::ctc_spm::decode_base64(spm_b64).map_err(|error| {
            NemotronTranscribeGgmlExecutorError::WordTimestampsFailed {
                reason: error.to_string(),
            }
        })?;
        let ctc_spm = CtcSpm::parse(&spm_bytes, CTC_VOCAB_SIZE - 1).map_err(|error| {
            NemotronTranscribeGgmlExecutorError::WordTimestampsFailed {
                reason: error.to_string(),
            }
        })?;
        let ctc_head = CtcHeadRuntime::new_from_preflight(preflight, backend, host_offload)?;
        Ok(Self {
            tokenizer,
            embed_table,
            encoder,
            decoder: NemotronTranscribeDecodeSession::new(decoder),
            ctc_head,
            ctc_spm,
        })
    }
}

fn downmix_prepared_audio<'a>(audio: &'a GgmlAsrPreparedAudioView<'_>) -> Cow<'a, [f32]> {
    if audio.channels <= 1 {
        return Cow::Borrowed(audio.samples_f32.as_ref());
    }
    let channels = audio.channels as usize;
    Cow::Owned(
        audio
            .samples_f32
            .chunks_exact(channels)
            .map(|frame| frame.iter().sum::<f32>() / channels as f32)
            .collect(),
    )
}

/// Mean/unbiased-std per-feature normalization over the valid frames
/// (reference `norm_stats`): mean per bin, `inv_std = 1/(sqrt(ss/(t-1)) +
/// 1e-5f)` with the epsilon added AFTER the sqrt. The padded tail frames are
/// zeroed.
fn normalize_per_feature(features: &[f32], frames: usize) -> Vec<f32> {
    let n_mels = super::frontend::NEMOTRON_TRANSCRIBE_N_MELS;
    debug_assert_eq!(features.len(), frames * n_mels);
    let mut out = features.to_vec();
    for m in 0..n_mels {
        let mut sum = 0.0f64;
        for t in 0..frames {
            sum += features[t * n_mels + m] as f64;
        }
        let mean = sum / frames as f64;
        let mut ss = 0.0f64;
        for t in 0..frames {
            let d = features[t * n_mels + m] as f64 - mean;
            ss += d * d;
        }
        let std = if frames > 1 {
            (ss / (frames - 1) as f64).sqrt()
        } else {
            0.0
        };
        let inv_std = 1.0 / (std as f32 + 1e-5);
        for t in 0..frames {
            out[t * n_mels + m] = (features[t * n_mels + m] - mean as f32) * inv_std;
        }
    }
    out
}

/// Zero-pad `features` (frame-major, `n_mels` columns) to `padded_frames`.
fn pad_mel(features: &[f32], frames: usize, padded_frames: usize) -> Vec<f32> {
    let n_mels = super::frontend::NEMOTRON_TRANSCRIBE_N_MELS;
    let mut out = vec![0.0f32; padded_frames * n_mels];
    out[..frames * n_mels].copy_from_slice(features);
    out
}

/// Whether a `<...>` span is a diarization speaker tag. Those are turn markers
/// the caller needs in diarize mode, unlike the SALM scaffolding below.
fn is_speaker_markup(span: &str) -> bool {
    let inner = span
        .strip_prefix("</")
        .or_else(|| span.strip_prefix('<'))
        .and_then(|rest| rest.strip_suffix('>'))
        .unwrap_or("");
    let Some(digits) = inner.strip_prefix("spk:") else {
        return false;
    };
    !digits.is_empty() && digits.bytes().all(|byte| byte.is_ascii_digit())
}

/// Remove SALM scaffolding markup from a decoded transcript.
///
/// The observed tags (`<ev:audio>`, `<ev:ev-step end="true" ...>`, `<voice>`,
/// `</ev:segment>`, misspellings like `<ev:asistant>`, and tags split across
/// word boundaries) are *free text* built from ordinary BPE pieces, not
/// vocabulary specials, so they can only be recognised textually. A literal
/// `<`/`>` pair in genuine speech is rare; anything that looks like a closed
/// tag is treated as markup. Speaker tags can be preserved for diarize mode.
fn strip_scaffolding_markup(text: &str, preserve_speaker_tags: bool) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    let mut removed = false;
    while let Some(open) = rest.find('<') {
        out.push_str(&rest[..open]);
        let tail = &rest[open..];
        let Some(close) = tail.find('>') else {
            // Unterminated '<': keep the remainder verbatim.
            out.push_str(tail);
            rest = "";
            break;
        };
        let span = &tail[..=close];
        let keep = preserve_speaker_tags && is_speaker_markup(span);
        if !keep {
            removed = true;
        } else {
            out.push_str(span);
        }
        rest = &tail[close + 1..];
    }
    out.push_str(rest);
    if !removed {
        return out;
    }
    // Tag removal leaves doubled/orphaned whitespace. The CTC alignment target
    // is whitespace-normalised by `clean_sot_transcript`, so do the same here
    // to keep `text` and the aligned words consistent.
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

impl NemotronTranscribeGgmlExecutor {
    fn required_self_kv_positions(
        decoder_state: &crate::models::ggml_asr_executor::GgmlAsrDecoderState,
    ) -> Result<usize, NemotronTranscribeGgmlExecutorError> {
        match decoder_state {
            crate::models::ggml_asr_executor::GgmlAsrDecoderState::Planned(state) => Ok(state
                .allocations()
                .iter()
                .find(|allocation| allocation.logical.id == crate::models::nemotron_transcribe::capacity::NEMOTRON_TRANSCRIBE_SELF_KV_STATE_ID)
                // Use the envelope's reserve span, not the exact slice's
                // logical span: all slices of a longform request then share
                // one actor instead of allocating a new ~15 GB pack residency
                // for every distinct slice length.
                .map(|allocation| allocation.reserve.positions)
                .ok_or_else(|| NemotronTranscribeGgmlExecutorError::CapacityFailed {
                    reason: "decoder-state plan is missing the nemotron self-KV stream"
                        .to_string(),
                })?),
            crate::models::ggml_asr_executor::GgmlAsrDecoderState::NoPersistentState => Err(
                NemotronTranscribeGgmlExecutorError::CapacityFailed {
                    reason: "decoder-state plan declared no persistent state; the hybrid trunk requires one"
                        .to_string(),
                },
            ),
        }
    }

    fn map_unified_actor_error(
        error: PinnedRuntimeActorError,
    ) -> NemotronTranscribeGgmlExecutorError {
        NemotronTranscribeGgmlExecutorError::RuntimeOwnershipFailed {
            reason: error.to_string(),
        }
    }

    /// Resident lane-keyed runtime: the pack's encoder and decoder graphs
    /// are uploaded once per (pack content, lane, span) and reused by later
    /// requests through the pinned actor pool; the pack itself is re-mmap'd
    /// from the request's already-open source, never re-read from disk.
    fn checkout_unified_runtime(
        &self,
        preflight: &GgufRuntimeSourcePreflight,
        backend: GgmlCpuGraphBackend,
        required_span: usize,
    ) -> Result<NemotronTranscribeUnifiedRuntimeActor, NemotronTranscribeGgmlExecutorError> {
        let key = NemotronTranscribeUnifiedRuntimeCacheKey {
            content: PackContentKey::for_runtime_source(&preflight.runtime_source),
            lane: current_execution_lane_key(backend),
            required_span,
        };
        let quote_preflight = preflight.clone();
        self.unified_runtimes.checkout_or_try_build_with(
            key,
            move || Ok((0, (quote_preflight, required_span, backend))),
            move |(preflight, required_span, backend)| {
                NemotronTranscribePreparedRuntime::build(
                    &preflight,
                    backend,
                    required_span
                        .max(NEMOTRON_TRANSCRIBE_MIN_KV_POSITIONS)
                        .saturating_add(NEMOTRON_TRANSCRIBE_KV_SPAN_MARGIN),
                )
                .map(|prepared| {
                    SystemMemoryOwner::without_allocation(NemotronTranscribeUnifiedRuntime {
                        prepared,
                    })
                })
            },
            Self::map_unified_actor_error,
        )
    }

    fn empty_transcription(audio_duration_seconds: f32) -> Transcription {
        Transcription {
            truncated_decodes: Vec::new(),
            unnamed_speakers: Vec::new(),
            text: String::new(),
            segments: vec![Segment {
                start: 0.0,
                end: audio_duration_seconds.max(0.0),
                text: String::new(),
                speaker: None,
                speaker_label: None,
                speaker_person_id: None,
                speaker_snapshot_label: None,
                words: Vec::new(),
            }],
            longform: None,
            language: None,
            ..Default::default()
        }
    }

    fn execute_inner(
        &self,
        request: &GgmlAsrExecutionViewRequest,
    ) -> Result<GgmlAsrExecutionResult, NemotronTranscribeGgmlExecutorError> {
        if request.selected_family.adapter_id != NEMOTRON_TRANSCRIBE_GGML_ADAPTER_ID {
            return Err(NemotronTranscribeGgmlExecutorError::AdapterMismatch {
                expected: NEMOTRON_TRANSCRIBE_GGML_ADAPTER_ID,
                found: request.selected_family.adapter_id.to_string(),
            });
        }
        if request.prepared_audio.sample_rate_hz
            != super::frontend::NEMOTRON_TRANSCRIBE_SAMPLE_RATE_HZ
        {
            return Err(NemotronTranscribeGgmlExecutorError::FrontendFailed {
                reason: format!(
                    "nemotron-transcribe requires a 16 kHz mono PCM signal, got {} Hz",
                    request.prepared_audio.sample_rate_hz
                ),
            });
        }
        let preflight = request.runtime_source_preflight();
        let samples = downmix_prepared_audio(&request.prepared_audio);
        let audio_duration_seconds = samples.len() as f32 / 16_000.0;
        if prompt::is_near_digital_silence(samples.as_ref()) {
            return Ok(GgmlAsrExecutionResult {
                transcription: Self::empty_transcription(audio_duration_seconds),
                carry_context: None,
                decode_truncation: None,
            });
        }

        // --- decoder-state span from the planner's output ---
        let required_span = Self::required_self_kv_positions(&request.decoder_state)?;

        // --- backend ---
        let backend = request.resolved_runtime.backend();

        // --- frontend ---
        let frontend = NemotronMelFrontend::from_preflight_fb(preflight)?;
        let NemotronMelFeatures { features, frames } = frontend.extract(samples.as_ref())?;
        let sub = super::frontend::NEMOTRON_TRANSCRIBE_HOP_LENGTH;
        let _ = sub;
        let t_out = frames.div_ceil(8);
        let t_padded = t_out * 8;
        let asr_mel = normalize_per_feature(&features, frames);
        let asr_mel = pad_mel(&asr_mel, frames, t_padded);
        let diar_mel = pad_mel(&features, frames, t_padded);

        // --- resident unified runtime (lane-keyed, reused across requests) ---
        let actor = self.checkout_unified_runtime(preflight, backend, required_span)?;
        let control = Arc::clone(&request.execution_context.control);
        let decode_work_progress = request
            .execution_context
            .decode_work_progress_observer()
            .cloned();
        let unstable_decode_text = request
            .execution_context
            .unstable_decode_text_observer()
            .cloned();
        let in_decoder_speakers = request.request_options.in_decoder_speakers;
        let language = request.request_options.language.clone();
        let word_timestamps = request.request_options.word_timestamps;

        actor
            .call_mut_fallible(move |runtime| -> Result<GgmlAsrExecutionResult, NemotronTranscribeGgmlExecutorError> {
                let prepared = &mut runtime.prepared;

                // --- PE encoder ---
                let encoder_output = prepared.encoder.encode(&diar_mel, &asr_mel, t_padded)?;
                prepared.encoder.release_transient_compute_memory()?;

                // No-voice gate: an utterance whose speaker probabilities never
                // exceed the activity threshold produces an empty transcript.
                let voice = encoder_output
                    .speaker_probs
                    .iter()
                    .any(|&p| p > NEMOTRON_TRANSCRIBE_SPEAKER_ACTIVITY_THRESHOLD);
                if !voice {
                    return Ok(GgmlAsrExecutionResult {
                        transcription: Self::empty_transcription(audio_duration_seconds),
                        carry_context: None,
                        decode_truncation: None,
                    });
                }

                // --- prompt ---
                let instruction = if in_decoder_speakers {
                    prompt::NEMOTRON_TRANSCRIBE_DIARIZE_PROMPT
                } else {
                    prompt::NEMOTRON_TRANSCRIBE_TRANSCRIBE_PROMPT
                };
                let plan = prompt::build_served_prompt(
                    &prepared.tokenizer,
                    instruction,
                    language.as_deref(),
                )?;
                let audio_rows = encoder_output.frames;
                let decode_text_token_ids =
                    |token_ids: &[u32]| -> Result<String, Seq2SeqGreedyDecodeError> {
                        prepared
                            .tokenizer
                            .decode_text_token_ids(token_ids)
                            .map_err(|error| Seq2SeqGreedyDecodeError::TokenizerDecodeFailed {
                                reason: error.to_string(),
                            })
                    };
                // One decode pass over the audio-spliced prompt. Each pass
                // prefills the resident session from scratch (it resets its
                // arenas at prefill), so a second pass re-decodes the SAME
                // encoder output under a different prompt: the reference's
                // one-shot empty-transcript fallback.
                let projections = encoder_output.projections;

                // Forced-alignment evidence for ladder round selection. The
                // CTC head projects only the round's target columns, so this
                // must re-run per candidate; it is still much cheaper than an
                // extra decoder pass. Returns (mean per-label path score,
                // aligned word count, failed-stream/unmatched count).
                let ev_frames = encoder_output.frames;
                let ev_speakers = super::encoder_graph::NEMOTRON_TRANSCRIBE_NUM_SPEAKERS;
                let mut ev_speaker_probs = vec![0.0f32; ev_frames * ev_speakers];
                for t in 0..ev_frames {
                    for spk in 0..ev_speakers {
                        ev_speaker_probs[t * ev_speakers + spk] =
                            encoder_output.speaker_probs[spk * ev_frames + t];
                    }
                }
                let ev_ctc_frames = ev_frames * CTC_UPSAMPLING;
                let ev_config = AlignmentConfig {
                    max_word_frames: 12 * CTC_UPSAMPLING,
                    ..Default::default()
                };
                let mut ctc_evidence = |text: &str| -> Option<(f32, usize, usize)> {
                    if !word_timestamps {
                        return None;
                    }
                    let transcript = clean_sot_transcript(text, &[]);
                    if transcript.is_empty() {
                        return None;
                    }
                    let targets = alignment_targets(
                        &prepared.ctc_spm,
                        &transcript,
                        CTC_BLANK_ID,
                        ev_ctc_frames,
                    )
                    .ok()?;
                    let logits = prepared
                        .ctc_head
                        .logits(&encoder_output.asr_states, ev_frames, &targets.columns)
                        .ok()?;
                    let _ = prepared.ctc_head.release_transient_compute_memory();
                    let emissions: Vec<f32> = logits.iter().map(|&v| bf16_round(v)).collect();
                    let inputs = AlignmentInputs {
                        log_probs: &emissions,
                        frames: ev_ctc_frames,
                        speaker_probs: Some(&ev_speaker_probs),
                        speaker_frames: ev_frames,
                        speakers: ev_speakers,
                        activity10: Some(&encoder_output.speaker_probs10),
                        label_frames: frames.min(8 * ev_frames),
                        label_speakers: ev_speakers,
                        label_seconds: 0.01,
                        duration: f64::from(audio_duration_seconds),
                        time_offset: 0.0,
                    };
                    let aligned = align_record(&targets, &inputs, &ev_config).ok()?;
                    // The Viterbi score accumulates a term on EVERY frame
                    // (mostly blank), so normalising by label count rewards
                    // short hallucinated texts. Per-frame normalisation keeps
                    // the denominator fixed across candidate texts.
                    let score = aligned.scores.iter().sum::<f32>() / ev_ctc_frames as f32;
                    let failed = aligned.failed_streams.len()
                        + aligned.refinement.unmatched_words.max(0) as usize;
                    Some((score, aligned.words.len(), failed))
                };

                let mut decode_pass = |plan: &prompt::NemotronTranscribePromptPlan,
                                       projections: &[f32],
                                       temperature: f32,
                                       sample_seed: u64|
                    -> Result<crate::models::seq2seq_greedy_decode::Seq2SeqGreedyDecodeResult,
                              NemotronTranscribeGgmlExecutorError> {
                    let decode_config = BuiltinSeq2SeqDecodePolicyConfigInput {
                        initial_prompt_tokens: plan.prompt_token_ids(audio_rows),
                        eot_token_id: NEMOTRON_TRANSCRIBE_EOT_TOKEN_ID,
                        vocab_size: 131_072,
                        // Match the KV topology's duration-scaled budget: the
                        // resident self-KV span is planned from
                        // `output_budget_for_seconds` for the envelope, so a
                        // larger driver cap would let a long decode walk off
                        // the end of the allocated KV positions.
                        max_generated_tokens: prompt::output_budget_for_seconds(
                            audio_duration_seconds,
                        ),
                    };
                    let prompt_len = decode_config.initial_prompt_tokens.len();
                    // The resident session resets its arenas itself at prefill.
                    let session = &mut prepared.decoder;
                    let mut step_executor = NemotronTranscribeResidentAudioDecodeStepExecutor::new(
                        session,
                        &prepared.embed_table,
                        plan.prefix_ids.clone(),
                        plan.suffix_ids.clone(),
                        projections.to_vec(),
                        2688,
                        prompt_len,
                        temperature,
                        sample_seed,
                    );
                    let decoded = run_builtin_seq2seq_decode_policy(
                        crate::arch::NEMOTRON_TRANSCRIBE_DECODE_POLICY_ID,
                        &decode_config,
                        &(),
                        None,
                        &mut step_executor,
                        &decode_text_token_ids,
                        |error: Seq2SeqGreedyDecodeError| error,
                        |error: Seq2SeqGreedyDecodeError| error,
                        map_registry_error,
                        &control,
                        decode_work_progress.as_ref(),
                        unstable_decode_text.as_ref(),
                    );
                    match decoded {
                        Ok(result) => Ok(result),
                        Err(Seq2SeqGreedyDecodeError::EotNotReachedBeforeMaxTokens {
                            generated_tokens,
                            generated_probabilities,
                            ..
                        }) => {
                            // Reaching the budget is a degraded decode, not a
                            // failed request: keep the generated prefix and
                            // report the shortfall as a budget truncation (the
                            // same salvage the other seq2seq families use).
                            let text = decode_text_token_ids(&generated_tokens).map_err(|error| {
                                NemotronTranscribeGgmlExecutorError::DecodeFailed {
                                    reason: error.to_string(),
                                }
                            })?;
                            Ok(crate::models::seq2seq_greedy_decode::Seq2SeqGreedyDecodeResult {
                                generated_tokens,
                                generated_probabilities,
                                text,
                                stop_reason:
                                    crate::models::seq2seq_greedy_decode::Seq2SeqGreedyDecodeStopReason::BudgetExhausted,
                                guard_trip_ngram_len: None,
                            })
                        }
                        Err(error) => Err(NemotronTranscribeGgmlExecutorError::DecodeFailed {
                            reason: error.to_string(),
                        }),
                    }
                };
                let mut result = decode_pass(&plan, &projections, 0.0, 0)?;
                // Strip SALM scaffolding before the emptiness check. A decode
                // that emitted only event markup carries no transcript, so it
                // must take the same recovery as a step-0 stop token; checking
                // before the strip left those slices empty with no fallback.
                let raw_text = result.text.trim();
                let mut text = strip_scaffolding_markup(raw_text, in_decoder_speakers);
                // A clean stop with no output is the model saying "nothing
                // here" (music, room tone): the fallback prompt would invent
                // a transcript. Recover only when the decode was cut short or
                // its entire output was scaffolding that stripped to empty.
                let needs_recovery = result.stop_reason.is_truncated()
                    || (!raw_text.is_empty() && text.is_empty());

                // Short CTC-scored recovery ladder. A sampled round replaces
                // the incumbent only if forced alignment says it fits the
                // SAME audio better (higher mean per-label path score) without
                // losing placed words or adding failed streams: unlike text
                // length, that signal cannot be gamed by fluent hallucination.
                let rungs = nemotron_temperature_ladder();
                if needs_recovery && !rungs.is_empty() && !in_decoder_speakers {
                    let incumbent = ctc_evidence(&text);
                    eprintln!(
                        "openasr_nemotron_transcribe stage=temperature_ladder event=ladder_start stop_reason={:?} incumbent={:?}",
                        result.stop_reason,
                        incumbent
                    );
                    // Evaluate every rung before choosing: single-word
                    // fillers can win an early race on a noisy score, and a
                    // later rung may carry the real recovery.
                    let mut best_round: Option<(
                        crate::models::seq2seq_greedy_decode::Seq2SeqGreedyDecodeResult,
                        String,
                        (f32, usize, usize),
                    )> = None;
                    for (i, &temperature) in rungs.iter().enumerate() {
                        let seed = NEMOTRON_LADDER_BASE_SEED
                            .wrapping_add(i as u64)
                            .wrapping_mul(0x2545_F491_4F6C_DD1D);
                        let round = match decode_pass(&plan, &projections, temperature, seed) {
                            Ok(round) if !round.stop_reason.is_truncated() => round,
                            Ok(round) => {
                                eprintln!(
                                    "openasr_nemotron_transcribe stage=temperature_ladder event=round_skipped round={} temperature={} reason=truncated stop_reason={:?}",
                                    i + 2,
                                    temperature,
                                    round.stop_reason
                                );
                                continue;
                            }
                            Err(error) => {
                                eprintln!(
                                    "openasr_nemotron_transcribe stage=temperature_ladder event=round_skipped round={} temperature={} reason=error error={error}",
                                    i + 2,
                                    temperature
                                );
                                continue;
                            }
                        };
                        let round_text = strip_scaffolding_markup(round.text.trim(), false);
                        if round_text.is_empty() {
                            continue;
                        }
                        let Some(evidence) = ctc_evidence(&round_text) else {
                            continue;
                        };
                        let (score, words, failed) = evidence;
                        // A one-word filler ("Yeah.", "electronic", "Um")
                        // is never a recovery of a multi-word dropped phrase.
                        let eligible = score >= NEMOTRON_MIN_ADOPT_CTC_SCORE
                            && words >= 3
                            && match &incumbent {
                                Some((inc_score, inc_words, inc_failed)) => {
                                    score > *inc_score + NEMOTRON_LADDER_SCORE_MARGIN
                                        && words >= *inc_words
                                        && failed <= *inc_failed
                                }
                                None => true,
                            };
                        let better = eligible
                            && best_round
                                .as_ref()
                                .is_none_or(|(_, _, (best_score, _, _))| score > *best_score);
                        eprintln!(
                            "openasr_nemotron_transcribe stage=temperature_ladder event=round_evidence round={} temperature={} score={:.3} words={} failed={} eligible={} text={:?}",
                            i + 2,
                            temperature,
                            score,
                            words,
                            failed,
                            eligible,
                            round_text
                        );
                        if better {
                            best_round = Some((round, round_text, evidence));
                        }
                    }
                    if let Some((round, round_text, evidence)) = best_round {
                        eprintln!(
                            "openasr_nemotron_transcribe stage=temperature_ladder event=round_adopted score={:.3} words={} text={:?}",
                            evidence.0,
                            evidence.1,
                            round_text
                        );
                        result = round;
                        text = round_text;
                    }
                }

                if text.is_empty() && !in_decoder_speakers {
                    if !needs_recovery {
                        // Clean stop token with no output: respect the model's
                        // "nothing here" rather than inventing music/noise text.
                        eprintln!(
                            "openasr_nemotron_transcribe stage=decode event=empty_transcript_fallback status=skipped_clean_stop"
                        );
                    } else {
                        // The reference's one-shot empty-transcript fallback:
                        // the decode was cut short or emitted only scaffolding,
                        // so re-decode the same encoder output once under the
                        // music/noise-tolerant prompt. At most once: if this
                        // also comes back empty the result stands, now reported
                        // as a stop-token-before-content truncation.
                        eprintln!(
                            "openasr_nemotron_transcribe stage=decode event=empty_transcript_fallback status=fired"
                        );
                        let fallback_plan = prompt::build_served_prompt(
                            &prepared.tokenizer,
                            prompt::NEMOTRON_TRANSCRIBE_FALLBACK_PROMPT,
                            language.as_deref(),
                        )?;
                        let fallback_result = decode_pass(&fallback_plan, &projections, 0.0, 0)?;
                        let fallback_text =
                            strip_scaffolding_markup(fallback_result.text.trim(), in_decoder_speakers);
                        // Same acoustic-evidence test the ladder uses: a clean
                        // decode of music/noise can be fluent but its words do
                        // not sit on voiced frames. Without evidence (no word
                        // timestamps requested) keep the historical behavior.
                        let accept = match ctc_evidence(&fallback_text) {
                            Some((score, words, failed)) => {
                                let ok = score >= NEMOTRON_MIN_ADOPT_CTC_SCORE
                                    && words >= 3
                                    && failed <= 3;
                                if !ok {
                                    eprintln!(
                                        "openasr_nemotron_transcribe stage=decode event=empty_transcript_fallback status=rejected_evidence score={score:.3} words={words} failed={failed} text={fallback_text:?}"
                                    );
                                }
                                ok
                            }
                            None => !fallback_text.is_empty(),
                        };
                        if accept && !fallback_text.is_empty() {
                            result = fallback_result;
                            text = fallback_text;
                        }
                    }
                }

                // --- CTC word timestamps (transcript-independent head + host aligner) ---
                // Alignment is a best-effort enhancement: the reference returns
                // the transcript with timestamps unavailable when the head or
                // the aligner fails (a long/recovering decode can outgrow the
                // CTC grid), so a failure here must never fail the transcription.
                let aligned_words = (|| -> Result<
                    Vec<crate::api::backend::WordTimestamp>,
                    NemotronTranscribeGgmlExecutorError,
                > {
                    if !word_timestamps {
                        return Ok(Vec::new());
                    }
                    let transcript = clean_sot_transcript(&text, &[]);
                    if transcript.is_empty() {
                        return Ok(Vec::new());
                    }
                    // The encoder returns speaker probabilities speaker-major
                    // ([speaker * frames + t]); the aligner expects frame-major
                    // ([t * speakers + speaker]).
                    let enc_frames = encoder_output.frames;
                    let speakers = super::encoder_graph::NEMOTRON_TRANSCRIBE_NUM_SPEAKERS;
                    let mut speaker_probs = vec![0.0f32; enc_frames * speakers];
                    for t in 0..enc_frames {
                        for spk in 0..speakers {
                            speaker_probs[t * speakers + spk] =
                                encoder_output.speaker_probs[spk * enc_frames + t];
                        }
                    }
                    let ctc_frames = encoder_output.frames * CTC_UPSAMPLING;
                    let targets = alignment_targets(
                        &prepared.ctc_spm,
                        &transcript,
                        CTC_BLANK_ID,
                        ctc_frames,
                    )
                    .map_err(|reason| NemotronTranscribeGgmlExecutorError::WordTimestampsFailed {
                        reason: format!("CTC alignment targets: {reason}"),
                    })?;
                    let logits = prepared.ctc_head.logits(
                        &encoder_output.asr_states,
                        encoder_output.frames,
                        &targets.columns,
                    )
                    .map_err(|error| NemotronTranscribeGgmlExecutorError::WordTimestampsFailed {
                        reason: error.to_string(),
                    })?;
                    prepared.ctc_head.release_transient_compute_memory().map_err(|error| {
                        NemotronTranscribeGgmlExecutorError::WordTimestampsFailed {
                            reason: error.to_string(),
                        }
                    })?;
                    let emissions: Vec<f32> = logits.iter().map(|&v| bf16_round(v)).collect();
                    let config = AlignmentConfig {
                        max_word_frames: 12 * CTC_UPSAMPLING,
                        ..Default::default()
                    };
                    let inputs = AlignmentInputs {
                        log_probs: &emissions,
                        frames: ctc_frames,
                        speaker_probs: Some(&speaker_probs),
                        speaker_frames: encoder_output.frames,
                        speakers: super::encoder_graph::NEMOTRON_TRANSCRIBE_NUM_SPEAKERS,
                        activity10: Some(&encoder_output.speaker_probs10),
                        label_frames: frames.min(8 * encoder_output.frames),
                        label_speakers: super::encoder_graph::NEMOTRON_TRANSCRIBE_NUM_SPEAKERS,
                        label_seconds: 0.01,
                        duration: f64::from(audio_duration_seconds),
                        time_offset: 0.0,
                    };
                    let aligned = align_record(&targets, &inputs, &config).map_err(|reason| {
                        NemotronTranscribeGgmlExecutorError::WordTimestampsFailed {
                            reason: format!("CTC alignment: {reason}"),
                        }
                    })?;
                    Ok(public_words(&aligned.words)
                        .into_iter()
                        .map(|row| {
                            let unit = &targets.words[row.word];
                            crate::api::backend::WordTimestamp {
                                word: if unit.display.is_empty() {
                                    unit.word.clone()
                                } else {
                                    unit.display.clone()
                                },
                                start: row.start as f32,
                                end: row.end as f32,
                                confidence: None,
                            }
                        })
                        .collect())
                })();
                let words = match aligned_words {
                    Ok(words) => words,
                    Err(error) => {
                        eprintln!(
                            "openasr_nemotron_transcribe stage=word_timestamps event=alignment_unavailable status=degraded reason={error}"
                        );
                        Vec::new()
                    }
                };

                Ok(GgmlAsrExecutionResult {
                    transcription: Transcription {
                        truncated_decodes: Vec::new(),
                        unnamed_speakers: Vec::new(),
                        text: text.clone(),
                        segments: vec![Segment {
                            start: 0.0,
                            end: audio_duration_seconds,
                            text,
                            speaker: None,
                            speaker_label: None,
                            speaker_person_id: None,
                            speaker_snapshot_label: None,
                            words,
                        }],
                        longform: None,
                        language: None,
                        ..Default::default()
                    },
                    carry_context: None,
                    decode_truncation: result.stop_reason.into_decode_truncation(None),
                })
            })
            .map_err(Self::map_unified_actor_error)?
    }
}

impl GgmlAsrViewExecutor for NemotronTranscribeGgmlExecutor {
    fn evict_prepared_runtime_content_id(&self, pack_content_id: &str) {
        // Drop the idle resident actors for this pack (any lane/span bucket).
        self.unified_runtimes
            .evict_where(|key| key.content.pack_content_id == pack_content_id);
    }

    fn executor_id(&self) -> &'static str {
        NEMOTRON_TRANSCRIBE_EXECUTOR_ID
    }

    fn supports_phrase_bias(&self) -> bool {
        // No phrase-bias plumbing for this family (Unsupported in the
        // descriptor); a hotword set must be silently ignored rather than
        // falsely acknowledged.
        false
    }

    fn decoder_state_contract(
        &self,
        _selected_family: &crate::GgmlFamilyAdapterDescriptor,
    ) -> Result<crate::models::ggml_asr_executor::GgmlAsrDecoderStateContract, GgmlAsrExecutionError>
    {
        Ok(
            crate::models::ggml_asr_executor::GgmlAsrDecoderStateContract::planned(
                super::capacity::plan_nemotron_transcribe_decoder_state,
                super::capacity::NEMOTRON_TRANSCRIBE_DECODER_STATE_STREAMS,
            ),
        )
    }

    fn execute_view(
        &self,
        request: &GgmlAsrExecutionViewRequest,
    ) -> Result<GgmlAsrExecutionResult, GgmlAsrExecutionError> {
        self.execute_inner(request)
            .map_err(|error| GgmlAsrExecutionError::ExecutorFailed {
                executor_id: GgmlAsrViewExecutor::executor_id(self),
                adapter_id: request.selected_family.adapter_id,
                reason: error.to_string(),
            })
    }

    fn unload_idle_state(&self) {
        // Drop every idle resident actor (all pack/lane/span buckets).
        self.unified_runtimes.clear();
    }
}

impl NemotronTranscribeGgmlExecutor {
    fn execute_streaming(
        &self,
        request: &GgmlAsrExecutionViewRequest,
    ) -> Result<GgmlAsrExecutionResult, GgmlAsrExecutionError> {
        self.execute_inner(request)
            .map_err(|error| GgmlAsrExecutionError::ExecutorFailed {
                executor_id: GgmlAsrViewExecutor::executor_id(self),
                adapter_id: request.selected_family.adapter_id,
                reason: error.to_string(),
            })
    }
}

const NEMOTRON_TRANSCRIBE_STREAMING_EXECUTOR_ID: &str =
    "nemotron-transcribe-ggml-snapshot-streaming-executor-v1";

impl crate::models::ggml_asr_executor::GgmlAsrStreamingExecutor for NemotronTranscribeGgmlExecutor {
    fn executor_id(&self) -> &'static str {
        NEMOTRON_TRANSCRIBE_STREAMING_EXECUTOR_ID
    }

    fn start_streaming_session(
        &self,
        request: &crate::models::ggml_asr_executor::GgmlAsrStreamingSessionRequest,
    ) -> Result<Box<dyn crate::NativeAsrSession>, GgmlAsrExecutionError> {
        crate::models::incremental_streaming_driver::build_seq2seq_streaming_session(
            self.clone(),
            NEMOTRON_TRANSCRIBE_STREAMING_EXECUTOR_ID,
            NEMOTRON_TRANSCRIBE_GGML_ADAPTER_ID,
            "nemotron-transcribe",
            request,
            crate::models::incremental_streaming_driver::STREAMING_PARTIAL_TUNING_HEAVY_SEQ2SEQ,
            NemotronTranscribeGgmlExecutor::execute_streaming,
        )
    }

    fn unload_idle_state(&self) {
        // Drop every idle resident actor (all pack/lane/span buckets).
        self.unified_runtimes.clear();
    }
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};
    use std::time::{Duration, Instant};

    use crate::arch::builtin_adapter_descriptor;
    use crate::ggml_runtime::install_request_backend_override;
    use crate::models::ggml_asr_executor::{
        GgmlAsrBackendPreference, GgmlAsrDecoderState, GgmlAsrDecoderStatePlanningInput,
        GgmlAsrPreparedAudioView,
    };

    use super::*;

    #[test]
    fn scaffolding_strip_removes_event_and_metadata_tags() {
        let text = "<ev:audio> Hello, <ev:ev-step end=\"true\" start=\"true\" type=\"text\"> \
                    world </ev:audio> <voice> again </ev:segment>";
        assert_eq!(strip_scaffolding_markup(text, false), "Hello, world again");
    }

    #[test]
    fn scaffolding_strip_handles_misspellings_and_split_tags() {
        let text = "<ev:asistant> Hi <ev:window-message>Two minute</ev:window-message> \
                    </ev:asistant> there";
        assert_eq!(strip_scaffolding_markup(text, false), "Hi Two minute there");
    }

    #[test]
    fn scaffolding_strip_preserves_speaker_tags_only_for_diarize() {
        let text = "<spk:0> Hello <ev:voice> world </spk:0>";
        assert_eq!(
            strip_scaffolding_markup(text, true),
            "<spk:0> Hello world </spk:0>"
        );
        assert_eq!(strip_scaffolding_markup(text, false), "Hello world");
    }

    #[test]
    fn scaffolding_strip_leaves_clean_transcripts_untouched() {
        let clean = "And so my fellow Americans, ask not what your country can do for you.";
        assert_eq!(strip_scaffolding_markup(clean, false), clean);
    }

    #[test]
    fn scaffolding_only_decode_is_an_empty_transcript() {
        // A decode that emitted nothing but event markup is empty content. The
        // executor decides the empty-transcript fallback on the stripped text,
        // so this shape takes the same recovery as a step-zero stop token.
        let raw = "<ev:audio> <ev:ev-step end=\"true\" type=\"text\"> </ev:ev-step> </ev:audio>";
        assert_eq!(strip_scaffolding_markup(raw, false), "");
    }

    /// Real pack for the opt-in scaling probe; never committed (a 15-37 GB
    /// weight-bearing fixture). Same skip-if-absent convention as the other
    /// families' dev-pack tests.
    fn dev_pack_path() -> Option<PathBuf> {
        match crate::testing::external_test_fixture_path(
            "OPENASR_NEMOTRON_TRANSCRIBE_PACK",
            "nemotron-transcribe .oasr pack",
        ) {
            Ok(path) => Some(path),
            Err(skip) => {
                eprintln!("skipping: {skip}");
                None
            }
        }
    }

    fn jfk_samples() -> Vec<f32> {
        crate::api::audio_io::load_wav_16khz_mono_f32_v0(
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/jfk.wav"),
            "nemotron slice scaling probe",
            "nemotron slice scaling probe",
        )
        .expect("load jfk.wav fixture")
    }

    /// Speech-like filler: loop the JFK fixture so the clip clears both the
    /// near-digital-silence gate and the no-voice speaker-activity gate,
    /// otherwise the executor would short-circuit before the encoder runs.
    fn looped_samples(seconds: f32, base: &[f32]) -> Vec<f32> {
        let target = (seconds * 16_000.0).ceil() as usize;
        let mut samples = Vec::with_capacity(target);
        while samples.len() < target {
            samples
                .extend_from_slice(&base[..target.saturating_sub(samples.len()).min(base.len())]);
        }
        samples
    }

    fn plan_test_request_decoder_state(
        executor: &NemotronTranscribeGgmlExecutor,
        request: &mut GgmlAsrExecutionViewRequest<'_>,
    ) {
        let decoder_state = {
            let planning_input = GgmlAsrDecoderStatePlanningInput::for_offline_view_request(
                request.runtime_source_preflight(),
                &request.prepared_audio,
                &request.request_options,
                request.resolved_runtime.backend(),
            )
            .expect("build nemotron decoder-state planning input");
            executor
                .decoder_state_contract(&request.selected_family)
                .expect("load nemotron decoder-state contract")
                .plan(&planning_input)
                .expect("plan nemotron decoder state")
        };
        request.decoder_state = decoder_state;
    }

    /// One direct invocation of the given length; the longform slicer is
    /// deliberately bypassed so the measurement is the cost of one slice,
    /// which is exactly the window `NEMOTRON_TRANSCRIBE_LONGFORM_SLICE_SECONDS`
    /// is meant to bound.
    fn transcribe_single_pass(
        executor: &NemotronTranscribeGgmlExecutor,
        samples: Vec<f32>,
        pack_path: &Path,
    ) -> Option<(String, std::time::Duration, f32)> {
        let runtime_source_preflight =
            crate::models::runtime_preflight::load_runtime_source_metadata_and_tensor_index(
                pack_path,
            )
            .expect("nemotron runtime must pass preflight");
        // `Accelerated` would try a 14.5 GiB device allocation here: the bare
        // harness lacks the production exact-route lane that shares one weight
        // context between encoder and decoder, so it OOMs. Keep `Auto` and
        // report the resolved backend so a CPU-lane run cannot be mistaken for
        // a VRAM measurement. For the real GPU number use the CLI with
        // OPENASR_MEASURE_VRAM=1.
        let backend_preference = GgmlAsrBackendPreference::Auto;
        // A direct execute_view bypasses dispatch, so the backend preference
        // must be installed here for the thread-local readers (same as the
        // other families' real-pack dev tests).
        let _backend_override_guard =
            install_request_backend_override(backend_preference.request_backend_override());
        let resolved_runtime = crate::ggml_runtime::ResolvedFamilyRuntimeInput::resolve(
            backend_preference.request_backend_override(),
            crate::arch::family_auto_gpu_policy_for_model_architecture(
                crate::arch::NEMOTRON_TRANSCRIBE_GGML_ARCHITECTURE_ID,
            ),
        );
        eprintln!(
            "nemotron slice probe: resolved_backend={:?}",
            resolved_runtime.backend()
        );
        let audio_duration_seconds = samples.len() as f32 / 16_000.0;
        let mut request = GgmlAsrExecutionViewRequest {
            execution_services:
                crate::models::native_execution_services::test_native_execution_services(),
            decoder_state: GgmlAsrDecoderState::NoPersistentState,
            verified_pack: crate::models::runtime_preflight::verified_pack_from_preflight_for_test(
                runtime_source_preflight,
                crate::arch::NEMOTRON_TRANSCRIBE_GGML_ARCHITECTURE_ID,
            ),
            selected_family: builtin_adapter_descriptor(
                crate::arch::NEMOTRON_TRANSCRIBE_GGML_ARCHITECTURE_ID,
            ),
            prepared_audio: GgmlAsrPreparedAudioView::mono_16khz(samples),
            request_options: Default::default(),
            backend_preference,
            resolved_runtime,
            execution_context: std::sync::Arc::new(crate::RequestExecutionContext::uncancellable(
                "nemotron slice scaling probe",
            )),
        };
        plan_test_request_decoder_state(executor, &mut request);

        let started_at = Instant::now();
        let result = executor
            .execute_view(&request)
            .expect("nemotron transcribe");
        Some((
            result.transcription.text,
            started_at.elapsed(),
            audio_duration_seconds,
        ))
    }

    /// Slice-length scaling probe: reports a single-pass transcription's RTF,
    /// peak RSS and sampled peak VRAM at each candidate window. This is the
    /// evidence owed by `NEMOTRON_TRANSCRIBE_LONGFORM_SLICE_SECONDS` (a
    /// measured per-family peak-activation coefficient); the assertions only
    /// check the call completed (an empty transcript is reported, not failed,
    /// since the gates can legitimately produce one). Peak RSS is a process
    /// high-water mark (run one candidate per
    /// process for an isolated reading); VRAM is sampled per call and is
    /// device-wide, so it is only attributable to this process on an idle
    /// GPU. VRAM reading requires building the test with `--features hip`.
    /// `OPENASR_NEMOTRON_SLICE_PROBE_WINDOWS` overrides the comma-separated
    /// candidate windows (seconds) for a quick smoke run.
    #[test]
    #[ignore = "host-local: needs OPENASR_NEMOTRON_TRANSCRIBE_PACK + fixtures/jfk.wav; \
                measures per-slice activation headroom to pin \
                NEMOTRON_TRANSCRIBE_LONGFORM_SLICE_SECONDS; run via --run-ignored on the target class"]
    fn longform_slice_scaling_probe() {
        let Some(pack_path) = dev_pack_path() else {
            return;
        };
        let base = jfk_samples();
        let executor = NemotronTranscribeGgmlExecutor::default();
        let windows: Vec<f32> = match std::env::var("OPENASR_NEMOTRON_SLICE_PROBE_WINDOWS") {
            Ok(value) => value
                .split(',')
                .filter_map(|part| part.trim().parse::<f32>().ok())
                .filter(|seconds| seconds.is_finite() && *seconds > 0.0)
                .collect(),
            Err(_) => vec![120.0, 300.0, 600.0, 1200.0],
        };
        if windows.is_empty() {
            eprintln!("skipping: no positive windows in OPENASR_NEMOTRON_SLICE_PROBE_WINDOWS");
            return;
        }
        for window_seconds in windows {
            let samples = looped_samples(window_seconds + 1.0, &base);
            // VRAM can fall between windows (unlike the RSS high-water mark),
            // so each candidate gets its own per-call sampled peak.
            let vram_sampler = crate::metrics::VramPeakSampler::start(Duration::from_millis(50));
            let outcome = transcribe_single_pass(&executor, samples, &pack_path);
            let peak_vram_bytes = vram_sampler.stop();
            let Some((text, elapsed, audio_duration_seconds)) = outcome else {
                return;
            };
            // An empty transcript is a legitimate outcome (silence / no-voice
            // gate) and does not invalidate the memory measurement.
            if text.trim().is_empty() {
                eprintln!(
                    "nemotron slice probe: window={window_seconds:.0}s produced no transcript"
                );
            }
            let rtf = elapsed.as_secs_f32() / audio_duration_seconds.max(0.001);
            let peak_rss_mb = crate::metrics::peak_rss_bytes()
                .map(|bytes| bytes as f64 / 1.0e6)
                .unwrap_or(0.0);
            let peak_vram_mb = peak_vram_bytes
                .map(|bytes| bytes as f64 / 1.0e6)
                .unwrap_or(0.0);
            eprintln!(
                "nemotron slice probe: window={window_seconds:.0}s \
                 audio={audio_duration_seconds:.2}s elapsed={elapsed:?} \
                 RTF={rtf:.3} peak_rss={peak_rss_mb:.0}MB peak_vram={peak_vram_mb:.0}MB"
            );
        }
    }
}
