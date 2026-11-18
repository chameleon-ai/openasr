//! Nemotron-transcribe dedicated executor — currently a fail-closed stub.
//!
//! This type exists so the family can be registered in the canonical
//! architecture inventory (which is what lets the published `.oasr` packs pass
//! the production `PackVerifier` and lets the release pipeline proceed). Its
//! execution paths all fail closed with "not yet implemented"; no audio is
//! read, no graph is built, and no output is produced. When the real runtime
//! lands, this file becomes the executor proper and the error paths become
//! the real ones.

use crate::NativeAsrSession;
use crate::arch::NEMOTRON_TRANSCRIBE_EXECUTOR_COMPONENT_ID;
use crate::capacity::topology::{DecoderStatePlan, StateKind};
use crate::models::ggml_asr_executor::{
    GgmlAsrDecoderStateContract, GgmlAsrDecoderStatePlanningError,
    GgmlAsrDecoderStatePlanningInput, GgmlAsrDecoderStateStreamContract, GgmlAsrExecutionError,
    GgmlAsrExecutionResult, GgmlAsrExecutionViewRequest, GgmlAsrStreamingExecutor,
    GgmlAsrStreamingSessionRequest, GgmlAsrViewExecutor,
};
use crate::models::ggml_family_adapter::GgmlFamilyAdapterDescriptor;
use crate::models::native_execution_services::ExecutionLaneKey;
use crate::models::prepared_runtime_cache::{
    HostNeutralPreparedRuntime, PreparedRuntimeCache, PreparedRuntimeQuoteBuilder,
    PreparedRuntimeQuoteContext, SystemMemoryMaterialization,
};
use crate::models::system_memory_owner::{SystemMemoryAllocationQuote, SystemMemoryOwnerError};

pub(crate) const NEMOTRON_TRANSCRIBE_EXECUTOR_ID: &str = NEMOTRON_TRANSCRIBE_EXECUTOR_COMPONENT_ID;
pub(crate) const NEMOTRON_TRANSCRIBE_STREAMING_EXECUTOR_ID: &str =
    "nemotron-transcribe-ggml-not-yet-implemented-streaming-executor-v1";

const NEMOTRON_TRANSCRIBE_SELF_KV_STATE_ID: &str = "nemotron-transcribe.decoder.self_kv";
const NEMOTRON_TRANSCRIBE_DECODER_STATE_STREAMS: &[GgmlAsrDecoderStateStreamContract] =
    &[GgmlAsrDecoderStateStreamContract::new(
        NEMOTRON_TRANSCRIBE_SELF_KV_STATE_ID,
        StateKind::SelfAttentionKv,
    )];

const NOT_YET_IMPLEMENTED_REASON: &str = "model 'nemotron-3.5-transcribe-30b-a4b' is not yet implemented in this build of OpenASR: the nemotron-transcribe runtime (PE encoder, Mamba-2/MoE decoder, Sortformer speaker branch, CTC timestamp head) has not landed, and the published packs are structurally verified but cannot be executed yet";

pub(crate) fn not_yet_implemented_error() -> GgmlAsrExecutionError {
    GgmlAsrExecutionError::ExecutorFailed {
        executor_id: NEMOTRON_TRANSCRIBE_EXECUTOR_ID,
        adapter_id: crate::arch::NEMOTRON_TRANSCRIBE_GGML_ADAPTER_ID,
        reason: NOT_YET_IMPLEMENTED_REASON.to_string(),
    }
}

/// Placeholder for the family's future resident runtime. The staged build
/// materializes no prepared state (execution fails closed before any is
/// needed); the cache and its content-id eviction exist so the eviction seam
/// behaves exactly like the implemented families once a runtime lands.
#[derive(Default)]
struct NemotronTranscribePreparedRuntime {
    _reserved: (),
}

impl SystemMemoryMaterialization for NemotronTranscribePreparedRuntime {
    fn retained_system_memory_bytes(&self) -> Result<u64, String> {
        // Nothing is retained by a staged stub; this is the truthful quote.
        Ok(0)
    }
}

impl HostNeutralPreparedRuntime for NemotronTranscribePreparedRuntime {
    fn system_memory_quote(
        _context: PreparedRuntimeQuoteContext<'_>,
        pack_content_id: &str,
    ) -> Result<SystemMemoryAllocationQuote, SystemMemoryOwnerError> {
        PreparedRuntimeQuoteBuilder::new::<Self>(pack_content_id).finish()
    }
}

fn fail_closed_planner(
    _input: &GgmlAsrDecoderStatePlanningInput<'_>,
) -> Result<DecoderStatePlan, GgmlAsrDecoderStatePlanningError> {
    // The staged build refuses to derive capacity plans: there is no runtime
    // to plan for yet, and a fabricated plan would be worse than none.
    Err(GgmlAsrDecoderStatePlanningError::MetadataUnavailable {
        family: "nemotron-transcribe",
        reason: "model 'nemotron-3.5-transcribe-30b-a4b' is not yet implemented in this build of OpenASR: the nemotron-transcribe runtime (PE encoder, Mamba-2/MoE decoder, Sortformer speaker branch, CTC timestamp head) has not landed, and the published packs are structurally verified but cannot be executed yet"
            .to_string(),
    })
}

#[derive(Default)]
pub(crate) struct NemotronTranscribeGgmlExecutor {
    prepared_runtime: PreparedRuntimeCache<NemotronTranscribePreparedRuntime>,
}

impl GgmlAsrViewExecutor for NemotronTranscribeGgmlExecutor {
    fn executor_id(&self) -> &'static str {
        NEMOTRON_TRANSCRIBE_EXECUTOR_ID
    }

    fn adapter_binding_strategy_for(
        &self,
        _selected_family: &GgmlFamilyAdapterDescriptor,
    ) -> Result<crate::models::ggml_family_adapter::GgmlAdapterBindingStrategy, GgmlAsrExecutionError>
    {
        Ok(crate::models::ggml_family_adapter::GgmlAdapterBindingStrategy::Unsupported)
    }

    fn supports_phrase_bias(&self) -> bool {
        false
    }

    fn decoder_state_contract(
        &self,
        _selected_family: &GgmlFamilyAdapterDescriptor,
    ) -> Result<GgmlAsrDecoderStateContract, GgmlAsrExecutionError> {
        Ok(GgmlAsrDecoderStateContract::planned(
            fail_closed_planner,
            NEMOTRON_TRANSCRIBE_DECODER_STATE_STREAMS,
        ))
    }

    fn execute_view(
        &self,
        request: &GgmlAsrExecutionViewRequest<'_>,
    ) -> Result<GgmlAsrExecutionResult, GgmlAsrExecutionError> {
        // Lane identity is resolved but never consumed: the resident runtime
        // that would be keyed on it does not exist yet. Referenced so the
        // eviction seam stays lane-aware the moment a runtime lands.
        let _lane_key: Option<&ExecutionLaneKey> =
            request.execution_context.native_execution_lane();
        let _ = _lane_key;
        Err(not_yet_implemented_error())
    }

    fn evict_prepared_runtime_content_id(&self, pack_content_id: &str) {
        // Eviction from the (empty) prepared cache; correct-by-construction
        // and free of false positives — a staged build never admits an entry.
        self.prepared_runtime.evict_content_id(pack_content_id);
    }

    fn unload_idle_state(&self) {
        self.prepared_runtime.clear();
    }
}

impl GgmlAsrStreamingExecutor for NemotronTranscribeGgmlExecutor {
    fn executor_id(&self) -> &'static str {
        NEMOTRON_TRANSCRIBE_STREAMING_EXECUTOR_ID
    }

    fn start_streaming_session(
        &self,
        _request: &GgmlAsrStreamingSessionRequest,
    ) -> Result<Box<dyn NativeAsrSession>, GgmlAsrExecutionError> {
        Err(not_yet_implemented_error())
    }

    fn unload_idle_state(&self) {
        self.prepared_runtime.clear();
    }
}
