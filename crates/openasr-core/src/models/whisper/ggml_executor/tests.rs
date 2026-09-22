use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use super::*;
use crate::arch::builtin_adapter_descriptor;
use crate::models::whisper::ggml_tensor_binding::WhisperGgufTensorSlot;
use crate::models::whisper::tokenizer::{
    TOKENIZER_GGML_EOT_TOKEN_ID_KEY, TOKENIZER_GGML_MERGES_KEY, TOKENIZER_GGML_MODEL_KEY,
    TOKENIZER_GGML_MODEL_VALUE_GPT2, TOKENIZER_GGML_NO_TIMESTAMPS_TOKEN_ID_KEY,
    TOKENIZER_GGML_SOT_TOKEN_ID_KEY, TOKENIZER_GGML_SPECIAL_TOKEN_IDS_KEY,
    TOKENIZER_GGML_TOKENS_KEY, TOKENIZER_GGML_TRANSCRIBE_TOKEN_ID_KEY,
};
use crate::testing::{
    TinyGgufFixtureSpec, WhisperExecutionFailureStage,
    assert_tiny_whisper_encoder_output_shape_and_finite, classify_whisper_execution_failure_stage,
    tiny_whisper_encoder_smoke_prepared_audio, tiny_whisper_encoder_smoke_real_mel_input,
    tiny_whisper_encoder_smoke_shape_for_default_fixture, write_tiny_gguf_runtime_source,
};
use crate::{
    GgufMetadata, GgufMetadataValue, read_gguf_metadata_from_runtime_source,
    validate_ggml_runtime_source_path,
};
use sha2::{Digest, Sha256};

const GOLDEN_DIFF_TINY_WHISPER_ENCODER_PRELUDE_SHA256: &str =
    "1249053500204b8b7e4b08b54e42a7d557e3a1d507a51ff709953ef003c2e826";
// Reference-platform (macOS aarch64) bit-identity golden. ggml CPU float
// compute is not bit-identical across architectures (libm / SIMD / reduction
// order differ), so the decoder-step logit hash is pinned to the capture
// platform; the cross-platform encoder-prelude golden above still gates every
// target (incl. the Linux CI runner).
#[cfg(target_os = "macos")]
const GOLDEN_DIFF_TINY_WHISPER_DECODER_STEP_LOGITS_SHA256: &str =
    "562f79a316fb538274ec4eb03a8ee28407dd5a8f258029d44fe6c1fba5882561";

fn exactly_addressable_preference(provider: ExecutionProvider) -> RequestBackendPreference {
    RequestBackendPreference::Exact(crate::device::execution_route::ResolvedExecutionRoute {
        provider,
        stable_id: format!("{}0", provider.as_str()),
        registry_ordinal: 0,
        kind: crate::device::execution_route::RouteDeviceKind::Accelerated,
        addressability: crate::device::execution_route::DeviceAddressability::ExactlyAddressable {
            physical_key: crate::device::execution_route::PhysicalResourceKey::new("0000:01:00.0")
                .expect("physical key"),
        },
    })
}

#[test]
fn unified_offline_path_checkouts_combined_owner_instead_of_encoder_only_actor() {
    assert!(
        whisper_should_checkout_unified_gpu_owner(false, false, true),
        "offline unified GPU must skip the encoder-only actor so prelude cannot install a second TLS backend"
    );
    assert!(!whisper_should_checkout_unified_gpu_owner(
        true, false, true
    ));
    assert!(!whisper_should_checkout_unified_gpu_owner(
        false, true, true
    ));
    assert!(!whisper_should_checkout_unified_gpu_owner(
        false, false, false
    ));
}

#[test]
fn unified_owner_is_limited_to_exact_direct_cuda_hip_vulkan_full_device() {
    let direct_gpu = GgmlCpuGraphConfig {
        backend: GgmlCpuGraphBackend::Gpu,
        use_scheduler: false,
        ..GgmlCpuGraphConfig::conservative_default()
    };
    let medium_geometry = WhisperUnifiedRuntimeGeometry {
        decoder_layers: 24,
        decoder_hidden_size: 1024,
        tensor_storage_bytes: Some(2 * 1024 * 1024 * 1024),
    };
    let large_geometry = WhisperUnifiedRuntimeGeometry {
        decoder_layers: 32,
        decoder_hidden_size: 1280,
        tensor_storage_bytes: Some(2 * 1024 * 1024 * 1024),
    };
    let vulkan = exactly_addressable_preference(ExecutionProvider::Vulkan);
    assert!(whisper_unified_runtime_enabled_with_override(
        GgmlCpuGraphBackend::Gpu,
        Some(&vulkan),
        Some(ExecutionPlacement::FullDevice),
        direct_gpu,
        direct_gpu,
        medium_geometry,
        false,
        None,
        None,
    ));
    let cuda = exactly_addressable_preference(ExecutionProvider::Cuda);
    let hip = exactly_addressable_preference(ExecutionProvider::Hip);
    for capture_gpu in [&cuda, &hip] {
        assert!(!whisper_unified_runtime_enabled_with_override(
            GgmlCpuGraphBackend::Gpu,
            Some(capture_gpu),
            Some(ExecutionPlacement::FullDevice),
            direct_gpu,
            direct_gpu,
            medium_geometry,
            false,
            None,
            None,
        ));
        assert!(whisper_unified_runtime_enabled_with_override(
            GgmlCpuGraphBackend::Gpu,
            Some(capture_gpu),
            Some(ExecutionPlacement::FullDevice),
            direct_gpu,
            direct_gpu,
            medium_geometry,
            true,
            None,
            None,
        ));
        assert!(whisper_unified_runtime_enabled_with_override(
            GgmlCpuGraphBackend::Gpu,
            Some(capture_gpu),
            Some(ExecutionPlacement::FullDevice),
            direct_gpu,
            direct_gpu,
            large_geometry,
            false,
            None,
            None,
        ));
        assert!(whisper_unified_runtime_enabled_with_override(
            GgmlCpuGraphBackend::Gpu,
            Some(capture_gpu),
            Some(ExecutionPlacement::FullDevice),
            direct_gpu,
            direct_gpu,
            medium_geometry,
            false,
            None,
            Some("1"),
        ));
    }
    assert!(!whisper_unified_runtime_enabled_with_override(
        GgmlCpuGraphBackend::Gpu,
        Some(&vulkan),
        Some(ExecutionPlacement::FullDevice),
        direct_gpu,
        direct_gpu,
        medium_geometry,
        false,
        Some("1"),
        None,
    ));
    for provider in [
        ExecutionProvider::Cpu,
        ExecutionProvider::Metal,
        ExecutionProvider::Accelerator,
        ExecutionProvider::Unknown,
    ] {
        let preference = exactly_addressable_preference(provider);
        assert!(!whisper_unified_runtime_enabled_with_override(
            GgmlCpuGraphBackend::Gpu,
            Some(&preference),
            Some(ExecutionPlacement::FullDevice),
            direct_gpu,
            direct_gpu,
            large_geometry,
            true,
            None,
            None,
        ));
    }
    let scheduled_gpu = GgmlCpuGraphConfig {
        use_scheduler: true,
        ..direct_gpu
    };
    assert!(!whisper_unified_runtime_enabled_with_override(
        GgmlCpuGraphBackend::Gpu,
        Some(&exactly_addressable_preference(ExecutionProvider::Cuda)),
        Some(ExecutionPlacement::FullDevice),
        direct_gpu,
        scheduled_gpu,
        large_geometry,
        true,
        None,
        None,
    ));
}

#[test]
fn gpu_loaded_f16_views_require_exact_direct_cuda_hip_vulkan_full_device() {
    let direct_gpu = GgmlCpuGraphConfig {
        backend: GgmlCpuGraphBackend::Gpu,
        use_scheduler: false,
        ..GgmlCpuGraphConfig::conservative_default()
    };
    for provider in [
        ExecutionProvider::Cuda,
        ExecutionProvider::Hip,
        ExecutionProvider::Vulkan,
    ] {
        let preference = exactly_addressable_preference(provider);
        assert_eq!(
            whisper_gpu_loaded_f16_weight_mode_with_override(
                GgmlCpuGraphBackend::Gpu,
                Some(&preference),
                Some(ExecutionPlacement::FullDevice),
                direct_gpu,
                None,
            ),
            WhisperGpuLoadedF16WeightMode::LoadedView
        );
        assert_eq!(
            whisper_gpu_loaded_f16_weight_mode_with_override(
                GgmlCpuGraphBackend::Gpu,
                Some(&preference),
                Some(ExecutionPlacement::FullDevice),
                direct_gpu,
                Some("1"),
            ),
            WhisperGpuLoadedF16WeightMode::ArenaCopy
        );
    }
    for provider in [
        ExecutionProvider::Cpu,
        ExecutionProvider::Metal,
        ExecutionProvider::Accelerator,
        ExecutionProvider::Unknown,
    ] {
        let preference = exactly_addressable_preference(provider);
        assert_eq!(
            whisper_gpu_loaded_f16_weight_mode_with_override(
                GgmlCpuGraphBackend::Gpu,
                Some(&preference),
                Some(ExecutionPlacement::FullDevice),
                direct_gpu,
                None,
            ),
            WhisperGpuLoadedF16WeightMode::ArenaCopy
        );
    }
    let scheduled_gpu = GgmlCpuGraphConfig {
        use_scheduler: true,
        ..direct_gpu
    };
    assert_eq!(
        whisper_gpu_loaded_f16_weight_mode_with_override(
            GgmlCpuGraphBackend::Gpu,
            Some(&exactly_addressable_preference(ExecutionProvider::Cuda)),
            Some(ExecutionPlacement::FullDevice),
            scheduled_gpu,
            None,
        ),
        WhisperGpuLoadedF16WeightMode::ArenaCopy
    );
    assert_eq!(
        whisper_gpu_loaded_f16_weight_mode_with_override(
            GgmlCpuGraphBackend::Gpu,
            Some(&exactly_addressable_preference(ExecutionProvider::Vulkan)),
            Some(ExecutionPlacement::Hybrid),
            direct_gpu,
            None,
        ),
        WhisperGpuLoadedF16WeightMode::ArenaCopy
    );
}

#[test]
fn same_retained_graph_serves_full_logits_and_native_first_max_plans() {
    use crate::ggml_runtime::{
        AutoGpuPolicy, GgmlDecodeOutputContract, GgmlDecodeOutputPlan, RequestBackendPreference,
        ResolvedFamilyRuntimeInput,
    };
    use crate::models::runtime_cache_coordinator::PackContentKey;

    let full = ResolvedFamilyRuntimeInput::resolve_with_output_contract(
        Some(RequestBackendPreference::CpuOnly),
        AutoGpuPolicy::AllBackends,
        GgmlDecodeOutputContract::FullLogits,
    );
    let compact = ResolvedFamilyRuntimeInput::resolve_with_output_contract(
        Some(RequestBackendPreference::CpuOnly),
        AutoGpuPolicy::AllBackends,
        GgmlDecodeOutputContract::NativeFirstMaxTokenOrFullLogits,
    );
    assert_eq!(full.output_plan(), GgmlDecodeOutputPlan::FullLogits);
    assert_eq!(
        compact.output_plan(),
        GgmlDecodeOutputPlan::NativeFirstMaxToken
    );

    let content = PackContentKey::new("sha256:whisper-output-plan-fixture");
    let lane = current_execution_lane_key(GgmlCpuGraphBackend::Cpu);
    let capacity = Seq2SeqResidentCapacity {
        self_attention_positions: 448,
        cross_attention_positions: 1500,
    };
    let weight_mode = WhisperGpuLoadedF16WeightMode::ArenaCopy;
    // Production checkout does not take output_plan: the retained decoder
    // graph always materializes complete logits, so both plans share one owner.
    let decoder_key = |_plan: GgmlDecodeOutputPlan| -> WhisperDecoderPersistentSessionKey {
        (content.clone(), lane.clone(), capacity, weight_mode)
    };
    let unified_key = |_plan: GgmlDecodeOutputPlan| -> WhisperUnifiedPersistentSessionKey {
        (content.clone(), lane.clone(), capacity, weight_mode)
    };
    assert_eq!(
        decoder_key(full.output_plan()),
        decoder_key(compact.output_plan()),
        "whisper decoder owner must serve FullLogits and NativeFirstMaxToken with one retained graph"
    );
    assert_eq!(
        unified_key(full.output_plan()),
        unified_key(compact.output_plan()),
        "whisper unified owner must serve FullLogits and NativeFirstMaxToken with one retained graph"
    );
}

#[test]
fn encoder_loaded_f16_view_proof_rejects_converted_and_transposed_sources() {
    let make_tensor =
        |source_ggml_type, source_dims: Vec<u64>, payload| WhisperMaterializedTensor {
            slot: WhisperGgufTensorSlot::EncoderLayerSelfAttnQWeight { layer_idx: 0 },
            tensor_name: "model.encoder.layers.0.self_attn.q_proj.weight".to_string(),
            source_ggml_type,
            dims: source_dims.clone(),
            source_dims,
            num_elements: 6,
            payload,
        };

    let mut source_f16 = make_tensor(
        1,
        vec![2, 3],
        WhisperMaterializedTensorPayload::F16Bits(vec![0; 6]),
    );
    prepare_encoder_linear_weight_tensor_input_output_f16(&mut source_f16, 2, 3)
        .expect("prepare source f16 input-output");
    assert!(source_f16.source_is_f16_input_output(2, 3));

    let mut converted_f32 = make_tensor(
        0,
        vec![2, 3],
        WhisperMaterializedTensorPayload::F32(vec![0.0; 6]),
    );
    prepare_encoder_linear_weight_tensor_input_output_f16(&mut converted_f32, 2, 3)
        .expect("prepare converted f32");
    assert!(!converted_f32.source_is_f16_input_output(2, 3));

    let mut transposed_f16 = make_tensor(
        1,
        vec![3, 2],
        WhisperMaterializedTensorPayload::F16Bits(vec![0; 6]),
    );
    prepare_encoder_linear_weight_tensor_input_output_f16(&mut transposed_f16, 2, 3)
        .expect("prepare transposed source f16");
    assert_eq!(transposed_f16.dims, vec![2, 3]);
    assert!(!transposed_f16.source_is_f16_input_output(2, 3));
}

#[test]
fn cuda_unified_owner_geometry_requires_large_depth_and_width() {
    assert!(
        WhisperUnifiedRuntimeGeometry {
            decoder_layers: 32,
            decoder_hidden_size: 1280,
            tensor_storage_bytes: Some(WHISPER_CUDA_UNIFIED_MIN_TENSOR_STORAGE_BYTES),
        }
        .favors_cuda_unified_runtime()
    );
    assert!(
        !WhisperUnifiedRuntimeGeometry {
            decoder_layers: 31,
            decoder_hidden_size: 1280,
            tensor_storage_bytes: Some(WHISPER_CUDA_UNIFIED_MIN_TENSOR_STORAGE_BYTES),
        }
        .favors_cuda_unified_runtime()
    );
    assert!(
        !WhisperUnifiedRuntimeGeometry {
            decoder_layers: 32,
            decoder_hidden_size: 1279,
            tensor_storage_bytes: Some(WHISPER_CUDA_UNIFIED_MIN_TENSOR_STORAGE_BYTES),
        }
        .favors_cuda_unified_runtime()
    );
    assert!(
        !WhisperUnifiedRuntimeGeometry {
            decoder_layers: 32,
            decoder_hidden_size: 1280,
            tensor_storage_bytes: Some(WHISPER_CUDA_UNIFIED_MIN_TENSOR_STORAGE_BYTES - 1),
        }
        .favors_cuda_unified_runtime()
    );
    assert!(
        !WhisperUnifiedRuntimeGeometry {
            decoder_layers: 32,
            decoder_hidden_size: 1280,
            tensor_storage_bytes: None,
        }
        .favors_cuda_unified_runtime()
    );
}

fn sha256_f32_le(values: &[f32]) -> String {
    let mut hasher = Sha256::new();
    for value in values {
        hasher.update(value.to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

#[derive(Debug, Clone)]
enum TestPreludeRunnerOutcome {
    Success,
}

struct TestPreludeRunner {
    called: Arc<AtomicBool>,
    outcome: TestPreludeRunnerOutcome,
}

impl WhisperEncoderPreludeRunner for TestPreludeRunner {
    fn runner_id(&self) -> &'static str {
        "test-whisper-encoder-prelude-runner-v0"
    }

    fn run_encoder_prelude(
        &self,
        _runtime_source: &GgmlRuntimeSource,
        _encoder_weights: &WhisperEncoderWeightBundle,
        plan: &WhisperEncoderPreludePlan,
        mel_input: &WhisperMelFeatureInput,
        _backend: GgmlCpuGraphBackend,
    ) -> Result<WhisperEncoderPreludeSeamResult, WhisperGgmlExecutorError> {
        self.called.store(true, Ordering::SeqCst);
        match &self.outcome {
            TestPreludeRunnerOutcome::Success => {
                let output_hidden_f32 = vec![0.0; plan.output_frames * plan.output_hidden_size];
                assert_eq!(
                    mel_input.values_f32.len(),
                    plan.input_shape.mel_frames * plan.input_shape.mel_bins
                );
                Ok(WhisperEncoderPreludeSeamResult::GraphExecuted {
                    runner_id: self.runner_id(),
                    output_frames: plan.output_frames,
                    output_hidden_size: plan.output_hidden_size,
                    output_hidden_f32,
                })
            }
        }
    }
}

#[derive(Debug, Clone)]
enum TestEncoderGraphRunnerOutcome {
    Success,
}

struct TestEncoderGraphRunner {
    called: Arc<AtomicBool>,
    outcome: TestEncoderGraphRunnerOutcome,
}

impl WhisperEncoderGraphRunner for TestEncoderGraphRunner {
    fn runner_id(&self) -> &'static str {
        "test-whisper-encoder-graph-runner-v0"
    }

    fn run_encoder_graph(
        &self,
        input: WhisperEncoderGraphInput<'_>,
        _session: &mut WhisperEncoderPersistentStaticSession,
    ) -> Result<WhisperEncoderGraphSeamResult, WhisperGgmlExecutorError> {
        let plan = input.plan;
        let encoder_hidden_input_f32 = input.encoder_hidden_input_f32;
        self.called.store(true, Ordering::SeqCst);
        assert_eq!(
            encoder_hidden_input_f32.len(),
            plan.output_frames * plan.output_hidden_size
        );
        assert!(
            encoder_hidden_input_f32
                .iter()
                .all(|value| value.is_finite()),
            "encoder graph seam input must stay finite"
        );
        match &self.outcome {
            TestEncoderGraphRunnerOutcome::Success => {
                Ok(WhisperEncoderGraphSeamResult::GraphExecuted {
                    runner_id: self.runner_id(),
                    layer_count: plan.layers.len(),
                    output_frames: plan.output_frames,
                    output_hidden_size: plan.output_hidden_size,
                    output_hidden_f32: vec![0.0; plan.output_frames * plan.output_hidden_size],
                })
            }
        }
    }
}

#[derive(Debug, Clone)]
enum TestMelFeatureInputProviderOutcome {
    RealFrontend,
    ExtractionFailed { reason: String },
}

struct TestMelFeatureInputProvider {
    called: Arc<AtomicBool>,
    outcome: TestMelFeatureInputProviderOutcome,
}

impl WhisperMelFeatureInputProvider for TestMelFeatureInputProvider {
    fn provider_id(&self) -> &'static str {
        "test-whisper-mel-provider-v0"
    }

    fn prepare_mel_feature_input(
        &self,
        execution: &WhisperGgmlExecutionMetadata,
        prepared_audio: &GgmlAsrPreparedAudioView,
    ) -> Result<WhisperMelFeatureInput, WhisperGgmlExecutorError> {
        self.called.store(true, Ordering::SeqCst);
        match &self.outcome {
            TestMelFeatureInputProviderOutcome::RealFrontend => {
                let prepared_audio = crate::GgmlAsrPreparedAudio {
                    sample_rate_hz: prepared_audio.sample_rate_hz,
                    channels: prepared_audio.channels,
                    samples_f32: prepared_audio.samples_f32.to_vec(),
                };
                let mel_input = tiny_whisper_encoder_smoke_real_mel_input(
                    &prepared_audio,
                    execution.encoder_mels_count,
                )
                .map_err(|reason| {
                    WhisperGgmlExecutorError::MelFeatureInputPreparationFailed {
                        reason: format!("provider='{}' {reason}", self.provider_id()),
                    }
                })?;
                Ok(WhisperMelFeatureInput {
                    source_label: mel_input.source_label,
                    shape: WhisperMelFeatureInputShape {
                        mel_bins: mel_input.mel_bins,
                        mel_frames: mel_input.mel_frames,
                    },
                    values_f32: mel_input.values_f32,
                })
            }
            TestMelFeatureInputProviderOutcome::ExtractionFailed { reason } => {
                Err(WhisperGgmlExecutorError::MelFeatureExtractionFailed {
                    reason: reason.clone(),
                })
            }
        }
    }
}

fn default_prepared_audio() -> GgmlAsrPreparedAudioView<'static> {
    let prepared = tiny_whisper_encoder_smoke_prepared_audio();
    GgmlAsrPreparedAudioView {
        sample_rate_hz: prepared.sample_rate_hz,
        channels: prepared.channels,
        samples_f32: prepared.samples_f32.into(),
    }
}

fn whisper_execution_and_tokenizer_fixture() -> (WhisperGgmlExecutionMetadata, WhisperTokenizer) {
    let mut values = std::collections::BTreeMap::new();
    values.insert(
        "general.architecture".to_string(),
        GgufMetadataValue::String("whisper".to_string()),
    );
    values.insert(
        "whisper.encoder.block_count".to_string(),
        GgufMetadataValue::U32(1),
    );
    values.insert(
        "whisper.encoder.embedding_length".to_string(),
        GgufMetadataValue::U32(4),
    );
    values.insert(
        "whisper.encoder.attention.head_count".to_string(),
        GgufMetadataValue::U32(2),
    );
    values.insert(
        "whisper.encoder.context_length".to_string(),
        GgufMetadataValue::U32(1500),
    );
    values.insert(
        "whisper.encoder.mels_count".to_string(),
        GgufMetadataValue::U32(80),
    );
    values.insert(
        "whisper.decoder.block_count".to_string(),
        GgufMetadataValue::U32(1),
    );
    values.insert(
        "whisper.decoder.embedding_length".to_string(),
        GgufMetadataValue::U32(4),
    );
    values.insert(
        "whisper.decoder.attention.head_count".to_string(),
        GgufMetadataValue::U32(2),
    );
    values.insert(
        "whisper.decoder.context_length".to_string(),
        GgufMetadataValue::U32(32),
    );
    values.insert("whisper.vocab_size".to_string(), GgufMetadataValue::U32(14));
    values.insert(
        TOKENIZER_GGML_MODEL_KEY.to_string(),
        GgufMetadataValue::String(TOKENIZER_GGML_MODEL_VALUE_GPT2.to_string()),
    );
    values.insert(
        TOKENIZER_GGML_TOKENS_KEY.to_string(),
        GgufMetadataValue::StringArray(vec![
            "\u{0120}".to_string(),
            "h".to_string(),
            "e".to_string(),
            "l".to_string(),
            "o".to_string(),
            "w".to_string(),
            "r".to_string(),
            "d".to_string(),
            "<|endoftext|>".to_string(),
            "<|startoftranscript|>".to_string(),
            "<|transcribe|>".to_string(),
            "<|notimestamps|>".to_string(),
            "<|startofprev|>".to_string(),
            "\u{010A}".to_string(),
        ]),
    );
    values.insert(
        TOKENIZER_GGML_MERGES_KEY.to_string(),
        GgufMetadataValue::StringArray(vec!["x y".to_string()]),
    );
    values.insert(
        TOKENIZER_GGML_SPECIAL_TOKEN_IDS_KEY.to_string(),
        GgufMetadataValue::U32Array(vec![8, 9, 10, 11, 12]),
    );
    values.insert(
        TOKENIZER_GGML_SOT_TOKEN_ID_KEY.to_string(),
        GgufMetadataValue::U32(9),
    );
    values.insert(
        TOKENIZER_GGML_EOT_TOKEN_ID_KEY.to_string(),
        GgufMetadataValue::U32(8),
    );
    values.insert(
        TOKENIZER_GGML_TRANSCRIBE_TOKEN_ID_KEY.to_string(),
        GgufMetadataValue::U32(10),
    );
    values.insert(
        TOKENIZER_GGML_NO_TIMESTAMPS_TOKEN_ID_KEY.to_string(),
        GgufMetadataValue::U32(11),
    );
    let metadata = GgufMetadata::from_values_for_test(values);
    let execution =
        validate_whisper_execution_metadata(&metadata).expect("validate whisper metadata");
    let tokenizer = WhisperTokenizer::from_gguf_metadata(&metadata).expect("load tokenizer");
    (execution, tokenizer)
}

// Pinned to the reference platform — see
// GOLDEN_DIFF_TINY_WHISPER_DECODER_STEP_LOGITS_SHA256.
#[cfg(target_os = "macos")]
#[test]
fn golden_diff_tiny_imported_decoder_graph_executes_one_step() {
    let temp = tempfile::tempdir().expect("tempdir");
    let runtime_path = temp.path().join("whisper-decoder-step.gguf");
    let spec = TinyGgufFixtureSpec::whisper_oasr_v1_encoder_graph_one_layer("whisper-fixture");
    write_tiny_gguf_runtime_source(&runtime_path, &spec).expect("write gguf fixture");
    let runtime_source =
        validate_ggml_runtime_source_path(&runtime_path).expect("validate runtime source");
    let metadata =
        read_gguf_metadata_from_runtime_source(&runtime_source).expect("read gguf metadata");
    let execution =
        validate_whisper_execution_metadata(&metadata).expect("validate whisper metadata");
    let tensor_index = load_whisper_tensor_index(&runtime_source).expect("load tensor index");
    let tensor_binding =
        bind_whisper_required_tensors(&tensor_index, &execution).expect("bind tensors");
    let tensor_reader =
        GgufTensorDataReader::from_runtime_source(&runtime_source).expect("create tensor reader");
    let decoder_weights =
        build_decoder_weight_seam(&tensor_reader, &tensor_binding.weights.bindings)
            .expect("materialize decoder weights");

    let encoder_frames = 2usize;
    let encoder_hidden = execution.decoder_hidden_size;
    let encoder_hidden_f32 = (0..encoder_frames * encoder_hidden)
        .map(|idx| (idx as f32) * 0.001)
        .collect::<Vec<_>>();
    let token_count = 1usize;
    let plan = build_whisper_decoder_graph_plan(
        WhisperDecoderGraphMetadata {
            decoder_layers: execution.decoder_layers,
            decoder_hidden_size: execution.decoder_hidden_size,
            decoder_attention_heads: execution.decoder_attention_heads,
            vocab_size: execution.vocab_size,
            semantic_context_positions: execution.max_target_positions,
        },
        &decoder_weights.graph_binding,
        &decoder_weights.graph_materialization,
        WhisperDecoderGraphInputShape {
            token_count,
            encoder_frames,
            hidden_size: encoder_hidden,
        },
    )
    .expect("build decoder plan");
    let mut decoder_tensor_cache = WhisperDecoderExecutionTensorCache::default();
    let mut decoder_graph_runner =
        GgmlCpuGraphRunner::new(GgmlCpuGraphConfig::default()).expect("decoder graph runner");
    let step_logits = run_whisper_decoder_step_ggml_v0(
        &execution,
        &decoder_weights,
        &plan,
        &WhisperDecoderGraphExecutionInput {
            decoder_prefix_tokens: vec![0_u32],
            encoder_hidden_state: encoder_hidden_f32,
            encoder_layout: WhisperDecoderHiddenStateLayout::SequenceHidden,
        },
        WhisperDecoderGraphExecutionConfig {
            attention_heads: execution.decoder_attention_heads,
            use_self_flash_attention: false,
            use_cross_flash_attention: false,
            collect_cross_attention: false,
            layer_norm_epsilon: 1.0e-5_f32,
        },
        &mut decoder_graph_runner,
        None,
        None,
        &mut decoder_tensor_cache,
        &WhisperDecoderStepSeamInput {
            encoder_frames,
            encoder_hidden_size: encoder_hidden,
            step_index: 0,
            position_offset: 0,
        },
    )
    .expect("decoder runner should execute one tiny step");

    assert_eq!(step_logits.logits.len(), execution.vocab_size);
    assert!(
        step_logits.logits.iter().all(|value| value.is_finite()),
        "decoder step logits must remain finite: {:?}",
        step_logits.logits
    );
    assert_eq!(
        sha256_f32_le(&step_logits.logits),
        GOLDEN_DIFF_TINY_WHISPER_DECODER_STEP_LOGITS_SHA256
    );
}

#[test]
fn whisper_preflight_fails_on_missing_metadata_before_encoder_prelude() {
    let temp = tempfile::tempdir().expect("tempdir");
    let runtime_path = temp.path().join("whisper-metadata-missing.gguf");
    let spec = TinyGgufFixtureSpec::whisper_oasr_v1_non_streaming_cpu("whisper-fixture");
    write_tiny_gguf_runtime_source(&runtime_path, &spec).expect("write gguf fixture");
    let runtime_source =
        validate_ggml_runtime_source_path(&runtime_path).expect("validate runtime source");
    let metadata =
        read_gguf_metadata_from_runtime_source(&runtime_source).expect("read gguf metadata");
    let called = Arc::new(AtomicBool::new(false));
    let runner = Arc::new(TestPreludeRunner {
        called: Arc::clone(&called),
        outcome: TestPreludeRunnerOutcome::Success,
    });
    let graph_runner = WhisperCpuEncoderGraphComputeRunnerV0;
    let mel_called = Arc::new(AtomicBool::new(false));
    let mel_provider = TestMelFeatureInputProvider {
        called: Arc::clone(&mel_called),
        outcome: TestMelFeatureInputProviderOutcome::RealFrontend,
    };
    let adapter = builtin_adapter_descriptor(crate::arch::WHISPER_GGML_ARCHITECTURE_ID);

    let error = execute_whisper_ggml_non_streaming_cpu(
        &adapter,
        &runtime_source,
        &metadata,
        &load_whisper_tensor_index(&runtime_source).expect("load tensor index"),
        &default_prepared_audio(),
        &mel_provider,
        runner.as_ref(),
        Arc::new(graph_runner),
    )
    .expect_err("missing whisper metadata must fail preflight");

    match error {
        WhisperGgmlExecutorError::MissingRequiredMetadata { key } => {
            assert_eq!(key, "general.architecture");
        }
        other => panic!("unexpected error: {other}"),
    }
    assert_eq!(
        classify_whisper_execution_failure_stage(&error.to_string()),
        WhisperExecutionFailureStage::MetadataPreflight
    );
    assert!(
        !called.load(Ordering::SeqCst),
        "encoder prelude seam must not run when metadata preflight fails"
    );
    assert!(
        !mel_called.load(Ordering::SeqCst),
        "mel/input seam must not run when metadata preflight fails"
    );
}

#[test]
fn whisper_tensor_shape_mismatch_fails_before_encoder_prelude() {
    let temp = tempfile::tempdir().expect("tempdir");
    let runtime_path = temp.path().join("whisper-shape-mismatch.gguf");
    let spec = TinyGgufFixtureSpec::whisper_oasr_v1_encoder_graph_shape_mismatch(
        "whisper-fixture",
        "model.encoder.conv1.weight",
        [3_u64, 3, 8],
    );
    write_tiny_gguf_runtime_source(&runtime_path, &spec).expect("write gguf fixture");
    let runtime_source =
        validate_ggml_runtime_source_path(&runtime_path).expect("validate runtime source");
    let metadata =
        read_gguf_metadata_from_runtime_source(&runtime_source).expect("read gguf metadata");
    let called = Arc::new(AtomicBool::new(false));
    let runner = Arc::new(TestPreludeRunner {
        called: Arc::clone(&called),
        outcome: TestPreludeRunnerOutcome::Success,
    });
    let graph_runner = WhisperCpuEncoderGraphComputeRunnerV0;
    let mel_called = Arc::new(AtomicBool::new(false));
    let mel_provider = TestMelFeatureInputProvider {
        called: Arc::clone(&mel_called),
        outcome: TestMelFeatureInputProviderOutcome::RealFrontend,
    };
    let adapter = builtin_adapter_descriptor(crate::arch::WHISPER_GGML_ARCHITECTURE_ID);

    let error = execute_whisper_ggml_non_streaming_cpu(
        &adapter,
        &runtime_source,
        &metadata,
        &load_whisper_tensor_index(&runtime_source).expect("load tensor index"),
        &default_prepared_audio(),
        &mel_provider,
        runner.as_ref(),
        Arc::new(graph_runner),
    )
    .expect_err("tensor shape mismatch must fail before prelude seam");

    assert!(matches!(
        error,
        WhisperGgmlExecutorError::InvalidRequiredTensor { .. }
    ));
    assert_eq!(
        classify_whisper_execution_failure_stage(&error.to_string()),
        WhisperExecutionFailureStage::TensorBindingPreflight
    );
    assert!(
        !called.load(Ordering::SeqCst),
        "encoder prelude seam must not run when tensor shape preflight fails"
    );
    assert!(
        !mel_called.load(Ordering::SeqCst),
        "mel/input seam must not run when tensor preflight fails"
    );
}

#[test]
fn whisper_tensor_type_mismatch_fails_before_encoder_prelude() {
    let temp = tempfile::tempdir().expect("tempdir");
    let runtime_path = temp.path().join("whisper-type-mismatch.gguf");
    let spec = TinyGgufFixtureSpec::whisper_oasr_v1_encoder_graph_type_mismatch(
        "whisper-fixture",
        "model.encoder.conv1.bias",
    );
    write_tiny_gguf_runtime_source(&runtime_path, &spec).expect("write gguf fixture");
    let runtime_source =
        validate_ggml_runtime_source_path(&runtime_path).expect("validate runtime source");
    let metadata =
        read_gguf_metadata_from_runtime_source(&runtime_source).expect("read gguf metadata");
    let called = Arc::new(AtomicBool::new(false));
    let runner = Arc::new(TestPreludeRunner {
        called: Arc::clone(&called),
        outcome: TestPreludeRunnerOutcome::Success,
    });
    let graph_runner = WhisperCpuEncoderGraphComputeRunnerV0;
    let mel_called = Arc::new(AtomicBool::new(false));
    let mel_provider = TestMelFeatureInputProvider {
        called: Arc::clone(&mel_called),
        outcome: TestMelFeatureInputProviderOutcome::RealFrontend,
    };
    let adapter = builtin_adapter_descriptor(crate::arch::WHISPER_GGML_ARCHITECTURE_ID);

    let error = execute_whisper_ggml_non_streaming_cpu(
        &adapter,
        &runtime_source,
        &metadata,
        &load_whisper_tensor_index(&runtime_source).expect("load tensor index"),
        &default_prepared_audio(),
        &mel_provider,
        runner.as_ref(),
        Arc::new(graph_runner),
    )
    .expect_err("tensor type mismatch must fail before prelude seam");

    assert!(matches!(
        error,
        WhisperGgmlExecutorError::InvalidRequiredTensor { .. }
    ));
    let message = error.to_string();
    assert_eq!(
        classify_whisper_execution_failure_stage(&message),
        WhisperExecutionFailureStage::TensorBindingPreflight
    );
    assert!(
        message.contains("does not satisfy expected f32/f16/bf16"),
        "unexpected mismatch reason: {message}"
    );
    assert!(
        !called.load(Ordering::SeqCst),
        "encoder prelude seam must not run when tensor type preflight fails"
    );
    assert!(
        !mel_called.load(Ordering::SeqCst),
        "mel/input seam must not run when tensor preflight fails"
    );
}

#[test]
fn mel_feature_extraction_failure_fails_before_encoder_execution() {
    let temp = tempfile::tempdir().expect("tempdir");
    let runtime_path = temp.path().join("whisper-mel-seam.gguf");
    let spec = TinyGgufFixtureSpec::whisper_oasr_v1_encoder_graph_one_layer("whisper-fixture");
    write_tiny_gguf_runtime_source(&runtime_path, &spec).expect("write gguf fixture");
    let runtime_source =
        validate_ggml_runtime_source_path(&runtime_path).expect("validate runtime source");
    let metadata =
        read_gguf_metadata_from_runtime_source(&runtime_source).expect("read gguf metadata");
    let prelude_called = Arc::new(AtomicBool::new(false));
    let prelude_runner = TestPreludeRunner {
        called: Arc::clone(&prelude_called),
        outcome: TestPreludeRunnerOutcome::Success,
    };
    let graph_called = Arc::new(AtomicBool::new(false));
    let graph_runner = TestEncoderGraphRunner {
        called: Arc::clone(&graph_called),
        outcome: TestEncoderGraphRunnerOutcome::Success,
    };
    let mel_called = Arc::new(AtomicBool::new(false));
    let mel_provider = TestMelFeatureInputProvider {
        called: Arc::clone(&mel_called),
        outcome: TestMelFeatureInputProviderOutcome::ExtractionFailed {
            reason: "frontend fft failed".to_string(),
        },
    };
    let adapter = builtin_adapter_descriptor(crate::arch::WHISPER_GGML_ARCHITECTURE_ID);

    let error = execute_whisper_ggml_non_streaming_cpu(
        &adapter,
        &runtime_source,
        &metadata,
        &load_whisper_tensor_index(&runtime_source).expect("load tensor index"),
        &default_prepared_audio(),
        &mel_provider,
        &prelude_runner,
        Arc::new(graph_runner),
    )
    .expect_err("mel seam should fail closed");

    assert!(
        matches!(
            error,
            WhisperGgmlExecutorError::MelFeatureExtractionFailed { .. }
                | WhisperGgmlExecutorError::TokenizerMissing { .. }
                | WhisperGgmlExecutorError::DecoderWeightsMissing { .. }
                | WhisperGgmlExecutorError::DecoderGraphExecutionFailed { .. }
        ),
        "unexpected fail-closed boundary error: {error}"
    );
    if matches!(
        error,
        WhisperGgmlExecutorError::MelFeatureExtractionFailed { .. }
    ) {
        assert_eq!(
            classify_whisper_execution_failure_stage(&error.to_string()),
            WhisperExecutionFailureStage::MelFeature
        );
        assert!(mel_called.load(Ordering::SeqCst), "mel/input seam must run");
        assert!(
            !prelude_called.load(Ordering::SeqCst),
            "encoder prelude must not run when mel seam fails"
        );
        assert!(
            !graph_called.load(Ordering::SeqCst),
            "encoder graph must not run when mel seam fails"
        );
    }
}

#[test]
fn golden_diff_prepared_audio_real_mel_and_real_encoder_compute_reach_decoder_fail_closed() {
    let temp = tempfile::tempdir().expect("tempdir");
    let runtime_path = temp.path().join("whisper-real-mel.gguf");
    let spec = TinyGgufFixtureSpec::whisper_oasr_v1_encoder_graph_one_layer("whisper-fixture")
        .with_whisper_minimal_tokenizer();
    write_tiny_gguf_runtime_source(&runtime_path, &spec).expect("write gguf fixture");
    let runtime_source =
        validate_ggml_runtime_source_path(&runtime_path).expect("validate runtime source");
    let metadata =
        read_gguf_metadata_from_runtime_source(&runtime_source).expect("read gguf metadata");

    let prepared_audio = default_prepared_audio();
    let mel_provider = TestMelFeatureInputProvider {
        called: Arc::new(AtomicBool::new(false)),
        outcome: TestMelFeatureInputProviderOutcome::RealFrontend,
    };
    let execution =
        validate_whisper_execution_metadata(&metadata).expect("metadata should be valid");
    let tensor_index = load_whisper_tensor_index(&runtime_source).expect("load tensor index");
    let tensor_binding =
        bind_whisper_required_tensors(&tensor_index, &execution).expect("bind tensors");
    let encoder_weights = materialize_whisper_encoder_weights(&runtime_source, &tensor_binding)
        .expect("materialize encoder");
    let mel_input = prepare_mel_feature_input_seam(&mel_provider, &execution, &prepared_audio)
        .expect("real frontend mel preparation");
    assert!(
        mel_input.values_f32.iter().all(|value| value.is_finite()),
        "mel values must stay finite"
    );
    let prelude_plan = build_whisper_encoder_prelude_plan(
        &tensor_binding.weights.bindings,
        infer_encoder_prelude_input_shape_from_mel_input(&mel_input)
            .expect("infer prelude input shape"),
        execution.encoder_hidden_size,
        execution.encoder_mels_count,
    )
    .expect("build prelude plan");
    let prelude_result = run_encoder_prelude_seam(
        &runtime_source,
        &encoder_weights,
        &prelude_plan,
        &mel_input,
        &WhisperCpuEncoderPreludeComputeRunnerV0,
        GgmlCpuGraphBackend::Cpu,
    )
    .expect("run prelude seam");
    let smoke_shape = tiny_whisper_encoder_smoke_shape_for_default_fixture();
    match prelude_result {
        WhisperEncoderPreludeSeamResult::GraphExecuted {
            output_frames,
            output_hidden_size,
            output_hidden_f32,
            ..
        } => {
            assert_eq!(output_frames, smoke_shape.output_frames);
            assert_eq!(output_hidden_size, smoke_shape.hidden_size);
            assert_tiny_whisper_encoder_output_shape_and_finite(&output_hidden_f32, smoke_shape);
            assert_eq!(
                sha256_f32_le(&output_hidden_f32),
                GOLDEN_DIFF_TINY_WHISPER_ENCODER_PRELUDE_SHA256
            );
        }
    }

    let mut cached_prelude = WhisperEncoderPreludeCachedRuntime::build(
        &encoder_weights,
        &prelude_plan,
        GgmlCpuGraphBackend::Cpu,
    )
    .expect("build cached prelude runtime");
    let first_cached = cached_prelude
        .run(&mel_input)
        .expect("run cached prelude first time");
    let second_cached = cached_prelude
        .run(&mel_input)
        .expect("reuse cached prelude second time");
    let (
        WhisperEncoderPreludeSeamResult::GraphExecuted {
            output_hidden_f32: first_values,
            ..
        },
        WhisperEncoderPreludeSeamResult::GraphExecuted {
            output_hidden_f32: second_values,
            ..
        },
    ) = (first_cached, second_cached);
    assert_eq!(
        first_values, second_values,
        "owner-thread cached prelude must be byte-identical across reuse"
    );
    assert_eq!(
        sha256_f32_le(&first_values),
        GOLDEN_DIFF_TINY_WHISPER_ENCODER_PRELUDE_SHA256
    );

    let services = crate::models::native_execution_services::test_native_execution_services();
    let broker = Arc::clone(services.memory_broker());
    let _services_scope =
        crate::models::native_execution_services::install_native_execution_services(&services);
    let system_domain = crate::device::execution_memory::MemoryDomainKey::SystemMemory;
    let preflight = GgufRuntimeSourcePreflight {
        runtime_source: runtime_source.clone(),
        metadata: Arc::new(metadata.clone()),
        tensor_index: Arc::new(tensor_index.clone()),
    };
    let prepared_owner = Arc::new(SystemMemoryOwner::without_allocation(
        build_whisper_prepared_runtime(&preflight).expect("build prepared runtime"),
    ));
    let executor = WhisperGgmlExecutor::default();
    let first_actor = checkout_whisper_encoder_runtime(
        &executor.encoder_runtimes,
        &runtime_source,
        Arc::clone(&prepared_owner),
        Arc::new(WhisperCpuEncoderGraphComputeRunnerV0),
        GgmlCpuGraphBackend::Cpu,
        WhisperGpuLoadedF16WeightMode::ArenaCopy,
    )
    .expect("checkout first encoder actor");
    let first_actor_output = run_whisper_encoder_prelude_actor(
        &first_actor,
        Arc::clone(&prepared_owner),
        prelude_plan.clone(),
        mel_input.clone(),
        GgmlCpuGraphBackend::Cpu,
    )
    .expect("run first actor prelude");
    let first_runtime_identity = first_actor
        .call_mut(|state| {
            state
                .prelude
                .as_ref()
                .map(|runtime| runtime as *const WhisperEncoderPreludeCachedRuntime as usize)
        })
        .expect("inspect first actor prelude")
        .expect("first actor must retain its prelude runtime");
    drop(first_actor);
    assert_eq!(executor.encoder_runtimes.usage_for_test(), (1, 0));
    let after_first = broker.usage(&system_domain);

    let second_actor = checkout_whisper_encoder_runtime(
        &executor.encoder_runtimes,
        &runtime_source,
        Arc::clone(&prepared_owner),
        Arc::new(WhisperCpuEncoderGraphComputeRunnerV0),
        GgmlCpuGraphBackend::Cpu,
        WhisperGpuLoadedF16WeightMode::ArenaCopy,
    )
    .expect("checkout returned encoder actor");
    let second_actor_output = run_whisper_encoder_prelude_actor(
        &second_actor,
        Arc::clone(&prepared_owner),
        prelude_plan.clone(),
        mel_input.clone(),
        GgmlCpuGraphBackend::Cpu,
    )
    .expect("run cached actor prelude");
    let second_runtime_identity = second_actor
        .call_mut(|state| {
            state
                .prelude
                .as_ref()
                .map(|runtime| runtime as *const WhisperEncoderPreludeCachedRuntime as usize)
        })
        .expect("inspect cached actor prelude")
        .expect("cached actor must retain its prelude runtime");
    assert_eq!(second_runtime_identity, first_runtime_identity);
    assert_eq!(second_actor_output, first_actor_output);
    drop(second_actor);
    assert_eq!(executor.encoder_runtimes.usage_for_test(), (1, 0));
    assert_eq!(broker.usage(&system_domain), after_first);

    let output = execute_whisper_ggml_non_streaming_cpu(
        &builtin_adapter_descriptor(crate::arch::WHISPER_GGML_ARCHITECTURE_ID),
        &runtime_source,
        &metadata,
        &load_whisper_tensor_index(&runtime_source).expect("load tensor index"),
        &prepared_audio,
        &mel_provider,
        &WhisperCpuEncoderPreludeComputeRunnerV0,
        Arc::new(WhisperCpuEncoderGraphComputeRunnerV0),
    );
    match output {
        Ok(text) => {
            assert!(
                !text.trim().is_empty(),
                "decoder graph + tokenizer path should not emit empty text"
            );
        }
        Err(error) => {
            assert!(
                matches!(
                    error,
                    WhisperGgmlExecutorError::DecoderNoEotBeforeMaxTokens { .. }
                        | WhisperGgmlExecutorError::DecoderInvalidTokenDecode { .. }
                        | WhisperGgmlExecutorError::DecoderGraphExecutionFailed { .. }
                        | WhisperGgmlExecutorError::DecoderGraphUnsupported { .. }
                        | WhisperGgmlExecutorError::DecoderWeightsMissing { .. }
                        | WhisperGgmlExecutorError::TokenizerMissing { .. }
                ),
                "unexpected decoder-stage fail-closed error: {error}"
            );
            assert!(
                matches!(
                    classify_whisper_execution_failure_stage(&error.to_string()),
                    WhisperExecutionFailureStage::MetadataPreflight
                        | WhisperExecutionFailureStage::EncoderExecuted
                        | WhisperExecutionFailureStage::Unknown
                ),
                "unexpected failure stage: {error}"
            );
        }
    }
}

#[test]
fn invalid_sample_rate_fails_closed_before_encoder_execution() {
    let temp = tempfile::tempdir().expect("tempdir");
    let runtime_path = temp.path().join("whisper-invalid-sample-rate.gguf");
    let spec = TinyGgufFixtureSpec::whisper_oasr_v1_encoder_graph_one_layer("whisper-fixture");
    write_tiny_gguf_runtime_source(&runtime_path, &spec).expect("write gguf fixture");
    let runtime_source =
        validate_ggml_runtime_source_path(&runtime_path).expect("validate runtime source");
    let metadata =
        read_gguf_metadata_from_runtime_source(&runtime_source).expect("read gguf metadata");

    let prelude_called = Arc::new(AtomicBool::new(false));
    let prelude_runner = TestPreludeRunner {
        called: Arc::clone(&prelude_called),
        outcome: TestPreludeRunnerOutcome::Success,
    };
    let graph_called = Arc::new(AtomicBool::new(false));
    let graph_runner = TestEncoderGraphRunner {
        called: Arc::clone(&graph_called),
        outcome: TestEncoderGraphRunnerOutcome::Success,
    };
    let mel_called = Arc::new(AtomicBool::new(false));
    let mel_provider = TestMelFeatureInputProvider {
        called: Arc::clone(&mel_called),
        outcome: TestMelFeatureInputProviderOutcome::RealFrontend,
    };
    let mut invalid_audio = default_prepared_audio();
    invalid_audio.sample_rate_hz = 8_000;

    let error = execute_whisper_ggml_non_streaming_cpu(
        &builtin_adapter_descriptor(crate::arch::WHISPER_GGML_ARCHITECTURE_ID),
        &runtime_source,
        &metadata,
        &load_whisper_tensor_index(&runtime_source).expect("load tensor index"),
        &invalid_audio,
        &mel_provider,
        &prelude_runner,
        Arc::new(graph_runner),
    )
    .expect_err("invalid sample rate must fail before encoder execution");
    assert!(
        matches!(
            error,
            WhisperGgmlExecutorError::MelFeatureInputPreparationFailed { .. }
                | WhisperGgmlExecutorError::TokenizerMissing { .. }
                | WhisperGgmlExecutorError::DecoderWeightsMissing { .. }
                | WhisperGgmlExecutorError::DecoderGraphExecutionFailed { .. }
        ),
        "unexpected fail-closed boundary error: {error}"
    );
    if matches!(
        error,
        WhisperGgmlExecutorError::MelFeatureInputPreparationFailed { .. }
    ) {
        let message = error.to_string();
        assert!(
            message.contains("sample_rate_hz=8000"),
            "unexpected error: {message}"
        );
        assert_eq!(
            classify_whisper_execution_failure_stage(&message),
            WhisperExecutionFailureStage::MelFeature
        );
        assert!(mel_called.load(Ordering::SeqCst), "mel seam must run");
        assert!(
            !prelude_called.load(Ordering::SeqCst),
            "encoder prelude must not run for invalid sample rate"
        );
        assert!(
            !graph_called.load(Ordering::SeqCst),
            "encoder graph must not run for invalid sample rate"
        );
    }
}

#[test]
fn nan_audio_fails_closed_before_encoder_execution() {
    let temp = tempfile::tempdir().expect("tempdir");
    let runtime_path = temp.path().join("whisper-nan-audio.gguf");
    let spec = TinyGgufFixtureSpec::whisper_oasr_v1_encoder_graph_one_layer("whisper-fixture");
    write_tiny_gguf_runtime_source(&runtime_path, &spec).expect("write gguf fixture");
    let runtime_source =
        validate_ggml_runtime_source_path(&runtime_path).expect("validate runtime source");
    let metadata =
        read_gguf_metadata_from_runtime_source(&runtime_source).expect("read gguf metadata");

    let prelude_called = Arc::new(AtomicBool::new(false));
    let prelude_runner = TestPreludeRunner {
        called: Arc::clone(&prelude_called),
        outcome: TestPreludeRunnerOutcome::Success,
    };
    let graph_called = Arc::new(AtomicBool::new(false));
    let graph_runner = TestEncoderGraphRunner {
        called: Arc::clone(&graph_called),
        outcome: TestEncoderGraphRunnerOutcome::Success,
    };
    let mel_called = Arc::new(AtomicBool::new(false));
    let mel_provider = TestMelFeatureInputProvider {
        called: Arc::clone(&mel_called),
        outcome: TestMelFeatureInputProviderOutcome::RealFrontend,
    };
    let mut nan_samples = default_prepared_audio().samples_f32.to_vec();
    nan_samples[5] = f32::NAN;
    let nan_audio = GgmlAsrPreparedAudioView::mono_16khz(nan_samples);

    let error = execute_whisper_ggml_non_streaming_cpu(
        &builtin_adapter_descriptor(crate::arch::WHISPER_GGML_ARCHITECTURE_ID),
        &runtime_source,
        &metadata,
        &load_whisper_tensor_index(&runtime_source).expect("load tensor index"),
        &nan_audio,
        &mel_provider,
        &prelude_runner,
        Arc::new(graph_runner),
    )
    .expect_err("non-finite audio must fail before encoder execution");
    assert!(
        matches!(
            error,
            WhisperGgmlExecutorError::MelFeatureInputPreparationFailed { .. }
                | WhisperGgmlExecutorError::TokenizerMissing { .. }
                | WhisperGgmlExecutorError::DecoderWeightsMissing { .. }
                | WhisperGgmlExecutorError::DecoderGraphExecutionFailed { .. }
        ),
        "unexpected fail-closed boundary error: {error}"
    );
    if matches!(
        error,
        WhisperGgmlExecutorError::MelFeatureInputPreparationFailed { .. }
    ) {
        let message = error.to_string();
        assert!(
            message.contains("samples_f32 contains non-finite values"),
            "unexpected error: {message}"
        );
        assert_eq!(
            classify_whisper_execution_failure_stage(&message),
            WhisperExecutionFailureStage::MelFeature
        );
        assert!(mel_called.load(Ordering::SeqCst), "mel seam must run");
        assert!(
            !prelude_called.load(Ordering::SeqCst),
            "encoder prelude must not run for non-finite audio"
        );
        assert!(
            !graph_called.load(Ordering::SeqCst),
            "encoder graph must not run for non-finite audio"
        );
    }
}

#[test]
fn unsupported_primitive_fixture_fails_closed_with_real_prelude_runner() {
    let temp = tempfile::tempdir().expect("tempdir");
    let runtime_path = temp.path().join("whisper-prelude-capacity.gguf");
    let spec =
        TinyGgufFixtureSpec::whisper_oasr_v1_encoder_graph_unsupported_primitive("whisper-fixture");
    write_tiny_gguf_runtime_source(&runtime_path, &spec).expect("write gguf fixture");
    let runtime_source =
        validate_ggml_runtime_source_path(&runtime_path).expect("validate runtime source");
    let metadata =
        read_gguf_metadata_from_runtime_source(&runtime_source).expect("read gguf metadata");
    let adapter = builtin_adapter_descriptor(crate::arch::WHISPER_GGML_ARCHITECTURE_ID);
    let runner = WhisperCpuEncoderPreludeComputeRunnerV0;
    let graph_runner = WhisperCpuEncoderGraphComputeRunnerV0;
    let mel_provider = TestMelFeatureInputProvider {
        called: Arc::new(AtomicBool::new(false)),
        outcome: TestMelFeatureInputProviderOutcome::RealFrontend,
    };

    let error = execute_whisper_ggml_non_streaming_cpu(
        &adapter,
        &runtime_source,
        &metadata,
        &load_whisper_tensor_index(&runtime_source).expect("load tensor index"),
        &default_prepared_audio(),
        &mel_provider,
        &runner,
        Arc::new(graph_runner),
    )
    .expect_err("fixture should force unsupported prelude primitive");

    assert!(
        matches!(
            error,
            WhisperGgmlExecutorError::EncoderPreludePrimitiveUnsupported { .. }
                | WhisperGgmlExecutorError::TokenizerMissing { .. }
                | WhisperGgmlExecutorError::DecoderWeightsMissing { .. }
                | WhisperGgmlExecutorError::DecoderGraphExecutionFailed { .. }
        ),
        "unexpected fail-closed boundary error: {error}"
    );
    if matches!(
        error,
        WhisperGgmlExecutorError::EncoderPreludePrimitiveUnsupported { .. }
    ) {
        let message = error.to_string();
        assert_eq!(
            classify_whisper_execution_failure_stage(&message),
            WhisperExecutionFailureStage::EncoderPrelude
        );
        assert!(
            message.contains("encoder.positional_embedding.slice"),
            "unexpected error: {message}"
        );
    }
}

#[test]
fn decode_generated_token_step_cap_is_bounded() {
    let cap =
        decode_generated_token_step_cap(448, 4).expect("cap should be derived from ctx budget");
    assert_eq!(cap, WHISPER_DEFAULT_DECODE_MAX_GENERATED_TOKENS_CAP);

    let cap = decode_generated_token_step_cap(64, 4).expect("cap should respect tiny budget");
    assert_eq!(cap, 60);
}

#[test]
fn decode_generated_token_step_cap_fails_when_prompt_exhausts_context() {
    let error = decode_generated_token_step_cap(8, 8).expect_err("zero budget should fail");
    assert!(matches!(
        error,
        WhisperGgmlExecutorError::DecoderGraphUnsupported { .. }
    ));
}

#[test]
fn build_whisper_initial_prompt_tokens_appends_encoded_prompt_text() {
    let (execution, tokenizer) = whisper_execution_and_tokenizer_fixture();
    let request_options = GgmlAsrExecutionOptions::from_transcription_request(
        None,
        Some(" hello world ".to_string()),
        None,
    );
    let decoder_start_token_id = tokenizer
        .start_of_transcript_token_id()
        .unwrap_or(execution.decoder_start_token_id);
    let prefix = tokenizer
        .decoder_prefix(
            decoder_start_token_id,
            &WhisperPrefixSpec::transcribe(false),
        )
        .expect("default prefix");
    let encoded_prompt = tokenizer
        .encode_prompt_text("hello world")
        .expect("encode prompt");

    let initial_prompt_tokens =
        build_whisper_initial_prompt_tokens(&execution, &tokenizer, &request_options, None)
            .expect("build initial prompt");

    assert_eq!(&initial_prompt_tokens[..prefix.len()], prefix.as_slice());
    assert_eq!(
        &initial_prompt_tokens[prefix.len()..],
        encoded_prompt.as_slice()
    );
}

#[test]
fn build_whisper_initial_prompt_tokens_truncates_prompt_to_context_tail() {
    let (execution, tokenizer) = whisper_execution_and_tokenizer_fixture();
    let repeated_prompt = std::iter::repeat_n(" hello world", 128).collect::<String>();
    let request_options = GgmlAsrExecutionOptions::from_transcription_request(
        None,
        Some(repeated_prompt.clone()),
        None,
    );
    let decoder_start_token_id = tokenizer
        .start_of_transcript_token_id()
        .unwrap_or(execution.decoder_start_token_id);
    let prefix = tokenizer
        .decoder_prefix(
            decoder_start_token_id,
            &WhisperPrefixSpec::transcribe(false),
        )
        .expect("default prefix");
    let encoded_prompt = tokenizer
        .encode_prompt_text("hello world hello world hello world")
        .expect("sanity encode prompt");
    assert!(
        !encoded_prompt.is_empty(),
        "fixture prompt tokens must be non-empty"
    );

    let initial_prompt_tokens =
        build_whisper_initial_prompt_tokens(&execution, &tokenizer, &request_options, None)
            .expect("build initial prompt");

    let max_prompt_tokens = execution
        .max_target_positions
        .saturating_sub(prefix.len())
        .saturating_sub(1);
    assert_eq!(
        initial_prompt_tokens.len(),
        prefix.len() + max_prompt_tokens
    );
    let full_prompt_tokens = tokenizer
        .encode_prompt_text(repeated_prompt.trim())
        .expect("encode full repeated prompt");
    assert!(full_prompt_tokens.len() > max_prompt_tokens);
    assert_eq!(
        &initial_prompt_tokens[prefix.len()..],
        &full_prompt_tokens[full_prompt_tokens.len() - max_prompt_tokens..]
    );
}

#[test]
fn build_whisper_initial_prompt_tokens_caps_longform_prompt_tail() {
    let (execution, tokenizer) = whisper_execution_and_tokenizer_fixture();
    let repeated_prompt = std::iter::repeat_n(" hello world", 128).collect::<String>();
    let request_options = GgmlAsrExecutionOptions::from_transcription_request(
        None,
        Some(repeated_prompt.clone()),
        Some(crate::LongFormOptions::default()),
    );
    let decoder_start_token_id = tokenizer
        .start_of_transcript_token_id()
        .unwrap_or(execution.decoder_start_token_id);
    let prefix = tokenizer
        .decoder_prefix(
            decoder_start_token_id,
            &WhisperPrefixSpec::transcribe(false),
        )
        .expect("default prefix");
    let prev_token_id = tokenizer
        .token_id_by_content("<|startofprev|>")
        .expect("fixture prev token");
    let full_prompt_tokens = tokenizer
        .encode_prompt_text(repeated_prompt.trim())
        .expect("encode full repeated prompt");

    let initial_prompt_tokens =
        build_whisper_initial_prompt_tokens(&execution, &tokenizer, &request_options, None)
            .expect("build initial prompt");
    let expected_tail = execution
        .max_target_positions
        .saturating_sub(prefix.len())
        .saturating_sub(1)
        .saturating_sub(1)
        .min(WHISPER_LONGFORM_PROMPT_TOKEN_TAIL_LIMIT);

    assert_eq!(
        initial_prompt_tokens.len(),
        prefix.len() + expected_tail + 1
    );
    assert_eq!(initial_prompt_tokens[0], prev_token_id);
    assert_eq!(
        &initial_prompt_tokens[1..1 + expected_tail],
        &full_prompt_tokens[full_prompt_tokens.len() - expected_tail..]
    );
    assert_eq!(
        &initial_prompt_tokens[1 + expected_tail..],
        prefix.as_slice()
    );
}

#[test]
fn build_whisper_initial_prompt_tokens_prefers_direct_prompt_token_ids() {
    let (execution, tokenizer) = whisper_execution_and_tokenizer_fixture();
    let direct_prompt_tokens = vec![9, 10, 11, 12];
    let request_options = GgmlAsrExecutionOptions {
        prompt: Some("hello world".to_string()),
        prompt_token_ids: Some(direct_prompt_tokens.clone()),
        longform: Some(crate::LongFormOptions::default()),
        ..GgmlAsrExecutionOptions::default()
    };
    let decoder_start_token_id = tokenizer
        .start_of_transcript_token_id()
        .unwrap_or(execution.decoder_start_token_id);
    let prefix = tokenizer
        .decoder_prefix(
            decoder_start_token_id,
            &WhisperPrefixSpec::transcribe(false),
        )
        .expect("default prefix");
    let prev_token_id = tokenizer
        .token_id_by_content("<|startofprev|>")
        .expect("fixture prev token");

    let initial_prompt_tokens =
        build_whisper_initial_prompt_tokens(&execution, &tokenizer, &request_options, None)
            .expect("build initial prompt");

    assert_eq!(initial_prompt_tokens[0], prev_token_id);
    assert_eq!(
        &initial_prompt_tokens[1..1 + direct_prompt_tokens.len()],
        direct_prompt_tokens.as_slice()
    );
    assert_eq!(
        &initial_prompt_tokens[1 + direct_prompt_tokens.len()..],
        prefix.as_slice()
    );
}

#[test]
fn diarization_forced_word_anchors_keep_whisper_decode_path_identical() {
    // F1 regression: word timestamps forced solely for diarization must not
    // switch the whisper decode path (cross flash attention off +
    // cross-attention collection on), because that perturbs the transcript via
    // FP accumulation differences between diarize on/off.
    let plain = GgmlAsrExecutionOptions::default();
    let diarize_forced = GgmlAsrExecutionOptions {
        word_timestamps: true,
        word_timestamps_forced_for_diarization: true,
        ..GgmlAsrExecutionOptions::default()
    };
    let user_requested = GgmlAsrExecutionOptions {
        word_timestamps: true,
        ..GgmlAsrExecutionOptions::default()
    };

    assert_eq!(
        whisper_word_timestamp_mode(&plain),
        WhisperWordTimestampMode::Off
    );
    assert_eq!(
        whisper_word_timestamp_mode(&diarize_forced),
        WhisperWordTimestampMode::PostHocAnchors
    );
    assert_eq!(
        whisper_word_timestamp_mode(&user_requested),
        WhisperWordTimestampMode::CrossAttention
    );

    for cross_flash_enabled in [false, true] {
        // Diarize-forced anchors: decoder flags byte-identical to a plain run.
        assert_eq!(
            whisper_decoder_cross_attention_flags(cross_flash_enabled, &diarize_forced),
            whisper_decoder_cross_attention_flags(cross_flash_enabled, &plain),
            "diarize-forced word anchors must not alter the decode path (cross_flash_enabled={cross_flash_enabled})"
        );
        // User-requested word timestamps keep the higher-fidelity
        // cross-attention behavior: collection on, cross flash attention off.
        assert_eq!(
            whisper_decoder_cross_attention_flags(cross_flash_enabled, &user_requested),
            (false, true)
        );
    }
}

#[test]
fn build_whisper_carry_prompt_token_ids_keeps_last_longform_tail() {
    let (_, tokenizer) = whisper_execution_and_tokenizer_fixture();
    let first_timestamp = tokenizer
        .first_timestamp_token_id()
        .expect("first timestamp id");
    let request_options = GgmlAsrExecutionOptions {
        language: None,
        task: crate::TranscriptionTask::Transcribe,
        prompt: None,
        prompt_token_ids: Some(vec![1; 40]),
        phrase_bias: None,
        inference_threads: None,
        word_timestamps: false,
        word_timestamps_forced_for_diarization: false,
        in_decoder_speakers: false,
        longform: Some(crate::LongFormOptions::default()),
        longform_chunk_count_hint: None,
        auto_prefer_cpu_decoder_for_multichunk_metal: false,
        serve_batch: crate::models::serve_batch_env::ServeBatchPolicy::serial(),
        runtime_build_identity: None,
        adapter_path: None,
    };

    // Seed and generated use ids below the first timestamp id so they are
    // treated as plain words, isolating the tail-trim from the timestamp
    // strip. The generated tail alternates so it is not loop-dominant (a
    // flat run would be refused by the carry loop-dominance gate).
    assert!(2 < first_timestamp);
    let generated = vec![2, 3, 2, 3, 2, 3, 2, 3, 2, 3];
    let carry_prompt_token_ids =
        build_whisper_carry_prompt_token_ids(&tokenizer, &request_options, &generated, None)
            .expect("carry prompt tokens")
            .expect("carry prompt token ids");

    assert_eq!(
        carry_prompt_token_ids.len(),
        WHISPER_LONGFORM_PROMPT_TOKEN_TAIL_LIMIT
    );
    // tail = last 32 of seed[1;40] ++ generated = 22 ones then the run.
    let mut expected = vec![1; 22];
    expected.extend_from_slice(&generated);
    assert_eq!(carry_prompt_token_ids.as_slice(), &expected);
}

#[test]
fn build_whisper_carry_prompt_token_ids_strips_prior_slice_timestamps() {
    let (_, tokenizer) = whisper_execution_and_tokenizer_fixture();
    let first_timestamp = tokenizer
        .first_timestamp_token_id()
        .expect("first timestamp id");
    let request_options = GgmlAsrExecutionOptions {
        prompt_token_ids: Some(vec![6, 6]),
        longform: Some(crate::LongFormOptions::default()),
        ..GgmlAsrExecutionOptions::default()
    };

    // A prior slice's decode interleaves word tokens (id 1) with its per-step
    // timestamp markers (at/above the first timestamp id). Only the words may
    // reach the next slice's carry; the prior slice's wall-clock offsets must
    // not leak across the boundary.
    let generated = vec![1, first_timestamp + 50, 6, first_timestamp + 120, 1];

    let carry =
        build_whisper_carry_prompt_token_ids(&tokenizer, &request_options, &generated, None)
            .expect("carry prompt tokens")
            .expect("carry prompt token ids");

    assert!(
        carry.iter().all(|id| *id < first_timestamp),
        "no per-step timestamp may be carried: {carry:?}"
    );
    // seed [6;2] ++ stripped generated [1,6,1]
    assert_eq!(carry.as_slice(), &[6, 6, 1, 6, 1]);
}

#[test]
fn build_whisper_carry_prompt_token_ids_empty_when_generated_is_all_timestamps() {
    let (_, tokenizer) = whisper_execution_and_tokenizer_fixture();
    let first_timestamp = tokenizer
        .first_timestamp_token_id()
        .expect("first timestamp id");
    let request_options = GgmlAsrExecutionOptions {
        prompt_token_ids: None,
        longform: Some(crate::LongFormOptions::default()),
        ..GgmlAsrExecutionOptions::default()
    };

    // A slice that emitted only timestamps (near-silence) leaves nothing to
    // carry; the caller then holds the previous context rather than a
    // timestamp-only seed.
    let generated = vec![first_timestamp, first_timestamp + 40];
    assert_eq!(
        build_whisper_carry_prompt_token_ids(&tokenizer, &request_options, &generated, None)
            .expect("valid"),
        None
    );
}

#[test]
fn build_whisper_carry_prompt_token_ids_refuses_a_loop_dominated_winner() {
    let (_, tokenizer) = whisper_execution_and_tokenizer_fixture();
    let first_timestamp = tokenizer
        .first_timestamp_token_id()
        .expect("first timestamp id");
    let request_options = GgmlAsrExecutionOptions {
        prompt_token_ids: Some(vec![1, 2]),
        longform: Some(crate::LongFormOptions::default()),
        ..GgmlAsrExecutionOptions::default()
    };

    // A winner that recited one short phrase over a groove: the cycle is
    // separated by per-step timestamps (so the in-decode guard, which reads
    // the raw token stream, may miss it) and the decode stopped mid-cycle.
    // Carrying it forward re-primes the next slice onto the same attractor,
    // so the carry must be refused and the previous context kept.
    let cycle = [3, 4, 5, 6];
    let mut generated = Vec::new();
    for _ in 0..3 {
        generated.extend_from_slice(&cycle);
        generated.push(first_timestamp + 50);
    }
    generated.extend_from_slice(&cycle[..2]);
    assert_eq!(
        build_whisper_carry_prompt_token_ids(&tokenizer, &request_options, &generated, None)
            .expect("valid"),
        None,
        "a loop-dominant winner must not be carried"
    );

    // The same phrase decoded only once (genuine speech) still carries.
    let generated = vec![3, first_timestamp + 50, 4, 5, 6];
    assert!(
        build_whisper_carry_prompt_token_ids(&tokenizer, &request_options, &generated, None)
            .expect("valid")
            .is_some()
    );
}

#[test]
fn build_whisper_carry_prompt_token_ids_strips_a_guard_trip_cycle() {
    let (_, tokenizer) = whisper_execution_and_tokenizer_fixture();
    let first_timestamp = tokenizer
        .first_timestamp_token_id()
        .expect("first timestamp id");
    let request_options = GgmlAsrExecutionOptions {
        prompt_token_ids: Some(vec![1, 2]),
        longform: Some(crate::LongFormOptions::default()),
        ..GgmlAsrExecutionOptions::default()
    };

    // A guard-cut stream: genuine speech, then the single kept occurrence of
    // the loop the driver truncated (keep_len = prefix + one cycle). After
    // the truncate the cycle is not detectable as a repeat - only the trip's
    // own ngram_len identifies it - so the carry must strip that many raw
    // tokens off the tail. The raw cycle carries an interleaved per-step
    // timestamp (5 raw tokens for 4 words), so the strip drops the cycle's
    // words plus one neighbouring content token: the loop's tokens must never
    // reach the next slice's prompt.
    let cycle_words = [3, 4, 5, 6];
    let generated = [
        9,
        8,
        7,
        first_timestamp + 50,
        9,
        8,
        7,
        first_timestamp + 120,
        cycle_words[0],
        cycle_words[1],
        first_timestamp + 200,
        cycle_words[2],
        cycle_words[3],
    ];
    let carry = build_whisper_carry_prompt_token_ids(
        &tokenizer,
        &request_options,
        &generated,
        Some(5), // raw cycle length: two words + timestamp + two words
    )
    .expect("valid")
    .expect("the genuine prefix still carries");

    // seed [1, 2] ++ stripped generated with the last 5 stripped tokens
    // removed: cycle words gone, no loop token left in the prompt tail.
    assert_eq!(carry.as_slice(), &[1, 2, 9, 8, 7, 9, 8]);
    assert!(
        !carry.iter().any(|id| cycle_words.contains(id)),
        "no loop token may be carried: {carry:?}"
    );

    // A guard-cut stream that kept only the loop (pure attractor) carries
    // nothing: the caller holds the previous slice's carry.
    assert_eq!(
        build_whisper_carry_prompt_token_ids(
            &tokenizer,
            &request_options,
            &[3, first_timestamp + 200, 4, 5],
            Some(4)
        )
        .expect("valid"),
        None
    );
}

#[test]
fn whisper_prompt_bounds_separate_current_prompt_from_future_carry() {
    let (execution, tokenizer) = whisper_execution_and_tokenizer_fixture();
    let request_options = GgmlAsrExecutionOptions {
        prompt_token_ids: Some(vec![1, 2, 3, 4]),
        longform: Some(crate::LongFormOptions::default()),
        ..GgmlAsrExecutionOptions::default()
    };
    let exact_only = super::super::prompt::whisper_prompt_position_bounds(
        &execution,
        &tokenizer,
        &request_options,
        0,
    )
    .expect("exact prompt bound");
    assert_eq!(exact_only.logical, exact_only.stable);

    let with_carry = super::super::prompt::whisper_prompt_position_bounds(
        &execution,
        &tokenizer,
        &request_options,
        WHISPER_LONGFORM_PROMPT_TOKEN_TAIL_LIMIT,
    )
    .expect("stable carry bound");
    assert_eq!(with_carry.logical, exact_only.logical);
    assert!(with_carry.stable > with_carry.logical);
}

#[test]
fn whisper_prompt_bounds_respect_mode_and_carry_as_independent_switches() {
    let (execution, tokenizer) = whisper_execution_and_tokenizer_fixture();
    let fixed_without_carry = GgmlAsrExecutionOptions {
        prompt_token_ids: Some(vec![1, 2, 3, 4]),
        longform: Some(crate::LongFormOptions {
            mode: crate::LongFormMode::Fixed,
            carry_prompt_across_slices: false,
            ..crate::LongFormOptions::default()
        }),
        ..GgmlAsrExecutionOptions::default()
    };
    let fixed = super::super::prompt::whisper_prompt_position_bounds(
        &execution,
        &tokenizer,
        &fixed_without_carry,
        WHISPER_LONGFORM_PROMPT_TOKEN_TAIL_LIMIT,
    )
    .expect("fixed prompt bound");
    assert_eq!(fixed.stable, fixed.logical);

    let disabled = GgmlAsrExecutionOptions {
        longform: Some(crate::LongFormOptions {
            mode: crate::LongFormMode::Off,
            ..crate::LongFormOptions::default()
        }),
        ..fixed_without_carry
    };
    let off = super::super::prompt::whisper_prompt_position_bounds(
        &execution,
        &tokenizer,
        &disabled,
        WHISPER_LONGFORM_PROMPT_TOKEN_TAIL_LIMIT,
    )
    .expect("disabled prompt bound");
    assert_eq!(off.stable, off.logical);
    assert_eq!(
        fixed.logical,
        off.logical + 1,
        "only active long-form adds <|startofprev|>"
    );
}

#[test]
fn whisper_carry_producer_honors_the_effective_carry_switch() {
    let (_, tokenizer) = whisper_execution_and_tokenizer_fixture();
    let request_options = GgmlAsrExecutionOptions {
        longform: Some(crate::LongFormOptions {
            carry_prompt_across_slices: false,
            ..crate::LongFormOptions::default()
        }),
        ..GgmlAsrExecutionOptions::default()
    };
    assert_eq!(
        build_whisper_carry_prompt_token_ids(&tokenizer, &request_options, &[1, 2, 3], None)
            .expect("disabled carry is valid"),
        None
    );
}

#[test]
fn whisper_serve_batch_requires_planner_reusable_graph() {
    let request_options = GgmlAsrExecutionOptions {
        longform: Some(crate::LongFormOptions::default()),
        ..GgmlAsrExecutionOptions::default()
    };

    assert!(!whisper_can_use_serve_batch(
        crate::ggml_runtime::GgmlDecodeReuseMode::FreshGraph,
        &request_options,
        true
    ));
    assert!(whisper_can_use_serve_batch(
        crate::ggml_runtime::GgmlDecodeReuseMode::ReusableGraph,
        &request_options,
        true
    ));
}

#[test]
fn whisper_serve_batch_rejects_unproven_reuse() {
    let request_options = GgmlAsrExecutionOptions::default();
    assert!(!whisper_can_use_serve_batch(
        crate::ggml_runtime::GgmlDecodeReuseMode::FreshGraph,
        &request_options,
        false
    ));
}

#[test]
fn decoder_quantized_tensor_is_indexed_in_quantized_source_map() {
    let mut tensors_f32_by_name = HashMap::new();
    let mut tensors_f16_bits_by_name = HashMap::new();
    let mut tensors_quantized_by_name = HashMap::new();
    insert_decoder_tensor_owned(
        &mut tensors_f32_by_name,
        &mut tensors_f16_bits_by_name,
        &mut tensors_quantized_by_name,
        WhisperMaterializedTensor {
            slot: WhisperGgufTensorSlot::DecoderLayerSelfAttnQWeight { layer_idx: 0 },
            tensor_name: "model.decoder.layers.0.self_attn.q_proj.weight".to_string(),
            source_ggml_type: 8,
            source_dims: vec![2, 2],
            dims: vec![2, 2],
            num_elements: 4,
            payload: WhisperMaterializedTensorPayload::Quantized {
                ggml_type: 8,
                bytes: vec![1, 2, 3, 4],
            },
        },
    )
    .expect("quantized decoder tensor should be indexed");
    assert!(tensors_f32_by_name.is_empty());
    assert!(tensors_f16_bits_by_name.is_empty());
    let (ggml_type, bytes) = tensors_quantized_by_name
        .get("model.decoder.layers.0.self_attn.q_proj.weight")
        .expect("quantized map must include tensor");
    assert_eq!(*ggml_type, 8);
    assert_eq!(bytes.as_ref(), &[1, 2, 3, 4]);
}

#[test]
fn decoder_quantized_tensor_with_empty_bytes_fails_closed() {
    let mut tensors_f32_by_name = HashMap::new();
    let mut tensors_f16_bits_by_name = HashMap::new();
    let mut tensors_quantized_by_name = HashMap::new();
    let error = insert_decoder_tensor_owned(
        &mut tensors_f32_by_name,
        &mut tensors_f16_bits_by_name,
        &mut tensors_quantized_by_name,
        WhisperMaterializedTensor {
            slot: WhisperGgufTensorSlot::DecoderLayerSelfAttnQWeight { layer_idx: 0 },
            tensor_name: "model.decoder.layers.0.self_attn.q_proj.weight".to_string(),
            source_ggml_type: 8,
            source_dims: vec![2, 2],
            dims: vec![2, 2],
            num_elements: 4,
            payload: WhisperMaterializedTensorPayload::Quantized {
                ggml_type: 8,
                bytes: Vec::new(),
            },
        },
    )
    .expect_err("empty quantized bytes must fail closed");
    assert!(matches!(
        error,
        WhisperGgmlExecutorError::DecoderWeightsMissing { .. }
    ));
    assert!(
        error
            .to_string()
            .contains("materialized quantized type 8 with empty bytes"),
        "unexpected error: {error}"
    );
}

#[test]
fn encoder_graph_upload_bytes_after_prepare_outputs_remains_supported() {
    const TEST_GGML_TYPE_F16: i32 = 1;
    let mut runner =
        GgmlCpuGraphRunner::new(GgmlCpuGraphConfig::conservative_default()).expect("runner");
    let mut graph = runner.start_graph();
    let output = graph.new_tensor_1d_f32(1, "output").expect("output tensor");
    let quantized_weight = graph
        .new_tensor_2d_typed(32, 1, TEST_GGML_TYPE_F16, "quantized_weight")
        .expect("quantized tensor");
    graph.set_input(output).expect("output input");
    graph.set_input(quantized_weight).expect("quantized input");
    graph.set_output(output).expect("set output");
    graph
        .prepare_outputs_for_upload(&[output])
        .expect("prepare outputs");

    let uploads = vec![
        WhisperEncoderGraphUpload::f32_owned(output, vec![0.0], "output"),
        WhisperEncoderGraphUpload::bytes(
            quantized_weight,
            vec![0_u8; 32 * std::mem::size_of::<u16>()],
            "quantized_weight",
        ),
    ];
    upload_encoder_graph_inputs(&mut graph, uploads).expect("upload bytes payload");

    let out = graph.compute_output_f32(output, 1).expect("compute output");
    assert_eq!(out, vec![0.0]);
}

#[test]
fn whisper_dtw_onset_lead_is_the_flat_baseline() {
    // The onset lead is the single flat constant (no density-scaled curve); the
    // runtime reads it via whisper_dtw_onset_lead, whose env-override fallback is
    // the compiled default. Pin the constant here rather than mutating process
    // env, which is unsafe in this edition and races under parallel nextest.
    assert!((WHISPER_DTW_ONSET_LEAD_SECONDS - 0.05).abs() < 1e-6);
}

#[test]
fn whisper_dtw_lead_silence_advance_fires_only_on_a_leading_leak() {
    let spf = 0.02_f32; // 1500 frames over a 30s window.
    let min_gap = WHISPER_DTW_LEAD_SILENCE_ADVANCE_MIN_GAP_SECONDS; // 0.2s -> 10 frames.

    // A run at the window front whose content onset sits well past the bound
    // (a leading silence leak) is advanced to that onset.
    let advance = whisper_dtw_lead_silence_advance_frame(0, Some(55), spf, min_gap); // 1.1s gap.
    assert_eq!(advance, Some(55));

    // The same onset but on a mid-run decoded `<|start|>` bound is NOT advanced:
    // the band_start == 0 gate is what keeps a real timestamp (which can mark a
    // large misalignment) from retargeting the lead word to an unrelated peak.
    let mid_run = whisper_dtw_lead_silence_advance_frame(300, Some(355), spf, min_gap); // 1.1s gap.
    assert_eq!(mid_run, None);

    // A window-front onset just over the minimum gap fires; a gap just under it
    // is normal `<|start|>` jitter, not a leak, so the sub-margin gate keeps it
    // untouched. (Values are kept clearly off the margin since the exact 0.2s
    // boundary is an arbitrary f32 knife-edge, not a meaningful threshold.)
    let over = whisper_dtw_lead_silence_advance_frame(0, Some(11), spf, min_gap); // 0.22s gap.
    assert_eq!(over, Some(11));
    let under = whisper_dtw_lead_silence_advance_frame(0, Some(9), spf, min_gap); // 0.18s gap.
    assert_eq!(under, None);

    // No usable content peak: nothing to advance to.
    let no_front = whisper_dtw_lead_silence_advance_frame(0, None, spf, min_gap);
    assert_eq!(no_front, None);
}

// ---------------------------------------------------------------------------
// whisper_refine_dtw_word_onsets
// ---------------------------------------------------------------------------

fn word_ts(word: &str, start: f32, end: f32) -> crate::WordTimestamp {
    crate::WordTimestamp {
        word: word.to_string(),
        start,
        end,
        confidence: None,
    }
}

/// A 15 s, 0.02 s/frame envelope (750 frames) at a 0.001 noise floor with a
/// single 0.5 peak at 8 s that sets the clip peak (and so the 5% silence
/// ceiling). The [2.0, 4.0) word-b window is filled from 0.001 up to 3.8 s and
/// a 0.25 speech onset occupies [3.8, 4.0).
fn refine_fixture_envelope() -> Vec<f32> {
    let mut env = vec![0.001f32; 750];
    env[400] = 0.5;
    for s in env[190..200].iter_mut() {
        *s = 0.25;
    }
    env
}

/// `whisper_refine_dtw_word_onsets` advances a word the fold parked in true
/// zero-silence to its real onset at 3.8 s: the previous word's boundary is
/// left untouched, so a real gap is opened where the pause sits.
#[test]
fn refine_dtw_onsets_pushes_true_silence_word_to_its_onset() {
    let words = vec![word_ts("a", 0.5, 0.6), word_ts("b", 2.0, 4.0)];
    let env = refine_fixture_envelope();
    let out = whisper_refine_dtw_word_onsets(words, Some(&env), 15.0);
    assert!((out[1].start - 3.8).abs() < 0.05, "start={}", out[1].start);
    // The first word is never modified.
    assert!((out[0].start - 0.5).abs() < 1e-4 && (out[0].end - 0.6).abs() < 1e-4);
}

/// The same window with a low music floor filling the front half (a sustained
/// level, so the front's mean sits above the floor) is *not* trusted as a
/// pause: a quiet passage over a music bed is ambiguous, so no push fires and
/// the word keeps its fold position. This is the gate that stops the refinement
/// from regressing continuous-speech / music-backed clips.
#[test]
fn refine_dtw_onsets_refuses_a_music_floor_front() {
    let mut env = refine_fixture_envelope();
    for s in env[100..190].iter_mut() {
        *s = 0.021;
    }
    let words = vec![word_ts("a", 0.5, 0.6), word_ts("b", 2.0, 4.0)];
    let out = whisper_refine_dtw_word_onsets(words, Some(&env), 15.0);
    assert!((out[1].start - 2.0).abs() < 1e-4, "start={}", out[1].start);
    assert!((out[1].end - 4.0).abs() < 1e-4);
}

/// A boundary word whose start maps to or past the last envelope frame -- common
/// at a longform slice end, where the frame array is shorter than
/// `duration_s / seconds_per_frame` -- must not overrun the slice. Pre-fix this
/// indexed out of bounds and panicked (`range end index ... out of range`);
/// post-fix the word is clamped into range and, finding no usable window, is
/// left unrefined rather than aborting the run.
#[test]
fn refine_dtw_onsets_clamps_a_word_at_or_past_the_end() {
    let env = refine_fixture_envelope(); // 750 frames
    let words = vec![
        word_ts("a", 0.5, 0.6),
        word_ts("b", 15.5, 16.0), // start past the 750-frame end, span 0.5 >= 0.3
    ];
    // duration_s larger than the envelope implies so `start_s / spf` overshoots.
    let out = whisper_refine_dtw_word_onsets(words, Some(&env), 16.0);
    assert_eq!(out[1].start, 15.5, "unrefined; must not panic");
    assert_eq!(out[1].end, 16.0, "unrefined; must not panic");
}

/// No envelope (a run without cross-attention word timestamps) is a byte-exact
/// no-op.
#[test]
fn refine_dtw_onsets_noop_without_envelope() {
    let words = vec![word_ts("a", 0.5, 0.6), word_ts("b", 2.0, 4.0)];
    let out = whisper_refine_dtw_word_onsets(words, None, 15.0);
    assert_eq!(out[0].start, 0.5);
    assert_eq!(out[1].start, 2.0);
    assert_eq!(out[1].end, 4.0);
}

// ---------------------------------------------------------------------------
// whisper_refine_dtw_word_offsets
// ---------------------------------------------------------------------------

/// A 15 s, 0.02 s/frame envelope (750 frames) at a 0.001 noise floor with a
/// single 0.5 peak at 8 s that sets the clip peak (and so the 5% silence
/// ceiling). The [2.0, 4.0) word window has a 0.25 speech run in [2.0, 2.6)
/// followed by digital-zero silence to 4.0 s -- the trailing-silence (hollow
/// back) shape the offset refinement retreats.
fn offset_fixture_envelope() -> Vec<f32> {
    let mut env = vec![0.001f32; 750];
    env[400] = 0.5;
    for s in env[100..130].iter_mut() {
        *s = 0.25;
    }
    env
}

/// `whisper_refine_dtw_word_offsets` retreats a word the fold let run past its
/// speech into the trailing silence back to its real offset at 2.6 s: the next
/// word's start is left untouched, so a real gap is opened where the pause sits.
#[test]
fn refine_dtw_offsets_pulls_true_silence_word_to_its_offset() {
    let words = vec![
        word_ts("a", 0.5, 0.6),
        word_ts("b", 2.0, 4.0),
        word_ts("c", 4.0, 4.5),
    ];
    let env = offset_fixture_envelope();
    let out = whisper_refine_dtw_word_offsets(words, Some(&env), 15.0);
    assert!((out[1].end - 2.6).abs() < 0.05, "end={}", out[1].end);
    // start is untouched, as is the next word.
    assert!((out[1].start - 2.0).abs() < 1e-4 && (out[2].end - 4.5).abs() < 1e-4);
}

/// The same window but with a low music floor filling the back half (a
/// sustained level, so the back's mean sits above the floor) is *not* trusted
/// as trailing silence: a quiet passage over a music bed is ambiguous, so no
/// pull fires and the word keeps its fold position.
#[test]
fn refine_dtw_offsets_refuses_a_music_floor_back() {
    let mut env = offset_fixture_envelope();
    for s in env[150..200].iter_mut() {
        *s = 0.021;
    }
    let words = vec![
        word_ts("a", 0.5, 0.6),
        word_ts("b", 2.0, 4.0),
        word_ts("c", 4.0, 4.5),
    ];
    let out = whisper_refine_dtw_word_offsets(words, Some(&env), 15.0);
    assert!((out[1].end - 4.0).abs() < 1e-4, "end={}", out[1].end);
    assert!((out[1].start - 2.0).abs() < 1e-4);
}

/// A middle word whose end maps to or past the last envelope frame -- common at
/// a longform slice end where the frame array is shorter than
/// `duration_s / seconds_per_frame` -- must not overrun the slice. The word is
/// clamped into range and, finding no usable window, is left unrefined rather
/// than aborting the run.
#[test]
fn refine_dtw_offsets_clamps_a_word_at_or_past_the_end() {
    let env = offset_fixture_envelope(); // 750 frames
    let words = vec![
        word_ts("a", 0.5, 0.6),
        word_ts("b", 15.4, 16.0), // end past the 750-frame end, span 0.6 >= 0.3
        word_ts("c", 16.0, 16.2), // the last word is skipped regardless
    ];
    // duration_s larger than the envelope implies so `end_s / spf` overshoots.
    let out = whisper_refine_dtw_word_offsets(words, Some(&env), 16.0);
    assert_eq!(out[1].end, 16.0, "unrefined; must not panic");
}

/// The last word is never retreated: its true end is the audio end, so the
/// silence after it is the clip's legitimate tail, not a fold leak.
#[test]
fn refine_dtw_offsets_skips_the_last_word() {
    let mut env = offset_fixture_envelope();
    for s in env[0..600].iter_mut() {
        *s = 0.25; // trailing word's window [11.5, 13.5) sits in speech to its end
    }
    let words = vec![
        word_ts("a", 0.5, 0.6),
        word_ts("b", 11.5, 13.5), // the last word
    ];
    let out = whisper_refine_dtw_word_offsets(words, Some(&env), 15.0);
    assert_eq!(out[1].end, 13.5, "last word untouched");
}

/// No envelope (a run without cross-attention word timestamps) is a byte-exact
/// no-op.
#[test]
fn refine_dtw_offsets_noop_without_envelope() {
    let words = vec![
        word_ts("a", 0.5, 0.6),
        word_ts("b", 2.0, 4.0),
        word_ts("c", 4.0, 4.5),
    ];
    let out = whisper_refine_dtw_word_offsets(words, None, 15.0);
    assert_eq!(out[1].end, 4.0);
}

// ---------------------------------------------------------------------------
// whisper_pad_dtw_word_windows
// ---------------------------------------------------------------------------

/// An interior word is widened on both sides by exactly the pads, while the
/// first word's start is clamped to 0.0 and the last word's end is clamped to
/// the audio duration; interior order is preserved.
#[test]
fn pad_dtw_word_windows_widens_toward_the_edges_and_clamps_to_the_audio() {
    let words = vec![
        word_ts("a", 0.02, 0.70),
        word_ts("b", 0.70, 1.30),
        word_ts("c", 1.30, 14.98),
    ];
    let out = whisper_pad_dtw_word_windows(words, 15.0);
    assert!(
        (out[0].start - 0.0).abs() < 1e-4,
        "a.start={}",
        out[0].start
    );
    assert!((out[0].end - 0.80).abs() < 1e-4, "a.end={}", out[0].end);
    assert!(
        (out[1].start - 0.60).abs() < 1e-4,
        "b.start={}",
        out[1].start
    );
    assert!((out[1].end - 1.40).abs() < 1e-4, "b.end={}", out[1].end);
    assert!(
        (out[2].start - 1.20).abs() < 1e-4,
        "c.start={}",
        out[2].start
    );
    assert!((out[2].end - 15.0).abs() < 1e-4, "c.end={}", out[2].end);
    for (index, word) in out.iter().enumerate() {
        assert!(word.start <= word.end, "word[{index}] inverted");
        if index + 1 < out.len() {
            assert!(word.end <= out[index + 1].end);
        }
    }
}

/// Zero-duration audio clamps every window edge to 0.0 and keeps each window
/// non-negative; an empty input is a byte-exact no-op.
#[test]
fn pad_dtw_word_windows_is_a_noop_when_empty_and_clamps_zero_duration() {
    let empty = whisper_pad_dtw_word_windows(Vec::new(), 15.0);
    assert!(empty.is_empty());
    let zero = whisper_pad_dtw_word_windows(vec![word_ts("a", 0.30, 0.90)], 0.0);
    assert_eq!(zero[0].start, 0.0);
    assert_eq!(zero[0].end, 0.0);
}

fn tail_repeat_is_timestamp(tokenizer: &WhisperTokenizer) -> impl Fn(u32) -> bool {
    let first_timestamp = tokenizer
        .first_timestamp_token_id()
        .expect("first timestamp id");
    move |token_id: u32| token_id >= first_timestamp
}

/// A long clause emitted verbatim twice within the final text run (a 2x
/// repeat of a block at least the min block long) is the no-speech
/// hallucination the pass targets: the range to remove is exactly the second
/// copy, leaving the first copy and the trailing timestamp.
#[test]
fn tail_repeat_range_collapses_a_within_run_verbatim_twice_repeat() {
    let (_, tokenizer) = whisper_execution_and_tokenizer_fixture();
    let is_timestamp = tail_repeat_is_timestamp(&tokenizer);
    let first_timestamp = tokenizer
        .first_timestamp_token_id()
        .expect("first timestamp id");
    // [ts_start] [a b c d e f] [a b c d e f] [ts_end]; single run of 12.
    let tokens = vec![
        first_timestamp,
        1,
        2,
        3,
        4,
        5,
        6,
        1,
        2,
        3,
        4,
        5,
        6,
        first_timestamp + 1,
    ];
    let removal = whisper_repeated_block_removal(&tokens, &is_timestamp).expect("a repeat");
    // The repeated unit is 6; drop the trailing copy -> indices 7..13.
    assert_eq!(removal.removed_range, (7, 13));
    assert_eq!(&tokens[..7], &[first_timestamp, 1, 2, 3, 4, 5, 6]);
}

/// A long clause emitted verbatim twice as two adjacent text runs (each a
/// single `<|start|> text <|end|>` segment separated by a mid-segment
/// timestamp) is the other no-speech hallucination shape: the range to remove
/// is exactly the second run, leaving the first run and its bracketing
/// timestamps.
#[test]
fn tail_repeat_range_collapses_two_verbatim_siblings_to_one_copy() {
    let (_, tokenizer) = whisper_execution_and_tokenizer_fixture();
    let is_timestamp = tail_repeat_is_timestamp(&tokenizer);
    let first_timestamp = tokenizer
        .first_timestamp_token_id()
        .expect("first timestamp id");
    // [ts] [a b c d e f] [ts_mid] [a b c d e f] [ts_end]; two runs of 6.
    let tokens = vec![
        first_timestamp,
        1,
        2,
        3,
        4,
        5,
        6,
        first_timestamp + 8,
        1,
        2,
        3,
        4,
        5,
        6,
        first_timestamp + 24,
    ];
    let removal = whisper_repeated_block_removal(&tokens, &is_timestamp).expect("a repeat");
    // Drop the second run: [lo_last, hi_last + 1) -> indices 8..14.
    assert_eq!(removal.removed_range, (8, 14));
    assert_eq!(
        &tokens[..removal.removed_range.0],
        &[first_timestamp, 1, 2, 3, 4, 5, 6, first_timestamp + 8]
    );
    assert_eq!(&tokens[removal.removed_range.1..], &[first_timestamp + 24]);
}

/// A clause a couple of tokens too short to be the no-speech shape (its
/// repeating unit is under the min block) is genuine backchannel and must not
/// collapse.
#[test]
fn tail_repeat_range_leaves_sub_min_block_repeats_alone() {
    let (_, tokenizer) = whisper_execution_and_tokenizer_fixture();
    let is_timestamp = tail_repeat_is_timestamp(&tokenizer);
    let first_timestamp = tokenizer
        .first_timestamp_token_id()
        .expect("first timestamp id");
    // "hoo-ah hoo-ah": unit n=3 < 6 -> no collapse.
    let short = vec![first_timestamp, 1, 2, 3, 1, 2, 3, first_timestamp];
    assert!(whisper_repeated_block_removal(&short, &is_timestamp).is_none());
}

/// A sentence that merely rhymes with the next (shares the tail but not a
/// verbatim block) is not a hallucination and must survive.
#[test]
fn tail_repeat_range_leaves_non_verbatim_tails_alone() {
    let (_, tokenizer) = whisper_execution_and_tokenizer_fixture();
    let is_timestamp = tail_repeat_is_timestamp(&tokenizer);
    let first_timestamp = tokenizer
        .first_timestamp_token_id()
        .expect("first timestamp id");
    let tokens = vec![
        first_timestamp,
        1,
        2,
        3,
        4,
        5,
        6,
        1,
        2,
        3,
        4,
        9,
        8,
        first_timestamp + 1,
    ];
    assert!(whisper_repeated_block_removal(&tokens, &is_timestamp).is_none());
}

/// A repeated long block in an earlier run is untouched; only the final run's
/// tail is eligible (a mid-clip repeat the prior slice already heard is real).
#[test]
fn tail_repeat_range_only_considers_the_final_text_run() {
    let (_, tokenizer) = whisper_execution_and_tokenizer_fixture();
    let is_timestamp = tail_repeat_is_timestamp(&tokenizer);
    let first_timestamp = tokenizer
        .first_timestamp_token_id()
        .expect("first timestamp id");
    // run1 = [a x6 a x6] (its own repeat) then ts, then run2 = [a b c] (no tail repeat).
    let tokens = vec![
        first_timestamp,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        first_timestamp + 9,
        5,
        6,
        7,
        first_timestamp + 15,
    ];
    assert!(
        whisper_repeated_block_removal(&tokens, &is_timestamp).is_none(),
        "only the final run's tail may collapse"
    );
}
/// A tiny rms envelope whose first half is loud and second half silent,
/// 0.02 s/frame, 1500 frames (the encoder frame space). Used by the collapse
/// tests so the acoustic gate reads real levels.
fn tail_repeat_silence_second_half_rms() -> Vec<f32> {
    let mut rms = vec![0.05_f32; 1500];
    for (i, slot) in rms.iter_mut().enumerate() {
        if i < 50 {
            *slot = 0.2_f32;
        } else if i < 100 {
            *slot = 0.002_f32;
        }
    }
    rms
}

/// A tiny rms envelope that is loud throughout the bracket window (a genuine
/// repeated line has real speech in both copies), so the gate must refuse.
fn tail_repeat_loud_everywhere_rms() -> Vec<f32> {
    (0..1500)
        .map(|i| if i < 100 { 0.2_f32 } else { 0.05_f32 })
        .collect()
}

fn tail_repeat_alignments(tokens: &[u32]) -> Vec<WhisperGeneratedTokenAlignment> {
    tokens
        .iter()
        .map(|&token_id| WhisperGeneratedTokenAlignment {
            token_id,
            frame_probs: vec![0.5_f32],
        })
        .collect()
}

fn tail_repeat_decode(
    tokens: Vec<u32>,
    stop_reason: Seq2SeqGreedyDecodeStopReason,
) -> WhisperGreedyDecodeResult {
    WhisperGreedyDecodeResult {
        generated_probabilities: vec![0.9_f32; tokens.len()],
        generated_tokens: tokens,
        text: String::new(),
        stop_reason,
        guard_trip_ngram_len: None,
    }
}

/// No-speech shape: a long clause emitted verbatim twice within the final text
/// run, second copy over near-silence. The collapse splices the tokens, the
/// parallel probabilities, the cross-attention alignments, and re-derives the
/// (single-copy) text.
#[test]
fn collapse_repeated_tail_splices_tokens_probs_alignments_and_text() {
    let (_, tokenizer) = whisper_execution_and_tokenizer_fixture();
    let first_timestamp = tokenizer
        .first_timestamp_token_id()
        .expect("first timestamp id");
    // [ts(0)] [a b c d e f] [a b c d e f] [ts(100)]; single run of 12, second
    // copy sits over silence.
    let tokens = vec![
        first_timestamp,
        1,
        2,
        3,
        4,
        5,
        6,
        1,
        2,
        3,
        4,
        5,
        6,
        first_timestamp + 100,
    ];
    let rms = tail_repeat_silence_second_half_rms();
    let mut alignments = tail_repeat_alignments(&tokens);
    let mut decode = tail_repeat_decode(tokens.clone(), Seq2SeqGreedyDecodeStopReason::StopToken);
    let result =
        whisper_collapse_repeated_decode_tail(&tokenizer, &mut decode, &mut alignments, Some(&rms))
            .expect("collapse");
    assert!(result, "no-speech second copy must collapse");
    // The second copy (tokens 7..13) is gone; first copy + both ts remain.
    assert_eq!(
        decode.generated_tokens,
        vec![first_timestamp, 1, 2, 3, 4, 5, 6, first_timestamp + 100]
    );
    assert_eq!(decode.generated_probabilities.len(), 8);
    assert_eq!(alignments.len(), 8);
    assert_eq!(
        alignments.iter().map(|a| a.token_id).collect::<Vec<_>>(),
        decode.generated_tokens.as_slice()
    );
    // Text re-derived from the surviving text tokens (one copy), not doubled.
    let expected_text = tokenizer
        .decode_text_token_ids(&[1, 2, 3, 4, 5, 6])
        .expect("decode text");
    assert_eq!(decode.text, expected_text);
    assert!(
        !decode
            .text
            .contains(&format!("{expected_text}{expected_text}")),
        "text should not contain both copies: {:?}",
        decode.text
    );
}

/// Genuine repeated line: the same clause emitted twice but both copies sit
/// over real speech (equal level). The acoustic gate refuses, so the decode is
/// left byte-identical.
#[test]
fn collapse_repeated_tail_is_a_noop_when_both_copies_are_real_speech() {
    let (_, tokenizer) = whisper_execution_and_tokenizer_fixture();
    let first_timestamp = tokenizer
        .first_timestamp_token_id()
        .expect("first timestamp id");
    let tokens = vec![
        first_timestamp,
        1,
        2,
        3,
        4,
        5,
        6,
        1,
        2,
        3,
        4,
        5,
        6,
        first_timestamp + 100,
    ];
    let rms = tail_repeat_loud_everywhere_rms();
    let mut alignments = tail_repeat_alignments(&tokens);
    let mut decode = tail_repeat_decode(tokens.clone(), Seq2SeqGreedyDecodeStopReason::StopToken);
    let before = tokenizer.decode_text_token_ids(&tokens).expect("decode");
    decode.text = before.clone();
    let result =
        whisper_collapse_repeated_decode_tail(&tokenizer, &mut decode, &mut alignments, Some(&rms))
            .expect("no-op");
    assert!(!result, "a genuine repeat must not collapse");
    assert_eq!(decode.generated_tokens, tokens);
    assert_eq!(decode.text, before);
}

/// A repeated long clause emitted as two adjacent `<|start|>/<|end|>` runs
/// (the cross-run shape), second over silence: the collapse splices the second
/// run and keeps the first run plus the bracketing timestamps.
#[test]
fn collapse_repeated_tail_splices_two_verbatim_runs_and_keeps_bracketing_ts() {
    let (_, tokenizer) = whisper_execution_and_tokenizer_fixture();
    let first_timestamp = tokenizer
        .first_timestamp_token_id()
        .expect("first timestamp id");
    // [ts(0)] [a b c d e f] [ts(50)] [a b c d e f] [ts(100)]; two runs of 6,
    // the second over silence.
    let tokens = vec![
        first_timestamp,
        1,
        2,
        3,
        4,
        5,
        6,
        first_timestamp + 50,
        1,
        2,
        3,
        4,
        5,
        6,
        first_timestamp + 100,
    ];
    let rms = tail_repeat_silence_second_half_rms();
    let mut alignments = tail_repeat_alignments(&tokens);
    let mut decode = tail_repeat_decode(tokens.clone(), Seq2SeqGreedyDecodeStopReason::StopToken);
    let result =
        whisper_collapse_repeated_decode_tail(&tokenizer, &mut decode, &mut alignments, Some(&rms))
            .expect("collapse");
    assert!(result, "no-speech second run must collapse");
    assert_eq!(
        decode.generated_tokens,
        vec![
            first_timestamp,
            1,
            2,
            3,
            4,
            5,
            6,
            first_timestamp + 50,
            first_timestamp + 100
        ]
    );
    assert_eq!(decode.generated_probabilities.len(), 9);
    assert_eq!(alignments.len(), 9);
    assert_eq!(
        alignments.iter().map(|a| a.token_id).collect::<Vec<_>>(),
        decode.generated_tokens.as_slice()
    );
    let expected_text = tokenizer
        .decode_text_token_ids(&[1, 2, 3, 4, 5, 6])
        .expect("decode text");
    assert_eq!(decode.text, expected_text);
}

/// No stop token, no collapse: a guard-cut decode ends on its own stop reason,
/// so even a verbatim repeated tail is left untouched (the cut prefix was a
/// salvage, not a finished hallucination).
#[test]
fn collapse_repeated_tail_is_a_noop_on_a_non_stop_reason() {
    let (_, tokenizer) = whisper_execution_and_tokenizer_fixture();
    let first_timestamp = tokenizer
        .first_timestamp_token_id()
        .expect("first timestamp id");
    let tokens = vec![
        first_timestamp,
        1,
        2,
        3,
        4,
        5,
        6,
        1,
        2,
        3,
        4,
        5,
        6,
        first_timestamp + 100,
    ];
    let rms = tail_repeat_silence_second_half_rms();
    let mut alignments = tail_repeat_alignments(&tokens);
    let mut decode = tail_repeat_decode(
        tokens.clone(),
        Seq2SeqGreedyDecodeStopReason::DegenerateRepeatGuard,
    );
    let before = tokenizer.decode_text_token_ids(&tokens).expect("decode");
    decode.text = before.clone();
    let result =
        whisper_collapse_repeated_decode_tail(&tokenizer, &mut decode, &mut alignments, Some(&rms))
            .expect("no-op");
    assert!(!result, "a guard-cut decode must not collapse");
    assert_eq!(decode.generated_tokens, tokens);
    assert_eq!(decode.text, before);
}

/// Layer (a): the placed words of a run that all folded onto one instant (a
/// 0.1 s span after the word-window pad) are a small fraction of the bracket
/// band, so the run is collapsed; words that visibly take their share of the
/// band are not.
#[test]
fn degenerate_tail_span_collapsed_fires_below_the_band_fraction() {
    // The ali tail shape: words on one point, the 3.18 s tail-slice band.
    assert!(whisper_degenerate_tail_span_collapsed(0.1, 3.18));
    // The fraction boundary: exactly 10% of the band is not below it.
    assert!(!whisper_degenerate_tail_span_collapsed(0.1 * 3.18, 3.18));
    assert!(whisper_degenerate_tail_span_collapsed(
        0.1 * 3.18 - 1e-9,
        3.18
    ));
    // A genuine short closing run IS span-collapsed on a wide band -- the
    // layer (b) envelope is what protects it.
    assert!(whisper_degenerate_tail_span_collapsed(1.0, 27.0));
    assert!(!whisper_degenerate_tail_span_collapsed(2.7, 27.0));
}

/// Layer (a) floor: with a degenerate (zero-width) bracket the absolute
/// 0.1 s floor is the only tolerance left.
#[test]
fn degenerate_tail_span_collapsed_uses_the_absolute_floor_on_a_zero_band() {
    assert!(whisper_degenerate_tail_span_collapsed(0.05, 0.0));
    assert!(!whisper_degenerate_tail_span_collapsed(0.1, 0.0));
    assert!(!whisper_degenerate_tail_span_collapsed(1.5, 0.0));
}

/// Layer (b): a sustained speech run (5+ consecutive 0.02 s frames above
/// median+5dB) anywhere in the region is real audio under the words; the
/// same level outside the region leaves it confirmed silent.
#[test]
fn degenerate_tail_region_silence_flags_sustained_speech_in_the_region() {
    // 159 frames at floor; speech (0.01, far above floor*10^0.25) on frames
    // 150..159.
    let mut levels = vec![0.0005_f32; 159];
    for sample in levels[150..159].iter_mut() {
        *sample = 0.01_f32;
    }
    assert_eq!(
        whisper_degenerate_tail_region_silence(Some(&levels), 3.03, 3.33, 3.18),
        Some(true)
    );
    // The same speech at the slice head leaves the tail region silent.
    let mut head_speech = vec![0.0005_f32; 159];
    for sample in head_speech[..50].iter_mut() {
        *sample = 0.01_f32;
    }
    assert_eq!(
        whisper_degenerate_tail_region_silence(Some(&head_speech), 3.03, 3.33, 3.18),
        Some(false)
    );
}

/// Layer (b) fail-open: a missing envelope, an immeasurably short region, and
/// a blip shorter than the sustain floor all refuse the verdict the splice
/// needs.
#[test]
fn degenerate_tail_region_silence_refuses_when_silence_cannot_be_confirmed() {
    let quiet = vec![0.0005_f32; 159];
    assert_eq!(
        whisper_degenerate_tail_region_silence(None, 3.03, 3.33, 3.18),
        None
    );
    assert_eq!(
        whisper_degenerate_tail_region_silence(Some(&quiet), 3.03, 3.33, 0.0),
        None
    );
    // The region clips down to fewer than four frames: not measurable.
    assert_eq!(
        whisper_degenerate_tail_region_silence(Some(&quiet), 3.13, 3.33, 3.18),
        None
    );
    // Four consecutive frames above the floor is a blip, not a sustain.
    let mut blip = vec![0.0005_f32; 159];
    for sample in blip[150..154].iter_mut() {
        *sample = 0.01_f32;
    }
    assert_eq!(
        whisper_degenerate_tail_region_silence(Some(&blip), 3.03, 3.33, 3.18),
        Some(false)
    );
}

// ---------------------------------------------------------------------------
// whisper_splice_degenerate_tail_run
// ---------------------------------------------------------------------------

/// The full pass: a clean-stop decode whose final run's words all fold onto
/// the window-end instant over confirmed silence is spliced out of the
/// tokens, the probabilities, and the alignments, its placed words are
/// drained from the word list, and the re-derived text is empty -- the
/// caller's empty-text degrade then reports the slice as honest no-speech.
#[test]
fn splice_degenerate_tail_run_splices_a_folded_tail_run_over_silence() {
    let (_, tokenizer) = whisper_execution_and_tokenizer_fixture();
    let ts = tokenizer
        .first_timestamp_token_id()
        .expect("first timestamp id");
    // [ts(0)] [hallu 6] [ts(159)]: one run brackets the whole 3.18 s tail
    // slice, its five placed words all fold onto the window end.
    let tokens = vec![ts, 1, 2, 3, 4, 5, 6, ts + 159];
    let mut alignments = tail_repeat_alignments(&tokens);
    let mut decode = tail_repeat_decode(tokens.clone(), Seq2SeqGreedyDecodeStopReason::StopToken);
    decode.text = tokenizer
        .decode_text_token_ids(&tokens)
        .expect("decode text");
    let mut words = vec![
        word_ts("a", 3.18, 3.18),
        word_ts("b", 3.18, 3.18),
        word_ts("c", 3.18, 3.18),
        word_ts("d", 3.18, 3.18),
        word_ts("e", 3.18, 3.18),
    ];
    let quiet = vec![0.0005_f32; 159];
    let spliced = whisper_splice_degenerate_tail_run(
        &tokenizer,
        &mut decode,
        &mut alignments,
        &mut words,
        &[(0, 5)],
        159,
        Some(&quiet),
        3.18,
    )
    .expect("splice");
    assert!(
        spliced,
        "a folded run over confirmed silence must be spliced"
    );
    assert_eq!(decode.generated_tokens, vec![ts, ts + 159]);
    assert_eq!(decode.generated_probabilities.len(), 2);
    assert_eq!(alignments.len(), 2);
    assert!(
        decode.text.trim().is_empty(),
        "no text tokens survive: {:?}",
        decode.text
    );
    assert!(words.is_empty(), "the run's placed words must be drained");
}

/// A real first run survives in the tokens, the text, and the word list: only
/// the folded final run's own words are drained.
#[test]
fn splice_degenerate_tail_run_keeps_earlier_runs_and_their_words() {
    let (_, tokenizer) = whisper_execution_and_tokenizer_fixture();
    let ts = tokenizer
        .first_timestamp_token_id()
        .expect("first timestamp id");
    // [ts(0)] [real 3] [ts(100)] [ts(150)] [hallu 3] [ts(159)]: the final run
    // folds two placed words onto the window end over silence.
    let tokens = vec![ts, 1, 2, 3, ts + 100, ts + 150, 4, 5, 6, ts + 159];
    let mut alignments = tail_repeat_alignments(&tokens);
    let mut decode = tail_repeat_decode(tokens.clone(), Seq2SeqGreedyDecodeStopReason::StopToken);
    decode.text = tokenizer
        .decode_text_token_ids(&tokens)
        .expect("decode text");
    let mut words = vec![
        word_ts("a", 0.10, 0.30),
        word_ts("b", 0.30, 0.70),
        word_ts("c", 0.70, 1.10),
        word_ts("d", 3.18, 3.18),
        word_ts("e", 3.18, 3.18),
    ];
    let quiet = vec![0.0005_f32; 159];
    let spliced = whisper_splice_degenerate_tail_run(
        &tokenizer,
        &mut decode,
        &mut alignments,
        &mut words,
        &[(0, 3), (3, 5)],
        159,
        Some(&quiet),
        3.18,
    )
    .expect("splice");
    assert!(spliced);
    assert_eq!(
        decode.generated_tokens,
        vec![ts, 1, 2, 3, ts + 100, ts + 150, ts + 159]
    );
    assert_eq!(decode.generated_probabilities.len(), 7);
    assert_eq!(alignments.len(), 7);
    assert_eq!(words.len(), 3);
    assert_eq!(words[0].word, "a");
    let expected_text = tokenizer
        .decode_text_token_ids(&[1, 2, 3])
        .expect("decode text");
    assert_eq!(decode.text, expected_text);
}

/// The layer (b) gate: a sustained speech run under a shape-collapsed final
/// run is real audio (a short closing line), so the splice must refuse.
#[test]
fn splice_degenerate_tail_run_is_a_noop_with_sustained_speech_under_the_run() {
    let (_, tokenizer) = whisper_execution_and_tokenizer_fixture();
    let ts = tokenizer
        .first_timestamp_token_id()
        .expect("first timestamp id");
    let tokens = vec![ts, 1, 2, 3, 4, 5, 6, ts + 159];
    let mut alignments = tail_repeat_alignments(&tokens);
    let mut decode = tail_repeat_decode(tokens.clone(), Seq2SeqGreedyDecodeStopReason::StopToken);
    decode.text = tokenizer
        .decode_text_token_ids(&tokens)
        .expect("decode text");
    let mut words = vec![word_ts("a", 3.18, 3.18), word_ts("b", 3.18, 3.18)];
    let mut speech = vec![0.0005_f32; 159];
    for sample in speech[140..159].iter_mut() {
        *sample = 0.01_f32;
    }
    let spliced = whisper_splice_degenerate_tail_run(
        &tokenizer,
        &mut decode,
        &mut alignments,
        &mut words,
        &[(0, 2)],
        159,
        Some(&speech),
        3.18,
    )
    .expect("no-op");
    assert!(!spliced, "sustained speech under the run is real audio");
    assert_eq!(decode.generated_tokens, tokens);
    assert_eq!(words.len(), 2);
}

/// A final run whose words actually spread over the band is not span-
/// collapsed, so the decode is left byte-identical.
#[test]
fn splice_degenerate_tail_run_is_a_noop_on_spread_words() {
    let (_, tokenizer) = whisper_execution_and_tokenizer_fixture();
    let ts = tokenizer
        .first_timestamp_token_id()
        .expect("first timestamp id");
    let tokens = vec![ts, 1, 2, 3, 4, 5, 6, ts + 159];
    let mut alignments = tail_repeat_alignments(&tokens);
    let mut decode = tail_repeat_decode(tokens.clone(), Seq2SeqGreedyDecodeStopReason::StopToken);
    decode.text = tokenizer
        .decode_text_token_ids(&tokens)
        .expect("decode text");
    let mut words = vec![
        word_ts("a", 0.40, 0.80),
        word_ts("b", 0.80, 1.20),
        word_ts("c", 1.20, 1.80),
    ];
    let quiet = vec![0.0005_f32; 159];
    let spliced = whisper_splice_degenerate_tail_run(
        &tokenizer,
        &mut decode,
        &mut alignments,
        &mut words,
        &[(0, 3)],
        159,
        Some(&quiet),
        3.18,
    )
    .expect("no-op");
    assert!(!spliced, "a run spread over its band must survive");
    assert_eq!(decode.generated_tokens, tokens);
    assert_eq!(words.len(), 3);
}

/// One placed word cannot be told apart from a tight bracket on a short
/// closing word: the min-words gate keeps it, even over confirmed silence.
#[test]
fn splice_degenerate_tail_run_is_a_noop_on_a_single_word_run() {
    let (_, tokenizer) = whisper_execution_and_tokenizer_fixture();
    let ts = tokenizer
        .first_timestamp_token_id()
        .expect("first timestamp id");
    let tokens = vec![ts, 1, 2, ts + 159];
    let mut alignments = tail_repeat_alignments(&tokens);
    let mut decode = tail_repeat_decode(tokens.clone(), Seq2SeqGreedyDecodeStopReason::StopToken);
    decode.text = tokenizer
        .decode_text_token_ids(&tokens)
        .expect("decode text");
    let mut words = vec![word_ts("bye", 3.18, 3.18)];
    let quiet = vec![0.0005_f32; 159];
    let spliced = whisper_splice_degenerate_tail_run(
        &tokenizer,
        &mut decode,
        &mut alignments,
        &mut words,
        &[(0, 1)],
        159,
        Some(&quiet),
        3.18,
    )
    .expect("no-op");
    assert!(!spliced, "a single placed word must survive");
    assert_eq!(decode.generated_tokens, tokens);
    assert_eq!(words.len(), 1);
}

/// A guard-cut decode's final run is a retained loop prefix the next slice may
/// continue: never splice it, even when it folds onto silence.
#[test]
fn splice_degenerate_tail_run_is_a_noop_on_a_guard_cut() {
    let (_, tokenizer) = whisper_execution_and_tokenizer_fixture();
    let ts = tokenizer
        .first_timestamp_token_id()
        .expect("first timestamp id");
    let tokens = vec![ts, 1, 2, 3, ts + 159];
    let mut alignments = tail_repeat_alignments(&tokens);
    let mut decode = tail_repeat_decode(
        tokens.clone(),
        Seq2SeqGreedyDecodeStopReason::DegenerateRepeatGuard,
    );
    decode.text = tokenizer
        .decode_text_token_ids(&tokens)
        .expect("decode text");
    let mut words = vec![
        word_ts("a", 3.18, 3.18),
        word_ts("b", 3.18, 3.18),
        word_ts("c", 3.18, 3.18),
    ];
    let quiet = vec![0.0005_f32; 159];
    let spliced = whisper_splice_degenerate_tail_run(
        &tokenizer,
        &mut decode,
        &mut alignments,
        &mut words,
        &[(0, 3)],
        159,
        Some(&quiet),
        3.18,
    )
    .expect("no-op");
    assert!(!spliced, "a guard-cut decode must not be spliced");
    assert_eq!(decode.generated_tokens, tokens);
    assert_eq!(words.len(), 3);
}

/// A range list that no longer pairs 1:1 with the token stream's runs is
/// untrustworthy: refuse rather than splice at shifted indices.
#[test]
fn splice_degenerate_tail_run_is_a_noop_when_the_run_ranges_misalign() {
    let (_, tokenizer) = whisper_execution_and_tokenizer_fixture();
    let ts = tokenizer
        .first_timestamp_token_id()
        .expect("first timestamp id");
    // Two runs in the tokens, one entry in the range list.
    let tokens = vec![ts, 1, 2, ts + 100, ts + 150, 4, 5, ts + 159];
    let mut alignments = tail_repeat_alignments(&tokens);
    let mut decode = tail_repeat_decode(tokens.clone(), Seq2SeqGreedyDecodeStopReason::StopToken);
    decode.text = tokenizer
        .decode_text_token_ids(&tokens)
        .expect("decode text");
    let mut words = vec![word_ts("a", 3.18, 3.18), word_ts("b", 3.18, 3.18)];
    let quiet = vec![0.0005_f32; 159];
    let spliced = whisper_splice_degenerate_tail_run(
        &tokenizer,
        &mut decode,
        &mut alignments,
        &mut words,
        &[(0, 0)],
        159,
        Some(&quiet),
        3.18,
    )
    .expect("no-op");
    assert!(!spliced, "misaligned ranges must refuse the splice");
    assert_eq!(decode.generated_tokens, tokens);
    assert_eq!(words.len(), 2);
}

/// No envelope means the silence cannot be confirmed: fail open and keep the
/// run even when its shape is a perfect fold.
#[test]
fn splice_degenerate_tail_run_is_a_noop_without_an_envelope() {
    let (_, tokenizer) = whisper_execution_and_tokenizer_fixture();
    let ts = tokenizer
        .first_timestamp_token_id()
        .expect("first timestamp id");
    let tokens = vec![ts, 1, 2, 3, 4, 5, 6, ts + 159];
    let mut alignments = tail_repeat_alignments(&tokens);
    let mut decode = tail_repeat_decode(tokens.clone(), Seq2SeqGreedyDecodeStopReason::StopToken);
    decode.text = tokenizer
        .decode_text_token_ids(&tokens)
        .expect("decode text");
    let mut words = vec![word_ts("a", 3.18, 3.18), word_ts("b", 3.18, 3.18)];
    let spliced = whisper_splice_degenerate_tail_run(
        &tokenizer,
        &mut decode,
        &mut alignments,
        &mut words,
        &[(0, 2)],
        159,
        None,
        3.18,
    )
    .expect("no-op");
    assert!(!spliced, "without an envelope the splice must refuse");
    assert_eq!(decode.generated_tokens, tokens);
    assert_eq!(words.len(), 2);
}

/// An alignment row that lost 1:1 parity with the tokens makes the run ranges
/// untrustworthy: refuse.
#[test]
fn splice_degenerate_tail_run_is_a_noop_when_the_alignments_misalign() {
    let (_, tokenizer) = whisper_execution_and_tokenizer_fixture();
    let ts = tokenizer
        .first_timestamp_token_id()
        .expect("first timestamp id");
    let tokens = vec![ts, 1, 2, 3, 4, 5, 6, ts + 159];
    // One alignment row short of the token count.
    let mut alignments = tail_repeat_alignments(&tokens);
    alignments.pop();
    let mut decode = tail_repeat_decode(tokens.clone(), Seq2SeqGreedyDecodeStopReason::StopToken);
    decode.text = tokenizer
        .decode_text_token_ids(&tokens)
        .expect("decode text");
    let mut words = vec![word_ts("a", 3.18, 3.18), word_ts("b", 3.18, 3.18)];
    let quiet = vec![0.0005_f32; 159];
    let spliced = whisper_splice_degenerate_tail_run(
        &tokenizer,
        &mut decode,
        &mut alignments,
        &mut words,
        &[(0, 2)],
        159,
        Some(&quiet),
        3.18,
    )
    .expect("no-op");
    assert!(!spliced, "a ragged alignment row must refuse the splice");
    assert_eq!(decode.generated_tokens, tokens);
    assert_eq!(words.len(), 2);
}

/// One alignment row whose cross-attention peaks at `frame` (0..1500).
fn ladder_alignment_peaking_at(frame: usize) -> WhisperGeneratedTokenAlignment {
    WhisperGeneratedTokenAlignment {
        token_id: 1,
        frame_probs: (0..1500)
            .map(|f| if f == frame { 1.0_f32 } else { 0.001_f32 })
            .collect(),
    }
}

fn ladder_candidate(text: &str, peak_frames: &[usize]) -> WhisperDecodeCandidate {
    WhisperDecodeCandidate {
        text_trimmed: text.to_string(),
        token_alignments: peak_frames
            .iter()
            .copied()
            .map(ladder_alignment_peaking_at)
            .collect(),
    }
}

fn ladder_candidate_evidence(candidate: &WhisperDecodeCandidate) -> Option<f32> {
    whisper_ladder_evidence_span_seconds(candidate, 30.0)
}

#[test]
fn ladder_evidence_span_measures_spread_and_collapses_to_zero() {
    let spread = ladder_candidate("real verse", &[100, 300, 500, 900, 1200, 1400]);
    let span = ladder_candidate_evidence(&spread).expect("span");
    // p10..p90 = frames 100..1400 over 1500, scaled to the 30 s slice.
    assert!((span - 1300.0_f32 / 1500.0 * 30.0).abs() < 1e-3);

    let collapsed = ladder_candidate("hallucinated filler", &[1499, 1498, 1499, 1497, 1499]);
    let span = ladder_candidate_evidence(&collapsed).expect("span");
    assert!(span < 0.05, "a collapsed run measures no span, got {span}");

    let sparse = ladder_candidate("short", &[10, 20, 30]);
    assert!(
        ladder_candidate_evidence(&sparse).is_none(),
        "fewer than 4 rows is not measurable"
    );
}

#[test]
fn ladder_cannot_win_by_length_when_attention_collapses() {
    // The incumbent tracked real audio; the challenger is the longer text but
    // its attention is pinned to the last frame (the no-speech attractor).
    let incumbent = ladder_candidate("real speech here", &[100, 400, 700, 1000, 1200]);
    let challenger = ladder_candidate(
        "a much longer hallucinated filler run of nonsense text",
        &[1499, 1498, 1499, 1497, 1499, 1500, 1499, 1496],
    );
    assert!(!whisper_decode_candidate_better(
        &challenger,
        &incumbent,
        ladder_candidate_evidence(&challenger),
        ladder_candidate_evidence(&incumbent)
    ));
}

#[test]
fn ladder_round_over_more_audio_wins_even_if_shorter() {
    // The guard-cut incumbent only covered the loop at the slice head; the
    // challenger actually tracked the audio further in.
    let incumbent = ladder_candidate("Lock, start! Rock!", &[100, 110, 120, 130]);
    let challenger = ladder_candidate("verse", &[100, 300, 1000, 1400]);
    assert!(whisper_decode_candidate_better(
        &challenger,
        &incumbent,
        ladder_candidate_evidence(&challenger),
        ladder_candidate_evidence(&incumbent)
    ));
}

#[test]
fn ladder_tie_on_evidence_falls_back_to_length_and_keeps_incumbent_on_shorter() {
    let incumbent = ladder_candidate("short", &[100, 400, 700, 1000, 1200]);
    let longer = ladder_candidate(
        "a longer text over the same audio",
        &[110, 400, 700, 1000, 1200, 1210],
    );
    assert!(whisper_decode_candidate_better(
        &longer,
        &incumbent,
        ladder_candidate_evidence(&longer),
        ladder_candidate_evidence(&incumbent)
    ));
    let shorter = ladder_candidate("a", &[110, 400, 700, 1000, 1200, 1210]);
    assert!(!whisper_decode_candidate_better(
        &shorter,
        &incumbent,
        ladder_candidate_evidence(&shorter),
        ladder_candidate_evidence(&incumbent)
    ));
}

#[test]
fn ladder_falls_back_to_length_when_evidence_unmeasurable() {
    let incumbent = ladder_candidate("short", &[]);
    let longer = ladder_candidate("a longer text", &[]);
    assert!(whisper_decode_candidate_better(
        &longer, &incumbent, None, None
    ));
    assert!(!whisper_decode_candidate_better(
        &incumbent, &longer, None, None
    ));
    // An empty challenger never wins, evidence or not.
    let empty = ladder_candidate("", &[100, 200, 300, 400]);
    let real = ladder_candidate("anything", &[100, 200, 300, 400]);
    assert!(!whisper_decode_candidate_better(&empty, &real, None, None));
}

#[test]
fn carry_loop_dominance_shapes() {
    // The lobster shape: a short phrase recited over a groove, stopped
    // mid-cycle (3 full cycles + 2 of the 6 tokens).
    let cycle = [10u32, 11, 12, 13, 14, 15];
    let mut tokens = Vec::new();
    for _ in 0..3 {
        tokens.extend_from_slice(&cycle);
    }
    tokens.extend_from_slice(&cycle[..2]);
    assert!(whisper_carry_is_loop_dominant(&tokens));

    // Single-token stutter reaches the floor at 8; 5 is below it.
    assert!(whisper_carry_is_loop_dominant(&[7u32; 8]));
    assert!(!whisper_carry_is_loop_dominant(&[7u32; 5]));

    // Two cycles of a phrase is emphatic speech, not a loop.
    let mut two = Vec::new();
    for _ in 0..2 {
        two.extend_from_slice(&cycle);
    }
    assert!(!whisper_carry_is_loop_dominant(&two));

    // A loop that only fills a minority of a long decode is not dominant: the
    // prefix carries the real speech, so its carry stays useful.
    let mut prefix = vec![99u32; 20];
    prefix.extend_from_slice(&[1u32, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]);
    assert!(!whisper_carry_is_loop_dominant(&prefix));

    // No repetition at all.
    assert!(!whisper_carry_is_loop_dominant(
        &(0..40).map(|i| i as u32).collect::<Vec<_>>()
    ));
}

#[test]
fn slice_head_is_audible_when_head_matches_speech_level() {
    // 100 x 20 ms frames = 2 s of flat, speech-level envelope.
    let rms = vec![0.5_f32; 100];
    // First word at 0.2 s, last ends at 1.8 s: head and region at equal level.
    assert!(super::whisper_slice_head_is_audible(&rms, 0.2, 1.8));
    // A louder head (music into speech) also passes: it is audible either way.
    let rms_head_louder = {
        let mut rms = vec![0.0_f32; 100];
        rms[..10].copy_from_slice(&[2.0_f32; 10]);
        rms[10..].copy_from_slice(&vec![0.5; 90]);
        rms
    };
    assert!(super::whisper_slice_head_is_audible(
        &rms_head_louder,
        0.2,
        1.8
    ));
}

#[test]
fn slice_head_is_silent_when_head_is_quiet_or_empty() {
    // Silent head, speech-level region: a legitimate onset, not a drop.
    let mut rms = vec![0.0_f32; 100];
    rms[10..].copy_from_slice(&vec![0.5_f32; 90]);
    assert!(!super::whisper_slice_head_is_audible(&rms, 0.2, 1.8));

    // Degenerate inputs fail closed.
    assert!(!super::whisper_slice_head_is_audible(&[], 10.0, 20.0));
    assert!(!super::whisper_slice_head_is_audible(
        &[0.5_f32],
        10.0,
        20.0
    ));
    assert!(!super::whisper_slice_head_is_audible(
        &[0.5_f32; 100],
        0.0,
        1.0
    ));
    assert!(!super::whisper_slice_head_is_audible(
        &[0.5_f32; 100],
        1.0,
        1.0
    ));
}

#[test]
fn slice_head_deficit_bound_sits_at_six_db() {
    // 2 s of 20 ms frames; the decoded region is [0.4, 1.9] s.
    let mut rms = vec![0.0_f32; 100];
    let region = 0.5_f32;
    rms[20..95].copy_from_slice(&vec![region; 75]);
    // 3 dB below the region: within the bound, audible.
    let head_3db = region * 10.0_f32.powf(-3.0 / 20.0);
    rms[..20].copy_from_slice(&[head_3db; 20]);
    assert!(super::whisper_slice_head_is_audible(&rms, 0.4, 1.9));
    // 10 dB below the region: outside the bound, not audible.
    let head_10db = region * 10.0_f32.powf(-10.0 / 20.0);
    rms[..20].copy_from_slice(&[head_10db; 20]);
    assert!(!super::whisper_slice_head_is_audible(&rms, 0.4, 1.9));
}

#[test]
fn token_stream_subsequence_shapes() {
    // In-order, non-contiguous: the longer stream carries everything the
    // shorter one says, in the same order, with more around it.
    let kept = vec![10u32, 20, 30];
    let other = vec![5u32, 10, 5, 20, 7, 30, 9];
    assert!(super::whisper_token_stream_is_subsequence(&kept, &other));

    // Order matters: the same tokens in a different order are a different
    // reading, not a superset.
    assert!(!super::whisper_token_stream_is_subsequence(
        &[30u32, 10],
        &[10, 30]
    ));

    // A drop anywhere fails: the subsequence is the content-preservation
    // guarantee, and losing one token defeats it.
    assert!(!super::whisper_token_stream_is_subsequence(
        &kept,
        &[10, 30]
    ));

    // Copies count positionally: two of a token need two in the other
    // stream, even with room around them.
    assert!(super::whisper_token_stream_is_subsequence(
        &[7u32, 7],
        &[7, 8, 7, 9]
    ));
    assert!(!super::whisper_token_stream_is_subsequence(
        &[7u32, 7],
        &[7, 8]
    ));

    // An empty kept stream is carried by anything.
    assert!(super::whisper_token_stream_is_subsequence(&[], &[1u32]));
    assert!(super::whisper_token_stream_is_subsequence(&[], &[]));
}

#[test]
fn dominant_cycle_stripping_shapes() {
    // The jc shape: a *Squeak* cycle (5 tokens) x 3 interleaved with real
    // words -- the cycle is the dominant one and is stripped wholesale, the
    // real content survives.
    let squeak = [1853u32, 50, 1077, 514, 9];
    let real = [3301u32, 485, 634, 603, 1699, 309, 484];
    let mut tokens = Vec::new();
    tokens.extend_from_slice(&squeak);
    tokens.extend_from_slice(&squeak);
    tokens.extend_from_slice(&real);
    tokens.extend_from_slice(&squeak);
    let stripped = super::whisper_stream_without_dominant_cycle(&tokens);
    assert_eq!(stripped, real);

    // A single or double echo is emphatic speech, not a loop: nothing is
    // stripped.
    let mut two = Vec::new();
    for _ in 0..2 {
        two.extend_from_slice(&squeak);
    }
    two.extend_from_slice(&real);
    assert_eq!(super::whisper_stream_without_dominant_cycle(&two), two);

    // Single-token stutter holds only at the guard's own floor of 8.
    assert_eq!(
        super::whisper_stream_without_dominant_cycle(&[7u32; 8]),
        Vec::<u32>::new()
    );
    let five = [7u32; 5];
    assert_eq!(super::whisper_stream_without_dominant_cycle(&five), five);

    // A phrase repeated twice is a repeated line, not a loop.
    let phrase = [11u32, 12, 13, 14, 15, 16];
    let mut twice = Vec::new();
    for _ in 0..2 {
        twice.extend_from_slice(&phrase);
    }
    assert_eq!(super::whisper_stream_without_dominant_cycle(&twice), twice);

    // No repetition at all: unchanged.
    let distinct: Vec<u32> = (0..24).map(|i| i as u32).collect();
    assert_eq!(
        super::whisper_stream_without_dominant_cycle(&distinct),
        distinct
    );

    // Shorter input than the longest period: still checked down to n=1.
    let short = [9u32, 9, 9, 9, 9];
    assert_eq!(super::whisper_stream_without_dominant_cycle(&short), short);
}
