//! Nemotron-transcribe Parallel Expert (PE) encoder: ASR expert (32-layer
//! RoPE transformer), Sortformer speaker branch (31-layer RoPE transformer +
//! high-resolution head), fusion (thresholded LayerNorm(8) -> sinusoid kernel
//! -> + LayerNorm(1280) -> Linear(2688)), producing the decoder's input
//! embeddings.
//!
//! Faithful port of the reference (`src/speechlm/pee/pee_model.cpp`,
//! `pee_rope_transformer.{h,cpp}` in NeMo-Speech.cpp), B=1, single utterance.
//! The ASR expert's post-`final_norm` states (and the 10 ms Sortformer
//! probabilities) are tapped in the output so the CTC timestamp head and
//! host aligner in `ctc_head`/`ctc_align` can bolt on after decode.
//!
//! Feature stacking (`feat_in=128`, `subsampling=8`) is a pure reshape of the
//! zero-padded mel, not a conv; the ASR expert's per-utterance per-feature
//! normalization happens on the host (reference's own `per_feature_normalize`
//! is bit-identical to doing it in-graph for an unpadded batch).

use thiserror::Error;

use crate::ggml_runtime::{
    GgmlCpuGraphBackend, GgmlCpuGraphBuilder, GgmlCpuGraphConfig, GgmlCpuGraphError,
    GgmlCpuGraphRunner, GgmlCpuTensor, GgmlLoadedTensor, GgmlLoadedWeightContext,
};
use crate::nn::norm::{AffineLayerNormSteps, apply_affine_layer_norm};

#[derive(Debug, Error)]
pub(crate) enum NemotronTranscribeEncoderError {
    #[error("nemotron-transcribe encoder shape error: {reason}")]
    Shape { reason: String },
    #[error("nemotron-transcribe encoder missing weight tensor '{name}'")]
    MissingWeight { name: String },
    #[error("nemotron-transcribe encoder GGML backend failed at {stage}: {source}")]
    Ggml {
        stage: &'static str,
        source: GgmlCpuGraphError,
    },
}

fn ggml_err(
    stage: &'static str,
) -> impl Fn(GgmlCpuGraphError) -> NemotronTranscribeEncoderError + Copy {
    move |source| NemotronTranscribeEncoderError::Ggml { stage, source }
}

/// One RoPE transformer configuration (ASR expert, diar expert, and the CTC
/// head share this block shape).
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub(crate) struct RopeTransformerConfig {
    pub d_model: usize,
    pub n_heads: usize,
    pub d_k: usize,
    pub n_layers: usize,
    pub ff_hidden: usize,
    pub rotary_dims: usize,
    pub qk_norm: bool,
    pub pre_block_norm: bool,
}

impl RopeTransformerConfig {
    pub(crate) fn validate(&self, name: &str) -> Result<(), NemotronTranscribeEncoderError> {
        if !self.d_model.is_multiple_of(self.n_heads) {
            return Err(NemotronTranscribeEncoderError::Shape {
                reason: format!(
                    "{name}: d_model {} not divisible by n_heads {}",
                    self.d_model, self.n_heads
                ),
            });
        }
        if self.rotary_dims < 2
            || !self.rotary_dims.is_multiple_of(2)
            || self.rotary_dims > self.d_k
        {
            return Err(NemotronTranscribeEncoderError::Shape {
                reason: format!("{name}: bad rotary_dims {}", self.rotary_dims),
            });
        }
        Ok(())
    }
}

pub(crate) const NEMOTRON_TRANSCRIBE_ASR_CONFIG: RopeTransformerConfig = RopeTransformerConfig {
    d_model: 1280,
    n_heads: 10,
    d_k: 128,
    n_layers: 32,
    ff_hidden: 5120,
    rotary_dims: 64,
    qk_norm: true,
    pre_block_norm: true,
};

pub(crate) const NEMOTRON_TRANSCRIBE_DIAR_CONFIG: RopeTransformerConfig = RopeTransformerConfig {
    d_model: 512,
    n_heads: 8,
    d_k: 64,
    n_layers: 31,
    ff_hidden: 2048,
    rotary_dims: 64,
    qk_norm: false,
    pre_block_norm: true,
};

pub(crate) const NEMOTRON_TRANSCRIBE_FUSION_PROJ_DIM: usize = 2688;
pub(crate) const NEMOTRON_TRANSCRIBE_SPEAKER_ACTIVITY_THRESHOLD: f32 = 0.5;
pub(crate) const NEMOTRON_TRANSCRIBE_NUM_SPEAKERS: usize = 8;
pub(crate) const NEMOTRON_TRANSCRIBE_TF_D_MODEL: usize = 192;
const NEMOTRON_TRANSCRIBE_UPSAMPLE_FACTOR: usize = 8;
const NEMOTRON_TRANSCRIBE_SUBSAMPLING_FACTOR: usize = 8;
const NEMOTRON_TRANSCRIBE_FEAT_IN: usize = 128;
pub(crate) const LAYERNORM_EPS: f32 = 1e-5;

fn rope_positions(t: usize) -> Vec<i32> {
    (0..t as i32).collect()
}

pub(crate) fn add_norm_weight<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    input: GgmlCpuTensor<'a>,
    weight: GgmlCpuTensor<'a>,
    bias: GgmlCpuTensor<'a>,
    name: &'static str,
) -> Result<GgmlCpuTensor<'a>, NemotronTranscribeEncoderError> {
    apply_affine_layer_norm(
        graph,
        input,
        LAYERNORM_EPS,
        weight,
        bias,
        AffineLayerNormSteps {
            norm: name,
            scale: name,
            bias: name,
        },
        |stage, source| NemotronTranscribeEncoderError::Ggml { stage, source },
    )
}

pub(crate) fn linear_no_bias<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    weight: GgmlCpuTensor<'a>,
    input: GgmlCpuTensor<'a>,
    stage: &'static str,
) -> Result<GgmlCpuTensor<'a>, NemotronTranscribeEncoderError> {
    graph.mul_mat(weight, input).map_err(ggml_err(stage))
}

pub(crate) fn linear_with_bias<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    weight: GgmlCpuTensor<'a>,
    input: GgmlCpuTensor<'a>,
    bias: GgmlCpuTensor<'a>,
    stage: &'static str,
) -> Result<GgmlCpuTensor<'a>, NemotronTranscribeEncoderError> {
    let projected = graph.mul_mat(weight, input).map_err(ggml_err(stage))?;
    graph.add(projected, bias).map_err(ggml_err(stage))
}

/// Per-head LayerNorm over `d_k` with its own gain/bias (the ASR expert's
/// `qk_norm`). Applied BEFORE rope.
pub(crate) fn head_norm<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    input: GgmlCpuTensor<'a>,
    weight: GgmlCpuTensor<'a>,
    bias: GgmlCpuTensor<'a>,
    name: &'static str,
) -> Result<GgmlCpuTensor<'a>, NemotronTranscribeEncoderError> {
    add_norm_weight(graph, input, weight, bias, name)
}

/// One pre-LN RoPE transformer block for the ASR/diar experts.
#[allow(clippy::too_many_arguments)]
pub(crate) fn rope_transformer_block<'a>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    input: GgmlCpuTensor<'a>,
    positions: GgmlCpuTensor<'a>,
    config: &RopeTransformerConfig,
    prefix: &str,
    weights: &EncoderBlockWeights<'a>,
    frames: usize,
) -> Result<GgmlCpuTensor<'a>, NemotronTranscribeEncoderError> {
    let _ = LAYERNORM_EPS;
    let stage = prefix;
    let map = ggml_err("block");
    let d_model = config.d_model;
    let n_head = config.n_heads;
    let d_k = config.d_k;

    let normed = add_norm_weight(graph, input, weights.norm1_w, weights.norm1_b, "pe_norm1")?;
    let qkv = linear_no_bias(graph, weights.qkv_w, normed, "pe_qkv")?;

    // Split the packed (3*d_model, T, 1) tensor into per-head 4D views.
    let esz = std::mem::size_of::<f32>();
    let packed_nb1 = 3 * d_model * esz;
    let packed_nb2 = packed_nb1 * frames;
    let q = graph
        .view_4d(
            qkv,
            d_k,
            n_head,
            frames,
            1,
            d_k * esz,
            packed_nb1,
            packed_nb2,
            0,
        )
        .map_err(map)?;
    let k = graph
        .view_4d(
            qkv,
            d_k,
            n_head,
            frames,
            1,
            d_k * esz,
            packed_nb1,
            packed_nb2,
            d_model * esz,
        )
        .map_err(map)?;
    let v = graph
        .view_4d(
            qkv,
            d_k,
            n_head,
            frames,
            1,
            d_k * esz,
            packed_nb1,
            packed_nb2,
            2 * d_model * esz,
        )
        .map_err(map)?;

    let _ = stage;
    let mut q = q;
    let mut k = k;
    if config.qk_norm {
        q = head_norm(graph, q, weights.q_norm_w, weights.q_norm_b, "pe_q_norm")?;
        k = head_norm(graph, k, weights.k_norm_w, weights.k_norm_b, "pe_k_norm")?;
    }

    // NeoX half-split RoPE of the first rotary_dims of each head.
    let rope = crate::ggml_runtime::GgmlRopeExtParams {
        n_dims: config.rotary_dims,
        mode: crate::ggml_runtime::ffi::GGML_ROPE_TYPE_NEOX,
        n_ctx_orig: 0,
        freq_base: 10000.0,
        freq_scale: 1.0,
        ext_factor: 0.0,
        attn_factor: 1.0,
        beta_fast: 0.0,
        beta_slow: 0.0,
    };
    q = graph.rope_ext(q, positions, rope).map_err(map)?;
    k = graph.rope_ext(k, positions, rope).map_err(map)?;

    // q,k: (d_k, T, n_head, 1) for flash attention.
    let q_p = graph.permute(q, 0, 2, 1, 3).map_err(map)?;
    let q_p = graph.cont(q_p).map_err(map)?;
    // Cast k/v to F16 *before* the permute/cont so the contiguous copy operates
    // on half-width elements. flash_attn_ext already accepts F16 k/v; q stays
    // F32 because flash_attn_ext requires an F32 q (nemotron-f16.md stage 1).
    let k_f16 = graph.cast_to_f16(k).map_err(map)?;
    let k_p = graph.permute(k_f16, 0, 2, 1, 3).map_err(map)?;
    let k_f16 = graph.cont(k_p).map_err(map)?;
    let v_f16 = graph.cast_to_f16(v).map_err(map)?;
    let v_p = graph.permute(v_f16, 0, 2, 1, 3).map_err(map)?;
    let v_f16 = graph.cont(v_p).map_err(map)?;
    let scale = 1.0f32 / (d_k as f32).sqrt();
    let attn = graph
        .flash_attn_ext_with_precision(
            q_p,
            k_f16,
            v_f16,
            None,
            scale,
            0.0,
            0.0,
            crate::ggml_runtime::GgmlFlashAttentionPrecision::F32,
        )
        .map_err(map)?;
    let merged = graph.reshape_3d(attn, d_model, frames, 1).map_err(map)?;
    let attn_out = linear_with_bias(
        graph,
        weights.out_proj_w,
        merged,
        weights.out_proj_b,
        "pe_out_proj",
    )?;
    let x1 = graph.add(input, attn_out).map_err(map)?;

    // Feed-forward sub-layer (exact GELU).
    let h2 = add_norm_weight(graph, x1, weights.norm2_w, weights.norm2_b, "pe_norm2")?;
    let ff = linear_with_bias(graph, weights.ffn_up_w, h2, weights.ffn_up_b, "pe_ffn_up")?;
    let ff = graph.gelu_erf(ff).map_err(map)?;
    let ff = linear_with_bias(
        graph,
        weights.ffn_down_w,
        ff,
        weights.ffn_down_b,
        "pe_ffn_down",
    )?;
    graph.add(x1, ff).map_err(map)
}

pub(crate) struct EncoderBlockWeights<'a> {
    pub(crate) norm1_w: GgmlCpuTensor<'a>,
    pub(crate) norm1_b: GgmlCpuTensor<'a>,
    pub(crate) qkv_w: GgmlCpuTensor<'a>,
    pub(crate) out_proj_w: GgmlCpuTensor<'a>,
    pub(crate) out_proj_b: GgmlCpuTensor<'a>,
    pub(crate) q_norm_w: GgmlCpuTensor<'a>,
    pub(crate) q_norm_b: GgmlCpuTensor<'a>,
    pub(crate) k_norm_w: GgmlCpuTensor<'a>,
    pub(crate) k_norm_b: GgmlCpuTensor<'a>,
    pub(crate) norm2_w: GgmlCpuTensor<'a>,
    pub(crate) norm2_b: GgmlCpuTensor<'a>,
    pub(crate) ffn_up_w: GgmlCpuTensor<'a>,
    pub(crate) ffn_up_b: GgmlCpuTensor<'a>,
    pub(crate) ffn_down_w: GgmlCpuTensor<'a>,
    pub(crate) ffn_down_b: GgmlCpuTensor<'a>,
}

struct RopeTransformerWeights<'a> {
    pre_encode_w: GgmlCpuTensor<'a>,
    embed_norm_w: Option<GgmlCpuTensor<'a>>,
    embed_norm_b: Option<GgmlCpuTensor<'a>>,
    layers: Vec<EncoderBlockWeights<'a>>,
    final_norm_w: GgmlCpuTensor<'a>,
    final_norm_b: GgmlCpuTensor<'a>,
}

fn loaded_tensor<'a>(
    loaded: &GgmlLoadedWeightContext,
    name: &str,
) -> Result<GgmlCpuTensor<'a>, NemotronTranscribeEncoderError> {
    loaded
        .tensor(name)
        .map(GgmlLoadedTensor::as_graph_tensor)
        .ok_or_else(|| NemotronTranscribeEncoderError::MissingWeight {
            name: name.to_string(),
        })
}

fn build_rope_transformer_weights<'a>(
    loaded: &GgmlLoadedWeightContext,
    prefix: &str,
    config: &RopeTransformerConfig,
    qkv_bias: bool,
) -> Result<RopeTransformerWeights<'a>, NemotronTranscribeEncoderError> {
    let _ = qkv_bias;
    let mut layers = Vec::with_capacity(config.n_layers);
    for index in 0..config.n_layers {
        let p = format!("{prefix}.layers.{index}");
        let qk_norm = config.qk_norm;
        layers.push(EncoderBlockWeights {
            norm1_w: loaded_tensor(loaded, &format!("{p}.norm1.weight"))?,
            norm1_b: loaded_tensor(loaded, &format!("{p}.norm1.bias"))?,
            qkv_w: loaded_tensor(loaded, &format!("{p}.attn.w_qkv.weight"))?,
            out_proj_w: loaded_tensor(loaded, &format!("{p}.attn.out_proj.weight"))?,
            out_proj_b: loaded_tensor(loaded, &format!("{p}.attn.out_proj.bias"))?,
            q_norm_w: if qk_norm {
                loaded_tensor(loaded, &format!("{p}.attn.q_norm.weight"))?
            } else {
                loaded_tensor(loaded, &format!("{p}.norm1.weight"))?
            },
            q_norm_b: if qk_norm {
                loaded_tensor(loaded, &format!("{p}.attn.q_norm.bias"))?
            } else {
                loaded_tensor(loaded, &format!("{p}.norm1.bias"))?
            },
            k_norm_w: if qk_norm {
                loaded_tensor(loaded, &format!("{p}.attn.k_norm.weight"))?
            } else {
                loaded_tensor(loaded, &format!("{p}.norm1.weight"))?
            },
            k_norm_b: if qk_norm {
                loaded_tensor(loaded, &format!("{p}.attn.k_norm.bias"))?
            } else {
                loaded_tensor(loaded, &format!("{p}.norm1.bias"))?
            },
            norm2_w: loaded_tensor(loaded, &format!("{p}.norm2.weight"))?,
            norm2_b: loaded_tensor(loaded, &format!("{p}.norm2.bias"))?,
            ffn_up_w: loaded_tensor(loaded, &format!("{p}.ffn.net.0.weight"))?,
            ffn_up_b: loaded_tensor(loaded, &format!("{p}.ffn.net.0.bias"))?,
            ffn_down_w: loaded_tensor(loaded, &format!("{p}.ffn.net.3.weight"))?,
            ffn_down_b: loaded_tensor(loaded, &format!("{p}.ffn.net.3.bias"))?,
        });
    }
    let pre_block_norm = config.pre_block_norm;
    Ok(RopeTransformerWeights {
        pre_encode_w: loaded_tensor(loaded, &format!("{prefix}.pre_encode.proj.weight"))?,
        embed_norm_w: pre_block_norm
            .then(|| loaded_tensor(loaded, &format!("{prefix}.embed_norm.weight")))
            .transpose()?,
        embed_norm_b: pre_block_norm
            .then(|| loaded_tensor(loaded, &format!("{prefix}.embed_norm.bias")))
            .transpose()?,
        layers,
        final_norm_w: loaded_tensor(loaded, &format!("{prefix}.final_norm.weight"))?,
        final_norm_b: loaded_tensor(loaded, &format!("{prefix}.final_norm.bias"))?,
    })
}

struct SpeakerHeadWeights<'a> {
    encoder_proj_w: GgmlCpuTensor<'a>,
    encoder_proj_b: GgmlCpuTensor<'a>,
    subpixel_upsample_w: GgmlCpuTensor<'a>,
    subpixel_upsample_b: GgmlCpuTensor<'a>,
    first_hidden_w: GgmlCpuTensor<'a>,
    first_hidden_b: GgmlCpuTensor<'a>,
    single_hidden_w: GgmlCpuTensor<'a>,
    single_hidden_b: GgmlCpuTensor<'a>,
}

struct FusionWeights<'a> {
    asr_norm_w: GgmlCpuTensor<'a>,
    asr_norm_b: GgmlCpuTensor<'a>,
    diar_norm_w: GgmlCpuTensor<'a>,
    diar_norm_b: GgmlCpuTensor<'a>,
    diar_kernel: GgmlCpuTensor<'a>,
    proj_w: GgmlCpuTensor<'a>,
    proj_b: GgmlCpuTensor<'a>,
}

/// Resident weights for the PE encoder, bound zero-copy from the loaded pack.
struct EncoderWeights<'a> {
    asr: RopeTransformerWeights<'a>,
    diar: RopeTransformerWeights<'a>,
    speaker: SpeakerHeadWeights<'a>,
    fusion: FusionWeights<'a>,
}

fn build_encoder_weights<'a>(
    loaded: &GgmlLoadedWeightContext,
) -> Result<EncoderWeights<'a>, NemotronTranscribeEncoderError> {
    let asr =
        build_rope_transformer_weights(loaded, "asr", &NEMOTRON_TRANSCRIBE_ASR_CONFIG, false)?;
    let diar =
        build_rope_transformer_weights(loaded, "diar", &NEMOTRON_TRANSCRIBE_DIAR_CONFIG, false)?;
    let speaker = SpeakerHeadWeights {
        encoder_proj_w: loaded_tensor(loaded, "diar.head.encoder_proj.weight")?,
        encoder_proj_b: loaded_tensor(loaded, "diar.head.encoder_proj.bias")?,
        subpixel_upsample_w: loaded_tensor(loaded, "diar.head.subpixel_upsample.weight")?,
        subpixel_upsample_b: loaded_tensor(loaded, "diar.head.subpixel_upsample.bias")?,
        first_hidden_w: loaded_tensor(loaded, "diar.head.first_hidden_to_hidden.weight")?,
        first_hidden_b: loaded_tensor(loaded, "diar.head.first_hidden_to_hidden.bias")?,
        single_hidden_w: loaded_tensor(loaded, "diar.head.single_hidden_to_spks.weight")?,
        single_hidden_b: loaded_tensor(loaded, "diar.head.single_hidden_to_spks.bias")?,
    };
    let fusion = FusionWeights {
        asr_norm_w: loaded_tensor(loaded, "fusion.asr_norm.weight")?,
        asr_norm_b: loaded_tensor(loaded, "fusion.asr_norm.bias")?,
        diar_norm_w: loaded_tensor(loaded, "fusion.diar_norm.weight")?,
        diar_norm_b: loaded_tensor(loaded, "fusion.diar_norm.bias")?,
        diar_kernel: loaded_tensor(loaded, "fusion.diar_kernel")?,
        proj_w: loaded_tensor(loaded, "fusion.proj.weight")?,
        proj_b: loaded_tensor(loaded, "fusion.proj.bias")?,
    };
    Ok(EncoderWeights {
        asr,
        diar,
        speaker,
        fusion,
    })
}

/// The graph's node budget: ~180 tensor nodes per layer, plus the speaker
/// head and fusion. Sized for the 30-second safe encoder ceiling (375
/// frames), not the full hour.
const ENCODER_GRAPH_NODES: usize = 8192;

fn encoder_graph_config(backend: GgmlCpuGraphBackend) -> GgmlCpuGraphConfig {
    crate::models::graph_runtime_config::apply_request_execution_placement(GgmlCpuGraphConfig {
        context_bytes: GgmlCpuGraphConfig::metadata_context_bytes(ENCODER_GRAPH_NODES),
        graph_size: ENCODER_GRAPH_NODES,
        n_threads: GgmlCpuGraphConfig::resolve_runtime_thread_count_for(
            backend,
            crate::ggml_runtime::GgmlCpuGraphThreadingWorkload::EncoderPrelude,
        ),
        backend,
        use_scheduler: true,
    })
}

/// The encoder's outputs, downloaded to the host frame-major.
pub(crate) struct NemotronTranscribeEncoderOutput {
    /// Output frames (80 ms grid): `ceil(t_mel / subsampling)`.
    pub frames: usize,
    /// `frames * 2688` f32, the decoder's input embeddings.
    pub projections: Vec<f32>,
    /// `frames * 8` f32 80 ms speaker probabilities.
    pub speaker_probs: Vec<f32>,
    /// `frames * 1280` f32 ASR expert post-`final_norm` states (80 ms grid),
    /// tapped for the CTC timestamp head.
    pub asr_states: Vec<f32>,
    /// `8 * frames * 8` f32 Sortformer probabilities on the 10 ms grid,
    /// frame-major (`[t * 8 + speaker]`), tapped for diarization refinement.
    pub speaker_probs10: Vec<f32>,
}

/// PE encoder runtime: a resident runner + loaded weight context. The graph
/// is rebuilt per request (shape depends on `T'`); weights stay bound.
pub(crate) struct NemotronTranscribeEncoderRuntime {
    runner: GgmlCpuGraphRunner,
    loaded: GgmlLoadedWeightContext,
}

impl NemotronTranscribeEncoderRuntime {
    pub(crate) fn new_from_preflight(
        preflight: &crate::ggml_runtime::GgufRuntimeSourcePreflight,
        backend: GgmlCpuGraphBackend,
        host_offload: Option<crate::ggml_runtime::HostWeightOffloadSpec>,
    ) -> Result<Self, NemotronTranscribeEncoderError> {
        let graph_config = encoder_graph_config(backend);
        let mut runner = GgmlCpuGraphRunner::new(graph_config).map_err(ggml_err("runner_init"))?;
        if let Some(spec) = host_offload {
            runner.set_host_weight_offload(spec);
        }
        let loaded = runner
            .load_gguf_weight_context_from_preflight(preflight)
            .map_err(ggml_err("load_gguf_weight_context"))?;
        Ok(Self { runner, loaded })
    }

    fn build_rope_transformer_graph<'a>(
        graph: &mut GgmlCpuGraphBuilder<'a>,
        weights: &RopeTransformerWeights<'a>,
        config: &RopeTransformerConfig,
        mel: GgmlCpuTensor<'a>,
        positions: GgmlCpuTensor<'a>,
        t_out: usize,
        name: &'static str,
    ) -> Result<GgmlCpuTensor<'a>, NemotronTranscribeEncoderError> {
        config.validate(name)?;
        // Feature stacking: (feat_in * sub, T') pure reshape of the
        // zero-padded mel. `mel` arrives as (feat_in, t_frames, 1) with
        // t_frames == t_out * sub (the host zero-pads).
        let stacked = graph
            .reshape_3d(
                mel,
                NEMOTRON_TRANSCRIBE_FEAT_IN * NEMOTRON_TRANSCRIBE_SUBSAMPLING_FACTOR,
                t_out,
                1,
            )
            .map_err(ggml_err("pe_stack"))?;
        let mut hidden = linear_no_bias(graph, weights.pre_encode_w, stacked, "pe_pre_encode")?;
        if let (Some(w), Some(b)) = (weights.embed_norm_w, weights.embed_norm_b) {
            hidden = add_norm_weight(graph, hidden, w, b, "pe_embed_norm")?;
        }
        for layer in &weights.layers {
            hidden = rope_transformer_block(graph, hidden, positions, config, name, layer, t_out)?;
        }
        add_norm_weight(
            graph,
            hidden,
            weights.final_norm_w,
            weights.final_norm_b,
            "pe_final_norm",
        )
    }

    /// Encode one single-window utterance. `mel` is the frontend's log-mel,
    /// `asr_mel` its per-feature-normalized (host, unbiased std) form; both
    /// are frame-major `t_out * sub`-padded by the caller.
    pub(crate) fn encode(
        &mut self,
        mel: &[f32],
        asr_mel: &[f32],
        t_mel: usize,
    ) -> Result<NemotronTranscribeEncoderOutput, NemotronTranscribeEncoderError> {
        let sub = NEMOTRON_TRANSCRIBE_SUBSAMPLING_FACTOR;
        let feat_in = NEMOTRON_TRANSCRIBE_FEAT_IN;
        let t_out = t_mel.div_ceil(sub);
        let t_padded = t_out * sub;
        if mel.len() != t_padded * feat_in || asr_mel.len() != t_padded * feat_in {
            return Err(NemotronTranscribeEncoderError::Shape {
                reason: format!(
                    "encoder input mismatch: mel {} values, asr_mel {} values, expected {}x{}",
                    mel.len(),
                    asr_mel.len(),
                    t_padded,
                    feat_in
                ),
            });
        }
        let weights = build_encoder_weights(&self.loaded)?;
        let positions_values = rope_positions(t_out);

        let mut graph = self.runner.start_graph();
        let asr_mel_t = graph
            .new_tensor_2d_f32(feat_in, t_padded, "pe_asr_mel")
            .map_err(ggml_err("asr_mel_alloc"))?;
        let diar_mel_t = graph
            .new_tensor_2d_f32(feat_in, t_padded, "pe_diar_mel")
            .map_err(ggml_err("diar_mel_alloc"))?;
        let pos_t = graph
            .new_tensor_1d_i32(t_out, "pe_pos")
            .map_err(ggml_err("positions_alloc"))?;
        graph
            .set_input(asr_mel_t)
            .map_err(ggml_err("set_input(asr_mel)"))?;
        graph
            .set_input(diar_mel_t)
            .map_err(ggml_err("set_input(diar_mel)"))?;
        graph.set_input(pos_t).map_err(ggml_err("set_input(pos)"))?;

        let asr_out = Self::build_rope_transformer_graph(
            &mut graph,
            &weights.asr,
            &NEMOTRON_TRANSCRIBE_ASR_CONFIG,
            asr_mel_t,
            pos_t,
            t_out,
            "asr",
        )?;
        let diar_out = Self::build_rope_transformer_graph(
            &mut graph,
            &weights.diar,
            &NEMOTRON_TRANSCRIBE_DIAR_CONFIG,
            diar_mel_t,
            pos_t,
            t_out,
            "diar",
        )?;

        // ---- Sortformer speaker head (high-resolution 10 ms) ----
        let speaker = &weights.speaker;
        let map = ggml_err("pe_speaker");
        let speaker_proj = linear_with_bias(
            &graph,
            speaker.encoder_proj_w,
            diar_out,
            speaker.encoder_proj_b,
            "pe_speaker_proj",
        )?;
        // (192, T') -> (T', 192) for the time-major conv, then back.
        let up_in = graph
            .cont(graph.permute(speaker_proj, 1, 0, 2, 3).map_err(map)?)
            .map_err(map)?;
        let up = graph
            .conv_1d(speaker.subpixel_upsample_w, up_in, 1, 1, 1)
            .map_err(map)?;
        // The conv returns (time, channels); broadcast the bias over ne1.
        let bias_t = graph
            .reshape_2d(
                speaker.subpixel_upsample_b,
                1,
                NEMOTRON_TRANSCRIBE_TF_D_MODEL * NEMOTRON_TRANSCRIBE_UPSAMPLE_FACTOR,
            )
            .map_err(map)?;
        let up = graph.add(up, bias_t).map_err(map)?;
        let up_c = graph
            .cont(graph.permute(up, 1, 0, 2, 3).map_err(map)?)
            .map_err(map)?;
        let hi = graph
            .reshape_3d(
                up_c,
                NEMOTRON_TRANSCRIBE_TF_D_MODEL,
                t_out * NEMOTRON_TRANSCRIBE_UPSAMPLE_FACTOR,
                1,
            )
            .map_err(map)?;
        let h1 = linear_with_bias(
            &graph,
            speaker.first_hidden_w,
            graph.relu(hi).map_err(map)?,
            speaker.first_hidden_b,
            "pe_speaker_hidden",
        )?;
        let logits = linear_with_bias(
            &graph,
            speaker.single_hidden_w,
            graph.relu(h1).map_err(map)?,
            speaker.single_hidden_b,
            "pe_speaker_spks",
        )?;
        let probs = graph.sigmoid(logits).map_err(map)?;

        // ---- 10 ms -> 80 ms: mean of 8 consecutive high-res frames ----
        let esz = std::mem::size_of::<f32>();
        let p80 = {
            // probs: (8 speakers [ne0, stride 1], 8*t_out 10 ms frames [ne1, stride 8]).
            // Group 8 consecutive frames per speaker into one 80 ms bin:
            // v[spk, sub, bin] = probs[spk, 8*bin + sub]; strides (1, 8, 64).
            let v = graph
                .view_3d(
                    probs,
                    NEMOTRON_TRANSCRIBE_NUM_SPEAKERS,
                    NEMOTRON_TRANSCRIBE_UPSAMPLE_FACTOR,
                    t_out,
                    NEMOTRON_TRANSCRIBE_NUM_SPEAKERS * esz,
                    NEMOTRON_TRANSCRIBE_NUM_SPEAKERS * NEMOTRON_TRANSCRIBE_UPSAMPLE_FACTOR * esz,
                    0,
                )
                .map_err(map)?;
            // Move subs to ne0 so sum_rows can reduce them.
            let v = graph.permute(v, 1, 0, 2, 3).map_err(map)?;
            let v = graph.cont(v).map_err(map)?;
            let sums = graph.sum_rows(v).map_err(map)?;
            let mean = graph
                .scale(sums, 1.0 / NEMOTRON_TRANSCRIBE_UPSAMPLE_FACTOR as f32)
                .map_err(map)?;
            let mean = graph.permute(mean, 2, 1, 0, 3).map_err(map)?;
            let mean = graph.cont(mean).map_err(map)?;
            graph
                .reshape_2d(mean, NEMOTRON_TRANSCRIBE_NUM_SPEAKERS, t_out)
                .map_err(map)?
        };

        // ---- fusion ----
        let fusion = &weights.fusion;
        let fmap = ggml_err("pe_fusion");
        let bin = graph
            .step(
                graph
                    .scale_bias(p80, 1.0, -NEMOTRON_TRANSCRIBE_SPEAKER_ACTIVITY_THRESHOLD)
                    .map_err(fmap)?,
            )
            .map_err(fmap)?;
        let ln_spk = add_norm_weight(
            &graph,
            bin,
            fusion.diar_norm_w,
            fusion.diar_norm_b,
            "pe_fusion_diar_norm",
        )?;
        let kernel_t = graph
            .cont(graph.transpose(fusion.diar_kernel).map_err(fmap)?)
            .map_err(fmap)?;
        let infusion = graph.mul_mat(kernel_t, ln_spk).map_err(fmap)?;
        let ln_asr = add_norm_weight(
            &graph,
            asr_out,
            fusion.asr_norm_w,
            fusion.asr_norm_b,
            "pe_fusion_asr_norm",
        )?;
        let fused = graph.add(ln_asr, infusion).map_err(fmap)?;
        let proj = linear_with_bias(
            &graph,
            fusion.proj_w,
            fused,
            fusion.proj_b,
            "pe_fusion_proj",
        )?;

        let taps: Vec<(GgmlCpuTensor<'_>, usize)> = vec![
            (proj, t_out * NEMOTRON_TRANSCRIBE_FUSION_PROJ_DIM),
            (p80, t_out * NEMOTRON_TRANSCRIBE_NUM_SPEAKERS),
            (asr_out, t_out * NEMOTRON_TRANSCRIBE_ASR_CONFIG.d_model),
            (
                probs,
                t_out * NEMOTRON_TRANSCRIBE_UPSAMPLE_FACTOR * NEMOTRON_TRANSCRIBE_NUM_SPEAKERS,
            ),
        ];
        graph
            .set_output(proj)
            .map_err(ggml_err("set_output(proj)"))?;
        graph.set_output(p80).map_err(ggml_err("set_output(p80)"))?;
        graph
            .set_output(asr_out)
            .map_err(ggml_err("set_output(asr_out)"))?;
        graph
            .set_output(probs)
            .map_err(ggml_err("set_output(probs10)"))?;
        graph
            .set_f32_slice(asr_mel_t, asr_mel, "pe_asr_mel")
            .map_err(ggml_err("upload_asr_mel"))?;
        graph
            .set_f32_slice(diar_mel_t, mel, "pe_diar_mel")
            .map_err(ggml_err("upload_diar_mel"))?;
        graph
            .set_i32_slice(pos_t, &positions_values, "pe_pos")
            .map_err(ggml_err("upload_pos"))?;

        let outputs = graph
            .compute_outputs_f32(&taps)
            .map_err(ggml_err("compute"))?;
        let mut outputs = outputs;
        let projections = outputs.remove(0);
        let speaker_probs = outputs.remove(0);
        let asr_states = outputs.remove(0);
        let speaker_probs10 = outputs.remove(0);
        Ok(NemotronTranscribeEncoderOutput {
            frames: t_out,
            projections,
            speaker_probs,
            asr_states,
            speaker_probs10,
        })
    }

    pub(crate) fn release_transient_compute_memory(
        &mut self,
    ) -> Result<(), NemotronTranscribeEncoderError> {
        self.runner
            .release_transient_scheduler_working_set()
            .map_err(ggml_err("release_transient_scheduler_working_set"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ggml_runtime::GGML_TYPE_F16;

    fn runner() -> GgmlCpuGraphRunner {
        GgmlCpuGraphRunner::new(GgmlCpuGraphConfig::default()).expect("graph runner")
    }

    /// The k/v path casts to F16 before the permute/cont so all three tensors
    /// after the cast are F16 (the contiguous copy operates on half-width data).
    #[test]
    fn kv_cast_precedes_cont_and_stays_f16() {
        let mut runner = runner();
        let graph = &mut runner.start_graph();
        // 4D (d_k, T, n_head, 1) as produced by rope_ext for k/v.
        let k = graph
            .new_tensor_4d_f32(4, 3, 2, 1, "kv_cast_test_k")
            .expect("k tensor");
        graph.set_input(k).expect("set k");
        let k_f16 = graph.cast_to_f16(k).expect("cast k");
        assert_eq!(graph.tensor_dtype(k_f16), GGML_TYPE_F16);
        let k_p = graph.permute(k_f16, 0, 2, 1, 3).expect("permute k");
        assert_eq!(graph.tensor_dtype(k_p), GGML_TYPE_F16);
        let k_cont = graph.cont(k_p).expect("cont k");
        assert_eq!(graph.tensor_dtype(k_cont), GGML_TYPE_F16);
    }
}
