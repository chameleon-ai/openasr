//! CTC timestamp head for nemotron-transcribe.
//!
//! Runs the transcript-independent part of `src/speechlm/pee/ctc_timestamp.cpp`
//! (`CtcGraph::build_graph`): the ASR expert's post-`final_norm` states (80 ms
//! grid) are linearly upsampled x2 to the 40 ms CTC grid, passed through an
//! `embed_norm` + two pre-LN RoPE transformer blocks + `final_norm`, and the
//! residual `x + residual_scale * y` is the CTC evidence `h`.
//!
//! The reference subtracts a full-vocabulary log-sum-exp and rounds log-probs
//! to bf16 before alignment. The LSE is a per-frame constant shared by every
//! CTC class, so it cancels out of the banded Viterbi's argmax/predecessor
//! comparisons (which is what determines word boundaries); only the overall
//! path score changes. We therefore project only the compact alignment columns
//! in-graph and bf16-round the logits on the host, which is numerically
//! equivalent for the word times and avoids materialising 23863 logits/frame.

use thiserror::Error;

use crate::ggml_runtime::{
    GgmlCpuGraphBackend, GgmlCpuGraphConfig, GgmlCpuGraphError, GgmlCpuGraphRunner,
    GgmlCpuGraphThreadingWorkload, GgmlCpuTensor, GgmlLoadedTensor, GgmlLoadedWeightContext,
};

use super::encoder_graph::{
    EncoderBlockWeights, RopeTransformerConfig, add_norm_weight, linear_with_bias,
    rope_transformer_block,
};

pub(crate) const CTC_D_MODEL: usize = 1280;
pub(crate) const CTC_N_HEADS: usize = 16;
pub(crate) const CTC_N_LAYERS: usize = 2;
pub(crate) const CTC_FF_HIDDEN: usize = 640;
pub(crate) const CTC_ROTARY_DIMS: usize = 40;
pub(crate) const CTC_UPSAMPLING: usize = 2;
pub(crate) const CTC_VOCAB_SIZE: usize = 23863;
pub(crate) const CTC_RESIDUAL_SCALE: f32 = 0.1;
pub(crate) const CTC_BLANK_ID: u32 = (CTC_VOCAB_SIZE - 1) as u32; // 23862

const HEAD_GRAPH_NODES: usize = 1024;

fn head_config() -> RopeTransformerConfig {
    RopeTransformerConfig {
        d_model: CTC_D_MODEL,
        n_heads: CTC_N_HEADS,
        d_k: CTC_D_MODEL / CTC_N_HEADS, // 80
        n_layers: CTC_N_LAYERS,
        ff_hidden: CTC_FF_HIDDEN,
        rotary_dims: CTC_ROTARY_DIMS,
        qk_norm: true,
        pre_block_norm: true,
    }
}

#[derive(Debug, Error)]
pub(crate) enum CtcHeadError {
    #[error("nemotron CTC head shape error: {reason}")]
    Shape { reason: String },
    #[error("nemotron CTC head missing weight tensor '{name}'")]
    MissingWeight { name: String },
    #[error("nemotron CTC head GGML backend failed at {stage}: {source}")]
    Ggml {
        stage: &'static str,
        source: GgmlCpuGraphError,
    },
    #[error("nemotron CTC head shared graph op failed: {0}")]
    EncoderOp(#[from] super::encoder_graph::NemotronTranscribeEncoderError),
}

fn ggml_err(stage: &'static str) -> impl Fn(GgmlCpuGraphError) -> CtcHeadError + Copy {
    move |source| CtcHeadError::Ggml { stage, source }
}

fn loaded_tensor<'a>(
    loaded: &GgmlLoadedWeightContext,
    name: &str,
) -> Result<GgmlCpuTensor<'a>, CtcHeadError> {
    loaded
        .tensor(name)
        .map(GgmlLoadedTensor::as_graph_tensor)
        .ok_or_else(|| CtcHeadError::MissingWeight {
            name: name.to_string(),
        })
}

struct CtcHeadWeights<'a> {
    upsample_w: GgmlCpuTensor<'a>,
    upsample_b: GgmlCpuTensor<'a>,
    embed_norm_w: GgmlCpuTensor<'a>,
    embed_norm_b: GgmlCpuTensor<'a>,
    layers: Vec<EncoderBlockWeights<'a>>,
    final_norm_w: GgmlCpuTensor<'a>,
    final_norm_b: GgmlCpuTensor<'a>,
    classifier_w: GgmlCpuTensor<'a>,
    classifier_b: GgmlCpuTensor<'a>,
}

fn build_weights<'a>(loaded: &GgmlLoadedWeightContext) -> Result<CtcHeadWeights<'a>, CtcHeadError> {
    let mut layers = Vec::with_capacity(CTC_N_LAYERS);
    for i in 0..CTC_N_LAYERS {
        let p = format!("ctc.transformer.layers.{i}");
        layers.push(EncoderBlockWeights {
            norm1_w: loaded_tensor(loaded, &format!("{p}.norm1.weight"))?,
            norm1_b: loaded_tensor(loaded, &format!("{p}.norm1.bias"))?,
            qkv_w: loaded_tensor(loaded, &format!("{p}.attn.w_qkv.weight"))?,
            out_proj_w: loaded_tensor(loaded, &format!("{p}.attn.out_proj.weight"))?,
            out_proj_b: loaded_tensor(loaded, &format!("{p}.attn.out_proj.bias"))?,
            q_norm_w: loaded_tensor(loaded, &format!("{p}.attn.q_norm.weight"))?,
            q_norm_b: loaded_tensor(loaded, &format!("{p}.attn.q_norm.bias"))?,
            k_norm_w: loaded_tensor(loaded, &format!("{p}.attn.k_norm.weight"))?,
            k_norm_b: loaded_tensor(loaded, &format!("{p}.attn.k_norm.bias"))?,
            norm2_w: loaded_tensor(loaded, &format!("{p}.norm2.weight"))?,
            norm2_b: loaded_tensor(loaded, &format!("{p}.norm2.bias"))?,
            ffn_up_w: loaded_tensor(loaded, &format!("{p}.ffn.net.0.weight"))?,
            ffn_up_b: loaded_tensor(loaded, &format!("{p}.ffn.net.0.bias"))?,
            ffn_down_w: loaded_tensor(loaded, &format!("{p}.ffn.net.3.weight"))?,
            ffn_down_b: loaded_tensor(loaded, &format!("{p}.ffn.net.3.bias"))?,
        });
    }
    Ok(CtcHeadWeights {
        upsample_w: loaded_tensor(loaded, "ctc.upsample.weight")?,
        upsample_b: loaded_tensor(loaded, "ctc.upsample.bias")?,
        embed_norm_w: loaded_tensor(loaded, "ctc.transformer.embed_norm.weight")?,
        embed_norm_b: loaded_tensor(loaded, "ctc.transformer.embed_norm.bias")?,
        layers,
        final_norm_w: loaded_tensor(loaded, "ctc.transformer.final_norm.weight")?,
        final_norm_b: loaded_tensor(loaded, "ctc.transformer.final_norm.bias")?,
        classifier_w: loaded_tensor(loaded, "ctc.classifier.weight")?,
        classifier_b: loaded_tensor(loaded, "ctc.classifier.bias")?,
    })
}

fn graph_config(backend: GgmlCpuGraphBackend) -> GgmlCpuGraphConfig {
    crate::models::graph_runtime_config::apply_request_execution_placement(GgmlCpuGraphConfig {
        context_bytes: GgmlCpuGraphConfig::metadata_context_bytes(HEAD_GRAPH_NODES),
        graph_size: HEAD_GRAPH_NODES,
        n_threads: GgmlCpuGraphConfig::resolve_runtime_thread_count_for(
            backend,
            GgmlCpuGraphThreadingWorkload::EncoderPrelude,
        ),
        backend,
        use_scheduler: true,
    })
}

/// Resident CTC head runtime: a runner plus zero-copy bound pack weights.
pub(crate) struct CtcHeadRuntime {
    runner: GgmlCpuGraphRunner,
    loaded: GgmlLoadedWeightContext,
}

impl CtcHeadRuntime {
    pub(crate) fn new_from_preflight(
        preflight: &crate::ggml_runtime::GgufRuntimeSourcePreflight,
        backend: GgmlCpuGraphBackend,
        host_offload: Option<crate::ggml_runtime::HostWeightOffloadSpec>,
    ) -> Result<Self, CtcHeadError> {
        let mut runner =
            GgmlCpuGraphRunner::new(graph_config(backend)).map_err(ggml_err("runner_init"))?;
        if let Some(spec) = host_offload {
            runner.set_host_weight_offload(spec);
        }
        let loaded = runner
            .load_gguf_weight_context_from_preflight(preflight)
            .map_err(ggml_err("load_weights"))?;
        Ok(Self { runner, loaded })
    }

    /// Project `columns` (CTC class ids, blank first) over every CTC frame.
    /// Returns `frames * columns.len()` f32 logits, frame-major (the same
    /// layout `align_record` expects). The caller bf16-rounds them.
    pub(crate) fn logits(
        &mut self,
        asr_states: &[f32],
        encoder_frames: usize,
        columns: &[u32],
    ) -> Result<Vec<f32>, CtcHeadError> {
        let frames = encoder_frames * CTC_UPSAMPLING;
        let k = columns.len();
        if encoder_frames == 0 || k == 0 {
            return Err(CtcHeadError::Shape {
                reason: "ctc head: empty frames or columns".to_string(),
            });
        }
        if asr_states.len() != encoder_frames * CTC_D_MODEL {
            return Err(CtcHeadError::Shape {
                reason: format!(
                    "ctc head: {} ASR states, expected {}",
                    asr_states.len(),
                    encoder_frames * CTC_D_MODEL
                ),
            });
        }
        if columns.iter().any(|&id| id >= CTC_VOCAB_SIZE as u32) {
            return Err(CtcHeadError::Shape {
                reason: "ctc head: class id outside vocabulary".to_string(),
            });
        }

        let weights = build_weights(&self.loaded)?;
        let config = head_config();
        config.validate("ctc").map_err(|e| CtcHeadError::Shape {
            reason: e.to_string(),
        })?;

        let mut graph = self.runner.start_graph();
        let states_t = graph
            .new_tensor_2d_f32(CTC_D_MODEL, encoder_frames, "ctc_states")
            .map_err(ggml_err("states_alloc"))?;
        let pos_t = graph
            .new_tensor_1d_i32(frames, "ctc_pos")
            .map_err(ggml_err("pos_alloc"))?;
        let ids_t = graph
            .new_tensor_1d_i32(k, "ctc_ids")
            .map_err(ggml_err("ids_alloc"))?;
        graph
            .set_input(states_t)
            .map_err(ggml_err("set_input(states)"))?;
        graph.set_input(pos_t).map_err(ggml_err("set_input(pos)"))?;
        graph.set_input(ids_t).map_err(ggml_err("set_input(ids)"))?;

        // ConvTranspose1d(k=stride=upsampling) as one Linear product: the
        // (upsampling*D, T) result reshapes to (D, upsampling*T).
        let up = linear_with_bias(
            &graph,
            weights.upsample_w,
            states_t,
            weights.upsample_b,
            "ctc_upsample",
        )?;
        let x = graph
            .reshape_3d(up, CTC_D_MODEL, frames, 1)
            .map_err(ggml_err("ctc_reshape"))?;

        let mut hidden = add_norm_weight(
            &graph,
            x,
            weights.embed_norm_w,
            weights.embed_norm_b,
            "ctc_embed_norm",
        )?;
        for layer in &weights.layers {
            hidden =
                rope_transformer_block(&mut graph, hidden, pos_t, &config, "ctc", layer, frames)?;
        }
        let y = add_norm_weight(
            &graph,
            hidden,
            weights.final_norm_w,
            weights.final_norm_b,
            "ctc_final_norm",
        )?;
        let scaled = graph
            .scale(y, CTC_RESIDUAL_SCALE)
            .map_err(ggml_err("ctc_residual_scale"))?;
        let h = graph.add(x, scaled).map_err(ggml_err("ctc_residual_add"))?;
        let h2 = graph
            .reshape_2d(h, CTC_D_MODEL, frames)
            .map_err(ggml_err("ctc_h2"))?;

        // Project only the compact classes: W[ids] (D x K) @ h (D x F) -> K x F.
        let sel_w = graph
            .get_rows(weights.classifier_w, ids_t)
            .map_err(ggml_err("ctc_get_rows_w"))?;
        let bias_1k = graph
            .get_rows(
                graph
                    .reshape_2d(weights.classifier_b, 1, CTC_VOCAB_SIZE)
                    .map_err(ggml_err("ctc_bias_reshape"))?,
                ids_t,
            )
            .map_err(ggml_err("ctc_get_rows_b"))?;
        let bias_k1 = graph
            .reshape_2d(bias_1k, k, 1)
            .map_err(ggml_err("ctc_bias_k1"))?;
        let logits = graph.mul_mat(sel_w, h2).map_err(ggml_err("ctc_project"))?;
        let logits = graph
            .add(logits, bias_k1)
            .map_err(ggml_err("ctc_project_bias"))?;

        graph
            .set_output(logits)
            .map_err(ggml_err("set_output(logits)"))?;
        graph
            .set_f32_slice(states_t, asr_states, "ctc_states")
            .map_err(ggml_err("upload_states"))?;
        let positions: Vec<i32> = (0..frames as i32).collect();
        graph
            .set_i32_slice(pos_t, &positions, "ctc_pos")
            .map_err(ggml_err("upload_pos"))?;
        let ids: Vec<i32> = columns.iter().map(|&id| id as i32).collect();
        graph
            .set_i32_slice(ids_t, &ids, "ctc_ids")
            .map_err(ggml_err("upload_ids"))?;

        let mut out = graph
            .compute_outputs_f32(&[(logits, k * frames)])
            .map_err(ggml_err("compute"))?;
        Ok(out.remove(0))
    }

    pub(crate) fn release_transient_compute_memory(&mut self) -> Result<(), CtcHeadError> {
        self.runner
            .release_transient_scheduler_working_set()
            .map_err(ggml_err("release_transient"))
    }
}

/// Round a float to bf16 (round-to-nearest-even), matching the reference's
/// `bf16_round` used for CTC emissions.
pub(crate) fn bf16_round(v: f32) -> f32 {
    let bits = v.to_bits();
    if bits & 0x7f80_0000 == 0x7f80_0000 {
        return v; // inf / NaN unchanged
    }
    let rounded = bits.wrapping_add(0x7fff).wrapping_add((bits >> 16) & 1) & 0xffff_0000;
    f32::from_bits(rounded)
}
