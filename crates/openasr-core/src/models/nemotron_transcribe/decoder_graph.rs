//! Nemotron-transcribe `nemotron_h_moe` decoder.
//!
//! 52-block trunk (23 Mamba-2 blocks interleaved with 24 MoE blocks and 6
//! attention blocks; the blk.52 MTP/NextN drafter is excluded - it is a
//! speculative-decoding drafter, not a trunk component). Every block is
//! preceded by one shared RMS norm (`attn_norm`); the attention blocks are
//! causal NoPE with a 32:2 GQA layout; the MoE blocks are sigmoid top-6
//! routing with relu-squared experts plus a relu-squared shared expert; the
//! Mamba blocks are the standard Mamba-2 mixer with a causal conv1d and a
//! grouped RMS norm before `ssm_out`. The trunk ends with an RMS norm
//! (`output_norm`) and the (untied) `output` linear head.
//!
//! State is split between a caller-owned resident arena (per-attention-block
//! K/V cache, per-mamba-block SSM state and conv state) and host-side
//! book-keeping (`NemotronTranscribeDecoderRuntime::reset` zeroes the mamba
//! state between utterances). The graph is rebuilt per call; weights stay
//! bound from the load context, only activations allocate.

use thiserror::Error;

use crate::ggml_runtime::{
    GgmlCpuGraphBackend, GgmlCpuGraphBuilder, GgmlCpuGraphConfig, GgmlCpuGraphError,
    GgmlCpuGraphRunner, GgmlCpuTensor, GgmlLoadedTensor, GgmlLoadedWeightContext, GgmlStaticTensor,
    GgmlStaticTensorArena,
};
use crate::nn::half::f32_to_f16_bits;
use crate::nn::norm::{RmsNormSteps, apply_rms_norm};

#[derive(Debug, Error)]
pub(crate) enum NemotronTranscribeDecoderError {
    #[error("nemotron-transcribe decoder shape error: {reason}")]
    Shape { reason: String },
    #[error("nemotron-transcribe decoder missing weight tensor '{name}'")]
    MissingWeight { name: String },
    #[error("nemotron-transcribe decoder GGML backend failed at {stage}: {source}")]
    Ggml {
        stage: &'static str,
        source: GgmlCpuGraphError,
    },
}

fn ggml_err(
    stage: &'static str,
) -> impl Fn(GgmlCpuGraphError) -> NemotronTranscribeDecoderError + Copy {
    move |source| NemotronTranscribeDecoderError::Ggml { stage, source }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BlockType {
    Mamba,
    Attention,
    Moe,
}

/// Trunk block classification from the pack's per-block hparam arrays
/// (`attention.head_count_kv` and `feed_forward_length`): attention iff kv
/// heads are declared and `n_ff == 0`, MoE iff `n_ff != 0`, mamba otherwise.
/// Const so the inventory stays exactly the published generation.
pub(crate) const BLOCK_TYPES: [BlockType; 52] = [
    // 0 mamba, 1 moe, 2 mamba, 3 moe, 4 mamba, 5 attn,
    BlockType::Mamba,
    BlockType::Moe,
    BlockType::Mamba,
    BlockType::Moe,
    BlockType::Mamba,
    BlockType::Attention,
    // 6 moe, 7 mamba, 8 moe, 9 mamba, 10 moe, 11 mamba, 12 attn
    BlockType::Moe,
    BlockType::Mamba,
    BlockType::Moe,
    BlockType::Mamba,
    BlockType::Moe,
    BlockType::Mamba,
    BlockType::Attention,
    // 13 moe, 14 mamba, 15 moe, 16 mamba, 17 moe, 18 mamba, 19 attn
    BlockType::Moe,
    BlockType::Mamba,
    BlockType::Moe,
    BlockType::Mamba,
    BlockType::Moe,
    BlockType::Mamba,
    BlockType::Attention,
    // 20 moe, 21 mamba, 22 moe, 23 mamba, 24 moe, 25 mamba, 26 attn
    BlockType::Moe,
    BlockType::Mamba,
    BlockType::Moe,
    BlockType::Mamba,
    BlockType::Moe,
    BlockType::Mamba,
    BlockType::Attention,
    // 27 moe, 28 mamba, 29 moe, 30 mamba, 31 moe, 32 mamba, 33 attn
    BlockType::Moe,
    BlockType::Mamba,
    BlockType::Moe,
    BlockType::Mamba,
    BlockType::Moe,
    BlockType::Mamba,
    BlockType::Attention,
    // 34 moe, 35 mamba, 36 moe, 37 mamba, 38 moe, 39 mamba, 40 moe
    BlockType::Moe,
    BlockType::Mamba,
    BlockType::Moe,
    BlockType::Mamba,
    BlockType::Moe,
    BlockType::Mamba,
    BlockType::Moe,
    // 41 mamba, 42 attn, 43 moe, 44 mamba, 45 moe, 46 mamba, 47 moe
    BlockType::Mamba,
    BlockType::Attention,
    BlockType::Moe,
    BlockType::Mamba,
    BlockType::Moe,
    BlockType::Mamba,
    BlockType::Moe,
    // 48 mamba, 49 moe, 50 mamba, 51 moe
    BlockType::Mamba,
    BlockType::Moe,
    BlockType::Mamba,
    BlockType::Moe,
];

const SSM_IN_ROW: usize = 2 * NEMOTRON_TRANSCRIBE_SSM_D_INNER
    + 2 * NEMOTRON_TRANSCRIBE_SSM_GROUP * NEMOTRON_TRANSCRIBE_SSM_D_STATE
    + NEMOTRON_TRANSCRIBE_SSM_DT_RANK;
const NEMOTRON_TRANSCRIBE_HIDDEN: usize = 2688;
const NEMOTRON_TRANSCRIBE_VOCAB: usize = 131072;
const NEMOTRON_TRANSCRIBE_SSM_D_CONV: usize = 4;
const NEMOTRON_TRANSCRIBE_SSM_D_STATE: usize = 128;
const NEMOTRON_TRANSCRIBE_SSM_GROUP: usize = 8;
const NEMOTRON_TRANSCRIBE_SSM_D_INNER: usize = 4096;
const NEMOTRON_TRANSCRIBE_SSM_DT_RANK: usize = 64;
const NEMOTRON_TRANSCRIBE_SSM_HEAD_DIM: usize =
    NEMOTRON_TRANSCRIBE_SSM_D_INNER / NEMOTRON_TRANSCRIBE_SSM_DT_RANK;
const NEMOTRON_TRANSCRIBE_CONV_CHANNELS: usize = NEMOTRON_TRANSCRIBE_SSM_D_INNER
    + 2 * NEMOTRON_TRANSCRIBE_SSM_GROUP * NEMOTRON_TRANSCRIBE_SSM_D_STATE;
const NEMOTRON_TRANSCRIBE_ATTN_HEADS: usize = 32;
const NEMOTRON_TRANSCRIBE_ATTN_KV_HEADS: usize = 2;
const NEMOTRON_TRANSCRIBE_ATTN_HEAD_DIM: usize = 128;
const NEMOTRON_TRANSCRIBE_ATTN_SCALE: f32 = 0.088_388_346; // 1 / sqrt(128)
const NEMOTRON_TRANSCRIBE_EXPERT_COUNT: usize = 128;
const NEMOTRON_TRANSCRIBE_EXPERT_USED: usize = 6;
const NEMOTRON_TRANSCRIBE_EXPERT_WEIGHTS_SCALE: f32 = 2.5;
const NEMOTRON_TRANSCRIBE_NORM_EPS: f32 = 1e-5;
const SSM_STATE_FLOATS: usize = NEMOTRON_TRANSCRIBE_SSM_D_STATE
    * NEMOTRON_TRANSCRIBE_SSM_HEAD_DIM
    * NEMOTRON_TRANSCRIBE_SSM_DT_RANK;
const CONV_STATE_FLOATS: usize =
    (NEMOTRON_TRANSCRIBE_SSM_D_CONV - 1) * NEMOTRON_TRANSCRIBE_CONV_CHANNELS;

fn rms_weight<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    input: GgmlCpuTensor<'a>,
    weight: GgmlCpuTensor<'a>,
    name: &'static str,
) -> Result<GgmlCpuTensor<'a>, NemotronTranscribeDecoderError> {
    apply_rms_norm(
        graph,
        input,
        NEMOTRON_TRANSCRIBE_NORM_EPS,
        weight,
        RmsNormSteps {
            norm: name,
            scale: name,
        },
        |stage, source| NemotronTranscribeDecoderError::Ggml { stage, source },
    )
}

fn linear<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    weight: GgmlCpuTensor<'a>,
    input: GgmlCpuTensor<'a>,
    stage: &'static str,
) -> Result<GgmlCpuTensor<'a>, NemotronTranscribeDecoderError> {
    graph.mul_mat(weight, input).map_err(ggml_err(stage))
}

fn relu_sqr<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    input: GgmlCpuTensor<'a>,
    stage: &'static str,
) -> Result<GgmlCpuTensor<'a>, NemotronTranscribeDecoderError> {
    let relu = graph.relu(input).map_err(ggml_err(stage))?;
    graph.sqr(relu).map_err(ggml_err(stage))
}

struct MambaBlockWeights<'a> {
    attn_norm: GgmlCpuTensor<'a>,
    ssm_in: GgmlCpuTensor<'a>,
    ssm_conv1d: GgmlCpuTensor<'a>,
    ssm_conv1d_b: GgmlCpuTensor<'a>,
    ssm_dt_b: GgmlCpuTensor<'a>,
    ssm_a: GgmlCpuTensor<'a>,
    ssm_d: GgmlCpuTensor<'a>,
    ssm_norm: GgmlCpuTensor<'a>,
    ssm_out: GgmlCpuTensor<'a>,
}

struct AttentionBlockWeights<'a> {
    attn_norm: GgmlCpuTensor<'a>,
    attn_q: GgmlCpuTensor<'a>,
    attn_k: GgmlCpuTensor<'a>,
    attn_v: GgmlCpuTensor<'a>,
    attn_output: GgmlCpuTensor<'a>,
}

struct MoeBlockWeights<'a> {
    attn_norm: GgmlCpuTensor<'a>,
    ffn_gate_inp: GgmlCpuTensor<'a>,
    exp_probs_b: GgmlCpuTensor<'a>,
    ffn_up_exps: GgmlCpuTensor<'a>,
    ffn_down_exps: GgmlCpuTensor<'a>,
    ffn_up_shexp: GgmlCpuTensor<'a>,
    ffn_down_shexp: GgmlCpuTensor<'a>,
}

enum BlockWeights<'a> {
    Mamba(MambaBlockWeights<'a>),
    Attention(AttentionBlockWeights<'a>),
    Moe(MoeBlockWeights<'a>),
}

pub(crate) struct DecoderWeights<'a> {
    output_norm: GgmlCpuTensor<'a>,
    output: GgmlCpuTensor<'a>,
    blocks: Vec<BlockWeights<'a>>,
}

fn loaded_tensor<'a>(
    loaded: &GgmlLoadedWeightContext,
    name: &str,
) -> Result<GgmlCpuTensor<'a>, NemotronTranscribeDecoderError> {
    loaded
        .tensor(name)
        .map(GgmlLoadedTensor::as_graph_tensor)
        .ok_or_else(|| NemotronTranscribeDecoderError::MissingWeight {
            name: name.to_string(),
        })
}

pub(crate) fn build_decoder_weights<'a>(
    loaded: &GgmlLoadedWeightContext,
) -> Result<DecoderWeights<'a>, NemotronTranscribeDecoderError> {
    let mut blocks = Vec::with_capacity(BLOCK_TYPES.len());
    for (index, &kind) in BLOCK_TYPES.iter().enumerate() {
        let p = format!("blk.{index}");
        match kind {
            BlockType::Mamba => blocks.push(BlockWeights::Mamba(MambaBlockWeights {
                attn_norm: loaded_tensor(loaded, &format!("{p}.attn_norm.weight"))?,
                ssm_in: loaded_tensor(loaded, &format!("{p}.ssm_in.weight"))?,
                ssm_conv1d: loaded_tensor(loaded, &format!("{p}.ssm_conv1d.weight"))?,
                ssm_conv1d_b: loaded_tensor(loaded, &format!("{p}.ssm_conv1d.bias"))?,
                ssm_dt_b: loaded_tensor(loaded, &format!("{p}.ssm_dt.bias"))?,
                ssm_a: loaded_tensor(loaded, &format!("{p}.ssm_a"))?,
                ssm_d: loaded_tensor(loaded, &format!("{p}.ssm_d"))?,
                ssm_norm: loaded_tensor(loaded, &format!("{p}.ssm_norm.weight"))?,
                ssm_out: loaded_tensor(loaded, &format!("{p}.ssm_out.weight"))?,
            })),
            BlockType::Attention => blocks.push(BlockWeights::Attention(AttentionBlockWeights {
                attn_norm: loaded_tensor(loaded, &format!("{p}.attn_norm.weight"))?,
                attn_q: loaded_tensor(loaded, &format!("{p}.attn_q.weight"))?,
                attn_k: loaded_tensor(loaded, &format!("{p}.attn_k.weight"))?,
                attn_v: loaded_tensor(loaded, &format!("{p}.attn_v.weight"))?,
                attn_output: loaded_tensor(loaded, &format!("{p}.attn_output.weight"))?,
            })),
            BlockType::Moe => blocks.push(BlockWeights::Moe(MoeBlockWeights {
                attn_norm: loaded_tensor(loaded, &format!("{p}.attn_norm.weight"))?,
                ffn_gate_inp: loaded_tensor(loaded, &format!("{p}.ffn_gate_inp.weight"))?,
                exp_probs_b: loaded_tensor(loaded, &format!("{p}.exp_probs_b.bias"))?,
                ffn_up_exps: loaded_tensor(loaded, &format!("{p}.ffn_up_exps.weight"))?,
                ffn_down_exps: loaded_tensor(loaded, &format!("{p}.ffn_down_exps.weight"))?,
                ffn_up_shexp: loaded_tensor(loaded, &format!("{p}.ffn_up_shexp.weight"))?,
                ffn_down_shexp: loaded_tensor(loaded, &format!("{p}.ffn_down_shexp.weight"))?,
            })),
        }
    }
    Ok(DecoderWeights {
        output_norm: loaded_tensor(loaded, "output_norm.weight")?,
        output: loaded_tensor(loaded, "output.weight")?,
        blocks,
    })
}

struct MambaCache<'a> {
    ssm_state: GgmlCpuTensor<'a>,
    conv_state: GgmlCpuTensor<'a>,
    ids: GgmlCpuTensor<'a>,
}

#[allow(clippy::too_many_arguments)]
fn mamba_block<'a>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    weights: &MambaBlockWeights<'a>,
    normed: GgmlCpuTensor<'a>,
    cache: &MambaCache<'a>,
    t: usize,
) -> Result<GgmlCpuTensor<'a>, NemotronTranscribeDecoderError> {
    let map = ggml_err("mamba");
    let esz = std::mem::size_of::<f32>();
    let zx_bcdt = linear(graph, weights.ssm_in, normed, "mamba_ssm_in")?; // (10304, T)
    // z: rows [0, 4096)
    let z = graph
        .view_3d(
            zx_bcdt,
            NEMOTRON_TRANSCRIBE_SSM_D_INNER,
            t,
            1,
            SSM_IN_ROW * esz,
            SSM_IN_ROW * esz * t,
            0,
        )
        .map_err(map)?;
    let z = graph
        .reshape_4d(
            graph.cont(z).map_err(map)?,
            NEMOTRON_TRANSCRIBE_SSM_HEAD_DIM,
            NEMOTRON_TRANSCRIBE_SSM_DT_RANK,
            t,
            1,
        )
        .map_err(map)?;
    // xBC: rows [4096, 10240)
    let x_bc = graph
        .view_3d(
            zx_bcdt,
            NEMOTRON_TRANSCRIBE_CONV_CHANNELS,
            t,
            1,
            SSM_IN_ROW * esz,
            SSM_IN_ROW * esz * t,
            NEMOTRON_TRANSCRIBE_SSM_D_INNER * esz,
        )
        .map_err(map)?;
    // dt: rows [10240, 10304)
    let dt = graph
        .view_3d(
            zx_bcdt,
            NEMOTRON_TRANSCRIBE_SSM_DT_RANK,
            t,
            1,
            SSM_IN_ROW * esz,
            SSM_IN_ROW * esz * t,
            (2 * NEMOTRON_TRANSCRIBE_SSM_D_INNER
                + 2 * NEMOTRON_TRANSCRIBE_SSM_GROUP * NEMOTRON_TRANSCRIBE_SSM_D_STATE)
                * esz,
        )
        .map_err(map)?;
    let dt = graph.cont(dt).map_err(map)?;
    let dt = graph.add(dt, weights.ssm_dt_b).map_err(map)?;

    // Causal conv1d: [conv_state | xBC] -> ssm_conv -> bias -> silu.
    let x_bc_t = graph
        .cont(graph.transpose(x_bc).map_err(map)?)
        .map_err(map)?;
    let conv_x = graph.concat(cache.conv_state, x_bc_t, 0).map_err(map)?;
    let x_bc_c = graph.ssm_conv(conv_x, weights.ssm_conv1d).map_err(map)?;
    let x_bc_c = graph.add(x_bc_c, weights.ssm_conv1d_b).map_err(map)?;
    let x_bc_c = graph.silu(x_bc_c).map_err(map)?;

    // Persist the trailing d_conv-1 rows as the next conv state.
    let conv_row = (t + NEMOTRON_TRANSCRIBE_SSM_D_CONV - 1) * esz;
    let last = graph
        .view_3d(
            conv_x,
            NEMOTRON_TRANSCRIBE_SSM_D_CONV - 1,
            NEMOTRON_TRANSCRIBE_CONV_CHANNELS,
            1,
            conv_row,
            conv_row,
            t * esz,
        )
        .map_err(map)?;
    let last = graph.cont(last).map_err(map)?;
    let conv_write = graph.cpy_into_view(last, cache.conv_state).map_err(map)?;
    graph.add_side_effect_root(conv_write).map_err(map)?;

    // SSM scan over the whole sequence at once (K = 1).
    let x_scan = graph
        .view_4d(
            x_bc_c,
            NEMOTRON_TRANSCRIBE_SSM_HEAD_DIM,
            NEMOTRON_TRANSCRIBE_SSM_DT_RANK,
            t,
            1,
            NEMOTRON_TRANSCRIBE_SSM_HEAD_DIM * esz,
            NEMOTRON_TRANSCRIBE_CONV_CHANNELS * esz,
            NEMOTRON_TRANSCRIBE_CONV_CHANNELS * esz * t,
            0,
        )
        .map_err(map)?;
    let b = graph
        .view_4d(
            x_bc_c,
            NEMOTRON_TRANSCRIBE_SSM_D_STATE,
            NEMOTRON_TRANSCRIBE_SSM_GROUP,
            t,
            1,
            NEMOTRON_TRANSCRIBE_SSM_D_STATE * esz,
            NEMOTRON_TRANSCRIBE_CONV_CHANNELS * esz,
            NEMOTRON_TRANSCRIBE_CONV_CHANNELS * esz * t,
            NEMOTRON_TRANSCRIBE_SSM_D_INNER * esz,
        )
        .map_err(map)?;
    let c = graph
        .view_4d(
            x_bc_c,
            NEMOTRON_TRANSCRIBE_SSM_D_STATE,
            NEMOTRON_TRANSCRIBE_SSM_GROUP,
            t,
            1,
            NEMOTRON_TRANSCRIBE_SSM_D_STATE * esz,
            NEMOTRON_TRANSCRIBE_CONV_CHANNELS * esz,
            NEMOTRON_TRANSCRIBE_CONV_CHANNELS * esz * t,
            (NEMOTRON_TRANSCRIBE_SSM_D_INNER
                + NEMOTRON_TRANSCRIBE_SSM_GROUP * NEMOTRON_TRANSCRIBE_SSM_D_STATE)
                * esz,
        )
        .map_err(map)?;
    let scan = graph
        .ssm_scan(
            cache.ssm_state,
            x_scan,
            dt,
            weights.ssm_a,
            b,
            c,
            cache.ids,
            1,
        )
        .map_err(map)?;
    // Scan output: [y (nelements(x)) | final state (d_state*d_inner)] 1D.
    let y_elems = NEMOTRON_TRANSCRIBE_SSM_D_INNER * t;
    let y = graph
        .view_4d(
            scan,
            NEMOTRON_TRANSCRIBE_SSM_HEAD_DIM,
            NEMOTRON_TRANSCRIBE_SSM_DT_RANK,
            t,
            1,
            NEMOTRON_TRANSCRIBE_SSM_HEAD_DIM * esz,
            NEMOTRON_TRANSCRIBE_SSM_D_INNER * esz,
            NEMOTRON_TRANSCRIBE_SSM_D_INNER * esz * t,
            0,
        )
        .map_err(map)?;
    let state_new = graph
        .view_1d(
            scan,
            NEMOTRON_TRANSCRIBE_SSM_D_STATE * NEMOTRON_TRANSCRIBE_SSM_D_INNER,
            y_elems * esz,
        )
        .map_err(map)?;
    let state_new = graph.cont(state_new).map_err(map)?;
    let state_new = graph
        .reshape_4d(
            state_new,
            NEMOTRON_TRANSCRIBE_SSM_D_STATE,
            NEMOTRON_TRANSCRIBE_SSM_HEAD_DIM,
            NEMOTRON_TRANSCRIBE_SSM_DT_RANK,
            1,
        )
        .map_err(map)?;
    let ssm_write = graph
        .cpy_into_view(state_new, cache.ssm_state)
        .map_err(map)?;
    graph.add_side_effect_root(ssm_write).map_err(map)?;

    // Skip connection (per SSM head) and z-gating.
    let y = graph
        .add(y, graph.mul(x_scan, weights.ssm_d).map_err(map)?)
        .map_err(map)?;
    let y = graph
        .swiglu_split(graph.cont(z).map_err(map)?, y)
        .map_err(map)?;

    // Grouped RMS norm over (d_inner/n_group, n_group), then ssm_out.
    let y = graph
        .reshape_4d(
            y,
            NEMOTRON_TRANSCRIBE_SSM_D_INNER / NEMOTRON_TRANSCRIBE_SSM_GROUP,
            NEMOTRON_TRANSCRIBE_SSM_GROUP,
            t,
            1,
        )
        .map_err(map)?;
    let y = rms_weight(graph, y, weights.ssm_norm, "mamba_ssm_norm")?;
    let y = graph
        .reshape_2d(y, NEMOTRON_TRANSCRIBE_SSM_D_INNER, t)
        .map_err(map)?;
    linear(graph, weights.ssm_out, y, "mamba_ssm_out")
}

struct AttentionCache<'a> {
    arena_k: GgmlCpuTensor<'a>,
    arena_v: GgmlCpuTensor<'a>,
    write_rows: GgmlCpuTensor<'a>,
    span: usize,
    max_positions: usize,
}

#[allow(clippy::too_many_arguments)]
fn attention_block<'a>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    weights: &AttentionBlockWeights<'a>,
    normed: GgmlCpuTensor<'a>,
    cache: &AttentionCache<'a>,
    t: usize,
    prefill_mask: Option<GgmlCpuTensor<'a>>,
) -> Result<GgmlCpuTensor<'a>, NemotronTranscribeDecoderError> {
    let map = ggml_err("attn");
    let esz = std::mem::size_of::<f32>();
    let q = linear(graph, weights.attn_q, normed, "attn_q")?;
    let k = linear(graph, weights.attn_k, normed, "attn_k")?;
    let v = linear(graph, weights.attn_v, normed, "attn_v")?;
    let q4 = graph
        .reshape_4d(
            q,
            NEMOTRON_TRANSCRIBE_ATTN_HEAD_DIM,
            NEMOTRON_TRANSCRIBE_ATTN_HEADS,
            t,
            1,
        )
        .map_err(map)?;
    let k4 = graph
        .reshape_4d(
            k,
            NEMOTRON_TRANSCRIBE_ATTN_HEAD_DIM,
            NEMOTRON_TRANSCRIBE_ATTN_KV_HEADS,
            t,
            1,
        )
        .map_err(map)?;
    let v4 = graph
        .reshape_4d(
            v,
            NEMOTRON_TRANSCRIBE_ATTN_HEAD_DIM,
            NEMOTRON_TRANSCRIBE_ATTN_KV_HEADS,
            t,
            1,
        )
        .map_err(map)?;
    // Write K/V at rows [p0, p0+t) of the resident span.
    let k_rows = graph
        .cont(graph.permute(k4, 0, 2, 1, 3).map_err(map)?)
        .map_err(map)?;
    let v_rows = graph
        .cont(graph.permute(v4, 0, 2, 1, 3).map_err(map)?)
        .map_err(map)?;
    let k_write = graph
        .set_kv_rows(cache.arena_k, k_rows, cache.write_rows)
        .map_err(map)?;
    graph.add_kv_write_root(k_write).map_err(map)?;
    let v_write = graph
        .set_kv_rows(cache.arena_v, v_rows, cache.write_rows)
        .map_err(map)?;
    graph.add_kv_write_root(v_write).map_err(map)?;
    // NoPE attention over the resident span. Flash attention is avoided on
    // this backend (it silently yields all-zero outputs for this shape);
    // the naive path (mul_mat + soft_max_ext) is what the rest of the
    // hybrid families use. GQA broadcasts natively in mul_mat.
    let span = cache.span;
    let k_span = graph
        .view_3d(
            cache.arena_k,
            NEMOTRON_TRANSCRIBE_ATTN_HEAD_DIM,
            span,
            NEMOTRON_TRANSCRIBE_ATTN_KV_HEADS,
            NEMOTRON_TRANSCRIBE_ATTN_HEAD_DIM * esz,
            NEMOTRON_TRANSCRIBE_ATTN_HEAD_DIM * cache.max_positions * esz,
            0,
        )
        .map_err(map)?;
    let v_span = graph
        .view_3d(
            cache.arena_v,
            NEMOTRON_TRANSCRIBE_ATTN_HEAD_DIM,
            span,
            NEMOTRON_TRANSCRIBE_ATTN_KV_HEADS,
            NEMOTRON_TRANSCRIBE_ATTN_HEAD_DIM * esz,
            NEMOTRON_TRANSCRIBE_ATTN_HEAD_DIM * cache.max_positions * esz,
            0,
        )
        .map_err(map)?;
    // Keep (head_dim, span, kv_heads) layout; q permuted to (head_dim, t, heads).
    let k3 = graph.cont(k_span).map_err(map)?;
    let v3 = graph.cont(v_span).map_err(map)?;
    let q3 = graph
        .cont(graph.permute(q4, 0, 2, 1, 3).map_err(map)?)
        .map_err(map)?;
    let scores = graph.mul_mat(k3, q3).map_err(map)?; // (span, t, heads)
    let probs = graph
        .soft_max_ext(scores, prefill_mask, NEMOTRON_TRANSCRIBE_ATTN_SCALE, 0.0)
        .map_err(map)?;
    let v_t = graph
        .cont(graph.permute(v3, 1, 0, 2, 3).map_err(map)?)
        .map_err(map)?; // (span, head_dim, kv_heads)
    let context = graph.mul_mat(v_t, probs).map_err(map)?; // (head_dim, t, heads)
    let merged = graph
        .cont(graph.permute(context, 0, 2, 1, 3).map_err(map)?)
        .map_err(map)?;
    let merged = graph
        .reshape_2d(
            merged,
            NEMOTRON_TRANSCRIBE_ATTN_HEAD_DIM * NEMOTRON_TRANSCRIBE_ATTN_HEADS,
            t,
        )
        .map_err(map)?;
    linear(graph, weights.attn_output, merged, "attn_output")
}

fn moe_block<'a>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    weights: &MoeBlockWeights<'a>,
    normed: GgmlCpuTensor<'a>,
    t: usize,
) -> Result<GgmlCpuTensor<'a>, NemotronTranscribeDecoderError> {
    let map = ggml_err("moe");
    let esz = std::mem::size_of::<f32>();
    // Router (F32 in trunk, BF16 at blk.52 which plain greedy never enters).
    let router = linear(graph, weights.ffn_gate_inp, normed, "moe_router")?; // (128, T)
    let probs = graph.sigmoid(router).map_err(map)?;
    // Top-6 by biased sigmoid score (DeepSeek-style selection bias; the raw
    // sigmoid probabilities stay unbiased and are used for the weights).
    let selection_probs = graph.add(probs, weights.exp_probs_b).map_err(map)?;
    let selected = graph
        .argsort_top_k(selection_probs, NEMOTRON_TRANSCRIBE_EXPERT_USED)
        .map_err(map)?; // (6, T)
    let selected = graph.cont(selected).map_err(map)?;
    // Gather the sigmoid weights, normalize, scale, apply.
    let probs_3d = graph
        .reshape_3d(probs, 1, NEMOTRON_TRANSCRIBE_EXPERT_COUNT, t)
        .map_err(map)?;
    let weights_3d = graph.get_rows(probs_3d, selected).map_err(map)?; // (1, 6, T)
    let weights_2d = graph
        .reshape_2d(weights_3d, NEMOTRON_TRANSCRIBE_EXPERT_USED, t)
        .map_err(map)?;
    let sum = graph.sum_rows(weights_2d).map_err(map)?; // (1, T)
    let sum = graph.clamp(sum, 6.103_515_6e-5, f32::MAX).map_err(map)?;
    let weights_2d = graph.div(weights_2d, sum).map_err(map)?;
    let weights_2d = graph
        .scale(weights_2d, NEMOTRON_TRANSCRIBE_EXPERT_WEIGHTS_SCALE)
        .map_err(map)?;
    let expert_weights = graph
        .reshape_3d(weights_2d, 1, NEMOTRON_TRANSCRIBE_EXPERT_USED, t)
        .map_err(map)?;

    // Expert FFN: relu-squared, no gate. mul_mat_id over [in, out, expert].
    let tokens = graph
        .reshape_3d(normed, NEMOTRON_TRANSCRIBE_HIDDEN, 1, t)
        .map_err(map)?;
    let up = graph
        .mul_mat_id(weights.ffn_up_exps, tokens, selected)
        .map_err(map)?; // (1856, 6, T)
    let act = relu_sqr(graph, up, "moe_relu_sqr")?;
    let experts = graph
        .mul_mat_id(weights.ffn_down_exps, act, selected)
        .map_err(map)?; // (2688, 6, T)
    let experts = graph.mul(experts, expert_weights).map_err(map)?;
    // Sum the selected experts (experts: (2688, 6, T), contiguous).
    let expert_stride = NEMOTRON_TRANSCRIBE_HIDDEN * esz;
    let t_stride = NEMOTRON_TRANSCRIBE_HIDDEN * NEMOTRON_TRANSCRIBE_EXPERT_USED * esz;
    let mut moe_out = graph
        .view_2d(experts, NEMOTRON_TRANSCRIBE_HIDDEN, t, t_stride, 0)
        .map_err(map)?;
    for expert_index in 1..NEMOTRON_TRANSCRIBE_EXPERT_USED {
        let part = graph
            .view_2d(
                experts,
                NEMOTRON_TRANSCRIBE_HIDDEN,
                t,
                t_stride,
                expert_index * expert_stride,
            )
            .map_err(map)?;
        moe_out = graph.add(moe_out, part).map_err(map)?;
    }

    // Shared expert (relu-squared, no router weights).
    let shexp = linear(graph, weights.ffn_up_shexp, normed, "moe_shexp_up")?;
    let shexp = relu_sqr(graph, shexp, "moe_shexp_relu_sqr")?;
    let shexp = linear(graph, weights.ffn_down_shexp, shexp, "moe_shexp_down")?;
    graph.add(moe_out, shexp).map_err(map)
}

pub(crate) struct NemotronTranscribeDecoderRuntime {
    runner: GgmlCpuGraphRunner,
    loaded: GgmlLoadedWeightContext,
    state_arena: GgmlStaticTensorArena,
    kv_arena: GgmlStaticTensorArena,
    ssm_tensors: Vec<(usize, GgmlStaticTensor)>,
    conv_tensors: Vec<(usize, GgmlStaticTensor)>,
    kv_tensors: Vec<(usize, GgmlStaticTensor, GgmlStaticTensor)>,
    kv_max_positions: usize,
    cached_positions: usize,
}

impl NemotronTranscribeDecoderRuntime {
    const STATE_ARENA_TENSORS: usize = 64;
    const KV_ARENA_TENSORS: usize = 16;

    pub(crate) fn new_from_preflight(
        preflight: &crate::ggml_runtime::GgufRuntimeSourcePreflight,
        max_positions: usize,
        backend: GgmlCpuGraphBackend,
    ) -> Result<Self, NemotronTranscribeDecoderError> {
        let graph_config = decoder_graph_config(backend);
        let runner = GgmlCpuGraphRunner::new(graph_config).map_err(ggml_err("runner_init"))?;
        let loaded = runner
            .load_gguf_weight_context_from_preflight(preflight)
            .map_err(ggml_err("load_gguf_weight_context"))?;

        let mamba_blocks: Vec<usize> = BLOCK_TYPES
            .iter()
            .enumerate()
            .filter(|(_, kind)| **kind == BlockType::Mamba)
            .map(|(index, _)| index)
            .collect();
        let attention_blocks: Vec<usize> = BLOCK_TYPES
            .iter()
            .enumerate()
            .filter(|(_, kind)| **kind == BlockType::Attention)
            .map(|(index, _)| index)
            .collect();

        let state_context = GgmlCpuGraphConfig::metadata_context_bytes(Self::STATE_ARENA_TENSORS);
        let state_arena = runner
            .start_state_tensor_arena(state_context)
            .map_err(ggml_err("state_arena"))?;
        let mut ssm_tensors = Vec::with_capacity(mamba_blocks.len());
        let mut conv_tensors = Vec::with_capacity(mamba_blocks.len());
        for &block in &mamba_blocks {
            let ssm = state_arena
                .new_tensor_4d_f32(
                    NEMOTRON_TRANSCRIBE_SSM_D_STATE,
                    NEMOTRON_TRANSCRIBE_SSM_HEAD_DIM,
                    NEMOTRON_TRANSCRIBE_SSM_DT_RANK,
                    1,
                    "nemotron_ssm_state",
                )
                .map_err(ggml_err("ssm_state_alloc"))?;
            let conv = state_arena
                .new_tensor_3d_f32(
                    NEMOTRON_TRANSCRIBE_SSM_D_CONV - 1,
                    NEMOTRON_TRANSCRIBE_CONV_CHANNELS,
                    1,
                    "nemotron_conv_state",
                )
                .map_err(ggml_err("conv_state_alloc"))?;
            ssm_tensors.push((block, ssm));
            conv_tensors.push((block, conv));
        }

        let kv_context = GgmlCpuGraphConfig::metadata_context_bytes(Self::KV_ARENA_TENSORS);
        let kv_arena = runner
            .start_state_tensor_arena(kv_context)
            .map_err(ggml_err("kv_arena"))?;
        let mut kv_tensors = Vec::with_capacity(attention_blocks.len());
        for &block in &attention_blocks {
            let k = kv_arena
                .new_tensor_3d_f32(
                    NEMOTRON_TRANSCRIBE_ATTN_HEAD_DIM,
                    max_positions,
                    NEMOTRON_TRANSCRIBE_ATTN_KV_HEADS,
                    "nemotron_kv_k",
                )
                .map_err(ggml_err("kv_k_alloc"))?;
            let v = kv_arena
                .new_tensor_3d_f32(
                    NEMOTRON_TRANSCRIBE_ATTN_HEAD_DIM,
                    max_positions,
                    NEMOTRON_TRANSCRIBE_ATTN_KV_HEADS,
                    "nemotron_kv_v",
                )
                .map_err(ggml_err("kv_v_alloc"))?;
            kv_tensors.push((block, k, v));
        }

        let mut runtime = Self {
            runner,
            loaded,
            state_arena,
            kv_arena,
            ssm_tensors,
            conv_tensors,
            kv_tensors,
            kv_max_positions: max_positions,
            cached_positions: 0,
        };
        runtime.reset_state_arenas()?;
        Ok(runtime)
    }

    /// Zero the mamba state between utterances; the KV span is tracked by
    /// `cached_positions` and its stale rows are never read (the span view
    /// shrinks to the written prefix).
    pub(crate) fn reset_state_arenas(&mut self) -> Result<(), NemotronTranscribeDecoderError> {
        let ssm_len = SSM_STATE_FLOATS;
        let conv_len = CONV_STATE_FLOATS;
        let ssm_zero = vec![0.0f32; ssm_len];
        let conv_zero = vec![0.0f32; conv_len];
        for &(_, tensor) in &self.ssm_tensors {
            self.state_arena
                .set_f32_slice(tensor, &ssm_zero, "nemotron_ssm_state_reset")
                .map_err(ggml_err("ssm_state_reset"))?;
        }
        for &(_, tensor) in &self.conv_tensors {
            self.state_arena
                .set_f32_slice(tensor, &conv_zero, "nemotron_conv_state_reset")
                .map_err(ggml_err("conv_state_reset"))?;
        }
        let kv_zero = vec![
            0.0f32;
            self.kv_max_positions
                * NEMOTRON_TRANSCRIBE_ATTN_HEAD_DIM
                * NEMOTRON_TRANSCRIBE_ATTN_KV_HEADS
        ];
        for &(_, k, v) in &self.kv_tensors {
            self.kv_arena
                .set_f32_slice(k, &kv_zero, "nemotron_kv_k_reset")
                .map_err(ggml_err("kv_k_reset"))?;
            self.kv_arena
                .set_f32_slice(v, &kv_zero, "nemotron_kv_v_reset")
                .map_err(ggml_err("kv_v_reset"))?;
        }
        self.cached_positions = 0;
        Ok(())
    }

    fn positions_filled(&self) -> usize {
        self.cached_positions
    }
}

const DECODER_GRAPH_NODES: usize = 4096;

fn decoder_graph_config(backend: GgmlCpuGraphBackend) -> GgmlCpuGraphConfig {
    crate::models::graph_runtime_config::apply_request_execution_placement(GgmlCpuGraphConfig {
        context_bytes: GgmlCpuGraphConfig::metadata_context_bytes(DECODER_GRAPH_NODES),
        graph_size: DECODER_GRAPH_NODES,
        n_threads: GgmlCpuGraphConfig::resolve_runtime_thread_count_for(
            backend,
            crate::ggml_runtime::GgmlCpuGraphThreadingWorkload::Decoder,
        ),
        backend,
        use_scheduler: true,
    })
}

struct ForwardInputs<'a> {
    embeddings: GgmlCpuTensor<'a>,
    write_rows: GgmlCpuTensor<'a>,
    ids: GgmlCpuTensor<'a>,
    mamba_states: Vec<(usize, GgmlCpuTensor<'a>, GgmlCpuTensor<'a>)>,
    kv_tensors: Vec<(usize, GgmlCpuTensor<'a>, GgmlCpuTensor<'a>)>,
    span: usize,
    kv_max_positions: usize,
    prefill_mask: Option<GgmlCpuTensor<'a>>,
}

impl NemotronTranscribeDecoderRuntime {
    fn forward<'a>(
        weights: &DecoderWeights<'a>,
        graph: &mut GgmlCpuGraphBuilder<'a>,
        inputs: ForwardInputs<'a>,
        t: usize,
    ) -> Result<GgmlCpuTensor<'a>, NemotronTranscribeDecoderError> {
        let map = ggml_err("forward");
        let mut hidden = inputs.embeddings;
        for (block_index, block) in weights.blocks.iter().enumerate() {
            match block {
                BlockWeights::Mamba(mamba) => {
                    let normed = rms_weight(graph, hidden, mamba.attn_norm, "decoder_norm")?;
                    let (_, ssm_state, conv_state) = inputs
                        .mamba_states
                        .iter()
                        .find(|(index, _, _)| *index == block_index)
                        .ok_or_else(|| NemotronTranscribeDecoderError::Shape {
                            reason: format!("missing mamba state for block {block_index}"),
                        })?;
                    let mamba_out = mamba_block(
                        graph,
                        mamba,
                        normed,
                        &MambaCache {
                            ssm_state: *ssm_state,
                            conv_state: *conv_state,
                            ids: inputs.ids,
                        },
                        t,
                    )?;
                    hidden = graph.add(hidden, mamba_out).map_err(map)?;
                }
                BlockWeights::Attention(attention) => {
                    let normed = rms_weight(graph, hidden, attention.attn_norm, "decoder_norm")?;
                    let (_, arena_k, arena_v) = inputs
                        .kv_tensors
                        .iter()
                        .find(|(index, _, _)| *index == block_index)
                        .ok_or_else(|| NemotronTranscribeDecoderError::Shape {
                            reason: format!("missing kv arena for attention block {block_index}"),
                        })?;
                    let attn_out = attention_block(
                        graph,
                        attention,
                        normed,
                        &AttentionCache {
                            arena_k: *arena_k,
                            arena_v: *arena_v,
                            write_rows: inputs.write_rows,
                            span: inputs.span,
                            max_positions: inputs.kv_max_positions,
                        },
                        t,
                        inputs.prefill_mask,
                    )?;
                    hidden = graph.add(hidden, attn_out).map_err(map)?;
                }
                BlockWeights::Moe(moe) => {
                    let normed = rms_weight(graph, hidden, moe.attn_norm, "decoder_norm")?;
                    let moe_out = moe_block(graph, moe, normed, t)?;
                    hidden = graph.add(hidden, moe_out).map_err(map)?;
                }
            }
        }
        let normed = rms_weight(graph, hidden, weights.output_norm, "decoder_output_norm")?;
        let logits = linear(graph, weights.output, normed, "decoder_output")?;
        // Return the last position's logits.
        let esz = std::mem::size_of::<f32>();
        graph
            .view_2d(
                logits,
                NEMOTRON_TRANSCRIBE_VOCAB,
                1,
                NEMOTRON_TRANSCRIBE_VOCAB * esz,
                (t - 1) * NEMOTRON_TRANSCRIBE_VOCAB * esz,
            )
            .map_err(map)
    }
}

fn build_prefill_mask_f16_bits(span: usize, t: usize, write_start: usize) -> Vec<u16> {
    debug_assert_eq!(span, t + write_start);
    let mut bits = vec![f32_to_f16_bits(0.0); span * t];
    let neg_inf = f32_to_f16_bits(f32::NEG_INFINITY);
    for query in 0..t {
        let kv_limit = write_start + query + 1;
        for kv in kv_limit..span {
            bits[query * span + kv] = neg_inf;
        }
    }
    bits
}

impl NemotronTranscribeDecoderRuntime {
    /// Prefill `t` embedding rows (t x hidden) through the trunk; resets the
    /// mamba recurrent state first. Returns the logits for the final position
    /// and the minted compute witness (when a collector is installed).
    pub(crate) fn prefill(
        &mut self,
        embeddings: &[f32],
        t: usize,
    ) -> Result<
        (
            Vec<f32>,
            Option<crate::ggml_runtime::GgmlSelectionEvidenceRef>,
        ),
        NemotronTranscribeDecoderError,
    > {
        if t == 0 || embeddings.len() != t * NEMOTRON_TRANSCRIBE_HIDDEN {
            return Err(NemotronTranscribeDecoderError::Shape {
                reason: format!(
                    "prefill expects t>0 embedding rows of width {NEMOTRON_TRANSCRIBE_HIDDEN}, got {} values for {t} rows",
                    embeddings.len()
                ),
            });
        }
        if t > self.kv_max_positions {
            return Err(NemotronTranscribeDecoderError::Shape {
                reason: format!(
                    "prefill of {t} frames exceeds the resident KV span {}",
                    self.kv_max_positions
                ),
            });
        }
        self.reset_state_arenas()?;
        let write_start = 0usize;
        let span = write_start + t;
        self.run_forward(embeddings, t, write_start, span, /* prefill */ true)
    }

    /// Advance one token; `cached_positions` positions must already be written.
    pub(crate) fn decode_step(
        &mut self,
        embedding: &[f32],
    ) -> Result<
        (
            Vec<f32>,
            Option<crate::ggml_runtime::GgmlSelectionEvidenceRef>,
        ),
        NemotronTranscribeDecoderError,
    > {
        if embedding.len() != NEMOTRON_TRANSCRIBE_HIDDEN {
            return Err(NemotronTranscribeDecoderError::Shape {
                reason: format!(
                    "decode step expects one {NEMOTRON_TRANSCRIBE_HIDDEN}-wide embedding row, got {} values",
                    embedding.len()
                ),
            });
        }
        if self.cached_positions >= self.kv_max_positions {
            return Err(NemotronTranscribeDecoderError::Shape {
                reason: format!(
                    "decode position {} exhausts the resident KV span {}",
                    self.cached_positions, self.kv_max_positions
                ),
            });
        }
        let write_start = self.cached_positions;
        self.run_forward(
            embedding,
            1,
            write_start,
            write_start + 1,
            /* prefill */ false,
        )
    }

    fn run_forward(
        &mut self,
        embeddings: &[f32],
        t: usize,
        write_start: usize,
        span: usize,
        prefill: bool,
    ) -> Result<
        (
            Vec<f32>,
            Option<crate::ggml_runtime::GgmlSelectionEvidenceRef>,
        ),
        NemotronTranscribeDecoderError,
    > {
        let weights = build_decoder_weights(&self.loaded)?;
        let write_rows: Vec<i32> = (write_start..write_start + t)
            .map(|position| i32::try_from(position).expect("span fits i32"))
            .collect();
        let ids = [0i32];
        let mamba_states: Vec<(usize, GgmlCpuTensor<'_>, GgmlCpuTensor<'_>)> = self
            .ssm_tensors
            .iter()
            .zip(self.conv_tensors.iter())
            .map(|((block, ssm), (_, conv))| {
                (
                    *block,
                    self.state_arena.graph_tensor(*ssm),
                    self.state_arena.graph_tensor(*conv),
                )
            })
            .collect();
        let kv_tensors: Vec<(usize, GgmlCpuTensor<'_>, GgmlCpuTensor<'_>)> = self
            .kv_tensors
            .iter()
            .map(|&(block, k, v)| {
                (
                    block,
                    self.kv_arena.graph_tensor(k),
                    self.kv_arena.graph_tensor(v),
                )
            })
            .collect();

        let mut graph = self.runner.start_graph();
        let embeddings_t = graph
            .new_tensor_2d_f32(NEMOTRON_TRANSCRIBE_HIDDEN, t, "decoder_embeddings")
            .map_err(ggml_err("embeddings_alloc"))?;
        let rows_t = graph
            .new_tensor_1d_i32(t, "decoder_write_rows")
            .map_err(ggml_err("write_rows_alloc"))?;
        let ids_t = graph
            .new_tensor_1d_i32(1, "decoder_ids")
            .map_err(ggml_err("ids_alloc"))?;
        let mask_t = if prefill && t > 1 {
            Some(
                graph
                    .new_tensor_3d_f16(span, t, 1, "decoder_mask")
                    .map_err(ggml_err("mask_alloc"))?,
            )
        } else {
            None
        };
        graph
            .set_input(embeddings_t)
            .map_err(ggml_err("set_input(embeddings)"))?;
        graph
            .set_input(rows_t)
            .map_err(ggml_err("set_input(write_rows)"))?;
        graph.set_input(ids_t).map_err(ggml_err("set_input(ids)"))?;
        if let Some(mask) = mask_t {
            graph.set_input(mask).map_err(ggml_err("set_input(mask)"))?;
        }

        let last_logits = Self::forward(
            &weights,
            &mut graph,
            ForwardInputs {
                embeddings: embeddings_t,
                write_rows: rows_t,
                ids: ids_t,
                mamba_states,
                kv_tensors,
                span,
                kv_max_positions: self.kv_max_positions,
                prefill_mask: mask_t,
            },
            t,
        )?;
        graph
            .set_output(last_logits)
            .map_err(ggml_err("set_output(logits)"))?;
        graph
            .set_f32_slice(embeddings_t, embeddings, "decoder_embeddings")
            .map_err(ggml_err("upload_embeddings"))?;
        graph
            .set_i32_slice(rows_t, &write_rows, "decoder_write_rows")
            .map_err(ggml_err("upload_write_rows"))?;
        graph
            .set_i32_slice(ids_t, &ids, "decoder_ids")
            .map_err(ggml_err("upload_ids"))?;
        if let Some(mask) = mask_t {
            let bits = build_prefill_mask_f16_bits(span, t, write_start);
            graph
                .set_f16_bits_slice(mask, &bits, "decoder_mask")
                .map_err(ggml_err("upload_mask"))?;
        }

        let taps: Vec<(GgmlCpuTensor<'_>, usize)> = vec![(last_logits, NEMOTRON_TRANSCRIBE_VOCAB)];
        let output = graph
            .compute_outputs_f32_with_evidence(&taps)
            .map_err(ggml_err("compute"))?;
        self.cached_positions += t;
        let (mut values, evidence) = output.into_parts();
        let logits = values.remove(0);
        Ok((logits, evidence))
    }

    pub(crate) fn cached_positions(&self) -> usize {
        self.positions_filled()
    }
}
