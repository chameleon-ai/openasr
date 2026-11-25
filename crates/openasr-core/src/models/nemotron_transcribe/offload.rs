//! VRAM-derived MoE-to-RAM offload policy for nemotron-transcribe.
//!
//! The 23 trunk MoE blocks plus the MTP drafter (block 52) hold about 73% of
//! the pack's weight bytes. When the detected device memory cannot keep the
//! whole pack and the worst-case context footprint under the 95% admission
//! policy ceiling, this module computes which expert tensors must move to
//! system RAM and returns the matching [`HostWeightOffloadSpec`]; the ggml
//! weight loader applies the split at buffer-load time. The offload count is
//! always a function of the detected device memory and the pack's tensor
//! sizes - never a fixed knob, never an environment variable.
//!
//! Tiering follows the reference implementation's "least important first"
//! ordering:
//! 1. block 52 (MTP drafter) experts are in the pack but greedy decode never
//!    executes them, so offloading them costs nothing at runtime - always
//!    first;
//! 2. the Q2_0-precision trunk MoE layers, earliest first;
//! 3. only if still short, the IQ4_NL-precision trunk MoE layers (kept at
//!    higher precision in the published pack, so most valuable), earliest
//!    first.
//!
//! Sizing constants: `nemotron-offloading.md` sections 2.3 and 3.1, plus the
//! Hybrid lane's per-graph BACKEND_PRIVATE admission measured on the discrete
//! card (the spec's 2 GiB context floor alone is not enough: the decoder graph
//! is quoted at ~7.3 GiB for the full 90 s reserve span).

use crate::device::execution_route::ExecutionProvider;
use crate::ggml_runtime::{
    GgmlBackendKind, GgmlCpuGraphBackend, GgufTensorIndex, HostWeightOffloadSpec,
    ggml_available_devices,
};

/// Fraction of the device's reported total memory the admission policy lets
/// the engine own, in basis points. Mirrors
/// `crate::device::execution_memory::DeviceMemoryPolicy`'s default
/// (`maximum_owned_basis_points = 9_500`): the split must keep weights and
/// context under that same ceiling, not a looser one.
const OWNED_FRACTION_BASIS_POINTS: u64 = 9_500;

/// Absolute driver/external-process reserve the broker holds back from the
/// live free snapshot. Mirrors `DeviceMemoryPolicy::default()`'s
/// `minimum_headroom_bytes`. The Hybrid lane admits the scheduler-private
/// graph buffers against **live free** VRAM, not the policy ceiling on total,
/// so the offload sizing must respect this observed ceiling too.
const BROKER_MINIMUM_HEADROOM_BYTES: u64 = 256 * 1024 * 1024;

/// Device-wide footprint of the context buffer plus KV/SSM/conv state for the
/// longest (90 s) slice, including the desktop/driver baseline that shares
/// the card. The floor is the recon's device-wide peak minus weights,
/// measured on a 94 s clip (1,778,193,920 B); rounded up to 2 GiB to cover
/// driver allocation spikes and slightly longer slices.
const CONTEXT_AND_STATE_FOOTPRINT_BYTES: u64 = 1024 * 1024 * 1024;

/// Extra live-free VRAM kept beyond the estimated graph/state footprint. The
/// scheduler admits the context buffer and graph scratch against the *live*
/// free heap, which also pays driver workspaces and transient allocation
/// spikes. A 1 GiB margin leaves ~1.5 GiB of the 16 GB card's VRAM unused at
/// peak; the 2 GiB margin used during bring-up was unnecessarily large for a
/// discrete card whose VRAM domain is separate from system memory. The APU
/// path never calls this module (`device_vram_bytes` returns `None` there), so
/// the 40% rule calibrated on gfx1151 is untouched.
const ADMISSION_SAFETY_MARGIN_BYTES: u64 = 512 * 1024 * 1024;

/// MoE block indices in offload-priority order: the MTP drafter first, then
/// the Q2_0 trunk layers earliest-first, then the IQ4_NL trunk layers
/// earliest-first. Block 52 is not part of the 52-block trunk
/// (`decoder_graph::BLOCK_TYPES`) but its tensors are in the pack and bound
/// at load time.
const MOE_BLOCK_OFFLOAD_ORDER: [usize; 24] = [
    52, 1, 3, 6, 8, 10, 13, 15, 17, 20, 22, 24, 27, 29, 31, 34, 36, 38, 40, 43, 45, 47, 49, 51,
];

/// Live `(total, free)` VRAM from the ggml backend the runner binds, `None`
/// when the class is not a discrete GPU device or the device reports no total.
/// Both come from the same `ggml_backend_dev_memory` snapshot, so they are
/// internally consistent; `free` is what the broker's observed-capacity check
/// samples when it admits the Hybrid lane's scheduler buffers.
fn ggml_device_vram_bytes(backend: GgmlCpuGraphBackend) -> Option<(u64, u64)> {
    let device = ggml_available_devices()
        .into_iter()
        .find(|device| match backend {
            GgmlCpuGraphBackend::Cpu => false,
            GgmlCpuGraphBackend::Metal => {
                ExecutionProvider::from_backend_name(&device.name) == ExecutionProvider::Metal
            }
            GgmlCpuGraphBackend::Gpu => {
                device.kind == GgmlBackendKind::Gpu
                    && !matches!(
                        ExecutionProvider::from_backend_name(&device.name),
                        ExecutionProvider::Metal | ExecutionProvider::Cpu
                    )
            }
        })?;
    let memory = device.memory.filter(|memory| memory.total_bytes > 0)?;
    Some((memory.total_bytes as u64, memory.free_bytes as u64))
}

/// Live `(total, free)` bytes of the memory domain the GPU actually allocates
/// from. On a discrete GPU that is the ggml VRAM snapshot. On a unified-memory
/// APU ggml reports an `IntegratedGpu` with no discrete device, so the GTT sysfs
/// counters are used instead: GTT (not the 512 MiB VRAM carveout, and not total
/// system RAM) is the pool graph buffers are charged against. Feeding both
/// device classes through this one function lets `resolve_discrete_residency`
/// size residency and slice identically on either architecture.
pub(crate) fn device_vram_bytes(backend: GgmlCpuGraphBackend) -> Option<(u64, u64)> {
    ggml_device_vram_bytes(backend).or_else(|| {
        if backend == GgmlCpuGraphBackend::Cpu {
            return None;
        }
        crate::metrics::gtt_memory_bytes().map(|(total, used)| (total, total.saturating_sub(used)))
    })
}

/// The binding device-memory ceiling for offload sizing. The broker enforces
/// two independent checks and admits only when both pass:
///
/// 1. policy: engine-owned bytes <= 95% of total VRAM;
/// 2. observed: newly requested bytes <= live free VRAM - headroom.
///
/// The Hybrid lane's scheduler-private graph buffers make (2) bind first on a
/// discrete card whose desktop/driver baseline is not counted in (1), so the
/// smaller of the two is the ceiling weights + context must fit under.
pub(crate) fn binding_ceiling(total_vram_bytes: u64, free_vram_bytes: u64) -> Option<u64> {
    let policy = u64::try_from(
        u128::from(total_vram_bytes) * u128::from(OWNED_FRACTION_BASIS_POINTS) / 10_000,
    )
    .ok()?;
    // A zero free reading means the backend did not report it; fall back to
    // the policy ceiling instead of offloading the entire pack on that basis.
    let observed = (free_vram_bytes > 0)
        .then(|| free_vram_bytes.saturating_sub(BROKER_MINIMUM_HEADROOM_BYTES));
    Some(observed.map_or(policy, |observed| policy.min(observed)))
}

/// Tensors the family packs but never binds into a ggml weight context.
///
/// `token_embd.weight` is consumed by `MappedTokenEmbeddingTable`, which gathers
/// rows straight from the mapped GGUF and uploads the resulting vectors as a
/// graph input; the graph never looks the weight up. `blk.52.*` is the
/// MTP/NextN speculative drafter, outside the 52-block trunk. Its expert
/// tensors stay in the offload candidate list because offloading them is free,
/// but its dense tensors are never executed.
pub(crate) fn skipped_weight_names(tensor_index: &GgufTensorIndex) -> Vec<String> {
    tensor_index
        .tensors()
        .iter()
        .filter(|tensor| {
            tensor.name == "token_embd.weight"
                || (tensor.name.starts_with("blk.52.")
                    && !tensor.name.ends_with("ffn_up_exps.weight")
                    && !tensor.name.ends_with("ffn_down_exps.weight"))
        })
        .map(|tensor| tensor.name.clone())
        .collect()
}

fn skipped_weight_bytes(tensor_index: &GgufTensorIndex) -> u64 {
    tensor_index
        .tensors()
        .iter()
        .filter(|tensor| {
            tensor.name == "token_embd.weight"
                || (tensor.name.starts_with("blk.52.")
                    && !tensor.name.ends_with("ffn_up_exps.weight")
                    && !tensor.name.ends_with("ffn_down_exps.weight"))
        })
        .fold(0u64, |sum, tensor| sum.saturating_add(tensor.size_bytes))
}

/// Payload bytes the runtime actually has to find room for: every pack tensor
/// except the family's dead weights. Used by the slice policy to reserve
/// device/host memory before deriving the encoder graph budget.
pub(crate) fn pack_weight_bytes(tensor_index: &GgufTensorIndex) -> u64 {
    let total = tensor_index
        .tensors()
        .iter()
        .fold(0u64, |sum, tensor| sum.saturating_add(tensor.size_bytes));
    total.saturating_sub(skipped_weight_bytes(tensor_index))
}

/// Picks whole MoE layers, in `MOE_BLOCK_OFFLOAD_ORDER`, until their expert
/// bytes cover `budget_bytes`. Pure over the per-layer byte lookup so the
/// tiering logic is testable without a pack; a layer whose tensors are absent
/// from the pack is skipped, never fatal.
fn plan_offload_layers(
    budget_bytes: u64,
    layer_bytes: &dyn Fn(usize) -> Option<u64>,
) -> Vec<usize> {
    let mut chosen = Vec::new();
    if budget_bytes == 0 {
        return chosen;
    }
    let mut freed = 0u64;
    for &block in &MOE_BLOCK_OFFLOAD_ORDER {
        let Some(bytes) = layer_bytes(block) else {
            continue;
        };
        chosen.push(block);
        freed = freed.saturating_add(bytes);
        if freed >= budget_bytes {
            break;
        }
    }
    chosen
}

/// Expert tensors of one MoE block, in bytes, or `None` when the pack does not
/// carry that block.
fn layer_bytes_of(tensor_index: &GgufTensorIndex, block: usize) -> Option<u64> {
    let up = tensor_index.get(&format!("blk.{block}.ffn_up_exps.weight"))?;
    let down = tensor_index.get(&format!("blk.{block}.ffn_down_exps.weight"))?;
    Some(up.size_bytes.saturating_add(down.size_bytes))
}

/// Build the offload spec for a byte budget. The loader's in-order walk binds
/// exactly the layers `plan_offload_layers` selects for that budget.
fn spec_for_budget(budget: u64, tensor_index: &GgufTensorIndex) -> Option<HostWeightOffloadSpec> {
    if budget == 0 {
        return None;
    }
    let layer_bytes = |block: usize| layer_bytes_of(tensor_index, block);
    let blocks = plan_offload_layers(budget, &layer_bytes);
    if blocks.is_empty() {
        return None;
    }
    let mut candidates = Vec::with_capacity(blocks.len() * 2);
    for block in &blocks {
        candidates.push(format!("blk.{block}.ffn_up_exps.weight"));
        candidates.push(format!("blk.{block}.ffn_down_exps.weight"));
    }
    let budget_bytes: u64 = blocks.iter().filter_map(|block| layer_bytes(*block)).sum();
    Some(HostWeightOffloadSpec::new(budget_bytes, candidates))
}

/// Device footprint of the encoder graph for a slice of `seconds`, using the
/// same budgeting rate the slice policy uses (measured 155 MiB/s plus a 24%
/// margin). The decoder graph is prefix-chunked and the topology spans the
/// derived slice, so the encoder is the component that actually scales with
/// slice length; the context/state arena is accounted separately below.
fn graph_footprint_bytes(seconds: u64) -> u64 {
    super::slice::ENCODER_SCRATCH_BYTES_PER_SECOND.saturating_mul(seconds)
}

/// Resolve the offload byte budget directly from a target slice length. The
/// two decisions are deliberately decoupled: the slice is an input chosen for
/// quality/throughput, not a fraction of whatever headroom the weights happen
/// to leave. Previously the slice was derived from post-weight headroom and
/// then fed back into the graph term of this budget, amplifying any margin
/// change by `1/(1-0.4) = 1.67x` and driving the slice to its floor. Here the
/// graph cost is fixed by the target, so the budget is a single linear solve.
fn resolve_offload_budget(
    total_vram_bytes: u64,
    free_vram_bytes: u64,
    pack_weight_bytes: u64,
    margin_bytes: u64,
    target_slice_seconds: u64,
) -> Option<u64> {
    let ceiling = binding_ceiling(total_vram_bytes, free_vram_bytes)?;
    let required = pack_weight_bytes
        .saturating_add(CONTEXT_AND_STATE_FOOTPRINT_BYTES)
        .saturating_add(graph_footprint_bytes(target_slice_seconds))
        .saturating_add(margin_bytes);
    Some(required.saturating_sub(ceiling))
}

/// Slice length the post-residency headroom can fund: half of it, budgeted at
/// the encoder scratch rate. The other half stays reserved for context/state,
/// the admission margin and driver transients. This is the ceiling that binds
/// on small cards and the grown slice on large cards or a big APU.
fn headroom_slice_cap(ceiling: u64, resident: u64) -> u64 {
    const DISCRETE_CAP_BASIS_POINTS: u128 = 5_000; // 50%
    let headroom = ceiling.saturating_sub(resident);
    u64::try_from(
        u128::from(headroom) * DISCRETE_CAP_BASIS_POINTS
            / 10_000
            / u128::from(super::slice::ENCODER_SCRATCH_BYTES_PER_SECOND),
    )
    .unwrap_or(u64::MAX)
    .clamp(
        super::slice::MIN_SLICE_SECONDS,
        super::slice::MAX_SLICE_SECONDS,
    )
}

/// Pick the final slice from the headroom-derived cap and an optional explicit
/// target. With no override the cap is the slice: on small cards it is exactly
/// the value that already bound (13/15/14 s), while larger cards grow instead
/// of being pinned at the old fixed 20 s. An explicit override is still capped
/// so a profiling target can never exceed the safety ceiling.
fn resolve_slice_seconds(explicit_target: Option<u64>, cap: u64) -> u64 {
    explicit_target.map_or(cap, |target| target.min(cap))
}

/// The resolved discrete plan: offload budget and the slice length that fits
/// alongside it. Both the longform slice policy and the weight loader call
/// this once so they cannot disagree about residency vs. graph size.
pub(crate) struct DiscreteResidencyPlan {
    pub offload_budget_bytes: u64,
    pub slice_seconds: u64,
}

/// Resolve the discrete plan in one pass:
/// 1. price the graph at the explicit target (or the 20 s default when deriving
///    it) and see which layers that would offload (provisional residency);
/// 2. cap the slice to 50% of the headroom that residency actually leaves;
/// 3. re-price the graph at the chosen slice for the final budget.
///
/// With no explicit target the cap itself is the slice: on 10/12/16 GiB cards
/// it is already the binding value (13/15/14 s), while larger cards grow with
/// headroom instead of being pinned at 20 s. The cap is a safety ceiling, not
/// a feedback term: the final budget uses the chosen slice, so a smaller card
/// loads more experts instead of requesting an over-sized graph. No fixed-point
/// loop and no 1.67x margin amplification.
pub(crate) fn resolve_discrete_residency(
    backend: GgmlCpuGraphBackend,
    tensor_index: &GgufTensorIndex,
) -> Option<DiscreteResidencyPlan> {
    let (total, free) = device_vram_bytes(backend)?;
    let ceiling = binding_ceiling(total, free)?;
    let pack = pack_weight_bytes(tensor_index);
    let explicit_target = super::slice::discrete_target_slice_override();
    // The provisional budget is priced at the explicit target, or at the old
    // 20 s default when deriving the slice. It only decides the starting
    // residency; the final budget is re-priced at the chosen slice below.
    let pricing_slice = explicit_target.unwrap_or(super::slice::PROVISIONAL_PRICING_SLICE_SECONDS);

    let provisional = resolve_offload_budget(
        total,
        free,
        pack,
        ADMISSION_SAFETY_MARGIN_BYTES,
        pricing_slice,
    )?;
    let chosen = plan_offload_layers(provisional, &|block| layer_bytes_of(tensor_index, block));
    let offloaded: u64 = chosen
        .iter()
        .filter_map(|block| layer_bytes_of(tensor_index, *block))
        .sum();
    let resident = pack.saturating_sub(offloaded);
    let cap = headroom_slice_cap(ceiling, resident);
    let slice_seconds = resolve_slice_seconds(explicit_target, cap);
    let budget = resolve_offload_budget(
        total,
        free,
        pack,
        ADMISSION_SAFETY_MARGIN_BYTES,
        slice_seconds,
    )?;
    Some(DiscreteResidencyPlan {
        offload_budget_bytes: budget,
        slice_seconds,
    })
}

/// Computes the host offload spec for one prepared runtime. On a discrete GPU
/// the plan is resolved once and shared with the longform slice policy.
pub(crate) fn compute_host_offload_spec(
    backend: GgmlCpuGraphBackend,
    tensor_index: &GgufTensorIndex,
) -> Option<HostWeightOffloadSpec> {
    let plan = resolve_discrete_residency(backend, tensor_index)?;
    spec_for_budget(plan.offload_budget_bytes, tensor_index)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pack expert-tensor layout from `nemotron-offloading.md` 3.1: Q2_0
    /// trunk layers 359,202,816 B each, IQ4_NL layers (and the MTP drafter)
    /// 718,405,632 B each.
    const Q2_0_LAYER_BYTES: u64 = 359_202_816;
    const IQ4_NL_LAYER_BYTES: u64 = 718_405_632;
    const GIB: u64 = 1024 * 1024 * 1024;
    const Q2_PACK: u64 = 15_242_619_392;
    const IQ4_PACK: u64 = 21_349_067_264;
    const Q8_PACK: u64 = 36_675_053_792;

    fn recon_layer_bytes(block: usize) -> Option<u64> {
        const Q2_0_BLOCKS: [usize; 17] = [
            1, 3, 6, 8, 10, 13, 15, 17, 20, 22, 24, 27, 29, 31, 34, 36, 38,
        ];
        const IQ4_NL_BLOCKS: [usize; 7] = [52, 40, 43, 45, 47, 49, 51];
        if Q2_0_BLOCKS.contains(&block) {
            Some(Q2_0_LAYER_BYTES)
        } else if IQ4_NL_BLOCKS.contains(&block) {
            Some(IQ4_NL_LAYER_BYTES)
        } else {
            None
        }
    }

    #[test]
    fn dead_weights_are_excluded_from_the_planner_footprint() {
        let mut index =
            crate::GgufTensorIndex::empty_for_test(std::path::PathBuf::from("probe.gguf"));
        let tensor = |name: &str, bytes: u64| crate::GgufTensorMetadata {
            name: name.to_owned(),
            dims: vec![bytes, 1],
            ggml_type: 0,
            type_name: "f32".to_owned(),
            size_bytes: bytes,
            offset_bytes: 0,
        };
        index.push_for_test(tensor("loaded.weight", 100));
        index.push_for_test(tensor("token_embd.weight", 374));
        index.push_for_test(tensor("blk.52.attn_q.weight", 52));
        index.push_for_test(tensor("blk.52.ffn_up_exps.weight", 718));
        assert_eq!(
            skipped_weight_names(&index),
            vec![
                "token_embd.weight".to_owned(),
                "blk.52.attn_q.weight".to_owned()
            ]
        );
        // Expert tensors stay in the offload budget; dead weights do not.
        assert_eq!(pack_weight_bytes(&index), 818);
    }

    #[test]
    fn binding_ceiling_uses_observed_free_when_desktop_holds_vram() {
        // 16 GB card with ~0.72 GiB held by the desktop/driver: the observed
        // ceiling (free - headroom) binds before the 95%-of-total policy one.
        let total = 17_163_091_968u64;
        let free = 15_389_671_424u64; // total - 1,773,420,544 (~1.65 GiB)
        let policy = total * 9_500 / 10_000; // 16,304,937,369
        let observed = free - 256 * 1024 * 1024; // 15,121,236,992
        assert_eq!(binding_ceiling(total, free), Some(observed));
        assert!(observed < policy);
    }

    #[test]
    fn binding_ceiling_falls_back_to_policy_when_free_is_unreported() {
        let total = 17_163_091_968u64;
        let policy = total * 9_500 / 10_000;
        assert_eq!(binding_ceiling(total, 0), Some(policy));
    }

    #[test]
    fn binding_ceiling_uses_policy_on_unified_memory() {
        // On a 112 GiB APU nearly all memory reports free, so the policy
        // ceiling binds and no offload is needed (need = 0).
        let total = 112 * 1024u64 * 1024 * 1024;
        assert_eq!(binding_ceiling(total, total), Some(total * 9_500 / 10_000));
    }

    #[test]
    fn zero_budget_offloads_nothing() {
        assert!(plan_offload_layers(0, &recon_layer_bytes).is_empty());
    }

    #[test]
    fn auto_slice_uses_the_headroom_cap() {
        // No override: the cap is the slice. On a 16 GiB card it is 14 s; a
        // 32 GiB card grows to 43 s instead of being pinned at 20 s.
        assert_eq!(resolve_slice_seconds(None, 14), 14);
        assert_eq!(resolve_slice_seconds(None, 43), 43);
    }

    #[test]
    fn explicit_target_is_still_capped_for_safety() {
        assert_eq!(resolve_slice_seconds(Some(20), 14), 14);
        assert_eq!(resolve_slice_seconds(Some(10), 14), 10);
        assert_eq!(resolve_slice_seconds(Some(60), 43), 43);
    }

    /// gfx1151 has 112 GiB of GTT. Routing it through the shared resolver with
    /// no override gives 245/230/192 s caps for q2/iq4/q8 (whole pack resident,
    /// no expert offload), replacing the old 40%-policy 208/196/166 s slices.
    #[test]
    fn unified_apu_cap_grows_with_gtt_headroom() {
        let gtt_total = 112 * GIB;
        let ceiling = binding_ceiling(gtt_total, gtt_total).expect("ceiling");
        for (pack, expected) in [(Q2_PACK, 245), (IQ4_PACK, 230), (Q8_PACK, 192)] {
            assert_eq!(headroom_slice_cap(ceiling, pack), expected, "pack {pack}");
        }
    }

    /// At those longer APU slices the whole pack still fits under the 95%-of-GTT
    /// ceiling, so the shared resolver offloads nothing on gfx1151.
    #[test]
    fn unified_apu_keeps_the_whole_pack_resident() {
        let gtt_total = 112 * GIB;
        for (pack, slice) in [(Q2_PACK, 245), (IQ4_PACK, 230), (Q8_PACK, 192)] {
            let budget = resolve_offload_budget(
                gtt_total,
                gtt_total,
                pack,
                ADMISSION_SAFETY_MARGIN_BYTES,
                slice,
            )
            .expect("budget");
            assert_eq!(budget, 0, "pack {pack} at {slice} s");
        }
    }

    #[test]
    fn recon_deficit_is_covered_by_the_mtp_block_alone() {
        // nemotron-offloading.md 2.3: 715,875,943 B deficit.
        let chosen = plan_offload_layers(715_875_943, &recon_layer_bytes);
        assert_eq!(chosen, vec![52]);
    }

    #[test]
    fn next_tier_is_the_earliest_q2_0_layers() {
        // A ~1 GiB budget covers blk.52 plus the first two Q2_0 trunk layers.
        let chosen = plan_offload_layers(1_085_165_671, &recon_layer_bytes);
        assert_eq!(chosen, vec![52, 1, 3]);
    }

    /// Budget is a direct function of the target slice: a longer target leaves
    /// less room for resident experts, with no feedback loop.
    #[test]
    fn decoupled_budget_shrinks_as_target_slice_grows() {
        let total = 17_163_091_968u64;
        let free = total - 773_000_000; // ~0.72 GiB desktop baseline
        let pack = 15_233_819_708u64;
        let short = resolve_offload_budget(total, free, pack, ADMISSION_SAFETY_MARGIN_BYTES, 10)
            .expect("budget");
        let long = resolve_offload_budget(total, free, pack, ADMISSION_SAFETY_MARGIN_BYTES, 20)
            .expect("budget");
        assert!(
            long > short,
            "a 20 s target must offload more than a 10 s target: {long} vs {short}"
        );
        assert_eq!(long - short, graph_footprint_bytes(10));
    }

    /// The decoupled budget must restore some experts on a 16 GB card while
    /// keeping the target slice's graph, the context arena and the admission
    /// margin under the binding ceiling.
    #[test]
    fn decoupled_budget_restores_experts_without_exceeding_the_ceiling() {
        let total = 17_163_091_968u64;
        let free = total - 773_000_000; // ~0.72 GiB desktop baseline
        let pack = 15_233_819_708u64;
        let seconds = 10u64;
        let ceiling = binding_ceiling(total, free).expect("ceiling");
        let budget =
            resolve_offload_budget(total, free, pack, ADMISSION_SAFETY_MARGIN_BYTES, seconds)
                .expect("budget");
        let chosen = plan_offload_layers(budget, &recon_layer_bytes);
        assert!(!chosen.is_empty(), "some experts must remain offloaded");
        assert!(
            chosen.len() < MOE_BLOCK_OFFLOAD_ORDER.len(),
            "the stale fixed footprint must not force all experts off"
        );
        assert_eq!(chosen[0], 52, "MTP block stays first in priority order");

        let offloaded: u64 = chosen.iter().filter_map(|b| recon_layer_bytes(*b)).sum();
        let resident = pack.saturating_sub(offloaded);
        let required = resident
            .saturating_add(CONTEXT_AND_STATE_FOOTPRINT_BYTES)
            .saturating_add(graph_footprint_bytes(seconds))
            .saturating_add(ADMISSION_SAFETY_MARGIN_BYTES);
        assert!(
            required <= ceiling,
            "resident {resident} + graph {} + context + margin must fit under {ceiling}",
            graph_footprint_bytes(seconds)
        );
    }

    #[test]
    fn iq4_nl_layers_only_after_every_q2_0_layer() {
        // blk.52 + all 17 Q2_0 layers = 6,824,853,504 B; one byte more takes
        // the first IQ4_NL trunk layer (40), not any earlier layer.
        let chosen = plan_offload_layers(6_824_853_505, &recon_layer_bytes);
        assert_eq!(
            chosen,
            vec![
                52, 1, 3, 6, 8, 10, 13, 15, 17, 20, 22, 24, 27, 29, 31, 34, 36, 38, 40
            ]
        );
    }

    #[test]
    fn impossible_budget_offloads_every_expert_layer() {
        let chosen = plan_offload_layers(u64::MAX, &recon_layer_bytes);
        assert_eq!(chosen.len(), MOE_BLOCK_OFFLOAD_ORDER.len());
        assert_eq!(chosen[0], 52);
        assert_eq!(*chosen.last().expect("non-empty"), 51);
    }

    #[test]
    fn absent_blocks_are_skipped_not_fatal() {
        // Only the MTP drafter present: any positive budget takes it and
        // stops, even if the budget is far larger.
        let chosen = plan_offload_layers(u64::MAX, &|block| {
            (block == 52).then_some(IQ4_NL_LAYER_BYTES)
        });
        assert_eq!(chosen, vec![52]);
    }
}
