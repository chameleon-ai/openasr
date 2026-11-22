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
const CONTEXT_AND_STATE_FOOTPRINT_BYTES: u64 = 2 * 1024 * 1024 * 1024;

/// Extra live-free VRAM kept beyond the estimated graph/state footprint. The
/// scheduler admits the context buffer and graph scratch against the *live*
/// free heap, which also pays driver workspaces and transient allocation
/// spikes; the observed 16 GB card already produced `CapacityUnavailable` on
/// `context-buffer` while device-wide peak was only ~9 GiB. This margin is the
/// explicit allowance for that gap, so restoring experts never consumes the
/// headroom graph admission actually needs.
const ADMISSION_SAFETY_MARGIN_BYTES: u64 = 2 * 1024 * 1024 * 1024;

/// Starting point for the fixed-point residency search. The old sizing (2 GiB
/// context/state + 10 GiB graph scratch) is deliberately conservative; it is
/// used only as the initial offload budget and is then relaxed until the
/// slice-dependent footprint estimate agrees with the residency.
const INITIAL_GRAPH_SCRATCH_BYTES: u64 = 10 * 1024 * 1024 * 1024;

/// How many times to iterate residency -> slice length -> graph footprint.
/// The map is monotonically contracting on this domain; four passes are plenty
/// and keep the recomputation (two calls per request) cheap.
const RESIDENCY_FIXED_POINT_PASSES: usize = 4;

/// MoE block indices in offload-priority order: the MTP drafter first, then
/// the Q2_0 trunk layers earliest-first, then the IQ4_NL trunk layers
/// earliest-first. Block 52 is not part of the 52-block trunk
/// (`decoder_graph::BLOCK_TYPES`) but its tensors are in the pack and bound
/// at load time.
const MOE_BLOCK_OFFLOAD_ORDER: [usize; 24] = [
    52, 1, 3, 6, 8, 10, 13, 15, 17, 20, 22, 24, 27, 29, 31, 34, 36, 38, 40, 43, 45, 47, 49, 51,
];

/// Live `(total, free)` VRAM of the device the runner's backend class binds,
/// `None` when the class is not a discrete GPU device or the device reports no
/// total. Both come from the same `ggml_backend_dev_memory` snapshot, so they
/// are internally consistent; `free` is what the broker's observed-capacity
/// check samples when it admits the Hybrid lane's scheduler buffers.
pub(crate) fn device_vram_bytes(backend: GgmlCpuGraphBackend) -> Option<(u64, u64)> {
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

/// Total payload bytes of every tensor in the pack. Used by the slice policy
/// to reserve device/host memory for the resident weights before deriving the
/// encoder graph budget.
pub(crate) fn pack_weight_bytes(tensor_index: &GgufTensorIndex) -> u64 {
    tensor_index
        .tensors()
        .iter()
        .fold(0u64, |sum, tensor| sum.saturating_add(tensor.size_bytes))
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

/// Resolve the offload byte budget by iterating residency -> derived slice ->
/// graph footprint until the two agree. Pure over the pack's layer sizes so
/// the policy is testable without a real pack.
fn resolve_offload_budget(
    total_vram_bytes: u64,
    free_vram_bytes: u64,
    pack_weight_bytes: u64,
    layer_bytes: &dyn Fn(usize) -> Option<u64>,
) -> Option<u64> {
    let ceiling = binding_ceiling(total_vram_bytes, free_vram_bytes)?;
    // Start from the conservative old sizing so every later pass is a
    // relaxation, never an optimistic first guess.
    let initial_required = pack_weight_bytes
        .saturating_add(CONTEXT_AND_STATE_FOOTPRINT_BYTES)
        .saturating_add(INITIAL_GRAPH_SCRATCH_BYTES);
    let mut budget = initial_required.saturating_sub(ceiling);

    for _ in 0..RESIDENCY_FIXED_POINT_PASSES {
        let chosen = plan_offload_layers(budget, layer_bytes);
        let offloaded: u64 = chosen.iter().filter_map(|block| layer_bytes(*block)).sum();
        let resident = pack_weight_bytes.saturating_sub(offloaded);
        let seconds = super::slice::slice_seconds_from_device_budget(
            total_vram_bytes,
            free_vram_bytes,
            resident,
        )?;
        let required = pack_weight_bytes
            .saturating_add(CONTEXT_AND_STATE_FOOTPRINT_BYTES)
            .saturating_add(graph_footprint_bytes(seconds))
            .saturating_add(ADMISSION_SAFETY_MARGIN_BYTES);
        let next = required.saturating_sub(ceiling);
        if next == budget {
            break;
        }
        budget = next;
    }
    Some(budget)
}

/// Computes the host offload spec for one prepared runtime. The offload budget
/// and the derived slice length are mutually dependent (more resident experts
/// shrink the slice, which shrinks the graph that must fit alongside them), so
/// resolve them with a short fixed-point iteration. The result is stable for
/// both callers -- the slice policy and the executor -- which each call this
/// pure function with the same pack index and backend.
pub(crate) fn compute_host_offload_spec(
    backend: GgmlCpuGraphBackend,
    tensor_index: &GgufTensorIndex,
) -> Option<HostWeightOffloadSpec> {
    let (total_vram_bytes, free_vram_bytes) = device_vram_bytes(backend)?;
    let pack_weight_bytes = pack_weight_bytes(tensor_index);
    let budget = resolve_offload_budget(
        total_vram_bytes,
        free_vram_bytes,
        pack_weight_bytes,
        &|block| layer_bytes_of(tensor_index, block),
    )?;
    spec_for_budget(budget, tensor_index)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pack expert-tensor layout from `nemotron-offloading.md` 3.1: Q2_0
    /// trunk layers 359,202,816 B each, IQ4_NL layers (and the MTP drafter)
    /// 718,405,632 B each.
    const Q2_0_LAYER_BYTES: u64 = 359_202_816;
    const IQ4_NL_LAYER_BYTES: u64 = 718_405_632;

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

    #[test]
    fn recon_16gb_card_offloads_every_expert_layer_with_the_old_footprint() {
        // The initial (conservative) footprint still offloads everything; the
        // slice-aware iteration below is what relaxes it. Keeping this test
        // pins the starting point of the fixed-point search.
        let total = 17_163_091_968u64;
        let ceiling = binding_ceiling(total, 0).expect("ceiling");
        let required =
            15_233_819_708u64 + CONTEXT_AND_STATE_FOOTPRINT_BYTES + INITIAL_GRAPH_SCRATCH_BYTES;
        let budget = required.saturating_sub(ceiling);
        let chosen = plan_offload_layers(budget, &recon_layer_bytes);
        assert_eq!(chosen.len(), MOE_BLOCK_OFFLOAD_ORDER.len());
        assert_eq!(chosen[0], 52);
    }

    /// The slice-aware budget must restore some experts on a 16 GB card while
    /// keeping the derived slice's graph, the context arena and the admission
    /// margin under the binding ceiling.
    #[test]
    fn slice_aware_budget_restores_experts_without_exceeding_the_ceiling() {
        let total = 17_163_091_968u64;
        let free = total - 773_000_000; // ~0.72 GiB desktop baseline
        let pack = 15_233_819_708u64;
        let ceiling = binding_ceiling(total, free).expect("ceiling");
        let budget = resolve_offload_budget(total, free, pack, &recon_layer_bytes).expect("budget");
        let chosen = plan_offload_layers(budget, &recon_layer_bytes);
        assert!(!chosen.is_empty(), "some experts must remain offloaded");
        assert!(
            chosen.len() < MOE_BLOCK_OFFLOAD_ORDER.len(),
            "the stale fixed footprint must not force all experts off"
        );
        assert_eq!(chosen[0], 52, "MTP block stays first in priority order");

        let offloaded: u64 = chosen.iter().filter_map(|b| recon_layer_bytes(*b)).sum();
        let resident = pack.saturating_sub(offloaded);
        let seconds = crate::models::nemotron_transcribe::slice::slice_seconds_from_device_budget(
            total, free, resident,
        )
        .expect("slice");
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
