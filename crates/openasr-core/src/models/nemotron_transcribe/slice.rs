//! VRAM-derived longform slice length for nemotron-transcribe.
//!
//! The PE encoder runs full (bidirectional) flash attention over every mel
//! frame of a slice, so its graph scratch grows ~linearly with slice length:
//! measured on the discrete card at 1.70 GiB for 11 s (`jfk`) and 13.6 GiB for
//! 88 s. A fixed slice length therefore either wastes a large card or cannot
//! be admitted on a small one. This module derives the largest safe slice from
//! the detected device memory instead, so the same code admits GPU inference
//! on a 10 GB, 12 GB, or 16 GB card (and a larger card gets longer slices).

use crate::ggml_runtime::GgmlCpuGraphBackend;

use super::offload::{binding_ceiling, device_vram_bytes};

/// Encoder graph scratch per second of audio, with safety margin. Measurements
/// on the discrete card (all experts offloaded to host): 1.70 GiB @ 11 s and
/// 13.6 GiB @ 88 s, i.e. ~155 MiB/s. Reserve 192 MiB/s (~24% margin) for
/// driver allocation spikes and the CTC/encoder projection buffers.
const ENCODER_SCRATCH_BYTES_PER_SECOND: u64 = 192 * 1024 * 1024;

/// Fraction of the post-weight headroom reserved for the encoder graph.
///
/// Bounded (40%) because the broker admits the graph against the *live* free
/// heap, which on a unified-memory APU is well below the policy ceiling:
/// driver workspaces, hipBLAS scratch, and resident KV/SSM arenas are not in
/// the broker ledger. Measured on gfx1151: device heap ~67 GB total; a q8
/// pack commits 36.7 GB, leaving ~30 GB live free. At 50% of headroom the
/// ~236 s slice requested a 36 GB graph and failed admission; at 30% the
/// ~135 s slice left too little KV span and decode overran it. 40% yields a
/// ~180 s q8 slice (~28 GB graph, safe under 30 GB; KV span ~3800, safe
/// beyond the degenerate-guard's extra steps) and longer slices for q2/iq4.
const ENCODER_GRAPH_MEMORY_FRACTION_BASIS_POINTS: u64 = 4_000; // 40%

/// Rate at which the ggml scheduler actually *charges* the encoder graph when
/// admitting it, i.e. the measured 155 MiB/s flash-attention graph scratch
/// (`arch/mod.rs`) without the 24% budgeting margin added to
/// `ENCODER_SCRATCH_BYTES_PER_SECOND`. Admission is the binding constraint, so
/// the second bound below uses this rate, not the budgeting rate.
const ADMISSION_GRAPH_BYTES_PER_SECOND: u64 = 155 * 1024 * 1024;

/// Memory the broker ledger does not account for on a unified-memory host:
/// driver workspaces, hipBLAS scratch, and the resident KV/SSM arenas. 1 GiB is
/// the value consistent with the known-good gfx1151 q8 setting: heap ~67 GiB,
/// pack 36.7 GiB, ~1 GiB reserve -> (67 - 36.7 - 1) GiB / 155 MiB/s = 189 s.
const NON_LEDGER_RESERVE_BYTES: u64 = 1024 * 1024 * 1024;

/// Env override for the device-heap size on unified-memory hosts where ggml
/// reports no discrete VRAM (gfx1151). The broker admits the graph against the
/// live-free device heap, which is NOT derivable from system `MemAvailable`
/// (the system-memory policy ceiling is ~114 GiB while the device heap is
/// ~67 GiB). Set this to the observed heap size to enable the admission-side
/// bound there; when unset, the policy bound alone applies as before.
const APU_DEVICE_HEAP_BYTES_ENV: &str = "OPENASR_NEMOTRON_DEVICE_HEAP_BYTES";

/// Floor so a tiny detected budget still yields a usable slice; the family's
/// 640-position KV span covers it comfortably.
const MIN_SLICE_SECONDS: u64 = 10;

/// Hard ceiling from the family's advertised invocation span (one hour,
/// `OpenAsrInvocationSpan::Bounded { max_seconds: 3600 }`), independent of any
/// device. Even a machine with unlimited memory never exceeds one hour per
/// slice; memory only ever narrows it further.
const MAX_SLICE_SECONDS: u64 = 3600;

/// The model-level hard ceiling a slice may never exceed, regardless of the
/// device. Replaces the old fixed 90 s as the descriptor's semantic bound.
pub(crate) const NEMOTRON_SLICE_HARD_CEILING_SECONDS: u32 = MAX_SLICE_SECONDS as u32;

/// Derive the largest safe longform slice (seconds) from the memory left
/// after resident weights. Pure over the budget so it is unit-testable.
///
/// The pack weights are already committed when a slice's encoder graph is
/// admitted, so they must be subtracted first: a 36.7 GB q8 pack leaves far
/// less headroom than a 15 GB q2 pack on the same host, and using the raw
/// ceiling for both makes the larger quant fall back to CPU.
fn slice_seconds_from_budget(available_bytes: u64, pack_weight_bytes: u64) -> u64 {
    let headroom = available_bytes.saturating_sub(pack_weight_bytes);
    let encoder_budget = u64::try_from(
        u128::from(headroom) * u128::from(ENCODER_GRAPH_MEMORY_FRACTION_BASIS_POINTS) / 10_000,
    )
    .unwrap_or(u64::MAX);
    (encoder_budget / ENCODER_SCRATCH_BYTES_PER_SECOND).clamp(MIN_SLICE_SECONDS, MAX_SLICE_SECONDS)
}

/// Admission-side bound: the scheduler charges the graph against the live-free
/// device heap at the measured quote rate, after an explicit reserve for memory
/// the broker does not ledger. `admission_ceiling` is that device domain; when
/// it is unknown (APU with no `APU_DEVICE_HEAP_BYTES_ENV` override) this bound
/// is absent and the policy bound alone governs, as before.
fn admission_slice_seconds(resident_bytes: u64, admission_ceiling: Option<u64>) -> Option<u64> {
    let usable = admission_ceiling?
        .saturating_sub(resident_bytes)
        .saturating_sub(NON_LEDGER_RESERVE_BYTES);
    Some((usable / ADMISSION_GRAPH_BYTES_PER_SECOND).clamp(MIN_SLICE_SECONDS, MAX_SLICE_SECONDS))
}

/// Both bounds: the policy share of post-weight headroom, and the admission
/// domain the graph is actually charged against. Taking the minimum makes the
/// calculation pack-aware in the *same* domain admission uses, so all quant
/// tiers are bounded by the same physical heap rather than by a flat fraction
/// of a different (system-memory) domain.
fn slice_seconds_from_budget_with_admission(
    available_bytes: u64,
    resident_bytes: u64,
    admission_ceiling: Option<u64>,
) -> u64 {
    slice_seconds_from_budget(available_bytes, resident_bytes)
        .min(admission_slice_seconds(resident_bytes, admission_ceiling).unwrap_or(u64::MAX))
}

/// GPU wrapper: apply the broker's binding device-memory ceiling to both the
/// policy budget and the admission bound, then take the tighter slice.
fn slice_seconds_from_device_budget(
    total_vram_bytes: u64,
    free_vram_bytes: u64,
    resident_bytes: u64,
) -> Option<u64> {
    let ceiling = binding_ceiling(total_vram_bytes, free_vram_bytes)?;
    Some(slice_seconds_from_budget_with_admission(
        ceiling,
        resident_bytes,
        Some(ceiling),
    ))
}

/// Optional device-heap size for unified-memory hosts (see
/// `APU_DEVICE_HEAP_BYTES_ENV`). Invalid or non-positive values are ignored so
/// a malformed override cannot produce a pathological slice.
fn apu_device_heap_bytes() -> Option<u64> {
    let raw = std::env::var(APU_DEVICE_HEAP_BYTES_ENV).ok()?;
    let bytes = raw.trim().parse::<u64>().ok()?;
    (bytes > 0).then_some(bytes)
}

/// Derive the runtime slice length for the given backend.
///
/// Every backend is sized from detected memory:
/// - discrete GPU: the broker's binding device-memory ceiling;
/// - APU with unified memory (device reports no discrete VRAM, e.g. gfx1151)
///   and CPU: available host memory, where the encoder graph actually lives.
///
/// More memory yields longer slices, less memory yields shorter ones, and the
/// result is always clamped to the model's hard context ceiling. No fixed
/// seconds value is consulted at runtime.
pub(crate) fn runtime_slice_seconds(
    backend: GgmlCpuGraphBackend,
    tensor_index: &crate::ggml_runtime::GgufTensorIndex,
) -> Option<u64> {
    let pack_bytes = super::offload::pack_weight_bytes(tensor_index);
    // On a discrete GPU the offload policy moves experts to host RAM, so only
    // the residual device-resident bytes compete with the encoder graph. On an
    // APU/CPU nothing is offloaded (or there is no discrete device), so the
    // whole pack is resident in system memory and must be reserved in full.
    let resident_bytes = match device_vram_bytes(backend) {
        Some(_) => super::offload::compute_host_offload_spec(backend, tensor_index)
            .map_or(pack_bytes, |spec| {
                pack_bytes.saturating_sub(spec.budget_bytes())
            }),
        None => pack_bytes,
    };
    match backend {
        GgmlCpuGraphBackend::Cpu => crate::host::host_available_memory_bytes()
            .map(|available| slice_seconds_from_budget(available, resident_bytes)),
        _ => match device_vram_bytes(backend) {
            Some((total, free)) => slice_seconds_from_device_budget(total, free, resident_bytes),
            // Unified-memory APU: ggml reports an IntegratedGpu, not a discrete
            // GPU, so device_vram_bytes returns None. The encoder graph lives
            // in system memory there, so size from host RAM instead. Without
            // this fallback the cap is absent and a long clip requests one
            // enormous graph that fails admission.
            None => crate::host::host_available_memory_bytes().map(|available| {
                slice_seconds_from_budget_with_admission(
                    available,
                    resident_bytes,
                    apu_device_heap_bytes(),
                )
            }),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A realistic free reading is total minus the ~0.72 GiB desktop baseline.
    fn free_of(total: u64) -> u64 {
        total.saturating_sub(773_000_000)
    }

    // Representative pack sizes (bytes) from the three quants.
    const Q2_PACK: u64 = 15_242_619_392;
    const IQ4_PACK: u64 = 21_349_067_264;
    const Q8_PACK: u64 = 36_675_053_792;

    // gfx1151 measurements from nemotron-offloading.md: ~123 GiB host
    // MemAvailable, ~67 GiB device heap. These reproduce the observed
    // 232/220/189 s slices through the policy bound.
    const APU_MEM_AVAILABLE: u64 = 123 * 1024u64 * 1024 * 1024 + 50 * 1024 * 1024;
    const APU_DEVICE_HEAP: u64 = 67 * 1024u64 * 1024 * 1024;

    #[test]
    fn small_card_gets_a_short_slice() {
        // ~10 GB, q2 pack fully offloaded on a discrete card -> ~4 GB resident.
        // Observed ceiling ~9.0 GB minus ~4 GB resident = ~5 GB, half / 192MiB ~= 13 s.
        let total = 10 * 1024 * 1024 * 1024;
        let seconds = slice_seconds_from_budget(
            binding_ceiling(total, free_of(total)).unwrap(),
            4 * 1024 * 1024 * 1024,
        );
        assert!((10..=40).contains(&seconds), "got {seconds}");
    }

    #[test]
    fn sixteen_gb_card_gets_a_medium_slice() {
        // 16 GB, q2 fully offloaded -> ~4 GB resident: ceiling ~16.1 - 4 = ~12.1 GB.
        let total = 17_163_091_968u64;
        let seconds = slice_seconds_from_budget(
            binding_ceiling(total, free_of(total)).unwrap(),
            4 * 1024 * 1024 * 1024,
        );
        assert!((10..=60).contains(&seconds), "got {seconds}");
    }

    #[test]
    fn large_apu_bigger_pack_gets_shorter_slice() {
        // On the 112 GB APU nothing is offloaded, so the whole pack is resident.
        // q8 (36.7 GB) must get a shorter slice than q2 (15.2 GB).
        let total = 112 * 1024u64 * 1024 * 1024;
        let ceiling = binding_ceiling(total, free_of(total)).unwrap();
        let q2 = slice_seconds_from_budget(ceiling, Q2_PACK);
        let q8 = slice_seconds_from_budget(ceiling, Q8_PACK);
        assert!(q8 < q2, "q8 slice {q8} should be shorter than q2 {q2}");
        assert!(
            q2 > q8 + 30,
            "difference should be material: q2 {q2} q8 {q8}"
        );
        assert!((MIN_SLICE_SECONDS..=MAX_SLICE_SECONDS).contains(&q2));
        assert!((MIN_SLICE_SECONDS..=MAX_SLICE_SECONDS).contains(&q8));
    }

    #[test]
    fn derived_slice_is_always_within_bounds() {
        for gb in [8u64, 10, 12, 16, 24, 48, 112] {
            let total = gb * 1024 * 1024 * 1024;
            let seconds =
                slice_seconds_from_budget(binding_ceiling(total, free_of(total)).unwrap(), Q2_PACK);
            assert!((MIN_SLICE_SECONDS..=MAX_SLICE_SECONDS).contains(&seconds));
        }
    }

    #[test]
    fn hard_ceiling_caps_an_absurd_budget() {
        // Even unbounded memory cannot exceed the model's 3600 s invocation.
        let seconds = slice_seconds_from_budget(u64::MAX / 2, 0);
        assert_eq!(seconds, MAX_SLICE_SECONDS);
    }

    #[test]
    fn iq4_pack_is_between_q2_and_q8() {
        let total = 112 * 1024u64 * 1024 * 1024;
        let ceiling = binding_ceiling(total, free_of(total)).unwrap();
        let q2 = slice_seconds_from_budget(ceiling, Q2_PACK);
        let iq4 = slice_seconds_from_budget(ceiling, IQ4_PACK);
        let q8 = slice_seconds_from_budget(ceiling, Q8_PACK);
        assert!(q2 > iq4 && iq4 > q8, "q2 {q2} iq4 {iq4} q8 {q8}");
    }

    /// With the observed device heap the admission bound does not bind, so all
    /// three quants keep their known-good slices. This calibrates the 1 GiB
    /// non-ledger reserve against the gfx1151 evidence instead of guessing.
    #[test]
    fn apu_heap_bound_reproduces_observed_quant_slices() {
        for (pack, expected) in [(Q2_PACK, 232), (IQ4_PACK, 220), (Q8_PACK, 189)] {
            let seconds = slice_seconds_from_budget_with_admission(
                APU_MEM_AVAILABLE,
                pack,
                Some(APU_DEVICE_HEAP),
            );
            assert_eq!(seconds, expected, "pack {pack}");
        }
    }

    /// A smaller device heap makes the admission bound bind; the largest pack
    /// must shrink first, which is the gfx1151/q8 OOM failure mode.
    #[test]
    fn apu_heap_bound_binds_for_the_largest_pack_first() {
        let tight_heap = 50 * 1024u64 * 1024 * 1024;
        let q2 =
            slice_seconds_from_budget_with_admission(APU_MEM_AVAILABLE, Q2_PACK, Some(tight_heap));
        let q8 =
            slice_seconds_from_budget_with_admission(APU_MEM_AVAILABLE, Q8_PACK, Some(tight_heap));
        assert!(q8 < 189, "q8 {q8} should shrink under a tight heap");
        assert!(q2 > q8, "q2 {q2} should stay longer than q8 {q8}");
    }

    /// Without a heap override (APU default, CPU) behavior is unchanged.
    #[test]
    fn no_admission_ceiling_preserves_policy_behavior() {
        for pack in [Q2_PACK, IQ4_PACK, Q8_PACK] {
            assert_eq!(
                slice_seconds_from_budget_with_admission(APU_MEM_AVAILABLE, pack, None),
                slice_seconds_from_budget(APU_MEM_AVAILABLE, pack),
            );
        }
    }

    /// The extra admission bound must not change the discrete-GPU path: its
    /// binding ceiling is already the admission domain and is much tighter.
    #[test]
    fn discrete_gpu_admission_bound_does_not_change_small_card_slices() {
        let total = 17_163_091_968u64;
        let free = free_of(total);
        let resident = 4 * 1024 * 1024 * 1024;
        let before = slice_seconds_from_budget(binding_ceiling(total, free).unwrap(), resident);
        let after = slice_seconds_from_device_budget(total, free, resident).unwrap();
        assert_eq!(before, after);
    }
}
