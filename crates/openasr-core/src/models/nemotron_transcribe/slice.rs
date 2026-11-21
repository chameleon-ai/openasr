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

/// GPU wrapper: apply the broker's binding device-memory ceiling, then the
/// pure derivation.
fn slice_seconds_from_device_budget(
    total_vram_bytes: u64,
    free_vram_bytes: u64,
    pack_weight_bytes: u64,
) -> Option<u64> {
    Some(slice_seconds_from_budget(
        binding_ceiling(total_vram_bytes, free_vram_bytes)?,
        pack_weight_bytes,
    ))
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
            None => crate::host::host_available_memory_bytes()
                .map(|available| slice_seconds_from_budget(available, resident_bytes)),
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
}
