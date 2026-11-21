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

/// Fraction of the binding device-memory ceiling reserved for the encoder
/// graph. The rest is left for device-resident weights, the decoder's chunked
/// graph, the state arenas, and the desktop/driver baseline.
const ENCODER_GRAPH_MEMORY_FRACTION_BASIS_POINTS: u64 = 5_000; // 50%

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

/// Derive the largest safe longform slice (seconds) from an available-memory
/// budget. Pure over the budget so the arithmetic is unit-testable.
fn slice_seconds_from_budget(available_bytes: u64) -> u64 {
    let encoder_budget = u64::try_from(
        u128::from(available_bytes) * u128::from(ENCODER_GRAPH_MEMORY_FRACTION_BASIS_POINTS)
            / 10_000,
    )
    .unwrap_or(u64::MAX);
    (encoder_budget / ENCODER_SCRATCH_BYTES_PER_SECOND).clamp(MIN_SLICE_SECONDS, MAX_SLICE_SECONDS)
}

/// GPU wrapper: apply the broker's binding device-memory ceiling, then the
/// pure derivation.
fn slice_seconds_from_device_budget(total_vram_bytes: u64, free_vram_bytes: u64) -> Option<u64> {
    Some(slice_seconds_from_budget(binding_ceiling(
        total_vram_bytes,
        free_vram_bytes,
    )?))
}

/// Derive the runtime slice length for the given backend.
///
/// Every backend is sized from detected memory:
/// - discrete GPU / APU: the broker's binding device-memory ceiling;
/// - CPU: available host memory (the encoder graph lives in host RAM).
///
/// More memory yields longer slices, less memory yields shorter ones, and the
/// result is always clamped to the model's hard context ceiling. No fixed
/// seconds value is consulted at runtime.
pub(crate) fn runtime_slice_seconds(backend: GgmlCpuGraphBackend) -> Option<u64> {
    match backend {
        GgmlCpuGraphBackend::Cpu => {
            crate::host::host_available_memory_bytes().map(slice_seconds_from_budget)
        }
        _ => {
            let (total, free) = device_vram_bytes(backend)?;
            slice_seconds_from_device_budget(total, free)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A realistic free reading is total minus the ~0.72 GiB desktop baseline.
    fn free_of(total: u64) -> u64 {
        total.saturating_sub(773_000_000)
    }

    #[test]
    fn small_card_gets_a_short_slice() {
        // ~10 GB: observed ceiling ~9.0 GB, half ~4.5 GB / 192 MiB/s ~= 24 s.
        let total = 10 * 1024 * 1024 * 1024;
        let seconds = slice_seconds_from_device_budget(total, free_of(total)).unwrap();
        assert!((15..=40).contains(&seconds), "got {seconds}");
    }

    #[test]
    fn sixteen_gb_card_gets_a_medium_slice() {
        // 16 GB: observed ceiling ~16.1 GB, half ~8.0 GB / 192 MiB/s ~= 43 s.
        let total = 17_163_091_968u64;
        let seconds = slice_seconds_from_device_budget(total, free_of(total)).unwrap();
        assert!((30..=60).contains(&seconds), "got {seconds}");
    }

    #[test]
    fn large_card_gets_a_long_but_capped_slice() {
        // 112 GB APU: observed ceiling ~106 GiB, half ~53 GiB / 192 MiB/s
        // ~= 283 s -- far longer than any small card, still under the 3600 s
        // model hard ceiling (which needs >1 TB of budget to reach).
        let total = 112 * 1024u64 * 1024 * 1024;
        let seconds = slice_seconds_from_device_budget(total, free_of(total)).unwrap();
        assert!((200..=400).contains(&seconds), "got {seconds}");
        assert!(seconds < MAX_SLICE_SECONDS);
    }

    #[test]
    fn derived_slice_is_always_within_bounds() {
        for gb in [8u64, 10, 12, 16, 24, 48, 112] {
            let total = gb * 1024 * 1024 * 1024;
            let seconds = slice_seconds_from_device_budget(total, free_of(total)).unwrap();
            assert!((MIN_SLICE_SECONDS..=MAX_SLICE_SECONDS).contains(&seconds));
        }
    }

    #[test]
    fn hard_ceiling_caps_an_absurd_budget() {
        // Even unbounded memory cannot exceed the model's 3600 s invocation.
        let seconds = slice_seconds_from_budget(u64::MAX / 2);
        assert_eq!(seconds, MAX_SLICE_SECONDS);
    }
}
