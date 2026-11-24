//! HIP VRAM probes for the performance harness.
//!
//! On Linux the reading comes from the driver's sysfs
//! `mem_info_vram_used` counter, which reports device-wide VRAM in bytes.
//! It is NOT per-process: on an otherwise idle GPU it approximates this
//! process's footprint, but it includes the desktop compositor and any other
//! device users. `hipMemGetInfo` from a sampler thread is deliberately not
//! used on Linux: HIP contexts are thread-local, so a worker thread without
//! the executor's context reports a meaningless near-zero occupancy even
//! while the executor holds gigabytes of VRAM.
//!
//! [`gtt_memory_bytes`] reads the companion GTT counters. On unified-memory
//! APUs the VRAM carveout is tiny (often 512 MiB) and the GPU allocates from
//! GTT instead, so GTT is the capacity admission must be sized against.
//!
//! On non-Linux HIP builds `hipMemGetInfo` is the best available fallback.
//! This module compiles everywhere; the probe returns `None` and the sampler
//! is inert unless the `hip` feature is enabled, so harness code can call it
//! unconditionally.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

#[cfg(all(feature = "hip", not(target_os = "linux")))]
#[link(name = "amdhip64")]
unsafe extern "C" {
    fn hipMemGetInfo(free: *mut usize, total: *mut usize) -> i32;
}

/// Device-wide VRAM in use right now, in bytes.
///
/// `None` on a non-HIP build or when no supported counter is available.
/// The reading includes other processes' allocations and runtime caching;
/// treat it as a host-level observation, not an exact accounting of this
/// process's live tensors.
#[cfg(all(feature = "hip", target_os = "linux"))]
pub fn hip_vram_used_bytes() -> Option<u64> {
    use std::path::PathBuf;
    use std::sync::OnceLock;

    // Discover the sysfs counter paths once; card numbering is host-specific.
    static COUNTERS: OnceLock<Vec<PathBuf>> = OnceLock::new();
    let counters = COUNTERS.get_or_init(|| {
        let Ok(entries) = std::fs::read_dir("/sys/class/drm") else {
            return Vec::new();
        };
        entries
            .filter_map(Result::ok)
            .filter_map(|entry| {
                let name = entry.file_name();
                let name = name.to_string_lossy();
                let digits = name.strip_prefix("card")?;
                if digits.is_empty() || !digits.chars().all(|c| c.is_ascii_digit()) {
                    return None;
                }
                let path = entry.path().join("device/mem_info_vram_used");
                path.exists().then_some(path)
            })
            .collect()
    });
    let mut total = 0u64;
    let mut any = false;
    for counter in counters {
        if let Ok(text) = std::fs::read_to_string(counter)
            && let Ok(bytes) = text.trim().parse::<u64>()
        {
            total = total.saturating_add(bytes);
            any = true;
        }
    }
    any.then_some(total)
}

/// Total and in-use GTT (Graphics Translation Table) bytes reported by the
/// Linux amdgpu driver, or `None` when the counters are unavailable.
///
/// On a unified-memory APU the small VRAM carveout is not the pool the GPU
/// allocates from. GTT is the system-memory range the GPU can map, sized by
/// `ttm.pages_limit` / `amdgpu.gttsize`; it is the capacity that bounds GPU
/// admission there, not total system RAM. Values from all `cardN` devices are
/// summed, matching [`hip_vram_used_bytes`]. On a machine with no amdgpu
/// sysfs (non-Linux, or another vendor) this returns `None` and callers keep
/// their previous behavior.
pub fn gtt_memory_bytes() -> Option<(u64, u64)> {
    use std::path::PathBuf;
    use std::sync::OnceLock;

    // Discover the sysfs counter paths once; card numbering is host-specific.
    static COUNTERS: OnceLock<Vec<(PathBuf, PathBuf)>> = OnceLock::new();
    let counters = COUNTERS.get_or_init(|| {
        let Ok(entries) = std::fs::read_dir("/sys/class/drm") else {
            return Vec::new();
        };
        entries
            .filter_map(Result::ok)
            .filter_map(|entry| {
                let name = entry.file_name();
                let name = name.to_string_lossy();
                let digits = name.strip_prefix("card")?;
                if digits.is_empty() || !digits.chars().all(|c| c.is_ascii_digit()) {
                    return None;
                }
                let device = entry.path().join("device");
                let total = device.join("mem_info_gtt_total");
                let used = device.join("mem_info_gtt_used");
                (total.exists() && used.exists()).then_some((total, used))
            })
            .collect()
    });
    if counters.is_empty() {
        return None;
    }
    let mut total = 0u64;
    let mut used = 0u64;
    let mut any = false;
    for (total_path, used_path) in counters {
        let Ok(total_text) = std::fs::read_to_string(total_path) else {
            continue;
        };
        let Ok(used_text) = std::fs::read_to_string(used_path) else {
            continue;
        };
        let (Ok(total_bytes), Ok(used_bytes)) = (
            total_text.trim().parse::<u64>(),
            used_text.trim().parse::<u64>(),
        ) else {
            continue;
        };
        total = total.saturating_add(total_bytes);
        used = used.saturating_add(used_bytes);
        any = true;
    }
    any.then_some((total, used))
}

/// Non-Linux HIP builds fall back to the runtime query.
#[cfg(all(feature = "hip", not(target_os = "linux")))]
pub fn hip_vram_used_bytes() -> Option<u64> {
    let mut free = 0usize;
    let mut total = 0usize;
    // SAFETY: both pointers point to caller-owned `usize`s and remain valid
    // for the call; `hipMemGetInfo` only writes through them.
    let status = unsafe { hipMemGetInfo(&mut free, &mut total) };
    if status != 0 || total == 0 || free > total {
        return None;
    }
    Some((total - free) as u64)
}

/// Non-HIP builds have no VRAM to report.
#[cfg(not(feature = "hip"))]
pub fn hip_vram_used_bytes() -> Option<u64> {
    None
}

/// Samples [`hip_vram_used_bytes`] on a background thread and keeps the max.
///
/// Inert on non-HIP builds: `stop()` returns `None`. On HIP builds the
/// interval should be short relative to the request (50 ms is a good default)
/// because this is a sampled peak, not a driver high-water counter.
pub struct VramPeakSampler {
    stop: Arc<AtomicBool>,
    peak: Arc<AtomicU64>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl VramPeakSampler {
    /// Start sampling. Early samples can miss allocations made before the
    /// driver counter is populated; those read as zero and are skipped.
    #[cfg(feature = "hip")]
    pub fn start(interval: Duration) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let peak = Arc::new(AtomicU64::new(0));
        let worker_stop = Arc::clone(&stop);
        let worker_peak = Arc::clone(&peak);
        let handle = std::thread::spawn(move || {
            while !worker_stop.load(Ordering::Relaxed) {
                if let Some(used) = hip_vram_used_bytes() {
                    worker_peak.fetch_max(used, Ordering::Relaxed);
                }
                std::thread::sleep(interval);
            }
        });
        Self {
            stop,
            peak,
            handle: Some(handle),
        }
    }

    /// Non-HIP builds keep the same API but never spawn a thread.
    #[cfg(not(feature = "hip"))]
    pub fn start(_interval: Duration) -> Self {
        Self {
            stop: Arc::new(AtomicBool::new(false)),
            peak: Arc::new(AtomicU64::new(0)),
            handle: None,
        }
    }

    /// Stop sampling and return the largest reading seen, if any.
    pub fn stop(self) -> Option<u64> {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle {
            let _ = handle.join();
        }
        let peak = self.peak.load(Ordering::Relaxed);
        (peak > 0).then_some(peak)
    }
}
