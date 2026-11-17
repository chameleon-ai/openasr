//! Safe ownership wrappers around ggml's optional physical-memory ABI.
//!
//! This layer deliberately does not infer policy or physical-domain aliases.
//! It preserves the backend's native UUID/heap/kind claims for the process
//! broker to map, reserve atomically, and reconcile after commit.

#![allow(dead_code)]

use std::mem;

use thiserror::Error;

use crate::device::execution_memory::{MemoryDomainKey, MemoryObservationConfidence};
use crate::device::execution_route::ExecutionProvider;

use super::ffi;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BackendMemoryLifecyclePoint {
    BackendInitialized,
    AdmissionQuote,
    PostAllocationReconciliation,
    TerminalFailure,
    AfterGraphCompute,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BackendMemoryUnknownReason {
    AbiUnavailable,
    StatsUnavailable,
    IncompatibleStats,
    DeviceBudgetUnavailable,
    ProviderDoesNotReportBackendOwned,
    ProviderOwnedAccountingIncomplete,
    ProviderReliabilityUnspecified,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BackendMemoryBytes {
    Known(u64),
    Unknown(BackendMemoryUnknownReason),
}

pub(crate) fn backend_owned_unknown_reason(
    provider: ExecutionProvider,
) -> Option<BackendMemoryUnknownReason> {
    match provider {
        ExecutionProvider::Vulkan => {
            Some(BackendMemoryUnknownReason::ProviderDoesNotReportBackendOwned)
        }
        ExecutionProvider::Cpu => None,
        ExecutionProvider::Cuda => {
            Some(BackendMemoryUnknownReason::ProviderOwnedAccountingIncomplete)
        }
        _ => Some(BackendMemoryUnknownReason::ProviderReliabilityUnspecified),
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BackendMemoryDomainKind {
    HostPageable,
    HostPinned,
    Unified,
    DeviceLocal,
    FileBacked,
    Unknown(u32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BackendDeviceHealth {
    Healthy,
    Degraded,
    Quarantined,
    DeviceLost,
    Unavailable,
    Unknown(u32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BackendTerminalStatusClass {
    Success,
    Validation,
    Capacity,
    Cancelled,
    Execution,
    DeviceLost,
    BackendPoisoned,
    Unknown(i32),
}

impl BackendTerminalStatusClass {
    fn from_raw(status: i32) -> Self {
        match status {
            ffi::GGML_STATUS_SUCCESS => Self::Success,
            ffi::GGML_STATUS_FAILED => Self::Validation,
            ffi::GGML_STATUS_ALLOC_FAILED => Self::Capacity,
            ffi::GGML_STATUS_ABORTED => Self::Cancelled,
            status => Self::Unknown(status),
        }
    }
}

/// Sanitized memory evidence attached to an Exact smoke observation. It never
/// exposes physical UUIDs, backend pointers, paths, or native error text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SafeBackendMemoryReceipt {
    pub(crate) provider: ExecutionProvider,
    pub(crate) lifecycle: BackendMemoryLifecyclePoint,
    pub(crate) device_health: BackendDeviceHealth,
    pub(crate) last_status: BackendTerminalStatusClass,
    pub(crate) last_native_error: i64,
    pub(crate) quarantine_generation: u64,
    pub(crate) domain_kind: Option<BackendMemoryDomainKind>,
    pub(crate) heap_index: Option<u32>,
    pub(crate) total_bytes: BackendMemoryBytes,
    pub(crate) budget_bytes: BackendMemoryBytes,
    pub(crate) stats_generation: BackendMemoryBytes,
    pub(crate) quote_generation: BackendMemoryBytes,
    pub(crate) claim_flags: u32,
    pub(crate) observation_confidence: MemoryObservationConfidence,
    pub(crate) device_used_bytes: BackendMemoryBytes,
    pub(crate) device_free_bytes: BackendMemoryBytes,
    pub(crate) backend_owned_live_bytes: BackendMemoryBytes,
    pub(crate) backend_owned_cached_bytes: BackendMemoryBytes,
    pub(crate) backend_owned_workspace_bytes: BackendMemoryBytes,
    /// Greatest provider-reported high-water or current commitment proven at
    /// this sample. The observation sink carries the maximum across samples.
    pub(crate) backend_owned_observed_high_water_bytes: BackendMemoryBytes,
}

impl SafeBackendMemoryReceipt {
    pub(crate) fn unknown(
        lifecycle: BackendMemoryLifecyclePoint,
        reason: BackendMemoryUnknownReason,
    ) -> Self {
        Self::unknown_for_provider(ExecutionProvider::Unknown, lifecycle, reason)
    }

    fn unknown_for_provider(
        provider: ExecutionProvider,
        lifecycle: BackendMemoryLifecyclePoint,
        reason: BackendMemoryUnknownReason,
    ) -> Self {
        let value = BackendMemoryBytes::Unknown(reason);
        Self {
            provider,
            lifecycle,
            device_health: BackendDeviceHealth::Unavailable,
            last_status: BackendTerminalStatusClass::Unknown(i32::MIN),
            last_native_error: 0,
            quarantine_generation: 0,
            domain_kind: None,
            heap_index: None,
            total_bytes: value,
            budget_bytes: value,
            stats_generation: value,
            quote_generation: BackendMemoryBytes::Unknown(reason),
            claim_flags: 0,
            observation_confidence: MemoryObservationConfidence::Unknown,
            device_used_bytes: value,
            device_free_bytes: value,
            backend_owned_live_bytes: value,
            backend_owned_cached_bytes: value,
            backend_owned_workspace_bytes: value,
            backend_owned_observed_high_water_bytes: value,
        }
    }
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub(crate) enum BackendMemoryAbiError {
    #[error("backend memory ABI is unavailable")]
    Unavailable,
    #[error("backend memory ABI v1 has an incompatible layout or version")]
    Incompatible,
    #[error("backend memory operation '{operation}' failed with ggml status {status}")]
    Status {
        operation: &'static str,
        status: i32,
    },
    #[error("backend memory operation '{operation}' returned an unstable item count")]
    UnstableCount { operation: &'static str },
    #[error(
        "backend memory reserve_private committed but returned an unstable actual-claim count: sized={sized}, returned={returned}"
    )]
    ReservePrivatePostCommitCountMismatch { sized: u32, returned: u32 },
    #[error("backend memory quote mixed requests from different primary backends")]
    MixedPrimaryBackend,
    #[error("scheduler memory plan returned an item without a primary backend")]
    MissingPrimaryBackend,
    #[error("backend memory quote token does not match its requests")]
    QuoteMismatch,
}

impl BackendMemoryAbiError {
    /// `reserve_private` returned native success before Rust discovered an
    /// invalid result shape. Unlike a native non-success (failure-atomic by
    /// ABI contract), this state may already retain private allocations and
    /// therefore must be quarantined rather than refunded.
    pub(crate) fn may_have_committed_private_state(&self) -> bool {
        matches!(self, Self::ReservePrivatePostCommitCountMismatch { .. })
    }

    pub(crate) fn terminal_status(&self) -> i32 {
        match self {
            Self::Status { status, .. } => *status,
            _ => i32::MAX,
        }
    }
}

// Stock ggml exposes exactly one device-wide free/total pair per device
// (`ggml_backend_dev_memory`) and no native transaction machinery. The ABI
// below preserves OpenASR's admission semantics (quote -> reserve -> reconcile)
// on top of those facts:
//
// * domains are synthesized per device (one domain; host for CPU-class
//   devices, device-local for GPU-class devices) keyed by a synthetic
//   process-local identity, never a fabricated hardware UUID;
// * quotes are conservative committed upper bounds: used-before from live
//   stats plus the requested footprint, reconciled against fresh stats after
//   the engine commits;
// * reserve_private validates the quote token and re-issues the quote's own
//   claims without touching native state -- the real allocation happens at
//   graph/buffer commit time, and the post-allocation reconciliation is what
//   binds the outcome to physical reality;
// * stats are always available (never an ABI error): when the device query
//   fails we report zeroed bytes with BUDGET_UNAVAILABLE so typed consumers
//   see Unknown, i.e. fail-closed admission, rather than a silent total.
static REQUEST_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// Stats generation derived from the device's reported accounting tuple.
/// Identical snapshots yield identical generations; any device-side movement
/// (or a move between measured and unmeasurable) changes it. Deterministic
/// reads (e.g. Linux CPU, which always reports free == total) keep one
/// stable generation, which the admission layer requires between a quote and
/// the stats sampled from it.
fn stats_generation(free: u64, total: u64) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for value in [free, total, 1] {
        for byte in value.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash | 1
}

fn next_request_id() -> u64 {
    REQUEST_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

static PROCESS_START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();

fn monotonic_ns() -> u64 {
    PROCESS_START
        .get_or_init(std::time::Instant::now)
        .elapsed()
        .as_nanos()
        .try_into()
        .unwrap_or(u64::MAX)
}

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// Synthetic process-local identity for one ggml registry device. This is
/// deliberately not presented as a hardware UUID: bytes 0..8 are the device
/// pointer, bytes 8..16 mix in the device name, and the result never leaves
/// this process (the broker keys its domains on it).
fn synthetic_device_uuid(device: ffi::GgmlBackendDevRaw) -> [u8; 16] {
    let mut uuid = [0u8; 16];
    let raw = device as u64;
    uuid[..8].copy_from_slice(&raw.to_le_bytes());
    let name_hash = if device.is_null() {
        0x9e37_79b9_7f4a_7c15
    } else {
        let name = unsafe { ffi::ggml_backend_dev_name(device) };
        if name.is_null() {
            0x9e37_79b9_7f4a_7c15
        } else {
            let bytes = unsafe { std::ffi::CStr::from_ptr(name) }.to_bytes();
            fnv1a(bytes) | 1
        }
    };
    uuid[8..].copy_from_slice(&name_hash.to_le_bytes());
    if uuid.iter().all(|byte| *byte == 0) {
        uuid[15] = 1;
    }
    uuid
}

fn domain_kind_for_device(device_kind: Option<super::GgmlBackendKind>) -> u32 {
    match device_kind {
        Some(super::GgmlBackendKind::Gpu | super::GgmlBackendKind::IntegratedGpu) => {
            ffi::GGML_BACKEND_MEMORY_DOMAIN_DEVICE_LOCAL
        }
        Some(super::GgmlBackendKind::Meta) => ffi::GGML_BACKEND_MEMORY_DOMAIN_UNIFIED,
        // CPU and Accelerator (BLAS/AMX) domains are backed by host memory.
        _ => ffi::GGML_BACKEND_MEMORY_DOMAIN_HOST_PAGEABLE,
    }
}

fn domain_id_for_device(
    device_kind: Option<super::GgmlBackendKind>,
    device: ffi::GgmlBackendDevRaw,
) -> ffi::GgmlBackendMemoryDomainIdV1 {
    let kind = domain_kind_for_device(device_kind);
    ffi::GgmlBackendMemoryDomainIdV1 {
        physical_device_uuid: if matches!(
            kind,
            ffi::GGML_BACKEND_MEMORY_DOMAIN_DEVICE_LOCAL | ffi::GGML_BACKEND_MEMORY_DOMAIN_UNIFIED
        ) {
            synthetic_device_uuid(device)
        } else {
            [0; 16]
        },
        heap_index: 0,
        kind,
    }
}

fn device_name_lossy(device: ffi::GgmlBackendDevRaw) -> String {
    if device.is_null() {
        return "unknown-device".to_string();
    }
    let name = unsafe { ffi::ggml_backend_dev_name(device) };
    if name.is_null() {
        return "unknown-device".to_string();
    }
    unsafe { std::ffi::CStr::from_ptr(name) }
        .to_string_lossy()
        .into_owned()
}

#[derive(Debug, Clone)]
pub(crate) struct BackendMemoryDomain {
    pub(crate) id: ffi::GgmlBackendMemoryDomainIdV1,
    #[allow(dead_code)]
    pub(crate) name: String,
}

fn fingerprint_requests(requests: &[ffi::GgmlBackendMemoryRequestV1]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    let mut mix = |value: u64| {
        for byte in value.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    for request in requests {
        mix(request.request_id);
        mix(u64::from(request.kind));
        mix(u64::from(request.flags));
        mix(u64::from(request.usage));
        mix(request.backend as usize as u64);
        mix(request.peer_backend as usize as u64);
        mix(request.buft as usize as u64);
        mix(request.requested_bytes);
        mix(request.currently_allocated_bytes);
        mix(request.max_tensor_bytes);
    }
    hash
}

#[derive(Clone, Copy)]
pub(crate) struct BackendMemoryAbi {
    backend: ffi::GgmlBackendRaw,
    device: ffi::GgmlBackendDevRaw,
}

impl BackendMemoryAbi {
    /// A live ggml backend (or a registry device) is the only context the
    /// stock memory model can speak about. The registry is process resident
    /// after loading, so the returned ABI holds no native pointers beyond the
    /// backend/device pair.
    pub(crate) unsafe fn from_backend(
        backend: ffi::GgmlBackendRaw,
    ) -> Result<Self, BackendMemoryAbiError> {
        if backend.is_null() {
            return Err(BackendMemoryAbiError::Unavailable);
        }
        // SAFETY: caller guarantees `backend` is live.
        let device = unsafe { ffi::ggml_backend_get_device(backend) };
        if device.is_null() {
            return Err(BackendMemoryAbiError::Unavailable);
        }
        Ok(Self { backend, device })
    }

    /// `device` must remain a live registry device.
    pub(crate) unsafe fn from_device(
        device: ffi::GgmlBackendDevRaw,
    ) -> Result<Self, BackendMemoryAbiError> {
        if device.is_null() {
            return Err(BackendMemoryAbiError::Unavailable);
        }
        Ok(Self {
            backend: std::ptr::null_mut(),
            device,
        })
    }

    pub(crate) fn domains(&self) -> Result<Vec<BackendMemoryDomain>, BackendMemoryAbiError> {
        let device_kind = self.device_kind();
        Ok(vec![BackendMemoryDomain {
            id: domain_id_for_device(device_kind, self.device),
            name: device_name_lossy(self.device),
        }])
    }

    fn device_memory(&self) -> Result<(u64, u64), BackendMemoryAbiError> {
        let mut free: usize = 0;
        let mut total: usize = 0;
        // SAFETY: device is live; both pointers are writable.
        unsafe { ffi::ggml_backend_dev_memory(self.device, &mut free, &mut total) };
        if total == 0 {
            return Err(BackendMemoryAbiError::Status {
                operation: "device_memory",
                status: ffi::GGML_STATUS_FAILED,
            });
        }
        Ok((free as u64, total as u64))
    }

    pub(crate) fn quote(
        &self,
        requests: &[ffi::GgmlBackendMemoryRequestV1],
    ) -> Result<BackendMemoryQuote, BackendMemoryAbiError> {
        self.validate_primary_backends(requests)?;
        let (free, total) = self.device_memory()?;
        let used = total.saturating_sub(free);
        let mut requested_upper: u64 = 0;
        for request in requests {
            requested_upper = requested_upper
                .checked_add(request.requested_bytes)
                .ok_or(BackendMemoryAbiError::UnstableCount { operation: "quote" })?;
        }
        let device_kind = self.device_kind();
        let domain = domain_id_for_device(device_kind, self.device);
        let generation = stats_generation(free, total);
        let claims = requests
            .iter()
            .map(|request| ffi::GgmlBackendMemoryClaimV1 {
                struct_size: std::mem::size_of::<ffi::GgmlBackendMemoryClaimV1>() as u32,
                // A committed upper bound: admission charges it, the
                // post-allocation reconciliation refunds whatever the fresh
                // device stats show was not used.
                flags: ffi::GGML_BACKEND_MEMORY_CLAIM_CONSERVATIVE_UPPER,
                request_id: request.request_id,
                domain,
                payload_requested_bytes: request.requested_bytes,
                committed_before_bytes: used,
                committed_after_upper_bytes: used.saturating_add(requested_upper),
                commit_peak_extra_upper_bytes: requested_upper,
                resident_after_upper_bytes: request.requested_bytes,
                retained_after_use_upper_bytes: request.requested_bytes,
                releasable_after_use_upper_bytes: 0,
            })
            .collect();
        let raw = ffi::GgmlBackendMemoryQuoteV1 {
            struct_size: std::mem::size_of::<ffi::GgmlBackendMemoryQuoteV1>() as u32,
            flags: 0,
            residual_flags: 0,
            residual_request_count: 0,
            provisional_requested_upper_bytes: requested_upper,
            stats_generation: generation,
            quote_token: next_request_id(),
            request_fingerprint: fingerprint_requests(requests),
        };
        Ok(BackendMemoryQuote { raw, claims })
    }

    /// Stock ggml cannot pre-reserve backend-private memory; the allocation
    /// happens at commit time and is reconciled against fresh stats. This
    /// validates the quote token so a stale or mismatched quote cannot be
    /// committed by accident, then re-issues the quote's own claims.
    pub(crate) fn reserve_private(
        &self,
        requests: &[ffi::GgmlBackendMemoryRequestV1],
        quote: &BackendMemoryQuote,
    ) -> Result<Vec<ffi::GgmlBackendMemoryClaimV1>, BackendMemoryAbiError> {
        self.validate_primary_backends(requests)?;
        if fingerprint_requests(requests) != quote.raw.request_fingerprint {
            return Err(BackendMemoryAbiError::QuoteMismatch);
        }
        Ok(quote.claims.clone())
    }

    pub(crate) fn stats(&self) -> Result<BackendMemoryStatsSnapshot, BackendMemoryAbiError> {
        let domain_kind = self.device_kind();
        let domain = domain_id_for_device(domain_kind, self.device);
        let base = ffi::GgmlBackendMemoryStatsV1 {
            struct_size: std::mem::size_of::<ffi::GgmlBackendMemoryStatsV1>() as u32,
            flags: 0,
            domain,
            generation: 1,
            timestamp_monotonic_ns: monotonic_ns(),
            total_bytes: 0,
            budget_bytes: 0,
            device_used_bytes: 0,
            device_free_bytes: 0,
            backend_owned_live_bytes: 0,
            backend_owned_cached_bytes: 0,
            backend_owned_workspace_bytes: 0,
            backend_owned_high_water_bytes: 0,
            allocation_count: 0,
            allocation_failure_count: 0,
            health: ffi::GGML_BACKEND_MEMORY_HEALTHY,
            last_ggml_status: ffi::GGML_STATUS_SUCCESS,
            last_native_error: 0,
            quarantine_generation: 0,
        };
        let stats = match self.device_memory() {
            Ok((free, total)) => ffi::GgmlBackendMemoryStatsV1 {
                flags: 0,
                total_bytes: total,
                budget_bytes: free,
                device_used_bytes: total.saturating_sub(free),
                device_free_bytes: free,
                generation: stats_generation(free, total),
                ..base
            },
            Err(_) => ffi::GgmlBackendMemoryStatsV1 {
                flags: ffi::GGML_BACKEND_MEMORY_STATS_BUDGET_UNAVAILABLE,
                health: ffi::GGML_BACKEND_MEMORY_DEGRADED,
                ..base
            },
        };
        Ok(BackendMemoryStatsSnapshot {
            domains: vec![stats],
        })
    }

    pub(crate) fn stats_at(
        &self,
        lifecycle: BackendMemoryLifecyclePoint,
    ) -> Result<BackendMemoryStatsSnapshot, BackendMemoryAbiError> {
        let snapshot = self.stats()?;
        crate::models::native_execution_services::record_current_execution_backend_memory_stats(
            self.backend as usize,
            lifecycle,
            &snapshot,
        );
        Ok(snapshot)
    }

    pub(crate) fn backend(&self) -> ffi::GgmlBackendRaw {
        self.backend
    }

    pub(crate) fn device_kind(&self) -> Option<super::GgmlBackendKind> {
        if self.device.is_null() {
            return None;
        }
        // SAFETY: ABI construction requires a live registry device.
        let kind = unsafe { ffi::ggml_backend_dev_type(self.device) };
        Some(super::backend::backend_kind(kind))
    }

    pub(crate) fn provider(&self) -> ExecutionProvider {
        if self.device.is_null() {
            return ExecutionProvider::Unknown;
        }
        let name = unsafe { ffi::ggml_backend_dev_name(self.device) };
        if name.is_null() {
            return ExecutionProvider::Unknown;
        }
        let name = unsafe { std::ffi::CStr::from_ptr(name) }.to_string_lossy();
        ExecutionProvider::from_backend_name(name.as_ref())
    }

    /// Stock ggml has no backend cache-trim hook. Scheduler buffer release
    /// happens when the scheduler is dropped; this call keeps its existing
    /// contract ("reclaim what can be reclaimed") without fabricating a
    /// native operation.
    pub(crate) fn trim(&self, _flags: u64) -> Result<(), BackendMemoryAbiError> {
        Ok(())
    }

    fn validate_primary_backends(
        &self,
        requests: &[ffi::GgmlBackendMemoryRequestV1],
    ) -> Result<(), BackendMemoryAbiError> {
        if requests
            .iter()
            .any(|request| !request.backend.is_null() && request.backend != self.backend)
        {
            return Err(BackendMemoryAbiError::MixedPrimaryBackend);
        }
        Ok(())
    }
}

#[derive(Debug)]
pub(crate) struct BackendMemoryQuote {
    raw: ffi::GgmlBackendMemoryQuoteV1,
    claims: Vec<ffi::GgmlBackendMemoryClaimV1>,
}

impl BackendMemoryQuote {
    pub(crate) fn raw(&self) -> &ffi::GgmlBackendMemoryQuoteV1 {
        &self.raw
    }

    pub(crate) fn claims(&self) -> &[ffi::GgmlBackendMemoryClaimV1] {
        &self.claims
    }

    pub(crate) fn is_provisional(&self) -> bool {
        self.raw.flags & ffi::GGML_BACKEND_MEMORY_QUOTE_PROVISIONAL != 0
    }
}

#[derive(Debug)]
pub(crate) struct BackendMemoryStatsSnapshot {
    domains: Vec<ffi::GgmlBackendMemoryStatsV1>,
}

impl BackendMemoryStatsSnapshot {
    pub(crate) fn domains(&self) -> &[ffi::GgmlBackendMemoryStatsV1] {
        &self.domains
    }

    pub(crate) fn safe_receipts(
        &self,
        provider: ExecutionProvider,
        lifecycle: BackendMemoryLifecyclePoint,
    ) -> Vec<SafeBackendMemoryReceipt> {
        self.domains
            .iter()
            .map(|raw| safe_receipt(provider, lifecycle, raw))
            .collect()
    }
}

fn safe_receipt(
    provider: ExecutionProvider,
    lifecycle: BackendMemoryLifecyclePoint,
    raw: &ffi::GgmlBackendMemoryStatsV1,
) -> SafeBackendMemoryReceipt {
    if raw.struct_size < mem::size_of::<ffi::GgmlBackendMemoryStatsV1>() as u32 {
        return SafeBackendMemoryReceipt::unknown_for_provider(
            provider,
            lifecycle,
            BackendMemoryUnknownReason::IncompatibleStats,
        );
    }
    let domain_kind = match raw.domain.kind {
        ffi::GGML_BACKEND_MEMORY_DOMAIN_HOST_PAGEABLE => BackendMemoryDomainKind::HostPageable,
        ffi::GGML_BACKEND_MEMORY_DOMAIN_HOST_PINNED => BackendMemoryDomainKind::HostPinned,
        ffi::GGML_BACKEND_MEMORY_DOMAIN_UNIFIED => BackendMemoryDomainKind::Unified,
        ffi::GGML_BACKEND_MEMORY_DOMAIN_DEVICE_LOCAL => BackendMemoryDomainKind::DeviceLocal,
        ffi::GGML_BACKEND_MEMORY_DOMAIN_FILE_BACKED => BackendMemoryDomainKind::FileBacked,
        kind => BackendMemoryDomainKind::Unknown(kind),
    };
    let observation_confidence = if raw.domain.kind == ffi::GGML_BACKEND_MEMORY_DOMAIN_UNIFIED {
        MemoryObservationConfidence::WorkingSetBudget
    } else {
        MemoryObservationConfidence::DeviceSnapshot
    };
    let total_bytes = if raw.total_bytes == 0 {
        BackendMemoryBytes::Unknown(BackendMemoryUnknownReason::IncompatibleStats)
    } else {
        BackendMemoryBytes::Known(raw.total_bytes)
    };
    let budget_bytes = if raw.flags & ffi::GGML_BACKEND_MEMORY_STATS_BUDGET_UNAVAILABLE != 0
        || raw.budget_bytes == 0
    {
        BackendMemoryBytes::Unknown(BackendMemoryUnknownReason::DeviceBudgetUnavailable)
    } else {
        BackendMemoryBytes::Known(raw.budget_bytes)
    };
    let stats_generation = if raw.generation == 0 {
        BackendMemoryBytes::Unknown(BackendMemoryUnknownReason::IncompatibleStats)
    } else {
        BackendMemoryBytes::Known(raw.generation)
    };
    let quote_generation =
        BackendMemoryBytes::Unknown(BackendMemoryUnknownReason::StatsUnavailable);
    let device = if raw.flags & ffi::GGML_BACKEND_MEMORY_STATS_BUDGET_UNAVAILABLE != 0 {
        BackendMemoryBytes::Unknown(BackendMemoryUnknownReason::DeviceBudgetUnavailable)
    } else {
        BackendMemoryBytes::Known(raw.device_used_bytes)
    };
    let device_free = if matches!(device, BackendMemoryBytes::Known(_)) {
        BackendMemoryBytes::Known(raw.device_free_bytes)
    } else {
        device
    };
    let current_owned = raw
        .backend_owned_live_bytes
        .saturating_add(raw.backend_owned_cached_bytes)
        .max(raw.backend_owned_workspace_bytes);
    let owned_unknown = backend_owned_unknown_reason(provider);
    let owned = |value| {
        owned_unknown.map_or(
            BackendMemoryBytes::Known(value),
            BackendMemoryBytes::Unknown,
        )
    };
    SafeBackendMemoryReceipt {
        provider,
        lifecycle,
        device_health: match raw.health {
            ffi::GGML_BACKEND_MEMORY_HEALTHY => BackendDeviceHealth::Healthy,
            ffi::GGML_BACKEND_MEMORY_DEGRADED => BackendDeviceHealth::Degraded,
            ffi::GGML_BACKEND_MEMORY_QUARANTINED => BackendDeviceHealth::Quarantined,
            ffi::GGML_BACKEND_MEMORY_DEVICE_LOST => BackendDeviceHealth::DeviceLost,
            health => BackendDeviceHealth::Unknown(health),
        },
        last_status: BackendTerminalStatusClass::from_raw(raw.last_ggml_status),
        last_native_error: raw.last_native_error,
        quarantine_generation: raw.quarantine_generation,
        domain_kind: Some(domain_kind),
        heap_index: Some(raw.domain.heap_index),
        total_bytes,
        budget_bytes,
        stats_generation,
        quote_generation,
        claim_flags: 0,
        observation_confidence,
        device_used_bytes: device,
        device_free_bytes: device_free,
        backend_owned_live_bytes: owned(raw.backend_owned_live_bytes),
        backend_owned_cached_bytes: owned(raw.backend_owned_cached_bytes),
        backend_owned_workspace_bytes: owned(raw.backend_owned_workspace_bytes),
        backend_owned_observed_high_water_bytes: owned(
            raw.backend_owned_high_water_bytes.max(current_owned),
        ),
    }
}

pub(crate) fn record_backend_memory_probe(
    backend: ffi::GgmlBackendRaw,
    lifecycle: BackendMemoryLifecyclePoint,
) {
    let backend_identity = backend as usize;
    let result = unsafe { BackendMemoryAbi::from_backend(backend) };
    match result {
        Ok(abi) => {
            if abi.stats_at(lifecycle).is_err() {
                crate::models::native_execution_services::record_current_execution_backend_memory_unavailable(
                    backend_identity,
                    lifecycle,
                    BackendMemoryUnknownReason::StatsUnavailable,
                );
            }
        }
        Err(_) => {
            crate::models::native_execution_services::record_current_execution_backend_memory_unavailable(
                backend_identity,
                lifecycle,
                BackendMemoryUnknownReason::AbiUnavailable,
            );
        }
    }
}

/// Rust-side replacement for the fork's frozen scheduler memory plan. Stock
/// ggml exposes `ggml_backend_sched_reserve_size`, which reports the exact
/// buffer sizes a later `ggml_backend_sched_alloc_graph` will allocate, and
/// `ggml_backend_sched_alloc_graph` to perform it. This plan is the pair of
/// those calls plus the per-backend requests the admission broker quotes.
///
/// `ggml_backend_sched_reserve_size` internally calls
/// `ggml_backend_sched_reset`, releasing any previously bound graph
/// allocation before measuring. Callers must treat the scheduler as unbound
/// after `create` returns.
pub(crate) struct SchedulerMemoryPlan {
    scheduler: ffi::GgmlBackendSchedRaw,
    graph: ffi::GgmlCgraphRaw,
    requests: Vec<ffi::GgmlBackendMemoryRequestV1>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BackendMutationEvidence {
    ProvenUnchanged,
    MayHaveMutated,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BackendReleaseProof {
    NotRequired,
    Proven,
    Unproven,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BackendFailureDisposition {
    Complete,
    Cancel,
    Refund,
    Quarantine,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BackendTerminalStage {
    BackendPrivateReserve,
    EngineOwnedCommit,
    EngineOwnedReconcile,
    EngineOwnedRelease,
    SchedulerPlanCommit,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum BackendTerminalResourceDomains {
    Exact(Vec<MemoryDomainKey>),
    Unavailable,
}

/// Failure-time evidence sampled from every concrete backend named by the
/// frozen scheduler plan. Successful receipts are retained even when another
/// backend is unavailable, but policy treats any missing backend as unknown.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BackendTerminalEvidence {
    pub(crate) receipts: Vec<SafeBackendMemoryReceipt>,
    pub(crate) unavailable_backend_count: usize,
}

impl BackendTerminalEvidence {
    fn unavailable() -> Self {
        Self {
            receipts: Vec::new(),
            unavailable_backend_count: 1,
        }
    }

    fn device_health(&self) -> BackendDeviceHealth {
        let mut health = if self.unavailable_backend_count == 0 && !self.receipts.is_empty() {
            BackendDeviceHealth::Healthy
        } else {
            BackendDeviceHealth::Unavailable
        };
        for receipt in &self.receipts {
            health = strongest_device_health(health, receipt.device_health);
        }
        health
    }

    pub(crate) fn capture(
        backends: &[BackendMemoryAbi],
        mut unavailable_backend_count: usize,
    ) -> Self {
        let mut receipts = Vec::new();
        for abi in backends {
            match abi.stats_at(BackendMemoryLifecyclePoint::TerminalFailure) {
                Ok(snapshot) => {
                    let mut backend_receipts = snapshot.safe_receipts(
                        abi.provider(),
                        BackendMemoryLifecyclePoint::TerminalFailure,
                    );
                    if backend_receipts.is_empty() {
                        unavailable_backend_count += 1;
                    } else {
                        receipts.append(&mut backend_receipts);
                    }
                }
                Err(_) => unavailable_backend_count += 1,
            }
        }
        if receipts.is_empty() && unavailable_backend_count == 0 {
            unavailable_backend_count = 1;
        }
        Self {
            receipts,
            unavailable_backend_count,
        }
    }
}

fn strongest_device_health(
    left: BackendDeviceHealth,
    right: BackendDeviceHealth,
) -> BackendDeviceHealth {
    let rank = |health| match health {
        BackendDeviceHealth::Healthy => 0,
        BackendDeviceHealth::Degraded => 1,
        BackendDeviceHealth::Unavailable | BackendDeviceHealth::Unknown(_) => 2,
        BackendDeviceHealth::Quarantined => 3,
        BackendDeviceHealth::DeviceLost => 4,
    };
    if rank(right) > rank(left) {
        right
    } else {
        left
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BackendTerminalIdentity {
    pub(crate) provider: ExecutionProvider,
    pub(crate) stable_device_id: String,
    pub(crate) resource_domains: BackendTerminalResourceDomains,
}

impl BackendTerminalIdentity {
    pub(crate) fn exact(
        provider: ExecutionProvider,
        stable_device_id: impl Into<String>,
        resource_domains: Vec<MemoryDomainKey>,
    ) -> Self {
        Self {
            provider,
            stable_device_id: stable_device_id.into(),
            resource_domains: BackendTerminalResourceDomains::Exact(resource_domains),
        }
    }

    pub(crate) fn unavailable(provider: ExecutionProvider, stable_device_id: &str) -> Self {
        Self {
            provider,
            stable_device_id: stable_device_id.to_owned(),
            resource_domains: BackendTerminalResourceDomains::Unavailable,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BackendTerminalOutcome {
    pub(crate) stage: BackendTerminalStage,
    pub(crate) identity: BackendTerminalIdentity,
    pub(crate) status: BackendTerminalStatusClass,
    pub(crate) mutation: BackendMutationEvidence,
    pub(crate) device_health: BackendDeviceHealth,
    pub(crate) release_proof: BackendReleaseProof,
    pub(crate) evidence: BackendTerminalEvidence,
}

impl BackendTerminalOutcome {
    pub(crate) fn backend_operation(
        stage: BackendTerminalStage,
        status: i32,
        may_have_mutated: bool,
        identity: BackendTerminalIdentity,
        evidence: BackendTerminalEvidence,
    ) -> Self {
        let status = BackendTerminalStatusClass::from_raw(status);
        let device_health = evidence.device_health();
        Self {
            stage,
            identity,
            status,
            mutation: if may_have_mutated {
                BackendMutationEvidence::MayHaveMutated
            } else {
                BackendMutationEvidence::ProvenUnchanged
            },
            device_health,
            release_proof: if may_have_mutated {
                BackendReleaseProof::Unproven
            } else {
                BackendReleaseProof::NotRequired
            },
            evidence,
        }
    }

    fn scheduler_commit(
        status: i32,
        flags: u32,
        identity: BackendTerminalIdentity,
        evidence: BackendTerminalEvidence,
    ) -> Self {
        let status = BackendTerminalStatusClass::from_raw(status);
        let device_health = evidence.device_health();
        const MAY_HAVE_MUTATED: u32 = 1 << 0;
        const RELEASE_PROVEN: u32 = 1 << 1;
        let mutation_flag = MAY_HAVE_MUTATED;
        let release_flag = RELEASE_PROVEN;
        let unknown_flags = flags & !(mutation_flag | release_flag);
        let may_have_mutated = flags & (mutation_flag | release_flag) != 0 || unknown_flags != 0;
        let mutation = if may_have_mutated {
            BackendMutationEvidence::MayHaveMutated
        } else {
            BackendMutationEvidence::ProvenUnchanged
        };
        let release_proof = if !may_have_mutated {
            BackendReleaseProof::NotRequired
        } else if flags & release_flag != 0 && unknown_flags == 0 {
            BackendReleaseProof::Proven
        } else {
            BackendReleaseProof::Unproven
        };
        Self {
            stage: BackendTerminalStage::SchedulerPlanCommit,
            identity,
            status,
            mutation,
            device_health,
            release_proof,
            evidence,
        }
    }

    pub(crate) fn with_release_proof(mut self, release_proof: BackendReleaseProof) -> Self {
        if self.mutation == BackendMutationEvidence::MayHaveMutated {
            self.release_proof = release_proof;
        }
        self
    }

    pub(crate) fn disposition(&self) -> BackendFailureDisposition {
        if self.status == BackendTerminalStatusClass::Success {
            return BackendFailureDisposition::Complete;
        }
        if matches!(
            self.status,
            BackendTerminalStatusClass::DeviceLost
                | BackendTerminalStatusClass::BackendPoisoned
                | BackendTerminalStatusClass::Unknown(_)
        ) {
            return BackendFailureDisposition::Quarantine;
        }
        // Cancellation remains the user-visible request result, but it cannot
        // authorize a physical refund after native state may have changed.
        // The transaction quarantines that lease first; error projection can
        // still report Canceled to the caller.
        if self.mutation == BackendMutationEvidence::MayHaveMutated
            && self.release_proof != BackendReleaseProof::Proven
        {
            return BackendFailureDisposition::Quarantine;
        }
        if self.status == BackendTerminalStatusClass::Cancelled {
            return BackendFailureDisposition::Cancel;
        }
        if matches!(
            self.device_health,
            BackendDeviceHealth::DeviceLost
                | BackendDeviceHealth::Quarantined
                | BackendDeviceHealth::Unavailable
                | BackendDeviceHealth::Unknown(_)
        ) {
            return BackendFailureDisposition::Quarantine;
        }
        BackendFailureDisposition::Refund
    }
}

#[derive(Debug, Error)]
#[error("{source} (scheduler terminal outcome={outcome:?})")]
pub(crate) struct SchedulerMemoryPlanCommitError {
    source: BackendMemoryAbiError,
    outcome: BackendTerminalOutcome,
}

impl SchedulerMemoryPlanCommitError {
    pub(crate) fn requires_quarantine(&self) -> bool {
        self.outcome.disposition() == BackendFailureDisposition::Quarantine
    }

    pub(crate) fn outcome(&self) -> BackendTerminalOutcome {
        self.outcome.clone()
    }

    pub(crate) fn into_source(self) -> BackendMemoryAbiError {
        self.source
    }
}

impl super::backend_memory_admission::NativeOwnerAttachedCommitOutcome
    for SchedulerMemoryPlanCommitError
{
    fn requires_quarantine(&self) -> bool {
        SchedulerMemoryPlanCommitError::requires_quarantine(self)
    }
}

impl SchedulerMemoryPlan {
    /// `scheduler`, `graph`, and every tensor reachable from `graph` must stay
    /// live and immutable until this plan is committed or dropped.
    /// `backends` is the scheduler's backend set in creation order; stock
    /// ggml assigns one default buffer type per scheduler backend, so
    /// `sizes[i]` is the footprint `sched_alloc_graph` will request from
    /// `backends[i]`.
    pub(crate) unsafe fn create(
        scheduler: ffi::GgmlBackendSchedRaw,
        graph: ffi::GgmlCgraphRaw,
        backends: &[ffi::GgmlBackendRaw],
    ) -> Result<Self, BackendMemoryAbiError> {
        if scheduler.is_null() || graph.is_null() || backends.is_empty() {
            return Err(BackendMemoryAbiError::Status {
                operation: "scheduler_plan/create",
                status: ffi::GGML_STATUS_FAILED,
            });
        }
        // Measure the footprint with a throwaway gallocr on the primary
        // backend's buffer type. For the stock-ggml scheduler this is exact
        // for the single-backend case and an upper bound for multi-backend
        // (the pinned split is unreachable from outside the scheduler;
        // a conservative node-byte sum stands in there). Crucially, this
        // never reserves on the scheduler's own allocator: calling the stock
        // `ggml_backend_sched_reserve_size` leaves `galloc->n_nodes` set with
        // `buffers == NULL`, which makes the subsequent
        // `ggml_backend_sched_alloc_graph` skip allocation and dereference a
        // null buffer (SIGSEGV).
        let mut sizes = vec![0_usize; backends.len()];
        let primary_buft = unsafe {
            let device = ffi::ggml_backend_get_device(backends[0]);
            if device.is_null() {
                return Err(BackendMemoryAbiError::Status {
                    operation: "scheduler_plan/create",
                    status: ffi::GGML_STATUS_FAILED,
                });
            }
            ffi::ggml_backend_dev_buffer_type(device)
        };
        if primary_buft.is_null() {
            return Err(BackendMemoryAbiError::Status {
                operation: "scheduler_plan/create",
                status: ffi::GGML_STATUS_FAILED,
            });
        }
        let measure = unsafe { ffi::ggml_gallocr_new(primary_buft) };
        if measure.is_null() {
            return Err(BackendMemoryAbiError::Status {
                operation: "scheduler_plan/create",
                status: ffi::GGML_STATUS_FAILED,
            });
        }
        if backends.len() == 1 {
            let mut measured = [0_usize; 1];
            // SAFETY: both graph and the fresh gallocr are live; `measured`
            // has exactly one slot, matching one buffer.
            unsafe {
                ffi::ggml_gallocr_reserve_n_size(
                    measure,
                    graph,
                    std::ptr::null(),
                    std::ptr::null(),
                    measured.as_mut_ptr(),
                )
            };
            sizes[0] = measured[0];
        } else {
            // Conservative upper bound: sum of node tensor bytes. The
            // scheduler never needs more than this for a single-backend
            // placement of any on-graph tensor, and binaries that place
            // tensors on a different backend can only move bytes within
            // the same total (backend-specific workspaces are typically
            // smaller than payload).
            let mut total = 0_u64;
            let n_nodes = unsafe { ffi::ggml_graph_n_nodes(graph) };
            for index in 0..n_nodes {
                let node = unsafe { ffi::ggml_graph_node(graph, index) };
                if !node.is_null() {
                    total = total.saturating_add(unsafe { ffi::ggml_nbytes(node) } as u64);
                }
            }
            sizes[0] = total as usize;
        }
        unsafe { ffi::ggml_gallocr_free(measure) };
        let requests = backends
            .iter()
            .enumerate()
            .map(|(index, backend)| ffi::GgmlBackendMemoryRequestV1 {
                struct_size: std::mem::size_of::<ffi::GgmlBackendMemoryRequestV1>() as u32,
                kind: ffi::GGML_BACKEND_MEMORY_REQUEST_GRAPH_PRIVATE,
                flags: 0,
                usage: ffi::GGML_BACKEND_BUFFER_USAGE_COMPUTE as u32,
                request_id: next_request_id(),
                backend: *backend,
                peer_backend: std::ptr::null_mut(),
                buft: if index == 0 {
                    primary_buft
                } else {
                    std::ptr::null_mut()
                },
                graph,
                host_ptr: std::ptr::null(),
                requested_bytes: sizes[index] as u64,
                currently_allocated_bytes: 0,
                max_tensor_bytes: 0,
            })
            .collect();
        Ok(Self {
            scheduler,
            graph,
            requests,
        })
    }

    pub(crate) fn requests(
        &self,
    ) -> Result<Vec<ffi::GgmlBackendMemoryRequestV1>, BackendMemoryAbiError> {
        Ok(self.requests.clone())
    }

    /// Partitions one multi-backend scheduler plan without altering request
    /// order inside each backend batch. Each batch must be quoted by the proc
    /// table resolved from that exact primary backend.
    pub(crate) fn requests_by_backend(
        &self,
    ) -> Result<
        Vec<(ffi::GgmlBackendRaw, Vec<ffi::GgmlBackendMemoryRequestV1>)>,
        BackendMemoryAbiError,
    > {
        let requests = self.requests()?;
        let mut batches: Vec<(ffi::GgmlBackendRaw, Vec<ffi::GgmlBackendMemoryRequestV1>)> =
            Vec::new();
        for request in requests {
            if request.backend.is_null() {
                return Err(BackendMemoryAbiError::MissingPrimaryBackend);
            }
            if let Some((_, batch)) = batches
                .iter_mut()
                .find(|(backend, _)| *backend == request.backend)
            {
                batch.push(request);
            } else {
                batches.push((request.backend, vec![request]));
            }
        }
        Ok(batches)
    }

    /// Commits the measured allocation. On stock ggml this is
    /// `ggml_backend_sched_alloc_graph`; on failure the scheduler is reset so
    /// the broker's refund of the reservation is accurate.
    pub(crate) fn commit(
        self,
        identity: BackendTerminalIdentity,
    ) -> Result<(), SchedulerMemoryPlanCommitError> {
        // Stock ggml asserts `!sched->is_alloc` inside
        // `ggml_backend_sched_alloc_graph`: the scheduler owns one
        // graph-specific allocation at a time. A sibling persistent graph may
        // have left the shared scheduler allocated, so reset first. Reset is
        // idempotent and does not release the gallocr buffers; the subsequent
        // reserve_n reuses them in place when they are large enough.
        unsafe { ffi::ggml_backend_sched_reset(self.scheduler) };
        // SAFETY: the scheduler is live and the graph is still unbound.
        let allocated = unsafe { ffi::ggml_backend_sched_alloc_graph(self.scheduler, self.graph) };
        if allocated {
            return Ok(());
        }
        // SAFETY: the scheduler is live; reset releases any buffers reserved
        // by the failed attempt.
        unsafe { ffi::ggml_backend_sched_reset(self.scheduler) };
        let evidence = self.terminal_evidence();
        Err(SchedulerMemoryPlanCommitError {
            source: BackendMemoryAbiError::Status {
                operation: "scheduler_plan/commit",
                status: ffi::GGML_STATUS_ALLOC_FAILED,
            },
            outcome: BackendTerminalOutcome::scheduler_commit(
                ffi::GGML_STATUS_ALLOC_FAILED,
                0,
                identity,
                evidence,
            ),
        })
    }

    fn terminal_evidence(&self) -> BackendTerminalEvidence {
        let batches = match self.requests_by_backend() {
            Ok(batches) => batches,
            Err(_) => return BackendTerminalEvidence::unavailable(),
        };
        let mut backends = Vec::new();
        let mut unavailable_backend_count = 0;
        for (backend, _) in batches {
            match unsafe { BackendMemoryAbi::from_backend(backend) } {
                Ok(abi) => backends.push(abi),
                Err(_) => unavailable_backend_count += 1,
            }
        }
        BackendTerminalEvidence::capture(&backends, unavailable_backend_count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_os = "macos")]
    use crate::ggml_runtime::ensure_backends_loaded;

    fn test_stats() -> ffi::GgmlBackendMemoryStatsV1 {
        ffi::GgmlBackendMemoryStatsV1 {
            struct_size: mem::size_of::<ffi::GgmlBackendMemoryStatsV1>() as u32,
            domain: ffi::GgmlBackendMemoryDomainIdV1 {
                kind: ffi::GGML_BACKEND_MEMORY_DOMAIN_DEVICE_LOCAL,
                heap_index: 2,
                ..Default::default()
            },
            device_used_bytes: 900,
            device_free_bytes: 100,
            backend_owned_live_bytes: 20,
            backend_owned_cached_bytes: 30,
            backend_owned_workspace_bytes: 40,
            backend_owned_high_water_bytes: 45,
            ..Default::default()
        }
    }

    #[test]
    fn safe_cuda_receipt_keeps_incomplete_owned_accounting_typed_unknown() {
        let receipt = safe_receipt(
            ExecutionProvider::Cuda,
            BackendMemoryLifecyclePoint::PostAllocationReconciliation,
            &test_stats(),
        );
        assert_eq!(receipt.device_used_bytes, BackendMemoryBytes::Known(900));
        assert_eq!(receipt.device_free_bytes, BackendMemoryBytes::Known(100));
        let unknown = BackendMemoryBytes::Unknown(
            BackendMemoryUnknownReason::ProviderOwnedAccountingIncomplete,
        );
        assert_eq!(receipt.backend_owned_live_bytes, unknown);
        assert_eq!(receipt.backend_owned_cached_bytes, unknown);
        assert_eq!(receipt.backend_owned_workspace_bytes, unknown);
        assert_eq!(receipt.backend_owned_observed_high_water_bytes, unknown);
    }

    #[test]
    fn safe_vulkan_receipt_never_presents_unreported_owned_zero_as_known() {
        let mut raw = test_stats();
        raw.backend_owned_live_bytes = 0;
        raw.backend_owned_cached_bytes = 0;
        raw.backend_owned_workspace_bytes = 0;
        raw.backend_owned_high_water_bytes = 0;
        let receipt = safe_receipt(
            ExecutionProvider::Vulkan,
            BackendMemoryLifecyclePoint::BackendInitialized,
            &raw,
        );
        let unknown = BackendMemoryBytes::Unknown(
            BackendMemoryUnknownReason::ProviderDoesNotReportBackendOwned,
        );
        assert_eq!(receipt.backend_owned_live_bytes, unknown);
        assert_eq!(receipt.backend_owned_cached_bytes, unknown);
        assert_eq!(receipt.backend_owned_workspace_bytes, unknown);
        assert_eq!(receipt.backend_owned_observed_high_water_bytes, unknown);
        assert_eq!(receipt.device_used_bytes, BackendMemoryBytes::Known(900));
    }

    #[test]
    fn safe_receipt_keeps_unavailable_device_budget_typed_unknown() {
        let mut raw = test_stats();
        raw.flags = ffi::GGML_BACKEND_MEMORY_STATS_BUDGET_UNAVAILABLE;
        let receipt = safe_receipt(
            ExecutionProvider::Vulkan,
            BackendMemoryLifecyclePoint::AdmissionQuote,
            &raw,
        );
        let unknown =
            BackendMemoryBytes::Unknown(BackendMemoryUnknownReason::DeviceBudgetUnavailable);
        assert_eq!(receipt.device_used_bytes, unknown);
        assert_eq!(receipt.device_free_bytes, unknown);
    }

    fn terminal_evidence(
        provider: ExecutionProvider,
        health: BackendDeviceHealth,
        last_status: i32,
        native_error: i64,
        quarantine_generation: u64,
    ) -> BackendTerminalEvidence {
        let raw_health = match health {
            BackendDeviceHealth::Healthy => ffi::GGML_BACKEND_MEMORY_HEALTHY,
            BackendDeviceHealth::Degraded => ffi::GGML_BACKEND_MEMORY_DEGRADED,
            BackendDeviceHealth::Quarantined => ffi::GGML_BACKEND_MEMORY_QUARANTINED,
            BackendDeviceHealth::DeviceLost => ffi::GGML_BACKEND_MEMORY_DEVICE_LOST,
            BackendDeviceHealth::Unavailable | BackendDeviceHealth::Unknown(_) => u32::MAX,
        };
        let mut raw = test_stats();
        raw.health = raw_health;
        raw.last_ggml_status = last_status;
        raw.last_native_error = native_error;
        raw.quarantine_generation = quarantine_generation;
        BackendTerminalEvidence {
            receipts: vec![safe_receipt(
                provider,
                BackendMemoryLifecyclePoint::TerminalFailure,
                &raw,
            )],
            unavailable_backend_count: 0,
        }
    }

    fn healthy_terminal_evidence(provider: ExecutionProvider) -> BackendTerminalEvidence {
        terminal_evidence(
            provider,
            BackendDeviceHealth::Healthy,
            ffi::GGML_STATUS_SUCCESS,
            0,
            0,
        )
    }

    #[test]
    fn scheduler_commit_quarantines_only_unrecoverable_failures() {
        let identity =
            || BackendTerminalIdentity::unavailable(ExecutionProvider::Vulkan, "Vulkan0");
        let error = |status, flags, evidence| SchedulerMemoryPlanCommitError {
            source: BackendMemoryAbiError::Status {
                operation: "scheduler_plan/commit",
                status,
            },
            outcome: BackendTerminalOutcome::scheduler_commit(status, flags, identity(), evidence),
        };
        let healthy = || healthy_terminal_evidence(ExecutionProvider::Vulkan);
        let mutated = 1_u32 << 0;
        let released = mutated | (1_u32 << 1);

        assert!(!error(ffi::GGML_STATUS_FAILED, 0, healthy()).requires_quarantine());
        assert!(!error(ffi::GGML_STATUS_ALLOC_FAILED, 0, healthy()).requires_quarantine());
        assert!(!error(ffi::GGML_STATUS_FAILED, 0, healthy()).requires_quarantine());
        assert!(error(ffi::GGML_STATUS_FAILED, mutated, healthy()).requires_quarantine());
        assert!(error(ffi::GGML_STATUS_ALLOC_FAILED, mutated, healthy()).requires_quarantine());
        assert!(!error(ffi::GGML_STATUS_ALLOC_FAILED, released, healthy()).requires_quarantine());
        assert!(error(i32::MAX, 0, healthy()).requires_quarantine());
        assert!(error(i32::MAX - 1, 0, healthy()).requires_quarantine());
        assert!(error(i32::MAX, 0, healthy()).requires_quarantine());
        assert_eq!(
            error(
                ffi::GGML_STATUS_ABORTED,
                mutated,
                BackendTerminalEvidence::unavailable()
            )
            .outcome()
            .disposition(),
            BackendFailureDisposition::Quarantine
        );
        assert_eq!(
            error(ffi::GGML_STATUS_ABORTED, 0, healthy())
                .outcome()
                .disposition(),
            BackendFailureDisposition::Cancel
        );
        assert_eq!(
            error(ffi::GGML_STATUS_ABORTED, released, healthy())
                .outcome()
                .disposition(),
            BackendFailureDisposition::Cancel
        );

        let released_outcome = BackendTerminalOutcome::scheduler_commit(
            ffi::GGML_STATUS_ALLOC_FAILED,
            released,
            identity(),
            healthy(),
        );
        assert_eq!(
            released_outcome.disposition(),
            BackendFailureDisposition::Refund
        );
        assert_eq!(
            released_outcome.identity.provider,
            ExecutionProvider::Vulkan
        );
        assert_eq!(released_outcome.identity.stable_device_id, "Vulkan0");

        let lost_evidence = terminal_evidence(
            ExecutionProvider::Vulkan,
            BackendDeviceHealth::DeviceLost,
            3, // retired fork device-lost alias; stock ggml has no such status
            -4,
            7,
        );
        assert!(error(ffi::GGML_STATUS_ALLOC_FAILED, 0, lost_evidence).requires_quarantine());
        assert!(
            error(
                ffi::GGML_STATUS_ALLOC_FAILED,
                0,
                BackendTerminalEvidence::unavailable(),
            )
            .requires_quarantine()
        );
    }

    #[test]
    fn terminal_outcome_preserves_exact_lane_and_resource_domains() {
        let identity = BackendTerminalIdentity::exact(
            ExecutionProvider::Hip,
            "HIP0",
            vec![MemoryDomainKey::SystemMemory],
        );
        let outcome = BackendTerminalOutcome::scheduler_commit(
            ffi::GGML_STATUS_ALLOC_FAILED,
            0,
            identity,
            healthy_terminal_evidence(ExecutionProvider::Hip),
        );
        assert_eq!(outcome.stage, BackendTerminalStage::SchedulerPlanCommit);
        assert_eq!(outcome.identity.provider, ExecutionProvider::Hip);
        assert_eq!(outcome.identity.stable_device_id, "HIP0");
        assert_eq!(
            outcome.identity.resource_domains,
            BackendTerminalResourceDomains::Exact(vec![MemoryDomainKey::SystemMemory])
        );
        assert_eq!(outcome.disposition(), BackendFailureDisposition::Refund);
    }

    #[test]
    fn safe_receipt_preserves_typed_terminal_health_evidence() {
        let mut raw = test_stats();
        raw.health = ffi::GGML_BACKEND_MEMORY_DEVICE_LOST;
        raw.last_ggml_status = 3; // retired fork device-lost alias
        raw.last_native_error = -4;
        raw.quarantine_generation = 7;
        let receipt = safe_receipt(
            ExecutionProvider::Vulkan,
            BackendMemoryLifecyclePoint::AfterGraphCompute,
            &raw,
        );
        assert_eq!(receipt.device_health, BackendDeviceHealth::DeviceLost);
        assert_eq!(receipt.last_status, BackendTerminalStatusClass::Unknown(3));
        assert_eq!(receipt.last_native_error, -4);
        assert_eq!(receipt.quarantine_generation, 7);
    }

    #[test]
    fn memory_api_resolves_from_registry_device_without_a_backend_context() {
        crate::ggml_runtime::ensure_backends_loaded();
        let cpu = crate::ggml_available_devices()
            .into_iter()
            .find(|device| device.kind == crate::GgmlBackendKind::Cpu)
            .expect("cpu device");
        let abi =
            unsafe { BackendMemoryAbi::from_device(cpu.as_ptr()) }.expect("device-only memory ABI");
        assert!(abi.backend().is_null());
        assert_eq!(abi.provider(), ExecutionProvider::Cpu);
        abi.domains()
            .expect("cpu registry device must expose memory domains");
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn repeated_metal_memory_quotes_reuse_one_device_context() {
        ensure_backends_loaded();
        let backend = unsafe { ffi::ggml_backend_init_best() };
        assert!(!backend.is_null(), "macOS must expose a ggml backend");

        let name = unsafe {
            std::ffi::CStr::from_ptr(ffi::ggml_backend_name(backend))
                .to_string_lossy()
                .to_ascii_lowercase()
        };
        if !name.contains("metal") && !name.starts_with("mtl") {
            unsafe { ffi::ggml_backend_free(backend) };
            return;
        }

        let abi = unsafe { BackendMemoryAbi::from_backend(backend) }
            .expect("Metal must expose the memory ABI");
        let device = unsafe { ffi::ggml_backend_get_device(backend) };
        assert!(!device.is_null());
        let buft = unsafe { ffi::ggml_backend_dev_buffer_type(device) };
        assert!(!buft.is_null());
        let request = ffi::GgmlBackendMemoryRequestV1 {
            kind: ffi::GGML_BACKEND_MEMORY_REQUEST_BUFFER,
            usage: ffi::GGML_BACKEND_BUFFER_USAGE_COMPUTE as u32,
            request_id: 1,
            backend,
            buft,
            requested_bytes: 64 * 1024,
            ..Default::default()
        };
        let before = unsafe { ffi::openasr_ggml_metal_cached_device_count() };
        for _ in 0..128 {
            abi.quote(&[request])
                .expect("repeated Metal memory quote must remain valid");
        }
        let after = unsafe { ffi::openasr_ggml_metal_cached_device_count() };

        unsafe { ffi::ggml_backend_free(backend) };
        assert_eq!(after, before, "memory quote created Metal device contexts");
    }
}
