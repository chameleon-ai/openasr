//! Backend-native captured graph lifecycle evidence.
//!
//! Stock ggml 1.0.0 has no captured-graph or capture-executable observation
//! hook (the fork's lifecycle ABI is gone and there is no upstream
//! equivalent). The binding therefore reports `Unavailable` for every
//! backend, which the graph layer treats as "no native capture evidence":
//! ordinary Rust-side lifecycle receipts (graph identity, generations,
//! compute counts) are produced exactly as before, and no capture event is
//! fabricated. The `Incompatible` state remains distinguishable so a
//! diagnostic build can still fail closed when a binding is injected for
//! tests.

use std::{ffi::c_void, ptr::NonNull};

use thiserror::Error;

/// Kept for native-evidence parity with the lifecycle receipts; stock ggml
/// cannot produce these, so only tests construct them.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NativeGraphExecutableChange {
    Instantiated,
    Updated,
    Replaced,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct NativeGraphLifecycleObservation {
    pub capture_supported: bool,
    pub graph_tracked: bool,
    pub capture_enabled: bool,
    pub executable_generation: Option<u64>,
    pub last_executable_change: Option<NativeGraphExecutableChange>,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum BackendGraphLifecycleBinding {
    Unavailable,
    /// Diagnostic/testing seam: a binding that fails closed on every observe.
    #[allow(dead_code)]
    Incompatible,
}

impl BackendGraphLifecycleBinding {
    /// Stock ggml exposes no per-backend lifecycle observation, so every
    /// backend resolves to `Unavailable`. The `unsafe` marker is kept for
    /// symmetry with the prior ABI-backed resolution (callers already hold a
    /// live backend).
    pub(crate) unsafe fn resolve(_backend: NonNull<c_void>) -> Self {
        Self::Unavailable
    }

    pub(crate) fn observe(
        self,
        _backend: NonNull<c_void>,
        _graph: NonNull<c_void>,
    ) -> Result<Option<NativeGraphLifecycleObservation>, BackendGraphLifecycleError> {
        match self {
            Self::Unavailable => Ok(None),
            Self::Incompatible => Err(BackendGraphLifecycleError::Incompatible),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub(crate) enum BackendGraphLifecycleError {
    #[error("backend graph lifecycle ABI is incompatible")]
    Incompatible,
    #[error(
        "backend graph lifecycle capture support or enablement changed within one graph generation"
    )]
    CapturePolicyDrift,
    #[error("backend graph lifecycle stopped tracking a live graph generation")]
    GraphTrackingDisappeared,
    #[error("backend graph lifecycle did not track a capture-capable graph after compute")]
    GraphNotTrackedAfterCompute,
    #[error("backend graph lifecycle capture executable disappeared within one graph generation")]
    CaptureExecutableDisappeared,
    #[error(
        "backend graph lifecycle capture generation regressed: previous={previous}, actual={actual}"
    )]
    CaptureGenerationRegressed { previous: u64, actual: u64 },
    #[error("backend graph lifecycle capture generation changed without a native change reason")]
    CaptureGenerationChangedWithoutReason,
    #[error("backend graph lifecycle capture generation changed outside the measured compute")]
    CaptureGenerationChangedOutsideCompute,
}
