use std::ffi::{c_char, c_int, c_void};

pub(crate) type GgmlBackendRaw = *mut c_void;
pub(crate) type GgmlBackendDevRaw = *mut c_void;
pub(crate) type GgmlBackendBufferRaw = *mut c_void;
pub(crate) type GgmlBackendBufferTypeRaw = *mut c_void;
pub(crate) type GgmlBackendSchedRaw = *mut c_void;
pub(crate) type GgmlGallocrRaw = *mut c_void;
pub(crate) type GgmlBackendRegRaw = *mut c_void;
pub(crate) type GgmlContextRaw = *mut c_void;
pub(crate) type GgmlTensorRaw = *mut c_void;
pub(crate) type GgmlCgraphRaw = *mut c_void;
pub(crate) type GgufContextRaw = *mut c_void;
pub(crate) const GGML_MAX_DIMS: usize = 4;
/// Mirrors `GGML_PREC_UNDEFINED` in ggml.h (`enum ggml_prec`, value 0). Used
/// where an op carries no precision constraint; `GGML_PREC_F32` pins the
/// accumulator to f32.
pub(crate) const GGML_PREC_UNDEFINED: c_int = 0;
pub(crate) const GGML_PREC_F32: c_int = 10;
pub(crate) const GGML_OP_POOL_MAX: c_int = 0;

pub(crate) type GgmlToFloatFn = unsafe extern "C" fn(x: *const c_void, y: *mut f32, k: i64);

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct GgmlTensorLayoutPrefix {
    pub type_: c_int,
    pub buffer: *mut c_void,
    pub ne: [i64; GGML_MAX_DIMS],
    pub nb: [usize; GGML_MAX_DIMS],
}

pub(crate) const GGML_MAX_SRC: usize = 10;
pub(crate) const GGML_MAX_OP_PARAMS_I32: usize = 16;

/// Mirrors the leading fields of `struct ggml_cgraph` (ggml-impl.h, vendored
/// stock ggml) up to the nodes/leafs pointers. The scheduler hands out a
/// shared gallocr whose buffers are reused across retained cgraphs, so the
/// runtime must walk a cgraph to unbind the tensor bindings that reference a
/// (possibly freed) buffer before another graph rebinds onto the same
/// scheduler. `#[repr(C)]` reproduces the C layout; keep it in lockstep with
/// the vendored ggml.c if the submodule pin changes cgraph fields ahead of
/// `leafs`.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct GgmlCgraphPrefix {
    pub size: c_int,
    pub n_nodes: c_int,
    pub n_leafs: c_int,
    pub nodes: *mut GgmlTensorRaw,
    pub grads: *mut GgmlTensorRaw,
    pub grad_accs: *mut GgmlTensorRaw,
    pub leafs: *mut GgmlTensorRaw,
}

/// Mirrors `struct ggml_tensor` (ggml.h) far enough to reach the `view_src`
/// and `data` fields the CPU step-buffer grow-to-fit reuse needs to inspect
/// (see `bind_unallocated_context_tensors` in cpu_graph.rs), extending the
/// `type_`/`buffer`/`ne`/`nb` prefix `GgmlTensorLayoutPrefix` already reads.
/// `#[repr(C)]` with matching field types/order reproduces the C compiler's
/// layout, so this must be kept in lockstep with the vendored ggml.h if the
/// submodule pin ever changes tensor struct fields ahead of `data`.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct GgmlTensorAllocPrefix {
    pub type_: c_int,
    pub buffer: *mut c_void,
    pub ne: [i64; GGML_MAX_DIMS],
    pub nb: [usize; GGML_MAX_DIMS],
    pub op: c_int,
    pub op_params: [i32; GGML_MAX_OP_PARAMS_I32],
    pub flags: i32,
    pub src: [*mut c_void; GGML_MAX_SRC],
    pub view_src: *mut c_void,
    pub view_offs: usize,
    pub data: *mut c_void,
}

/// Mirrors `struct ggml_tallocr` (ggml-alloc.h), a stable public ggml struct
/// (not an internal implementation detail) exposing the bump-allocator ggml
/// itself uses to bind tensors into an already-allocated backend buffer.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct GgmlTallocr {
    pub buffer: GgmlBackendBufferRaw,
    pub base: *mut c_void,
    pub alignment: usize,
    pub offset: usize,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct GgufInitParams {
    pub no_alloc: bool,
    pub ctx: *mut *mut c_void,
}

#[derive(Clone, Copy)]
pub(crate) struct GgufParseLimits {
    pub max_tensors: u64,
    pub max_kv: u64,
    pub max_string_bytes: u64,
    pub max_array_elements: u64,
    pub max_header_bytes: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct GgmlInitParams {
    pub mem_size: usize,
    pub mem_buffer: *mut c_void,
    pub no_alloc: bool,
}

pub(crate) const GGML_BACKEND_DEVICE_TYPE_CPU: c_int = 0;
pub(crate) const GGML_BACKEND_DEVICE_TYPE_GPU: c_int = 1;
pub(crate) const GGML_BACKEND_DEVICE_TYPE_IGPU: c_int = 2;
pub(crate) const GGML_BACKEND_DEVICE_TYPE_ACCEL: c_int = 3;
pub(crate) const GGML_BACKEND_DEVICE_TYPE_META: c_int = 4;

pub(crate) const GGML_STATUS_ALLOC_FAILED: c_int = -2;
pub(crate) const GGML_STATUS_FAILED: c_int = -1;
pub(crate) const GGML_STATUS_SUCCESS: c_int = 0;
/// Mirrors `enum ggml_status` in ggml.h. `ggml_backend_graph_compute` and
/// `ggml_backend_sched_graph_compute` return the merged submit + completion
/// status, so this is the single terminal outcome for a compute call -- no
/// separate synchronize step is needed to observe a completion-phase failure.
/// `GGML_STATUS_ABORTED` is the cooperative cancellation result; Rust maps it
/// to the typed per-job cancellation path rather than a generic compute
/// failure. All other non-success values are terminal failures.
pub(crate) const GGML_STATUS_ABORTED: c_int = 1;
pub(crate) const GGML_BACKEND_BUFFER_USAGE_WEIGHTS: c_int = 1;
pub(crate) const GGML_BACKEND_BUFFER_USAGE_COMPUTE: c_int = 2;
// Only the graph-metadata tests read this; stock ggml itself propagates the
// flag through view chains, which is what the regression gates assert.
#[allow(dead_code)]
pub(crate) const GGML_TENSOR_FLAG_OUTPUT: c_int = 2;

// The stock-ggml memory model exposes exactly one device-wide free/total pair
// via `ggml_backend_dev_memory`; there are no domains, heaps, or provider
// failure taxonomy. The OpenASR broker keys synthetic GPU domains on a
// process-local identity derived from the device pointer (see backend_memory.rs).

// Scheduler measured reservations carry no backend-native provenance; the
// broker treats them as committed upper bounds until the post-allocation
// reconciliation against live device stats lands.
pub(crate) const GGML_BACKEND_MEMORY_REQUEST_BUFFER: u32 = 1;
pub(crate) const GGML_BACKEND_MEMORY_REQUEST_HOST_IMPORT: u32 = 2;
pub(crate) const GGML_BACKEND_MEMORY_REQUEST_GRAPH_PRIVATE: u32 = 3;
pub(crate) const GGML_BACKEND_MEMORY_REQUEST_TRANSFER: u32 = 4;
pub(crate) const GGML_BACKEND_MEMORY_QUOTE_PROVISIONAL: u32 = 1 << 0;
pub(crate) const GGML_BACKEND_MEMORY_QUOTE_HAS_RESIDUAL_UNCERTAINTY: u32 = 1 << 1;
pub(crate) const GGML_BACKEND_MEMORY_QUOTE_OPAQUE_DRIVER_COSTS_REQUIRE_DOMAIN_HEADROOM: u32 =
    1 << 2;
pub(crate) const GGML_BACKEND_MEMORY_CLAIM_EXACT: u32 = 1 << 0;
pub(crate) const GGML_BACKEND_MEMORY_CLAIM_CONSERVATIVE_UPPER: u32 = 1 << 1;
pub(crate) const GGML_BACKEND_MEMORY_CLAIM_DRIVER_ESTIMATE: u32 = 1 << 2;
pub(crate) const GGML_BACKEND_MEMORY_CLAIM_PROVISIONAL: u32 = 1 << 6;
#[cfg(test)]
pub(crate) const GGML_BACKEND_MEMORY_CLAIM_FILE_BACKED: u32 = 1 << 5;
#[cfg(test)]
pub(crate) const GGML_BACKEND_MEMORY_RESIDUAL_BACKEND_PRIVATE: u32 = 1 << 0;
pub(crate) const GGML_BACKEND_MEMORY_HEALTHY: u32 = 0;
pub(crate) const GGML_BACKEND_MEMORY_DEGRADED: u32 = 1;
pub(crate) const GGML_BACKEND_MEMORY_QUARANTINED: u32 = 2;
pub(crate) const GGML_BACKEND_MEMORY_DEVICE_LOST: u32 = 3;

pub(crate) const GGML_BACKEND_MEMORY_DOMAIN_DEVICE_LOCAL: u32 = 1;
pub(crate) const GGML_BACKEND_MEMORY_DOMAIN_HOST_PAGEABLE: u32 = 2;
pub(crate) const GGML_BACKEND_MEMORY_DOMAIN_HOST_PINNED: u32 = 3;
pub(crate) const GGML_BACKEND_MEMORY_DOMAIN_UNIFIED: u32 = 4;
pub(crate) const GGML_BACKEND_MEMORY_DOMAIN_FILE_BACKED: u32 = 5;
pub(crate) const GGML_BACKEND_MEMORY_STATS_BUDGET_UNAVAILABLE: u32 = 1 << 0;

/// One backend-specific memory request. This is the OpenASR-owned quoting
/// currency, not an FFI layout; stock ggml has no equivalent struct.
#[derive(Clone, Copy, Debug)]
pub(crate) struct GgmlBackendMemoryRequestV1 {
    // Native-evidence parity fields: never read by the broker, retained so
    // the struct remains self-describing for diagnostics and debugging.
    #[allow(dead_code)]
    pub struct_size: u32,
    pub kind: u32,
    pub flags: u32,
    pub usage: u32,
    pub request_id: u64,
    pub backend: GgmlBackendRaw,
    #[allow(dead_code)]
    pub peer_backend: GgmlBackendRaw,
    #[allow(dead_code)]
    pub buft: GgmlBackendBufferTypeRaw,
    #[allow(dead_code)]
    pub graph: GgmlCgraphRaw,
    #[allow(dead_code)]
    pub host_ptr: *const c_void,
    pub requested_bytes: u64,
    pub currently_allocated_bytes: u64,
    #[allow(dead_code)]
    pub max_tensor_bytes: u64,
}

impl Default for GgmlBackendMemoryRequestV1 {
    fn default() -> Self {
        Self {
            struct_size: std::mem::size_of::<Self>() as u32,
            kind: 0,
            flags: 0,
            usage: 0,
            request_id: 0,
            backend: std::ptr::null_mut(),
            peer_backend: std::ptr::null_mut(),
            buft: std::ptr::null_mut(),
            graph: std::ptr::null_mut(),
            host_ptr: std::ptr::null(),
            requested_bytes: 0,
            currently_allocated_bytes: 0,
            max_tensor_bytes: 0,
        }
    }
}

/// Canonical identity of the domain a claim belongs to. `physical_device_uuid`
/// is a synthetic process-local identity derived from the backend-device
/// pointer for GPU domains (stock ggml exposes no hardware UUID), and all
/// zeroes for host/system domains. Never fabricated across processes.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub(crate) struct GgmlBackendMemoryDomainIdV1 {
    pub physical_device_uuid: [u8; 16],
    pub heap_index: u32,
    pub kind: u32,
}

#[derive(Clone, Copy, Default, Debug)]
pub(crate) struct GgmlBackendMemoryClaimV1 {
    pub struct_size: u32,
    pub flags: u32,
    pub request_id: u64,
    pub domain: GgmlBackendMemoryDomainIdV1,
    pub payload_requested_bytes: u64,
    pub committed_before_bytes: u64,
    #[allow(dead_code)]
    pub committed_after_upper_bytes: u64,
    pub commit_peak_extra_upper_bytes: u64,
    #[allow(dead_code)]
    pub resident_after_upper_bytes: u64,
    pub retained_after_use_upper_bytes: u64,
    #[allow(dead_code)]
    pub releasable_after_use_upper_bytes: u64,
}

#[derive(Clone, Copy, Default, Debug)]
pub(crate) struct GgmlBackendMemoryQuoteV1 {
    #[allow(dead_code)]
    pub struct_size: u32,
    pub flags: u32,
    pub residual_flags: u32,
    pub residual_request_count: u32,
    pub provisional_requested_upper_bytes: u64,
    pub stats_generation: u64,
    #[allow(dead_code)]
    pub quote_token: u64,
    pub request_fingerprint: u64,
}

#[derive(Clone, Copy, Default, Debug)]
pub(crate) struct GgmlBackendMemoryStatsV1 {
    pub struct_size: u32,
    pub flags: u32,
    pub domain: GgmlBackendMemoryDomainIdV1,
    pub generation: u64,
    #[allow(dead_code)]
    pub timestamp_monotonic_ns: u64,
    pub total_bytes: u64,
    pub budget_bytes: u64,
    pub device_used_bytes: u64,
    pub device_free_bytes: u64,
    pub backend_owned_live_bytes: u64,
    pub backend_owned_cached_bytes: u64,
    pub backend_owned_workspace_bytes: u64,
    pub backend_owned_high_water_bytes: u64,
    #[allow(dead_code)]
    pub allocation_count: u64,
    #[allow(dead_code)]
    pub allocation_failure_count: u64,
    pub health: u32,
    pub last_ggml_status: i32,
    pub last_native_error: i64,
    pub quarantine_generation: u64,
}

/// ggml abort predicate: return true to abort the in-flight graph. Called from
/// ggml worker threads -- must stay panic-free and lock-light.
pub(crate) type GgmlAbortCallback = Option<unsafe extern "C" fn(data: *mut c_void) -> bool>;

pub(crate) const GGML_TYPE_F32: c_int = 0;
pub(crate) const GGML_TYPE_F16: c_int = 1;
pub(crate) const GGML_TYPE_Q4_0: c_int = 2;
pub(crate) const GGML_TYPE_Q8_0: c_int = 8;
pub(crate) const GGML_TYPE_Q3_K: c_int = 11;
pub(crate) const GGML_TYPE_Q4_K: c_int = 12;
pub(crate) const GGML_TYPE_Q5_K: c_int = 13;
pub(crate) const GGML_TYPE_Q6_K: c_int = 14;
pub(crate) const GGML_TYPE_I32: c_int = 26;
pub(crate) const GGML_TYPE_Q2_0: c_int = 42;

/// Exclusive upper bound of `enum ggml_type` in vendored
/// `third_party/openasr-ggml/include/ggml.h` (`GGML_TYPE_COUNT = 44`,
/// `GGML_TYPE_F8_E4M3 = 43` added in v1.0.0).
/// ggml indexes `type_traits[type]` without a range check, so ids outside
/// `0..GGML_TYPE_COUNT` are an out-of-bounds read. Retired slots inside that
/// range (removed Q4_2/Q4_3, Q4_0_*, IQ4_NL_*) keep `blck_size == 0`;
/// NDEBUG `ggml_row_size` would then divide by zero.
///
/// When bumping the vendored ggml pin: copy the new `GGML_TYPE_COUNT` from
/// ggml.h, add any new live type constants, and re-run
/// `checked_ggml_type_matches_vendored_count` plus the retired-slot tests
/// (type ids 4 and 31).
/// Kept for pin documentation; stock ggml v1.0.0 adds GGML_TYPE_F8_E4M3.
#[allow(dead_code)]
pub(crate) const GGML_TYPE_F8_E4M3: c_int = 43;
pub(crate) const GGML_TYPE_COUNT: c_int = 44;

const _: () = assert!(GGML_TYPE_Q2_0 < GGML_TYPE_COUNT);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct InvalidGgmlType {
    pub raw: i64,
}

impl std::fmt::Display for InvalidGgmlType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ggml type {} is outside 0..{GGML_TYPE_COUNT} or is a retired slot",
            self.raw
        )
    }
}

/// Reject a raw ggml type id before any further `type_traits` query.
/// Retired in-range slots (`blck_size == 0`) fail the same way as OOB ids.
pub(crate) fn checked_ggml_type(raw: u32) -> Result<c_int, InvalidGgmlType> {
    checked_ggml_type_i64(i64::from(raw))
}

/// GGUF / host metadata store ggml types as `i32` (may be negative).
pub(crate) fn checked_ggml_type_i32(raw: i32) -> Result<c_int, InvalidGgmlType> {
    match u32::try_from(raw) {
        Ok(raw) => checked_ggml_type(raw),
        Err(_) => Err(InvalidGgmlType {
            raw: i64::from(raw),
        }),
    }
}

fn checked_ggml_type_i64(raw: i64) -> Result<c_int, InvalidGgmlType> {
    if raw < 0 || raw >= i64::from(GGML_TYPE_COUNT) {
        return Err(InvalidGgmlType { raw });
    }
    let ty = raw as c_int;
    // One FFI probe: retired ggml slots keep a type_traits entry with
    // blck_size == 0. Calling ggml_row_size on those is integer-divide-by-zero
    // under NDEBUG.
    if unsafe { ggml_blck_size(ty) } <= 0 {
        return Err(InvalidGgmlType { raw });
    }
    Ok(ty)
}

pub(crate) fn ggml_is_quantized_checked(raw: i32) -> Result<bool, InvalidGgmlType> {
    let ty = checked_ggml_type_i32(raw)?;
    Ok(unsafe { ggml_is_quantized(ty) })
}

pub(crate) fn ggml_blck_size_checked(raw: i32) -> Result<i64, InvalidGgmlType> {
    let ty = checked_ggml_type_i32(raw)?;
    Ok(unsafe { ggml_blck_size(ty) })
}

pub(crate) fn ggml_type_size_checked(raw: i32) -> Result<usize, InvalidGgmlType> {
    let ty = checked_ggml_type_i32(raw)?;
    Ok(unsafe { ggml_type_size(ty) })
}

pub(crate) fn ggml_row_size_checked(raw: i32, ne: i64) -> Result<usize, InvalidGgmlType> {
    let ty = checked_ggml_type_i32(raw)?;
    Ok(unsafe { ggml_row_size(ty, ne) })
}

pub(crate) fn ggml_get_type_traits_checked(
    raw: i32,
) -> Result<*const GgmlTypeTraits, InvalidGgmlType> {
    let ty = checked_ggml_type_i32(raw)?;
    Ok(unsafe { ggml_get_type_traits(ty) })
}

pub(crate) fn ggml_type_name_checked(raw: i32) -> Result<*const c_char, InvalidGgmlType> {
    let ty = checked_ggml_type_i32(raw)?;
    Ok(unsafe { ggml_type_name(ty) })
}

pub(crate) const GGML_LSTM_GATE_ORDER_IOFC: c_int = 0;
pub(crate) const GGML_LSTM_GATE_ORDER_IFGO: c_int = 1;

#[allow(dead_code)]
pub(crate) const GGML_ROPE_TYPE_NEOX: c_int = 2;

/// GPT-J / interleaved RoPE layout (rotates adjacent pairs x[2i], x[2i+1]).
/// Matches HuggingFace `repeat_interleave(2)` rotary embedding (e.g. Moonshine).
#[allow(dead_code)]
pub(crate) const GGML_ROPE_TYPE_NORMAL: c_int = 0;

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct GgmlTypeTraits {
    pub type_name: *const c_char,
    pub blck_size: i64,
    pub blck_size_interleave: i64,
    pub type_size: usize,
    pub is_quantized: bool,
    pub to_float: Option<GgmlToFloatFn>,
    pub from_float_ref: *const c_void,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct GgmlBackendDevCaps {
    pub async_: bool,
    pub host_buffer: bool,
    pub buffer_from_host_ptr: bool,
    pub events: bool,
    pub mmap_support: bool,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct GgmlBackendDevProps {
    pub name: *const c_char,
    pub description: *const c_char,
    pub memory_free: usize,
    pub memory_total: usize,
    pub type_: c_int,
    pub device_id: *const c_char,
    pub caps: GgmlBackendDevCaps,
}

pub(crate) const GGUF_TYPE_UINT32: c_int = 4;
pub(crate) const GGUF_TYPE_FLOAT32: c_int = 6;
pub(crate) const GGUF_TYPE_BOOL: c_int = 7;
pub(crate) const GGUF_TYPE_STRING: c_int = 8;
pub(crate) const GGUF_TYPE_ARRAY: c_int = 9;
pub(crate) const GGUF_TYPE_UINT64: c_int = 10;

/// Custom operator callback for stock ggml's `GGML_OP_CUSTOM`. Invoked from
/// ggml's CPU threadpool with `ith`/`nth` task indices; openasr kernels set
/// `n_tasks = 1`, so every callback runs once on the calling thread's ggml
/// executor.
pub(crate) type GgmlCustomOpFn =
    unsafe extern "C" fn(dst: GgmlTensorRaw, ith: c_int, nth: c_int, userdata: *mut c_void);

unsafe extern "C" {
    pub(crate) fn ggml_backend_name(backend: GgmlBackendRaw) -> *const c_char;
    pub(crate) fn ggml_backend_free(backend: GgmlBackendRaw);
    pub(crate) fn ggml_backend_buffer_free(buffer: GgmlBackendBufferRaw);
    pub(crate) fn ggml_backend_init_best() -> GgmlBackendRaw;
    pub(crate) fn ggml_backend_init_by_type(
        dev_type: c_int,
        params: *const c_char,
    ) -> GgmlBackendRaw;
    // Backend plugins are never loaded through the stock `ggml_backend_load*`
    // family from openasr: every module activation goes through the Rust-side
    // verified loader (ggml_runtime/verified_load.rs), which checks this
    // host's signed file identity against the signed catalog/bundle before
    // initialization and then registers the loaded module with the stock
    // registry. Scanning an application directory for `ggml-*.dll` remains
    // disabled: an unverified module must never activate.
    pub(crate) fn ggml_backend_register(reg: GgmlBackendRegRaw);
    pub(crate) fn ggml_backend_reg_get_proc_address(
        reg: GgmlBackendRegRaw,
        name: *const c_char,
    ) -> *mut c_void;
    pub(crate) fn ggml_backend_reg_name(reg: GgmlBackendRegRaw) -> *const c_char;
    pub(crate) fn ggml_backend_buffer_is_host(buffer: GgmlBackendBufferRaw) -> bool;
    pub(crate) fn ggml_backend_buffer_set_usage(buffer: GgmlBackendBufferRaw, usage: c_int);
    pub(crate) fn ggml_backend_buffer_get_usage(buffer: GgmlBackendBufferRaw) -> c_int;
    // Keep raw graph execution inside ggml_runtime. Model families must use
    // GgmlCpuGraphRunner so request cancellation and typed terminal-status
    // mapping cannot be bypassed as new executors are added.
    pub(super) fn ggml_backend_graph_compute(
        backend: GgmlBackendRaw,
        cgraph: GgmlCgraphRaw,
    ) -> c_int;
    pub(crate) fn ggml_backend_sched_new(
        backends: *mut GgmlBackendRaw,
        bufts: *mut GgmlBackendBufferTypeRaw,
        n_backends: c_int,
        graph_size: usize,
        parallel: bool,
        op_offload: bool,
    ) -> GgmlBackendSchedRaw;
    pub(crate) fn ggml_backend_sched_free(sched: GgmlBackendSchedRaw);
    pub(crate) fn ggml_backend_sched_reset(sched: GgmlBackendSchedRaw);
    pub(crate) fn ggml_backend_sched_alloc_graph(
        sched: GgmlBackendSchedRaw,
        graph: GgmlCgraphRaw,
    ) -> bool;
    pub(crate) fn ggml_backend_sched_get_tensor_backend(
        sched: GgmlBackendSchedRaw,
        node: GgmlTensorRaw,
    ) -> GgmlBackendRaw;
    pub(crate) fn ggml_backend_sched_graph_compute(
        sched: GgmlBackendSchedRaw,
        cgraph: GgmlCgraphRaw,
    ) -> c_int;
    pub(crate) fn ggml_backend_tensor_set(
        tensor: GgmlTensorRaw,
        data: *const c_void,
        offset: usize,
        size: usize,
    );
    pub(crate) fn ggml_backend_tensor_get(
        tensor: GgmlTensorRaw,
        data: *mut c_void,
        offset: usize,
        size: usize,
    );
    pub(crate) fn ggml_backend_tensor_alloc(
        buffer: GgmlBackendBufferRaw,
        tensor: GgmlTensorRaw,
        addr: *mut c_void,
    ) -> c_int;
    pub(crate) fn ggml_backend_alloc_ctx_tensors(
        ctx: GgmlContextRaw,
        backend: GgmlBackendRaw,
    ) -> GgmlBackendBufferRaw;
    // Read-only sizing query for the CPU step-buffer pool's grow-to-fit check:
    // computes what `ggml_backend_alloc_ctx_tensors_from_buft` would allocate
    // for `ctx`'s currently-unallocated tensors without allocating anything.
    pub(crate) fn ggml_backend_alloc_ctx_tensors_from_buft_size(
        ctx: GgmlContextRaw,
        buft: GgmlBackendBufferTypeRaw,
    ) -> usize;
    pub(crate) fn ggml_backend_get_default_buffer_type(
        backend: GgmlBackendRaw,
    ) -> GgmlBackendBufferTypeRaw;
    pub(crate) fn ggml_backend_buft_alloc_buffer(
        buft: GgmlBackendBufferTypeRaw,
        size: usize,
    ) -> GgmlBackendBufferRaw;
    pub(crate) fn ggml_backend_buffer_get_size(buffer: GgmlBackendBufferRaw) -> usize;
    pub(crate) fn ggml_gallocr_new(buft: GgmlBackendBufferTypeRaw) -> GgmlGallocrRaw;
    pub(crate) fn ggml_gallocr_free(galloc: GgmlGallocrRaw);
    // Pre-allocate buffers for a worst-case graph without mutating it. This is
    // the stock-ggml measurement step: it computes the buffer sizes a later
    // allocation would use.
    pub(crate) fn ggml_gallocr_reserve_n(
        galloc: GgmlGallocrRaw,
        graph: GgmlCgraphRaw,
        node_buffer_ids: *const c_int,
        leaf_buffer_ids: *const c_int,
    ) -> bool;
    pub(crate) fn ggml_gallocr_reserve_n_size(
        galloc: GgmlGallocrRaw,
        graph: GgmlCgraphRaw,
        node_buffer_ids: *const c_int,
        leaf_buffer_ids: *const c_int,
        sizes: *mut usize,
    );
    pub(crate) fn ggml_gallocr_alloc_graph(galloc: GgmlGallocrRaw, graph: GgmlCgraphRaw) -> bool;
    // Tensor allocator (ggml-alloc.h): binds tensors into an already-allocated
    // buffer -- the primitive `ggml_backend_alloc_ctx_tensors` itself uses,
    // exposed here to bind the CPU step pool's *reused* buffer without
    // allocating a fresh one every step.
    pub(crate) fn ggml_tallocr_new(buffer: GgmlBackendBufferRaw) -> GgmlTallocr;
    pub(crate) fn ggml_tallocr_alloc(talloc: *mut GgmlTallocr, tensor: GgmlTensorRaw) -> c_int;
    pub(crate) fn ggml_backend_view_init(tensor: GgmlTensorRaw) -> c_int;
    pub(crate) fn ggml_backend_get_device(backend: GgmlBackendRaw) -> GgmlBackendDevRaw;

    pub(crate) fn ggml_backend_dev_count() -> usize;
    pub(crate) fn ggml_backend_dev_get(index: usize) -> GgmlBackendDevRaw;
    pub(crate) fn ggml_backend_dev_name(device: GgmlBackendDevRaw) -> *const c_char;
    pub(crate) fn ggml_backend_dev_description(device: GgmlBackendDevRaw) -> *const c_char;
    pub(crate) fn ggml_backend_dev_type(device: GgmlBackendDevRaw) -> c_int;
    pub(crate) fn ggml_backend_dev_memory(
        device: GgmlBackendDevRaw,
        free: *mut usize,
        total: *mut usize,
    );
    pub(crate) fn ggml_backend_dev_backend_reg(device: GgmlBackendDevRaw) -> GgmlBackendRegRaw;
    pub(crate) fn ggml_backend_dev_get_props(
        device: GgmlBackendDevRaw,
        props: *mut GgmlBackendDevProps,
    );
    pub(crate) fn ggml_backend_dev_buffer_type(
        device: GgmlBackendDevRaw,
    ) -> GgmlBackendBufferTypeRaw;
    pub(crate) fn ggml_backend_buft_get_alignment(buft: GgmlBackendBufferTypeRaw) -> usize;
    pub(crate) fn ggml_backend_dev_init(
        device: GgmlBackendDevRaw,
        params: *const c_char,
    ) -> GgmlBackendRaw;
    pub(crate) fn ggml_backend_dev_buffer_from_host_ptr(
        device: GgmlBackendDevRaw,
        ptr: *mut c_void,
        size: usize,
        max_tensor_size: usize,
    ) -> GgmlBackendBufferRaw;
    pub(crate) fn ggml_backend_dev_supports_op(
        device: GgmlBackendDevRaw,
        op: GgmlTensorRaw,
    ) -> bool;

    // The host sets CPU threads through the registry proc-address table
    // (`backend_set_n_threads`), which works under GGML_BACKEND_DL where the
    // ggml-cpu plugin's symbols are not linked into the core. The macOS
    // BLAS-accelerator path calls ggml_backend_blas_set_n_threads directly.
    #[cfg(target_os = "macos")]
    pub(crate) fn ggml_backend_blas_init() -> GgmlBackendRaw;
    #[cfg(target_os = "macos")]
    pub(crate) fn ggml_backend_blas_set_n_threads(backend: GgmlBackendRaw, n_threads: c_int);
    pub(crate) fn ggml_init(params: GgmlInitParams) -> GgmlContextRaw;
    pub(crate) fn ggml_reset(ctx: GgmlContextRaw);
    pub(crate) fn ggml_free(ctx: GgmlContextRaw);
    pub(crate) fn ggml_blck_size(type_: c_int) -> i64;
    pub(crate) fn ggml_type_size(type_: c_int) -> usize;
    pub(crate) fn ggml_row_size(type_: c_int, ne: i64) -> usize;
    #[allow(dead_code)]
    pub(crate) fn ggml_is_quantized(type_: c_int) -> bool;
    pub(crate) fn ggml_get_type_traits(type_: c_int) -> *const GgmlTypeTraits;
    pub(crate) fn ggml_quantize_chunk(
        type_: c_int,
        src: *const f32,
        dst: *mut c_void,
        start: i64,
        nrows: i64,
        n_per_row: i64,
        imatrix: *const f32,
    ) -> usize;
    pub(crate) fn ggml_get_data(tensor: GgmlTensorRaw) -> *mut c_void;
    pub(crate) fn ggml_get_first_tensor(ctx: GgmlContextRaw) -> GgmlTensorRaw;
    pub(crate) fn ggml_get_next_tensor(ctx: GgmlContextRaw, tensor: GgmlTensorRaw)
    -> GgmlTensorRaw;
    pub(crate) fn ggml_get_name(tensor: GgmlTensorRaw) -> *const c_char;
    pub(crate) fn ggml_op_desc(tensor: GgmlTensorRaw) -> *const c_char;
    pub(crate) fn ggml_nbytes(tensor: GgmlTensorRaw) -> usize;
    pub(crate) fn ggml_new_tensor_1d(ctx: GgmlContextRaw, type_: c_int, ne0: i64) -> GgmlTensorRaw;
    pub(crate) fn ggml_new_tensor_2d(
        ctx: GgmlContextRaw,
        type_: c_int,
        ne0: i64,
        ne1: i64,
    ) -> GgmlTensorRaw;
    pub(crate) fn ggml_new_tensor_3d(
        ctx: GgmlContextRaw,
        type_: c_int,
        ne0: i64,
        ne1: i64,
        ne2: i64,
    ) -> GgmlTensorRaw;
    pub(crate) fn ggml_new_tensor_4d(
        ctx: GgmlContextRaw,
        type_: c_int,
        ne0: i64,
        ne1: i64,
        ne2: i64,
        ne3: i64,
    ) -> GgmlTensorRaw;
    pub(crate) fn ggml_set_name(tensor: GgmlTensorRaw, name: *const c_char) -> GgmlTensorRaw;
    pub(crate) fn ggml_add(
        ctx: GgmlContextRaw,
        a: GgmlTensorRaw,
        b: GgmlTensorRaw,
    ) -> GgmlTensorRaw;
    pub(crate) fn ggml_sub(
        ctx: GgmlContextRaw,
        a: GgmlTensorRaw,
        b: GgmlTensorRaw,
    ) -> GgmlTensorRaw;
    pub(crate) fn ggml_mul(
        ctx: GgmlContextRaw,
        a: GgmlTensorRaw,
        b: GgmlTensorRaw,
    ) -> GgmlTensorRaw;
    pub(crate) fn ggml_div(
        ctx: GgmlContextRaw,
        a: GgmlTensorRaw,
        b: GgmlTensorRaw,
    ) -> GgmlTensorRaw;
    pub(crate) fn ggml_sqr(ctx: GgmlContextRaw, a: GgmlTensorRaw) -> GgmlTensorRaw;
    pub(crate) fn ggml_sqrt(ctx: GgmlContextRaw, a: GgmlTensorRaw) -> GgmlTensorRaw;
    pub(crate) fn ggml_abs(ctx: GgmlContextRaw, a: GgmlTensorRaw) -> GgmlTensorRaw;
    pub(crate) fn ggml_log(ctx: GgmlContextRaw, a: GgmlTensorRaw) -> GgmlTensorRaw;
    pub(crate) fn ggml_sin(ctx: GgmlContextRaw, a: GgmlTensorRaw) -> GgmlTensorRaw;
    pub(crate) fn ggml_cos(ctx: GgmlContextRaw, a: GgmlTensorRaw) -> GgmlTensorRaw;
    pub(crate) fn ggml_leaky_relu(
        ctx: GgmlContextRaw,
        a: GgmlTensorRaw,
        negative_slope: f32,
        inplace: bool,
    ) -> GgmlTensorRaw;
    pub(crate) fn ggml_mul_mat(
        ctx: GgmlContextRaw,
        a: GgmlTensorRaw,
        b: GgmlTensorRaw,
    ) -> GgmlTensorRaw;
    // Sets the accumulator precision (op_params[0] for MUL_MAT, op_params[3] for
    // FLASH_ATTN_EXT). Returns false for ops that carry no precision.
    pub(crate) fn ggml_prec_set_acc(a: GgmlTensorRaw, prec: c_int) -> bool;
    pub(crate) fn ggml_get_rows(
        ctx: GgmlContextRaw,
        a: GgmlTensorRaw,
        b: GgmlTensorRaw,
    ) -> GgmlTensorRaw;
    #[allow(dead_code)] // last-max binding; unused until a native last-max lane is authorized
    pub(crate) fn ggml_argmax(ctx: GgmlContextRaw, a: GgmlTensorRaw) -> GgmlTensorRaw;
    // Custom operator support (stock ggml `GGML_OP_CUSTOM`). OpenASR's
    // first-max argmax, Transformer-XL relative-position attention, LSTM scan,
    // and swoosh activation are implemented as custom operators with Rust
    // callbacks in `custom_ops.rs`; stock ggml has no native kernels for them.
    pub(crate) fn ggml_custom_4d(
        ctx: GgmlContextRaw,
        type_: c_int,
        ne0: i64,
        ne1: i64,
        ne2: i64,
        ne3: i64,
        args: *mut GgmlTensorRaw,
        n_args: c_int,
        fun: GgmlCustomOpFn,
        n_tasks: c_int,
        userdata: *mut c_void,
    ) -> GgmlTensorRaw;
    #[cfg(test)]
    pub(crate) fn ggml_top_k(ctx: GgmlContextRaw, a: GgmlTensorRaw, k: c_int) -> GgmlTensorRaw;
    pub(crate) fn ggml_scale(ctx: GgmlContextRaw, a: GgmlTensorRaw, s: f32) -> GgmlTensorRaw;
    pub(crate) fn ggml_sum(ctx: GgmlContextRaw, a: GgmlTensorRaw) -> GgmlTensorRaw;
    pub(crate) fn ggml_sum_rows(ctx: GgmlContextRaw, a: GgmlTensorRaw) -> GgmlTensorRaw;
    pub(crate) fn ggml_mean(ctx: GgmlContextRaw, a: GgmlTensorRaw) -> GgmlTensorRaw;
    pub(crate) fn ggml_norm(ctx: GgmlContextRaw, a: GgmlTensorRaw, eps: f32) -> GgmlTensorRaw;
    // group normalize along ne0*ne1*n_groups (ggml.h:1382). wav2vec2 base uses
    // feat_extract_norm=="group" with n_groups == n_channels (per-channel norm).
    pub(crate) fn ggml_group_norm(
        ctx: GgmlContextRaw,
        a: GgmlTensorRaw,
        n_groups: c_int,
        eps: f32,
    ) -> GgmlTensorRaw;
    // concat a and b along `dim` (ggml.h:1084). Used to stitch per-group conv_1d
    // outputs back into one [out_channels, T] tensor for the grouped pos-conv.
    pub(crate) fn ggml_concat(
        ctx: GgmlContextRaw,
        a: GgmlTensorRaw,
        b: GgmlTensorRaw,
        dim: c_int,
    ) -> GgmlTensorRaw;
    #[allow(dead_code)]
    pub(crate) fn ggml_rms_norm(ctx: GgmlContextRaw, a: GgmlTensorRaw, eps: f32) -> GgmlTensorRaw;
    #[allow(dead_code)]
    pub(crate) fn ggml_repeat(
        ctx: GgmlContextRaw,
        a: GgmlTensorRaw,
        b: GgmlTensorRaw,
    ) -> GgmlTensorRaw;
    #[allow(dead_code)]
    pub(crate) fn ggml_repeat_4d(
        ctx: GgmlContextRaw,
        a: GgmlTensorRaw,
        ne0: i64,
        ne1: i64,
        ne2: i64,
        ne3: i64,
    ) -> GgmlTensorRaw;
    pub(crate) fn ggml_soft_max(ctx: GgmlContextRaw, a: GgmlTensorRaw) -> GgmlTensorRaw;
    pub(crate) fn ggml_soft_max_ext(
        ctx: GgmlContextRaw,
        a: GgmlTensorRaw,
        mask: GgmlTensorRaw,
        scale: f32,
        max_bias: f32,
    ) -> GgmlTensorRaw;
    pub(crate) fn ggml_flash_attn_ext(
        ctx: GgmlContextRaw,
        q: GgmlTensorRaw,
        k: GgmlTensorRaw,
        v: GgmlTensorRaw,
        mask: GgmlTensorRaw,
        scale: f32,
        max_bias: f32,
        logit_softcap: f32,
    ) -> GgmlTensorRaw;
    #[allow(dead_code)]
    pub(crate) fn ggml_ssm_conv(
        ctx: GgmlContextRaw,
        sx: GgmlTensorRaw,
        c: GgmlTensorRaw,
    ) -> GgmlTensorRaw;
    pub(crate) fn ggml_can_repeat(a: GgmlTensorRaw, b: GgmlTensorRaw) -> bool;
    pub(crate) fn ggml_gelu(ctx: GgmlContextRaw, a: GgmlTensorRaw) -> GgmlTensorRaw;
    pub(crate) fn ggml_gelu_erf(ctx: GgmlContextRaw, a: GgmlTensorRaw) -> GgmlTensorRaw;
    pub(crate) fn ggml_tanh(ctx: GgmlContextRaw, a: GgmlTensorRaw) -> GgmlTensorRaw;
    pub(crate) fn ggml_relu(ctx: GgmlContextRaw, a: GgmlTensorRaw) -> GgmlTensorRaw;
    pub(crate) fn ggml_sigmoid(ctx: GgmlContextRaw, a: GgmlTensorRaw) -> GgmlTensorRaw;
    pub(crate) fn ggml_softplus(ctx: GgmlContextRaw, a: GgmlTensorRaw) -> GgmlTensorRaw;
    pub(crate) fn ggml_exp(ctx: GgmlContextRaw, a: GgmlTensorRaw) -> GgmlTensorRaw;
    #[allow(dead_code)]
    pub(crate) fn ggml_silu(ctx: GgmlContextRaw, a: GgmlTensorRaw) -> GgmlTensorRaw;
    #[allow(dead_code)]
    pub(crate) fn ggml_cast(ctx: GgmlContextRaw, a: GgmlTensorRaw, type_: c_int) -> GgmlTensorRaw;
    pub(crate) fn ggml_cont(ctx: GgmlContextRaw, a: GgmlTensorRaw) -> GgmlTensorRaw;
    pub(crate) fn ggml_nelements(tensor: GgmlTensorRaw) -> i64;
    pub(crate) fn ggml_is_transposed(tensor: GgmlTensorRaw) -> bool;
    pub(crate) fn ggml_is_contiguous(tensor: GgmlTensorRaw) -> bool;
    pub(crate) fn ggml_is_contiguous_rows(tensor: GgmlTensorRaw) -> bool;
    pub(crate) fn ggml_reshape_2d(
        ctx: GgmlContextRaw,
        a: GgmlTensorRaw,
        ne0: i64,
        ne1: i64,
    ) -> GgmlTensorRaw;
    pub(crate) fn ggml_reshape_1d(ctx: GgmlContextRaw, a: GgmlTensorRaw, ne0: i64)
    -> GgmlTensorRaw;
    pub(crate) fn ggml_reshape_3d(
        ctx: GgmlContextRaw,
        a: GgmlTensorRaw,
        ne0: i64,
        ne1: i64,
        ne2: i64,
    ) -> GgmlTensorRaw;
    pub(crate) fn ggml_reshape_4d(
        ctx: GgmlContextRaw,
        a: GgmlTensorRaw,
        ne0: i64,
        ne1: i64,
        ne2: i64,
        ne3: i64,
    ) -> GgmlTensorRaw;
    pub(crate) fn ggml_view_2d(
        ctx: GgmlContextRaw,
        a: GgmlTensorRaw,
        ne0: i64,
        ne1: i64,
        nb1: usize,
        offset: usize,
    ) -> GgmlTensorRaw;
    pub(crate) fn ggml_view_1d(
        ctx: GgmlContextRaw,
        a: GgmlTensorRaw,
        ne0: i64,
        offset: usize,
    ) -> GgmlTensorRaw;
    pub(crate) fn ggml_view_3d(
        ctx: GgmlContextRaw,
        a: GgmlTensorRaw,
        ne0: i64,
        ne1: i64,
        ne2: i64,
        nb1: usize,
        nb2: usize,
        offset: usize,
    ) -> GgmlTensorRaw;
    #[allow(dead_code)]
    pub(crate) fn ggml_view_4d(
        ctx: GgmlContextRaw,
        a: GgmlTensorRaw,
        ne0: i64,
        ne1: i64,
        ne2: i64,
        ne3: i64,
        nb1: usize,
        nb2: usize,
        nb3: usize,
        offset: usize,
    ) -> GgmlTensorRaw;
    pub(crate) fn ggml_cpy(
        ctx: GgmlContextRaw,
        a: GgmlTensorRaw,
        b: GgmlTensorRaw,
    ) -> GgmlTensorRaw;
    #[allow(dead_code)]
    pub(crate) fn ggml_set_rows(
        ctx: GgmlContextRaw,
        a: GgmlTensorRaw,
        b: GgmlTensorRaw,
        c: GgmlTensorRaw,
    ) -> GgmlTensorRaw;
    pub(crate) fn ggml_transpose(ctx: GgmlContextRaw, a: GgmlTensorRaw) -> GgmlTensorRaw;
    pub(crate) fn ggml_permute(
        ctx: GgmlContextRaw,
        a: GgmlTensorRaw,
        axis0: c_int,
        axis1: c_int,
        axis2: c_int,
        axis3: c_int,
    ) -> GgmlTensorRaw;
    pub(crate) fn ggml_conv_1d(
        ctx: GgmlContextRaw,
        a: GgmlTensorRaw,
        b: GgmlTensorRaw,
        s0: c_int,
        p0: c_int,
        d0: c_int,
    ) -> GgmlTensorRaw;
    pub(crate) fn ggml_pool_1d(
        ctx: GgmlContextRaw,
        a: GgmlTensorRaw,
        op: c_int,
        k0: c_int,
        s0: c_int,
        p0: c_int,
    ) -> GgmlTensorRaw;
    pub(crate) fn ggml_conv_2d(
        ctx: GgmlContextRaw,
        a: GgmlTensorRaw,
        b: GgmlTensorRaw,
        s0: c_int,
        s1: c_int,
        p0: c_int,
        p1: c_int,
        d0: c_int,
        d1: c_int,
    ) -> GgmlTensorRaw;
    pub(crate) fn ggml_conv_2d_direct(
        ctx: GgmlContextRaw,
        a: GgmlTensorRaw,
        b: GgmlTensorRaw,
        s0: c_int,
        s1: c_int,
        p0: c_int,
        p1: c_int,
        d0: c_int,
        d1: c_int,
    ) -> GgmlTensorRaw;
    pub(crate) fn ggml_conv_2d_dw_direct(
        ctx: GgmlContextRaw,
        a: GgmlTensorRaw,
        b: GgmlTensorRaw,
        stride0: c_int,
        stride1: c_int,
        pad0: c_int,
        pad1: c_int,
        dilation0: c_int,
        dilation1: c_int,
    ) -> GgmlTensorRaw;
    #[allow(dead_code)]
    pub(crate) fn ggml_rope_ext(
        ctx: GgmlContextRaw,
        a: GgmlTensorRaw,
        b: GgmlTensorRaw,
        c: GgmlTensorRaw,
        n_dims: c_int,
        mode: c_int,
        n_ctx_orig: c_int,
        freq_base: f32,
        freq_scale: f32,
        ext_factor: f32,
        attn_factor: f32,
        beta_fast: f32,
        beta_slow: f32,
    ) -> GgmlTensorRaw;
    pub(crate) fn ggml_new_graph_custom(
        ctx: GgmlContextRaw,
        size: usize,
        grads: bool,
    ) -> GgmlCgraphRaw;
    pub(crate) fn ggml_build_forward_expand(cgraph: GgmlCgraphRaw, tensor: GgmlTensorRaw);
    pub(crate) fn ggml_graph_n_nodes(cgraph: GgmlCgraphRaw) -> c_int;
    pub(crate) fn ggml_graph_node(cgraph: GgmlCgraphRaw, index: c_int) -> GgmlTensorRaw;
    pub(crate) fn ggml_set_input(tensor: GgmlTensorRaw);
    pub(crate) fn ggml_set_output(tensor: GgmlTensorRaw);
    // Sizing helpers for no_alloc metadata contexts. `ggml_tensor_overhead`
    // returns the per-tensor bookkeeping cost (ggml_object + ggml_tensor) and
    // `ggml_graph_overhead_custom` the bytes a cgraph of `size` nodes consumes
    // inside its context — together they give the EXACT capacity a metadata-only
    // context needs, mirroring llama.cpp's compute-meta buffer sizing.
    pub(crate) fn ggml_tensor_overhead() -> usize;
    pub(crate) fn ggml_graph_overhead_custom(size: usize, grads: bool) -> usize;
    pub(crate) fn ggml_used_mem(ctx: GgmlContextRaw) -> usize;
    pub(crate) fn ggml_get_mem_size(ctx: GgmlContextRaw) -> usize;
    // ggml_cpu_has_* / ggml_cpu_get_* are not declared here: under
    // GGML_BACKEND_DL they live in the loaded ggml-cpu plugin, not the linked
    // core. GgmlCpuFeatures::detect() reads CPU features via the Rust stdlib
    // instead (build-mode-agnostic).

    pub(crate) fn gguf_init_from_buffer(
        data: *const c_void,
        size: usize,
        params: GgufInitParams,
    ) -> GgufContextRaw;
    pub(crate) fn gguf_free(ctx: GgufContextRaw);
    pub(crate) fn gguf_get_n_kv(ctx: *const c_void) -> i64;
    pub(crate) fn gguf_get_key(ctx: *const c_void, key_id: i64) -> *const c_char;
    pub(crate) fn gguf_get_kv_type(ctx: *const c_void, key_id: i64) -> c_int;
    pub(crate) fn gguf_get_arr_type(ctx: *const c_void, key_id: i64) -> c_int;
    pub(crate) fn gguf_get_val_u32(ctx: *const c_void, key_id: i64) -> u32;
    pub(crate) fn gguf_get_val_u64(ctx: *const c_void, key_id: i64) -> u64;
    pub(crate) fn gguf_get_val_f32(ctx: *const c_void, key_id: i64) -> f32;
    pub(crate) fn gguf_get_val_bool(ctx: *const c_void, key_id: i64) -> bool;
    pub(crate) fn gguf_get_val_str(ctx: *const c_void, key_id: i64) -> *const c_char;
    pub(crate) fn gguf_get_arr_n(ctx: *const c_void, key_id: i64) -> usize;
    pub(crate) fn gguf_get_arr_data(ctx: *const c_void, key_id: i64) -> *const c_void;
    pub(crate) fn gguf_get_arr_str(ctx: *const c_void, key_id: i64, i: usize) -> *const c_char;
    pub(crate) fn gguf_get_data_offset(ctx: *const c_void) -> usize;
    pub(crate) fn gguf_get_n_tensors(ctx: *const c_void) -> i64;
    pub(crate) fn gguf_get_tensor_offset(ctx: *const c_void, tensor_id: i64) -> usize;
    pub(crate) fn gguf_get_tensor_name(ctx: *const c_void, tensor_id: i64) -> *const c_char;
    pub(crate) fn gguf_get_tensor_ne(ctx: *const c_void, tensor_id: i64) -> *const i64;
    pub(crate) fn gguf_get_tensor_type(ctx: *const c_void, tensor_id: i64) -> c_int;
    pub(crate) fn gguf_get_tensor_size(ctx: *const c_void, tensor_id: i64) -> usize;
    pub(crate) fn ggml_type_name(type_: c_int) -> *const c_char;

    pub(crate) fn gguf_init_empty() -> GgufContextRaw;
    pub(crate) fn gguf_set_val_u32(ctx: GgufContextRaw, key: *const c_char, val: u32);
    pub(crate) fn gguf_set_val_u64(ctx: GgufContextRaw, key: *const c_char, val: u64);
    pub(crate) fn gguf_set_val_f32(ctx: GgufContextRaw, key: *const c_char, val: f32);
    pub(crate) fn gguf_set_val_bool(ctx: GgufContextRaw, key: *const c_char, val: bool);
    pub(crate) fn gguf_set_val_str(ctx: GgufContextRaw, key: *const c_char, val: *const c_char);
    pub(crate) fn gguf_set_arr_data(
        ctx: GgufContextRaw,
        key: *const c_char,
        type_: c_int,
        data: *const c_void,
        n: usize,
    );
    pub(crate) fn gguf_set_arr_str(
        ctx: GgufContextRaw,
        key: *const c_char,
        data: *const *const c_char,
        n: usize,
    );
    pub(crate) fn gguf_add_tensor(ctx: GgufContextRaw, tensor: GgmlTensorRaw);
    pub(crate) fn gguf_set_tensor_type(ctx: GgufContextRaw, name: *const c_char, type_: c_int);
    pub(crate) fn gguf_set_tensor_data(
        ctx: GgufContextRaw,
        name: *const c_char,
        data: *const c_void,
    );
    pub(crate) fn gguf_write_to_file(
        ctx: *const c_void,
        fname: *const c_char,
        only_meta: bool,
    ) -> bool;

    #[cfg(target_os = "macos")]
    pub(crate) fn ggml_backend_metal_init() -> GgmlBackendRaw;
    #[cfg(target_os = "macos")]
    pub(crate) fn ggml_backend_metal_set_abort_callback(
        backend: GgmlBackendRaw,
        abort_callback: GgmlAbortCallback,
        abort_callback_data: *mut c_void,
    );
}

#[cfg(test)]
mod ggml_type_tests {
    use std::ffi::CStr;

    use super::{
        GGML_TYPE_COUNT, InvalidGgmlType, checked_ggml_type, checked_ggml_type_i32, ggml_type_name,
    };

    #[test]
    fn checked_ggml_type_accepts_count_minus_one() {
        assert_eq!(
            checked_ggml_type((GGML_TYPE_COUNT - 1) as u32),
            Ok(GGML_TYPE_COUNT - 1)
        );
        assert_eq!(
            checked_ggml_type_i32(GGML_TYPE_COUNT - 1),
            Ok(GGML_TYPE_COUNT - 1)
        );
    }

    #[test]
    fn checked_ggml_type_rejects_count_and_u32_max() {
        assert_eq!(
            checked_ggml_type(GGML_TYPE_COUNT as u32),
            Err(InvalidGgmlType {
                raw: i64::from(GGML_TYPE_COUNT)
            })
        );
        assert_eq!(
            checked_ggml_type(u32::MAX),
            Err(InvalidGgmlType {
                raw: i64::from(u32::MAX)
            })
        );
        assert_eq!(
            checked_ggml_type_i32(GGML_TYPE_COUNT),
            Err(InvalidGgmlType {
                raw: i64::from(GGML_TYPE_COUNT)
            })
        );
        assert_eq!(checked_ggml_type_i32(-1), Err(InvalidGgmlType { raw: -1 }));
    }

    #[test]
    fn checked_ggml_type_rejects_retired_slots() {
        assert_eq!(checked_ggml_type_i32(4), Err(InvalidGgmlType { raw: 4 }));
        assert_eq!(checked_ggml_type_i32(31), Err(InvalidGgmlType { raw: 31 }));
        assert!(checked_ggml_type(4).is_err());
        assert!(checked_ggml_type(31).is_err());
    }

    #[test]
    fn checked_ggml_type_matches_vendored_count() {
        let name_ptr = unsafe { ggml_type_name(GGML_TYPE_COUNT - 1) };
        assert!(
            !name_ptr.is_null(),
            "ggml_type_name(COUNT-1) must name a live type"
        );
        let name = unsafe { CStr::from_ptr(name_ptr) }.to_string_lossy();
        let upper = name.to_ascii_uppercase();
        assert!(
            !upper.contains("REMOVED") && !upper.contains("DEPRECATED"),
            "ggml_type_name(COUNT-1) was {name:?}"
        );
    }
}
