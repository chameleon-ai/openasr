# Stock ggml 1.0.0 migration (experimental branch)

Status: **experimental**. This branch moves the vendored ggml submodule from the
OpenASR fork (`chameleon-ai/ggml` @ `6775e391`, based on upstream v0.17.0 + 37
fork commits) to stock upstream **v1.0.0** (`ggml-org/ggml`, tag `v1.0.0` =
`417a024b`). It is a feasibility exercise, not a release candidate. See
`ggml-migration-report.md` (repo root) for the pre-migration analysis this
change implements.

## What changed

### Submodule and build

- `.gitmodules` now points at `https://github.com/ggml-org/ggml.git`; the
  submodule gitlink moves from fork `6775e391` to stock tag `v1.0.0`.
- `build.rs` no longer passes the fork-only CMake flags
  `OPENASR_VERIFIED_BACKEND_LOADING_ONLY` and `OPENASR_BACKEND_ABI_V1`. Stock
  ggml has no code to honor them. The `windows_cmake_build_contract` cache
  expectations and `tooling/release-manifest/backend_catalog.py` (and its test)
  drop the matching required entry. The `verified_backend_loading_only: true`
  build-manifest flag is kept: it now describes the Rust-side activation
  policy below, not a CMake define.

### Trust boundary, reimplemented

Stock ggml has no verified-load API and no physical-memory admission ABI.
Both were rebuilt in Rust:

- **Verified backend loading** (`ggml_runtime/verified_load.rs`). Modules are
  loaded by openasr itself (restricted search path: plugin directory +
  verified dependency dirs + System32/default loader path; `dlopen`/`LoadLibraryExW`),
  their `ggml_backend_init` is called, the registry name must match the
  expected provider, and the module is registered with the stock registry via
  `ggml_backend_register`. The pre-load file-identity checks are unchanged:
  bundled CPU rescue modules are matched against the signed bundle manifest
  (full SHA-256 + PE image identity), and optional accelerator packs against
  the signed catalog entry, the installed-pack record, and a locked, re-hashed
  file inventory. Stock `ggml_backend_load*` is never used by openasr.
  - **Live qualification probes are unavailable.** Stock ggml has no
    attestation symbol for the physical target (sm_86) or the driver version,
    so `probe_exact_backend_plugin_candidate` /
    `probe_backend_plugin_identity_candidate` always return
    `LiveProbeFailed`. Fresh optional GPU packs therefore cannot be qualified
    or activated on this branch until a hardware attestation story is
    designed. Already-persisted exact activation records still activate
    because their target/driver binding is carried in the record. GPU users
    must build static GPU hosts (`--features cuda|vulkan|hip|metal`); those
    need no DL plugin at all.
- **Memory admission** (`backend_memory.rs`). The fork's domain/UUID/quota
  ABI is gone. The admission broker now consumes a Rust-only ABI: one domain
  per device (host for CPU-class, device-local for GPU-class, synthetic
  process-local identity derived from the device pointer — never a fabricated
  hardware UUID), quotes are committed upper bounds reconciled against fresh
  `ggml_backend_dev_memory` free/total after commit, `reserve_private` is a
  token-validated no-op, and stats are always present (zeroed with
  `BUDGET_UNAVAILABLE` when the device cannot report, i.e. fail-closed).
  Scheduler plans measure the graph footprint with a throwaway
  `ggml_gallocr` on the primary backend's buffer type
  (`ggml_gallocr_reserve_n_size`, exact for the single-backend case; a
  conservative sum-of-node-tensor-bytes for multi-backend) and commit
  through `ggml_backend_sched_alloc_graph` with `sched_reset` on failure.

  Design note: the scheduler's own `ggml_gallocr` must NOT be used for
  measurement. Stock `ggml_backend_sched_reserve_size` sets
  `galloc->n_nodes` without allocating buffers, so a subsequent
  `ggml_backend_sched_alloc_graph` believes the graph is already reserved,
  skips allocation, and dereferences a null buffer (SIGSEGV in
  `ggml_backend_sched_alloc_graph`). The throwaway-gallocr measurement
  avoids leaving the scheduler in that state.
- **Backend lifecycle observability** (`backend_graph_lifecycle.rs`). Stock
  ggml has no captured-graph/capture-executable observation hook. The
  binding reports `Unavailable` everywhere; Rust-side lifecycle receipts
  (graph ids, generations, compute counts) are unchanged, and no capture
  evidence is fabricated. Lifecycle JSON receipts keep their shape; capture
  fields are simply absent.
- **Cooperative cancellation** (`cpu_graph.rs`). Stock ggml dispatches a
  per-backend abort hook from the registry proc table
  (`ggml_backend_set_abort_callback`, or the Metal direct symbol on macOS).
  The job's cancel flag is installed on every participating scheduler backend
  that supports it before each compute and cleared afterwards. Backends
  without a native hook (stock CUDA/Vulkan/HIP register none) run in degraded
  mode: the graph runs to completion, cancel is still honored before the
  compute starts and after it returns (so no readback follows an aborted job),
  and any participant that did take the hook (e.g. the CPU fallback in a
  scheduler) keeps node-level abort. A participant reporting a fatal lookup
  failure (no device or registry, not merely a missing hook) fails the compute
  closed. CPU cancellation is per-node; Metal cancellation works through the
  Metal hook; CUDA/Vulkan/HIP cancellation is job/boundary-scoped only (no
  native hook exists in stock v1.0.0), matching llama.cpp's non-cooperative
  GPU path — this replaces the earlier fail-closed-on-absent-hook behavior,
  which broke every server-side GPU job carrying a cancellation flag (e.g.
  whisper encoder prelude on HIP).

### Graph operators (stock GGML_OP_CUSTOM)

`ggml_argmax_first`, `ggml_flash_attn_rel_pos`, `ggml_lstm_seq`, and
`ggml_swoosh` were fork-only kernels. Stock ggml has no equivalents. They are
reimplemented as custom operators (`ggml_custom_4d`) with Rust callbacks in
`ggml_runtime/custom_ops.rs` (CPU backend only, `n_tasks = 1`):

- `ggml_argmax_first` (i32 out; first index of the max per row; any non-finite
  entry forces -1);
- `ggml_swoosh` (element-wise `softplus(x - offset) - linear_scale*x - shift`);
- `ggml_lstm_seq` (full scan, both gate orders, forward/reverse, optional
  concatenated recurrent bias; no workspace tensor needed);
- `ggml_flash_attn_rel_pos` (Transformer-XL AC+BD relative attention with
  online softmax; CPU-only, unchanged policy).

Custom ops are only executable by CPU-class devices in stock ggml, so the
builders fail closed at graph-build time when there is neither a CPU backend
nor a CPU-participating scheduler. GPU-lane direct builds get a typed
unsupported error instead of a crash inside the GPU backend.

### GGUF bounded parsing

The fork's `gguf_init_from_buffer_with_limits` was replaced by a full-Rust
header validator (`ggml_runtime/gguf_bounded_parser.rs`) that enforces every
resource dimension of `GgufParseLimits` before the bytes are handed to stock
`gguf_init_from_buffer`. Complexity and failure semantics are preserved and
tested; the known caveats: the stock C parser itself has no caller-supplied
limits, so the Rust gate is the whole enforcement (allocation inside the C
parser for adversarial-but-in-budget input is possible but bounded by the
declared header).

### FFI layer

- `GGML_PREC_DEFAULT` is now `GGML_PREC_UNDEFINED` (same value 0); the
  precision setters use stock `ggml_prec_set_acc` (returns bool) instead of
  the removed `ggml_mul_mat_set_prec` / `ggml_flash_attn_ext_set_prec`.
- `GGML_TYPE_COUNT` 43 -> 44 (`GGML_TYPE_F8_E4M3 = 43` added in v1.0.0).
- `GgmlBackendDevCaps` gained `mmap_support` (stock ggml appended it; the
  fork-era mirror was one bool too small, which was an out-of-bounds write on
  the C side).
- `ggml_backend_tensor_set/get` now return `void`; the fail-closed checks on
  their result were removed because the new status-capable `*_n` variants are
  not yet wired in (see `KNOWN_LIMITATIONS` in this doc).
- The fork's terminal-status extension values (`EXECUTION_FAILED`,
  `DEVICE_LOST`, `BACKEND_POISONED`) and all `_v1`/`_v2`/`_v3_utf8` FFI
  symbols are deleted; `ggml_status` again ends at `GGML_STATUS_ABORTED`.

### Validation

All executed on a Linux x86_64 host with `rust-toolchain.toml` (1.100.0):

- `cargo fmt --check`: clean.
- `cargo clippy --all-targets -- -D warnings`: clean.
- `cargo nextest run --workspace`: **5632 passed, 0 failed, 217 skipped**.
  Plain `cargo test` is not evidence for this workspace (see `AGENTS.md`);
  nextest runs every test in its own process, which is required for the
  process-global trackers these tests assert against.
- `cargo test --workspace --doc`: clean (0 doc tests).
- `cargo test -p openasr-core golden_diff -- --nocapture`: ok (3 passed,
  14 host-local ignored).
- `cargo test -p openasr-core bundled_catalog_signature_verifies_committed_catalog_and_epoch -- --nocapture`: ok.
- The whisper tiny-encoder prelude golden hash moved to the stock v1.0.0
  Linux build (`657fd641...`, see `GOLDEN_DIFF_TINY_WHISPER_ENCODER_PRELUDE_SHA256`)
  because stock's F32 CPU kernels are not bit-identical to the fork's. The
  byte-identity assertion now runs on Linux only; the macOS reference was not
  re-verifiable in this environment. The macOS-only decoder-step logits golden
  was left in place but is almost certainly stale on v1.0.0 and must be
  regenerated on the macOS runner before the fork-pinning migration can merge.
- Catalog artifacts (`model-registry/catalog*.json` + signatures + epoch) are
  NOT regenerated here: `OPENASR_BACKEND_HOST_ABI_FINGERPRINT` changes with
  the submodule, so the committed catalog's `host_abi` entries are inert for
  new builds. Regenerating them requires the release manifest tooling and
  signing keys; this branch is not publishable as a release.

### Post-migration correctness fix: scheduler-rebind tensor detach

Root-caused and fixed a fork-only semantic stock ggml does not carry over.
The fork's `6775e391`/`0a44f2b0` added `ggml_gallocr_detach_graph_tensors_v1`:
on `ggml_backend_sched_reset` it walks the scheduler's split graph and clears
`tensor->buffer`/`tensor->data` for every tensor bound into the scheduler's
shared gallocr buffers (leaving external/WEIGHTS bindings intact). OpenASR
builds sibling persistent cgraphs that alternate on one shared scheduler;
stock's gallocr reallocates (frees) its buffer when a larger sibling binds,
and a retained sibling graph is left with `data` pointers into the freed
buffer. The next read/write to that graph's inputs is a heap-use-after-free
(ASan: write in `write_tensor_data` into a freed 2080-byte gallocr buffer,
also observed as a deterministic SIGSEGV at `pthread_create` inside
`ggml_graph_compute` from unsorted-bin corruption).

`ggml_runtime::cpu_graph::detach_scheduler_graph_bindings` reimplements the
semantic in Rust: it walks a retained cgraph's nodes/leafs (via the vendored
`struct ggml_cgraph` layout and `ggml_graph_n_nodes`), and for every tensor
bound into a COMPUTE-usage backend buffer (i.e. gallocr-owned, confirmed via
stock `ggml_backend_buffer_get_usage`) it clears `buffer`/`data`.
`ggml_backend_sched_alloc_graph` then rebinds the graph fresh against the
current buffers. It runs before every `sched_reset`/sibling rebind:

- `allocate_scheduler_graph` unbinds the previously active graph before
  committing the next one (the sibling-alternation path);
- `GgmlSchedulerGraphLifetime::drop`, `park_idle_runner_graph_context`, and
  `start_graph` detach the active graph before `ggml_backend_sched_reset`.

External WEIGHTS-usage bindings (arena, loaded packs) are always kept.
Note this makes rebinds **required**: any caller that interferes with the
shared scheduler (rebind, park, reset) must go through
`ensure_scheduler_graph_active`, never call `ggml_backend_sched_reset`
directly on a graph that may still hold gallocr-bound tensors.

### Known ABI differences from the fork (behavior changes)

- Fork-only terminal statuses (`EXECUTION_FAILED`, `DEVICE_LOST`,
  `BACKEND_POISONED`) are gone; unknown statuses (including the retired
  3/4 raw values) map to `UnknownComputeStatus` and poison the session but
  not the backend handle. Tests that asserted sticky backend poisoning on
  statuses 3/4 were rewritten to the stock contract.
- Stock `ggml_set_output` propagates `GGML_TENSOR_FLAG_OUTPUT` through the
  `view_src` chain (the fork did not). The weight-metadata regression test
  now asserts the flag delta only up to that stock behavior and pins
  data/buffer/shape identity instead of bitwise flags.
- Stock `GGML_TYPE_COUNT = 44` (`GGML_TYPE_F8_E4M3 = 43`). The
  "COUNT-1 must remain a live quantized slot" assertion now uses COUNT-2
  (stock's last quantized slot, `GGML_TYPE_Q2_0`).

## Known limitations / intentional degradations

1. Optional downloadable GPU packs cannot be qualified or freshly activated
   (no live target/driver probe on stock ggml). Already-persisted exact
   activation records still work. Static GPU builds are unaffected.
2. Cooperative cancellation is per-node on CPU, node-level via the Metal hook
   on Metal, and job/boundary-scoped (pre/post-compute) only on CUDA/Vulkan/HIP.
   The fork additionally polled at scheduler split boundaries (C-level), which
   is not reimplemented on stock ggml; callers needing intra-decode cancel on
   GPU lanes get it at job/boundary granularity only.
3. The backend-private trim hook (`abi.trim`) is a no-op; scheduler working
   sets are released by dropping/recreating schedulers, not by a native trim.
4. Backend-private accounting is reported as provider-unknown typed values;
   the `backend_owned_*` receipts never fabricate GPU-pool residency.
5. `ggml_backend_tensor_set/get` can no longer surface a synchronous failure.
   The `*_n` batched variants with status returns are not wired in; a
   follow-up should adopt them.
6. GPU lanes that previously ran the fork's fused kernels for swoosh,
   argmax-first, LSTM, rel-pos attention (or its non-rel-pos siblings) now
   route those ops to a CPU participant in a scheduler, or fail closed on a
   direct GPU build. Numbers to validate on hardware before calling this
   ready.
7. The `ggml-sync` tooling and `OPENASR_FORK.md`-style pin management are
   obsolete on stock ggml and tracked for cleanup; `tooling/ggml-sync/README.md`
   was already stale before this change.
8. Windows BACKEND_DL topology is re-baselined but unverified on this host:
   `cargo build` for `x86_64-pc-windows-msvc` was not run here. The
   `toml`/CMake contracts were updated by inspection only; CI must re-run the
   release windows contract before any tag.
9. The macOS whisper goldens (`GOLDEN_DIFF_TINY_WHISPER_DECODER_STEP_LOGITS_SHA256`,
   and any macOS encoder-prelude bytes) could not be re-verified on this
   Linux host. `ggml_backend_buffer_get_usage` distinguishes gallocr-owned
   COMPUTE buffers from arena/loaded-weight WEIGHTS buffers; other buffer
   usages cannot be distinguished from gallocr ownership, so the rebind-detach
   walks assume any COMPUTE-usage binding on a retained graph is
   gallocr-owned (holds for this runtime's scheduler graphs).
