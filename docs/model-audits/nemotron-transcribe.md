# Model release audit: nemotron-transcribe

> **Policy.** Should-support items MUST be `Supported`; anything else requires a
> detailed justification and an explicit unlock condition. This form ships with
> the model release. A family without a completed form does not enter the
> release flow: `tooling/publish-model/scripts/_manifest.py --public` fails
> closed on a missing or half-filled form.

**STATUS: RUNTIME IMPLEMENTED, BENCH COMPLETE, NOT READY FOR RELEASE.** The
nemotron-transcribe family's offline execution path (frontend -> PE
ASR+Sortformer encoder -> nemotron_h_moe trunk -> shared greedy driver) is
implemented and verified verbatim against the NeMo-Speech.cpp oracle golden
(`tooling/nemotron-reference-dumper/work/goldens/jfk.transcript.json`) on three
lanes: **CPU** (best-of-3 per quant, all three quants, sha-bound to the
published packs), **HIP** (full-residency runs on a gfx1151 Strix Halo,
112 GiB unified GTT; all three quants verbatim), and **CUDA** (Hybrid receipt
at `7944b7f7` on a 10 GB RTX 3080: all four short-clip runs pass on the first
candidate, both fixtures x both AD quants verbatim). Peak RSS per quant measured
on CPU/HIP; peak VRAM, GPU utilisation and Windows process RAM measured on CUDA.
**Remaining release blockers:** min_core_version / min_cli_version must name
the first release shipping this family; CUDA has one card and no >10 min
fixture or automated parity test, and the second file of a multi-file run still
loses Hybrid and completes via CPU fallback; the wider discrete-card GPU verdict
is unsupported outside the listed receipts; the AMI diarization fixtures are
unverified through this runtime; streaming is the shared snapshot driver.
**Do not flip `release_public = true`, run `publish_catalog.sh`, or treat any
cell as evidence beyond the listed lanes until those are closed.**

| Field | Value |
| --- | --- |
| Family (`models-core.toml` `family`) | nemotron-transcribe |
| Models covered | nemotron-3.5-transcribe-30b-a4b (q8_0, ad_iq4_nl, ad_q2_0_m; packs published at chameleon-ai/nemotron-3.5-transcribe-30b-a4b, hf_revision 4aefd4391c57bb5cf18ffcabac635f526c636142) |
| Auditor / date | OpenASR onboarding, 2026-11-18 |
| Core version + commit audited | workspace 0.2.1, `nemotron` branch (uncommitted working tree at audit time) |
| Bench hardware | AMD Ryzen AI MAX+ 395 / Radeon 8060S (gfx1151, 112 GiB unified GTT), Arch 2026-11-18 (CPU + HIP best-of-3); AMD RX 6800 XT (gfx1030, 16 GB) HIP bring-up, AMD ROCm/HIP; Windows 11 Pro (build 26300.10075), i7-12700F, 32 GB RAM (`31.85 GiB`), RTX 3080 (10239 MiB, `sm_86`, driver 621.40, CUDA 13.0), 2026-11-25 (CUDA Hybrid verification at `7944b7f7`, both fixtures x both AD quants); no Mac/Metal host benched |

### CUDA hybrid receipts (2026-11-25, external tester)

Recorded in `bug-report.md`, `bug-report-followup.md` and
`verification-report.md` for branch `cuda-hybrid`. These are manual hardware
receipts, not CI gates.

- Build at both `c54ec7f5` and `7944b7f7`:
  `cargo build --release -p openasr-cli --features "cuda,legacy-windows-static-sidecar"`
  with `OPENASR_CUDA_GPU_TARGETS=86`, no `RUSTFLAGS` workaround. The missing
  `cublasLt` link (`077f5f5a`) is confirmed fixed.
- `StaleStatsGeneration` ("quote is stale") did not recur in any run.
- **Initial receipt at `c54ec7f5`:** JFK (11 s) passed on both AD quants with
  VRAM 7284 / 7296 MiB at 61% util, but `en_zh_mixed.wav` (13 s) failed on both
  quants during encoder `load_gguf_weight_context` with a `system-memory`
  `observed` budget error on `pack-weight-buffer-chunk-0` (iq4 requested
  20,903,768,128 B, available 11,437,273,088 B; `binding_check=observed`,
  committed only 473,984 B). The requested size was the whole IQ4 pack minus
  the dead-weight skip set, i.e. a copied host buffer with no expert split.
  Root cause: the HOST_IMPORT quote charged an already-open mapping a second
  time, and the Hybrid host half used an anonymous buffer sized 11-17 GB
  against a live `ullAvailPhys` floor near zero. Fixed by `561a4685` and
  `7944b7f7` respectively.
- **Verified at `7944b7f7`:** all four 11/13 s runs exit 0 with **zero candidate
  retries** (Hybrid committed on the first candidate):

  | Quant / fixture | Wall / inference | RTF | VRAM peak / GPU util |
  | --- | ---: | ---: | ---: |
  | `ad_q2_0_m` / jfk | 29.14 s / 27.57 s | 2.51x | 7451 MiB / 77% |
  | `ad_q2_0_m` / en_zh_mixed | 42.22 s / 40.43 s | 3.11x | 7460 MiB / 73% |
  | `ad_iq4_nl` / jfk | 52.79 s / 50.43 s | 4.58x | 7469 MiB / 94% |
  | `ad_iq4_nl` / en_zh_mixed | 67.88 s / 65.61 s | 5.05x | 7465 MiB / 67% |

- Memory mechanism: peak `PrivateMemorySize64` fell from 15-23 GiB to ~7.25 GiB
  on all four runs; lowest `FreePhysicalMemory` improved from 0.00-0.59 GiB to
  0.85-5.88 GiB. The mmap import path is active on Windows.
- Transcripts: JFK matches the oracle golden on both quants. `en_zh_mixed.wav`
  produces the expected mixed-language text on both quants; iq4 ends
  `...去伊家新开。` where q2 reads `...去一家新开...` (no OpenASR golden exists
  for this fixture, so this is recorded, not scored).
- Two-file `--continue-on-error` runs (`ad_q2_0_m`, both orders) also pass 2/2,
  but the second file takes one `context-buffer` capacity retry and completes
  via CPU fallback (~9.5-11.6 s inference, peak VRAM ~9.57-9.58 GiB with both
  runtimes resident). Hybrid does not stay available across requests in one
  process; see section 7.

**How to fill.** Status is exactly one of:

- `Supported` -- implemented and verified for this family in this repo. Cite
  the evidence (test name, bench run, code path).
- `Not applicable` -- architecturally impossible or meaningless for this
  family. Say why, so nobody re-derives it.
- `Deferred` -- applicable but intentionally not done yet. Give the detailed
  justification AND the unlock condition (what measurement, upstream change,
  or milestone flips it to Supported).

The release gate still applies as written: this form deliberately does NOT
pass it today.

## 1. Graph & scheduling

| Item | Status | Justification / evidence (+ unlock condition if not Supported) |
| --- | --- | --- |
| Graph reuse / persistent session (no per-request graph rebuild) | Deferred | The prepared unified runtime (tokenizer, mmap'd embedding table, encoder+decoder sessions, KV/SSM/conv arenas) is resident across requests via the lane-keyed actor pool (`AdmittedPinnedRuntimeActorCheckoutPool`, key `(pack content, ExecutionLaneKey, required_span)`); transient forward graphs are still built per step. Unlock: persistent (build-once, re-run) decode graph across steps. |
| Op fusion opportunities reviewed (norm+matmul, QKV, rope, ...) | Deferred | Graphs are built but no fusion review has been done. Unlock: executor graph assembly review (Mamba-2 conv/SSM chain, MoE router+experts, PE attention blocks) with explicit fused-node decisions. |
| Batching / serve-batch path | Deferred | Batch-1 only; no serve-batch wiring. Unlock: serve-batch wiring once the executor path is benchmarked. |
| Encode-decode pipelining | Deferred | Executor is serial (encode, then decode). Unlock: pipelining design + measurement. |
| Arena / gallocr reuse across steps (no per-step allocator churn) | Deferred | KV/SSM/conv state arenas are allocated once per resident runtime (zeroed per request at prefill) and reused for all decode steps and across requests; graph compute buffers are allocated per step. Unlock: step-pool growth audit (`GgmlCpuStepBufferPool`) + allocator-reuse measurement across decode steps. |

## 2. Precision & quantization

| Item | Status | Justification / evidence (+ unlock condition if not Supported) |
| --- | --- | --- |
| KV cache quantization | Deferred | Attention KV is F32 in the resident span (7 attention blocks, GQA 2-head KV); no quantized-KV policy. Unlock: KV dtype policy decision (F16 vs F32) with measured residency impact on the release host. |
| Activation precision policy chosen deliberately (f32 vs f16) | Deferred | Activations are F32 end-to-end on the HIP lane (small activations; quantized operand types only where ggml fuses them into the Q-block kernels). No deliberate f16/BF16 policy and no per-op dtype map. Unlock: executor numerics policy documented with per-op dtype map. |
| Keep-quantized matmul (native Q blocks bound, no load-time dequant; RAM orders q4 < q8 < fp16) | Supported | Weights are bound from the verified pack in their native Q formats (Q8_0 dense, Q2_0/IQ4_NL expert blocks) and consumed by ggml's quantized kernels (`mul_mat` / `mul_mat_id`) at runtime; no host-side dequant pass exists anywhere in the executor. Pack-quant floor evidence: all three published tiers pass `openasr model-pack audit-quant` (zero below-floor tensors). RAM-ordering comparison across tiers is unmeasured. Unlock (full cell): quiet-host RAM-ordering measurement q8_0 > ad_iq4_nl > ad_q2_0_m. |
| Quant tiers complete (q4_k / q8_0 / fp16) | Not applicable | The three published tiers are q8_0 / ad_iq4_nl (imatrix IQ4_NL experts + Q8_0 dense) / ad_q2_0_m (imatrix Q2_0 experts + Q8_0 dense + IQ4_NL for the six largest MoE blocks and MTP head). No q4_k/fp16 tier is published upstream; the catalog lists exactly the three existing packs. |

## 3. Memory & data movement

| Item | Status | Justification / evidence (+ unlock condition if not Supported) |
| --- | --- | --- |
| mmap weight loading | Supported | Weights are bound from the verified pack through the shared preflight reader (`load_gguf_weight_context_from_preflight`): Q2_0/IQ4_NL/Q8_0 tensors are bound in place (no host-side dequantization pass), activations flow F32. Evidence: executor weight binding + `audit-quant` clean on all three packs. |
| Resident pool reuse across requests (weights stay resident) | Supported | `NemotronTranscribeGgmlExecutor` owns an `AdmittedPinnedRuntimeActorCheckoutPool` keyed by `(pack content, ExecutionLaneKey, required_span)` with limits 4 idle entries / 2 instances per key; the unified runtime (tokenizer, mmap'd embedding table, encoder+decoder sessions, KV/SSM/conv arenas) is leased per request and returned to the pool; the actor worker thread keeps every native context. `evict_prepared_runtime_content_id` evicts idle actors per pack; `unload_idle_state` drops them all. K2/K3 audit gates pass; `resident_runtime_audit` green. KV/SSM/conv arena reuse within a request AND across requests both exercised. |
| View contiguity tradeoffs audited (`cont`/copy nodes justified) | Deferred | Graphs are built with explicit `cont` calls at layout boundaries (e.g. SSM `zxBCdt` slicing, KV row writes, debug-free tap-free outputs); no audit pass has recorded the reasons per node. Unlock: graph review enumerating every `cont` and why the input could not be consumed strided. |
| Peak RSS/VRAM per shipped quant measured (quiet host) and reconciled against the weights+KV+activations budget; unexplained excess blocks release; catalog RAM requirement matches the measured peak | Supported (CPU, HIP, CUDA) | Measured on the Strix Halo bench host (2026-11-18, quiet, sha-bound to the published packs): ad_q2_0_m peak RSS 18.3 GB / HIP 15.1 GB; ad_iq4_nl 24.0 / 20.4; q8_0 38.2 / 34.7. Weights alone match (card: 15.2 / 21.3 / 36.7 GB); no excess observed. Catalog RAM requirement should cite these peaks at flip time. HIP VRAM not separately attributed (unified GTT). CUDA (2026-11-25, Windows, not a quiet host): all four 11/13 s AD runs peak at 7451-7469 MiB VRAM with 67-94% GPU util and ~7.25 GiB `PrivateMemorySize64` after the mmap-import fix (before: 15-23 GiB private, free RAM floor 0.00-0.59 GiB). Multi-file runs peak at ~9.57 GiB VRAM and ~9.81 GiB private with both runtimes resident. |
| Dead-weight skip (unused pack tensors excluded from the loaded weight context) | Supported by wiring + Linux unit tests + CUDA/HIP hardware runs | `token_embd.weight` and the dense `blk.52.*` tensors are never looked up by a ggml graph; both AD quants load and run on CUDA with the skip installed; see the detailed note below. |

### Dead-weight skip: rationale and the MTP dependency

The runtime now omits two tensor groups from the loaded ggml weight context even
though they remain in the `.oasr` pack and index:

- `token_embd.weight` (374,341,632 B in the AD tiers). The decoder does not
  bind this weight into a graph. `MappedTokenEmbeddingTable` gathers the needed
  rows directly from the mapped GGUF, and `upload_embeddings` supplies the
  resulting vectors as a graph input. The data is used; the ggml binding was
  redundant.
- Dense `blk.52.*` tensors (~52 MB: attention projections, norms, shared-expert
  projections, `nextn.eh_proj`). The decoder topology has exactly 52 blocks
  (`BLOCK_TYPES`, indices 0..=51); block 52 is the MTP/NextN speculative drafter
  and is outside the trunk. Its two *expert* tensors
  (`ffn_up_exps.weight` / `ffn_down_exps.weight`) are deliberately **not**
  skipped: the VRAM-derived offload policy already moves them to host memory
  first, where they cost nothing at runtime.

The pack is unchanged. `validate_runtime_pack_contract` checks the tensor index
(count, dims and pinned names) and never inspects the loaded context, so skipping
a binding at load time remains contract-valid. The skip set participates in the
loaded-weight owner-cache fingerprint, so a cached full load cannot be reused for
a skipped load; skipped names are also excluded from buffer sizing and binding,
and their bytes are subtracted in `offload::pack_weight_bytes` so the slice and
offload planner budgets the bytes the runtime actually has to place. Any future
graph lookup of a skipped name fails closed with a typed missing-weight error
rather than reading dead memory.

**MTP warning (must be addressed explicitly):** if speculative / NextN decoding
is implemented for this family, block 52 stops being dead. The MTP drafter will
need its dense `blk.52.*` weights and likely the embedding table, so the skip
list in `offload::skipped_weight_names` must be reduced accordingly and the
cache-fingerprint, planner-footprint and missing-weight tests updated in the same
change. Do not treat the skip list as a permanent property of the pack; it is a
statement about today's greedy-only decoder. The audit verdict for speculative
decode stays **Deferred**.

## 4. Decode algorithms

| Item | Status | Justification / evidence (+ unlock condition if not Supported) |
| --- | --- | --- |
| Greedy logits shortcuts (argmax path skips needless softmax/sort work) | Supported | Decode goes through the shared `run_builtin_seq2seq_decode_policy` with the registered shared policy `nemotron-transcribe.greedy.seq2seq.v1` (`NEMOTRON_TRANSCRIBE_DECODE_POLICY_COMPONENT`, decode_policy_component_registry.rs); the inventory row is `SharedSeq2SeqGreedy` and `shared_decode_topologies_call_their_declared_driver` enforces it. No separate softmax-skip shortcut is implemented (the shared loop owns the argmax path). |
| Speculative decode: per-family verdict recorded (do it, defer it, or dead) | Deferred | Verdict: **defer**. The pack's single nextn/MTP layer (blk.52) is inert without speculative-decode runtime work that does not exist anywhere in OpenASR; revisit with the MTP milestone. |
| CTC blank-skip fast path (CTC families; otherwise Not applicable) | Not applicable | The CTC head is a timestamp/diarize-auxiliary branch, not the decode driver. A word-time path now exists (`ctc_head`/`ctc_align`) but only projects the compact alignment columns, never full-vocabulary decode. Unlock: if the CTC head ever becomes the decode driver, revisit. |
| Decode guards are zero-cost on the hot path (degenerate-loop guard etc.) | Supported | The degenerate-loop guard and the stop-token accounting come from the shared driver (`run_seq2seq_greedy_decode_loop_v0`); the family-specific executor supplies only `Seq2SeqGreedyDecodeStepExecutor` + prompt plan credentials. |

## 5. Frontend & IO

| Item | Status | Justification / evidence (+ unlock condition if not Supported) |
| --- | --- | --- |
| Mel/fbank frontend SIMD + parallelized | Deferred | Host-side f32 mel128/16 kHz frontend (`preprocessor.fb` from the pack, periodic Hann pre-emphasis; verified against an independent numpy reference dump, max abs diff 2.8e-4). No SIMD/parallel audit. Unlock: frontend perf pass. |
| Zero-copy audio path (no avoidable resample/copy hops) | Deferred | Audio enters as f32 samples; the mel is computed host-side into one contiguous f32 buffer and uploaded; no resample hop. Unlock: frontend wiring review (mel into ggml_mel or fused frontend op). |
| VAD cost measured and accounted | Deferred | Executor runs, VAD cost not measured. The longform pipeline VAD is shared and already metered; SALM accepts <=1 hour audio so slicing runs only past 60 s integral. Unlock: measurement on the release host. |

## 6. Platform-specific

| Item | Status | Justification / evidence (+ unlock condition if not Supported) |
| --- | --- | --- |
| Metal command batching + wired memory budget respected | Deferred | No OpenASR Metal lane (this family has only been run on HIP, CUDA and CPU-build); Vulkan-class smoke on the upstream SALM server (see `tooling/nemotron-reference-dumper/README.md`, validated ad_q2_0_m on gfx1031/RADV 16 GB on 2026-11-16) is NOT evidence about OpenASR Metal. Unlock: OpenASR Metal smoke on reference quants. |
| CPU thread pool sized for P/E cores | Deferred | The CPU lane is correctness-verified and benched: JFK fixture transcribes verbatim on ad_q2_0_m (best-of-3, RTF 0.627, peak RSS ~19.5 GB on a 62 GB x86-64 host; load 96.3 s / compute 6.9 s, 2026-11-18); threadpool still not audited. Unlock: CPU threadpool audit + full per-quant CPU bench on the release host (ad_iq4_nl / q8_0 do not fit a 62 GB host under the residency regime). |
| Accelerate/BLAS used where it wins | Deferred | CPU lane not profiled. Unlock: CPU bring-up + BLAS comparison. |

## 7. Backend coverage matrix

Every cell must be answered. `Supported? = Yes` means the OpenASR backend
loads and runs the pack; `Golden-verified? = Yes` means byte/parity fixtures
pass ON that backend; `Utilization measured? = Yes` means the weight
placement gate (or equivalent) proved the compute actually runs there.
CPU and HIP are green on the listed lanes; CUDA is green for both short
fixtures (11 s and 13 s) on both AD quants at `7944b7f7`. Metal and Vulkan
remain unanswered.

| Backend | Supported? | Golden-verified? | Utilization measured? | Justification + unlock plan if unsupported |
| --- | --- | --- | --- | --- |
| CPU | Yes | Yes | N/A | All three quants bench on CPU (best-of-3, Ryzen AI MAX+ 395, 2026-11-18, sha-bound to the published packs, JFK verbatim): ad_q2_0_m RTF 0.234 / peak RSS 18.3 GB; ad_iq4_nl RTF 0.242 / 24.0 GB; q8_0 RTF 0.251 / 38.2 GB. Utilization: N/A on CPU. |
| Metal | No | No | No | See CPU. Unlock: same gate, on the Apple-Silicon bench host with the Metal backend. |
| CUDA | Yes | Yes (JFK oracle; both fixtures manual) | Yes | VRAM-derived Hybrid lane with MoE expert offload, declared by default at `c54ec7f5` and verified at `7944b7f7`: Windows 11 / RTX 3080 (10 GB, sm_86, CUDA 13.0), all four 11/13 s runs (both AD quants) exit 0 with zero retries and Hybrid committed on the first candidate; JFK matches the oracle golden, `en_zh_mixed.wav` produces expected text on both quants; VRAM 7451-7469 MiB at 67-94% util and ~7.25 GiB private bytes after the mmap-import fixes. `GGML_SCHED_DEBUG=2` confirms experts on CPU and trunk on CUDA. Not a general discrete-card verdict: one card, no >10 min fixture, no automated parity test, and the second file of a two-file run takes a `context-buffer` capacity retry and completes via CPU fallback. Unlock for full green: second CUDA card + >10 min golden + fix the cross-request Hybrid reuse loss. |
| Vulkan | No | No | No | See CPU; note upstream `nemo-speech vulkan-server` smoke (ad_q2_0_m, gfx1031/RADV, validated 2026-11-16) is oracle evidence, not OpenASR evidence. Unlock: same gate on the Vulkan bench host. |
| HIP | Yes | Yes | No | All three quants run and transcribe verbatim on ROCm/HIP (best-of-3, Radeon 8060S / gfx1151 / 112 GiB unified GTT, 2026-11-18, sha-bound packs): ad_q2_0_m RTF 0.090 / peak 15.1 GB; ad_iq4_nl RTF 0.092 / 20.4 GB; q8_0 RTF 0.105 / 34.7 GB. Utilization not measured. Discrete-card verdict (RX 6800 XT, gfx1030, 16 GB, ad_q2_0_m): the all-device lane silently fell back to CPU on every clip (the earlier "no retry" note was wrong -- candidate retries are gated on `OPENASR_TIMING`, not `RUST_LOG`); the VRAM-derived Hybrid split offloads all MoE experts to system RAM, decoder prefill is chunked (64 rows/call), and the longform slice length is derived from detected memory (capped at the family's 3600 s hard ceiling). JFK (11 s) now runs on the GPU with a verbatim transcript at ~6.9 GB peak VRAM; gilbert (94 s) runs in 3 x 40 s GPU slices at ~11.4 GB with no CPU fallback. The full-context single-segment path still needs chunked attention / a unified-memory GPU (see KNOWN_LIMITATIONS). |

## 8. Correctness & quality

| Item | Status | Justification / evidence (+ unlock condition if not Supported) |
| --- | --- | --- |
| WER vs fp16 measured for every shipped quant tier | Deferred | Measured upstream, not by OpenASR: the source repo reports <= +/-0.18 WER decoder-vs-BF16 for every tier from 15.55 GB up, and ad_q2_0_m +1.53 CER on FLEURS Chinese (model card). OpenASR's quality evidence is the verbatim JFK fixture (CPU/HIP, all three quants; CUDA, both AD quants) plus manually reviewed `en_zh_mixed.wav` transcripts on CUDA, neither scored against a frozen OpenASR golden. Unlock: OpenASR-side WER sweep per tier on the frozen en eval set before any public flip. |
| Model ref alias forms resolve identically everywhere (bare family / `family:canonical` / every `quant_tag_cases.json` alias accepted by CLI and server match logic; covered by the catalog-wide alias matrix test) | Deferred | The catalog row does not exist yet (staged); alias behavior is exercised by the catalog-wide matrix only after the entry is public. Unlock: catalog entry lands publicly; matrix test covers `nemotron-3.5-transcribe-30b-a4b` and its `:ad_q2_0_m`/`:ad_iq4_nl`/`:q8_0` suffix forms (note: the `ad_*` tags are exact, not aliased). |
| Golden coverage includes long audio AND a cross-backend parity fixture | Deferred | JFK verbatim transcript reproduces byte-identically through the OpenASR HIP runtime and, on the 2026-11-25 CUDA receipts, through the CUDA Hybrid runtime on both AD quants, matching the oracle golden (`tooling/nemotron-reference-dumper/work/goldens/jfk.transcript.json`); the mel path is pinned by an ignored golden unit test. The GPU receipts are manual comparisons, not automated parity tests. CUDA also produced `en_zh_mixed.wav` transcripts on both AD quants, with a q2/iq4 character difference (`一家` vs `伊家`) that has no OpenASR golden to score against. AMI 2-speaker fixtures are not validated by OpenASR; no long-audio golden, no cross-backend parity fixture. Unlock: pin `GOLDEN_JFK_TEXT` in a pack-backed test; add AMI validation + a >10 min fixture + a CPU/GPU parity fixture. |
| Official decode parameters honored (suppression, stop tokens, upstream reference settings) | Deferred | The served prompt renders the SALM chat template with an empty `<think></think>` block (thinking disabled), byte-identical prefix/suffix to the SALM reference (`<|im_start|>system\n<|im_end|>\n<|im_start|>user\n` + instruction + `<|audio|>` locator + `<|im_end|>\n<|im_start|>assistant\n<think></think>`); the greedy shared driver enforces the EOT stop token and the serve budget. Not covered by an automated render-parity test. Unlock: prompt-render parity test + suppression table audit against the pack's embedded template. |
| Long-audio degradation checked (repetition, drift, truncation) | Deferred | Not checked by OpenASR; only 11 s and 13 s fixtures have been run on CUDA, with no >10 min receipt on any lane. Upstream validates up to 1 h; the AMI 2-speaker 60 s fixture decodes verbatim on the oracle. Unlock: long-audio golden + degenerate-loop guard exercise on >10 min fixtures. |

## 9. Resource limits & fail-closed

| Item | Status | Justification / evidence (+ unlock condition if not Supported) |
| --- | --- | --- |
| Max audio length / context budget derived and over-limit behavior fails closed | Deferred | Envelope declared: `OpenAsrInvocationSpan::Bounded { max_seconds: 3600.0 }` (the SALM protocol's own one-hour limit) and `ScopedSlices { integral: 60, target: 30, max: 30 }`. The resident KV span is planned from the packed prompt by `capacity.rs` (min 640 positions) and prefill/decode are bounded by the planned span, but the over-limit path is not yet tested end-to-end. Unlock: over-limit fixture test (audio past the invocation span must fail closed before any network/graph work). |
| Streaming first-token latency floor documented (chunk accumulation math; streaming families, otherwise Not applicable) | Deferred | SALM refuses live audio outright; OpenASR models this family as `UtteranceComplete`. Unlock: streaming session driver lands (or the family documents a live-protocol refusal path and this row becomes Not applicable). |
| KV growth rate per audio second known | Deferred | Decode runs but KV/sec has not been measured. Decoder: 2688-width, 53 blocks (7 attention blocks with GQA KV; KV growth is a function of the 7 attention blocks only). Unlock: measured KV/sec on the release host. |
| Metal wired-memory profile captured | Deferred | No Metal run exists for this family. Unlock: Metal smoke with `GGML_SCHED_DEBUG=2`. |

## 10. Engineering completeness

| Item | Status | Justification / evidence (+ unlock condition if not Supported) |
| --- | --- | --- |
| Runtime/encoder unit tests beyond goldens | Deferred | Unit tests cover the prompt planner, the mel frontend (incl. a pack-backed ignored golden), the runtime contract/classifier (`runtime_contract::*`, `package_import::*`), and the merge-tokenizer. The executor/decoder graphs and the end-to-end path have no automated test (the CPU/HIP/CUDA JFK runs are all manual); the backend/FFI seams are covered elsewhere in the workspace. Unlock: pack-backed executor test (`#[ignore]`d, JFK verbatim) + decoder block-level unit tests. |
| Golden-verified pack in the repo or fetchable in CI | Deferred | Packs are too large for CI (15-37 GB); they live on HF at chameleon-ai/nemotron-3.5-transcribe-30b-a4b with sha256 pins, and CI verifies them by hash via the catalog once public. Unlock: public catalog row + `tooling/public-hf-e2e/` covers the pull path. |
| Pinned upstream revisions for weights, tokenizer, and fixtures | Supported | `hf_revision` pinned: `4aefd4391c57bb5cf18ffcabac635f526c636142` (chameleon-ai repo, 2026-11-17); pack sha256s verified 2026-11-16 and re-verified 2026-11-17 by this onboarding (`tmp/publish/nemotron-3.5-transcribe-30b-a4b/packs/`); upstream AtomicChat conversion repo pinned at `dbd06c5cad7a2f33c8e06be2fb3ef814dbd9aa59`; base checkpoint id `0b5820fad1103208f65895b586bd65d2` recorded in every pack; fixtures pinned under `tooling/nemotron-reference-dumper/`. |
| Reference dumper / parity oracle exists for this family | Supported | `tooling/nemotron-reference-dumper/` pins NVIDIA's NeMo-Speech.cpp (`e790e463b2913cff9f989a3df224c587f14b4c5c`) as the parity oracle of record; transcript-level goldens for JFK and AMI 2-speaker (both modes) are captured and sha256-checked by the script. OpenASR's HIP runtime reproduces the JFK golden byte-identically; the CUDA Hybrid run does the same manually on both AD quants. It does NOT dump intermediate tensors; block-level parity is a separate, explicitly scoped tool if ever needed. |
