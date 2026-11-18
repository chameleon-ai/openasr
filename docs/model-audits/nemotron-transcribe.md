# Model release audit: nemotron-transcribe

> **Policy.** Should-support items MUST be `Supported`; anything else requires a
> detailed justification and an explicit unlock condition. This form ships with
> the model release. A family without a completed form does not enter the
> release flow: `tooling/publish-model/scripts/_manifest.py --public` fails
> closed on a missing or half-filled form.

**STATUS: STAGED, NOT READY FOR RELEASE.** The nemotron-transcribe family is
registered in the canonical architecture inventory as a fail-closed,
not-yet-implemented executor so its packs can pass the production
`PackVerifier` and the release pipeline can proceed. **No backend, no decode
path, and no quality measurement has been produced by OpenASR for this
family.** Every cell below is written to be honest about that state. Do not
flip `release_public = true`, run `publish_catalog.sh`, or treat any cell as
evidence of working inference until the runtime lands and this form is re-cut
against it.

| Field | Value |
| --- | --- |
| Family (`models-core.toml` `family`) | nemotron-transcribe |
| Models covered | nemotron-3.5-transcribe-30b-a4b (q8_0, ad_iq4_nl, ad_q2_0_m; packs published at chameleon-ai/nemotron-3.5-transcribe-30b-a4b, hf_revision 4aefd4391c57bb5cf18ffcabac635f526c636142) |
| Auditor / date | OpenASR onboarding, 2026-11-17 |
| Core version + commit audited | workspace 0.2.1, `nemotron` branch (uncommitted working tree at audit time) |
| Bench hardware | none |

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
| Graph reuse / persistent session (no per-request graph rebuild) | Deferred | No executor exists; the registered `NemotronTranscribeGgmlExecutor` fails closed on every call (see `crates/openasr-core/src/models/nemotron_transcribe/executor.rs`). Unlock: the family executor lands and `resident_runtime_audit::k2_*`/`k3_*` pass against a real module that owns a `PreparedRuntimeCache<...>`. |
| Op fusion opportunities reviewed (norm+matmul, QKV, rope, ...) | Deferred | Nothing to fuse; no graphs are built. Unlock: executor graph assembly review (Mamba-2 conv/SSM chain, MoE router+experts, PE attention blocks). |
| Batching / serve-batch path | Deferred | No executor. Unlock: serve-batch wiring once the executor can produce a batch-1 result. |
| Encode-decode pipelining | Deferred | No executor. Unlock: same as above. |
| Arena / gallocr reuse across steps (no per-step allocator churn) | Deferred | No executor. Unlock: executor lands; allocator reuse measured. |

## 2. Precision & quantization

| Item | Status | Justification / evidence (+ unlock condition if not Supported) |
| --- | --- | --- |
| KV cache quantization | Deferred | No executor; the attention KV is not built. Unlock: KV dtype policy chosen with the executor (upstream embeds the 7 attention blocks + blk.52 MTP head only). |
| Activation precision policy chosen deliberately (f32 vs f16) | Deferred | See above; upstream runs decoder in f16/BF16-mixed. Unlock: executor numerics policy documented with per-op dtype map. |
| Keep-quantized matmul (native Q blocks bound, no load-time dequant; RAM orders q4 < q8 < fp16) | Deferred | The published packs bind Q2_0/IQ4_NL/Q8_0 natively in any future executor; nothing is loaded today. The install-time `pack_quant_audit` floor is declared (`SemanticRolesV1` classifier, `models/nemotron_transcribe/package_import.rs`). Unlock: executor weight binding via `weight_tensor_payload_by_name` + `new_matmul_weight_2d_typed`, then this cell cites the K1/RAM-ordering evidence. |
| Quant tiers complete (q4_k / q8_0 / fp16) | Not applicable | The three published tiers are q8_0 / ad_iq4_nl (imatrix IQ4_NL experts + Q8_0 dense) / ad_q2_0_m (imatrix Q2_0 experts + Q8_0 dense + IQ4_NL for the six largest MoE blocks and MTP head). No q4_k/fp16 tier is published upstream; the catalog lists exactly the three existing packs. |

## 3. Memory & data movement

| Item | Status | Justification / evidence (+ unlock condition if not Supported) |
| --- | --- | --- |
| mmap weight loading | Deferred | No executor. Unlock: resident weight load via the shared mmap seams (`runtime_source`/`map_verified` path) with the executor's weight context. |
| Resident pool reuse across requests (weights stay resident) | Deferred | No executor; the family is registered with `FamilyOwned` prepared runtime and an empty `PreparedRuntimeCache` seam (empty by construction today). Unlock: executor caching with `resident_runtime_audit::k3_registered_families_reference_physical_execution_lane_identity` passing on real code. |
| View contiguity tradeoffs audited (`cont`/copy nodes justified) | Deferred | No executor. Unlock: graph review. |
| Peak RSS/VRAM per shipped quant measured (quiet host) and reconciled against the weights+KV+activations budget; unexplained excess blocks release; catalog RAM requirement matches the measured peak | Deferred | Not measured; no OpenASR process has ever loaded this model. Weights alone: q8_0 36.7 GB / ad_iq4_nl 21.3 GB / ad_q2_0_m 15.2 GB (card). Unlock: quiet-host measurement per quant on the release-bench host. |

## 4. Decode algorithms

| Item | Status | Justification / evidence (+ unlock condition if not Supported) |
| --- | --- | --- |
| Greedy logits shortcuts (argmax path skips needless softmax/sort work) | Deferred | No decode loop exists. Decode driver is declared `Dedicated` on the inventory row pending registration of `nemotron-transcribe.greedy.seq2seq.v1` as a shared policy; the implementation will reuse `run_builtin_seq2seq_decode_policy`. Unlock: shared-policy registration + `shared_decode_topologies_call_their_declared_driver` green. |
| Speculative decode: per-family verdict recorded (do it, defer it, or dead) | Deferred | Verdict: **defer**. The pack's single nextn/MTP layer (blk.52) is inert without speculative-decode runtime work that does not exist anywhere in OpenASR; revisit with the MTP milestone. |
| CTC blank-skip fast path (CTC families; otherwise Not applicable) | Not applicable | The CTC head is a timestamp/diarize-auxiliary branch, not the decode driver. Unlock: if a word-time path is built on the CTC branch, revisit. |
| Decode guards are zero-cost on the hot path (degenerate-loop guard etc.) | Deferred | No decode loop. Unlock: shared-driver reuse brings the guard for free once registered. |

## 5. Frontend & IO

| Item | Status | Justification / evidence (+ unlock condition if not Supported) |
| --- | --- | --- |
| Mel/fbank frontend SIMD + parallelized | Deferred | No frontend exists. The pack embeds a mel128/16 kHz filterbank (`preprocessor.fb`, documented in the card) and the upstream nn mel is the reference. Unlock: family mel frontend + ggml_mel SIMD audit. |
| Zero-copy audio path (no avoidable resample/copy hops) | Deferred | No executor. Unlock: executor frontend wiring review. |
| VAD cost measured and accounted | Deferred | No executor. The longform pipeline VAD is shared and already metered; SALM accepts <=1 hour audio so slicing runs only past 60 s integral. Unlock: measurement on the release host. |

## 6. Platform-specific

| Item | Status | Justification / evidence (+ unlock condition if not Supported) |
| --- | --- | --- |
| Metal command batching + wired memory budget respected | Deferred | No executor; Vulkan-class smoke on the upstream SALM server (see `tooling/nemotron-reference-dumper/README.md`, validated ad_q2_0_m on gfx1031/RADV 16 GB on 2026-11-16) is NOT evidence about OpenASR Metal. Unlock: OpenASR Metal smoke on reference quants. |
| CPU thread pool sized for P/E cores | Deferred | No executor. Unlock: CPU bring-up + threadpool audit. |
| Accelerate/BLAS used where it wins | Deferred | No executor. Unlock: CPU bring-up. |

## 7. Backend coverage matrix

Every cell must be answered. `Supported? = Yes` means the OpenASR backend
loads and runs the pack; `Golden-verified? = Yes` means byte/parity fixtures
pass ON that backend; `Utilization measured? = Yes` means the weight
placement gate (or equivalent) proved the compute actually runs there.
**No cell is Yes today.**

| Backend | Supported? | Golden-verified? | Utilization measured? | Justification + unlock plan if unsupported |
| --- | --- | --- | --- | --- |
| CPU | No | No | No | The runtime does not exist; packs are structurally verified and hash-pinned, never executed by OpenASR. Unlock: executor lands on CPU first; golden fixture (`GOLDEN_JFK_TEXT` from `tooling/nemotron-reference-dumper/work/goldens/jfk.transcript.json`) runs green; then this cell flips. |
| Metal | No | No | No | See CPU. Unlock: same gate, on the Apple-Silicon bench host with the Metal backend. |
| CUDA | No | No | No | See CPU. Unlock: executor on CUDA + gpu-weight-placement gate with one-shot `GGML_SCHED_DEBUG=2`. |
| Vulkan | No | No | No | See CPU; note upstream `nemo-speech vulkan-server` smoke (ad_q2_0_m, gfx1031/RADV, validated 2026-11-16) is oracle evidence, not OpenASR evidence. Unlock: same gate on the Vulkan bench host. |
| HIP | No | No | No | See CPU; no HIP-capable bench host identified. Unlock: a HIP host and the same four gates. |

## 8. Correctness & quality

| Item | Status | Justification / evidence (+ unlock condition if not Supported) |
| --- | --- | --- |
| WER vs fp16 measured for every shipped quant tier | Deferred | Measured upstream, not by OpenASR: the source repo reports <= +/-0.18 WER decoder-vs-BF16 for every tier from 15.55 GB up, and ad_q2_0_m +1.53 CER on FLEURS Chinese (model card). OpenASR has run zero inferences. Unlock: OpenASR-side WER sweep per tier on the frozen en eval set before any public flip. |
| Model ref alias forms resolve identically everywhere (bare family / `family:canonical` / every `quant_tag_cases.json` alias accepted by CLI and server match logic; covered by the catalog-wide alias matrix test) | Deferred | The catalog row does not exist yet (staged); alias behavior is exercised by the catalog-wide matrix only after the entry is public. Unlock: catalog entry lands publicly; matrix test covers `nemotron-3.5-transcribe-30b-a4b` and its `:ad_q2_0_m`/`:ad_iq4_nl`/`:q8_0` suffix forms (note: the `ad_*` tags are exact, not aliased). |
| Golden coverage includes long audio AND a cross-backend parity fixture | Deferred | Transcript-level goldens exist and are pinned by the oracle dumper (`tooling/nemotron-reference-dumper/`: `jfk.transcript.json`, `ami-2spk.transcript.json`, `ami-2spk.diarize.transcript.json`; JFK verbatim-stable across runs). No long-audio OpenASR golden, no cross-backend parity fixture. Unlock: executor lands; pin `GOLDEN_JFK_TEXT`; add a 10-minute+ fixture and a CPU/GPU parity fixture in the same change. |
| Official decode parameters honored (suppression, stop tokens, upstream reference settings) | Deferred | No decode loop. The pack's embedded chat template defaults `enable_thinking=ON`; transcription requires rendering with thinking disabled (upstream llama.cpp PR #31471 documents the same). This constraint MUST be carried into the executor's prompt construction. Unlock: executor lands with prompt-render parity to the SALM reference and the constraint above enforced + tested. |
| Long-audio degradation checked (repetition, drift, truncation) | Deferred | No executor. Upstream validates up to 1 h; the AMIO 2-speaker 60 s fixture decodes verbatim on the oracle. Unlock: long-audio golden + degenerate-loop guard exercise on >10 min fixtures. |

## 9. Resource limits & fail-closed

| Item | Status | Justification / evidence (+ unlock condition if not Supported) |
| --- | --- | --- |
| Max audio length / context budget derived and over-limit behavior fails closed | Deferred | Envelope declared: `OpenAsrInvocationSpan::Bounded { max_seconds: 3600.0 }` (the SALM protocol's own one-hour limit) and `ScopedSlices { integral: 60, target: 30, max: 30 }`; the staged executor fails closed on ALL inputs before any budget is consulted. Unlock: executor enforces the bound and the over-limit path is tested. |
| Streaming first-token latency floor documented (chunk accumulation math; streaming families, otherwise Not applicable) | Deferred | SALM refuses live audio outright; OpenASR models this family as `UtteranceComplete`. Unlock: streaming session driver lands (or the family documents a live-protocol refusal path and this row becomes Not applicable). |
| KV growth rate per audio second known | Deferred | No executor. Decoder: 2688-width, 53 blocks (7 attention blocks with GQA KV); KV growth is measurable only once decode runs. Unlock: measured KV/sec on the release host. |
| Metal wired-memory profile captured | Deferred | No executor, no Metal run. Unlock: Metal smoke with `GGML_SCHED_DEBUG=2`. |

## 10. Engineering completeness

| Item | Status | Justification / evidence (+ unlock condition if not Supported) |
| --- | --- | --- |
| Runtime/encoder unit tests beyond goldens | Deferred | None exist; no code to test beyond the fail-closed stub (`models/nemotron_transcribe/`). Its contract/classifier tests (`runtime_contract::*`, `package_import::*`) pin the published packs' metadata and quant roles. Unlock: executor unit tests land with the runtime. |
| Golden-verified pack in the repo or fetchable in CI | Deferred | Packs are too large for CI (15-37 GB); they live on HF at chameleon-ai/nemotron-3.5-transcribe-30b-a4b with sha256 pins, and CI verifies them by hash via the catalog once public. Unlock: public catalog row + `tooling/public-hf-e2e/` covers the pull path. |
| Pinned upstream revisions for weights, tokenizer, and fixtures | Supported | `hf_revision` pinned: `4aefd4391c57bb5cf18ffcabac635f526c636142` (chameleon-ai repo, 2026-11-17); pack sha256s verified 2026-11-16 and re-verified 2026-11-17 by this onboarding (`tmp/publish/nemotron-3.5-transcribe-30b-a4b/packs/`); upstream AtomicChat conversion repo pinned at `dbd06c5cad7a2f33c8e06be2fb3ef814dbd9aa59`; base checkpoint id `0b5820fad1103208f65895b586bd65d2` recorded in every pack; fixtures pinned under `tooling/nemotron-reference-dumper/`. |
| Reference dumper / parity oracle exists for this family | Supported | `tooling/nemotron-reference-dumper/` pins NVIDIA's NeMo-Speech.cpp (`e790e463b2913cff9f989a3df224c587f14b4c5c`) as the parity oracle of record; transcript-level goldens for JFK and AMI 2-speaker (both modes) are captured and sha256-checked by the script. It does NOT dump intermediate tensors; block-level parity is a separate, explicitly scoped tool if ever needed. |
