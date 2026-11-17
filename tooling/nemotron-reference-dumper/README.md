# nemotron-reference-dumper

Transcript-level reference goldens for the `nemotron-transcribe` family
(`nemotron-3.5-transcribe-30b-a4b`), produced with the official NVIDIA
**NeMo-Speech.cpp** SALM runtime — the parity oracle of record for this
family (see `docs/MODEL_ONBOARDING.md`, `docs/design/model-family-lifecycle.md`,
and `docs/model-audits/TEMPLATE.md` §10 "Reference dumper exists for this
family"). The descriptor's `reference_dumper_source` points at this directory.

The script `dump_golden.py` clones the pinned oracle, builds it, fetches the
AtomicChat GGUF pair (sha256-verified), and runs the two committed fixtures
through `nemo-speech transcribe` in the modes needed by the acceptance
ladder:

- `test_files/asr/wav/test/jfk.wav` — 11 s verbatim JFK clip (default mode);
  the same fixture used for the acceptance-criteria smoke run.
- `test_files/diar/ami_en2002d_2132.wav` — 60 s AMI 2-speaker clip, run in
  both modes: default (verbatim transcript) and `--diarize` (the speaker
  prompt: `<spk:N>`-tagged output and per-word `speaker` ids; SALM switches
  prompt with `--diarize`, see `docs/salm/README.md`).

## Scope (read before relying on it)

These are **transcript-level goldens**: verbatim text, punctuation, and
per-word start/end times plus speaker labels. This tool does **not** dump
intermediate tensors (mel, encoder states, mamba/MoE blocks, logits). Block-level
parity is covered by the Rust per-block tests instead (see the family's
onboarding/implementation planning documents, the verification-ladder and
"op-coverage and numeric risks" sections). If the encoder or a decoder block
ever needs a tensor dump, that is a separate, explicitly scoped tool.

## Pinned references

| What | Pin |
|:-----|:----|
| NeMo-Speech.cpp | `e790e463b2913cff9f989a3df224c587f14b4c5c` (repo `.git` HEAD verified by the script) |
| its llama.cpp submodule | `bd4f514db14d87fded667787a7a963bfbaa98e89` |
| AtomicChat GGUF repo | `AtomicChat/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-GGUF` revision `dbd06c5cad7a2f33c8e06be2fb3ef814dbd9aa59` |
| base checkpoint | `nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B`, checkpoint id `0b5820fad1103208f65895b586bd65d2` |

Model files (sha256 verified by the script after download — do not substitute):

| File | Size | sha256 |
|:-----|-----:|:-------|
| `...-AD-Q2_0_M.gguf` (default decoder) | 13,572,208,800 | `764dc3aedbc800a96346e188d74f56fc4d0932299e4771df1bc2e1c952ceed81` |
| `...-AD-IQ4_NL.gguf` | 19,678,656,672 | `0c47d4ab6c783e7606ec03714b354fab0b2052027a7ea1991e7e2d3009c345ad` |
| `...-Q8_0.gguf` | 35,004,643,232 | `3d99075b58f15153d87e8922f54e12dda818bbb505c2e0d7d50cd4f30beac9f2` |
| `...-Encoder-F16.gguf` (shared) | 1,670,410,080 | `4221029c8a565c4a80c507ffa760c716e595acf663f9f4273ce69910050e7ba2` |

## Requirements

- git, CMake ≥ 3.26, Ninja, C/C++17 toolchain, SentencePiece development
  files (`libsentencepiece-dev`).
- A GPU for the chosen preset. `vulkan-server` (default) works on Linux with
  Mesa Vulkan (validated on gfx1031 / RADV, 16 GB, with `ad_q2_0_m`);
  `cuda-server` for NVIDIA (CUDA ≥ 12.8, needs the `patches/` it builds by
  default); `cpu-server` anywhere (slow).
- ~40 GB free disk; Python ≥ 3.9, stdlib only.
- The default SALM flags (`--asr.salm.*`) assume ≥ 16 GB VRAM. Override with
  `--salm-args "..."` or `--no-default-salm-args` for other hardware.

## Usage

From the repository root:

```bash
# full pipeline: clone, build, download models, dump goldens
python3 tooling/nemotron-reference-dumper/dump_golden.py

# verify pins/hashes only (no clone/build/run)
python3 tooling/nemotron-reference-dumper/dump_golden.py --check

# reuse existing checkout/models, just produce goldens
python3 tooling/nemotron-reference-dumper/dump_golden.py --skip-clone --skip-build --skip-download

# run on the reference (fidelity) tier instead of the catalog default
python3 tooling/nemotron-reference-dumper/dump_golden.py --quant q8_0
```

Stages are independently skippable (`--skip-clone|--skip-build|--skip-download|--skip-dump`).
Everything lands under `tooling/nemotron-reference-dumper/work/` by default
(git-ignored); override with `--repo-dir`, `--model-dir`, `--out-dir`.

Outputs (in `--out-dir`):

- `jfk.transcript.json` — verbatim transcript + word times.
- `ami-2spk.transcript.json` — 60 s AMI 2-speaker clip, verbatim transcript + word times.
- `ami-2spk.diarize.transcript.json` — same clip with the speaker (`--diarize`) prompt:
  `<spk:N>`-tagged text and per-word `speaker` ids. Note this output is the
  *speaker-prompted* rendering (no punctuation/capitalization), not a
  punctuated transcript; it exists to pin the speaker path (v1a).
- `manifest.json` — oracle pin, model sha256s, SALM args, fixture texts.

## Determinism (Vulkan preset, observed 2026-11-16)

Greedy decode is mostly stable but not bit-exact across runs on the Vulkan
path. JFK is verbatim and stable across repeated runs; the AMI clip in the
default mode was byte-identical across consecutive runs; the `--diarize`
AMI pass wobbled by a couple of words between runs (145 vs 147) with the
first token differing. Pin `jfk.transcript.json` verbatim in Rust tests;
for the speaker path prefer property assertions (>= 2 distinct speaker ids,
non-empty speaker-tagged turns) over exact equality. Whether the CUDA or
CPU presets are more strictly deterministic is unvalidated.

## After the dump: wiring goldens into tests

Nothing under `work/` is committed. The Rust family pins the transcripts
(e.g. `GOLDEN_JFK_TEXT`) in `#[ignore]`d golden tests, transcribed manually
from these files — the same pattern as `qwen-reference-dumper` and
`moss-reference-dumper`. Real-weight runs are manual/release-gated; CI never
invokes the oracle.

## Notes / gotchas

- `vulkan-server` builds only `libllama` for the SALM path; no speculative
  decoding (Vulkan runs upstream llama.cpp semantics) — fine for greedy
  parity, and it is what the 16 GB smoke test validated.
- The first `git clone` pulls several submodules too (llama.cpp,
  cpp-httplib); `--full-submodules` additionally initializes the gRPC/TTS
  submodules (only needed for `cuda-full`-style builds).
- The `.oasr` merged packs are **not** used here — the oracle loads the
  two-file AtomicChat layout (decoder + sidecar), which is what was
  validated on 2026-11-16.
- JFK produced a verbatim, punctuated transcript on this host on 2026-11-16;
  if your run diverges, the first thing to check is the pin and the sha256s
  (`--check`), then the SALM flags (`--salm-args`).
