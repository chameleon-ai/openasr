#!/usr/bin/env python3
"""Produce transcript-level goldens for the ``nemotron-transcribe`` family
from the official NVIDIA NeMo-Speech.cpp reference runtime.

NeMo-Speech.cpp is the parity oracle of record for the
``nemotron-3.5-transcribe-30b-a4b`` model (see ``docs/model-audits/TEMPLATE.md``
section 10, "Reference dumper exists for this family", and
``docs/design/model-family-lifecycle.md``). It is the only runtime that
implements the full SALM pipeline this family ports: the Parallel Expert
Encoder (ASR + Sortformer diar experts), thresholded fusion, the
``nemotron_h_moe`` decoder (Mamba-2 + MoE + GQA NoPE attention), and the
CTC word-timestamp head.

This script automates exactly what was validated on a 16 GB Vulkan host
(gfx1031, 2026-11-16): clone the pinned oracle, build it, fetch the
AtomicChat GGUF pair, and run the two committed fixtures through
``nemo-speech transcribe``, writing one ``<fixture>.transcript.json`` per
fixture (plus ``manifest.json``) under ``--out-dir``.

What this tool produces is **transcript-level parity** (verbatim text,
punctuation, per-word timestamps + speaker tags). It does **not** dump
intermediate tensors; block-level goldens (mel / encoder / mamba / MoE /
logits) are out of scope here (see ``translation-plan.md`` 6.6 and 8.1 R5
in the parent planning documents).

House rules honored:
  - Nothing third-party is vendored into this repository. The oracle is
    cloned into ``--repo-dir`` at a pinned revision; the GGUFs are fetched
    into ``--model-dir`` at a pinned HF revision and sha256-verified.
  - Artifacts (clone, build tree, models, dumps) live under
    ``tooling/nemotron-reference-dumper/work/`` by default and are covered
    by ``.gitignore``; nothing generated is committed. The Rust goldens
    (``GOLDEN_*_TEXT`` etc.) are transcribed manually from these dumps into
    the family's tests.
  - Fail closed: pin mismatch, hash mismatch, missing toolkit, missing
    binary, non-zero oracle exit -> abort with a specific error.

Pinned references (keep in sync with the audit form and the descriptor):
  - NeMo-Speech.cpp ``e790e463b2913cff9f989a3df224c587f14b4c5c`` (main @
    2026-11-16); its llama.cpp submodule is pinned by the repo itself
    (``bd4f514db14d87fded667787a7a963bfbaa98e89``).
  - AtomicChat GGUF conversion repo
    ``AtomicChat/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-GGUF`` revision
    ``dbd06c5cad7a2f33c8e06be2fb3ef814dbd9aa59`` (2026-10-28), converted
    from base checkpoint ``nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B``
    checkpoint id ``0b5820fad1103208f65895b586bd65d2``.

Requirements on the host:
  - git, CMake >= 3.26, Ninja, a C++17 toolchain, SentencePiece dev files.
  - A GPU supported by the chosen preset (``vulkan-server`` on Linux/Vulkan,
    ``cuda-server`` on NVIDIA with CUDA >= 12.8, ``cpu-server`` anywhere,
    just slow). The default SALM flags assume >= 16 GB VRAM; for smaller
    GPUs/CPU pass ``--salm-args`` (and expect long runtimes).
  - ~40 GB free disk (models 15.2 GB for ad_q2_0_m + clone/build ~10 GB).
  - Python 3.9+ stdlib only (no third-party Python deps).

Usage (from the repository root):
  python3 tooling/nemotron-reference-dumper/dump_golden.py            # all stages
  python3 tooling/nemotron-reference-dumper/dump_golden.py --check    # verify pins only
  python3 tooling/nemotron-reference-dumper/dump_golden.py --skip-build --skip-download
  python3 tooling/nemotron-reference-dumper/dump_golden.py --quant q8_0

Outputs:
  <out-dir>/jfk.transcript.json            # jfk.wav verbatim transcript (+ word times)
  <out-dir>/ami-2spk.transcript.json       # 60 s AMI 2-speaker clip, verbatim transcript
  <out-dir>/ami-2spk.diarize.transcript.json  # same clip, speaker-tagged (<spk:N>) prompt
  <out-dir>/manifest.json                  # revisions, sha256s, argv used, texts
"""

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import urllib.request

NEMO_SPEECH_CPP_URL = "https://github.com/NVIDIA/NeMo-Speech.cpp.git"
NEMO_SPEECH_CPP_PIN = "e790e463b2913cff9f989a3df224c587f14b4c5c"
LLAMA_CPP_PIN = "bd4f514db14d87fded667787a7a963bfbaa98e89"  # the repo's own submodule pin

HF_REPO = "AtomicChat/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-GGUF"
HF_REVISION = "dbd06c5cad7a2f33c8e06be2fb3ef814dbd9aa59"
BASE_CHECKPOINT_ID = "0b5820fad1103208f65895b586bd65d2"

QUANTS = {
    # name: (filename, size in bytes, sha256) - sha256s verified 2026-11-16
    # against the source repo (see nemotron-3.5-transcribe-oasr/README.md).
    "ad_q2_0_m": (
        "NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-AD-Q2_0_M.gguf",
        13_572_208_800,
        "764dc3aedbc800a96346e188d74f56fc4d0932299e4771df1bc2e1c952ceed81",
    ),
    "ad_iq4_nl": (
        "NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-AD-IQ4_NL.gguf",
        19_678_656_672,
        "0c47d4ab6c783e7606ec03714b354fab0b2052027a7ea1991e7e2d3009c345ad",
    ),
    "q8_0": (
        "NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-Q8_0.gguf",
        35_004_643_232,
        "3d99075b58f15153d87e8922f54e12dda818bbb505c2e0d7d50cd4f30beac9f2",
    ),
}

ENCODER = (
    "NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-Encoder-F16.gguf",
    1_670_410_080,
    "4221029c8a565c4a80c507ffa760c716e595acf663f9f4273ce69910050e7ba2",
)

# Fixtures shipped in the NeMo-Speech.cpp checkout (relative to repo root).
# Modes: "default" = verbatim transcription prompt + word times;
# "diarize" = speaker-tagged prompt (<spk:N> tags, no punctuation/caps) +
# word times + per-word speaker ids. SALM switches prompt with --diarize
# (docs/salm/README.md), so the two are deliberately separate passes.
FIXTURES = (
    {"name": "jfk", "wav": "test_files/asr/wav/test/jfk.wav", "modes": ("default",)},
    {"name": "ami-2spk", "wav": "test_files/diar/ami_en2002d_2132.wav", "modes": ("default", "diarize")},
)

# 16 GB-host-validated SALM settings (gfx1031 smoke test, 2026-11-16).
DEFAULT_SALM_ARGS = [
    "--asr.salm.max_sequences", "2",
    "--asr.salm.n_ctx_per_sequence", "8192",
    "--asr.salm.max_context", "8192",
    "--asr.salm.mtp_draft", "0",
    "--asr.salm.n_batch", "512",
    "--asr.salm.encoder_max_batch", "2",
]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WORK_DIR = os.path.join(SCRIPT_DIR, "work")


def log(msg):
    print(f"[dump_golden] {msg}", flush=True)


def fail(msg):
    print(f"[dump_golden] ERROR: {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


def run(cmd, cwd=None, check=True):
    log("$ " + " ".join(shlex.quote(c) for c in cmd))
    proc = subprocess.run(cmd, cwd=cwd)
    if check and proc.returncode != 0:
        fail(f"command failed with exit code {proc.returncode}: {' '.join(cmd)}")
    return proc.returncode


def sha256_file(path, progress_every_gb=True):
    h = hashlib.sha256()
    total = 0
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 22)  # 4 MiB
            if not chunk:
                break
            h.update(chunk)
            total += len(chunk)
    return h.hexdigest(), total


def check_tool(name, install_hint):
    if shutil.which(name) is None:
        fail(f"required tool '{name}' not found on PATH. {install_hint}")


def repo_head(repo_dir):
    out = subprocess.run(
        ["git", "-C", repo_dir, "rev-parse", "HEAD"],
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        fail(f"cannot read HEAD of {repo_dir}: {out.stderr.strip()}")
    return out.stdout.strip()


def ensure_repo(repo_dir, full_submodules, force_reclone):
    if os.path.isdir(repo_dir) and force_reclone:
        log(f"--force-reclone: removing {repo_dir}")
        shutil.rmtree(repo_dir)
    if os.path.isdir(os.path.join(repo_dir, ".git")):
        head = repo_head(repo_dir)
        if head != NEMO_SPEECH_CPP_PIN:
            fail(
                f"{repo_dir} is at {head}, expected {NEMO_SPEECH_CPP_PIN}.\n"
                f"Refusing to reuse a wrong revision. Either pass --force-reclone\n"
                f"or remove the directory and re-run."
            )
        dirty = subprocess.run(
            ["git", "-C", repo_dir, "status", "--porcelain"],
            capture_output=True, text=True,
        ).stdout.strip()
        if dirty:
            log(f"WARNING: {repo_dir} has uncommitted changes (expected for a build tree):\n{dirty[:2000]}")
        else:
            log(f"oracle repo already at pinned revision {NEMO_SPEECH_CPP_PIN}")
        return
    if os.path.exists(repo_dir):
        fail(f"{repo_dir} exists but is not a git checkout; remove it or pass --force-reclone")
    os.makedirs(os.path.dirname(repo_dir) or ".", exist_ok=True)
    log(f"cloning {NEMO_SPEECH_CPP_URL} -> {repo_dir} (this is a few hundred MB)")
    run(["git", "clone", NEMO_SPEECH_CPP_URL, repo_dir])
    run(["git", "-C", repo_dir, "checkout", NEMO_SPEECH_CPP_PIN])
    if full_submodules:
        run(["git", "-C", repo_dir, "submodule", "update", "--init", "--recursive"])
    else:
        # vulkan-server/cuda-server need only llama.cpp (SALM) and cpp-httplib
        # (HTTP server); the remaining submodules serve gRPC/TTS/diar builds.
        run(["git", "-C", repo_dir, "submodule", "update", "--init",
              "llama.cpp", "third_party/cpp-httplib"])
    head = repo_head(repo_dir)
    if head != NEMO_SPEECH_CPP_PIN:
        fail(f"checkout landed on {head}, expected {NEMO_SPEECH_CPP_PIN}")
    log(f"oracle repo pinned at {head}")


def file_matches(path, size, sha256):
    if not os.path.isfile(path):
        return False
    actual_size = os.path.getsize(path)
    if actual_size != size:
        log(f"{os.path.basename(path)}: size {actual_size} != expected {size}")
        return False
    log(f"verifying sha256 of {os.path.basename(path)} ({size / 1e9:.2f} GB) ...")
    digest, _ = sha256_file(path)
    if digest != sha256:
        log(f"{os.path.basename(path)}: sha256 {digest} != expected {sha256}")
        return False
    return True


def download_file(url, dest, size, sha256):
    tmp = dest + ".part"
    log(f"downloading {url}")
    log(f"  -> {tmp} ({size / 1e9:.2f} GB; resume not supported, expect minutes)")
    h = hashlib.sha256()
    received = 0
    req = urllib.request.Request(url, headers={"User-Agent": "openasr-nemotron-reference-dumper"})
    with urllib.request.urlopen(req) as resp, open(tmp, "wb") as out:
        while True:
            chunk = resp.read(1 << 22)
            if not chunk:
                break
            out.write(chunk)
            h.update(chunk)
            received += len(chunk)
            if received % (1 << 30) < (1 << 22):  # ~each GiB
                log(f"  ... {received / 1e9:.2f} / {size / 1e9:.2f} GB")
    if received != size:
        os.remove(tmp)
        fail(f"download truncated: {received} bytes, expected {size}")
    digest = h.hexdigest()
    if digest != sha256:
        os.remove(tmp)
        fail(f"downloaded file failed sha256 verification: {digest} != {sha256}.\n"
             f"The pinned HF revision may have changed; do NOT proceed.")
    os.replace(tmp, dest)
    log(f"verified {os.path.basename(dest)} (sha256 ok)")


def ensure_models(model_dir, quant, skip_download):
    os.makedirs(model_dir, exist_ok=True)
    wanted = [QUANTS[quant], ENCODER]
    for fname, size, sha in wanted:
        dest = os.path.join(model_dir, fname)
        if file_matches(dest, size, sha):
            log(f"{fname}: present, sha256 verified")
            continue
        if skip_download:
            fail(f"{dest} missing or fails verification and --skip-download was given")
        if os.path.isfile(dest):
            log(f"{fname}: present but failed verification; re-downloading")
            os.remove(dest)
        url = f"https://huggingface.co/{HF_REPO}/resolve/{HF_REVISION}/{fname}"
        download_file(url, dest, size, sha)
    return (
        os.path.join(model_dir, QUANTS[quant][0]),
        os.path.join(model_dir, ENCODER[0]),
    )


def ensure_preset_listed(repo_dir, preset):
    path = os.path.join(repo_dir, "CMakePresets.json")
    try:
        with open(path) as f:
            presets = {p.get("name") for p in json.load(f).get("configurePresets", [])}
    except (OSError, json.JSONDecodeError) as e:
        fail(f"cannot read {path}: {e}")
    if preset not in presets:
        fail(f"preset '{preset}' not in {path}. Available: {sorted(presets)}")


def build_repo(repo_dir, preset, jobs):
    check_tool("cmake", "Install CMake >= 3.26 (https://cmake.org/download/).")
    check_tool("ninja", "Install Ninja (e.g. 'apt-get install ninja-build').")
    ensure_preset_listed(repo_dir, preset)
    run(["cmake", "--preset", preset], cwd=repo_dir)
    run(["cmake", "--build", "--preset", preset, "-j", str(jobs)], cwd=repo_dir)


def find_binary(repo_dir, preset):
    candidate = os.path.join(repo_dir, "build", preset, "bin", "nemo-speech")
    if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
        return candidate
    fail(
        f"nemo-speech binary not found at {candidate}.\n"
        f"Did the build succeed? Re-run without --skip-build, or check the "
        f"preset ('{preset}')."
    )


def run_fixture(binary, repo_dir, fixture, decoder, encoder, salm_args, out_path):
    wav = os.path.join(repo_dir, fixture["wav"])
    if not os.path.isfile(wav):
        fail(f"fixture missing from oracle checkout: {wav}")
    mode = fixture["mode"]
    cmd = [
        binary, "transcribe", wav,
        "--model", decoder,
        "--salm-encoder", encoder,
        "--device", "auto",
        "--format", "json",
        "--word-times",
        "--force",  # re-dumps overwrite our own generated artifacts
        *salm_args,
        *(["--diarize"] if mode == "diarize" else []),
        "-o", out_path,
    ]
    log(f"transcribing {fixture['wav']} [{mode}] -> {out_path}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-4000:]
        fail(f"oracle failed on {fixture['wav']} [{mode}] (exit {proc.returncode}):\n{tail}")
    try:
        with open(out_path) as f:
            doc = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        fail(f"oracle wrote unreadable JSON to {out_path}: {e}")
    log(f"  text: {doc.get('text', '')[:200]!r}")
    log(f"  words: {len(doc.get('words', []))}, duration: {doc.get('duration')} s")
    return doc


def main():
    parser = argparse.ArgumentParser(
        description="Build the pinned NeMo-Speech.cpp oracle and dump transcript-level goldens.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--quant", choices=sorted(QUANTS), default="ad_q2_0_m",
                        help="decoder quant to run (default: ad_q2_0_m, the catalog default)")
    parser.add_argument("--preset", default="vulkan-server",
                        help="CMake preset to build; must have SPEECHLM on (default: vulkan-server)")
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--repo-dir", default=os.path.join(DEFAULT_WORK_DIR, "NeMo-Speech.cpp"))
    parser.add_argument("--model-dir", default=os.path.join(DEFAULT_WORK_DIR, "models"))
    parser.add_argument("--out-dir", default=os.path.join(DEFAULT_WORK_DIR, "goldens"))
    parser.add_argument("--salm-args", default=None,
                        help="extra/replacement SALM args, shell-split, e.g. "
                             "'--asr.salm.n_batch 1024'. Replaces the default 16 GB settings.")
    parser.add_argument("--no-default-salm-args", action="store_true",
                        help="do not pass the default 16 GB SALM settings at all")
    parser.add_argument("--fixture", choices=[f["name"] for f in FIXTURES] + ["all"], default="all")
    parser.add_argument("--check", action="store_true",
                        help="verify toolchain, pins, hashes and binary; do not build/run")
    parser.add_argument("--skip-clone", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-download", action="store_true",
                        help="use existing files in --model-dir; fail if missing/invalid")
    parser.add_argument("--skip-dump", action="store_true")
    parser.add_argument("--full-submodules", action="store_true",
                        help="init ALL oracle submodules (default: llama.cpp + cpp-httplib only)")
    parser.add_argument("--force-reclone", action="store_true",
                        help="wipe --repo-dir and re-clone from scratch")
    args = parser.parse_args()

    salm_args = list(DEFAULT_SALM_ARGS)
    if args.no_default_salm_args:
        salm_args = []
    if args.salm_args:
        salm_args = shlex.split(args.salm_args)

    if args.check:
        check_tool("git", "Install git.")
        if not args.skip_build:
            check_tool("cmake", "Install CMake >= 3.26.")
            check_tool("ninja", "Install Ninja.")
        if not args.skip_clone:
            if os.path.isdir(os.path.join(args.repo_dir, ".git")):
                head = repo_head(args.repo_dir)
                status = "OK" if head == NEMO_SPEECH_CPP_PIN else f"MISMATCH (got {head})"
                log(f"oracle repo pin: {status}")
                if head != NEMO_SPEECH_CPP_PIN:
                    sys.exit(1)
            else:
                log(f"oracle repo not cloned yet at {args.repo_dir}")
        ensure_models(args.model_dir, args.quant, skip_download=True)
        if not args.skip_dump:
            binary = find_binary(args.repo_dir, args.preset)
            log(f"nemo-speech binary: {binary}")
        log("check OK")
        return

    check_tool("git", "Install git.")
    if not args.skip_clone:
        ensure_repo(args.repo_dir, args.full_submodules, args.force_reclone)
    elif not os.path.isdir(os.path.join(args.repo_dir, ".git")):
        fail(f"--skip-clone given but {args.repo_dir} is not a git checkout")

    if not args.skip_build:
        build_repo(args.repo_dir, args.preset, args.jobs)

    decoder, encoder = ensure_models(args.model_dir, args.quant, args.skip_download)
    if args.skip_dump:
        log("--skip-dump given; setup complete, no goldens produced")
        return
    binary = find_binary(args.repo_dir, args.preset)

    os.makedirs(args.out_dir, exist_ok=True)
    fixture_defs = FIXTURES if args.fixture == "all" else tuple(
        f for f in FIXTURES if f["name"] == args.fixture)
    results = {}
    for fdef in fixture_defs:
        for mode in fdef["modes"]:
            fx = {**fdef, "mode": mode}
            out_name = (f"{fdef['name']}.transcript.json" if mode == "default"
                        else f"{fdef['name']}.{mode}.transcript.json")
            out_path = os.path.join(args.out_dir, out_name)
            results[(fdef["name"], mode)] = {
                "file": out_name,
                "doc": run_fixture(binary, args.repo_dir, fx,
                                   decoder, encoder, salm_args, out_path),
            }

    manifest = {
        "tool": "tooling/nemotron-reference-dumper/dump_golden.py",
        "nemo_speech_cpp": {"url": NEMO_SPEECH_CPP_URL, "pin": NEMO_SPEECH_CPP_PIN,
                            "llama_cpp_submodule": LLAMA_CPP_PIN},
        "source_gguf": {"repo": HF_REPO, "revision": HF_REVISION,
                        "base_checkpoint_id": BASE_CHECKPOINT_ID,
                        "decoder": os.path.basename(decoder), "decoder_sha256": QUANTS[args.quant][2],
                        "encoder": ENCODER[0], "encoder_sha256": ENCODER[2]},
        "preset": args.preset,
        "salm_args": salm_args,
        "fixtures": [
            {"name": name,
             "mode": mode,
             "file": results[(name, mode)]["file"],
             "text": results[(name, mode)]["doc"].get("text"),
             "duration": results[(name, mode)]["doc"].get("duration"),
             "word_count": len(results[(name, mode)]["doc"].get("words", []))}
            for name, mode in sorted(results)
        ],
    }
    manifest_path = os.path.join(args.out_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    log(f"wrote {manifest_path}")
    log("goldens are transcript-level only; transcribe them into the family's "
        "GOLDEN_* constants manually (see README.md). Do not commit the work/ directory.")


if __name__ == "__main__":
    main()
