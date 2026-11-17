#!/usr/bin/env python3
"""Merge a Nemotron 3.5 Transcribe 30B-A4B decoder GGUF and the shared F16
encoder sidecar GGUF into a single `.oasr` pack.

NVIDIA-Nemotron-3.5-Transcribe-30B-A4B ships as two GGUF files: a decoder
(architecture ``nemotron_h_moe``; 53 blocks of the Mamba-2/Transformer/MoE
hybrid, 417 tensors, tokenizer + chat template embedded) and a shared
~1.67 GB F16 encoder sidecar (architecture ``pee``; 886 tensors: 32-layer ASR
encoder, 31-layer Sortformer speaker branch, CTC timestamp head, mel
filterbank). The `.oasr` v1 payload is a single standard GGUF byte stream
(``docs/format/OASR_PACKAGE_CONTRACT_V1.md``), so a pack is built by merging
both tensor tables into one GGUF. This script is the pack-prep step for that
merge (an ``external_converter``-seam script, declared per model via
``models-publish.toml`` ``prep_scripts`` and run by ``prep.sh``); the pure-Rust
``nemotron_transcribe::package_import`` importer is the follow-on
self-containment milestone and supersedes it once the family module lands.

Merge rules:
  - Tensor tables concatenated: decoder tensors first (offsets unchanged,
    relative to the data blob), then encoder tensors (offsets shifted by the
    decoder data region size). The two files are fail-closed-checked for
    tensor-name collisions (none exist: ``blk.*``/``token_embd``/``output*``
    vs ``asr.*``/``diar.*``/``ctc.*``/``fusion.*``/``preprocessor.*``).
  - Metadata: decoder keys are authoritative; the encoder contributes every
    non-conflicting key (``pee.*``, ``ctc.*``, ``nemo_speech.components``).
    The four conflicting keys keep the decoder value (decoder-wins set below).
  - The ``openasr.*`` envelope keys are stamped (values below); a source file
    already carrying one fails closed.

Layout invariants implemented (``third_party/openasr-ggml/include/gguf.h`` and
``src/gguf.cpp``):
  - GGUF v3: magic, version, tensor count, kv count, kvs, tensor infos,
    pad-to-32, tensor data blob (each tensor padded to 32 bytes)
  - the tensor offset field is relative to the data blob start and a multiple
    of the alignment (32; neither source sets ``general.alignment``)
  - metadata is emitted sorted by key bytes, matching the Rust BTreeMap writer
  - tensor data is copied byte-identical from the two sources, so the merged
    blob is exactly ``decoder blob || encoder blob``

Fail-closed verification runs on every write (and on its own via
``--verify-only``): re-parse the output; check counts, offset invariants
(first offset 0, 32-aligned, strictly monotonic), exact total size, that every
source metadata key survives byte-identically, that the envelope keys are
present with the exact expected bytes, and 4 MB head/tail spot checks of both
data regions. ``--full-hash`` upgrades the spot checks to a sha256 over the
entire tensor data regions of both sources vs the output blob.

Family constants below (envelope values, decoder-wins set, expected decoder
architecture) belong to the ``nemotron-transcribe`` family; they are pinned
here so the prep output is reproducible, and must be kept in sync with the
Rust descriptor row when the family module is onboarded.

Usage:
  python3 tooling/publish-model/scripts/nemotron_transcribe_gguf_to_oasr.py \
      --decoder <decoder>.gguf --encoder <encoder>.gguf --out <pack>.oasr \
      [--full-hash] [--dry-run] [--verify-only]

``prep_scripts`` entry shape (one line per quant, from the repo root):
  python3 tooling/publish-model/scripts/nemotron_transcribe_gguf_to_oasr.py \
      --decoder {src}/<decoder>.gguf --encoder {src}/<encoder>.gguf \
      --out {work}/<registry_id>-<quant>.oasr --full-hash
"""
import argparse
import hashlib
import os
import struct
import time

ALIGN = 32
MAGIC = b'GGUF'
CHUNK = 256 * 1024 * 1024

# openasr.* envelope keys for the nemotron-transcribe family (string-typed
# descriptors per OASR_PACKAGE_CONTRACT_V1.md). openasr.package.version is
# OASR_PACKAGE_VERSION_V1; openasr.model.* / audio.frontend / decode.policy /
# tokenizer.id mirror the proposed descriptor facet values in the onboarding
# plan. openasr.source.revision is the base-model checkpoint id recorded in
# nemo_speech.checkpoint_id in every source GGUF.
OASR_KEYS = [
    ('openasr.audio.frontend', 'nemotron-transcribe.mel128.16khz.mono.v0'),
    ('openasr.decode.policy', 'nemotron-transcribe.greedy.seq2seq.v1'),
    ('openasr.license.name', 'OpenMDW-1.1'),
    ('openasr.license.source', 'https://openmdw.ai/license/1-1/'),
    ('openasr.model.architecture', 'nemotron-transcribe-hybrid-moe'),
    ('openasr.model.family', 'nemotron-transcribe'),
    ('openasr.model.id', 'nemotron-3.5-transcribe-30b-a4b'),
    ('openasr.package.version', '1'),
    ('openasr.source.name', 'nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B'),
    ('openasr.source.revision', '0b5820fad1103208f65895b586bd65d2'),
    ('openasr.tokenizer.id', 'nemotron-transcribe.bpe-131k.v1'),
]

# Metadata keys present in both files; the decoder value wins.
DECODER_WINS = {
    'general.architecture',       # nemotron_h_moe (decoder) vs pee (sidecar)
    'general.file_type',          # decoder quant class vs sidecar f16
    'general.name',               # decoder name vs "... Encoder F16"
    'nemo_speech.checkpoint_id',  # identical value in both files
}

# The merged pack must keep the decoder's GGUF architecture string; any other
# value means the wrong decoder file was passed.
EXPECTED_ARCHITECTURE = 'nemotron_h_moe'


class Buf:
    """Buffered forward reader. Tracks the file position consumed.

    One Buf per file for the whole header parse; it reads ahead, so never
    create a fresh Buf on the same file mid-stream.
    """

    def __init__(self, f):
        self.f = f
        self.buf = b''
        self.off = 0
        self.base = 0  # file position of buf[0]

    @property
    def pos(self):
        return self.base + self.off

    def take(self, n):
        while len(self.buf) - self.off < n:
            chunk = self.f.read(1 << 24)
            if not chunk:
                raise EOFError('unexpected end of GGUF header')
            if self.off > (1 << 26):
                self.base += self.off
                self.buf = self.buf[self.off:]
                self.off = 0
            self.buf += chunk
        out = self.buf[self.off:self.off + n]
        self.off += n
        return out

    def u32(self):
        return struct.unpack('<I', self.take(4))[0]

    def u64(self):
        return struct.unpack('<Q', self.take(8))[0]

    def value_bytes(self, file_start, file_end):
        """Raw bytes (type tag + payload) of a value in [file_start, file_end)."""
        if file_end - file_start <= 0 or file_start < self.base:
            raise ValueError('value range outside buffered region')
        return self.buf[file_start - self.base:file_end - self.base]


def skip_value_body(r, t, depth=0):
    """Skip the payload (not the 4-byte type tag) of a value of type t."""
    if t in (0, 1, 7):
        r.take(1)
    elif t in (2, 3):
        r.take(2)
    elif t in (4, 5, 6):
        r.take(4)
    elif t in (10, 11, 12):
        r.take(8)
    elif t == 8:
        ln = r.u64()
        if ln > 64 * (1 << 20):
            raise ValueError('implausible string length %d' % ln)
        r.take(ln)
    elif t == 9:
        st = r.u32()
        n = r.u64()
        if n > 100 * (1 << 20) or depth > 4:
            raise ValueError('implausible array (n=%d depth=%d)' % (n, depth))
        for _ in range(n):
            skip_value_body(r, st, depth + 1)
    else:
        raise ValueError('unknown gguf value type %d' % t)


class Gguf:
    def __init__(self, path, meta, tensors, data_start, file_size):
        self.path = path
        self.meta = meta        # key -> raw (type tag + payload) bytes
        self.tensors = tensors  # (name, ndim, dims, ttype, off_relative)
        self.data_start = data_start
        self.file_size = file_size
        self.region_size = file_size - data_start

    def summary(self):
        return ('%s: tensors=%d kv=%d data_start=%d region=%d (%.2f GB)'
                % (os.path.basename(self.path), len(self.tensors), len(self.meta),
                   self.data_start, self.region_size, self.region_size / 1e9))


def parse_gguf(path):
    with open(path, 'rb') as f:
        r = Buf(f)
        magic = r.take(4)
        if magic != MAGIC:
            raise ValueError('%s: bad magic %r' % (path, magic))
        ver = r.u32()
        if ver != 3:
            raise ValueError('%s: unsupported version %d' % (path, ver))
        n_tensors = r.u64()
        n_kv = r.u64()
        if n_tensors == 0 or n_tensors > 10_000_000:
            raise ValueError('implausible tensor count %d' % n_tensors)
        if n_kv == 0 or n_kv > 1_000_000:
            raise ValueError('implausible kv count %d' % n_kv)

        meta = {}
        for _ in range(n_kv):
            klen = r.u64()
            if klen > (1 << 20):
                raise ValueError('implausible key length %d' % klen)
            key = r.take(klen).decode('utf-8')
            if key in meta:
                raise ValueError('%s: duplicate metadata key %s' % (path, key))
            vstart = r.pos
            vtype = r.u32()
            skip_value_body(r, vtype)
            meta[key] = r.value_bytes(vstart, r.pos)

        tensors = []
        names = set()
        for _ in range(n_tensors):
            nlen = r.u64()
            if nlen > (1 << 20):
                raise ValueError('implausible tensor name length')
            name = r.take(nlen).decode('utf-8')
            if name in names:
                raise ValueError('%s: duplicate tensor %s' % (path, name))
            names.add(name)
            ndim = r.u32()
            if ndim == 0 or ndim > 8:
                raise ValueError('implausible ndim %d for %s' % (ndim, name))
            dims = [r.u64() for _ in range(ndim)]
            ttype = r.u32()
            if ttype >= 43:
                raise ValueError('implausible tensor type %d for %s' % (ttype, name))
            off = r.u64()
            tensors.append((name, ndim, dims, ttype, off))

        data_start = (r.pos + ALIGN - 1) // ALIGN * ALIGN
        if tensors[0][4] != 0:
            raise ValueError('%s: first tensor offset %d != 0' % (path, tensors[0][4]))
        offs = [t[4] for t in tensors]
        if any(o % ALIGN for o in offs):
            raise ValueError('%s: tensor offset not %d-aligned' % (path, ALIGN))
        if any(b <= a for a, b in zip(offs, offs[1:])):
            raise ValueError('%s: non-monotonic tensor offsets' % path)
        file_size = r.f.seek(0, 2)
        if file_size < data_start or file_size - data_start <= offs[-1]:
            raise ValueError('%s: tensor data offsets exceed file' % path)
    return Gguf(path, meta, tensors, data_start, file_size)


def tensor_meta_bytes(name, ndim, dims, ttype, off):
    b = name.encode('utf-8')
    out = struct.pack('<Q', len(b)) + b + struct.pack('<I', ndim)
    for d in dims:
        out += struct.pack('<Q', d)
    out += struct.pack('<I', ttype) + struct.pack('<Q', off)
    return out


def build_header(dec, enc, merged_meta):
    keys = sorted(merged_meta)
    hdr = bytearray()
    hdr += MAGIC
    hdr += struct.pack('<I', 3)
    hdr += struct.pack('<Q', len(dec.tensors) + len(enc.tensors))
    hdr += struct.pack('<Q', len(keys))
    for k in keys:
        kb = k.encode('utf-8')
        hdr += struct.pack('<Q', len(kb)) + kb + merged_meta[k]
    dec_region = dec.region_size
    for (name, ndim, dims, ttype, off) in dec.tensors:
        hdr += tensor_meta_bytes(name, ndim, dims, ttype, off)
    for (name, ndim, dims, ttype, off) in enc.tensors:
        hdr += tensor_meta_bytes(name, ndim, dims, ttype, dec_region + off)
    data_start = (len(hdr) + ALIGN - 1) // ALIGN * ALIGN
    hdr += b'\x00' * (data_start - len(hdr))
    return bytes(hdr), data_start, dec_region


def copy_region(src, src_off, size, out, label, t0):
    src.seek(src_off)
    remaining = size
    while remaining > 0:
        data = src.read(min(CHUNK, remaining))
        if len(data) != min(CHUNK, remaining):
            raise IOError('%s: short read' % label)
        out.write(data)
        remaining -= len(data)
        done = size - remaining
        if done % (1 << 30) == 0 or remaining == 0:
            rate = done / max(time.time() - t0, 1e-9) / 1e6
            print('    %s: %d/%d MB (%.0f MB/s)' % (label, done // (1 << 20),
                                                   size // (1 << 20), rate), flush=True)


def hash_region_into(h, src, src_off, size):
    """Stream the region [src_off, src_off+size) of src into an open hash h."""
    src.seek(src_off)
    remaining = size
    while remaining > 0:
        data = src.read(min(CHUNK, remaining))
        if not data:
            raise IOError('short read in hash')
        h.update(data)
        remaining -= len(data)


def verify_pack(out_path, dec, enc, data_start, dec_region, full_hash=False):
    print('  verifying %s' % out_path)
    with open(out_path, 'rb') as o, open(dec.path, 'rb') as d, open(enc.path, 'rb') as e:
        r = Buf(o)
        if r.take(4) != MAGIC:
            raise ValueError('output magic mismatch')
        if r.u32() != 3:
            raise ValueError('output version mismatch')
        n_tensors = r.u64()
        n_kv = r.u64()
        if n_tensors != len(dec.tensors) + len(enc.tensors):
            raise ValueError('output tensor count %d != %d'
                             % (n_tensors, len(dec.tensors) + len(enc.tensors)))
        out_meta = {}
        for _ in range(n_kv):
            klen = r.u64()
            key = r.take(klen).decode('utf-8')
            if key in out_meta:
                raise ValueError('output duplicate metadata key %s' % key)
            vstart = r.pos
            vtype = r.u32()
            skip_value_body(r, vtype)
            out_meta[key] = r.value_bytes(vstart, r.pos)
        first_off = None
        last_off = 0
        for _ in range(n_tensors):
            nlen = r.u64()
            r.take(nlen)
            ndim = r.u32()
            for _ in range(ndim):
                r.u64()
            r.u32()
            off = r.u64()
            if first_off is None:
                first_off = off
            elif off <= last_off:
                raise ValueError('output tensor offsets not monotonic')
            if off % ALIGN:
                raise ValueError('output tensor offset not aligned')
            last_off = off
        if first_off != 0:
            raise ValueError('output first tensor offset %d != 0' % first_off)

        # metadata content checks
        arch_expect = struct.pack('<I', 8) + struct.pack('<Q', len(EXPECTED_ARCHITECTURE)) \
            + EXPECTED_ARCHITECTURE.encode('utf-8')
        if out_meta.get('general.architecture') != arch_expect:
            raise ValueError('output general.architecture wrong: %r'
                             % out_meta.get('general.architecture'))
        for k, v in OASR_KEYS:
            expect = struct.pack('<I', 8) + struct.pack('<Q', len(v)) + v.encode('utf-8')
            if out_meta.get(k) != expect:
                raise ValueError('output metadata %s wrong' % k)
        # every source key must survive byte-identically (decoder-wins keys
        # keep the decoder value in the output)
        for src, skip in ((dec, set()), (enc, DECODER_WINS)):
            for k, raw in src.meta.items():
                if k in skip:
                    continue
                if out_meta.get(k) != raw:
                    raise ValueError('output metadata %s differs from source bytes' % k)

        out_size = o.seek(0, 2)
        expect_size = data_start + dec_region + enc.region_size
        if out_size != expect_size:
            raise ValueError('output size %d != expected %d' % (out_size, expect_size))

        def spot(off_out, src_f, src_off, region, label):
            n = min(4 << 20, region)
            for tag, at in (('head', 0), ('tail', region - n)):
                o.seek(off_out + at)
                src_f.seek(src_off + at)
                if o.read(n) != src_f.read(n):
                    raise ValueError('%s %s spot-check mismatch' % (label, tag))

        spot(data_start, d, dec.data_start, dec_region, 'decoder region')
        spot(data_start + dec_region, e, enc.data_start, enc.region_size, 'encoder region')
        print('    layout + metadata + spot checks OK (%d tensors, %d kv, size %d)'
              % (n_tensors, n_kv, out_size))
        if full_hash:
            # One stream hash on each side: H(decoder_region || encoder_region).
            h_src = hashlib.sha256()
            hash_region_into(h_src, d, dec.data_start, dec.region_size)
            hash_region_into(h_src, e, enc.data_start, enc.region_size)
            o.seek(data_start)
            h_out = hashlib.sha256()
            hash_region_into(h_out, o, data_start, dec_region + enc.region_size)
            if h_src.hexdigest() != h_out.hexdigest():
                raise ValueError('full data-region hash mismatch: %s vs %s'
                                 % (h_out.hexdigest(), h_src.hexdigest()))
            print('    full data-region hash OK (%s...)' % h_src.hexdigest()[:16])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--decoder', required=True)
    ap.add_argument('--encoder', required=True)
    ap.add_argument('--out')
    ap.add_argument('--full-hash', action='store_true',
                    help='sha256 the entire data regions of sources and output')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--verify-only', action='store_true',
                    help='do not write; verify an existing --out against the sources')
    args = ap.parse_args()

    t0 = time.time()
    dec = parse_gguf(args.decoder)
    enc = parse_gguf(args.encoder)
    print(dec.summary())
    print(enc.summary())

    if dec.region_size % ALIGN or enc.region_size % ALIGN:
        raise ValueError('source data region not %d-byte sized' % ALIGN)

    coll = set(t[0] for t in dec.tensors) & set(t[0] for t in enc.tensors)
    if coll:
        raise ValueError('tensor name collision: %s' % sorted(coll)[:5])

    merged_meta = dict(dec.meta)
    conflicts = []
    for k, v in enc.meta.items():
        if k in merged_meta:
            if k in DECODER_WINS:
                conflicts.append(k)
                continue
            raise ValueError('unexpected metadata conflict: %s' % k)
        merged_meta[k] = v
    for k, val in OASR_KEYS:
        if k in merged_meta:
            raise ValueError('source already carries %s' % k)
        vb = val.encode('utf-8')
        merged_meta[k] = struct.pack('<I', 8) + struct.pack('<Q', len(vb)) + vb
    print('decoder-wins conflicts: %s' % conflicts)
    print('merged kv=%d tensors=%d' % (len(merged_meta), len(dec.tensors) + len(enc.tensors)))

    hdr, data_start, dec_region = build_header(dec, enc, merged_meta)
    total = len(hdr) + dec_region + enc.region_size
    print('header=%d data_start=%d total_out=%d (%.2f GB)'
          % (len(hdr), data_start, total, total / 1e9))
    if args.dry_run:
        print('dry run: no output written (elapsed %.1fs)' % (time.time() - t0))
        return

    if args.verify_only:
        if not args.out:
            raise ValueError('--verify-only requires --out')
        if not os.path.exists(args.out):
            raise ValueError('output missing for --verify-only: %s' % args.out)
        verify_pack(args.out, dec, enc, data_start, dec_region, full_hash=args.full_hash)
        print('verification passed (total %.0fs)' % (time.time() - t0))
        return

    if not args.out:
        raise ValueError('--out is required when not --dry-run')

    if os.path.exists(args.out):
        raise ValueError('output exists: %s' % args.out)
    tmp = args.out + '.tmp'
    t1 = time.time()
    with open(tmp, 'wb') as out, open(args.decoder, 'rb') as d, open(args.encoder, 'rb') as e:
        out.write(hdr)
        copy_region(d, dec.data_start, dec.region_size, out, 'decoder data', t1)
        copy_region(e, enc.data_start, enc.region_size, out, 'encoder data', time.time())
        out.flush()
        os.fsync(out.fileno())
    os.replace(tmp, args.out)
    print('wrote %s in %.0fs' % (args.out, time.time() - t1))

    verify_pack(args.out, dec, enc, data_start, dec_region, full_hash=args.full_hash)
    print('done (total %.0fs)' % (time.time() - t0))


if __name__ == '__main__':
    main()
