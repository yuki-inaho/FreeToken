#!/usr/bin/env python3
"""Benchmark the Engram row store (``freetoken.engram.store.SafetensorsRowStore``) on disk.

DeepSeek-V4.1-Flash keeps two n-gram hash tables (layers 1 and 14, FP8 ``[384M, 256]`` codes +
E8M0 ``[384M, 8]`` scales, ~101 GB each) that never fit in RAM; every token touches 24 random
256-byte rows per layer, served from the shipped shards through the OS page cache. This script
measures exactly that gather, at the two batch shapes the engine issues:

- **decode**: ``--decode-rows`` (48 = one token x 24 hash columns x 2 layers) random rows per
  step, split over the layers and read concurrently like ``EngramRuntime.prepare_decode``;
- **prefill**: ``--prefill-tokens`` x 24 rows per layer (8192 tokens -> 393,216 rows over two
  layers), the shape of one ``prepare_prefill`` chunk.

for each ``--io`` backend (``mmap``: page faults from numpy fancy-index copies; ``pread``: one
``preadv`` syscall per row, POSIX only), each ``--readers`` thread-pool width, and each id
pattern: ``--warm`` (same ids every repeat: page-cache-hot ceiling), ``--cold`` (fresh uniform
ids every repeat: the miss floor, only truly cold while the table dwarfs the page cache or with
``--drop-caches``) and ``--zipf ALPHA`` (fresh Zipf-skewed ids: the n-gram reuse a real prompt
shows, so the cache warms across repeats). Per configuration it prints p50/p90/p99 wall time of
the whole step, MB/s of bytes actually fetched (unique rows x 256 B; scales are resident), rows/s
of requested rows and the unique-row fraction; ``--json`` dumps everything with raw samples.

``--model DIR`` uses the real shards (``resolve_engram_tables``: ``--engram-dir`` /
``$FREETOKEN_ENGRAM_DIR`` first, then the checkpoint dir); ``--synthetic ROWS`` writes a
temporary hand-made safetensors shard (same layout as ``tests/models/test_dsv41_engram_store.py``)
so the script runs on any box, GPU or not. Torch is never imported.

What to expect (Linux, tables on their own NVMe, 64 readers):

- decode, 48 rows: page-cache-hot < 50 us of copying, plus ~0.1 ms of executor hand-off (the
  store fans out through a thread pool; WSL2 measures ~0.17 ms wall); cold NVMe ~0.2-0.5 ms with
  the rows spread over the readers. With the store's default ``--chunk-rows 2048`` a 48-row batch
  is ONE chunk on ONE thread, i.e. 48 serial misses (~3-5 ms cold at ~80 us per 4 KB random read):
  ``--chunk-rows 4`` shows whether fanning the decode batch out pays. Each extra task costs
  ~0.1 ms on hot rows, so it only wins on real misses -- compare cold, not warm.
- prefill, 8K tokens (393K rows, ~100 MB, ~99.9% unique): ~0.4-0.7 s cold on a Gen4/Gen5 NVMe
  (0.6-1.0 M random 4 KB IOPS), ~0.1 s hot. The readers sweep finds the IOPS knee: below it the
  gather is latency-bound and more threads help linearly.
- ``pread`` pays one syscall per row (~10-12 us each even when hot, ~72 ms for 6K rows in
  WSL2), so it only competes with ``mmap`` when the drive's miss latency, not the syscall, is the
  bottleneck; ``mmap`` is the default for a reason, but pread keeps the page tables small.
- NTFS (native Windows, or /mnt/<drive> 9p/drvfs under WSL2) is an order of magnitude slower on
  both shapes; ext4/xfs on the same drive is what the serving box should use
  (``FREETOKEN_ENGRAM_DIR``, see docs/models.md).

How the numbers feed the Engram decisions in the plan:

- decode p99 vs the GPU step (~20-30 ms at bs=1): if a cold 48-row gather fits with margin, the
  fetch issued at ``prepare_decode`` hides behind the step; if not (NTFS, 9p, serial chunk), the
  rows must be prefetched a step ahead (speculative drafts included) or the tables moved.
- prefill p50 vs the encoder's compute time for one 8K chunk decides whether one-chunk-ahead
  prefetch (``prefetch_prefill_next``) is enough or the chunk must shrink; MB/s and the readers
  knee pick ``FREETOKEN_ENGRAM_READERS`` and ``FREETOKEN_ENGRAM_IO`` (mmap vs pread).
- ``--zipf`` vs ``--cold`` quantifies what the ~65 GiB of page cache left next to the pinned
  experts is worth; the resident-scale load time/bytes (6 GB for both layers) is the RAM that
  ``resident_scales=True`` costs to halve the random reads per row.

Examples::

    # dev box, no checkpoint: 200K-row synthetic table in $TMPDIR
    scripts/engram-bench.py --synthetic 200000 --io mmap,pread --readers 4,16 \\
        --decode-rows 48 --prefill-tokens 256 --repeat 3

    # serving box, real shards on ext4; drop the page cache between cold repeats
    FREETOKEN_ENGRAM_DIR=/mnt/ftw/DeepSeek-V4.1-Flash-engram \\
    scripts/engram-bench.py --model "/media/nuko/New Volume/DeepSeek-V4.1-Flash" \\
        --readers 16,32,64,128 --cold --zipf 1.1 --drop-caches --json engram-bench.json

``--drop-caches`` is best effort as a normal user: ``posix_fadvise(DONTNEED)`` on the shards
(after ``madvise(MADV_DONTNEED)`` on the maps) between repeats, which the kernel honours only for
unmapped, clean pages. A guaranteed cold run needs root: ``sync; echo 3 > /proc/sys/vm/drop_caches``
(done automatically here when running as root), or a table larger than RAM.
"""

from __future__ import annotations

import argparse
import json
import math
import mmap
import os
import shutil
import struct
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))

from freetoken.engram.store import SafetensorsRowStore  # noqa: E402
from freetoken.engram.tables import EngramTableLoc, resolve_engram_tables  # noqa: E402

DIM, SCALE_COLS = 256, 8
SYNTHETIC_LAYER = 1
SYNTHETIC_SHARD = "model-00047-of-00048.safetensors"
DEFAULT_LAYERS = (1, 14)
DEFAULT_ROWS_PER_TOKEN = 24  # (max_ngram_size - 1) x n_heads = 3 x 8


# ----------------------------------------------------------------------------- tables
def synthetic_model(rows: int, out_dir: str, seed: int) -> str:
    """Write ``model.safetensors.index.json`` + one shard holding a random ``[rows, 256]`` table
    (decoy tensor first so the offsets are non-trivial). Chunked so big tables need no RAM."""
    os.makedirs(out_dir, exist_ok=True)
    wname = f"layers.{SYNTHETIC_LAYER}.engram.embed.weight"
    sname = f"layers.{SYNTHETIC_LAYER}.engram.embed.scale"
    decoy = np.arange(100 * 8, dtype=np.uint16).tobytes()
    w_bytes, s_bytes = rows * DIM, rows * SCALE_COLS
    header = {
        "__metadata__": {"format": "pt"},
        "decoy": {"dtype": "BF16", "shape": [100, 8], "data_offsets": [0, len(decoy)]},
        wname: {"dtype": "F8_E4M3", "shape": [rows, DIM], "data_offsets": [len(decoy), len(decoy) + w_bytes]},
        sname: {
            "dtype": "F8_E8M0",
            "shape": [rows, SCALE_COLS],
            "data_offsets": [len(decoy) + w_bytes, len(decoy) + w_bytes + s_bytes],
        },
    }
    hj = json.dumps(header).encode()
    hj += b" " * (-len(hj) % 8)
    rng = np.random.default_rng(seed)
    step = 1 << 16
    with open(os.path.join(out_dir, SYNTHETIC_SHARD), "wb") as f:
        f.write(struct.pack("<Q", len(hj)))
        f.write(hj)
        f.write(decoy)
        for s in range(0, rows, step):
            n = min(step, rows - s)
            vals = rng.integers(0, 256, size=(n, DIM), dtype=np.uint8)
            vals[(vals == 0x7F) | (vals == 0xFF)] = 0x40  # no e4m3 NaN codes
            f.write(vals.tobytes())
        for s in range(0, rows, step):
            n = min(step, rows - s)
            f.write(rng.integers(100, 140, size=(n, SCALE_COLS), dtype=np.uint8).tobytes())
    with open(os.path.join(out_dir, "model.safetensors.index.json"), "w") as f:
        json.dump({"weight_map": {wname: SYNTHETIC_SHARD, sname: SYNTHETIC_SHARD, "decoy": SYNTHETIC_SHARD}}, f)
    return out_dir


def engram_args(model_path: str) -> dict:
    """``engram_*`` fields of the checkpoint: via ``load_args`` when its (torch-backed) import
    works, else straight from ``inference/config.json``; defaults for a bare directory."""
    out = {
        "layer_ids": DEFAULT_LAYERS,
        "num_embeddings": None,
        "head_dim": DIM,
        "rows_per_token": DEFAULT_ROWS_PER_TOKEN,
        "source": "defaults",
    }
    try:
        from freetoken.models.deepseek_v41.args import load_args

        a = load_args(model_path)
        out.update(
            layer_ids=tuple(a.engram_layer_ids),
            num_embeddings=dict(zip(a.engram_layer_ids, a.engram_num_embeddings)),
            head_dim=int(a.engram_head_dim),
            rows_per_token=(int(a.engram_max_ngram_size) - 1) * int(a.engram_n_heads),
            source="load_args",
        )
        return out
    except Exception as e:  # noqa: BLE001 - torch missing, config missing: fall back to the raw JSON
        reason = f"{type(e).__name__}: {e}"
    cfg = os.path.join(model_path, "inference", "config.json")
    if os.path.isfile(cfg):
        with open(cfg) as f:
            raw = json.load(f)
        ids = tuple(int(x) for x in raw.get("engram_layer_ids", DEFAULT_LAYERS))
        emb = raw.get("engram_num_embeddings")
        out.update(
            layer_ids=ids,
            num_embeddings=dict(zip(ids, (int(x) for x in emb))) if emb else None,
            head_dim=int(raw.get("engram_head_dim", DIM)),
            rows_per_token=(int(raw.get("engram_max_ngram_size", 4)) - 1) * int(raw.get("engram_n_heads", 8)),
            source=f"inference/config.json (load_args unavailable: {reason})",
        )
    else:
        out["source"] = f"defaults (no inference/config.json; load_args: {reason})"
    return out


# ----------------------------------------------------------------------------- filesystem
def shard_filesystem(path: str) -> dict:
    """Which filesystem/device the shard lives on: longest ``/proc/mounts`` entry on the same
    ``st_dev`` (Linux), ``GetVolumeInformationW`` on Windows, ``statvfs`` block size where present."""
    real = os.path.realpath(path)
    info = {"path": real, "fstype": "unknown", "mount": None, "device": None, "block_size": None}
    try:
        st_dev = os.stat(real).st_dev
    except OSError:
        return info
    if os.path.isfile("/proc/mounts"):
        best = None
        try:
            with open("/proc/mounts") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) < 3:
                        continue
                    dev, mnt, fstype = parts[0], parts[1].replace("\\040", " "), parts[2]
                    try:
                        if os.stat(mnt).st_dev != st_dev:
                            continue
                    except OSError:
                        continue
                    if real == mnt or real.startswith(mnt.rstrip("/") + "/"):
                        if best is None or len(mnt) > len(best[1]):
                            best = (dev, mnt, fstype)
        except OSError:
            pass
        if best:
            info.update(device=best[0], mount=best[1], fstype=best[2])
    elif os.name == "nt":
        try:
            import ctypes

            root = os.path.splitdrive(real)[0] + "\\"
            buf = ctypes.create_unicode_buffer(64)
            if ctypes.windll.kernel32.GetVolumeInformationW(root, None, 0, None, None, None, buf, 64):
                info.update(fstype=buf.value, mount=root, device=root)
        except Exception:  # noqa: BLE001
            pass
    if hasattr(os, "statvfs"):
        try:
            info["block_size"] = os.statvfs(real).f_bsize
        except OSError:
            pass
    return info


def drop_page_cache(paths: list[str], stores: list[SafetensorsRowStore]) -> str:
    """Evict the shards from the page cache: the real thing as root, ``fadvise(DONTNEED)`` otherwise."""
    if os.name == "nt":
        return "unsupported on Windows"
    if hasattr(os, "geteuid") and os.geteuid() == 0 and os.access("/proc/sys/vm/drop_caches", os.W_OK):
        try:
            os.sync()
            with open("/proc/sys/vm/drop_caches", "w") as f:
                f.write("3\n")
            return "drop_caches"
        except OSError:
            pass
    # mapped pages are skipped by fadvise: unmap them first
    for st in stores:
        for mm, _ in getattr(st, "_maps", {}).values():
            try:
                if hasattr(mm, "madvise") and hasattr(mmap, "MADV_DONTNEED"):
                    mm.madvise(mmap.MADV_DONTNEED)
            except (OSError, ValueError):
                pass
    if not (hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED")):
        return "unsupported (no posix_fadvise)"
    for p in set(paths):
        try:
            fd = os.open(p, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
        except OSError:
            pass
    return "fadvise"


# ----------------------------------------------------------------------------- ids
def coprime_multiplier(rng: np.random.Generator, rows: int) -> int:
    """A random odd multiplier coprime to ``rows`` (so ``rank * m % rows`` permutes the rows)
    small enough that ``rank * m`` stays inside int64."""
    hi = max(3, min(1 << 40, (1 << 62) // max(rows, 1)))
    while True:
        m = int(rng.integers(1, hi)) | 1
        if math.gcd(m, rows) == 1:
            return m


def zipf_ids(rng: np.random.Generator, alpha: float, n: int, rows: int, mult: int) -> np.ndarray:
    """``n`` row ids with Zipf(alpha)-distributed popularity, the hot rows scattered over the table."""
    out = np.empty(n, dtype=np.int64)
    filled = 0
    while filled < n:
        z = rng.zipf(alpha, size=(n - filled) + (n - filled) // 8 + 16).astype(np.int64)
        z = z[(z >= 1) & (z <= rows)] - 1
        take = min(z.shape[0], n - filled)
        out[filled : filled + take] = z[:take]
        filled += take
    return (out * mult) % rows


class IdSource:
    """Per-layer row ids for one (pattern, batch): fixed for ``warm``, fresh per repeat otherwise."""

    def __init__(self, pattern: str, alpha: float | None, per_layer_rows: list[int], counts: list[int], seed: int):
        self.pattern, self.alpha, self.rows, self.counts = pattern, alpha, per_layer_rows, counts
        self.rng = np.random.default_rng(seed)
        self.mults = [coprime_multiplier(self.rng, r) for r in per_layer_rows]
        self._fixed = self._draw() if pattern == "warm" else None

    def _draw(self) -> list[np.ndarray]:
        ids = []
        for li, (rows, n) in enumerate(zip(self.rows, self.counts)):
            if self.pattern == "zipf":
                ids.append(zipf_ids(self.rng, self.alpha, n, rows, self.mults[li]))
            else:
                ids.append(self.rng.integers(0, rows, size=n, dtype=np.int64))
        return ids

    def next(self) -> list[np.ndarray]:
        return self._fixed if self._fixed is not None else self._draw()


# ----------------------------------------------------------------------------- bench
def percentiles(samples_ms: list[float]) -> tuple[float, float, float]:
    p = np.percentile(np.asarray(samples_ms, dtype=np.float64), [50, 90, 99])
    return float(p[0]), float(p[1]), float(p[2])


def _read_step(stores, pool, ids, vals, scales) -> float:
    """One engine step: every layer's rows read concurrently (like ``FetchDriver.submit_multi``)."""
    t0 = time.perf_counter()
    futs = [
        pool.submit(st.read_rows, i, vals[li][: i.shape[0]], scales[li][: i.shape[0]])
        for li, (st, i) in enumerate(zip(stores, ids))
    ]
    for f in futs:
        f.result()
    return time.perf_counter() - t0


def run_config(
    stores: list[SafetensorsRowStore],
    pool: ThreadPoolExecutor,
    source: IdSource,
    vals: list[np.ndarray],
    scales: list[np.ndarray],
    repeat: int,
    drop: bool,
    paths: list[str],
) -> dict:
    samples_ms, bytes_per, uniq_frac = [], [], []
    if source.pattern == "warm":  # one untimed pass fills the cache
        _read_step(stores, pool, source.next(), vals, scales)
    for _ in range(repeat):
        if drop and source.pattern == "cold":
            drop_page_cache(paths, stores)
        ids = source.next()
        before = sum(s.stats.bytes_read for s in stores)
        dt = _read_step(stores, pool, ids, vals, scales)
        samples_ms.append(dt * 1e3)
        bytes_per.append(sum(s.stats.bytes_read for s in stores) - before)
        total = sum(int(i.shape[0]) for i in ids)
        uniq_frac.append(sum(int(np.unique(i).shape[0]) for i in ids) / max(1, total))
    p50, p90, p99 = percentiles(samples_ms)
    rows = sum(source.counts)
    mb_s = (float(np.mean(bytes_per)) / 1e6) / (p50 / 1e3) if p50 > 0 else float("inf")
    return {
        "rows": rows,
        "rows_per_layer": list(source.counts),
        "unique_frac": float(np.mean(uniq_frac)),
        "samples_ms": samples_ms,
        "bytes_fetched": [int(b) for b in bytes_per],
        "p50_ms": p50,
        "p90_ms": p90,
        "p99_ms": p99,
        "mb_s": mb_s,
        "rows_s": rows / (p50 / 1e3) if p50 > 0 else float("inf"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="See the module docstring for the numbers to expect and what they decide.",
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--model", help="checkpoint dir (model.safetensors.index.json + inference/config.json)")
    src.add_argument("--synthetic", type=int, metavar="ROWS", help="write a temporary ROWS x 256 table instead of using a checkpoint")
    ap.add_argument("--engram-dir", default=None, help="where the engram shards live (default: $FREETOKEN_ENGRAM_DIR, then --model)")
    ap.add_argument("--synthetic-dir", default=None, help="parent dir for the synthetic shard (default: the system temp dir; put it on the drive under test)")
    ap.add_argument("--keep-synthetic", action="store_true", help="do not delete the synthetic shard afterwards")
    ap.add_argument("--layers", default=None, help="engram layer ids, comma separated (default: all from inference/config.json)")
    ap.add_argument("--io", default="mmap,pread", help="store backends to test (default: mmap,pread; pread is skipped where os.pread is missing)")
    ap.add_argument("--readers", default="16,32,64,128", help="thread-pool widths to test")
    ap.add_argument("--chunk-rows", type=int, default=2048, help="rows per reader task (SafetensorsRowStore chunk_rows; default 2048)")
    ap.add_argument("--decode-rows", type=int, default=48, help="rows per decode step over all layers (default 48 = 1 token x 24 x 2 layers)")
    ap.add_argument("--prefill-tokens", type=int, default=8192, help="tokens per prefill chunk; rows = tokens x 24 per layer")
    ap.add_argument("--rows-per-token", type=int, default=None, help="hash columns per token per layer (default: from the args, else 24)")
    ap.add_argument("--repeat", type=int, default=5, help="timed repeats per configuration")
    ap.add_argument("--warm", action="store_true", help="same ids every repeat (page-cache-hot)")
    ap.add_argument("--cold", action="store_true", help="fresh uniform random ids every repeat")
    ap.add_argument("--zipf", type=float, default=None, metavar="ALPHA", help="fresh Zipf(ALPHA)-skewed ids every repeat (ALPHA > 1, e.g. 1.1)")
    ap.add_argument("--drop-caches", action="store_true", help="evict the shards from the page cache before each cold repeat (best effort; see docstring)")
    ap.add_argument("--no-resident-scales", action="store_true", help="read scales from disk per row instead of holding them in RAM")
    ap.add_argument("--skip-prefill", action="store_true", help="only the decode-sized batch")
    ap.add_argument("--skip-decode", action="store_true", help="only the prefill-sized batch")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None, help="write all results (with raw samples) to this file")
    args = ap.parse_args()

    if args.zipf is not None and args.zipf <= 1.0:
        ap.error("--zipf ALPHA must be > 1")
    if args.repeat < 1:
        ap.error("--repeat must be >= 1")
    patterns: list[tuple[str, float | None]] = []
    if args.warm:
        patterns.append(("warm", None))
    if args.cold:
        patterns.append(("cold", None))
    if args.zipf is not None:
        patterns.append(("zipf", args.zipf))
    if not patterns:
        patterns = [("warm", None), ("cold", None)]
    ios = [x.strip() for x in args.io.split(",") if x.strip()]
    readers_list = [int(x) for x in args.readers.split(",") if x.strip()]
    for io in ios:
        if io not in ("mmap", "pread"):
            ap.error(f"unknown --io {io!r}")
    if "pread" in ios and not (hasattr(os, "preadv") or hasattr(os, "pread")):
        print("note: io=pread needs os.pread (POSIX); skipping it", file=sys.stderr)
        ios = [x for x in ios if x != "pread"]
    if not ios or not readers_list:
        ap.error("nothing to run: empty --io or --readers")

    # ---- tables
    tmp_dir = None
    if args.synthetic is not None:
        if args.synthetic < 1:
            ap.error("--synthetic ROWS must be >= 1")
        tmp_dir = tempfile.mkdtemp(prefix="engram-bench-", dir=args.synthetic_dir)
        t0 = time.perf_counter()
        model_path = synthetic_model(args.synthetic, os.path.join(tmp_dir, "model"), args.seed)
        mb = args.synthetic * (DIM + SCALE_COLS) / 1e6
        print(f"synthetic table: {args.synthetic:,} rows x {DIM} B (+{SCALE_COLS} B scales) = {mb:.1f} MB written to {model_path} in {time.perf_counter() - t0:.2f} s")
        layer_ids: tuple[int, ...] = (SYNTHETIC_LAYER,)
        meta = {
            "layer_ids": layer_ids,
            "num_embeddings": {SYNTHETIC_LAYER: args.synthetic},
            "head_dim": DIM,
            "rows_per_token": DEFAULT_ROWS_PER_TOKEN,
            "source": "synthetic",
        }
        engram_dir = ""
    else:
        model_path = args.model
        meta = engram_args(model_path)
        layer_ids = tuple(meta["layer_ids"])
        engram_dir = args.engram_dir
    if args.layers:
        layer_ids = tuple(int(x) for x in args.layers.split(",") if x.strip())
    rows_per_token = args.rows_per_token or int(meta["rows_per_token"])
    try:
        locs = resolve_engram_tables(model_path, layer_ids, engram_dir, expected_rows=meta["num_embeddings"], expected_dim=meta["head_dim"])
    except Exception as e:  # noqa: BLE001
        print(f"error: cannot locate the engram tables: {type(e).__name__}: {e}", file=sys.stderr)
        if tmp_dir and not args.keep_synthetic:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        return 2
    locs_list: list[EngramTableLoc] = [locs[l] for l in layer_ids]
    n_layers = len(locs_list)
    paths = sorted({p for loc in locs_list for p in (loc.path, loc.scale_path)})

    print(f"engram args: layers {list(layer_ids)}, {rows_per_token} rows/token/layer ({meta['source']})")
    fs_info = {}
    for p in paths:
        fi = fs_info[p] = shard_filesystem(p)
        print(f"shard {p}: {os.path.getsize(p) / 1e9:.2f} GB on {fi['fstype']} ({fi['device'] or '?'} at {fi['mount'] or '?'}; bsize {fi['block_size']})")
    for loc in locs_list:
        print(
            f"  layer {loc.layer_id}: rows {loc.rows:,} x {loc.dim} B @ {loc.weight_offset:,}, scales x {loc.scale_cols} B @ {loc.scale_offset:,}"
            f" ({loc.weight_bytes / 1e9:.2f} GB + {loc.scale_bytes / 1e9:.2f} GB)"
        )
    if any(p == "cold" for p, _ in patterns) and not args.drop_caches:
        table_gb = sum(loc.weight_bytes for loc in locs_list) / 1e9
        print(f"note: --cold without --drop-caches only stays cold while the {table_gb:.2f} GB of tables exceed the page cache")

    # ---- batches
    dec_per_layer = [max(1, math.ceil(args.decode_rows / n_layers))] * n_layers
    pre_per_layer = [args.prefill_tokens * rows_per_token] * n_layers
    batches: list[tuple[str, list[int]]] = []
    if not args.skip_decode:
        batches.append(("decode", dec_per_layer))
    if not args.skip_prefill:
        batches.append(("prefill", pre_per_layer))
    if not batches:
        ap.error("nothing to run (both --skip-decode and --skip-prefill)")
    max_rows = max(max(c) for _, c in batches)
    # output arrays allocated once (the engine's staging ring does the same); touched so the
    # first-use page faults of the destination are not timed
    vals = [np.zeros((max_rows, loc.dim), dtype=np.uint8) for loc in locs_list]
    scales = [np.zeros((max_rows, loc.scale_cols), dtype=np.uint8) for loc in locs_list]
    for v, s in zip(vals, scales):
        v[:] = 1
        s[:] = 1
    print(
        "batches: "
        + ", ".join(f"{name} = {sum(c):,} rows ({c[0]:,}/layer)" for name, c in batches)
        + f"; patterns: {', '.join(p if a is None else f'{p}({a:g})' for p, a in patterns)}; repeat {args.repeat}; chunk_rows {args.chunk_rows}"
    )

    resident = not args.no_resident_scales
    shared_scales: dict[int, np.ndarray] = {}
    resident_info: dict[int, dict] = {}
    results = []
    pool = ThreadPoolExecutor(max_workers=n_layers, thread_name_prefix="engram-bench")
    header = (
        f"{'io':<6}{'readers':>8}{'pattern':>10}{'batch':>9}{'rows':>10}{'uniq%':>7}"
        f"{'p50 ms':>10}{'p90 ms':>10}{'p99 ms':>10}{'MB/s':>9}{'rows/s':>12}"
    )
    print()
    print(header)
    print("-" * len(header))
    try:
        for io in ios:
            for readers in readers_list:
                stores = [
                    SafetensorsRowStore(loc, io=io, readers=readers, resident_scales=resident, chunk_rows=args.chunk_rows)
                    for loc in locs_list
                ]
                try:
                    if resident:
                        for loc, st in zip(locs_list, stores):
                            arr = shared_scales.get(loc.layer_id)
                            if arr is None:
                                t0 = time.perf_counter()
                                arr = st.load_resident_scales()
                                dt = time.perf_counter() - t0
                                shared_scales[loc.layer_id] = arr
                                resident_info[loc.layer_id] = {"bytes": int(loc.scale_bytes), "load_s": dt}
                                print(f"  resident scales layer {loc.layer_id}: {loc.scale_bytes / 1e9:.2f} GB in {dt:.3f} s ({loc.scale_bytes / 1e9 / max(dt, 1e-9):.2f} GB/s)")
                            else:  # share the one copy across configurations instead of re-reading GBs per config
                                with st._scales_lock:
                                    st._scales_ram = arr
                    seed = args.seed
                    for pattern, alpha in patterns:
                        for bname, counts in batches:
                            seed += 1
                            source = IdSource(pattern, alpha, [loc.rows for loc in locs_list], counts, seed)
                            r = run_config(stores, pool, source, vals, scales, args.repeat, args.drop_caches, paths)
                            r.update(io=io, readers=readers, chunk_rows=args.chunk_rows, pattern=pattern, zipf_alpha=alpha, batch=bname)
                            results.append(r)
                            pl = pattern if alpha is None else f"zipf{alpha:g}"
                            print(
                                f"{io:<6}{readers:>8}{pl:>10}{bname:>9}{r['rows']:>10,}{100 * r['unique_frac']:>7.1f}"
                                f"{r['p50_ms']:>10.3f}{r['p90_ms']:>10.3f}{r['p99_ms']:>10.3f}{r['mb_s']:>9.1f}{r['rows_s']:>12,.0f}",
                                flush=True,
                            )
                finally:
                    for st in stores:
                        st.close()
    finally:
        pool.shutdown(wait=True)
        if tmp_dir and not args.keep_synthetic:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        elif tmp_dir:
            print(f"kept synthetic table in {tmp_dir}")

    if args.json:
        out = {
            "args": vars(args),
            "model_path": model_path,
            "engram_args": {k: ({str(a): b for a, b in v.items()} if isinstance(v, dict) else v) for k, v in meta.items()},
            "layers": [
                {"layer_id": loc.layer_id, "rows": loc.rows, "dim": loc.dim, "scale_cols": loc.scale_cols, "path": loc.path, "scale_path": loc.scale_path}
                for loc in locs_list
            ],
            "filesystems": fs_info,
            "resident_scales": {str(k): v for k, v in resident_info.items()},
            "platform": {"os": os.name, "platform": sys.platform, "python": sys.version.split()[0], "numpy": np.__version__, "cpus": os.cpu_count()},
            "results": results,
        }
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
