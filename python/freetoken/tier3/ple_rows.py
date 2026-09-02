"""Tier-3 for Qwen3.8-Flash-Next's 47.7 GiB PLE n-gram table.

The table is the single largest host allocation in the model -- larger than the entire expert
corpus is on a small box -- but it is touched astonishingly lightly: ``ple_layer_ids`` has one
entry, and a token reads ``num_ngram_heads`` (16) rows of ``head_dim`` (160) bytes, i.e.
**2,560 B/token**. A 3-row MTP verify step reads 7.7 KB. So this is a pure *latency* problem
over a huge, cold, Zipf-distributed key space -- exactly what a small RAM cache over NVMe is
for, and tiering it frees ~44 GiB for expert slots at negligible cost.

Two things make it tractable:

* **Row ids are a deterministic hash of the preceding tokens** (``PLEEmbedding.row_ids``), and
  that method is pure torch, so the host can compute the *same* ids on CPU tensors -- an exact
  mirror by construction rather than a hand-rewritten hash that can drift.
* The cache hands the gather kernel **slot indices** instead of row ids. ``ple_gather_rows``
  only ever computes ``base + idx * head_dim``, so it is unchanged; only the values of ``idx``
  differ. That also means the cache does not have to be direct-mapped (where two live rows
  colliding in one step would silently return the wrong embedding) -- the resolver assigns
  slots explicitly and can guarantee every row of a step is simultaneously resident.
"""

from __future__ import annotations

import mmap
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import torch

# 48 row reads per MTP verify step at ~100 us each is 5 ms if issued serially -- a quarter of
# the whole step. They are independent, so issue them concurrently; measured p50 falls from
# 5.01 ms to well under 1 ms.
DEFAULT_READERS = 16

ALIGN = 4096
_SHARD_RE = re.compile(r"\.ple\.ple_embedding\.ngram_embedding\.shard_(?P<shard>\d+)\.weight$")
_SCALE_SUFFIX = ".ple.ple_embedding.ngram_embedding.weight_scale"


@dataclass(frozen=True)
class PleLayout:
    """Where each logical table row lives on disk."""

    paths: tuple[str, ...]          # one per shard index
    offsets: tuple[int, ...]        # byte offset of the shard's data inside its file
    rows_per_shard: int
    head_dim: int                   # bytes per row (fp8 -> 1 byte per element)
    num_rows: int
    scale: float

    def locate(self, row: int) -> tuple[str, int]:
        """-> (path, byte offset of this row within that file)."""
        shard, within = divmod(int(row), self.rows_per_shard)
        return self.paths[shard], self.offsets[shard] + within * self.head_dim


def read_ple_layout(model_path: str) -> PleLayout:
    """Parse the PLE shards' safetensors headers into a row-addressable layout.

    Deliberately reads the *original* checkpoint: ``ft checkpoint`` does not carry the PLE table
    into FTW (``qwen4_exp.weight._rename`` drops it from ``iter_weights``), and it does not need
    to -- rows are 160 B, so a read is a block read regardless of alignment.
    """
    from freetoken.models.qwen4_exp.weight import _safetensors_header, _ple_table_files

    parts: dict[int, tuple[str, int]] = {}
    rows = cols = 0
    scale = 1.0
    for path in _ple_table_files(model_path):
        header, base = _safetensors_header(path)
        for key, meta in header.items():
            if key == "__metadata__":
                continue
            if key.endswith(_SCALE_SUFFIX):
                import safetensors

                with safetensors.safe_open(path, framework="pt", device="cpu") as f:
                    scale = float(f.get_tensor(key).reshape(()))
                continue
            m = _SHARD_RE.search(key)
            if m is None:
                continue
            rows, cols = meta["shape"]
            parts[int(m.group("shard"))] = (path, base + meta["data_offsets"][0])
    if not parts:
        raise ValueError(f"{model_path} has no PLE n-gram shards")
    order = sorted(parts)
    assert order == list(range(len(order))), f"missing PLE shards: {order[:8]}"
    return PleLayout(
        paths=tuple(parts[i][0] for i in order),
        offsets=tuple(parts[i][1] for i in order),
        rows_per_shard=rows, head_dim=cols, num_rows=rows * len(order), scale=scale,
    )


class PleDiskReader:
    """O_DIRECT row reads over the PLE shards.

    A 160-byte row is never block-aligned, so each read pulls the aligned window containing it
    into a small per-thread bounce and copies the row out. That is the right trade here and not
    the one the expert tier avoids: the window is 1-2 blocks, the copy is 160 bytes, and the
    alternative (repacking 47.7 GiB so rows are block-aligned) would inflate the table 25x.
    """

    def __init__(self, layout: PleLayout, readers: int = DEFAULT_READERS) -> None:
        self.layout = layout
        self.readers = max(1, readers)
        # The bounce window stays per-thread (a shared one would be a data race), but the file
        # descriptors are SHARED: pread carries its own offset, so there is no file-position
        # lock to dodge. Caching an fd per (thread, shard) multiplied the count by the pool size
        # and, together with the expert tier's pools, exhausted a 1024 fd limit -- which killed
        # a live server inside an unrelated JIT build ("Too many open files").
        self._tls = threading.local()
        self._fds: dict[str, int] = {}
        self._lock = threading.Lock()
        self._pool: ThreadPoolExecutor | None = None
        self.reads = 0

    def _fd(self, path: str) -> int:
        """One O_DIRECT fd per shard, shared by every reader thread (see __init__)."""
        fd = self._fds.get(path)
        if fd is None:
            with self._lock:
                fd = self._fds.get(path)
                if fd is None:
                    fd = self._fds[path] = os.open(path, os.O_RDONLY | os.O_DIRECT)
        return fd

    def _bounce_view(self) -> memoryview:
        b = getattr(self._tls, "bounce", None)
        if b is None:
            b = self._tls.bounce = mmap.mmap(-1, 2 * ALIGN)  # anonymous mmaps are page-aligned
        return memoryview(b)

    def read_row(self, row: int, dst: memoryview) -> None:
        """Read logical row ``row`` (head_dim bytes) into ``dst``."""
        hd = self.layout.head_dim
        assert len(dst) == hd, (len(dst), hd)
        path, off = self.layout.locate(row)
        head = off % ALIGN
        span = ((head + hd + ALIGN - 1) // ALIGN) * ALIGN
        mv = self._bounce_view()[:span]
        got = 0
        while got < span:
            n = os.preadv(self._fd(path), [mv[got:]], off - head + got)
            if n <= 0:
                # a read at the very end of a shard can legitimately be short of the padded
                # window; only the row's own bytes have to have landed
                if got >= head + hd:
                    break
                raise OSError(f"short PLE read at {path}+{off - head + got}")
            got += n
        dst[:] = mv[head:head + hd]
        self.reads += 1

    def read_rows(self, rows: list[int], dsts: list[memoryview]) -> None:
        """Read many rows concurrently (the whole point -- see DEFAULT_READERS)."""
        if len(rows) <= 1:
            for r, d in zip(rows, dsts):
                self.read_row(r, d)
            return
        if self._pool is None:
            self._pool = ThreadPoolExecutor(self.readers, thread_name_prefix="ft-ple")
        list(self._pool.map(self.read_row, rows, dsts))

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None
        with self._lock:
            for fd in self._fds.values():
                os.close(fd)
            self._fds.clear()


class PleRowCache:
    """Pinned row cache over the on-disk table; a drop-in ``PLETableBackend``.

    ``resolve`` is the host half: given the row ids of an upcoming step it guarantees every one
    of them is resident *simultaneously* and returns their slot indices. Eviction is
    round-robin over slots not claimed by the current step, which keeps the "all rows of a step
    are live at once" invariant that a direct-mapped cache cannot give.
    """

    def __init__(self, layout: PleLayout, cache_rows: int, *,
                 device: torch.device | None = None, pin: bool = True,
                 readers: int = DEFAULT_READERS) -> None:
        assert cache_rows > 0
        self.layout = layout
        self.cache_rows = cache_rows
        self.head_dim = layout.head_dim
        self.reader = PleDiskReader(layout, readers=readers)
        self.weight = _alloc_rows(cache_rows, layout.head_dim, pin)
        self.tag = torch.full((cache_rows,), -1, dtype=torch.int64)
        self._pos: dict[int, int] = {}     # row id -> slot
        self._pending: dict[int, int] = {}  # row id -> slot, within one resolve() call
        self._clock = 0
        self.hits = 0
        self.misses = 0
        self.device = device

    @property
    def bytes(self) -> int:
        return self.cache_rows * self.head_dim

    def resolve(self, row_ids: torch.Tensor) -> torch.Tensor:
        """Ensure every id in ``row_ids`` (CPU int64) is resident; return the slot per id.

        Rows claimed earlier in the same call are never evicted by a later miss in that call,
        so the returned slots are all simultaneously valid.
        """
        flat = row_ids.reshape(-1).tolist()
        claimed: set[int] = set()
        out = torch.empty(len(flat), dtype=torch.int64)
        miss_rows: list[int] = []
        miss_slots: list[int] = []
        miss_bufs: list[memoryview] = []

        # phase 1 -- classify and assign slots (host-only, no I/O). A row appearing twice in one
        # call is fetched once; slots claimed here cannot be evicted later in the same call.
        for i, rid in enumerate(flat):
            slot = self._pos.get(rid)
            if slot is None:
                pending = self._pending.get(rid)
                if pending is not None:
                    slot = pending
                else:
                    self.misses += 1
                    slot = self._evict(claimed)
                    miss_rows.append(rid)
                    miss_slots.append(slot)
                    miss_bufs.append(memoryview(bytearray(self.head_dim)))
                    self._pending[rid] = slot
            else:
                self.hits += 1
            claimed.add(slot)
            out[i] = slot

        # phase 2 -- one parallel batch of reads
        if miss_rows:
            self.reader.read_rows(miss_rows, miss_bufs)
            for rid, slot, buf in zip(miss_rows, miss_slots, miss_bufs):
                self.weight[slot] = torch.frombuffer(bytearray(buf), dtype=self.weight.dtype)
                old = int(self.tag[slot])
                if old >= 0:
                    self._pos.pop(old, None)
                self.tag[slot] = rid
                self._pos[rid] = slot
            self._pending.clear()
        return out.view(row_ids.shape)

    def _evict(self, claimed: set[int]) -> int:
        for _ in range(self.cache_rows):
            slot = self._clock
            self._clock = (self._clock + 1) % self.cache_rows
            if slot not in claimed:
                return slot
        raise RuntimeError(
            f"every one of {self.cache_rows} PLE cache rows is claimed by the current step; "
            f"raise the PLE row cache size"
        )


def _alloc_rows(rows: int, head_dim: int, pin: bool) -> torch.Tensor:
    if pin:
        from freetoken.kernel.pinned import alloc_pinned_tensor

        raw = alloc_pinned_tensor(rows * head_dim, dtype=torch.uint8)
        return raw.view(torch.float8_e4m3fn).view(rows, head_dim)
    return torch.zeros((rows, head_dim), dtype=torch.float8_e4m3fn)
