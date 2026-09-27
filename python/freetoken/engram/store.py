"""Row stores: gather Engram table rows (FP8 codes + E8M0 scales) by row id.

The tables never fit in RAM, so rows are fetched from the shipped shards on demand and the OS
page cache is the hot-row cache. Two backends, both fanning out over a thread pool because a
cold gather is IOPS-bound (a prefill chunk asks for ~400K random 256-byte rows):

- ``mmap``: one read-only map of the shard (``MADV_RANDOM``: no read-ahead, which would pull
  4 KB+ of neighbours per 256-byte row) and numpy fancy-index copies; numpy drops the GIL
  during the copy, so page faults from different threads overlap. Portable (Windows CI).
- ``pread``: ``os.preadv`` straight into the destination row (POSIX only), one syscall per
  row, ``POSIX_FADV_RANDOM`` on the fd.

The scale tensor (8 bytes/row, 3 GB per layer for V4.1) is optionally kept resident so each
row costs one random read, not two. Ids are deduplicated before touching the disk: the same
n-gram recurs constantly within a chunk. Torch-free by design — this is the host side.
"""

from __future__ import annotations

import gc
import mmap
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .tables import EngramTableLoc


class EngramRowSource(Protocol):
    """Anything that can fill ``vals[R, dim]`` / ``scales[R, scale_cols]`` (uint8) for ``row_ids[R]``."""

    rows: int
    dim: int
    scale_cols: int

    def read_rows(self, row_ids: np.ndarray, vals: np.ndarray, scales: np.ndarray) -> None: ...

    def close(self) -> None: ...


@dataclass
class RowStoreStats:
    calls: int = 0
    rows_requested: int = 0
    rows_unique: int = 0
    bytes_read: int = 0
    wall_s: float = 0.0
    max_call_s: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, requested: int, unique: int, nbytes: int, wall: float) -> None:
        with self._lock:
            self.calls += 1
            self.rows_requested += requested
            self.rows_unique += unique
            self.bytes_read += nbytes
            self.wall_s += wall
            self.max_call_s = max(self.max_call_s, wall)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "calls": self.calls,
                "rows_requested": self.rows_requested,
                "rows_unique": self.rows_unique,
                "bytes_read": self.bytes_read,
                "wall_s": self.wall_s,
                "max_call_s": self.max_call_s,
            }


def _check_request(row_ids: np.ndarray, vals: np.ndarray, scales: np.ndarray, rows: int, dim: int, scale_cols: int):
    ids = np.asarray(row_ids).reshape(-1)
    if ids.dtype != np.int64:
        ids = ids.astype(np.int64)
    R = ids.shape[0]
    if vals.shape != (R, dim) or vals.dtype != np.uint8:
        raise ValueError(f"vals must be uint8[{R}, {dim}], got {vals.dtype}{vals.shape}")
    if scales.shape != (R, scale_cols) or scales.dtype != np.uint8:
        raise ValueError(f"scales must be uint8[{R}, {scale_cols}], got {scales.dtype}{scales.shape}")
    if R and (ids.min() < 0 or ids.max() >= rows):
        raise IndexError(f"row id out of range [0, {rows})")
    return ids


class ArrayRowStore:
    """In-RAM oracle: the whole table as ``vals uint8[rows, dim]`` / ``scales uint8[rows, scale_cols]``."""

    def __init__(self, vals: np.ndarray, scales: np.ndarray):
        vals = np.ascontiguousarray(vals, dtype=np.uint8)
        scales = np.ascontiguousarray(scales, dtype=np.uint8)
        if vals.ndim != 2 or scales.ndim != 2 or vals.shape[0] != scales.shape[0]:
            raise ValueError(f"vals {vals.shape} / scales {scales.shape} must be [rows, *] with equal rows")
        self._vals, self._scales = vals, scales
        self.rows, self.dim = vals.shape
        self.scale_cols = scales.shape[1]
        self.stats = RowStoreStats()

    def read_rows(self, row_ids: np.ndarray, vals: np.ndarray, scales: np.ndarray) -> None:
        t0 = time.perf_counter()
        ids = _check_request(row_ids, vals, scales, self.rows, self.dim, self.scale_cols)
        np.take(self._vals, ids, axis=0, out=vals)
        np.take(self._scales, ids, axis=0, out=scales)
        self.stats.add(ids.shape[0], int(np.unique(ids).shape[0]), ids.shape[0] * (self.dim + self.scale_cols), time.perf_counter() - t0)

    def close(self) -> None:
        pass


class SafetensorsRowStore:
    """Rows straight from a safetensors shard at the offsets in an :class:`EngramTableLoc`.

    ``io``: ``"mmap"`` or ``"pread"`` (see the module docstring). ``readers``: thread-pool
    width. ``resident_scales``: read the whole scale region once (lazily, in chunks) and serve
    scales from RAM.
    """

    def __init__(
        self,
        loc: EngramTableLoc,
        io: str = "mmap",
        readers: int = 64,
        resident_scales: bool = True,
        *,
        chunk_rows: int = 2048,
    ):
        if io not in ("mmap", "pread"):
            raise ValueError(f"io must be 'mmap' or 'pread', got {io!r}")
        if io == "pread" and not hasattr(os, "preadv") and not hasattr(os, "pread"):
            raise OSError("io='pread' needs os.pread (POSIX); use io='mmap'")
        self.loc = loc
        self.io = io
        self.rows, self.dim, self.scale_cols = loc.rows, loc.dim, loc.scale_cols
        self.readers = max(1, int(readers))
        self.chunk_rows = max(1, int(chunk_rows))
        self.stats = RowStoreStats()
        self._pool = ThreadPoolExecutor(max_workers=self.readers, thread_name_prefix="engram-io")
        self._resident_scales = bool(resident_scales)
        self._scales_ram: np.ndarray | None = None
        self._scales_lock = threading.Lock()
        self._closed = False

        self._fds: dict[str, int] = {}
        self._maps: dict[str, tuple[mmap.mmap, np.ndarray]] = {}
        for path in {loc.path, loc.scale_path}:
            fd = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0))
            self._fds[path] = fd
            if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_RANDOM"):
                try:
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)
                except OSError:
                    pass
            if io == "mmap":
                mm = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
                if hasattr(mm, "madvise") and hasattr(mmap, "MADV_RANDOM"):
                    try:
                        mm.madvise(mmap.MADV_RANDOM)
                    except (OSError, ValueError):
                        pass
                self._maps[path] = (mm, np.frombuffer(mm, dtype=np.uint8))
        if io == "mmap":
            wview = self._maps[loc.path][1]
            self._weight_view = wview[loc.weight_offset : loc.weight_offset + loc.weight_bytes].reshape(loc.rows, loc.dim)
            sview = self._maps[loc.scale_path][1]
            self._scale_view = sview[loc.scale_offset : loc.scale_offset + loc.scale_bytes].reshape(loc.rows, loc.scale_cols)

    # ------------------------------------------------------------------ resident scales
    def load_resident_scales(self, chunk_bytes: int = 256 << 20) -> np.ndarray:
        """Read the whole scale tensor into RAM (chunked sequential reads); idempotent."""
        with self._scales_lock:
            if self._scales_ram is not None:
                return self._scales_ram
            loc = self.loc
            buf = np.empty(loc.scale_bytes, dtype=np.uint8)
            fd = self._fds[loc.scale_path]
            pos = 0
            while pos < loc.scale_bytes:
                n = min(chunk_bytes, loc.scale_bytes - pos)
                if hasattr(os, "preadv"):
                    got = os.preadv(fd, [memoryview(buf[pos : pos + n])], loc.scale_offset + pos)
                else:
                    data = os.pread(fd, n, loc.scale_offset + pos)
                    got = len(data)
                    buf[pos : pos + got] = np.frombuffer(data, dtype=np.uint8)
                if got <= 0:
                    raise OSError(f"{loc.scale_path}: short read of the scale tensor at {pos}")
                pos += got
            self._scales_ram = buf.reshape(loc.rows, loc.scale_cols)
            self.stats.add(0, 0, loc.scale_bytes, 0.0)
            return self._scales_ram

    # ------------------------------------------------------------------ gather
    def read_rows(self, row_ids: np.ndarray, vals: np.ndarray, scales: np.ndarray) -> None:
        if self._closed:
            raise RuntimeError("store is closed")
        t0 = time.perf_counter()
        ids = _check_request(row_ids, vals, scales, self.rows, self.dim, self.scale_cols)
        R = ids.shape[0]
        if R == 0:
            return
        uniq, inverse = np.unique(ids, return_inverse=True)
        U = uniq.shape[0]
        # The gather runs over the SORTED unique ids (dedup + monotone reads); the rows must
        # come back in REQUEST order. Aliasing the caller's buffers is only right when the
        # request already is the sorted unique sequence -- writing the sorted gather straight
        # into ``vals`` for any other all-distinct request permuted the rows (found on the
        # 5090, 2026-09-10: every token's 24 hash columns and the tokens of a chunk swapped
        # rows unless the chunk happened to repeat an n-gram, which is why long / repetitive
        # prompts looked fine and short single-turn prompts came out as garbage).
        direct = U == R and np.array_equal(uniq, ids)
        uvals = vals if direct else np.empty((U, self.dim), dtype=np.uint8)
        uscales = scales if direct else np.empty((U, self.scale_cols), dtype=np.uint8)

        want_scales = True
        if self._resident_scales:
            np.take(self.load_resident_scales(), uniq, axis=0, out=uscales)
            want_scales = False

        futures = []
        for s in range(0, U, self.chunk_rows):
            sl = slice(s, min(U, s + self.chunk_rows))
            futures.append(self._pool.submit(self._gather_chunk, uniq[sl], uvals[sl], uscales[sl] if want_scales else None))
        for f in futures:
            f.result()

        if not direct:
            np.take(uvals, inverse, axis=0, out=vals)
            np.take(uscales, inverse, axis=0, out=scales)
        nbytes = U * (self.dim + (self.scale_cols if want_scales else 0))
        self.stats.add(R, U, nbytes, time.perf_counter() - t0)

    def _gather_chunk(self, ids: np.ndarray, vals: np.ndarray, scales: np.ndarray | None) -> None:
        if self.io == "mmap":
            np.take(self._weight_view, ids, axis=0, out=vals)
            if scales is not None:
                np.take(self._scale_view, ids, axis=0, out=scales)
            return
        loc = self.loc
        wfd, sfd = self._fds[loc.path], self._fds[loc.scale_path]
        preadv = getattr(os, "preadv", None)
        for i in range(ids.shape[0]):
            rid = int(ids[i])
            self._pread_into(wfd, vals[i], loc.weight_offset + rid * loc.dim, preadv)
            if scales is not None:
                self._pread_into(sfd, scales[i], loc.scale_offset + rid * loc.scale_cols, preadv)

    @staticmethod
    def _pread_into(fd: int, dst: np.ndarray, offset: int, preadv) -> None:
        n = dst.shape[0]
        if preadv is not None:
            got = preadv(fd, [memoryview(dst)], offset)
        else:
            data = os.pread(fd, n, offset)
            got = len(data)
            dst[:got] = np.frombuffer(data, dtype=np.uint8)
        if got != n:
            raise OSError(f"short read: {got}/{n} bytes at offset {offset}")

    def warm(self, row_ids: np.ndarray) -> None:
        """Touch rows so they sit in the page cache (e.g. after a bank load dropped it)."""
        ids = np.unique(np.asarray(row_ids, dtype=np.int64).reshape(-1))
        step = 1 << 16
        for s in range(0, ids.shape[0], step):
            part = ids[s : s + step]
            vals = np.empty((part.shape[0], self.dim), dtype=np.uint8)
            scales = np.empty((part.shape[0], self.scale_cols), dtype=np.uint8)
            self.read_rows(part, vals, scales)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._pool.shutdown(wait=True)
        # numpy views export the map's buffer; every reference must be gone before close()
        self._weight_view = self._scale_view = None
        maps = [mm for mm, _ in self._maps.values()]
        self._maps.clear()
        gc.collect()
        for mm in maps:
            try:
                mm.close()
            except BufferError:
                pass  # a caller still holds a view: the map is released with it
            except (OSError, ValueError):
                pass
        for fd in self._fds.values():
            try:
                os.close(fd)
            except OSError:
                pass
        self._fds.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


__all__ = ["ArrayRowStore", "EngramRowSource", "RowStoreStats", "SafetensorsRowStore"]
