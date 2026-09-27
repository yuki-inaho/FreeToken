"""Host staging buffers for fetched Engram rows, and the fetch handle around a row read.

Rows gathered from the store land in a fixed ring of uint8 buffers (``[cap_rows, dim]`` codes
+ ``[cap_rows, scale_cols]`` scales) so the H2D copy can be a single ``copy_(non_blocking)``
from pinned memory — and, for CUDA-graph decode, from a stable address. The buffers are torch
pinned tensors when torch + CUDA are available and plain numpy otherwise, exposed to the store
as numpy views either way; torch is imported lazily so the host-side tests run without it.

A buffer is handed out by ``acquire`` and must be ``release``d once its consumer (the device
copy) is done with it; acquiring a buffer whose previous fetch was never released raises rather
than silently overwriting rows a copy may still be reading.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np


_pin_warned = False


def _pinned_uint8(shape: tuple[int, int]):
    """A pinned torch uint8 tensor, or None when torch/CUDA are unavailable.

    A CUDA device that is present but refuses the allocation degrades to pageable numpy memory
    (correct, just slower H2D); that is worth one warning, since nothing else would show it.
    """
    global _pin_warned
    try:
        import torch
    except Exception:
        return None
    try:
        if not torch.cuda.is_available():
            return None
        return torch.empty(shape, dtype=torch.uint8, pin_memory=True)
    except Exception as e:  # noqa: BLE001 - any CUDA/allocator failure means "fall back"
        if not _pin_warned:
            _pin_warned = True
            logging.getLogger(__name__).warning("engram staging: pinned allocation failed (%s); using pageable memory", e)
        return None


class EngramStagingRing:
    """``n_buffers`` staging slots of ``cap_rows`` rows each."""

    def __init__(self, n_buffers: int, cap_rows: int, dim: int, scale_cols: int, pin: bool = True):
        if n_buffers < 1 or cap_rows < 1:
            raise ValueError("n_buffers and cap_rows must be >= 1")
        self.n_buffers, self.cap_rows, self.dim, self.scale_cols = int(n_buffers), int(cap_rows), int(dim), int(scale_cols)
        self.pinned = False
        self._torch: list[tuple[object, object]] = []
        self.vals: list[np.ndarray] = []
        self.scales: list[np.ndarray] = []
        for _ in range(self.n_buffers):
            tv = _pinned_uint8((self.cap_rows, self.dim)) if pin else None
            ts = _pinned_uint8((self.cap_rows, self.scale_cols)) if tv is not None else None
            if tv is not None and ts is not None:
                self.pinned = True
                self._torch.append((tv, ts))
                self.vals.append(tv.numpy())
                self.scales.append(ts.numpy())
            else:
                self._torch.append((None, None))
                self.vals.append(np.empty((self.cap_rows, self.dim), dtype=np.uint8))
                self.scales.append(np.empty((self.cap_rows, self.scale_cols), dtype=np.uint8))
        self._owner: list[EngramFetch | None] = [None] * self.n_buffers
        self._next = 0
        self._lock = threading.Lock()
        self.acquires = 0

    @property
    def bytes(self) -> int:
        return self.n_buffers * self.cap_rows * (self.dim + self.scale_cols)

    def torch_buffers(self, index: int):
        """``(vals, scales)`` torch tensors backing slot ``index`` (None, None when not pinned)."""
        return self._torch[index]

    def acquire(self, n_rows: int, fetch: "EngramFetch") -> int:
        if n_rows > self.cap_rows:
            raise ValueError(f"{n_rows} rows exceed the staging capacity {self.cap_rows}")
        with self._lock:
            index = self._next
            prev = self._owner[index]
            if prev is not None and not prev.released:
                raise RuntimeError(
                    f"staging slot {index} still owned by an unreleased fetch ({prev.n_rows} rows); "
                    "release() fetches once their device copy is done, or use more buffers"
                )
            self._owner[index] = fetch
            self._next = (index + 1) % self.n_buffers
            self.acquires += 1
            return index

    def release(self, index: int, fetch: "EngramFetch") -> None:
        with self._lock:
            if self._owner[index] is fetch:
                self._owner[index] = None


@dataclass
class FetchStats:
    n_rows: int
    n_unique: int
    submitted_at: float
    done_at: float | None = None
    waited_s: float = 0.0  # time the consumer actually blocked in wait()

    @property
    def wall_s(self) -> float | None:
        return None if self.done_at is None else self.done_at - self.submitted_at


class EngramFetch:
    """One outstanding gather: ``row_ids`` (flat) into staging slot ``buffer_index``.

    ``wait()`` joins the read and returns the ``[n_rows, dim]`` / ``[n_rows, scale_cols]`` views
    (uint8 numpy, backed by pinned memory when the ring is pinned). Call ``release()`` when the
    device copy has consumed them.
    """

    def __init__(self, ring: EngramStagingRing, row_ids: np.ndarray):
        self.ring = ring
        self.row_ids = np.ascontiguousarray(row_ids, dtype=np.int64).reshape(-1)
        self.n_rows = int(self.row_ids.shape[0])
        self.buffer_index = ring.acquire(self.n_rows, self)
        self.futures: list[Future] = []
        self.released = False
        self._done = False
        self.stats = FetchStats(self.n_rows, int(np.unique(self.row_ids).shape[0]) if self.n_rows else 0, time.perf_counter())

    @property
    def vals(self) -> np.ndarray:
        return self.ring.vals[self.buffer_index][: self.n_rows]

    @property
    def scales(self) -> np.ndarray:
        return self.ring.scales[self.buffer_index][: self.n_rows]

    def torch_views(self):
        """``(vals, scales)`` torch tensors over the same rows (None, None when the ring is not pinned)."""
        tv, ts = self.ring.torch_buffers(self.buffer_index)
        if tv is None:
            return None, None
        return tv[: self.n_rows], ts[: self.n_rows]

    def done(self) -> bool:
        return self._done or all(f.done() for f in self.futures)

    def wait(self) -> tuple[np.ndarray, np.ndarray]:
        if not self._done:
            t0 = time.perf_counter()
            for f in self.futures:
                f.result()
            self._done = True
            now = time.perf_counter()
            self.stats.waited_s += now - t0
            self.stats.done_at = now
        return self.vals, self.scales

    def release(self) -> None:
        if not self.released:
            self.released = True
            self.ring.release(self.buffer_index, self)


class FetchDriver:
    """Submits store reads off the caller's thread so a fetch overlaps whatever issued it."""

    def __init__(self, workers: int = 2):
        self._pool = ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix="engram-fetch")

    def submit(self, ring: EngramStagingRing, store, row_ids: np.ndarray) -> EngramFetch:
        fetch = EngramFetch(ring, row_ids)
        if fetch.n_rows:
            fetch.futures.append(self._pool.submit(store.read_rows, fetch.row_ids, fetch.vals, fetch.scales))
        return fetch

    def submit_multi(self, ring: EngramStagingRing, parts: list[tuple[object, np.ndarray]]) -> EngramFetch:
        """One fetch whose rows come from several stores: ``parts = [(store, row_ids), ...]`` are
        laid out consecutively in the slot (used for the per-layer tables of one chunk)."""
        ids = [np.asarray(r, dtype=np.int64).reshape(-1) for _, r in parts]
        fetch = EngramFetch(ring, np.concatenate(ids) if ids else np.zeros(0, dtype=np.int64))
        s = 0
        for (store, _), r in zip(parts, ids):
            n = r.shape[0]
            if n:
                fetch.futures.append(self._pool.submit(store.read_rows, r, fetch.vals[s : s + n], fetch.scales[s : s + n]))
            s += n
        return fetch

    def close(self) -> None:
        self._pool.shutdown(wait=True)


__all__ = ["EngramFetch", "EngramStagingRing", "FetchDriver", "FetchStats"]
