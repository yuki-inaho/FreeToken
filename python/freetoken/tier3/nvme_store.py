"""Tier-3: expert rows read straight off an FTW checkpoint with O_DIRECT.

FTW is what makes this cheap. It stores the **post-assembly** expert banks (there is no byte
range in the original safetensors that corresponds to one expert's bank row -- ``expert_banks``
assembles ``gate_up`` from separate checkpoint tensors), one entry per ``(bank, layer)``, and
every entry starts on a 4096-byte boundary. So expert ``e``'s row of a bank lives at
``entry.global_off + e * feat_bytes``, and when ``feat_bytes`` is itself a whole number of
blocks -- true for the packed NVFP4 banks, 1,638,400 and 819,200 B -- source offset, length and
(page-aligned) destination are all block-aligned at once. **Every read is one aligned DMA into a
pinned host slot: no bounce buffer, no memcpy.**

Reads are issued by a small thread pool of blocking ``preadv`` calls. Measured on this box at
the true request shape (2,764,800 B): 2.58 GB/s at QD1, 5.90 GB/s at QD8, then *falling* to
4.50 GB/s at QD32 -- so the pool is capped, and io_uring buys nothing a blocking pool at QD8
does not already get.
"""

from __future__ import annotations

import json
import mmap
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

ALIGN = 4096
INDEX_NAME = "freetoken_weight.json"
# Measured on the real 72 GiB FTW, random (layer, expert) reads: 0.99 GB/s at 1 reader, 4.19 at
# 8, 5.61 at 16, 6.31 at 24, 5.80 at 32 -- a broad knee around 16-24, then it turns over. 16 is
# the default because the tail latency past that buys little (the earlier single-block probe
# turned over at 8). Read-ahead should widen the *window* misses are drawn from, not this cap.
DEFAULT_READERS = 16


class UnalignedRead(RuntimeError):
    """An expert row is not block-aligned, so it cannot be read straight into a pinned slot.

    Raised eagerly at construction rather than silently falling back to a bounce buffer: the
    bounce path costs a full memcpy per expert on the decode critical path, and the planner
    (``budget.host_bank_split``) is supposed to have excluded such banks already. Hitting this
    means the two disagree.
    """


@dataclass(frozen=True)
class Piece:
    """One contiguous read: a byte range of one shard file."""

    path: str
    file_offset: int
    nbytes: int


class NvmeExpertStore:
    """Random-access reader for the paged expert banks of an FTW checkpoint."""

    def __init__(self, ftw_dir: str, bank_names: Sequence[str], *,
                 readers: int = DEFAULT_READERS, verify_aligned: bool = True) -> None:
        with open(os.path.join(ftw_dir, INDEX_NAME), encoding="utf-8") as fh:
            index = json.load(fh)
        assert index.get("align", ALIGN) == ALIGN, index.get("align")
        self.dir = ftw_dir
        self.bank_names = tuple(bank_names)
        self.readers = max(1, readers)
        self.verify_aligned = verify_aligned
        self.bounce_reads = 0  # must stay 0; the round-trip test asserts it

        self._shards = [
            (int(s["global_off"]), int(s["nbytes"]), os.path.join(ftw_dir, s["file"]))
            for s in index["shards"]
        ]
        self._shards.sort()

        # (bank, layer) -> (global_off, feat_bytes, num_experts)
        self._entry: dict[tuple[str, int], tuple[int, int, int]] = {}
        for t in index["tensors"]:
            name = t["name"]
            if "#L" not in name:
                continue
            bank, _, suffix = name.partition("#L")
            if bank not in self.bank_names:
                continue
            layer = int(suffix)
            shape = t["shape"]
            num_experts = int(shape[0])
            feat = int(t["nbytes"]) // num_experts
            self._entry[(bank, layer)] = (int(t["global_off"]), feat, num_experts)

        if not self._entry:
            raise ValueError(
                f"{ftw_dir} has no per-layer expert-bank entries for {list(bank_names)}; tier 3 "
                f"needs a per-layer FTW conversion (ft checkpoint --moe-backend offload)"
            )
        if verify_aligned:
            for (bank, layer), (off, feat, _) in self._entry.items():
                if off % ALIGN or feat % ALIGN:
                    raise UnalignedRead(
                        f"{bank}#L{layer:05d}: entry offset {off} / row {feat} B is not a "
                        f"multiple of {ALIGN} -- this bank must not be NVMe-paged"
                    )

        self.num_layers = 1 + max(l for _, l in self._entry)
        self.num_experts = next(iter(self._entry.values()))[2]
        self.feat_bytes = {b: self._entry[(b, 0)][1] for b in self.bank_names}
        self.bytes_per_expert = sum(self.feat_bytes.values())
        self._fds: dict[str, int] = {}
        self._fd_lock = threading.Lock()
        self._pool: ThreadPoolExecutor | None = None

    # -- addressing -----------------------------------------------------------------

    def pieces(self, layer: int, expert: int, bank: str) -> list[Piece]:
        """Byte ranges holding one expert's row of one bank.

        Normally one piece; a row that straddles a shard boundary splits, and because both the
        entry start and the shard cut are 4096-aligned each piece stays aligned.
        """
        off, feat, _ = self._entry[(bank, layer)]
        return self._slice(off + expert * feat, feat)

    def _slice(self, global_off: int, nbytes: int) -> list[Piece]:
        out: list[Piece] = []
        pos, left = global_off, nbytes
        for base, size, path in self._shards:
            if pos >= base + size or left <= 0:
                continue
            if pos < base:
                continue
            take = min(left, base + size - pos)
            out.append(Piece(path, pos - base, take))
            pos += take
            left -= take
            if left <= 0:
                break
        if left > 0:
            raise ValueError(f"range {global_off}+{nbytes} runs past the last shard")
        return out

    # -- reading --------------------------------------------------------------------

    def _fd(self, path: str) -> int:
        """One O_DIRECT fd per shard, SHARED by every reader thread.

        preadv takes an explicit offset and never touches the file position, so there is no
        shared-position lock to avoid -- the per-thread cache this used to keep just multiplied
        the descriptor count by the pool size and blew a 1024 fd limit.
        """
        fd = self._fds.get(path)
        if fd is None:
            with self._fd_lock:
                fd = self._fds.get(path)
                if fd is None:
                    fd = self._fds[path] = os.open(path, os.O_RDONLY | os.O_DIRECT)
        return fd

    def read_bank(self, layer: int, expert: int, bank: str, dst: memoryview) -> int:
        """Read one expert's row of one bank into ``dst`` (page-aligned, exactly feat_bytes)."""
        feat = self.feat_bytes[bank]
        if len(dst) != feat:
            raise ValueError(f"destination is {len(dst)} B, need {feat}")
        if self.verify_aligned and _addr(dst) % ALIGN:
            self.bounce_reads += 1
            raise UnalignedRead(
                f"destination for {bank}#L{layer:05d} expert {expert} is not page-aligned; "
                f"O_DIRECT would need a bounce buffer"
            )
        done = 0
        for p in self.pieces(layer, expert, bank):
            got = 0
            while got < p.nbytes:
                n = os.preadv(self._fd(p.path), [dst[done + got:done + p.nbytes]],
                              p.file_offset + got)
                if n <= 0:
                    raise OSError(f"short O_DIRECT read at {p.path}+{p.file_offset + got}")
                got += n
            done += p.nbytes
        return done

    def read_block(self, layer: int, bank: str, first: int, count: int,
                   dst: memoryview) -> int:
        """Read experts ``[first, first+count)`` of one bank as one contiguous range.

        Prewarming wants a whole leading block of every layer, and experts are contiguous
        inside an FTW entry -- so this is a few hundred large sequential reads instead of
        ~122k scattered ones. Unaligned banks (NVFP4's tiny ``*_global``) take an
        aligned-window bounce; they are 0.3% of the bytes.
        """
        feat = self.feat_bytes[bank]
        off, _, _ = self._entry[(bank, layer)]
        src, nbytes = off + first * feat, count * feat
        if len(dst) < nbytes:
            raise ValueError(f"destination holds {len(dst)} B, need {nbytes}")
        if src % ALIGN == 0 and nbytes % ALIGN == 0 and _addr(dst) % ALIGN == 0:
            self._read_into(src, nbytes, dst)
            return nbytes
        head = src % ALIGN
        span = ((head + nbytes + ALIGN - 1) // ALIGN) * ALIGN
        bounce = mmap.mmap(-1, span)  # anonymous mmaps are page-aligned
        mv = memoryview(bounce)
        try:
            self._read_into(src - head, span, mv)
            dst[:nbytes] = mv[head:head + nbytes]
        finally:
            mv.release()
            bounce.close()
        return nbytes

    def _read_into(self, global_off: int, nbytes: int, dst: memoryview) -> None:
        """Fill ``dst[:nbytes]`` from the FTW byte region, splitting at shard boundaries."""
        done = 0
        for p in self._slice(global_off, nbytes):
            got = 0
            while got < p.nbytes:
                n = os.preadv(self._fd(p.path), [dst[done + got:done + p.nbytes]],
                              p.file_offset + got)
                if n <= 0:
                    raise OSError(f"short read at {p.path}+{p.file_offset + got}")
                got += n
            done += p.nbytes

    def read_expert(self, layer: int, expert: int, dsts: Mapping[str, memoryview]) -> int:
        """Read every paged bank of one expert. ``dsts`` maps bank name -> destination."""
        return sum(self.read_bank(layer, expert, b, dsts[b]) for b in self.bank_names)

    def submit_rows(self, jobs: Sequence[tuple[int, int, str, memoryview]]):
        """Start a batch of row reads and return a handle; pair with :meth:`wait_rows`.

        Lets a caller that knows what it needs one step early (prefill, which runs a layer
        ahead) overlap the reads with GPU work instead of blocking on them.
        """
        jobs = list(jobs)
        if not jobs:
            return None
        if self._pool is None:
            self._pool = ThreadPoolExecutor(self.readers, thread_name_prefix="ft-nvme")
        return [self._pool.submit(self.read_bank, *j) for j in jobs]

    def wait_rows(self, handle) -> int:
        return 0 if handle is None else sum(f.result() for f in handle)

    def read_rows(self, jobs: Sequence[tuple[int, int, str, memoryview]]) -> int:
        """Read a batch of ``(layer, expert, bank, dst)`` rows in parallel.

        The flat form: one job is one bank's row, which is the unit of parallelism (see
        :meth:`read_experts`). Callers that only want some banks -- prefill staging, which
        handles the unaligned ones itself -- use this directly.
        """
        jobs = list(jobs)
        if not jobs:
            return 0
        if self._pool is None:
            self._pool = ThreadPoolExecutor(self.readers, thread_name_prefix="ft-nvme")
        return sum(self._pool.map(lambda j: self.read_bank(*j), jobs))

    def read_experts(self, requests: Iterable[tuple[int, int, Mapping[str, memoryview]]]) -> int:
        """Read a batch of experts in parallel, capped at ``readers`` IOs in flight.

        The unit of parallelism is one **(expert, bank) row**, not one expert, so the in-flight
        IO count equals ``readers`` regardless of how many banks a format pages. Measured on the
        real checkpoint this is throughput-neutral (per-expert vs per-row at 8 readers: 4.69 vs
        4.19 GB/s, i.e. within run-to-run noise) -- it is kept for the predictable depth, not
        for speed.
        """
        reqs = list(requests)
        return self.read_rows([(layer, expert, bank, dsts[bank])
                               for layer, expert, dsts in reqs for bank in self.bank_names])

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None
        with self._fd_lock:
            for fd in self._fds.values():
                os.close(fd)
            self._fds.clear()


def _addr(mv: memoryview) -> int:
    import ctypes

    return ctypes.addressof(ctypes.c_char.from_buffer(mv))
