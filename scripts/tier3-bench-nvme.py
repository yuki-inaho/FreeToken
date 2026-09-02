#!/usr/bin/env python3
"""Measure NVMe random-read throughput AND latency at the exact shape the tier-3 reader uses.

The design issues aligned O_DIRECT reads of one expert record (4 NVFP4 banks = 2,764,800 B =
675 pages) into page-aligned host memory. What matters is not peak sequential bandwidth but
**per-read latency at low queue depth**: the simulator says a warm cache misses well under one
expert per layer, so without read-ahead the achievable QD is ~1.

Reports GB/s plus p50/p99 completion latency per (block size, queue depth).
"""

from __future__ import annotations

import argparse
import ctypes
import mmap
import os
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

BLK = 4096
EXPERT_BYTES = 2_764_800  # the 4 aligned NVFP4 banks of one expert


def _aligned_buf(n: int) -> tuple[mmap.mmap, memoryview]:
    """Anonymous mmap: page-aligned, which O_DIRECT requires of the destination."""
    m = mmap.mmap(-1, n)
    assert ctypes.addressof(ctypes.c_char.from_buffer(m)) % BLK == 0
    return m, memoryview(m)


def run(path: str, size: int, bs: int, qd: int, reads: int, seed: int) -> tuple[float, float, float]:
    """-> (GB/s, p50_ms, p99_ms). One fd per worker: O_DIRECT preadv takes no shared state and
    per-thread fds keep the kernel off a shared file-position lock."""
    rng = random.Random(seed)
    span = max(1, (size - bs) // BLK)
    offsets = [rng.randrange(span) * BLK for _ in range(reads)]
    fds = [os.open(path, os.O_RDONLY | os.O_DIRECT) for _ in range(qd)]
    bufs = [_aligned_buf(bs) for _ in range(qd)]
    lat: list[float] = []

    def one(i: int) -> None:
        slot = i % qd
        t0 = time.perf_counter()
        got = os.preadv(fds[slot], [bufs[slot][1]], offsets[i])
        lat.append((time.perf_counter() - t0) * 1e3)
        assert got == bs, (got, bs)

    try:
        t0 = time.perf_counter()
        if qd == 1:
            for i in range(reads):
                one(i)
        else:
            with ThreadPoolExecutor(qd) as ex:
                list(ex.map(one, range(reads)))
        dt = time.perf_counter() - t0
    finally:
        for fd in fds:
            os.close(fd)
        for m, mv in bufs:
            mv.release()  # a live memoryview keeps the mmap exported and close() raises
            m.close()
    lat.sort()
    return (reads * bs / dt / 1e9, statistics.median(lat), lat[min(len(lat) - 1, int(0.99 * len(lat)))])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="a large file on the target filesystem")
    ap.add_argument("--block-sizes", nargs="*", type=int,
                    default=[EXPERT_BYTES, 1_638_400, 819_200, 262_144])
    ap.add_argument("--queue-depths", nargs="*", type=int, default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--reads", type=int, default=192)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    size = os.path.getsize(args.path)
    print(f"{args.path}  ({size / 2**30:.1f} GiB)\n")
    print(f"{'block':>10} {'QD':>4} {'GB/s':>8} {'p50 ms':>8} {'p99 ms':>8} {'us/expert-equiv':>16}")
    for bs in args.block_sizes:
        if bs % BLK:
            print(f"  skip {bs}: not block-aligned")
            continue
        for qd in args.queue_depths:
            gbs, p50, p99 = run(args.path, size, bs, qd, args.reads, args.seed)
            # time to move one whole expert record at this rate, the number the step budget cares about
            us = EXPERT_BYTES / (gbs * 1e9) * 1e6
            print(f"{bs:>10} {qd:>4} {gbs:>8.2f} {p50:>8.3f} {p99:>8.3f} {us:>16.0f}")
        print()


if __name__ == "__main__":
    main()
