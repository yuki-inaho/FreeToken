#!/usr/bin/env python3
"""Measure NVMe random-read throughput AND latency at the exact shape the tier-3 reader uses.

The design issues aligned O_DIRECT reads of one expert record (4 NVFP4 banks = 2,764,800 B =
675 pages) into page-aligned host memory. What matters is not peak sequential bandwidth but
**per-read latency at low queue depth**: the simulator says a warm cache misses well under one
expert per layer, so without read-ahead the achievable QD is ~1.

Reports GB/s plus p50/p99 completion latency per (block size, queue depth).

``--mode expert`` answers the question the serving reader actually poses: how long does ONE
expert take? The C++ resolver issues one pread per bank (the 1,638,400 B gate_up dominates) on
one thread, so a single miss runs at queue depth 1 even though the drive needs QD 4-8 to reach
its bandwidth. This mode replays the real bank shape, optionally splitting each bank row into
``--chunk-bytes`` aligned pieces submitted to a reader pool, and reports the per-expert latency
that the decode step actually waits on -- p50/p99 over experts, not over individual reads.
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
# The aligned NVFP4 bank rows of one expert, in the order the resolver pushes them. The two
# unaligned *_global banks (2,560 / 5,120 B) go through a bounce buffer and are 0.3% of the
# bytes, so they are left out: they are not what a miss waits on.
EXPERT_BANKS = (1_638_400, 819_200, 204_800, 102_400)


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


def run_experts(path: str, size: int, chunk: int, readers: int, concurrent: int,
                experts: int, seed: int) -> tuple[float, float, float, float]:
    """-> (GB/s, p50_ms, p99_ms, mean_jobs_per_expert), timing WHOLE EXPERTS.

    One expert = EXPERT_BANKS read into page-aligned buffers, each bank optionally split into
    ``chunk``-byte aligned pieces. ``concurrent`` experts are in flight at once (a multi-miss
    layer), and every expert's latency is measured from its first submission to its last
    completion -- which is what the coordinator blocks on.
    """
    rng = random.Random(seed)
    total_bytes = sum(EXPERT_BANKS)
    # one destination buffer per bank per in-flight expert, page-aligned as O_DIRECT requires
    bufs = [[_aligned_buf(n) for n in EXPERT_BANKS] for _ in range(concurrent)]
    fds = [os.open(path, os.O_RDONLY | os.O_DIRECT) for _ in range(max(1, readers))]
    lat: list[float] = []
    jobs_per_expert = 0
    # sub-slices of a destination buffer keep the mmap exported; release them before close()
    slices: list[memoryview] = []

    def jobs_for(slot: int) -> list[tuple[int, int, memoryview]]:
        """(offset, nbytes, dst) for one expert at a random aligned position."""
        out = []
        for bi, n in enumerate(EXPERT_BANKS):
            span = max(1, (size - n) // BLK)
            base = rng.randrange(span) * BLK
            mv = bufs[slot][bi][1]
            if chunk <= 0 or n <= chunk:
                out.append((base, n, mv))
                continue
            for off in range(0, n, chunk):
                take = min(chunk, n - off)
                sub = mv[off:off + take]
                slices.append(sub)
                out.append((base + off, take, sub))
        return out

    def do(job, fd_idx: int) -> None:
        off, n, mv = job
        got = os.preadv(fds[fd_idx % len(fds)], [mv], off)
        assert got == n, (got, n)

    try:
        with ThreadPoolExecutor(max(1, readers)) as ex:
            t0 = time.perf_counter()
            for group in range(0, experts, concurrent):
                batch = [jobs_for(s) for s in range(min(concurrent, experts - group))]
                jobs_per_expert = sum(len(b) for b in batch) / len(batch)
                # submit every expert's jobs together (the resolver builds ONE batch per layer),
                # then time each expert by when its own last chunk lands
                t_start = time.perf_counter()
                futs = [[ex.submit(do, j, i) for i, j in enumerate(b)] for b in batch]
                for fl in futs:
                    for f in fl:
                        f.result()
                    lat.append((time.perf_counter() - t_start) * 1e3)
            dt = time.perf_counter() - t0
    finally:
        for fd in fds:
            os.close(fd)
        for sub in slices:
            sub.release()
        for per_expert in bufs:
            for m, mv in per_expert:
                mv.release()
                m.close()
    lat.sort()
    gbs = experts * total_bytes / dt / 1e9
    return (gbs, statistics.median(lat), lat[min(len(lat) - 1, int(0.99 * len(lat)))],
            jobs_per_expert)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="a large file on the target filesystem")
    ap.add_argument("--block-sizes", nargs="*", type=int,
                    default=[EXPERT_BYTES, 1_638_400, 819_200, 262_144])
    ap.add_argument("--queue-depths", nargs="*", type=int, default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--reads", type=int, default=192)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--mode", choices=["block", "expert"], default="block",
                    help="block: the raw (block size, QD) table. expert: whole-expert latency "
                         "at the real bank shape, sweeping --chunk-bytes.")
    ap.add_argument("--chunk-bytes", nargs="*", type=int,
                    default=[0, 819_200, 409_600, 204_800, 102_400],
                    help="expert mode: split each bank row into pieces this large (0 = today's "
                         "one pread per bank). Must be a multiple of 4096.")
    ap.add_argument("--readers", type=int, default=16, help="expert mode: reader pool size")
    ap.add_argument("--concurrent-experts", nargs="*", type=int, default=[1, 2, 3],
                    help="expert mode: experts in flight at once (a multi-miss layer)")
    ap.add_argument("--experts", type=int, default=96, help="expert mode: experts per point")
    args = ap.parse_args()

    size = os.path.getsize(args.path)
    print(f"{args.path}  ({size / 2**30:.1f} GiB)\n")
    if args.mode == "expert":
        print(f"whole-expert latency: banks {EXPERT_BANKS} = {sum(EXPERT_BANKS)} B, "
              f"{args.readers} readers, {args.experts} experts per point\n")
        print(f"{'chunk':>10} {'inflight':>9} {'jobs/exp':>9} {'GB/s':>8} {'p50 ms':>8} {'p99 ms':>8}")
        for chunk in args.chunk_bytes:
            if chunk and chunk % BLK:
                print(f"  skip {chunk}: not block-aligned")
                continue
            for conc in args.concurrent_experts:
                gbs, p50, p99, jobs = run_experts(args.path, size, chunk, args.readers, conc,
                                                  args.experts, args.seed)
                print(f"{chunk:>10} {conc:>9} {jobs:>9.1f} {gbs:>8.2f} {p50:>8.3f} {p99:>8.3f}")
            print()
        return
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
