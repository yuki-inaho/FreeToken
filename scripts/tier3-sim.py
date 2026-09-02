#!/usr/bin/env python3
"""Sweep (GPU slots x host slots x policy) over a compacted route trace and report what the
NVMe tier would actually have to carry.

    python scripts/tier3-sim.py --trace routes-m1.pt \
        --gpu-slots 1650 3000 --host-slots 128 256 356 448 512
"""

from __future__ import annotations

import argparse
import csv
import sys

sys.path.insert(0, "python")

from freetoken.tier3.sim import (  # noqa: E402
    frequency_hotset, load_trace, oracle_host_hit, prepare_steps, simulate_steps,
)

GB = 1e9


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--gpu-slots", nargs="*", type=int, default=[1650, 3000])
    ap.add_argument("--host-slots", nargs="*", type=int,
                    default=[128, 200, 256, 320, 356, 400, 448, 480, 512])
    ap.add_argument("--policies", nargs="*", default=["lru"])
    ap.add_argument("--num-experts", type=int, default=512)
    ap.add_argument("--verify-rows", type=int, default=3)
    ap.add_argument("--verify-stride", type=int, default=2)
    ap.add_argument("--skip", type=int, default=32)
    ap.add_argument("--bytes-per-expert", type=int, default=2_764_800,
                    help="NVMe-paged bank bytes per expert (the 4 aligned NVFP4 banks)")
    ap.add_argument("--bw", nargs="*", type=float, default=[1.0, 2.4, 4.9],
                    help="NVMe bandwidths (GB/s) to price the ceiling at")
    ap.add_argument("--accepted-per-step", type=float, default=2.4)
    ap.add_argument("--lookaheads", nargs="*", type=int, default=[1, 2, 4, 8])
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    chunks, top_k = load_trace(args.trace)
    L = int(chunks[0].shape[0])
    tot = sum(int(c.shape[1]) for c in chunks)
    print(f"trace: {len(chunks)} chunks, {tot} tokens, {L} layers, top_k={top_k}")

    steps = prepare_steps(chunks, verify_rows=args.verify_rows,
                          verify_stride=args.verify_stride, skip_steps=args.skip)
    active = sum(len(a) for s in steps for a in s) / (len(steps) * L)
    print(f"steps={len(steps)}  distinct active experts/layer/step = {active:.2f}\n")

    hot = frequency_hotset(chunks, args.num_experts) if "hotset" in args.policies else None
    rows = []
    for policy in args.policies:
        for g in args.gpu_slots:
            print(f"=== policy={policy}  gpu_slots={g} "
                  f"({100 * g / (L * args.num_experts):.1f}% of corpus) ===")
            hdr = (f"{'hostS':>6} {'resid%':>7} {'gpuMiss':>8} {'ramHit%':>8} {'orcl%':>6} "
                   f"{'rd/step':>8} {'MB/step':>8} " +
                   " ".join(f"{b:g}GB/s".rjust(9) for b in args.bw) +
                   " " + " ".join(f"QD{n}".rjust(7) for n in args.lookaheads))
            print(hdr)
            for h in args.host_slots:
                r = simulate_steps(steps, gpu_slots=g, host_slots=h,
                                   num_experts=args.num_experts, policy=policy,
                                   bytes_per_expert=args.bytes_per_expert, hot=hot)
                orc = oracle_host_hit(chunks, h, args.num_experts) if h < args.num_experts else 1.0
                ceils = [r.tok_s_ceiling(b * GB, args.accepted_per_step) for b in args.bw]
                qds = [r.inflight(n)[0] for n in args.lookaheads]
                print(f"{h:>6} {100 * h / args.num_experts:>6.1f}% {r.gpu_miss_rate:>8.3f} "
                      f"{100 * r.host_hit_rate:>7.2f}% {100 * orc:>5.1f}% "
                      f"{r.reads_per_step:>8.2f} {r.nvme_bytes_per_step / 1e6:>8.1f} " +
                      " ".join(f"{c:9.1f}" if c != float('inf') else "      inf" for c in ceils) +
                      " " + " ".join(f"{q:7.1f}" for q in qds))
                rows.append(dict(policy=policy, gpu_slots=g, host_slots=h,
                                 residency=h / args.num_experts,
                                 gpu_miss_rate=r.gpu_miss_rate, host_hit_rate=r.host_hit_rate,
                                 oracle_hit=orc, reads_per_step=r.reads_per_step,
                                 nvme_mb_per_step=r.nvme_bytes_per_step / 1e6,
                                 **{f"tok_s_at_{b:g}": c for b, c in zip(args.bw, ceils)},
                                 **{f"qd_p50_n{n}": q for n, q in zip(args.lookaheads, qds)}))
            print()

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
