#!/usr/bin/env python3
"""Is speculative tier3 -> tier2 promotion worth it AFTER residency-aware routing?

`tier3-promote-recall.py` answered "no" and its arithmetic still holds -- for the world it
measured. Two things about that world are no longer true, and both move the answer:

1. **It counted every routed expert missing from the host pool as a miss.** The GPU cache
   filters first (the ensure kernel hands the resolver `-1` for its own hits), so the real
   demand is a fraction of the 7.2 misses/token it charged, and the bandwidth ratio it
   rejected promotion on was computed against a demand figure that is too large.
2. **Residency routing now removes most of what is left.** The misses that survive are the
   ones the substitution REFUSED -- experts the router weights too highly to swap away. That
   is a smaller set, and, being the head of the router distribution rather than its tail, a
   very different one to predict.

So this re-asks the question at the real operating point: replay the GPU cache, the host pool
and the substitution together, then ask a cross-layer predictor (layer J's residual through
layer J+N's own gate -- the xnorm proxy) to name the experts that actually reach the disk.

Two budgets decide it, and both are reported:

* **Bandwidth.** Promotions cost a 2.64 MiB read each. Demand traffic at this operating point
  leaves most of the drive idle, which is the thing that changed.
* **Lead time.** A promotion issued N layers early has N * (step_ms / num_layers) to complete,
  against a measured ~0.95 ms per expert. Below that it does not hide the stall, it moves it.

Usage:
  scripts/tier3-promote-after-routing.py --tau 0.05 --max-subs 1 --step-ms 27
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import statistics

import torch
from safetensors import safe_open

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_replay():
    """Reuse the replay's cache and substitution semantics rather than restating them."""
    spec = importlib.util.spec_from_file_location(
        "tier3_replay", os.path.join(_HERE, "tier3-residency-replay.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/media/nuko/New Volume/Qwen3.8-Flash-Next-NVFP4")
    ap.add_argument("--dumps", default="/media/nuko/New Volume/route-dumps-m1")
    ap.add_argument("--gpu-slots", type=int, default=2296)
    ap.add_argument("--host-slots", type=int, default=352)
    ap.add_argument("--num-experts", type=int, default=512)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--tau", type=float, default=0.05, help="0 = routing off")
    ap.add_argument("--max-subs", type=int, default=1)
    ap.add_argument("--verify-rows", type=int, default=3)
    ap.add_argument("--verify-stride", type=int, default=2)
    ap.add_argument("--skip", type=int, default=200)
    ap.add_argument("--cand-scan", type=int, default=32)
    ap.add_argument("--lookahead", nargs="*", type=int, default=[2, 4, 6])
    ap.add_argument("--topk-pred", nargs="*", type=int, default=[4, 8, 16, 32])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--step-ms", type=float, default=27.0, help="decode step, for the budgets")
    ap.add_argument("--read-ms", type=float, default=0.95, help="measured latency of one expert")
    ap.add_argument("--drive-gbs", type=float, default=5.7)
    ap.add_argument("--accepted-per-step", type=float, default=2.33)
    args = ap.parse_args()

    R = _load_replay()
    idx = json.load(open(f"{args.model}/model.safetensors.index.json"))["weight_map"]

    def load(name):
        with safe_open(f"{args.model}/{idx[name]}", framework="pt") as f:
            return f.get_tensor(name)

    files = sorted(glob.glob(f"{args.dumps}/route_*.pt"))
    if not files:
        raise SystemExit(f"no route_*.pt under {args.dumps}")
    d0 = torch.load(files[0], map_location="cpu", weights_only=True)
    layer_ids = sorted(d0["layers"])
    del d0
    gates = {l: load(f"model.language_model.layers.{l}.mlp.gate.weight").float()
             for l in layer_ids}
    hidden = gates[layer_ids[0]].shape[1]
    print(f"{len(layer_ids)} MoE layers, gate hidden {hidden}, gpu {args.gpu_slots} slots, "
          f"host {args.host_slots}/layer, tau {args.tau}, max_subs {args.max_subs}")

    KS, NS = args.topk_pred, args.lookahead
    hit = {(n, k): 0 for n in NS for k in KS}      # reads a promotion would have covered
    cand = {(n, k): 0 for n in NS for k in KS}     # promotions issued (predicted, not resident)
    reads_total = 0
    steps = 0
    read_rank: list[int] = []       # router rank of the experts that still reach the disk
    read_weight: list[float] = []

    gpu = R.FlatLru(args.gpu_slots)
    host = [R.HostLru(args.host_slots) for _ in layer_ids]

    for path in files:
        blob = torch.load(path, map_location="cpu", weights_only=True)
        layers = blob["layers"]
        T = layers[layer_ids[0]]["logits"].shape[0]
        T = min(T, args.limit) if args.limit else T

        pre, pred = {}, {}
        for li, l in enumerate(layer_ids):
            lg = layers[l]["logits"][:T].float()
            w, ids = R.topk_softmax(lg, args.top_k)
            order = torch.argsort(lg, dim=-1, descending=True)[:, : args.cand_scan]
            l0 = lg.gather(1, ids[:, :1])
            cw = torch.exp(lg.gather(1, order) - l0) * w[:, :1]
            # rank of every expert in this token's router order, for the profile of what is left
            rank = torch.empty_like(lg, dtype=torch.int16)
            full = torch.argsort(lg, dim=-1, descending=True)
            rank.scatter_(1, full, torch.arange(lg.shape[1], dtype=torch.int16).expand(T, -1))
            pre[li] = (ids.tolist(), w.tolist(), order.tolist(), cw.tolist(), rank)
        # layer l predicted from layer (l - n)'s residual through layer l's own gate
        for n in NS:
            for i, l in enumerate(layer_ids):
                if i < n:
                    continue
                r = layers[layer_ids[i - n]]["r"][:T].float()
                h = r.view(T, -1, hidden).mean(1)
                h = h / (h.pow(2).mean(-1, keepdim=True) + 1e-6).sqrt()
                pred[(n, i)] = torch.topk(h @ gates[l].T, max(KS), -1).indices.tolist()
        del layers, blob

        n_steps = max(0, (T - args.verify_rows) // args.verify_stride + 1)
        for si in range(n_steps):
            t0 = si * args.verify_stride
            rows = range(t0, t0 + args.verify_rows)
            warm = si >= args.skip
            if warm:
                steps += 1
            for li, l in enumerate(layer_ids):
                ids_l, w_l, order_l, cw_l, rank_l = pre[li]
                routed = []
                for t in rows:
                    row_ids = ids_l[t]
                    if args.tau <= 0:
                        routed.append(row_ids)
                        continue
                    new_ids, _, _, _, _ = R.substitute(
                        order_l[t], dict(zip(order_l[t], cw_l[t])), host[li].last,
                        row_ids, w_l[t], args.tau, args.tau, args.max_subs, None,
                    )
                    routed.append(new_ids)

                # what a promoter would have issued, judged against residency BEFORE the step
                if warm:
                    for n in NS:
                        p = pred.get((n, li))
                        if p is None:
                            continue
                        for k in KS:
                            union = {e for t in rows for e in p[t][:k]}
                            cand[(n, k)] += sum(1 for e in union if not host[li].contains(e))

                distinct = sorted({e for r_ in routed for e in r_})
                base = l * args.num_experts
                missed_gpu = gpu.access(base + e for e in distinct)
                need = sorted(f - base for f in missed_gpu)
                # the reads themselves, before the pool is updated
                reads = [e for e in need if not host[li].contains(e)]
                if warm:
                    reads_total += len(reads)
                    for e in reads:
                        # an expert may be routed by any row of the verify batch; score it by
                        # the row that wanted it MOST (using row t0 alone mixes rows and
                        # reports ranks for experts that row never routed)
                        best_rank, best_w = args.num_experts, 0.0
                        for t in rows:
                            r_ = int(rank_l[t, e])
                            if r_ < best_rank:
                                best_rank = r_
                            if e in ids_l[t]:
                                best_w = max(best_w, w_l[t][ids_l[t].index(e)])
                        read_rank.append(best_rank)
                        read_weight.append(best_w)
                        for n in NS:
                            p = pred.get((n, li))
                            if p is None:
                                continue
                            # the promoter predicts once per step; credit it if ANY verify
                            # row's prediction named the expert
                            for k in KS:
                                if any(e in p[t][:k] for t in rows):
                                    hit[(n, k)] += 1
                host[li].access(need)
        del pre, pred

    steps = max(1, steps)
    reads_step = reads_total / steps
    tok_s = args.accepted_per_step * 1000.0 / args.step_ms
    demand_gbs = reads_step * 2_764_800 * (1000.0 / args.step_ms) / 1e9
    layer_ms = args.step_ms / len(layer_ids)

    print(f"\nsteps {steps}, reads/step {reads_step:.2f} "
          f"({reads_step * 2.64:.1f} MiB/step, {demand_gbs:.2f} GB/s demand at "
          f"{args.step_ms:.0f} ms/step = {tok_s:.0f} tok/s)")
    print(f"drive {args.drive_gbs:.1f} GB/s -> {100 * demand_gbs / args.drive_gbs:.0f}% used, "
          f"{args.drive_gbs - demand_gbs:.2f} GB/s idle "
          f"= room for {(args.drive_gbs - demand_gbs) * args.step_ms / 1000 / 0.0027648:.0f} "
          f"promotions/step at the same latency")

    if read_rank:
        rr = sorted(read_rank)
        q = lambda v, f: v[min(len(v) - 1, int(f * len(v)))]  # noqa: E731
        print(f"\nwhat still reaches the disk: router rank p50 {q(rr, 0.5)}, p90 {q(rr, 0.9)}, "
              f"mean {statistics.mean(rr):.1f}  (top-k is {args.top_k})")
        rw = [x for x in read_weight if x > 0]
        if rw:
            rw.sort()
            print(f"                            routed weight p50 {q(rw, 0.5):.4f}, "
                  f"p90 {q(rw, 0.9):.4f} (mean routed weight is {1 / args.top_k:.4f})")

    print(f"\nlead time: a promotion issued N layers early has N x {layer_ms:.2f} ms; "
          f"one expert takes ~{args.read_ms:.2f} ms")
    print(f"\n{'N':>3} {'lead ms':>8} {'top-K':>6} {'recall':>8} {'promos/step':>12} "
          f"{'GB/s':>7} {'fits?':>6}")
    for n in NS:
        for k in KS:
            c = cand[(n, k)] / steps
            gbs = c * 2_764_800 * (1000.0 / args.step_ms) / 1e9
            lead = n * layer_ms
            ok = "yes" if (gbs + demand_gbs) <= args.drive_gbs and lead >= args.read_ms else "no"
            print(f"{n:>3} {lead:>8.2f} {k:>6} {hit[(n, k)] / max(1, reads_total):>8.1%} "
                  f"{c:>12.1f} {gbs:>7.2f} {ok:>6}")
    print("\n(recall = of the experts that actually came off NVMe, the fraction a top-K\n"
          " prediction made N layers earlier would have named. 'fits' requires BOTH that the\n"
          " promotion traffic plus demand stays under the drive AND that the lead time covers\n"
          " one read -- a promotion that lands late has moved the stall, not hidden it.)")


if __name__ == "__main__":
    main()
