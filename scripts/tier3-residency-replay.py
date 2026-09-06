#!/usr/bin/env python3
"""What residency-aware routing actually saves, counted the way the disk counts it.

``tier3-routing-margin.py`` measures the perturbation budget over individual routed slots. It
cannot predict `nvme_reads/step`, and the gap showed up live: at tau=0.05 it predicted 88.7 % of
host-missing routes substituted (which the `rroute_cover` counter confirms) yet reads only
halved. Three effects it does not model, all of which this script does:

1. **A read is per DISTINCT expert per (layer, step)**, and a speculative step verifies several
   rows. An expert routed by two rows costs one read, and is avoided only if *every* row
   substitutes it.
2. **The GPU cache filters first.** The tier-3 resolver only ever sees routes the GPU cache
   missed (the ensure kernel hands it `-1` for hits), so substituting a host-missing expert that
   happens to be GPU-resident perturbs the model and saves nothing.
3. **v1 substitutes at most one expert per (row, layer)**, so a row with two misses keeps one.

Policies compared (all share one replay so the residency streams are identical):

  ``off``        no substitution -- the original model, and the reads everything else is scored against
  ``v1``         shipped: one substitution per row, gated on displacement <= tau
  ``v1+gpu``     v1, but only for experts the GPU cache does not already hold
  ``multi``      up to --max-subs substitutions per row, each <= tau, cumulative <= tau-total
  ``multi+gpu``  both
  ``drop:<eps>`` the MEASURED policy: do not substitute at all, DROP the host-missing expert,
                 gated on w_j*n_j/||y|| -- the fraction of the MoE block output it moves.
                 Needs --table (``ft moe subst``). Substitution is not offered because
                 measuring every expert against every other showed it is never worth it.

Usage:
  scripts/tier3-residency-replay.py --dumps /media/nuko/New\\ Volume/route-dumps-m1 --tau 0.05
"""

from __future__ import annotations

import argparse
import glob
import math
import statistics
import torch


def topk_softmax(lg: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The MoE's own routing: top-k over float32 logits, softmax renormalised across them."""
    v, i = torch.topk(lg, k, dim=-1)
    return torch.softmax(v, dim=-1), i


class FlatLru:
    """The GPU cache: one unified timestamp-LRU over ``layer * num_experts + expert``."""

    def __init__(self, slots: int) -> None:
        self.slots, self.last, self.clock = slots, {}, 0

    def contains(self, flat: int) -> bool:
        return flat in self.last

    def access(self, flat_ids) -> list[int]:
        self.clock += 1
        missed = []
        for f in flat_ids:
            if f not in self.last:
                missed.append(f)
                if len(self.last) >= self.slots:
                    victim = min(self.last, key=self.last.get)
                    del self.last[victim]
            self.last[f] = self.clock
        return missed


class HostLru:
    """One layer's pinned pool, seeing only the GPU-miss stream (as tier3/sim.py does)."""

    def __init__(self, slots: int) -> None:
        self.slots, self.last, self.clock = slots, {}, 0

    def contains(self, e: int) -> bool:
        return e in self.last

    def access(self, experts) -> int:
        """Touch a step's distinct experts; returns how many had to be read from NVMe."""
        self.clock += 1
        reads = 0
        for e in experts:
            if e not in self.last:
                reads += 1
                if len(self.last) >= self.slots:
                    victim = min(self.last, key=self.last.get)
                    del self.last[victim]
            self.last[e] = self.clock
        return reads


def cand_pos_rank(order, cand):
    """Position of ``cand`` in the router's full descending order (0 = the model's top pick)."""
    try:
        return order.index(cand)
    except ValueError:
        return len(order)


def drop_action(ids, p, norms, resident, eps, eps_total, max_acts, gpu_hold=None):
    """The measured policy on ONE row: drop the cheapest host-missing routes under a budget.

    Mirrors ``_residency_route_drop_kernel``: zero-drop (the survivors keep their weights, which
    is L2-optimal), ||y|| estimated as sqrt(sum_k w_k^2 n_k^2), greedy cheapest-first, each
    action <= eps and their squares summing to <= eps_total^2.
    """
    ids = list(ids)
    weights = list(p)
    s0 = sum(weights)
    n = [norms[e] for e in ids]
    missing = [(e not in resident) and (gpu_hold is None or not gpu_hold(e)) for e in ids]
    cum2, acts = 0.0, []
    for _ in range(max(1, min(max_acts, len(ids) - 1))):
        w = [v / max(s0, 1e-20) for v in weights]
        y2 = sum((w[j] * n[j]) ** 2 for j in range(len(ids)))
        yn = max(math.sqrt(y2), 1e-12)
        best, slot = None, None
        for j in range(len(ids)):
            if not missing[j]:
                continue
            c = w[j] * n[j] / yn
            if best is None or c < best:
                best, slot = c, j
        if best is None or best > eps or cum2 + best * best > eps_total ** 2:
            break
        weights[slot] = 0.0
        ids[slot] = -1
        missing[slot] = False
        cum2 += best * best
        acts.append(best)
    return ids, sum(acts), len(acts), acts, []


def substitute(order, cand_w, resident, ids, p, tau, tau_total, max_subs, gpu_hold=None):
    """Greedy residency substitution on ONE row, mirroring the kernel.

    ``order`` is the row's expert ids sorted by logit descending and ``cand_w[e]`` each one's
    weight on the same scale as ``p`` (the routed weights are a softmax over the top-k logits,
    so ``exp(l_e - l_0) * p_0`` places any expert on that scale exactly -- the partition function
    cancels, which is the identity the kernel relies on). Returns (new_ids, displaced, n_subs).

    Each swap moves the cheapest missing slot onto the best resident expert not yet selected and
    costs ``p_j/s - p_c/(s - p_j + p_c)``; it is taken only while that is <= tau and the running
    total stays <= tau_total, so the per-token perturbation bound is the one v1 ships with.

    ``gpu_hold`` marks experts the GPU cache already holds: those cost no disk read, so
    substituting them is perturbation for nothing.
    """
    ids = list(ids)
    weights = list(p)
    sel = set(ids)
    s = sum(weights)
    cum, n, cand_pos = 0.0, 0, 0
    moved = []   # the ROUTED weight each swap takes off its expert (not the same as disp)
    gaps = []    # how near-tied the two experts were, in the router's own logit space
    cranks = []  # where the substitute sits in the router's OWN order: rank 10 means the
                 # model's next choice, rank 200 means an expert it had all but excluded
    for _ in range(max_subs):
        missing = [j for j, e in enumerate(ids)
                   if e not in resident and (gpu_hold is None or not gpu_hold(e))]
        if not missing:
            break
        while cand_pos < len(order) and (order[cand_pos] in sel or order[cand_pos] not in resident):
            cand_pos += 1
        if cand_pos >= len(order):
            break
        cand = order[cand_pos]
        p_c = cand_w[cand]
        best_j, best_d = None, None
        for j in missing:
            d = weights[j] / s - p_c / (s - weights[j] + p_c)
            if best_d is None or d < best_d:
                best_j, best_d = j, d
        if best_d > tau or cum + best_d > tau_total:
            break
        cranks.append(cand_pos_rank(order, cand))
        moved.append(weights[best_j] / s)
        gaps.append(math.log(max(weights[best_j], 1e-30)) - math.log(max(p_c, 1e-30)))
        s = s - weights[best_j] + p_c
        ids[best_j], weights[best_j] = cand, p_c
        sel.add(cand)
        cum += best_d
        n += 1
        cand_pos += 1
    return ids, cum, n, moved, gaps, cranks


def run(args) -> None:
    files = sorted(glob.glob(f"{args.dumps}/route_*.pt"))
    if not files:
        raise SystemExit(f"no route_*.pt under {args.dumps}")
    policies = [x.strip() for x in args.policies.split(",") if x.strip()]
    norms = None
    if any(p.startswith("drop:") for p in policies):
        if not args.table:
            raise SystemExit("drop:<eps> needs --table (build it with 'ft moe subst')")
        from safetensors import safe_open
        with safe_open(args.table, framework="pt") as f:
            norms = f.get_tensor("norm").tolist()   # [L][E], indexed by layer position
    assert "off" in policies, "the tau=0 arm is the reference every other policy is scored against"
    reads = {p: 0 for p in policies}
    subs = {p: 0 for p in policies}
    disp = {p: 0.0 for p in policies}
    # The routed weight a swap MOVES to a different expert, which is not what tau bounds --
    # tau bounds the DIFFERENCE between the two experts' weights. Both matter and they differ
    # by an order of magnitude, so report them side by side.
    mass = {p: [] for p in policies}
    gap = {p: [] for p in policies}
    crank = {p: [] for p in policies}
    rows_multi_miss = 0
    rows_seen = 0
    gpu_wasted = 0
    steps = 0

    # The caches persist ACROSS dumps: the live pool is prewarmed and then churns for thousands
    # of steps, so a per-chunk reset would measure cold-start, which is the trap FINDINGS records.
    gpu = host = None
    for path in files:
        blob = torch.load(path, map_location="cpu", weights_only=True)
        layers = blob["layers"]
        layer_ids = sorted(layers)
        T = layers[layer_ids[0]]["logits"].shape[0]
        T = min(T, args.limit) if args.limit else T
        if gpu is None:
            gpu = {p: FlatLru(args.gpu_slots) for p in policies}
            host = {p: [HostLru(args.host_slots) for _ in layer_ids] for p in policies}

        # Everything that depends only on the logits, computed once per (layer, token) in torch.
        # Doing it per (layer, step, policy) with .item() was the whole runtime.
        pre = {}
        C = args.cand_scan
        for li, l in enumerate(layer_ids):
            lg = layers[l]["logits"][:T].float()
            w, ids = topk_softmax(lg, args.top_k)
            order = torch.argsort(lg, dim=-1, descending=True)[:, :C]
            l0 = lg.gather(1, ids[:, :1])                       # the top-1 logit per token
            cw = torch.exp(lg.gather(1, order) - l0) * w[:, :1]  # candidates on the routed scale
            pre[li] = (ids.tolist(), w.tolist(), order.tolist(), cw.tolist())

        n_steps = max(0, (T - args.verify_rows) // args.verify_stride + 1)
        for si in range(n_steps):
            t0 = si * args.verify_stride
            rows = range(t0, t0 + args.verify_rows)
            if si >= args.skip:
                steps += 1
            for li, l in enumerate(layer_ids):
                ids_l, w_l, order_l, cw_l = pre[li]
                ordr_all = {t: order_l[t] for t in rows}
                candw_all = {t: dict(zip(order_l[t], cw_l[t])) for t in rows}
                for policy in policies:
                    routed = []
                    for t in rows:
                        row_ids = ids_l[t]
                        row_w = w_l[t]
                        if policy.startswith("drop:"):
                            eps_v = float(policy.split(":", 1)[1])
                            new_ids, d, n, moved, gaps = drop_action(
                                row_ids, row_w, norms[li], res, eps_v, eps_v,
                                args.max_subs, hold)
                            if si >= args.skip:
                                subs[policy] += n
                                disp[policy] += d
                            routed.append([e for e in new_ids if e >= 0])
                            continue
                        if policy == "off":
                            routed.append(row_ids)
                            continue
                        res = host[policy][li].last
                        gp = gpu[policy]
                        base = l * args.num_experts
                        hold = (lambda e, gp=gp, base=base: gp.contains(base + e)) \
                            if policy.endswith("+gpu") else None
                        ordr = ordr_all[t]
                        cand_w = candw_all[t]
                        mx = 1 if policy.startswith("v1") else args.max_subs
                        new_ids, d, n, moved, gaps, cranks = substitute(
                            ordr, cand_w, res, row_ids, row_w, args.tau,
                            args.tau_total or args.tau, mx, hold,
                        )
                        if si >= args.skip:
                            subs[policy] += n
                            disp[policy] += d
                            mass[policy].extend(moved)
                            gap[policy].extend(gaps)
                            crank[policy].extend(cranks)
                        routed.append(new_ids)
                    if si >= args.skip and policy == "off":
                        rows_seen += len(routed)
                        for row_ids in routed:
                            miss = sum(1 for e in row_ids if e not in host["off"][li].last)
                            rows_multi_miss += 1 if miss >= 2 else 0
                            gpu_wasted += sum(
                                1 for e in row_ids
                                if e not in host["off"][li].last
                                and gpu["off"].contains(l * args.num_experts + e)
                            )
                    distinct = sorted({e for r in routed for e in r})
                    flat = [l * args.num_experts + e for e in distinct]
                    missed = gpu[policy].access(flat)
                    r = host[policy][li].access(sorted(f - l * args.num_experts for f in missed))
                    if si >= args.skip:
                        reads[policy] += r
        del pre

    print(f"steps {steps}  (verify {args.verify_rows} rows, stride {args.verify_stride}, "
          f"gpu {args.gpu_slots} slots, host {args.host_slots}/layer, tau {args.tau}, "
          f"max_subs {args.max_subs})\n")
    print(f"{'policy':>10} {'reads/step':>11} {'vs off':>8} {'subs/step':>10} "
          f"{'displaced/step':>15}")
    base = reads["off"] / max(1, steps)
    for p in policies:
        rs = reads[p] / max(1, steps)
        print(f"{p:>10} {rs:>11.2f} {100 * (rs / base - 1):>7.1f}% {subs[p] / max(1, steps):>10.2f}"
              f" {disp[p] / max(1, steps):>15.4f}")
    print(f"\n{'policy':>10} {'weight MOVED per swap (p50 / p90 / max)':>42} "
          f"{'logit gap p50':>14}")
    for p in policies:
        m = sorted(mass[p])
        if not m:
            continue
        g = sorted(gap[p])
        q = lambda v, f: v[min(len(v) - 1, int(f * len(v)))]  # noqa: E731
        c = sorted(crank[p])
        print(f"{p:>10} {q(m, 0.5):>16.4f} {q(m, 0.9):>12.4f} {m[-1]:>12.4f} "
              f"{q(g, 0.5):>14.3f}   substitute rank p50 {q(c, 0.5)} p90 {q(c, 0.9)}")
    print("  (tau caps the weight DIFFERENCE between the two experts; the mass that actually\n"
          "   lands on a different expert is the routed weight of the slot, ~10x larger. The\n"
          "   logit gap is how near-tied the router itself considered them.)")

    print(f"\nrows with >=2 host misses: {100 * rows_multi_miss / max(1, rows_seen):.2f}% "
          f"(what a second substitution per row can reach)")
    print(f"host-missing routes that the GPU cache already holds: "
          f"{gpu_wasted / max(1, steps):.2f}/step (substituting these perturbs for nothing)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", default="/media/nuko/New Volume/route-dumps-m1")
    ap.add_argument("--gpu-slots", type=int, default=2047)
    ap.add_argument("--host-slots", type=int, default=352)
    ap.add_argument("--num-experts", type=int, default=512)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--tau-total", type=float, default=0.0, help="0 = same as --tau")
    ap.add_argument("--max-subs", type=int, default=4)
    ap.add_argument("--verify-rows", type=int, default=3)
    ap.add_argument("--verify-stride", type=int, default=2)
    ap.add_argument("--skip", type=int, default=32, help="cold steps to drop per chunk")
    ap.add_argument("--table", default="",
                    help="expert substitution table (ft moe subst); required by drop:<eps>")
    ap.add_argument("--policies", default="off,v1,multi",
                    help="subset of off,v1,v1+gpu,multi,multi+gpu")
    ap.add_argument("--cand-scan", type=int, default=32,
                    help="how deep down the logit order to look for a resident substitute")
    ap.add_argument("--limit", type=int, default=0, help="tokens per dump (0 = all)")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
