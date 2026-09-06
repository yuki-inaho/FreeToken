"""CPU-only paired replay of captured top-k routes through hybrid admission policies.

This measures exact cache decisions on a fixed reconstructed trace, not throughput.
Run with CUDA_VISIBLE_DEVICES='' and one CPU thread while another server is busy.
"""
from __future__ import annotations

import argparse
import bisect
import collections
import hashlib
import heapq
import json
from pathlib import Path
import random
import time

Q = 1 << 16
FRACTION_Q16 = round(39.67 / (39.67 + 90.55) * Q)
HORIZONS = (1, 8, 32)


class ReplayCache:
    """Exact timestamp/slot-index LRU ties and full-host hybrid admission semantics."""

    def __init__(self, layers, experts, slots, credit):
        self.layers, self.experts, self.slots = layers, experts, slots
        self.credit_enabled = credit
        self.slot = {}
        self.owner = [-1] * slots
        self.usage = [0] * slots
        self.recency = [-1] * (layers * experts)
        self.credit = [0] * layers
        self.heap = [(0, slot) for slot in range(slots)]
        self.clock = 0

    def visit(self, layer, counts):
        self.clock += 1
        clock, base = self.clock, layer * self.experts
        active = {base + expert for expert in counts if expert >= 0}
        missing = active - self.slot.keys()
        for expert in active - missing:
            slot = self.slot[expert]
            self.usage[slot] = clock
            heapq.heappush(self.heap, (clock, slot))
        m = len(missing)
        lo = m * FRACTION_Q16 // Q
        cost = lambda f: max(f * (Q - FRACTION_Q16), (m - f) * FRACTION_Q16)
        fetch = min(m, lo if cost(lo) <= cost(lo + 1) else lo + 1)
        candidates, funded = missing, False
        if self.credit_enabled:
            if fetch:
                self.credit[layer] = 0
            else:
                recurring = {e for e in missing if self.recency[e] >= 0}
                self.credit[layer] += len(recurring) * FRACTION_Q16
                if self.credit[layer] >= Q:
                    self.credit[layer] -= Q
                    fetch, candidates, funded = 1, recurring, True
        fetched = sorted(candidates, key=lambda e: (-self.recency[e], e))[:fetch]
        evicted = []
        for expert in fetched:
            while True:
                stamp, slot = heapq.heappop(self.heap)
                if self.usage[slot] == stamp:
                    break
            old = self.owner[slot]
            assert old not in active, "an active route cannot be evicted"
            if old >= 0:
                del self.slot[old]
                evicted.append(old)
            self.owner[slot] = expert
            self.slot[expert] = slot
            self.usage[slot] = clock
            heapq.heappush(self.heap, (clock, slot))
        for expert in active:
            self.recency[expert] = clock
        if len(self.heap) > self.slots * 8:
            self.heap = list(zip(self.usage, range(self.slots)))
            heapq.heapify(self.heap)
        assert 0 <= self.credit[layer] < Q
        return missing, set(fetched), evicted, funded


def check_cpu_reference():
    """Fail before analysis if the efficient implementation differs from production."""
    import torch
    from freetoken.moe.offload_cache import OffloadMoeCache
    from freetoken.moe import offload_kernels

    old = offload_kernels._HYBRID_ADMISSION_CREDIT
    try:
        for enabled in (False, True):
            offload_kernels._HYBRID_ADMISSION_CREDIT = enabled
            ref = OffloadMoeCache(2, 32, 40, torch.device("cpu"), decode_target="hybrid",
                                 hybrid_max_fetch=32, hybrid_fetch_fraction=FRACTION_Q16 / Q)
            sim = ReplayCache(2, 32, 40, enabled)
            rng = random.Random(11)
            calls = [(0, [30] * 4)] * 7 + [(1, [29, 30] * 3)] * 8
            calls += [(i % 2, [rng.randrange(32) for _ in range(i % 23 + 1)]) for i in range(240)]
            for layer, raw in calls:
                ids = torch.tensor(raw, dtype=torch.int32)
                cpu_ids = ref.ensure_experts_hybrid(layer, ids)
                missing, fetched, _, _ = sim.visit(layer, collections.Counter(raw))
                assert ref.num_missing_full.item() == len(missing)
                assert ref.num_indices.item() == len(fetched)
                assert ref.hybrid_admission_credit.tolist() == sim.credit
                assert ref.id_of_slot.tolist() == sim.owner
                assert ref.usage.tolist() == sim.usage
                assert ref.expert_recency.flatten().tolist() == sim.recency
                assert ref.slot_for_id.flatten().tolist() == [sim.slot.get(e, -1) for e in range(64)]
                assert ids.tolist() == [sim.slot.get(layer * 32 + e, -1) for e in raw]
                assert cpu_ids.tolist() == [e if layer * 32 + e in missing - fetched else -1 for e in raw]
    finally:
        offload_kernels._HYBRID_ADMISSION_CREDIT = old
    return len(calls) * 2


def prepare(chunk, rows, stride):
    ids = chunk["ids"].numpy()
    layers, tokens, _ = ids.shape
    steps = []
    future = collections.defaultdict(list)
    for index, pos in enumerate(range(0, tokens - rows + 1, stride)):
        step = []
        for layer in range(layers):
            counts = collections.Counter(ids[layer, pos:pos + rows].reshape(-1).tolist())
            step.append(counts)
            for expert in counts:
                future[layer * 512 + expert].append(index)
        steps.append(step)
    return steps, future


def replay(chunk, slots, rows, stride, warmup, expert_bytes):
    steps, future = prepare(chunk, rows, stride)
    layers = len(steps[0])
    arms = {name: ReplayCache(layers, 512, slots, enabled)
            for name, enabled in (("baseline", False), ("admission", True))}
    counters = {name: collections.Counter() for name in arms}
    diff = collections.Counter()
    for index, step in enumerate(steps):
        for layer, counts in enumerate(step):
            outcomes = {name: arm.visit(layer, counts) for name, arm in arms.items()}
            if index < warmup:
                continue
            for name, (missing, fetched, evicted, funded) in outcomes.items():
                c = counters[name]
                cpu = missing - fetched
                c.update(calls=1, active=len(counts), missing=len(missing), fetched=len(fetched),
                         cpu=len(cpu), cpu_routes=sum(counts[e - layer * 512] for e in cpu),
                         evictions=len(evicted), credit_admissions=int(funded))
                for victim in evicted:
                    occurrences = future[victim]
                    loc = bisect.bisect_right(occurrences, index)
                    nxt = occurrences[loc] if loc < len(occurrences) else None
                    for horizon in HORIZONS:
                        if index + horizon >= len(steps):
                            continue  # exclude right-censored events from both counts
                        c[f"eligible_evictions_h{horizon}"] += 1
                        reused = nxt is not None and nxt - index <= horizon
                        c[f"evicted_reused_h{horizon}"] += int(reused)
                        if funded:
                            c[f"eligible_credit_evictions_h{horizon}"] += 1
                            c[f"credit_evicted_reused_h{horizon}"] += int(reused)
                if funded:
                    expert = next(iter(fetched))
                    occurrences = future[expert]
                    loc = bisect.bisect_right(occurrences, index)
                    nxt = occurrences[loc] if loc < len(occurrences) else None
                    for horizon in HORIZONS:
                        if index + horizon < len(steps):
                            c[f"eligible_credit_admissions_h{horizon}"] += 1
                            c[f"credit_admitted_reused_h{horizon}"] += int(
                                nxt is not None and nxt - index <= horizon)
            bm, bf, _, _ = outcomes["baseline"]
            am, af, _, _ = outcomes["admission"]
            diff.update(avoided_cache_misses=len(bm - am), additional_cache_misses=len(am - bm),
                        avoided_cpu_assignments=len((bm - bf) - (am - af)),
                        additional_cpu_assignments=len((am - af) - (bm - bf)))
    for c in counters.values():
        c["fetched_bytes"] = c["fetched"] * expert_bytes
    return {"source": chunk["path"], "tokens": chunk["ids"].shape[1],
            "total_steps": len(steps), "scored_steps": max(0, len(steps) - warmup),
            "arms": counters, "counterfactual": diff}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--slots", type=int, default=7968)
    ap.add_argument("--verify-rows", type=int, default=3)
    ap.add_argument("--verify-stride", type=int, default=2)
    ap.add_argument("--warmup-steps", type=int, default=128)
    args = ap.parse_args()
    import torch
    torch.set_num_threads(1)
    from freetoken.moe.offload_cache import _BANK_BYTES_PER_EXPERT

    started = time.monotonic()
    checks = check_cpu_reference()
    trace = torch.load(args.trace, map_location="cpu", weights_only=True)
    expert_bytes = _BANK_BYTES_PER_EXPERT["nvfp4"](2560, 640)
    results = []
    for chunk in trace["chunks"]:
        result = replay(chunk, args.slots, args.verify_rows, args.verify_stride,
                        args.warmup_steps, expert_bytes)
        results.append(result)
        print(Path(chunk["path"]).name, result["scored_steps"],
              {a: {k: c[k] for k in ("missing", "cpu", "fetched", "credit_admissions")}
               for a, c in result["arms"].items()}, flush=True)
    pooled = {name: collections.Counter() for name in ("baseline", "admission")}
    diff = collections.Counter()
    for result in results:
        for name, c in result["arms"].items():
            pooled[name].update(c)
        diff.update(result["counterfactual"])
    output = {"trace": str(Path(args.trace).resolve()),
              "trace_sha256": hashlib.sha256(Path(args.trace).read_bytes()).hexdigest(),
              "parameters": vars(args), "fraction_q16": FRACTION_Q16,
              "bytes_per_expert": expert_bytes, "cpu_reference_checked_calls": checks,
              "elapsed_s": time.monotonic() - started, "chunks": results,
              "pooled": pooled, "counterfactual": diff,
              "limits": ["Historical prefill-derived top-k traces, not the current ablation requests.",
                         "Fixed verify-row/stride grouping approximates speculation; no rejected-branch or MTP-layer routes.",
                         "Both arms start each chunk empty, replay a warmup, then score; no prefill cache materialization.",
                         "Soon-reused eviction horizons are replay steps, with right-censored events excluded.",
                         "Same traces and geometry isolate cache policy, but do not predict latency or model quality."]}
    Path(args.out).write_text(json.dumps(output, indent=2) + "\n")
    print("POOLED", json.dumps(pooled), "COUNTERFACTUAL", json.dumps(diff), flush=True)
    print(f"wrote {args.out} ({output['elapsed_s']:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
