#!/usr/bin/env python3
"""Per-layer attribution of a speculative verify step from a torch.profiler chrome trace.

Capture with the live server::

    FREETOKEN_SPEC_PROFILE=300:10 FREETOKEN_SPEC_PROFILE_DIR=/tmp/prof ./scripts/serve-qwen38.sh
    # ... send a long generation; the engine writes /tmp/prof/spec-trace-<pid>.json
    scripts/spec-trace-report.py /tmp/prof/spec-trace-*.json

The report reads only GPU-side events (kernels + memcpys, as CUPTI records the nodes of the
replayed CUDA graphs) and answers, per MoE layer of graph A: how long each kernel family ran,
how long the compute stream sat idle waiting for the CPU pool (the gap between the last
expert kernel and the HtoD copy of the CPU partial: cuStreamWaitValue64 is a front-end
memop and never appears as an event), whether the demand PCIe gather overlapped any GEMV,
and how much time tiny kernels (<5 us) and inter-node gaps cost. Step level: the gaps
between graph A / B / C and the eager segments between them.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict


def _load(path: str) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    events = data["traceEvents"] if isinstance(data, dict) else data
    return [e for e in events if e.get("ph") == "X" and "ts" in e and "dur" in e]


def _family(name: str) -> str:
    n = name
    if "ensure_experts_hybrid" in n:
        return "moe.ensure"
    if "fast_index_copy_multi_layered" in n or "MultiLayeredIndexCopy" in n:
        return "moe.lookahead_copy"
    if "fast_index_copy" in n or "IndexCopy" in n:
        return "moe.demand_copy"
    if "_decode_nvfp4_marlin_kernel" in n or "_decode_nvfp4_moe_kernel" in n:
        return "moe.gemv"
    if "moe_sum_reduce" in n:
        return "moe.sum_reduce"
    if "silu_and_mul" in n or "act_and_mul" in n:
        return "moe.act"
    if "fused_topk" in n or "topk_softmax" in n or "moe_router" in n:
        return "moe.router"
    if "shared_gate" in n:
        return "moe.shared_gate"
    if "Memcpy DtoH" in n:
        return "memcpy.d2h"
    if "Memcpy HtoD" in n:
        return "memcpy.h2d"
    if "Memcpy DtoD" in n or "Memcpy" in n:
        return "memcpy.other"
    if "qsa" in n.lower() or "_qsa_" in n:
        return "attn.qsa"
    if "fused_sigmoid_gating" in n or "causal_conv1d" in n or "rms_norm_gated" in n or "chunk_" in n:
        return "attn.gdn"
    if "_gemv_rows_splitk" in n or "_splitk_rows_reduce" in n or "_gemv_splitk" in n or "_splitk_reduce" in n or "_dynamic_quant" in n or "scaled_mm" in n or "enable_3x_kernel" in n:
        return "dense.fp8"
    if "gemm" in n.lower() or "cutlass" in n.lower() or "gemv" in n.lower() or "Kernel" in n and "sm" in n:
        return "dense.gemm"
    if "hc_" in n or "grouped_gemma_rmsnorm" in n or "rmsnorm" in n.lower():
        return "hc.norm_mix"
    if "elementwise" in n or "vectorized" in n or "fill" in n or "copy_" in n or "index_" in n:
        return "misc.elementwise"
    return "other"


def _segment_layers(events: list[dict]) -> list[list[dict]]:
    """Split one stream's GPU events into MoE layers: a layer ends at the HtoD copy of the
    CPU partial that follows an expert GEMV (hybrid backend); everything since the previous
    end belongs to it (attention/GDN/HC of the same decoder layer included)."""
    layers: list[list[dict]] = []
    cur: list[dict] = []
    seen_gemv = False
    for e in events:
        fam = _family(e["name"])
        cur.append(e)
        if fam == "moe.gemv":
            seen_gemv = True
        if seen_gemv and fam == "memcpy.h2d":
            layers.append(cur)
            cur = []
            seen_gemv = False
    return layers


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace")
    ap.add_argument("--stream", type=int, default=None, help="compute stream id (default: busiest)")
    ap.add_argument("--tiny-us", type=float, default=5.0)
    ap.add_argument("--layers", type=int, default=48, help="MoE layers per graph A (for step grouping)")
    ap.add_argument("--per-layer", action="store_true", help="print one row per layer index")
    args = ap.parse_args()

    events = _load(args.trace)
    gpu = [e for e in events if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    if not gpu:
        print("no GPU events in trace (cat kernel/gpu_memcpy)", file=sys.stderr)
        return 1
    by_stream: dict[int, list[dict]] = defaultdict(list)
    for e in gpu:
        by_stream[int(e.get("args", {}).get("stream", -1))].append(e)
    stream = args.stream if args.stream is not None else max(by_stream, key=lambda k: len(by_stream[k]))
    main_ev = sorted(by_stream[stream], key=lambda e: e["ts"])
    side = sorted((e for k, v in by_stream.items() if k != stream for e in v), key=lambda e: e["ts"])
    print(f"streams: {dict((k, len(v)) for k, v in by_stream.items())}; compute stream = {stream}")

    layers = _segment_layers(main_ev)
    print(f"segmented {len(layers)} MoE layers ({len(layers) / max(1, args.layers):.1f} graph-A equivalents)")
    if not layers:
        return 1

    fam_time: dict[str, list[float]] = defaultdict(list)
    stall: list[float] = []
    copy_dur: list[float] = []
    copy_overlap: list[float] = []
    tiny_count: list[int] = []
    tiny_time: list[float] = []
    gaps: list[float] = []
    layer_wall: list[float] = []
    lookahead_overlap_layers = 0
    rows = []
    for li, layer in enumerate(layers):
        t0, t1 = layer[0]["ts"], layer[-1]["ts"] + layer[-1]["dur"]
        layer_wall.append(t1 - t0)
        fams: dict[str, float] = defaultdict(float)
        n_tiny, t_tiny, gap = 0, 0.0, 0.0
        prev_end = None
        last_moe_end = None
        h2d_start = None
        gemv_spans = []
        copy_span = None
        for e in layer:
            fam = _family(e["name"])
            fams[fam] += e["dur"]
            if e["dur"] < args.tiny_us:
                n_tiny += 1
                t_tiny += e["dur"]
            if prev_end is not None and e["ts"] > prev_end:
                gap += e["ts"] - prev_end
            prev_end = max(prev_end or 0, e["ts"] + e["dur"])
            if fam in ("moe.gemv", "moe.act", "moe.sum_reduce"):
                last_moe_end = e["ts"] + e["dur"]
            if fam == "moe.gemv":
                gemv_spans.append((e["ts"], e["ts"] + e["dur"]))
            if fam == "moe.demand_copy":
                copy_span = (e["ts"], e["ts"] + e["dur"])
            if fam == "memcpy.h2d":
                h2d_start = e["ts"]
        for k, v in fams.items():
            fam_time[k].append(v)
        if last_moe_end is not None and h2d_start is not None:
            stall.append(max(0.0, h2d_start - last_moe_end))
        if copy_span is not None:
            copy_dur.append(copy_span[1] - copy_span[0])
            ov = 0.0
            for a, b in gemv_spans:
                ov += max(0.0, min(b, copy_span[1]) - max(a, copy_span[0]))
            copy_overlap.append(ov)
            # any lookahead copy on a side stream during the demand copy?
            for e in side:
                if _family(e["name"]) == "moe.lookahead_copy" and e["ts"] < copy_span[1] and e["ts"] + e["dur"] > copy_span[0]:
                    lookahead_overlap_layers += 1
                    break
        tiny_count.append(n_tiny)
        tiny_time.append(t_tiny)
        gaps.append(gap)
        rows.append((li, t1 - t0, fams.get("moe.gemv", 0.0), fams.get("moe.demand_copy", 0.0), stall[-1] if stall and last_moe_end is not None and h2d_start is not None else float("nan"), n_tiny, gap))

    def med(xs):
        return statistics.median(xs) if xs else float("nan")

    def p90(xs):
        if not xs:
            return float("nan")
        xs = sorted(xs)
        return xs[min(len(xs) - 1, int(0.9 * len(xs)))]

    print("\nper-layer medians (us) [p90]:")
    print(f"  layer wall          {med(layer_wall):8.1f} [{p90(layer_wall):.1f}]")
    print(f"  CPU-branch stall S  {med(stall):8.1f} [{p90(stall):.1f}]   (HtoD(y) start - last expert kernel end)")
    print(f"  demand copy dur     {med(copy_dur):8.1f} [{p90(copy_dur):.1f}]   overlap with GEMV {med(copy_overlap):.1f}")
    print(f"  tiny (<{args.tiny_us:.0f}us) kernels  {med(tiny_count):8.1f} count, {med(tiny_time):.1f} us; inter-node gaps {med(gaps):.1f} us")
    if copy_dur:
        print(f"  layers with a lookahead copy overlapping the demand copy: {lookahead_overlap_layers}/{len(copy_dur)}")
    print("\nkernel-family busy time per layer (median us, share of layer wall):")
    total = med(layer_wall)
    for fam, xs in sorted(fam_time.items(), key=lambda kv: -statistics.median(kv[1])):
        m = statistics.median(xs) if len(xs) == len(layers) else statistics.median(xs + [0.0] * (len(layers) - len(xs)))
        print(f"  {fam:22s} {m:8.1f}  {100 * m / total:5.1f}%   (present in {len(xs)}/{len(layers)} layers)")

    # step-level: gaps between consecutive layers longer than the median inter-layer gap x5
    inter = []
    for a, b in zip(layers, layers[1:]):
        inter.append(b[0]["ts"] - (a[-1]["ts"] + a[-1]["dur"]))
    if inter:
        thr = 5 * med(inter)
        big = [g for g in inter if g > thr]
        print(f"\ninter-layer gap median {med(inter):.1f} us; {len(big)} gaps > {thr:.0f} us (graph boundaries / eager segments): median {med(big):.1f} us, max {max(big) if big else 0:.1f} us")
    if args.per_layer:
        print("\n idx   wall   gemv   copy  stall  tiny  gaps")
        for r in rows:
            print(f"{r[0]:4d} {r[1]:7.1f} {r[2]:6.1f} {r[3]:6.1f} {r[4]:6.1f} {r[5]:5d} {r[6]:6.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
