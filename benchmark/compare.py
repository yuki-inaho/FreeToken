"""Variant-vs-variant comparison tables from one or more results directories.

Joins records by (suite, target_input_tokens, run index) across two variant labels and
prints/writes absolute + percentage deltas for decode tok/s, TTFT, prefill rate, TPOT and
MTP acceptance. Used for the MTP on/off ablation and the upstream-vs-custom comparison.

Usage:
    python -m benchmark.compare <dir> --a mtp_on --b mtp_off [--csv out.csv]
    python -m benchmark.compare <dir> --a custom --b upstream
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from collections import defaultdict


def load(out_dir: str) -> list[dict]:
    path = os.path.join(out_dir, "runs.jsonl")
    return [json.loads(l) for l in open(path) if l.strip()]


def group(recs: list[dict], variant: str):
    g = defaultdict(list)
    for r in recs:
        if r.get("variant") != variant or r.get("suite") == "concurrency":
            continue
        if r.get("valid") is False or "error" in (r.get("result") or {}):
            continue
        g[(r["suite"], r.get("target_input_tokens"))].append(r)
    return g


def _mean(rows, fn):
    vals = [v for v in (fn(r) for r in rows) if v is not None]
    return statistics.mean(vals) if vals else None


def summarize(rows: list[dict]) -> dict:
    res = lambda r: r.get("result") or {}
    eng = lambda r: res(r).get("engine") or {}
    out = {
        "n": len(rows),
        "decode_tps": _mean(rows, lambda r: res(r).get("decode_tps")),
        "ttft_s": _mean(rows, lambda r: res(r).get("ttft_s")),
        "tpot_ms": _mean(rows, lambda r: res(r).get("tpot_ms")),
        "prefill_tps": _mean(rows, lambda r: (r["target_input_tokens"] / eng(r)["prefill_s"])
                             if eng(r).get("prefill_s") else None),
        "tokens_per_step": _mean(rows, lambda r: (res(r).get("completion_tokens", 0) /
                                                  eng(r)["decode_steps"])
                                 if eng(r).get("decode_steps") else None),
        "accept_rate": _mean(rows, lambda r: (eng(r)["accepted_tokens"] / eng(r)["draft_tokens"])
                             if eng(r).get("draft_tokens") else None),
        "vram_mib": _mean(rows, lambda r: (r.get("telemetry") or {}).get("gpu_mem_used_mib_max")),
    }
    return out


def fmt(v, pct=False):
    if v is None:
        return "-"
    return f"{v * 100:.1f}%" if pct else (f"{v:.2f}" if isinstance(v, float) else str(v))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dirs", nargs="+")
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()
    recs = [r for d in args.out_dirs for r in load(d)]
    ga, gb = group(recs, args.a), group(recs, args.b)
    keys = sorted(set(ga) & set(gb))
    if not keys:
        print(f"no overlapping (suite, context) cells between {args.a!r} and {args.b!r}")
        print(f"  {args.a}: {sorted(set(ga))}\n  {args.b}: {sorted(set(gb))}")
        return
    lines = [f"| suite | ctx | metric | {args.a} | {args.b} | delta | delta% |",
             "| --- | ---: | --- | ---: | ---: | ---: | ---: |"]
    rows_csv = [["suite", "ctx", "metric", args.a, args.b, "delta", "delta_pct"]]
    for k in keys:
        sa, sb = summarize(ga[k]), summarize(gb[k])
        for metric in ("decode_tps", "ttft_s", "tpot_ms", "prefill_tps",
                       "tokens_per_step", "accept_rate", "vram_mib"):
            va, vb = sa.get(metric), sb.get(metric)
            if va is None or vb is None:
                continue
            delta = va - vb
            pct = (va / vb - 1) * 100 if vb else None
            lines.append(f"| {k[0]} | {k[1]} | {metric} | {fmt(va)} | {fmt(vb)} | "
                         f"{fmt(delta)} | {'-' if pct is None else f'{pct:+.1f}%'} |")
            rows_csv.append([k[0], k[1], metric, va, vb, delta, pct])
    print("\n".join(lines))
    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            csv.writer(f).writerows(rows_csv)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
