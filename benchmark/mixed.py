"""Mixed single-request decode suite and balanced, sampling-separated reporting.

Report with ``python -m benchmark.mixed <results-dir>``. The canonical measurement
matrix has 3 content families, 4 contexts, 4096 output tokens and 5 repeats in each
of two independent server boots. Partial matrices are explicitly exploratory.
Matrix completeness and throughput scoring are separate from quality and
regression gates, which must also pass before overall validation is complete.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import re
import statistics
from collections import defaultdict

from .validate import check

CONTEXTS = (4096, 32768, 131072, 253952)
KINDS = ("code", "english", "japanese")
SAMPLING = {
    "greedy": {"temperature": 0.0, "top_k": 1, "top_p": 1.0},
    "sampled": {"temperature": 1.0, "top_k": 20, "top_p": 0.95},
}
TASKS = {
    "code": "Using the code above as context, write a complete Python module implementing "
            "an LRU cache with TTL, thread safety, statistics, a CLI and extensive pytest "
            "tests. Include type hints, docstrings, detailed examples and edge cases. "
            "Continue for at least 6000 words; do not stop after an outline.",
    "english": "Write a long English essay about coastal communities adapting to changing "
               "seasons, weather and local traditions, drawing on the document above. "
               "Develop concrete examples, anecdotes and a clear narrative across many "
               "sections. Write at least 6000 words and avoid repeating paragraphs.",
    "japanese": "上の資料を参考に、日本の四季と食文化の関係について日本語で詳しく説明してください。"
                "地域ごとの料理、祭り、保存方法、人々の暮らしの変化を具体例とともに論じてください。"
                "多くの章に分けて少なくとも12000字を書き、同じ段落を繰り返さないでください。",
}


def read_moe_windows(path: str, offset: int) -> dict:
    """Read rounded diagnostic windows, not exact per-request hardware counters.

    The first completed window may include the preceding request's tail. Preserve
    raw window values and label the boundary limitation rather than fabricate totals.
    """
    with open(path, "rb") as f:
        f.seek(offset)
        text = f.read().decode(errors="replace")
    rows = []
    for line in text.splitlines():
        if "[moe-stats]" not in line:
            continue
        pairs = re.findall(r"([\w/]+)=([0-9.]+)", line.split("[moe-stats]", 1)[1])
        row = {k: float(v) for k, v in pairs}
        if row.get("layer_calls"):
            rows.append(row)
    return {"rows": rows, "source": os.path.abspath(path),
            "note": "Rounded periodic decode windows; first window can span request boundary."}


def run_mixed(runner, args) -> None:
    contexts = tuple(int(v) for v in args.contexts.split(","))
    kinds = tuple(args.kinds.split(","))
    modes = tuple(args.sampling_modes.split(","))
    if (not contexts or any(c <= 0 or c + args.output_tokens > 262144 for c in contexts)
            or any(k not in TASKS for k in kinds) or any(m not in SAMPLING for m in modes)
            or args.output_tokens < 2 or (args.runs is not None and args.runs < 1)
            or len(set(contexts)) != len(contexts) or len(set(kinds)) != len(kinds)
            or len(set(modes)) != len(modes)):
        raise ValueError("invalid mixed workload matrix")
    # Grow the elastic pool monotonically within a boot. Rotate family and sampling
    # order between repetitions so the same language does not always get first use.
    for ctx in sorted(contexts):
        for repeat in range(args.runs or 5):
            ordered_kinds = kinds[repeat % len(kinds):] + kinds[:repeat % len(kinds)]
            ordered_modes = modes if repeat % 2 == 0 else modes[::-1]
            for kind, mode in itertools.product(ordered_kinds, ordered_modes):
                key = f"mixed/{runner.variant}/{args.boot_id}/{mode}/{kind}/{ctx}/{repeat}"
                # Configuration-independent bytes permit paired A/B comparison across
                # independent boots. Each request in one boot has a different prefix.
                nonce = f"mixed-{args.boot_id}-{mode}-{kind}-{ctx}-{repeat}"
                settings = SAMPLING[mode]
                runner.measure(
                    key, kind, ctx, args.output_tokens, seed=12000 + repeat,
                    temperature=settings["temperature"],
                    sampling={k: v for k, v in settings.items() if k != "temperature"},
                    task=TASKS[kind], nonce=nonce, server_log=args.server_log,
                    extra={"sampling_mode": mode, "boot_id": args.boot_id,
                           "repeat": repeat, "mode": "cold"},
                )


def eligible(record: dict) -> bool:
    res = record.get("result") or {}
    target, output = record.get("target_input_tokens"), record.get("requested_output_tokens")
    seconds = res.get("decode_wall_s")
    mode = (record.get("extra") or {}).get("sampling_mode")
    if (not isinstance(target, int) or target <= 0
            or not isinstance(output, int) or output < 2
            or res.get("prompt_tokens") != target or res.get("completion_tokens") != output
            or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0
            or mode not in SAMPLING or record.get("sampling") != SAMPLING[mode]):
        return False
    ok, flags = check(record)
    # Exact counts and accounting are requirements for this suite, even where the
    # legacy suite reports those deviations as warnings.
    bad = ("prompt-off-target", "output-short", "engine-client-decode-gap", "spec-accounting")
    return (ok and record.get("valid") is not False
            and not any(f.startswith(bad) for f in flags))


def summarize(rows: list[dict]) -> dict:
    results = [r["result"] for r in rows]
    tokens = sum(r["completion_tokens"] - 1 for r in results)
    seconds = sum(r["decode_wall_s"] for r in results)
    steps = sum((r.get("engine") or {}).get("decode_steps") or 0 for r in results)
    accepted = sum((r.get("engine") or {}).get("accepted_tokens") or 0 for r in results)
    drafts = sum((r.get("engine") or {}).get("draft_tokens") or 0 for r in results)
    rates = [(r["completion_tokens"] - 1) / r["decode_wall_s"] for r in results]
    windows = [w for r in rows for w in (r.get("moe_windows") or {}).get("rows", [])]
    calls = sum(w["layer_calls"] for w in windows)
    def weighted(field):
        present = [w for w in windows if field in w]
        n = sum(w["layer_calls"] for w in present)
        return sum(w[field] * w["layer_calls"] for w in present) / n if n else None
    return {
        "n": len(rows), "tokens": tokens, "decode_s": seconds,
        "decode_tps": tokens / seconds if seconds else None,
        "median_tps": statistics.median(rates) if rates else None,
        "min_tps": min(rates) if rates else None, "max_tps": max(rates) if rates else None,
        "step_ms": sum((r.get("engine") or {}).get("decode_s") or 0 for r in results)
                   * 1000 / steps if steps else None,
        "tokens_per_step": tokens / steps if steps else None,
        "acceptance": accepted / drafts if drafts else None,
        "cpu_experts_per_layer_est": weighted("cpu/layer"),
        "fetch_mib_per_step_est": weighted("fetch_MB/step"),
        "moe_window_layer_calls": calls,
    }


def qualification(rows: list[dict]) -> bool:
    """Require the full canonical matrix; never turn cherry-picked cells into success."""
    keys = [r.get("key") for r in rows]
    if (any(not isinstance(key, str) or not key for key in keys)
            or len(set(keys)) != len(keys) or not all(eligible(r) for r in rows)):
        return False
    # A caller must not combine sampling modes or variants into a complete matrix.
    if (len({r.get("variant") for r in rows}) != 1
            or len({r.get("extra", {}).get("sampling_mode") for r in rows}) != 1):
        return False
    cells = defaultdict(list)
    for r in rows:
        cells[(r["family"], r["target_input_tokens"])].append(r)
    if set(cells) != set(itertools.product(KINDS, CONTEXTS)):
        return False
    all_boots = {r.get("extra", {}).get("boot_id") for r in rows}
    if len(all_boots) != 2 or None in all_boots:
        return False
    instances = [{r.get("server_instance") for r in rows if r["extra"]["boot_id"] == boot}
                 for boot in all_boots]
    if (any(len(ids) != 1 or None in ids for ids in instances)
            or len(set.union(*instances)) != 2):
        return False
    for records in cells.values():
        if len(records) != 10 or any(r["requested_output_tokens"] != 4096 for r in records):
            return False
        for boot in all_boots:
            repeats = [r["extra"].get("repeat") for r in records if r["extra"]["boot_id"] == boot]
            if len(repeats) != 5 or any(type(r) is not int for r in repeats) or set(repeats) != set(range(5)):
                return False
    return True


def write_report(out: str) -> str:
    with open(os.path.join(out, "runs.jsonl")) as f:
        records = [json.loads(line) for line in f if line.strip()]
    records = [r for r in records if r.get("suite") == "mixed"]
    groups, cells = defaultdict(list), defaultdict(list)
    for r in records:
        if not eligible(r):
            continue
        mode = r.get("extra", {}).get("sampling_mode")
        if mode not in SAMPLING or r.get("sampling") != SAMPLING[mode]:
            continue
        groups[(r["variant"], mode)].append(r)
        cells[(r["variant"], mode, r["family"], r["target_input_tokens"])].append(r)
    lines = ["# Mixed decode results", "",
             "Rates use emitted tokens / total first-to-last-token decode seconds. "
             "Greedy and sampled modes are separate. Partial matrices are exploratory. "
             "Full matrix complete reports measurement coverage and validity. Quality and "
             "regression gates must also pass before overall validation is complete.", "",
             "| Variant | Sampling | Runs | tok/s | Full matrix complete | 220 tok/s rate |",
             "| --- | --- | ---: | ---: | --- | --- |"]
    for (variant, mode), rows in sorted(groups.items()):
        s = summarize(rows)
        full = qualification(rows)
        result = ("met" if s["decode_tps"] >= 220 else "not met") if full else "unqualified"
        lines.append(f"| {variant} | {mode} | {s['n']} | {s['decode_tps']:.2f} | "
                     f"{'yes' if full else 'no'} | {result} |")
    lines += ["", "| Variant | Sampling | Content | Input | n | tok/s | min–max | ms/step | tok/step |",
              "| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: |"]
    csv_rows = []
    def fmt(v):
        return "—" if v is None else f"{v:.2f}"
    for (variant, mode, kind, ctx), rows in sorted(cells.items()):
        s = summarize(rows)
        csv_rows.append({"variant": variant, "sampling": mode, "kind": kind, "context": ctx, **s})
        lines.append(f"| {variant} | {mode} | {kind} | {ctx} | {s['n']} | {fmt(s['decode_tps'])} | "
                     f"{fmt(s['min_tps'])}–{fmt(s['max_tps'])} | {fmt(s['step_ms'])} | "
                     f"{fmt(s['tokens_per_step'])} |")
    excluded = len(records) - sum(len(rows) for rows in groups.values())
    lines += ["", f"Excluded records: {excluded}. Raw data and GPU telemetry accompany each run.",
              "MoE counters in mixed-summary.csv are rounded periodic-window estimates, "
              "not exact request totals. Missing diagnostics remain unavailable.", ""]
    if csv_rows:
        with open(os.path.join(out, "mixed-summary.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
            writer.writeheader()
            writer.writerows(csv_rows)
    path = os.path.join(out, "mixed-report.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out")
    print(write_report(ap.parse_args().out))
