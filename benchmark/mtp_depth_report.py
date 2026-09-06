"""Validate and report matched 128K MTP depth experiments without server access.

Usage: python -m benchmark.mtp_depth_report RESULTS_DIR [--control adaptive2-3]
Only mixed-suite rows are measured. Every arm must contain the same complete
code/English/Japanese x greedy/sampled matrix for each boot and repeat. Invalid
inputs produce a diagnostic report and nonzero exit, never a partial ranking.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from itertools import product
import json
import math
from pathlib import Path
import re

from .mixed import KINDS, SAMPLING

CONTEXT = 131072
OUTPUT = 1024
INTERPRETATION = (
    "Exploratory matched 128K screening; no automatic winner, statistical significance, "
    "or no-regression claim. Sampling modes are separate. Sampled continuations may "
    "differ stochastically; matching prompts do not match generated trajectories. "
    "Greedy output hash differences are divergence, not a quality failure. "
    "Separate scored quality checks are required to assess quality."
)
COUNT_FIELDS = ("budget_clipped_rounds", "budget_clipped_drafts", "truncated_rounds",
                "truncated_tokens", "truncated_accepted_tokens")
POSITION_FIELDS = ("proposed", "accepted", "useful_accepted")
HIST_FIELDS = ("verified_depth_histogram", "staged_depth_histogram")


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


def _positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _object(value):
    return value if isinstance(value, dict) else {}


def _identity(row):
    extra = _object(row.get("extra"))
    return (extra.get("boot_id"), extra.get("sampling_mode"), row.get("family"),
            extra.get("repeat"))


def _reference(row):
    return {"key": row.get("key"), "source": row.get("_source"), "variant": row.get("variant")}


def _counter_errors(engine, completion):
    errors = []
    if not all(_integer(engine.get(k), 1 if k.endswith("steps") else 0)
               for k in ("decode_steps", "spec_steps", "draft_tokens", "accepted_tokens")):
        return ["missing-or-invalid-engine-counters"]
    stats = _object(engine.get("spec_positions"))
    if not _integer(stats.get("rounds"), 1):
        return ["missing-or-invalid-position-rounds"]
    rounds = stats["rounds"]
    if rounds != engine["spec_steps"] or engine["decode_steps"] < rounds:
        errors.append("rounds-spec-step-mismatch")
    arrays = [stats.get(k) for k in POSITION_FIELDS]
    if (not all(isinstance(a, list) and 1 <= len(a) <= 4
                and all(_integer(v) for v in a) for a in arrays)
            or len({len(a) for a in arrays}) != 1):
        return errors + ["missing-or-invalid-position-arrays"]
    proposed, accepted, useful = arrays
    if (any(not 0 <= u <= a <= p <= rounds for p, a, u in zip(*arrays))
            or any(any(a[i] < a[i + 1] for i in range(len(a) - 1)) for a in arrays)):
        errors.append("invalid-position-count-order")
    if sum(proposed) != engine["draft_tokens"] or sum(accepted) != engine["accepted_tokens"]:
        errors.append("position-engine-total-mismatch")
    histograms = [stats.get(k) for k in HIST_FIELDS]
    if not all(isinstance(h, dict) and h and all(
            isinstance(d, str) and d in ("1", "2", "3", "4") and _integer(n, 1)
            for d, n in h.items()) for h in histograms):
        return errors + ["missing-or-invalid-depth-histograms"]
    verified, staged = histograms
    if any(sum(h.values()) != rounds for h in histograms):
        errors.append("histogram-rounds-mismatch")
    if (len(proposed) != max(map(int, staged))
            or any(proposed[i] != sum(n for d, n in verified.items() if int(d) > i)
                   for i in range(len(proposed)))
            or max(map(int, verified)) > len(proposed)):
        errors.append("histogram-position-mismatch")
    if any(sum(n for d, n in staged.items() if int(d) >= i)
           < sum(n for d, n in verified.items() if int(d) >= i) for i in range(1, 5)):
        errors.append("verified-depth-exceeds-staged-depth")
    if not all(_integer(stats.get(k)) for k in COUNT_FIELDS):
        return errors + ["missing-or-invalid-clipping-counters"]
    clipped = sum(int(d) * n for d, n in staged.items()) - sum(proposed)
    if (clipped != stats["budget_clipped_drafts"]
            or not 0 <= stats["budget_clipped_rounds"] <= min(rounds, clipped)
            or clipped > 3 * stats["budget_clipped_rounds"]
            or bool(clipped) != bool(stats["budget_clipped_rounds"])):
        errors.append("invalid-budget-clipping-accounting")
    if (sum(accepted) - sum(useful) != stats["truncated_accepted_tokens"]
            or stats["truncated_tokens"] != stats["truncated_accepted_tokens"] + stats["truncated_rounds"]
            or not 0 <= stats["truncated_rounds"] <= rounds
            or stats["truncated_tokens"] > 4 * stats["truncated_rounds"]):
        errors.append("invalid-truncation-accounting")
    # A truncated verify loses its correction/bonus token, even when all accepted
    # drafts survive. Counting useful accepted drafts alone misses this boundary.
    expected = 1 + engine["decode_steps"] + sum(useful) - stats["truncated_rounds"]
    if completion != expected:
        errors.append("emitted-token-accounting-mismatch")
    return errors


def _row_errors(row):
    errors = []
    if not isinstance(row.get("key"), str) or not row["key"]:
        errors.append("missing-record-key")
    if not isinstance(row.get("variant"), str) or not row["variant"]:
        errors.append("missing-variant")
    boot, mode, family, repeat = _identity(row)
    if (not isinstance(boot, str) or not boot or not isinstance(mode, str) or mode not in SAMPLING
            or not isinstance(family, str) or family not in KINDS
            or not _integer(repeat)):
        errors.append("invalid-matrix-coordinates")
    elif row.get("sampling") != SAMPLING[mode]:
        errors.append("sampling-settings-mismatch")
    prompt = _object(row.get("prompt"))
    if not _sha(prompt.get("sha256")):
        errors.append("missing-or-invalid-prompt-hash")
    if not isinstance(row.get("server_instance"), str) or not row["server_instance"]:
        errors.append("missing-server-instance")
    result = _object(row.get("result"))
    if "error" in result or row.get("valid") is False:
        errors.append("failed-measurement")
    if (type(row.get("target_input_tokens")) is not int or row["target_input_tokens"] != CONTEXT
            or type(result.get("prompt_tokens")) is not int or result["prompt_tokens"] != CONTEXT
            or type(row.get("requested_output_tokens")) is not int or row["requested_output_tokens"] != OUTPUT
            or type(result.get("completion_tokens")) is not int or result["completion_tokens"] != OUTPUT):
        errors.append("input-output-count-mismatch")
    if result.get("finish_reason") != "length":
        errors.append("non-length-finish")
    if (_object(row.get("extra")).get("mode") != "cold"
            or type(result.get("cached_tokens")) is not int or result["cached_tokens"] != 0):
        errors.append("not-cold-request")
    if not _sha(result.get("output_sha256")):
        errors.append("missing-or-invalid-output-hash")
    engine = _object(result.get("engine"))
    if not all(_positive(v) for v in (result.get("ttft_s"), result.get("decode_wall_s"), engine.get("decode_s"))):
        errors.append("missing-or-nonpositive-timing")
    elif abs(result["decode_wall_s"] - engine["decode_s"]) > .1 * max(result["decode_wall_s"], engine["decode_s"]):
        errors.append("engine-client-decode-gap")
    errors += _counter_errors(engine, result.get("completion_tokens"))
    staged = _object(_object(engine.get("spec_positions")).get("staged_depth_histogram"))
    fixed = re.fullmatch(r"fixed([1-4])", row.get("variant", "")) if isinstance(row.get("variant"), str) else None
    if fixed and set(staged) != {fixed.group(1)}:
        errors.append("fixed-depth-configuration-mismatch")
    if row.get("variant") == "adaptive2-3" and not set(staged) <= {"2", "3"}:
        errors.append("adaptive-depth-configuration-mismatch")
    return errors


def _geometry(rows):
    result = {}
    for phase in ("before", "after"):
        geometries = [_object(r.get(f"geometry_{phase}")) for r in rows]
        fields = set.union({"moe_cache_size", "num_pages", "page_size"}, *(set(g) for g in geometries))
        result[phase] = {}
        for field in sorted(fields):
            values = [g[field] for g in geometries if type(g.get(field)) in (int, float)
                      and math.isfinite(g[field]) and g[field] >= 0]
            result[phase][field] = {"min": min(values) if values else None,
                                   "max": max(values) if values else None,
                                   "available": len(values), "total": len(rows)}
    return result


def summarize(rows):
    results = [r["result"] for r in rows]
    engines = [r["engine"] for r in results]
    stats = [e["spec_positions"] for e in engines]
    tokens = sum(r["completion_tokens"] - 1 for r in results)
    seconds = sum(r["decode_wall_s"] for r in results)
    steps = sum(e["decode_steps"] for e in engines)
    rounds = sum(e["spec_steps"] for e in engines)
    positions = []
    for i in range(max(len(s["proposed"]) for s in stats)):
        p, a, u = [sum(s[field][i] if i < len(s[field]) else 0 for s in stats)
                   for field in POSITION_FIELDS]
        positions.append({"position": i + 1, "proposed": p, "accepted": a, "useful_accepted": u,
                          "proposal_frequency": p / rounds,
                          "acceptance_when_proposed": a / p if p else None,
                          "useful_accepted_per_round": u / rounds})
    histograms = {}
    for field in HIST_FIELDS:
        counts = Counter()
        for s in stats:
            counts.update(s[field])
        histograms[field] = dict(sorted(counts.items(), key=lambda item: int(item[0])))
    return {
        "n": len(rows), "decode_tps": tokens / seconds, "emitted_decode_tokens": tokens,
        "client_decode_s": seconds, "decode_steps": steps, "spec_steps": rounds,
        "useful_accepted_tokens": sum(sum(s["useful_accepted"]) for s in stats),
        "useful_accepted_per_round": sum(sum(s["useful_accepted"]) for s in stats) / rounds,
        "emitted_tokens_per_decode_step": tokens / steps,
        "engine_step_ms": 1000 * sum(e["decode_s"] for e in engines) / steps,
        "positions": positions, **histograms,
        **{field: sum(s[field] for s in stats) for field in COUNT_FIELDS},
        "geometry": _geometry(rows),
    }


def analyze(rows, control="adaptive2-3"):
    measured = [r for r in rows if isinstance(r, dict) and r.get("suite") == "mixed"]
    report = {"schema_version": 1, "valid": False, "control_variant": control,
              "interpretation": INTERPRETATION, "input_records": len(rows),
              "measured_records": len(measured), "excluded_nonmixed_records": len(rows) - len(measured),
              "protocol": {"input_tokens": CONTEXT, "output_tokens": OUTPUT,
                           "families": list(KINDS), "sampling": SAMPLING, "concurrency": 1},
              "errors": [], "variants": {}, "greedy_pairs": []}
    errors = report["errors"]
    for row in measured:
        reasons = _row_errors(row)
        if reasons:
            errors.append({**_reference(row), "reasons": reasons})
    if not measured:
        errors.append({"reasons": ["no-measured-mixed-records"]})
    if errors:
        return report
    indices = defaultdict(dict)
    keys = Counter(r["key"] for r in measured)
    instances = defaultdict(set)
    for row in measured:
        variant, identity = row["variant"], _identity(row)
        if keys[row["key"]] != 1 or identity in indices[variant]:
            errors.append({**_reference(row), "reasons": ["duplicate-key-or-matrix-coordinate"]})
        indices[variant][identity] = row
        instances[(variant, identity[0])].add(row["server_instance"])
    if control not in indices or len(indices) < 2:
        errors.append({"reasons": ["control-and-candidate-required"]})
    for (variant, boot), ids in instances.items():
        if len(ids) != 1:
            errors.append({"variant": variant, "boot_id": boot, "reasons": ["multiple-processes-in-one-boot"]})
    if len(set.union(set(), *instances.values())) != len(instances):
        errors.append({"reasons": ["server-process-shared-across-arms-or-boots"]})
    for variant, index in indices.items():
        blocks = {(identity[0], identity[3]) for identity in index}
        expected = {(boot, mode, family, repeat) for boot, repeat in blocks
                    for mode, family in product(SAMPLING, KINDS)}
        if set(index) != expected:
            errors.append({"variant": variant, "reasons": ["incomplete-six-cell-matrix"],
                           "missing_coordinates": sorted(expected - set(index))})
        if control in indices and set(index) != set(indices[control]):
            errors.append({"variant": variant, "reasons": ["unmatched-matrix-coordinates"]})
        if control in indices:
            for identity in index.keys() & indices[control].keys():
                row, reference = index[identity], indices[control][identity]
                if row["prompt"]["sha256"] != reference["prompt"]["sha256"]:
                    errors.append({**_reference(row), "control_record": _reference(reference),
                                   "reasons": ["unmatched-prompt-hash"]})
    if errors:
        return report
    report["valid"] = True
    for variant, index in sorted(indices.items()):
        selected = list(index.values())
        boots = sorted({r["extra"]["boot_id"] for r in selected})
        report["variants"][variant] = {
            "boot_ids": boots, "independent_boot_count": len(boots),
            "one_boot_only": len(boots) == 1,
            "modes": {mode: summarize([r for r in selected if r["extra"]["sampling_mode"] == mode])
                      for mode in SAMPLING},
            "cells": [{"sampling_mode": mode, "family": family,
                       **summarize([r for r in selected if r["extra"]["sampling_mode"] == mode and r["family"] == family])}
                      for mode, family in product(SAMPLING, KINDS)],
        }
        if variant == control:
            continue
        for identity, row in sorted(index.items()):
            if identity[1] != "greedy":
                continue
            reference = indices[control][identity]
            report["greedy_pairs"].append({
                "variant": variant, "boot_id": identity[0], "family": identity[2], "repeat": identity[3],
                "prompt_sha256": row["prompt"]["sha256"], "control_record": _reference(reference),
                "candidate_record": _reference(row), "control_output_sha256": reference["result"]["output_sha256"],
                "candidate_output_sha256": row["result"]["output_sha256"],
                "status": "identical" if row["result"]["output_sha256"] == reference["result"]["output_sha256"] else "divergence",
            })
    return report


def render(report):
    lines = ["# MTP depth comparison at 128K", "", report["interpretation"], ""]
    if not report["valid"]:
        lines += ["**Validation failed. No performance comparison was produced.**", ""]
        for error in report["errors"]:
            lines.append(f"- {error.get('key', error.get('variant', 'Input'))}: {', '.join(error['reasons'])}")
        return "\n".join(lines) + "\n"
    lines += [f"Input: {CONTEXT:,} tokens; output: {OUTPUT:,}; concurrency: 1. "
              f"Measured requests: {report['measured_records']}; excluded nonmixed rows: {report['excluded_nonmixed_records']}.", "",
              "Rates use total (completion tokens − 1) / total client decode seconds. "
              "Useful accepted drafts use verification rounds as denominator; emitted tokens and step latency use all decode steps.", "",
              "| Variant | Sampling | n | Boots | tok/s | Useful accepted/round | Emitted/step | ms/step |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for variant, data in report["variants"].items():
        for mode, s in data["modes"].items():
            lines.append(f"| {variant} | {mode} | {s['n']} | {len(data['boot_ids'])} | {s['decode_tps']:.2f} | "
                         f"{s['useful_accepted_per_round']:.3f} | {s['emitted_tokens_per_decode_step']:.3f} | {s['engine_step_ms']:.3f} |")
    if any(data["one_boot_only"] for data in report["variants"].values()):
        lines += ["", "**One boot per arm is exploratory screening; request repetitions do not establish independent boot replication.**"]
    lines += ["", "| Variant | Sampling | Position | Proposed | Accepted | Proposal frequency | Acceptance when proposed | Useful accepted/round |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for variant, data in report["variants"].items():
        for mode, s in data["modes"].items():
            for p in s["positions"]:
                acceptance = "—" if p["acceptance_when_proposed"] is None else f"{100 * p['acceptance_when_proposed']:.2f}%"
                lines.append(f"| {variant} | {mode} | {p['position']} | {p['proposed']} | {p['accepted']} | "
                             f"{100 * p['proposal_frequency']:.2f}% | {acceptance} | {p['useful_accepted_per_round']:.3f} |")
    lines += ["", "Acceptance is accepted prefix count / proposed count. Proposal frequency is proposed count / verification rounds. "
              "The JSON preserves both depth histograms and budget/terminal clipping counts.", "",
              "| Variant | Sampling | Family | n | tok/s | Useful accepted/round |",
              "| --- | --- | --- | ---: | ---: | ---: |"]
    for variant, data in report["variants"].items():
        for cell in data["cells"]:
            lines.append(f"| {variant} | {cell['sampling_mode']} | {cell['family']} | {cell['n']} | "
                         f"{cell['decode_tps']:.2f} | {cell['useful_accepted_per_round']:.3f} |")
    lines += ["", "| Variant | Sampling | Cache slots before | Cache slots after | KV pages before | KV pages after |",
              "| --- | --- | --- | --- | --- | --- |"]
    def span(s, phase, field):
        value = s["geometry"][phase][field]
        if value["available"] != value["total"]:
            return f"unavailable ({value['available']}/{value['total']} recorded)"
        return str(value["min"]) if value["min"] == value["max"] else f"{value['min']}–{value['max']}"
    for variant, data in report["variants"].items():
        for mode, s in data["modes"].items():
            cells = [span(s, phase, field) for field in ("moe_cache_size", "num_pages") for phase in ("before", "after")]
            lines.append(f"| {variant} | {mode} | " + " | ".join(cells) + " |")
    pairs = report["greedy_pairs"]
    divergent = sum(p["status"] == "divergence" for p in pairs)
    lines += ["", f"Greedy outputs compared against {report['control_variant']}: {len(pairs) - divergent}/{len(pairs)} identical hashes; "
              f"{divergent} divergences. Actual record keys, source lines and both hashes are retained in JSON. "
              "Divergence is not a scored quality failure; identical outputs on these prompts do not establish general quality equivalence.", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--control", default="adaptive2-3")
    args = parser.parse_args(argv)
    path = args.out / "runs.jsonl"
    try:
        rows = []
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"Line {number} must contain an object")
                rows.append({**row, "_source": f"{path.resolve()}:{number}"})
        report = analyze(rows, args.control)
    except (OSError, ValueError) as error:
        report = analyze([], args.control)
        report["errors"] = [{"reasons": [str(error)]}]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "mtp-depth-report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (args.out / "mtp-depth-report.md").write_text(render(report))
    print(f"MTP depth report: {'valid exploratory comparison' if report['valid'] else 'validation failed'}; {args.out / 'mtp-depth-report.md'}")
    return 0 if report["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
