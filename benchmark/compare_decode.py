"""Compare only matched mixed-decode requests, without contacting a server.

Example (run from the repository root)::

    python -m benchmark.compare_decode benchmark/results/decode-round6-qualification

Pairs must share boot label, sampling mode, content family, context, repetition,
prompt SHA-256 and output budget. Extra candidate repetitions are reported but
never included in the paired rate. Results are descriptive and exploratory;
two boot pairs per cell cannot establish statistical significance or absence of
regressions. No performance result changes this command's exit status.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path

from .mixed import CONTEXTS, KINDS, SAMPLING, eligible, qualification
from .validate import check


def load_records(directories: list[str]) -> list[dict]:
    rows = []
    for directory in directories:
        path = Path(directory) / "runs.jsonl"
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            if line.strip():
                row = json.loads(line)
                row = {**row, "_source": f"{path.resolve()}:{line_number}"}
                rows.append(row)
    return rows


def _identity(row: dict) -> tuple | None:
    extra = row.get("extra") or {}
    boot, mode, repeat = extra.get("boot_id"), extra.get("sampling_mode"), extra.get("repeat")
    family, context = row.get("family"), row.get("target_input_tokens")
    if (not isinstance(boot, str) or not boot or mode not in SAMPLING or family not in KINDS
            or type(repeat) is not int or repeat < 0 or type(context) is not int or context <= 0):
        return None
    return boot, mode, family, context, repeat


def _coordinates(identity: tuple) -> dict:
    return dict(zip(("boot_id", "sampling_mode", "family", "context_tokens", "repeat"), identity))


def _reference(row: dict) -> dict:
    return {"key": row.get("key"), "source": row.get("_source"), "variant": row.get("variant")}


def _finite(value) -> bool:
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def summarize(rows: list[dict]) -> dict:
    """Compute ratios from totals, preserving missing engine counters as unavailable."""
    results = [r["result"] for r in rows]
    tokens = sum(r["completion_tokens"] - 1 for r in results)
    seconds = sum(r["decode_wall_s"] for r in results)
    engines = [r.get("engine") or {} for r in results]

    def total(field):
        values = [e.get(field) for e in engines]
        return sum(values) if values and all(_finite(v) and v >= 0 for v in values) else None

    steps, accepted, drafts, engine_seconds = [total(f) for f in
                                              ("decode_steps", "accepted_tokens", "draft_tokens", "decode_s")]
    ttfts = [r.get("ttft_s") for r in results]
    return {
        "n": len(rows), "emitted_decode_tokens": tokens, "decode_s": seconds,
        "decode_tps": tokens / seconds if seconds else None,
        "tpot_ms": seconds * 1000 / tokens if tokens else None,
        "mean_decode_s": seconds / len(rows) if rows else None,
        "mean_ttft_s": statistics.mean(ttfts) if ttfts and all(_finite(v) for v in ttfts) else None,
        "decode_steps": steps, "accepted_tokens": accepted, "draft_tokens": drafts,
        "engine_step_ms": engine_seconds * 1000 / steps if steps and engine_seconds is not None else None,
        "emitted_tokens_per_step": tokens / steps if steps else None,
        "accepted_tokens_per_step": accepted / steps if steps and accepted is not None else None,
        "acceptance": accepted / drafts if drafts and accepted is not None else None,
        "engine_field_coverage": {field: sum(_finite(e.get(field)) for e in engines)
                                  for field in ("decode_steps", "accepted_tokens", "draft_tokens", "decode_s")},
    }


def _geometry(a: dict, b: dict) -> dict:
    differences, unavailable, unavailable_fields, transitions = [], [], [], {}
    def available(value):
        return (isinstance(value, dict) and "error" not in value
                and any(v is not None for v in value.values()))
    for phase in ("before", "after"):
        left, right = a.get(f"geometry_{phase}"), b.get(f"geometry_{phase}")
        if not available(left) or not available(right):
            unavailable.append(phase)
            continue
        for field in sorted(left.keys() | right.keys()):
            missing = [name for name, value in (("baseline", left.get(field)), ("candidate", right.get(field)))
                       if value is None]
            if missing:
                unavailable_fields.append({"field": f"{phase}.{field}", "sides": missing})
            elif left[field] != right[field]:
                differences.append({"field": f"{phase}.{field}",
                                    "baseline": left[field], "candidate": right[field]})
    for name, row in (("baseline", a), ("candidate", b)):
        before, after = row.get("geometry_before"), row.get("geometry_after")
        if available(before) and available(after):
            changes = [{"field": field, "before": before[field], "after": after[field]}
                       for field in sorted(before.keys() | after.keys())
                       if before.get(field) is not None and after.get(field) is not None
                       and before[field] != after[field]]
            if changes:
                transitions[name] = changes
    return {"differences": differences, "unavailable_phases": unavailable,
            "unavailable_fields": unavailable_fields,
            "within_request_changes": transitions}


def _comparison(pairs: list[tuple[dict, dict]]) -> dict:
    baseline = summarize([a for a, _ in pairs])
    candidate = summarize([b for _, b in pairs])
    return {"pairs": len(pairs), "baseline": baseline, "candidate": candidate,
            "decode_ratio": candidate["decode_tps"] / baseline["decode_tps"] if pairs else None}


def analyze(rows: list[dict], baseline: str = "baseline", candidate: str = "combined") -> dict:
    if baseline == candidate:
        raise ValueError("baseline and candidate must be different variants")
    selected = [r for r in rows if r.get("suite") == "mixed" and r.get("variant") in (baseline, candidate)]
    indices = {name: defaultdict(list) for name in (baseline, candidate)}
    key_counts = Counter((r.get("variant"), r.get("key")) for r in selected)
    invalid, unmatched, eligible_rows = [], [], {baseline: [], candidate: []}
    for row in selected:
        identity = _identity(row)
        if identity is None:
            invalid.append({**_reference(row), "reasons": ["missing-or-invalid-pair-coordinates"]})
        else:
            indices[row["variant"]][identity].append(row)

    valid = {baseline: {}, candidate: {}}
    for variant, index in indices.items():
        for identity, records in index.items():
            for row in records:
                reasons = []
                if len(records) != 1 or key_counts[(variant, row.get("key"))] != 1:
                    reasons.append("duplicate-record-key-or-pair-coordinates")
                if not isinstance(row.get("key"), str) or not row["key"]:
                    reasons.append("missing-record-key")
                sha = (row.get("prompt") or {}).get("sha256")
                if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", sha):
                    reasons.append("missing-or-invalid-prompt-sha256")
                if (row.get("extra") or {}).get("mode", "cold") != "cold":
                    reasons.append("not-cold-mode")
                if not eligible(row):
                    _, flags = check(row)
                    reasons.extend(["ineligible-mixed-record", *flags])
                if reasons:
                    invalid.append({**_reference(row), **_coordinates(identity), "reasons": reasons})
                else:
                    valid[variant][identity] = row
                    eligible_rows[variant].append(row)

    pairs_by_mode = defaultdict(list)
    pair_details = []
    for identity in sorted(valid[baseline].keys() | valid[candidate].keys()):
        a, b = valid[baseline].get(identity), valid[candidate].get(identity)
        reason = None
        if a is None or b is None:
            reason = "no-eligible-counterpart"
        elif a["prompt"]["sha256"].lower() != b["prompt"]["sha256"].lower():
            reason = "prompt-sha256-mismatch"
        elif a["requested_output_tokens"] != b["requested_output_tokens"]:
            reason = "output-budget-mismatch"
        elif (not a.get("server_instance") or not b.get("server_instance")
              or a["server_instance"] == b["server_instance"]):
            reason = "missing-or-shared-server-instance"
        if reason:
            for row in (a, b):
                if row is not None:
                    unmatched.append({**_reference(row), **_coordinates(identity), "reason": reason})
            continue
        pairs_by_mode[identity[1]].append((a, b))
        pair_details.append({**_coordinates(identity), "prompt_sha256": a["prompt"]["sha256"],
                             "output_tokens": a["requested_output_tokens"],
                             "baseline_record": _reference(a), "candidate_record": _reference(b),
                             "baseline_instance": a["server_instance"], "candidate_instance": b["server_instance"],
                             **_comparison([(a, b)]), "geometry": _geometry(a, b)})

    modes = {}
    for mode in SAMPLING:
        pairs = pairs_by_mode[mode]
        cells = defaultdict(list)
        boot_instances = {name: defaultdict(set) for name in (baseline, candidate)}
        for a, b in pairs:
            cells[(a["family"], a["target_input_tokens"])].append((a, b))
            for name, row in ((baseline, a), (candidate, b)):
                boot_instances[name][row["extra"]["boot_id"]].add(row["server_instance"])
        cell_results = []
        for (family, context), cell_pairs in sorted(cells.items()):
            boots = defaultdict(list)
            for a, b in cell_pairs:
                boots[a["extra"]["boot_id"]].append((a, b))
            boot_ratios = {boot: _comparison(bp)["decode_ratio"] for boot, bp in sorted(boots.items())}
            # Labels alone do not prove separate boots. Each side must have one unique
            # process per label, and all physical processes must be distinct.
            instances = [boot_instances[name][boot]
                         for name in (baseline, candidate) for boot in boots]
            independent = (len(boots) >= 2 and all(len(ids) == 1 for ids in instances)
                           and len(set.union(*instances)) == len(instances))
            negative = [boot for boot, ratio in boot_ratios.items() if ratio < 1]
            cell_results.append({"family": family, "context_tokens": context,
                                 **_comparison(cell_pairs), "boot_ratios": boot_ratios,
                                 "independent_boots": independent, "slower_boots": negative,
                                 "follow_up": bool(negative),
                                 "slower_across_all_independent_boots": independent and len(negative) == len(boots)})
        canonical = set(product(KINDS, CONTEXTS))
        mode_rows = {name: [r for r in eligible_rows[name] if r["extra"]["sampling_mode"] == mode]
                     for name in (baseline, candidate)}
        modes[mode] = {**_comparison(pairs), "cells": cell_results,
                       "boot_instances": {name: {boot: sorted(ids) for boot, ids in sorted(mapping.items())}
                                          for name, mapping in boot_instances.items()},
                       "all_canonical_cells_matched": set(cells) == canonical,
                       "minimum_pairs_per_cell": min((len(p) for p in cells.values()), default=0),
                       "eligible_records": {name: len(rs) for name, rs in mode_rows.items()},
                       "full_qualification_of_all_eligible_records": {name: qualification(rs) for name, rs in mode_rows.items()},
                       "unmatched_records": sum(r["sampling_mode"] == mode for r in unmatched)}
    return {
        "baseline_variant": baseline, "candidate_variant": candidate,
        "interpretation": "Exploratory matched comparisons only; no significance or no-regression claim. "
                          "Sampling modes remain separate; unmatched extra candidate repeats do not affect paired rates. "
                          "These requests use a synthetic template corpus. Geometry and generated outputs can differ.",
        "input_records": len(selected), "matched_pairs": len(pair_details),
        "invalid_records": invalid, "unmatched_records": unmatched,
        "modes": modes, "pair_details": pair_details,
    }


def render(report: dict) -> str:
    def fmt(value, digits=2):
        return "—" if value is None else f"{value:.{digits}f}"

    def change(ratio):
        return "—" if ratio is None else f"{(ratio - 1) * 100:+.2f}%"

    lines = [f"# Matched decode comparison: {report['candidate_variant']} / {report['baseline_variant']}", "",
             report["interpretation"], "",
             f"Matched pairs: {report['matched_pairs']}; invalid records: {len(report['invalid_records'])}; "
             f"unmatched records: {len(report['unmatched_records'])}.", "",
             "Rates divide total emitted decode tokens by total first-to-last-token seconds. "
             "Every aggregate below uses exactly the same paired requests on both sides. "
             "A slower cell is a follow-up signal, not a significance test.", "",
             "| Sampling | Pairs | Baseline tok/s | Candidate tok/s | Change | Cells | Min pairs/cell |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for mode, data in report["modes"].items():
        lines.append(f"| {mode} | {data['pairs']} | {fmt(data['baseline']['decode_tps'])} | "
                     f"{fmt(data['candidate']['decode_tps'])} | {change(data['decode_ratio'])} | "
                     f"{len(data['cells'])}/12 | {data['minimum_pairs_per_cell']} |")
    lines += ["", "| Sampling | Metric | Baseline | Candidate |", "| --- | --- | ---: | ---: |"]
    for mode, data in report["modes"].items():
        for field in ("tpot_ms", "mean_decode_s", "mean_ttft_s", "engine_step_ms",
                      "emitted_tokens_per_step", "accepted_tokens_per_step", "acceptance"):
            lines.append(f"| {mode} | {field} | {fmt(data['baseline'][field], 3)} | {fmt(data['candidate'][field], 3)} |")
    lines += ["", "| Sampling | Content | Input | Pairs | Baseline tok/s | Candidate tok/s | Change | Boot changes | Independent boots | Follow-up |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |"]
    for mode, data in report["modes"].items():
        for cell in data["cells"]:
            boots = ", ".join(f"{boot}: {change(ratio)}" for boot, ratio in cell["boot_ratios"].items())
            follow_up = ("slower in every independent boot" if cell["slower_across_all_independent_boots"]
                         else "slower in some observed boots" if cell["follow_up"] else "no slower observed boot")
            lines.append(f"| {mode} | {cell['family']} | {cell['context_tokens']} | {cell['pairs']} | "
                         f"{fmt(cell['baseline']['decode_tps'])} | {fmt(cell['candidate']['decode_tps'])} | "
                         f"{change(cell['decode_ratio'])} | {boots} | {cell['independent_boots']} | {follow_up} |")
    lines += ["", "Matrix completeness is evaluated over each variant's complete eligible dataset, "
              "independently of this smaller paired subset. Quality and regression gates are assessed separately:", ""]
    for mode, data in report["modes"].items():
        states = "; ".join(f"{variant}: {data['eligible_records'][variant]} eligible, full matrix={full}"
                           for variant, full in data["full_qualification_of_all_eligible_records"].items())
        lines.append(f"- {mode}: {states}.")
    geometries = defaultdict(list)
    for pair in report["pair_details"]:
        for difference in pair["geometry"]["differences"]:
            geometries[(pair["sampling_mode"], difference["field"])].append(difference)
    missing = sum(bool(p["geometry"]["unavailable_phases"] or p["geometry"]["unavailable_fields"])
                  for p in report["pair_details"])
    transitions = sum(bool(p["geometry"]["within_request_changes"]) for p in report["pair_details"])
    lines += ["", f"Geometry unavailable for {missing} pairs; geometry changed during a request in {transitions} pairs. "
              "All changes and record references are retained in the JSON report.", "",
              "| Sampling | Geometry field | Different pairs | Baseline values | Candidate values |",
              "| --- | --- | ---: | --- | --- |"]
    for (mode, field), differences in sorted(geometries.items()):
        values = lambda side: ", ".join(sorted({json.dumps(d[side], sort_keys=True) for d in differences}))
        lines.append(f"| {mode} | {field} | {len(differences)} | {values('baseline')} | {values('candidate')} |")
    reasons = Counter(r["reason"] for r in report["unmatched_records"])
    reasons.update(reason for row in report["invalid_records"] for reason in row["reasons"])
    lines += ["", "Excluded/unmatched reasons (an invalid row can have multiple reasons):", ""]
    lines.extend(f"- {reason}: {count}" for reason, count in sorted(reasons.items()))
    if not reasons:
        lines.append("- None.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+")
    parser.add_argument("--baseline", default="baseline")
    parser.add_argument("--candidate", default="combined")
    parser.add_argument("--out-prefix", help="Output filename prefix; default: first directory/decode-comparison")
    args = parser.parse_args()
    report = analyze(load_records(args.directories), args.baseline, args.candidate)
    prefix = Path(args.out_prefix or str(Path(args.directories[0]) / "decode-comparison"))
    prefix.parent.mkdir(parents=True, exist_ok=True)
    Path(str(prefix) + ".json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    markdown = render(report)
    Path(str(prefix) + ".md").write_text(markdown)
    print(markdown)


if __name__ == "__main__":
    main()
