"""Describe completed captured outputs without scoring quality or contacting a server.

Run against one cell directory; an active JSONL append is ignored until newline.
Example: python -m benchmark.output_diagnostics RESULTS/cells/greedy-japanese-253952 \
    --original-results ORIGINAL --out-prefix RESULTS/japanese-output-diagnostics
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

from .mixed import eligible

CHAPTER = re.compile(r"第\s*[\d一二三四五六七八九十百千万〇零]+\s*章")


def complete_rows(path: Path) -> tuple[list[dict], bool]:
    raw = path.read_bytes()
    complete, _, tail = raw.rpartition(b"\n")
    return [json.loads(line) for line in complete.splitlines() if line.strip()], bool(tail)


def text_features(text: str) -> dict:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    counts = Counter(lines)
    grams = [text[i:i + 16] for i in range(max(0, len(text) - 15))]
    tail = text[-1000:]
    return {
        "characters": len(text), "decimal_digits": sum(c.isdecimal() for c in text),
        "decimal_digit_fraction": sum(c.isdecimal() for c in text) / len(text) if text else None,
        "chapter_references": len(CHAPTER.findall(text)),
        "last_1000_characters": len(tail),
        "last_1000_decimal_digit_fraction": sum(c.isdecimal() for c in tail) / len(tail) if tail else None,
        "last_1000_chapter_references": len(CHAPTER.findall(tail)),
        "nonempty_lines": len(lines), "unique_nonempty_lines": len(counts),
        "duplicate_line_occurrences": len(lines) - len(counts),
        "longest_nonempty_line_characters": max(map(len, lines), default=0),
        "repeated_16char_window_fraction": (len(grams) - len(set(grams))) / len(grams) if grams else None,
        "repeated_lines": [{"text": line[:160], "count": count}
                           for line, count in counts.most_common(5) if count > 1],
        "head": text[:200], "tail": text[-200:],
    }


def request_metrics(row: dict) -> dict:
    result = row["result"]
    engine = result.get("engine") or {}
    steps, drafts = engine.get("decode_steps"), engine.get("draft_tokens")
    accepted = engine.get("accepted_tokens")
    windows = (row.get("moe_windows") or {}).get("rows", [])
    def weighted(field):
        present = [w for w in windows if field in w and w.get("layer_calls", 0) > 0]
        count = sum(w["layer_calls"] for w in present)
        return sum(w[field] * w["layer_calls"] for w in present) / count if count else None
    return {
        "decode_tps": (result["completion_tokens"] - 1) / result["decode_wall_s"],
        "completion_tokens": result["completion_tokens"],
        "emitted_tokens_per_step": (result["completion_tokens"] - 1) / steps if steps else None,
        "acceptance": accepted / drafts if drafts and accepted is not None else None,
        "engine_step_ms": engine["decode_s"] * 1000 / steps if steps and engine.get("decode_s") is not None else None,
        "characters_per_output_token": result["output_chars"] / result["completion_tokens"],
        "moe_window_count": len(windows), "cpu_experts_per_layer_est": weighted("cpu/layer"),
        "missing_experts_per_layer_est": weighted("missing/layer"),
        "fetch_mib_per_step_est": weighted("fetch_MB/step"),
        "last_moe_window": windows[-1] if windows else None,
    }


def analyze(directory: Path, *, variant: str = "baseline-followup", original: Path | None = None) -> dict:
    rows, incomplete = complete_rows(directory / "runs.jsonl")
    selected = [r for r in rows if r.get("suite") == "mixed" and r.get("variant") == variant]
    report = {"snapshot_utc": datetime.now(timezone.utc).isoformat(), "directory": str(directory.resolve()),
              "variant": variant, "complete_selected_records": len(selected), "incomplete_tail_ignored": incomplete,
              "requests": [], "unavailable": [],
              "interpretation": "Descriptive text features only. Chapter references, digits and repetition "
                                "are not quality-failure labels. Rates, content and cache behavior can covary; "
                                "this comparison does not establish causality or statistical significance."}
    for row in selected:
        result = row.get("result") or {}
        reference = {"key": row.get("key"), "boot": row.get("extra", {}).get("boot_id"),
                     "repeat": row.get("extra", {}).get("repeat")}
        relative = result.get("output_artifact")
        if not eligible(row) or not relative:
            report["unavailable"].append({**reference, "reason": "ineligible record or uncaptured output"})
            continue
        path = (directory / relative).resolve()
        try:
            if not path.is_relative_to(directory.resolve()):
                raise ValueError("output artifact is outside the result directory")
            artifact = json.loads(path.read_text())
            text = artifact["text"]
            if (not isinstance(text, str) or artifact.get("record_key") != row.get("key")
                    or len(text) != result.get("output_chars")
                    or hashlib.sha256(text.encode()).hexdigest() != result.get("output_sha256")
                    or artifact.get("output_sha256") != result.get("output_sha256")):
                raise ValueError("captured output identity, length or hash mismatch")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            report["unavailable"].append({**reference, "reason": str(exc)})
            continue
        report["requests"].append({**reference, "artifact": str(path), "output_sha256": result["output_sha256"],
                                   "geometry_before": row.get("geometry_before"), "geometry_after": row.get("geometry_after"),
                                   "metrics": request_metrics(row), "text": text_features(text)})
    if original is not None:
        original_rows, original_tail = complete_rows(original / "runs.jsonl")
        cells = {(r.get("family"), r.get("target_input_tokens"), r.get("extra", {}).get("sampling_mode")) for r in selected}
        combined = [r for r in original_rows if r.get("suite") == "mixed" and r.get("variant") == "combined"
                    and (r.get("family"), r.get("target_input_tokens"), r.get("extra", {}).get("sampling_mode")) in cells]
        report["original_combined_output_availability"] = {
            "directory": str(original.resolve()), "matching_cell_records": len(combined),
            "records_with_capture_reference": sum(bool((r.get("result") or {}).get("output_artifact")) for r in combined),
            "incomplete_tail_ignored": original_tail,
            "note": "No original combined text is reconstructed from hashes, character counts or truncated output heads.",
        }
    return report


def render(report: dict) -> str:
    def fmt(value, percent=False):
        return "—" if value is None else f"{value * (100 if percent else 1):.2f}"
    lines = ["# Captured-output diagnostics", "", f"Snapshot: {report['snapshot_utc']}. "
             f"Variant: `{report['variant']}`; {len(report['requests'])} verified captured outputs from "
             f"{report['complete_selected_records']} completed records. "
             f"Incomplete append ignored: {report['incomplete_tail_ignored']}.", "", report["interpretation"], "",
             "| Boot / repeat | tok/s | Tokens/step | Acceptance % | Step ms | Chars | Digit % | Chapter refs | Tail refs / digit % | Lines / duplicate occurrences | Repeated 16-char windows % |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in report["requests"]:
        m, t = row["metrics"], row["text"]
        lines.append(f"| {row['boot']} / {row['repeat']} | {fmt(m['decode_tps'])} | {fmt(m['emitted_tokens_per_step'])} | "
                     f"{fmt(m['acceptance'],True)} | {fmt(m['engine_step_ms'])} | {t['characters']} | "
                     f"{fmt(t['decimal_digit_fraction'],True)} | {t['chapter_references']} | "
                     f"{t['last_1000_chapter_references']} / {fmt(t['last_1000_decimal_digit_fraction'],True)} | "
                     f"{t['nonempty_lines']} / {t['duplicate_line_occurrences']} | {fmt(t['repeated_16char_window_fraction'],True)} |")
    lines += ["", "Digit fraction counts Unicode decimal digits / all Unicode characters. Chapter references "
              "match `第<number>章` anywhere, including prose references, using decimal or Japanese numeral characters. "
              "Tail measures use at most the last 1000 characters. Lines exclude blanks and strip surrounding whitespace. "
              "Duplicate occurrences count occurrences beyond each distinct line's first appearance. "
              "Repeated 16-character windows = (all overlapping windows − distinct windows) / all windows; "
              "the text is not normalized for this metric.", "",
              "| Boot / repeat | MoE windows | CPU experts/layer est. | Misses/layer est. | Fetch MiB/step est. |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for row in report["requests"]:
        m = row["metrics"]
        lines.append(f"| {row['boot']} / {row['repeat']} | {m['moe_window_count']} | "
                     f"{fmt(m['cpu_experts_per_layer_est'])} | {fmt(m['missing_experts_per_layer_est'])} | {fmt(m['fetch_mib_per_step_est'])} |")
    lines += ["", "MoE estimates weight rounded diagnostic windows by layer calls. Windows may cross request "
              "boundaries; they are not exact request totals or aligned with text segments. Last-window values "
              "and cache geometry remain in the JSON report.", ""]
    if original := report.get("original_combined_output_availability"):
        lines += [f"Original combined: {original['matching_cell_records']} matching-cell records, "
                  f"{original['records_with_capture_reference']} with full-output capture references. "
                  "Original uncaptured continuations are not retrospectively available; no text-style "
                  "comparison against them is made.", ""]
    for row in report["requests"]:
        lines += [f"## Boot {row['boot']}, repeat {row['repeat']}", "",
                  f"[Verified output artifact](<{row['artifact']}>). First 200 characters:", ""]
        lines.extend("> " + s for s in row["text"]["head"].splitlines())
        lines += ["", "Last 200 characters:", ""]
        lines.extend("> " + s for s in row["text"]["tail"].splitlines())
        lines.append("")
    if report["unavailable"]:
        lines += ["Unavailable records:", ""]
        lines.extend(f"- {r['key']}: {r['reason']}" for r in report["unavailable"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--variant", default="baseline-followup")
    parser.add_argument("--original-results", type=Path)
    parser.add_argument("--out-prefix", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.directory, variant=args.variant, original=args.original_results)
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    Path(str(args.out_prefix) + ".json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    text = render(report)
    Path(str(args.out_prefix) + ".md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
