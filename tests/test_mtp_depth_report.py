"""MTP reports preserve matched matrices and distinguish acceptance denominators."""
from copy import deepcopy
import hashlib
from itertools import product
import json

import pytest

from benchmark.mixed import KINDS, SAMPLING
from benchmark.mtp_depth_report import analyze, main, render


def record(variant="adaptive2-3", boot="1", mode="greedy", family="code", repeat=0,
           seconds=10, rounds=400):
    identity = f"{boot}/{mode}/{family}/{repeat}"
    # Distinct verification and decode counts catch denominator mixups.
    accepted = [300, 222, 100] if rounds == 400 else [300, 100, 22]
    return {
        "key": f"mixed/{variant}/{identity}", "suite": "mixed", "variant": variant,
        "family": family, "target_input_tokens": 131072, "requested_output_tokens": 1024,
        "server_instance": f"server-{variant}-{boot}", "sampling": SAMPLING[mode],
        "extra": {"boot_id": boot, "sampling_mode": mode, "repeat": repeat, "mode": "cold"},
        "prompt": {"sha256": hashlib.sha256(identity.encode()).hexdigest()},
        "result": {
            "prompt_tokens": 131072, "completion_tokens": 1024, "cached_tokens": 0,
            "ttft_s": 36, "decode_wall_s": seconds, "finish_reason": "length",
            "output_sha256": "a" * 64,
            "engine": {"decode_s": seconds, "decode_steps": rounds + 1, "spec_steps": rounds,
                       "draft_tokens": rounds * 2 + rounds // 2, "accepted_tokens": sum(accepted),
                       "spec_positions": {
                           "rounds": rounds, "proposed": [rounds, rounds, rounds // 2],
                           "accepted": accepted, "useful_accepted": list(accepted),
                           "verified_depth_histogram": {"2": rounds // 2, "3": rounds // 2},
                           "staged_depth_histogram": {"2": rounds // 2, "3": rounds // 2},
                           "budget_clipped_rounds": 0, "budget_clipped_drafts": 0,
                           "truncated_rounds": 0, "truncated_tokens": 0, "truncated_accepted_tokens": 0,
                       }},
        },
        "geometry_before": {"num_pages": 2304, "page_size": 64, "moe_cache_size": 6900},
        "geometry_after": {"num_pages": 2304, "page_size": 64, "moe_cache_size": 6900},
    }


def matrix(variants=("adaptive2-3", "candidate"), boots=("1",), repeats=(0,)):
    return [record(variant, boot, mode, family, repeat)
            for variant, boot, repeat, mode, family in product(variants, boots, repeats, SAMPLING, KINDS)]


def test_ratios_use_pooled_time_and_correct_round_denominators():
    rows = matrix()
    row = next(r for r in rows if r["variant"] == "candidate" and r["family"] == "code"
               and r["extra"]["sampling_mode"] == "greedy")
    replacement = record("candidate", seconds=30, rounds=600)
    rows[rows.index(row)] = replacement
    report = analyze(rows)
    assert report["valid"], report["errors"]
    s = report["variants"]["candidate"]["modes"]["greedy"]
    assert s["decode_tps"] == pytest.approx(3 * 1023 / 50)
    assert s["useful_accepted_per_round"] == pytest.approx((422 + 622 + 622) / 1400)
    assert s["emitted_tokens_per_decode_step"] == pytest.approx(3069 / 1403)
    assert s["engine_step_ms"] == pytest.approx(50000 / 1403)
    third = s["positions"][2]
    assert third["proposal_frequency"] == .5
    assert third["acceptance_when_proposed"] == pytest.approx(222 / 700)
    assert third["useful_accepted_per_round"] == pytest.approx(222 / 1400)
    assert report["variants"]["candidate"]["modes"]["sampled"]["decode_tps"] == pytest.approx(102.3)
    assert len(report["variants"]["candidate"]["cells"]) == 6


def test_clipping_preserves_raw_acceptance_and_retained_token_accounting():
    rows = matrix()
    for row in rows:
        engine = row["result"]["engine"]
        stats = engine["spec_positions"]
        # Final verify accepted 3 drafts; one accepted draft and bonus were cut.
        engine["accepted_tokens"] += 2
        stats["accepted"][1] += 1
        stats["accepted"][2] += 1
        stats["useful_accepted"][1] += 1
        stats.update(truncated_rounds=1, truncated_tokens=2, truncated_accepted_tokens=1)
        # One other round staged 3 but verified 2 drafts.
        stats["staged_depth_histogram"] = {"2": 199, "3": 201}
        stats.update(budget_clipped_rounds=1, budget_clipped_drafts=1)
    report = analyze(rows)
    assert report["valid"], report["errors"]
    stats = report["variants"]["candidate"]["modes"]["greedy"]
    assert stats["truncated_tokens"] == 6 and stats["truncated_accepted_tokens"] == 3
    assert stats["useful_accepted_per_round"] == pytest.approx(623 / 400)
    assert stats["positions"][2]["accepted"] == 303
    assert stats["positions"][2]["useful_accepted"] == 300


@pytest.mark.parametrize("tamper,reason", [
    ("duplicate", "duplicate-key-or-matrix-coordinate"),
    ("different-key-same-coordinate", "duplicate-key-or-matrix-coordinate"),
    ("missing-cell", "incomplete-six-cell-matrix"),
    ("unmatched-repeat", "unmatched-matrix-coordinates"),
    ("prompt-hash", "unmatched-prompt-hash"),
    ("missing-counter", "missing-or-invalid-engine-counters"),
    ("missing-positions", "missing-or-invalid-position-rounds"),
    ("missing-clipping", "missing-or-invalid-clipping-counters"),
    ("boolean-counter", "missing-or-invalid-engine-counters"),
    ("malformed-mode", "invalid-matrix-coordinates"),
    ("position-order", "invalid-position-count-order"),
    ("proposal-histogram", "histogram-position-mismatch"),
    ("wrong-rounds", "rounds-spec-step-mismatch"),
    ("wrong-output", "input-output-count-mismatch"),
    ("accounting", "emitted-token-accounting-mismatch"),
    ("zero-time", "missing-or-nonpositive-timing"),
    ("nan-time", "missing-or-nonpositive-timing"),
    ("warm-cache", "not-cold-request"),
    ("shared-server", "server-process-shared-across-arms-or-boots"),
])
def test_invalid_data_refuses_partial_rankings(tamper, reason):
    rows = matrix()
    row = rows[-1]
    engine = row["result"]["engine"]
    if tamper == "duplicate":
        rows.append(deepcopy(row))
    elif tamper == "different-key-same-coordinate":
        rows.append({**deepcopy(row), "key": "different-but-duplicate-coordinate"})
    elif tamper == "missing-cell":
        rows.pop()
    elif tamper == "unmatched-repeat":
        rows += matrix(variants=("candidate",), repeats=(1,))
    elif tamper == "prompt-hash":
        row["prompt"]["sha256"] = "b" * 64
    elif tamper == "missing-counter":
        engine.pop("spec_steps")
    elif tamper == "missing-positions":
        engine.pop("spec_positions")
    elif tamper == "missing-clipping":
        engine["spec_positions"].pop("truncated_tokens")
    elif tamper == "boolean-counter":
        engine["spec_steps"] = True
    elif tamper == "malformed-mode":
        row["extra"]["sampling_mode"] = []
    elif tamper == "position-order":
        engine["spec_positions"]["accepted"][2] = 300
    elif tamper == "proposal-histogram":
        engine["spec_positions"]["proposed"][2] -= 1
    elif tamper == "wrong-rounds":
        engine["spec_positions"]["rounds"] += 1
    elif tamper == "wrong-output":
        row["result"]["completion_tokens"] = 1023
    elif tamper == "accounting":
        engine["decode_steps"] += 1
    elif tamper in ("zero-time", "nan-time"):
        row["result"]["decode_wall_s"] = 0 if tamper == "zero-time" else float("nan")
    elif tamper == "warm-cache":
        row["result"]["cached_tokens"] = 64
    elif tamper == "shared-server":
        for r in rows:
            r["server_instance"] = "same-server"
    report = analyze(rows)
    assert not report["valid"]
    assert reason in json.dumps(report["errors"])
    assert report["variants"] == {} and report["greedy_pairs"] == []
    assert "No performance comparison" in render(report)


def test_greedy_pairs_use_actual_record_keys_and_divergence_is_not_quality_failure():
    rows = matrix()
    row = next(r for r in rows if r["variant"] == "candidate" and r["family"] == "code"
               and r["extra"]["sampling_mode"] == "greedy")
    row["key"] = "arbitrary-valid-captured-record-key"
    row["result"]["output_sha256"] = "b" * 64
    # Same output hash under sampled does not add an exact-match quality pair.
    report = analyze(rows)
    assert report["valid"]
    pairs = report["greedy_pairs"]
    assert len(pairs) == 3 and sum(p["status"] == "divergence" for p in pairs) == 1
    assert next(p for p in pairs if p["family"] == "code")["candidate_record"]["key"] == row["key"]
    assert "not a quality failure" in report["interpretation"]
    assert "One boot per arm" in render(report)


def test_independent_boots_and_geometry_ranges_remain_visible():
    rows = matrix(boots=("1", "2"))
    for row in rows:
        if row["variant"] == "candidate" and row["extra"]["boot_id"] == "2":
            row["geometry_after"]["moe_cache_size"] = 6800
    report = analyze(rows)
    assert report["valid"]
    data = report["variants"]["candidate"]
    assert data["independent_boot_count"] == 2 and not data["one_boot_only"]
    assert data["modes"]["greedy"]["geometry"]["after"]["moe_cache_size"] == {
        "min": 6800, "max": 6900, "available": 6, "total": 6}
    assert "6800–6900" in render(report)
    rows[-1]["server_instance"] = "restarted-under-same-label"
    assert not analyze(rows)["valid"]


def test_cli_preserves_raw_records_excludes_warmup_and_returns_failure(tmp_path):
    rows = [{"suite": "warmup", "key": "unmeasured", "result": {"error": "not-measured"}}] + matrix()
    path = tmp_path / "runs.jsonl"
    raw = "\n".join(json.dumps(row) for row in rows) + "\n"
    path.write_text(raw)
    assert main([str(tmp_path)]) == 0
    report = json.loads((tmp_path / "mtp-depth-report.json").read_text())
    assert report["excluded_nonmixed_records"] == 1 and report["measured_records"] == 12
    assert report["greedy_pairs"][0]["control_record"]["source"].endswith("runs.jsonl:2")
    assert (tmp_path / "mtp-depth-report.md").exists() and path.read_text() == raw
    rows.pop()
    path.write_text("\n".join(json.dumps(row) for row in rows))
    assert main([str(tmp_path)]) == 2
    assert not json.loads((tmp_path / "mtp-depth-report.json").read_text())["valid"]


def test_terminal_zero_position_has_no_fabricated_acceptance_rate():
    rows = matrix(variants=("control", "candidate"))
    for row in rows:
        e = row["result"]["engine"]
        e["spec_positions"]["proposed"] = [400, 400, 0]
        e["draft_tokens"] = 800
        e["spec_positions"]["accepted"] = [400, 222, 0]
        e["spec_positions"]["useful_accepted"] = [400, 222, 0]
        e["spec_positions"]["verified_depth_histogram"] = {"2": 400}
        e["spec_positions"]["staged_depth_histogram"] = {"2": 399, "3": 1}
        e["spec_positions"].update(budget_clipped_rounds=1, budget_clipped_drafts=1)
    report = analyze(rows, control="control")
    assert report["valid"], report["errors"]
    third = report["variants"]["candidate"]["modes"]["greedy"]["positions"][2]
    assert third["proposal_frequency"] == 0 and third["acceptance_when_proposed"] is None
