"""Paired comparisons preserve the matched sample and expose incomplete evidence."""
from copy import deepcopy
import hashlib
from itertools import product
import json

import pytest

from benchmark.compare_decode import analyze, main, render
from benchmark.mixed import CONTEXTS, KINDS, SAMPLING


def record(variant="baseline", boot="1", mode="greedy", family="code", context=4096,
           repeat=0, seconds=20, output=4096):
    identity = f"{boot}/{mode}/{family}/{context}/{repeat}"
    steps = max(1, (output - 1) // 3)
    return {
        "key": f"mixed/{variant}/{identity}", "suite": "mixed", "variant": variant,
        "family": family, "target_input_tokens": context, "requested_output_tokens": output,
        "server_instance": f"instance-{variant}-{boot}", "sampling": SAMPLING[mode],
        "extra": {"boot_id": boot, "sampling_mode": mode, "repeat": repeat, "mode": "cold"},
        "prompt": {"sha256": hashlib.sha256(identity.encode()).hexdigest()},
        "result": {"prompt_tokens": context, "completion_tokens": output, "cached_tokens": 0,
                   "ttft_s": 1, "decode_wall_s": seconds, "finish_reason": "length",
                   "engine": {"decode_s": seconds, "decode_steps": steps,
                              "accepted_tokens": output - 1 - steps, "draft_tokens": output - 1}},
        "geometry_before": {"num_pages": 1024, "page_size": 64, "moe_cache_size": 7900},
        "geometry_after": {"num_pages": 1024, "page_size": 64, "moe_cache_size": 7900},
    }


def test_only_matched_repetitions_enter_weighted_rate_and_modes_stay_separate():
    rows = [record(boot="1", seconds=10), record(boot="2", seconds=30),
            record("combined", boot="1", seconds=20), record("combined", boot="2", seconds=15)]
    # These fast candidate-only repetitions must never inflate the comparison.
    rows += [record("combined", boot=boot, repeat=repeat, seconds=.01)
             for boot, repeat in product(("1", "2"), range(1, 5))]
    rows += [record(mode="sampled", seconds=20), record("combined", mode="sampled", seconds=10)]
    report = analyze(rows)
    greedy, sampled = report["modes"]["greedy"], report["modes"]["sampled"]
    assert greedy["pairs"] == 2
    assert greedy["baseline"]["decode_tps"] == pytest.approx(8190 / 40)
    assert greedy["candidate"]["decode_tps"] == pytest.approx(8190 / 35)
    assert greedy["decode_ratio"] == pytest.approx(40 / 35)
    assert greedy["unmatched_records"] == 8
    assert sampled["pairs"] == 1 and sampled["decode_ratio"] == 2
    cell = greedy["cells"][0]
    assert cell["follow_up"] and cell["slower_boots"] == ["1"]
    assert not cell["slower_across_all_independent_boots"]
    assert greedy["baseline"]["accepted_tokens_per_step"] == 2
    assert greedy["baseline"]["emitted_tokens_per_step"] == 3
    assert greedy["baseline"]["engine_step_ms"] == pytest.approx(40000 / 2730)


def test_candidate_full_matrix_does_not_qualify_smaller_matched_subset():
    rows = [record("combined", boot=boot, family=family, context=context, repeat=repeat)
            for boot, family, context, repeat in product(("1", "2"), KINDS, CONTEXTS, range(5))]
    rows += [record(boot=boot, family=family, context=context)
             for boot, family, context in product(("1", "2"), KINDS, CONTEXTS)]
    data = analyze(rows)["modes"]["greedy"]
    assert data["pairs"] == 24 and data["minimum_pairs_per_cell"] == 2
    assert data["eligible_records"] == {"baseline": 24, "combined": 120}
    assert data["full_qualification_of_all_eligible_records"] == {"baseline": False, "combined": True}
    assert data["all_canonical_cells_matched"]
    assert data["unmatched_records"] == 96


@pytest.mark.parametrize("tamper,reason", [
    ("different_hash", "prompt-sha256-mismatch"),
    ("missing_hash", "missing-or-invalid-prompt-sha256"),
    ("invalid_hash_type", "missing-or-invalid-prompt-sha256"),
    ("different_output", "output-budget-mismatch"),
    ("duplicate", "duplicate-record-key-or-pair-coordinates"),
    ("invalid", "ineligible-mixed-record"),
    ("warm", "not-cold-mode"),
    ("missing_instance", "missing-or-shared-server-instance"),
    ("same_instance", "missing-or-shared-server-instance"),
])
def test_unusable_pairs_are_reported_without_partially_comparing(tamper, reason):
    a, b = record(), record("combined")
    rows = [a, b]
    if tamper == "different_hash":
        b["prompt"]["sha256"] = "0" * 64
    elif tamper == "missing_hash":
        b.pop("prompt")
    elif tamper == "invalid_hash_type":
        b["prompt"]["sha256"] = 123
    elif tamper == "different_output":
        rows[1] = record("combined", output=1024)
    elif tamper == "duplicate":
        rows.append(deepcopy(b))
    elif tamper == "invalid":
        b["result"]["cached_tokens"] = 64
    elif tamper == "warm":
        b["extra"]["mode"] = "warm"
    elif tamper == "missing_instance":
        b.pop("server_instance")
    else:
        b["server_instance"] = a["server_instance"]
    report = analyze(rows)
    assert report["matched_pairs"] == 0
    assert reason in json.dumps(report["invalid_records"] + report["unmatched_records"])
    assert report["modes"]["greedy"]["baseline"]["decode_tps"] is None
    assert report["modes"]["greedy"]["candidate"]["decode_tps"] is None


def test_consistent_negative_boots_only_flag_follow_up_not_statistical_proof():
    rows = [record(variant, boot=boot, seconds=20 if variant == "baseline" else 22)
            for variant, boot in product(("baseline", "combined"), ("1", "2"))]
    report = analyze(rows)
    cell = report["modes"]["greedy"]["cells"][0]
    assert cell["independent_boots"] and cell["slower_across_all_independent_boots"]
    assert cell["follow_up"]
    assert "no significance or no-regression claim" in render(report)
    # Reusing physical processes under different labels removes independent-boot evidence.
    for row in rows:
        row["server_instance"] = f"instance-{row['variant']}"
    cell = analyze(rows)["modes"]["greedy"]["cells"][0]
    assert cell["follow_up"] and not cell["independent_boots"]
    assert not cell["slower_across_all_independent_boots"]


def test_boot_process_identity_is_consistent_across_cells():
    rows = [record(variant, boot=boot, family=family, seconds=20 if variant == "baseline" else 22)
            for variant, boot, family in product(("baseline", "combined"), ("1", "2"), ("code", "english"))]
    assert all(c["independent_boots"] for c in analyze(rows)["modes"]["greedy"]["cells"])
    rows[1]["server_instance"] = "unexpected-restarted-process"
    data = analyze(rows)["modes"]["greedy"]
    assert len(data["boot_instances"]["baseline"]["1"]) == 2
    assert all(not c["independent_boots"] for c in data["cells"])


def test_geometry_differences_and_missing_counters_are_explicit():
    a, b = record(), record("combined")
    b["geometry_before"]["moe_cache_size"] = 8000
    b["geometry_after"] = {"num_pages": 4096, "page_size": 64, "moe_cache_size": 7900}
    b["result"].pop("engine")
    report = analyze([a, b])
    geometry = report["pair_details"][0]["geometry"]
    assert {d["field"] for d in geometry["differences"]} == {"before.moe_cache_size", "after.num_pages"}
    assert "candidate" in geometry["within_request_changes"]
    assert report["modes"]["greedy"]["candidate"]["accepted_tokens_per_step"] is None
    a.pop("geometry_before")
    report = analyze([a, b])
    assert report["pair_details"][0]["geometry"]["unavailable_phases"] == ["before"]
    assert "Geometry unavailable for 1 pairs" in render(report)
    for missing in ({"error": "connection timed out"}, {"num_pages": None, "page_size": None}):
        a["geometry_before"] = b["geometry_before"] = missing
        geometry = analyze([a, b])["pair_details"][0]["geometry"]
        assert geometry["unavailable_phases"] == ["before"]
        assert geometry["within_request_changes"] == {}


def test_cli_writes_machine_readable_results_without_mutating_raw_rows(tmp_path, monkeypatch):
    raw = "\n".join(json.dumps(r) for r in (record(), record("combined", seconds=25))) + "\n"
    path = tmp_path / "runs.jsonl"
    path.write_text(raw)
    monkeypatch.setattr("sys.argv", ["benchmark.compare_decode", str(tmp_path)])
    main()  # A slower candidate is reported; it does not raise SystemExit or fail a gate.
    report = json.loads((tmp_path / "decode-comparison.json").read_text())
    assert report["matched_pairs"] == 1
    assert report["pair_details"][0]["baseline_record"]["source"].endswith("runs.jsonl:1")
    assert (tmp_path / "decode-comparison.md").exists()
    assert path.read_text() == raw


def test_partial_geometry_does_not_turn_new_telemetry_into_a_change():
    a, b = record(), record("combined")
    a["geometry_before"]["moe_cache_size"] = None
    b["geometry_before"].pop("moe_cache_size")
    geometry = analyze([a, b])["pair_details"][0]["geometry"]
    assert geometry["unavailable_phases"] == []
    assert geometry["unavailable_fields"] == [{"field": "before.moe_cache_size", "sides": ["baseline", "candidate"]}]
    assert geometry["within_request_changes"] == {}
    assert geometry["differences"] == []
    b["geometry_before"]["moe_cache_size"] = 7900
    geometry = analyze([a, b])["pair_details"][0]["geometry"]
    assert geometry["unavailable_fields"] == [{"field": "before.moe_cache_size", "sides": ["baseline"]}]
    assert geometry["differences"] == []
