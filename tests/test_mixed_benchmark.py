"""Scoring cannot conceal a slow language, blend sampling modes or qualify partial runs."""
from copy import deepcopy
from itertools import product

import pytest

from benchmark.mixed import CONTEXTS, KINDS, SAMPLING, eligible, qualification, summarize, write_report


@pytest.mark.parametrize("template_offset", [-2, -1, 1, 2])
def test_prompt_bias_stays_correct_for_repeated_template_offset(template_offset):
    from types import SimpleNamespace
    from benchmark.benchmark import Runner

    # Exercise fitting and correction without loading a tokenizer or contacting a server.
    runner = Runner.__new__(Runner)
    runner.target_bias = {}
    runner.fitter = SimpleNamespace(build=lambda family, target, seed, nonce:
                                   SimpleNamespace(expected_prompt_tokens=target))
    for repeat in range(5):
        for target in (4096, 253952):
            prompt = runner._prompt("english", target, 12000 + repeat, f"repeat-{repeat}")
            actual = prompt.expected_prompt_tokens + template_offset
            runner._learn_bias(target, prompt.expected_prompt_tokens, actual)
            if repeat:
                assert actual == target

    # If a later boundary has no offset, the next prompt must recover too.
    for repeat in range(2):
        prompt = runner._prompt("code", 4096, repeat, f"zero-offset-{repeat}")
        actual = prompt.expected_prompt_tokens
        runner._learn_bias(4096, prompt.expected_prompt_tokens, actual)
    assert actual == 4096


def record(kind="code", context=4096, boot="1", repeat=0, seconds=20):
    return {
        "key": f"mixed/test/{boot}/greedy/{kind}/{context}/{repeat}",
        "suite": "mixed", "variant": "test", "family": kind,
        "server_instance": f"instance-{boot}",
        "target_input_tokens": context, "requested_output_tokens": 4096,
        "sampling": SAMPLING["greedy"],
        "extra": {"sampling_mode": "greedy", "boot_id": boot, "repeat": repeat},
        "result": {"completion_tokens": 4096, "prompt_tokens": context, "cached_tokens": 0,
                   "ttft_s": 1, "decode_wall_s": seconds, "finish_reason": "length",
                   "engine": {"decode_steps": 1365, "accepted_tokens": 2730,
                              "draft_tokens": 3000, "decode_s": seconds}},
    }


def test_throughput_uses_emitted_tokens_and_total_time():
    rows = [record(seconds=10), record(seconds=30)]
    result = summarize(rows)
    assert result["decode_tps"] == pytest.approx(8190 / 40)
    assert result["tokens_per_step"] == 3
    assert result["acceptance"] == pytest.approx(0.91)
    assert result["cpu_experts_per_layer_est"] is None


def test_full_qualification_requires_two_balanced_boots():
    rows = [record(kind, ctx, boot, repeat)
            for kind, ctx, boot, repeat in product(KINDS, CONTEXTS, ("1", "2"), range(5))]
    assert qualification(rows)
    assert not qualification(rows[:-1])
    one_boot = deepcopy(rows)
    for r in one_boot:
        r["extra"]["boot_id"] = "1"
    assert not qualification(one_boot)
    rows[0]["requested_output_tokens"] = 1024
    assert not qualification(rows)


@pytest.mark.parametrize("field,value", [("cached_tokens", 32), ("completion_tokens", 4095),
                                         ("completion_tokens", 4097), ("prompt_tokens", 4097),
                                         ("prompt_tokens", None), ("decode_wall_s", 0),
                                         ("decode_wall_s", float("inf")),
                                         ("decode_wall_s", float("nan"))])
def test_inaccurate_runs_excluded(field, value):
    r = record()
    assert eligible(r)
    r["result"][field] = value
    assert not eligible(r)


@pytest.mark.parametrize("tamper", ["duplicate_row", "duplicate_repeat", "missing_repeat",
                                  "wrong_sampling", "mixed_sampling", "mixed_variant"])
def test_qualification_rejects_duplicate_repeats_and_mixed_settings(tamper):
    rows = [record(kind, ctx, boot, repeat)
            for kind, ctx, boot, repeat in product(KINDS, CONTEXTS, ("1", "2"), range(5))]
    if tamper == "duplicate_row":
        rows[1] = deepcopy(rows[0])
    elif tamper == "duplicate_repeat":
        rows[1]["extra"]["repeat"] = 0
    elif tamper == "missing_repeat":
        rows[0]["extra"].pop("repeat")
    elif tamper == "wrong_sampling":
        rows[0]["sampling"] = SAMPLING["sampled"]
    elif tamper == "mixed_sampling":
        rows[0]["sampling"] = SAMPLING["sampled"]
        rows[0]["extra"]["sampling_mode"] = "sampled"
    else:
        rows[0]["variant"] = "other"
    assert not qualification(rows)


def test_report_keeps_sampling_separate_and_does_not_qualify_partial(tmp_path):
    import json
    greedy, sampled, wrong = record(), record(), record()
    sampled["sampling"] = SAMPLING["sampled"]
    sampled["extra"]["sampling_mode"] = "sampled"
    sampled["result"]["decode_wall_s"] = 10
    sampled["result"]["engine"]["decode_s"] = 10
    wrong["sampling"] = {"temperature": 0.5}
    (tmp_path / "runs.jsonl").write_text("\n".join(json.dumps(r) for r in (greedy, sampled, wrong)))
    write_report(str(tmp_path))
    report = (tmp_path / "mixed-report.md").read_text()
    assert "| test | greedy | 1 | 204.75 | no | unqualified |" in report
    assert "| test | sampled | 1 | 409.50 | no | unqualified |" in report
    assert "Excluded records: 1" in report


def test_moe_diagnostic_windows_preserve_units_and_missing_fields(tmp_path):
    from benchmark.mixed import read_moe_windows
    log = tmp_path / "server.log"
    prefix = "old request\n"
    log.write_text(prefix + "[moe-stats] cpu/layer=2.0 fetch_MB/step=50 layer_calls=100\n"
                   + "[moe-stats] cpu/layer=4.0 fetch_MB/step=80 layer_calls=200\n")
    r = record()
    r["moe_windows"] = read_moe_windows(str(log), len(prefix))
    assert summarize([r])["cpu_experts_per_layer_est"] == pytest.approx(10 / 3)
    assert summarize([r])["fetch_mib_per_step_est"] == 70


def test_legacy_csv_keeps_mixed_languages_and_sampling_separate(tmp_path):
    import csv
    from benchmark.report import write_csvs

    rows = []
    expected = {}
    for i, (kind, mode) in enumerate(product(("code", "japanese"), SAMPLING)):
        r = record(kind=kind, seconds=10 + i * 5)
        r["key"] += f"/{mode}"
        r["sampling"] = SAMPLING[mode]
        r["extra"]["sampling_mode"] = mode
        r["result"]["decode_tps"] = 4095 / r["result"]["decode_wall_s"]
        rows.append(r)
        expected[(kind, mode)] = round(r["result"]["decode_tps"], 3)
    # Legacy grouping stays unchanged even when its content family differs.
    for kind in ("code", "english"):
        r = record(kind=kind)
        r.update(suite="context", key=f"context/{kind}")
        r["result"]["decode_tps"] = 200
        rows.append(r)
    invalid = deepcopy(rows[0])
    invalid["key"] += "/invalid"
    invalid["result"]["completion_tokens"] += 1
    rows.append(invalid)
    write_csvs(rows, str(tmp_path))
    with (tmp_path / "summary_agg.csv").open() as f:
        aggregates = list(csv.DictReader(f))
    mixed = [r for r in aggregates if r["suite"] == "mixed"]
    assert len(mixed) == 4
    for row in mixed:
        assert row["n"] == "1"
        assert float(row["decode_tps_median"]) == expected[(row["family"], row["sampling_mode"])]
    legacy = [r for r in aggregates if r["suite"] == "context"]
    assert len(legacy) == 1 and legacy[0]["n"] == "2"
    assert legacy[0]["family"] == legacy[0]["sampling_mode"] == ""
    with (tmp_path / "summary.csv").open() as f:
        flat = {r["key"]: r for r in csv.DictReader(f)}
    for r in rows[:4]:
        actual = flat[r["key"]]
        assert actual["sampling_mode"] == r["extra"]["sampling_mode"]
        for field in ("temperature", "top_k", "top_p"):
            assert float(actual[field]) == r["sampling"][field]
    assert flat[invalid["key"]]["valid"] == "False"


def test_legacy_report_entrypoint_writes_and_links_mixed_report(tmp_path, monkeypatch, capsys):
    import json
    from benchmark.report import main

    (tmp_path / "runs.jsonl").write_text(json.dumps(record()) + "\n")
    monkeypatch.setattr("sys.argv", ["benchmark.report", str(tmp_path)])
    main()
    assert (tmp_path / "mixed-report.md").exists()
    report = (tmp_path / "report.md").read_text()
    assert "[mixed decode report](mixed-report.md)" in report
    assert "no legacy headline is reported" in report
    assert "HEADLINE:" not in capsys.readouterr().out
