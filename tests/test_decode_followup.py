"""The follow-up plan validates references before any server or output activity."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import socket

import pytest


REPO = Path(__file__).resolve().parents[1]


def reference_rows():
    rows = []
    for boot in ("1", "2"):
        for repeat in range(5):
            nonce = f"mixed-{boot}-greedy-japanese-131072-{repeat}"
            rows.append({
                "key": f"mixed/combined/{boot}/greedy/japanese/131072/{repeat}",
                "suite": "mixed", "variant": "combined", "family": "japanese",
                "target_input_tokens": 131072, "requested_output_tokens": 4096,
                "extra": {"boot_id": boot, "sampling_mode": "greedy", "repeat": repeat, "mode": "cold"},
                "sampling": {"temperature": 0.0, "top_k": 1, "top_p": 1.0},
                "server_instance": f"combined-instance-{boot}",
                "result": {"prompt_tokens": 131072, "completion_tokens": 4096, "cached_tokens": 0,
                           "ttft_s": 38, "decode_wall_s": 20, "finish_reason": "length"},
                "prompt": {"nonce": nonce, "sha256": hashlib.sha256(nonce.encode()).hexdigest(),
                           "target_tokens": 131072, "expected_prompt_tokens": 131072, "overhead_tokens": 12},
            })
    return rows


def run_plan(tmp_path, rows, cells="greedy:japanese:131072", *, mode="baseline", warmup="1", variant=""):
    reference = tmp_path / "reference"
    reference.mkdir(exist_ok=True)
    (reference / "runs.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    source = tmp_path / "old-source"
    (source / "freetoken").mkdir(parents=True, exist_ok=True)
    current = tmp_path / "current-source"
    (current / "freetoken").mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["bash", "benchmark/run_decode_followup.sh", "--plan-only", str(tmp_path / "output"),
         str(reference), cells], cwd=REPO, capture_output=True, text=True,
        env={**os.environ, "FREETOKEN_VENV": str(Path(sys.executable).parent.parent),
             "FREETOKEN_DECODE_BASELINE_PYTHON": str(source),
             "FREETOKEN_DECODE_PYTHONPATH": str(current),
             "FREETOKEN_FOLLOWUP_MODE": mode, "FREETOKEN_FOLLOWUP_WARMUP": warmup,
             "FREETOKEN_FOLLOWUP_VARIANT": variant, "FREETOKEN_PORT": "1920"},
        timeout=10,
    )
    return result


def test_plan_preserves_selected_hashes_without_creating_output(tmp_path):
    rows = reference_rows()
    result = run_plan(tmp_path, rows)
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["variant"] == "baseline-followup"
    assert plan["control"] == "baseline"
    assert set(plan["runtime_controls"].values()) == {"0"}
    assert plan["runtime_source"] == plan["baseline_source"] == str(tmp_path / "old-source")
    assert plan["warmup_first_cell"] is True
    assert plan["boots"] == ["1", "2"]
    assert plan["runs_per_cell_per_boot"] == 5
    assert plan["output_tokens"] == 4096
    assert len(plan["selected_reference_records"]) == 10
    assert [r["prompt_sha256"] for r in plan["selected_reference_records"]] == [r["prompt"]["sha256"] for r in rows]
    assert "conditional descriptive evidence" in " ".join(plan["limitations"])
    assert not (tmp_path / "output").exists()


def test_execution_only_selects_current_source_and_disables_only_admission(tmp_path):
    result = run_plan(tmp_path, reference_rows(), mode="execution-only")
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["variant"] == "execution-only-followup"
    assert plan["runtime_source"] == str(tmp_path / "current-source")
    assert plan["baseline_source"] is None
    assert plan["runtime_controls"] == {
        "FREETOKEN_HYBRID_ADMISSION_CREDIT": "0", "FREETOKEN_HYBRID_SHARED_OVERLAP": "1",
        "FREETOKEN_GDN_STRIDED_NORM": "1", "FREETOKEN_QSA_PARALLEL_HEADS": "1", "FREETOKEN_QSA_DIRECT_PERMUTE": "1",
    }
    assert "otherwise match the combined runtime" in " ".join(plan["limitations"])
    assert plan["capture_output"] is True
    assert not (tmp_path / "output").exists()


def test_optional_no_warmup_and_custom_variant_are_recorded(tmp_path):
    result = run_plan(tmp_path, reference_rows(), mode="execution-only", warmup="0", variant="admission-control")
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["warmup_first_cell"] is False and plan["variant"] == "admission-control"
    assert "startup calibration" in " ".join(plan["limitations"])


@pytest.mark.parametrize("options,reason", [
    ({"mode": "unknown"}, "Unsupported follow-up mode"),
    ({"warmup": "2"}, "FREETOKEN_FOLLOWUP_WARMUP must be 0 or 1"),
])
def test_invalid_controls_fail_during_plan(tmp_path, options, reason):
    result = run_plan(tmp_path, reference_rows(), **options)
    assert result.returncode != 0 and reason in result.stderr
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("tamper,reason", [
    ("missing", "Need one completed combined reference"),
    ("duplicate", "Need one completed combined reference"),
    ("fit_bias", "token-fit correction requiring manual replay"),
    ("nonce", "token-fit correction requiring manual replay"),
    ("same_instance", "reuse an actual server instance"),
    ("duplicate_cell", "Duplicate selected cells"),
])
def test_plan_rejects_unreproducible_or_duplicate_reference_cells(tmp_path, tamper, reason):
    rows = reference_rows()
    cells = "greedy:japanese:131072"
    if tamper == "missing":
        rows.pop()
    elif tamper == "duplicate":
        rows.append(rows[0])
    elif tamper == "fit_bias":
        rows[0]["prompt"]["target_tokens"] -= 1
    elif tamper == "nonce":
        rows[0]["prompt"]["nonce"] = "other-prompt-prefix"
    elif tamper == "same_instance":
        for row in rows:
            row["server_instance"] = "same-process"
    else:
        cells += "," + cells
    result = run_plan(tmp_path, rows, cells)
    assert result.returncode != 0
    assert reason in result.stderr
    assert not (tmp_path / "output").exists()


def test_plan_never_overwrites_existing_output(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("existing results")
    result = run_plan(tmp_path, reference_rows())
    assert result.returncode != 0 and "OUTPUT_DIR already exists" in result.stderr
    assert marker.read_text() == "existing results"


def test_port_reuse_probe_still_rejects_an_existing_listener():
    script = (REPO / "benchmark/run_decode_followup.sh").read_text()
    probe = script.split('assert_port_free() {\n  "$PY" - "$PORT" <<\'PY\'\n', 1)[1].split("\nPY\n", 1)[0]
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = str(listener.getsockname()[1])
        result = subprocess.run([sys.executable, "-c", probe, port], capture_output=True, text=True, timeout=5)
        assert result.returncode != 0 and "occupied or unavailable" in result.stderr
    result = subprocess.run([sys.executable, "-c", probe, port], capture_output=True, text=True, timeout=5)
    assert result.returncode == 0
