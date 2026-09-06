"""Complete output capture is opt-in and stays outside timing/JSONL payloads."""
from copy import deepcopy
import hashlib
import io
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmark.benchmark import Runner, stream_chat


def stream_result(monkeypatch, capture_output=False):
    content = [f"第{i}章: " + "complete output beyond the preview. " * 4 for i in range(9)]
    choices = [{"index": 0, "delta": {"role": "assistant"}},
               {"index": 0, "delta": {"reasoning_content": "検討します。"}}]
    choices += [{"index": 0, "delta": {"content": piece}} for piece in content]
    choices += [{"index": 0, "delta": {"content": "おわり", "reasoning_content": "同時の推論"}},
                {"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{}"}}]}},
                {"index": 0, "delta": {}, "finish_reason": "length"}]
    events = [{"choices": [choice]} for choice in choices]
    events.append({"choices": [], "usage": {"prompt_tokens": 4096, "completion_tokens": 4096}})
    body = "".join("data: " + json.dumps(event, ensure_ascii=False) + "\n\n" for event in events)
    body += "data: [DONE]\n\n"
    sent = []

    def urlopen(request, **kwargs):
        sent.append(json.loads(request.data))
        return io.BytesIO(body.encode())

    clock = itertools.count(100)
    monkeypatch.setattr("benchmark.benchmark.urllib.request.urlopen", urlopen)
    monkeypatch.setattr("benchmark.benchmark.time.monotonic", lambda: next(clock))
    monkeypatch.setattr("benchmark.benchmark.time.time", lambda: 10000)
    result = stream_chat("http://unused.invalid/v1", "test", [], 4096, capture_output=capture_output)
    assert len(sent) == 1 and "capture_output" not in sent[0]
    text = "検討します。" + "".join(content) + "おわり"
    return result, text, choices


def make_runner(tmp_path):
    runner = Runner.__new__(Runner)
    runner.out = str(tmp_path)
    runner.runs_path = str(tmp_path / "runs.jsonl")
    runner.done = set()
    return runner


def test_capture_preserves_full_channels_and_deltas_without_changing_metrics(monkeypatch):
    ordinary, text, choices = stream_result(monkeypatch)
    captured, _, _ = stream_result(monkeypatch, True)
    output = captured.pop("_captured_output")
    assert captured == ordinary
    assert len(ordinary["output_head"]) == 160 < len(text)
    assert output["text"] == text
    assert output["content"] == text.removeprefix("検討します。")
    assert output["reasoning_content"] == "検討します。同時の推論"
    assert output["deltas"] == choices
    assert captured["output_sha256"] == hashlib.sha256(text.encode()).hexdigest()
    assert captured["output_chars"] == len(text)


def test_append_stores_output_artifact_and_keeps_full_text_out_of_jsonl(tmp_path, monkeypatch):
    runner = make_runner(tmp_path)
    result, text, _ = stream_result(monkeypatch, True)
    row = {"key": "mixed/diagnostic/1/greedy/japanese/4096/0", "result": result}
    runner._append(row)
    saved = json.loads((tmp_path / "runs.jsonl").read_text())
    assert "_captured_output" not in saved["result"]
    assert "text" not in saved["result"]
    artifact_path = Path(saved["result"]["output_artifact"])
    assert not artifact_path.is_absolute() and artifact_path.parent == Path("outputs")
    output = json.loads((tmp_path / artifact_path).read_text())
    assert output["record_key"] == row["key"]
    assert output["text"] == text
    assert output["output_sha256"] == saved["result"]["output_sha256"]
    assert row["key"] in runner.done


def test_capture_disabled_creates_no_artifacts(tmp_path, monkeypatch):
    runner = make_runner(tmp_path)
    result, _, _ = stream_result(monkeypatch)
    runner._append({"key": "ordinary", "result": result})
    assert not (tmp_path / "outputs").exists()
    saved = json.loads((tmp_path / "runs.jsonl").read_text())
    assert "output_artifact" not in saved["result"]
    assert "_captured_output" not in saved["result"]


def test_concurrent_requests_get_separate_artifacts_and_preserve_previous_files(tmp_path, monkeypatch):
    runner = make_runner(tmp_path)
    result, text, _ = stream_result(monkeypatch, True)
    record = {"key": "concurrency/diagnostic/C2", "requests": [deepcopy(result), deepcopy(result), None, {"error": "failed"}]}
    runner._append(record)
    references = [r["output_artifact"] for r in record["requests"][:2]]
    assert len(set(references)) == 2
    previous = {path: (tmp_path / path).read_bytes() for path in references}
    for index, path in enumerate(references):
        output = json.loads(previous[path])
        assert output["record_key"] == f"concurrency/diagnostic/C2/request/{index}"
        assert output["text"] == text
    # An orphaned earlier capture with this key must survive a fresh attempt.
    runner._store_output("concurrency/diagnostic/C2/request/0", deepcopy(result))
    assert len(list((tmp_path / "outputs").glob("*.json"))) == 3
    assert all((tmp_path / path).read_bytes() == data for path, data in previous.items())


@pytest.mark.parametrize("capture", [False, True])
def test_measure_forwards_capture_option_and_persists_only_when_enabled(tmp_path, monkeypatch, capture):
    runner = make_runner(tmp_path)
    runner.base, runner.model, runner.variant, runner.server = "unused", "test", "diagnostic", "freetoken"
    runner.server_instance, runner.capture_output, runner.target_bias = "test-instance", capture, {}
    prompt = SimpleNamespace(messages=lambda: [], record=lambda: {}, expected_prompt_tokens=4096)
    runner._prompt = lambda *args, **kwargs: prompt
    result, _, _ = stream_result(monkeypatch, capture)

    def measured_stream(*args, **kwargs):
        assert kwargs["capture_output"] is capture
        return result

    monkeypatch.setattr("benchmark.benchmark.stream_chat", measured_stream)
    monkeypatch.setattr("benchmark.benchmark.geometry", lambda base: {})
    row = runner.measure("mixed/diagnostic/test", "japanese", 4096, 4096, 1, telemetry=False)
    assert ("output_artifact" in row["result"]) is capture
