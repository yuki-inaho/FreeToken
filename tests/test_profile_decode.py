"""Trace duration sums and interval unions answer different questions."""
import gzip
import json

import pytest

from benchmark.profile_decode import analyze, main, render


def event(name, ts, duration, category="kernel", device=0, stream=7, **fields):
    return {"ph": "X", "cat": category, "name": name, "ts": ts, "dur": duration,
            "pid": 123, "tid": 123, "args": {"device": device, "stream": stream}, **fields}


def test_all_stream_work_is_counted_but_overlap_is_not_wall_time():
    rows = [event("_gemv_rows_splitk_kernel", 0, 10),
            event("fast_index_copy_multi", 5, 10, stream=9),
            event("_decode_nvfp4_marlin_kernel", 20, 5),
            event("Memcpy HtoD", 24, 3, "gpu_memcpy", stream=9),
            event("Memset", 28, 2, "gpu_memset")]
    d = analyze(rows, label="synthetic", steps=2)["devices"]["0"]
    assert d["duration_sum_us"] == 30
    assert d["events"] == 5 and set(d["streams"]) == {"7", "9"}
    assert d["duration_sum_us_per_step"] == 15
    assert d["timeline"]["span_us"] == 30
    assert d["timeline"]["active_us"] == 24
    assert d["timeline"]["inactive_us"] == 6
    assert d["timeline"]["cross_bucket_overlap_us"] == 6
    assert sum(row["share_of_gpu_duration_sum"] for row in d["buckets"].values()) == pytest.approx(1)
    assert d["buckets"]["transfer"]["only_this_bucket_active_us"] == 7


def test_same_bucket_overlap_and_different_devices_are_separate():
    rows = [event("_gemv_rows_splitk_kernel", 0, 10),
            event("_gemv_rows_splitk_kernel", 5, 10, stream=8),
            event("qsa_attention", 0, 20, device=1)]
    report = analyze(rows, label="synthetic")
    d = report["devices"]["0"]
    assert d["duration_sum_us"] == 20 and d["timeline"]["active_us"] == 15
    assert d["timeline"]["simultaneous_events_us"] == 5
    assert d["timeline"]["cross_bucket_overlap_us"] == 0
    assert d["buckets"]["dense"]["only_this_bucket_active_us"] == 15
    assert report["devices"]["1"]["timeline"]["active_us"] == 20
    assert d["duration_sum_us_per_step"] is None


def test_nested_host_apis_are_unioned_not_added_to_gpu_time():
    rows = [event("qsa_attention", 0, 30),
            event("cudaGraphLaunch", 0, 10, "cuda_runtime"),
            event("cuLaunchKernel", 2, 5, "cuda_driver"),
            event("cudaEventSynchronize", 20, 15, "cuda_runtime"),
            event("aten::huge_inclusive_parent", 0, 9999, "cpu_op")]
    report = analyze(rows, label="synthetic")
    t = report["host_cuda_api"]["per_thread"]["123:123"]
    assert t["duration_sum_us"] == 30
    assert t["timeline"]["active_us"] == 25
    assert t["timeline"]["bucket_active_us"] == {"launch": 10, "synchronize_or_wait": 15}
    assert report["devices"]["0"]["duration_sum_us"] == 30
    assert "not exposed host overhead" in render(report)


def test_invalid_gpu_events_and_unknown_device_are_visible():
    rows = [event("unknown", 0, 1, args={}), event("invalid", 0, float("nan")),
            event("invalid", 0, -3), event("flow", 0, 5, ph="s")]
    report = analyze(rows, label="synthetic")
    assert report["invalid_gpu_complete_events"] == 2
    assert report["devices"]["unknown"]["events"] == 1
    with pytest.raises(ValueError, match="positive"):
        analyze(rows, label="bad", steps=0)
    with pytest.raises(ValueError, match="no valid"):
        analyze([], label="empty")


def test_cli_gzip_preserves_source_identity(tmp_path, monkeypatch, capsys):
    raw = gzip.compress(json.dumps({"traceEvents": [event("qsa_attention", 10, 2)]}).encode())
    path = tmp_path / "trace.json.gz"
    path.write_bytes(raw)
    prefix = tmp_path / "historical-validation"
    monkeypatch.setattr("sys.argv", ["profile_decode", str(path), "--label", "historical-validation",
                                     "--steps", "6", "--out-prefix", str(prefix)])
    main()
    d = json.loads(prefix.with_suffix(".json").read_text())
    assert d["source_bytes"] == len(raw) and len(d["trace_sha256"]) == 64
    assert d["label"] == "historical-validation"
    assert "not an unprofiled throughput result" in capsys.readouterr().out
    assert path.read_bytes() == raw
