"""CPU-only whole-trace accounting for PyTorch Chrome traces.

Example: python -m benchmark.profile_decode /tmp/profile/spec-trace-123.json \
    --label combined-japanese-253952 --steps 6 --out-prefix /tmp/profile/report

Steps must come from the capture configuration/log, not inferred layer counts.
The capture counter spans requests: use --no-warmup for a single target request
and confirm the profiler window belongs to it in the server log.
GPU duration sums measure recorded work; overlapping work is counted repeatedly.
Union intervals describe activity, not utilization, occupancy or a critical path.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path

GPU_CATEGORIES = {"kernel", "gpu_memcpy", "gpu_memset"}
API_CATEGORIES = {"cuda_runtime", "cuda_driver"}
BUCKETS = ("dense", "expert", "transfer", "attention", "other")
LIMITATIONS = [
    "Kernel buckets are name-based attribution; the complete per-name inventory is retained.",
    "Dense includes named linear/quantization kernels; expert includes routed expert compute, routing "
    "and finalize/control kernels; attention includes named QSA and GDN activation/state kernels.",
    "Transfer includes GPU memcpy events and expert-gather kernels. Gather duration does not "
    "establish PCIe traffic or bandwidth; tensor source memory and bytes are not encoded in its name.",
    "Summed GPU duration double-counts concurrent activity. Shares use that work sum, not wall time.",
    "Interval unions only describe recorded activity. GPU-active time is not utilization, occupancy "
    "or a critical path; bucket-only activity is not time guaranteed recoverable by an optimization.",
    "Gaps can include dependencies, CPU work, unrecorded stream wait-value memops and profiling effects. "
    "Do not label gaps CPU stalls or attribute them to a particular route without dependency evidence.",
    "CUDA API durations are host call time, including possible GPU backpressure. Nested driver/runtime "
    "calls and different threads can overlap; CPU operator inclusive times are intentionally not summed.",
    "The profiler starts at a verify-step boundary without an initial device synchronization and drains "
    "at the end. Supplied step normalization is approximate at capture boundaries; final synchronization "
    "and profiling overhead are not normal decode throughput.",
    "The FREETOKEN_SPEC_PROFILE step counter spans requests since executor creation. Benchmark warmup "
    "can consume the window; confirm capture start/end belong to the intended request in its server log.",
    "No request throughput, accepted tokens per step or fresh-performance claim is inferred from a trace. "
    "Pair a new trace with its configuration, source manifest, request counters and unprofiled timing.",
]


def finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def valid_duration(event: dict) -> bool:
    return (event.get("ph") == "X" and finite(event.get("ts"))
            and finite(event.get("dur")) and event["dur"] >= 0)


def bucket(event: dict) -> str:
    name = event.get("name", "").lower()
    if event.get("cat") == "gpu_memcpy" or "fast_index_copy" in name or "indexcopy" in name:
        return "transfer"
    if any(part in name for part in ("_decode_nvfp4", "_moe_", "ensure_experts_hybrid", "_router_triton")):
        return "expert"
    if any(part in name for part in ("_gemv_rows", "_splitk_rows", "_gemv_splitk", "_splitk_reduce",
                                     "_dynamic_quant", "_nvfp4_gemv", "_nvfp4_gemm", "scaled_mm")):
        return "dense"
    if any(part in name for part in ("qsa", "qk_norm_rope", "fused_sigmoid_gating_delta",
                                     "small_conv", "causal_conv", "rms_norm_gated",
                                     "layer_norm_fwd_1pass", "chunk_gated_delta", "fused_recurrent_gated_delta")):
        return "attention"
    return "other"


def intervals(events: list[dict], group) -> dict:
    """Sweep intervals once; concurrent events of the same bucket remain bucket-only."""
    positive = [e for e in events if e["dur"] > 0]
    if not positive:
        return {"span_us": 0.0, "active_us": 0.0, "inactive_us": 0.0,
                "simultaneous_events_us": 0.0, "cross_bucket_overlap_us": 0.0,
                "bucket_active_us": {}, "bucket_only_us": {}}
    origin = min(e["ts"] for e in positive)
    changes = defaultdict(Counter)
    for event in positive:
        start = event["ts"] - origin
        key = group(event)
        changes[start][key] += 1
        changes[start + event["dur"]][key] -= 1
    active = Counter()
    active_time = overlap = concurrent = 0.0
    bucket_active, bucket_only = Counter(), Counter()
    last = 0.0
    for timestamp, delta in sorted(changes.items()):
        duration = timestamp - last
        keys = [key for key, count in active.items() if count > 0]
        if keys:
            active_time += duration
            for key in keys:
                bucket_active[key] += duration
            if len(keys) == 1:
                bucket_only[keys[0]] += duration
            else:
                overlap += duration
            if sum(active.values()) > 1:
                concurrent += duration
        active.update(delta)
        last = timestamp
    return {"span_us": last, "active_us": active_time, "inactive_us": max(0.0, last - active_time),
            "simultaneous_events_us": concurrent, "cross_bucket_overlap_us": overlap,
            "bucket_active_us": dict(bucket_active), "bucket_only_us": dict(bucket_only)}


def event_totals(events: list[dict], steps: int | None) -> dict:
    work = sum(e["dur"] for e in events)
    return {"events": len(events), "duration_sum_us": work,
            "duration_mean_us": work / len(events) if events else None,
            "events_per_step": len(events) / steps if steps else None,
            "duration_sum_us_per_step": work / steps if steps else None}


def api_group(event: dict) -> str:
    name = event.get("name", "").lower()
    if "launch" in name:
        return "launch"
    if "synchronize" in name or "wait" in name:
        return "synchronize_or_wait"
    return "other_api"


def analyze(data, *, label: str, steps: int | None = None) -> dict:
    if steps is not None and (type(steps) is not int or steps < 1):
        raise ValueError("steps must be a positive integer")
    events = data.get("traceEvents", []) if isinstance(data, dict) else data
    if not isinstance(events, list):
        raise ValueError("traceEvents must be a list")
    events = [e for e in events if isinstance(e, dict)]
    gpu = [e for e in events if e.get("cat") in GPU_CATEGORIES and valid_duration(e)]
    if not gpu:
        raise ValueError("trace contains no valid kernel/gpu_memcpy/gpu_memset complete events")
    devices = defaultdict(list)
    for event in gpu:
        # A missing device remains visible, rather than silently merging into device 0.
        devices[str((event.get("args") or {}).get("device", "unknown"))].append(event)
    reports = {}
    for device, recorded in sorted(devices.items()):
        timeline = intervals(recorded, bucket)
        work = sum(e["dur"] for e in recorded)
        buckets = {}
        for name in BUCKETS:
            selected = [e for e in recorded if bucket(e) == name]
            buckets[name] = {**event_totals(selected, steps),
                             "share_of_gpu_duration_sum": sum(e["dur"] for e in selected) / work if work else None,
                             "active_union_us": timeline["bucket_active_us"].get(name, 0.0),
                             "only_this_bucket_active_us": timeline["bucket_only_us"].get(name, 0.0)}
        streams, names = defaultdict(list), defaultdict(list)
        for event in recorded:
            streams[str((event.get("args") or {}).get("stream", "unknown"))].append(event)
            names[(event.get("cat"), event.get("name", "<unnamed>"), bucket(event))].append(event)
        inventory = [{"category": key[0], "name": key[1], "bucket": key[2], **event_totals(value, steps)}
                     for key, value in names.items()]
        reports[device] = {
            **event_totals(recorded, steps), "timeline": timeline, "buckets": buckets,
            "streams": {stream: {**event_totals(value, steps), "timeline": intervals(value, bucket)}
                        for stream, value in sorted(streams.items())},
            "kernel_events": sum(e["cat"] == "kernel" for e in recorded),
            "tiny_kernel_events_under_5us": sum(e["cat"] == "kernel" and e["dur"] < 5 for e in recorded),
            "inventory": sorted(inventory, key=lambda row: -row["duration_sum_us"]),
        }
    api = [e for e in events if e.get("cat") in API_CATEGORIES and valid_duration(e)]
    threads, api_names = defaultdict(list), defaultdict(list)
    for event in api:
        threads[f"{event.get('pid', 'unknown')}:{event.get('tid', 'unknown')}"].append(event)
        api_names[event.get("name", "<unnamed>")].append(event)
    return {
        "label": label, "normalization_steps_supplied_by_user": steps,
        "trace_categories": dict(Counter(str(e.get("cat", "<none>")) for e in events)),
        "invalid_gpu_complete_events": sum(e.get("cat") in GPU_CATEGORIES and e.get("ph") == "X"
                                           and not valid_duration(e) for e in events),
        "devices": reports,
        "host_cuda_api": {
            "per_thread": {thread: {**event_totals(value, steps), "timeline": intervals(value, api_group)}
                           for thread, value in sorted(threads.items())},
            "inventory": sorted(({"name": name, "group": api_group(value[0]), **event_totals(value, steps)}
                                 for name, value in api_names.items()), key=lambda row: -row["duration_sum_us"]),
        },
        "profiler_overhead_events": [{"name": e.get("name"), "duration_us": e.get("dur")}
                                     for e in events if e.get("cat") == "overhead"],
        "limitations": LIMITATIONS,
    }


def render(report: dict) -> str:
    def number(value):
        return "—" if value is None else f"{value:.3f}"
    def safe(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    lines = [f"# Decode trace: {safe(report['label'])}", "",
             f"Source: `{report.get('trace_path', 'in-memory')}`. "
             f"SHA-256: `{report.get('trace_sha256', 'unavailable')}`.", "",
             f"Step normalization supplied by caller: {report['normalization_steps_supplied_by_user']}. "
             "This is a trace activity report, not an unprofiled throughput result.", ""]
    for device, data in report["devices"].items():
        t = data["timeline"]
        lines += [f"## Device {safe(device)}", "",
                  f"{data['events']} GPU events, including {data['kernel_events']} kernels; "
                  f"{data['tiny_kernel_events_under_5us']} kernels under 5 µs.", "",
                  f"GPU duration sum: {data['duration_sum_us'] / 1000:.3f} ms. "
                  f"Observed GPU span: {t['span_us'] / 1000:.3f} ms; "
                  f"union active: {t['active_us'] / 1000:.3f} ms; "
                  f"no recorded GPU activity: {t['inactive_us'] / 1000:.3f} ms. "
                  f"Cross-bucket overlap: {t['cross_bucket_overlap_us'] / 1000:.3f} ms.", "",
                  "| Bucket | Events | Sum ms | Work share | Sum µs/step | Active union ms | Only this bucket ms |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for name, row in sorted(data["buckets"].items(), key=lambda item: -item[1]["duration_sum_us"]):
            share = row["share_of_gpu_duration_sum"]
            lines.append(f"| {name} | {row['events']} | {row['duration_sum_us']/1000:.3f} | "
                         f"{number(share * 100 if share is not None else None)}% | "
                         f"{number(row['duration_sum_us_per_step'])} | {row['active_union_us']/1000:.3f} | "
                         f"{row['only_this_bucket_active_us']/1000:.3f} |")
        lines += ["", "`attention` includes GDN and QSA activation/state kernels. "
                  "`transfer` includes device copies as well as host transfers and expert-gather kernels; "
                  "it is not a measured PCIe-bandwidth bucket.", "",
                  "| Stream | Events | Sum ms | Active union ms | Span ms |", "| --- | ---: | ---: | ---: | ---: |"]
        for stream, row in data["streams"].items():
            lines.append(f"| {safe(stream)} | {row['events']} | {row['duration_sum_us']/1000:.3f} | "
                         f"{row['timeline']['active_us']/1000:.3f} | {row['timeline']['span_us']/1000:.3f} |")
        lines += ["", "Largest recorded GPU names (complete inventory in JSON):", "",
                  "| Name | Bucket | Calls | Sum ms | Mean µs |", "| --- | --- | ---: | ---: | ---: |"]
        for row in data["inventory"][:20]:
            lines.append(f"| {safe(row['name'])} | {row['bucket']} | {row['events']} | "
                         f"{row['duration_sum_us']/1000:.3f} | {number(row['duration_mean_us'])} |")
    lines += ["", "## Host CUDA APIs", "",
              "Inclusive call durations below may nest or overlap GPU work; they are not exposed host overhead.", "",
              "| Name | Group | Calls | Sum ms | Mean µs |", "| --- | --- | ---: | ---: | ---: |"]
    for row in report["host_cuda_api"]["inventory"][:20]:
        lines.append(f"| {safe(row['name'])} | {row['group']} | {row['events']} | "
                     f"{row['duration_sum_us']/1000:.3f} | {number(row['duration_mean_us'])} |")
    lines += ["", "Per-thread API unions avoid adding nested driver/runtime intervals:", "",
              "| PID:TID | API union ms | Launch union ms | Sync/wait union ms |",
              "| --- | ---: | ---: | ---: |"]
    for thread, row in report["host_cuda_api"]["per_thread"].items():
        t = row["timeline"]
        lines.append(f"| {safe(thread)} | {t['active_us']/1000:.3f} | "
                     f"{t['bucket_active_us'].get('launch',0)/1000:.3f} | "
                     f"{t['bucket_active_us'].get('synchronize_or_wait',0)/1000:.3f} |")
    lines += ["", f"Invalid GPU complete events excluded: {report['invalid_gpu_complete_events']}. "
              f"Profiler overhead records: {len(report['profiler_overhead_events'])}.", "", "## Interpretation limits", ""]
    lines.extend(f"- {text}" for text in report["limitations"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace")
    parser.add_argument("--label", required=True, help="Capture identity; mark historical validation as historical.")
    parser.add_argument("--steps", type=int, help="Verify steps actually captured, from capture configuration/log.")
    parser.add_argument("--out-prefix", help="Write .json and .md reports; otherwise only print Markdown.")
    args = parser.parse_args()
    path = Path(args.trace)
    raw = path.read_bytes()
    data = json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)
    report = analyze(data, label=args.label, steps=args.steps)
    report.update(trace_path=str(path.resolve()), trace_sha256=hashlib.sha256(raw).hexdigest(),
                  source_bytes=len(raw))
    text = render(report)
    if args.out_prefix:
        prefix = Path(args.out_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        Path(str(prefix) + ".json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        Path(str(prefix) + ".md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
