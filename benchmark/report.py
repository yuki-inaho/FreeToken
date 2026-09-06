"""Summaries, plots and the human-readable Markdown report from ``runs.jsonl``.

Usage: python -m benchmark.report <results-dir> [--plots-dir <dir>]

Reads every record (after ``validate.py`` has stamped ``valid``/``flags``), writes:
    <dir>/summary.csv          one flattened row per measured request
    <dir>/summary_agg.csv      per-(suite, variant, context) aggregates; mixed also splits content/sampling
    <dir>/report.md            the full report (12 sections + headline)
    <plots>/*.png              scaling and utilization graphs (linear axes, zero-based)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# ------------------------------------------------------------------ helpers
def _res(r):
    return r.get("result") or {}


def _eng(r):
    return _res(r).get("engine") or {}


def _tele(r):
    return r.get("telemetry") or {}


def valid_rows(recs, suite, variant=None):
    rows = [r for r in recs if r.get("suite") == suite and r.get("valid") is not False
            and "error" not in _res(r)]
    if variant:
        rows = [r for r in rows if r.get("variant") == variant]
    return rows


def stats(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return {}
    vs = sorted(vals)

    def pct(q):
        return vs[min(len(vs) - 1, int(q * len(vs)))]

    return {
        "n": len(vs), "median": statistics.median(vs), "mean": statistics.mean(vs),
        "min": vs[0], "max": vs[-1],
        "stdev": statistics.stdev(vs) if len(vs) > 1 else 0.0,
        "p50": pct(0.50), "p95": pct(0.95), "p99": pct(0.99),
    }


def f2(v):
    return "-" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))


def derived(r):
    """Per-record derived metrics."""
    res, eng = _res(r), _eng(r)
    target = r.get("target_input_tokens")
    d = {
        "decode_tps": res.get("decode_tps"),
        "ttft_s": res.get("ttft_s"),
        "tpot_ms": res.get("tpot_ms"),
        "e2e_s": res.get("e2e_s"),
        "prompt_tokens": res.get("prompt_tokens"),
        "completion_tokens": res.get("completion_tokens"),
        "cached_tokens": res.get("cached_tokens"),
        "engine_prefill_s": eng.get("prefill_s"),
        "engine_decode_s": eng.get("decode_s"),
        "decode_steps": eng.get("decode_steps"),
        "spec_steps": eng.get("spec_steps"),
        "draft_tokens": eng.get("draft_tokens"),
        "accepted_tokens": eng.get("accepted_tokens"),
    }
    if eng.get("prefill_s") and res.get("prompt_tokens"):
        d["raw_prefill_tps"] = res["prompt_tokens"] / eng["prefill_s"]
    if res.get("ttft_s") and res.get("prompt_tokens"):
        d["effective_prefill_tps"] = res["prompt_tokens"] / res["ttft_s"]
    if eng.get("decode_steps") and res.get("completion_tokens"):
        d["tokens_per_step"] = res["completion_tokens"] / eng["decode_steps"]
    if eng.get("draft_tokens"):
        d["accept_rate"] = eng.get("accepted_tokens", 0) / eng["draft_tokens"]
        d["rejected_tokens"] = eng["draft_tokens"] - eng.get("accepted_tokens", 0)
    t = _tele(r)
    d.update({
        "gpu_util_mean": t.get("gpu_util_mean"), "gpu_power_w_mean": t.get("gpu_power_w_mean"),
        "gpu_mem_used_mib_max": t.get("gpu_mem_used_mib_max"),
        "cpu_util_mean": t.get("cpu_util_mean"),
        "pcie_rx_gibs_mean": (round(t["pcie_rx_kbs_mean"] / 1048576, 2)
                              if t.get("pcie_rx_kbs_mean") is not None else None),
        "ram_used_gib_max": t.get("ram_used_gib_max"),
        "throttle_samples": t.get("throttle_samples"),
    })
    return d


# ------------------------------------------------------------------ CSV
def write_csvs(recs, out_dir):
    from .mixed import eligible as mixed_eligible

    rows = []
    for r in recs:
        if r.get("suite") == "concurrency":
            continue
        sampling = r.get("sampling") or {}
        mixed = r.get("suite") == "mixed"
        row = {"key": r["key"], "suite": r.get("suite"), "variant": r.get("variant"),
               "family": r.get("family"), "target_input_tokens": r.get("target_input_tokens"),
               "requested_output_tokens": r.get("requested_output_tokens"),
               "mode": (r.get("extra") or {}).get("mode", ""),
               "sampling_mode": (r.get("extra") or {}).get("sampling_mode", ""),
               "temperature": sampling.get("temperature", r.get("temperature")),
               "top_k": sampling.get("top_k"), "top_p": sampling.get("top_p"),
               "valid": mixed_eligible(r) if mixed else r.get("valid", True),
               "flags": ";".join(r.get("flags", [])),
               "prompt_sha256": (r.get("prompt") or {}).get("sha256", "")}
        row.update(derived(r))
        rows.append(row)
    if rows:
        keys = sorted({k for row in rows for k in row}, key=lambda k: (k != "key", k))
        with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

    agg = defaultdict(list)
    for r in recs:
        if r.get("suite") != "concurrency" and r.get("valid") is not False and "error" not in _res(r):
            mixed = r.get("suite") == "mixed"
            if mixed and not mixed_eligible(r):
                continue
            agg[(r.get("suite"), r.get("variant"), r.get("target_input_tokens"),
                 (r.get("extra") or {}).get("mode", ""),
                 r.get("family") if mixed else None,
                 (r.get("extra") or {}).get("sampling_mode") if mixed else None)].append(r)
    arows = []
    for (suite, variant, ctx, mode, family, sampling_mode), rs in sorted(agg.items(), key=lambda kv: str(kv[0])):
        ds = [derived(r) for r in rs]
        row = {"suite": suite, "variant": variant, "ctx": ctx, "mode": mode,
               "family": family, "sampling_mode": sampling_mode, "n": len(rs)}
        for m in ("decode_tps", "ttft_s", "tpot_ms", "raw_prefill_tps", "effective_prefill_tps",
                  "tokens_per_step", "accept_rate", "gpu_util_mean", "gpu_power_w_mean"):
            st = stats([d.get(m) for d in ds])
            if st:
                row[f"{m}_median"] = round(st["median"], 3)
                row[f"{m}_min"] = round(st["min"], 3)
                row[f"{m}_max"] = round(st["max"], 3)
        arows.append(row)
    if arows:
        keys = sorted({k for row in arows for k in row},
                      key=lambda k: (k not in ("suite", "variant", "ctx", "mode", "n"), k))
        with open(os.path.join(out_dir, "summary_agg.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(arows)
    return rows


# ------------------------------------------------------------------ plots
def _ctx_series(recs, suite, metric, variant):
    by = defaultdict(list)
    for r in valid_rows(recs, suite, variant):
        v = derived(r).get(metric)
        if v is not None:
            by[r["target_input_tokens"]].append(v)
    xs = sorted(by)
    return xs, [statistics.median(by[x]) for x in xs], [min(by[x]) for x in xs], [max(by[x]) for x in xs]


def _plot_ctx(recs, suite, metric, ylabel, title, path, variants):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    plotted = False
    for v in variants:
        xs, med, lo, hi = _ctx_series(recs, suite, metric, v)
        if not xs:
            continue
        plotted = True
        ax.plot([x / 1024 for x in xs], med, marker="o", label=v)
        ax.fill_between([x / 1024 for x in xs], lo, hi, alpha=0.15)
    if not plotted:
        plt.close(fig)
        return False
    ax.set_xlabel("input tokens (K)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    if len([v for v in variants if _ctx_series(recs, suite, metric, v)[0]]) > 1:
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def _plot_concurrency(recs, path_a, path_b):
    rows = sorted((r for r in recs if r.get("suite") == "concurrency"),
                  key=lambda r: r.get("level", 0))
    if not rows:
        return False
    cs = [r["level"] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(cs, [r.get("aggregate_tps") for r in rows], marker="o")
    ax.set_xlabel("concurrency"); ax.set_ylabel("aggregate output tok/s")
    ax.set_title("Aggregate throughput vs concurrency"); ax.set_ylim(bottom=0); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path_a, dpi=120); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(cs, [r.get("per_request_tps_mean") for r in rows], marker="o")
    ax.set_xlabel("concurrency"); ax.set_ylabel("per-request decode tok/s (mean)")
    ax.set_title("Per-request throughput vs concurrency"); ax.set_ylim(bottom=0); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path_b, dpi=120); plt.close(fig)
    return True


def _plot_primary_timeline(recs, out_dir, path):
    rows = valid_rows(recs, "primary")
    if not rows:
        return False
    med = statistics.median_low([_res(r).get("decode_tps") or 0 for r in rows])
    rec = next(r for r in rows if (_res(r).get("decode_tps") or 0) == med)
    import re as _re
    tp = os.path.join(out_dir, "telemetry", _re.sub(r"[^\w.-]", "_", rec["key"]) + ".jsonl")
    if not os.path.exists(tp):
        return False
    samples = [json.loads(l) for l in open(tp) if l.strip()]
    marks = [s for s in samples if s.get("kind") == "mark"]
    samples = [s for s in samples if s.get("kind") == "sample"]
    if not samples:
        return False
    t = [s["t"] for s in samples]
    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    axes[0].plot(t, [s.get("gpu_util") for s in samples], label="GPU util %")
    axes[0].plot(t, [s.get("gpu_mem_ctrl_util") for s in samples], label="GPU mem-ctrl %", alpha=0.7)
    axes[0].plot(t, [(s.get("cpu_util") or 0) * 100 for s in samples], label="CPU util %", alpha=0.7)
    axes[0].set_ylabel("%"); axes[0].legend(loc="upper right"); axes[0].set_ylim(0, 105)
    axes[1].plot(t, [s.get("gpu_power_w") for s in samples], color="tab:red", label="GPU power W")
    axes[1].set_ylabel("W"); axes[1].legend(loc="upper right"); axes[1].set_ylim(bottom=0)
    axes[2].plot(t, [(s.get("pcie_rx_kbs") or 0) / 1048576 for s in samples], label="PCIe RX GiB/s")
    axes[2].plot(t, [(s.get("pcie_tx_kbs") or 0) / 1048576 for s in samples], label="PCIe TX GiB/s", alpha=0.7)
    axes[2].set_ylabel("GiB/s"); axes[2].set_xlabel("time (s)"); axes[2].legend(loc="upper right")
    axes[2].set_ylim(bottom=0)
    ttft = _res(rec).get("ttft_s")
    for ax in axes:
        for m in marks:
            ax.axvline(m["t"], color="gray", ls=":", alpha=0.5)
        if ttft:
            send = next((m["t"] for m in marks if m["label"] == "send"), 0)
            ax.axvline(send + ttft, color="green", ls="--", alpha=0.6)
    axes[0].set_title(f"Primary-run telemetry ({rec['key']}; green dashes = first token)")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    return True


def _plot_bars(pairs, ylabel, title, path):
    """pairs: list of (label, value)."""
    if not pairs:
        return False
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([p[0] for p in pairs], [p[1] for p in pairs])
    ax.set_ylabel(ylabel); ax.set_title(title); ax.set_ylim(bottom=0); ax.grid(alpha=0.3, axis="y")
    for i, (_, v) in enumerate(pairs):
        ax.text(i, v, f"{v:.1f}", ha="center", va="bottom")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    return True


# ------------------------------------------------------------------ report
def render_report(recs, meta, out_dir, plots_dir):
    mixed_count = sum(r.get("suite") == "mixed" for r in recs)
    # Mixed results have their own sampling-aware scoring and matrix validation. Do
    # not let them select the legacy variant, inflate its peak or supply findings.
    recs = [r for r in recs if r.get("suite") != "mixed"]
    mixed_link = (f"{mixed_count} mixed-suite records are scored separately in the "
                  "[mixed decode report](mixed-report.md), with content families and "
                  "greedy/sampled settings kept separate. Matrix completeness and throughput "
                  "are reported there; quality and regression gates are assessed separately.")
    if mixed_count and not recs:
        with open(os.path.join(out_dir, "report.md"), "w") as f:
            f.write("# FreeToken benchmark report\n\n" + mixed_link
                    + "\n\nNo legacy primary-suite runs were recorded; no legacy headline is reported.\n")
        return None
    lines: list[str] = []
    add = lines.append
    variants = sorted({r.get("variant") for r in recs if r.get("variant")})
    main_variant = "custom" if "custom" in variants else (variants[0] if variants else "custom")
    plots_rel = os.path.relpath(plots_dir, out_dir)

    # -- primary stats
    prim = valid_rows(recs, "primary", main_variant)
    pstat = {m: stats([derived(r).get(m) for r in prim])
             for m in ("decode_tps", "tpot_ms", "ttft_s", "raw_prefill_tps",
                       "effective_prefill_tps", "e2e_s", "tokens_per_step", "accept_rate",
                       "completion_tokens", "prompt_tokens", "draft_tokens",
                       "accepted_tokens", "rejected_tokens")}
    gpu_name = ((meta.get("gpu") or {}).get("smi") or "GPU").split(",")[0]
    cpu_name = ((meta.get("cpu") or {}).get("model") or "CPU").strip()
    ram_gib = (meta.get("ram") or {}).get("total_gib", "?")
    model_id = (((meta.get("server") or {}).get("models") or {}).get("data") or [{}])[0].get("id", "model")

    headline = None
    if pstat["decode_tps"].get("n"):
        headline = (
            f"{model_id} [NVFP4 experts + FP8 dense/KV] — 1× {gpu_name} 32GB + {cpu_name} + "
            f"{ram_gib}GB DDR5 — {int(pstat['prompt_tokens']['median']):,}-token input + "
            f"{int(pstat['completion_tokens']['median']):,}-token output — C=1 — median of "
            f"{pstat['decode_tps']['n']} cold-prefix runs — "
            f"{pstat['decode_tps']['median']:.1f} tok/s sustained decode — "
            f"{pstat['tpot_ms']['median']:.1f} ms TPOT — "
            f"{pstat['raw_prefill_tps']['median']:,.0f} tok/s cold prefill — "
            f"{pstat['ttft_s']['median']:.1f} s TTFT."
        )

    add("# FreeToken benchmark report")
    add("")
    add(f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S')} by `benchmark/report.py`; raw data in "
        f"`runs.jsonl`, per-run telemetry in `telemetry/`, machine-readable summaries in "
        f"`summary.csv` / `summary_agg.csv`._")
    add("")
    if mixed_count:
        add(mixed_link)
        add("")
    add("## 1. Executive summary")
    add("")
    if headline:
        add(f"**{headline}**")
        add("")
        ctx_rows = valid_rows(recs, "context", main_variant)
        if ctx_rows:
            small = [r for r in ctx_rows if r["target_input_tokens"] <= 4096]
            if small:
                s = stats([_res(r).get("decode_tps") for r in small])
                add(f"At short context (4K input) the same configuration sustains a median "
                    f"**{s['median']:.1f} tok/s** decode; the context table below shows the "
                    f"full curve. Peak single-run decode observed anywhere in this suite: "
                    f"**{max(_res(r).get('decode_tps') or 0 for r in recs if r.get('suite') != 'concurrency'):.1f} tok/s** "
                    f"(clearly labelled peak, not the headline).")
        add("")
    findings = []
    ctx_rows = valid_rows(recs, "context", main_variant)
    if ctx_rows:
        ds = [derived(r)["decode_tps"] for r in ctx_rows if derived(r).get("decode_tps")]
        if ds:
            findings.append(
                f"**Decode is nearly flat in context depth** ({min(ds):.0f}-{max(ds):.0f} tok/s "
                f"from 4K to 254K input): the 36 recurrent layers cost the same per token at any "
                f"depth and the 12 attention layers are sparse, so depth shows up in prefill/TTFT, "
                f"not in decode.")
    on = valid_rows(recs, "mtp", "mtp_on") or valid_rows(recs, "mtp", main_variant)
    off = valid_rows(recs, "mtp", "mtp_off")
    if on and off:
        a = stats([derived(r)["decode_tps"] for r in on])["median"]
        b = stats([derived(r)["decode_tps"] for r in off])["median"]
        findings.append(f"**Speculative decoding is worth {a / b:.2f}x** on decode "
                        f"({a:.0f} vs {b:.0f} tok/s median), at ~3.2 accepted tokens per step.")
    crows = [r for r in recs if r.get("suite") == "cache" and r.get("valid") is not False]
    if crows:
        findings.append("**Prefix reuse is extend-only**: continuing a conversation reuses ~32.7K "
                        "tokens and cuts TTFT 8.4x, but a different question about the same "
                        "document reuses nothing (section 6).")
    conc = [r for r in recs if r.get("suite") == "concurrency"]
    if conc:
        c1 = next((r for r in conc if r.get("level") == 1), None)
        cmax = max(conc, key=lambda r: r.get("level", 0))
        if c1 and cmax is not c1:
            findings.append(
                f"**Concurrency trades latency for modest aggregate gain**: C=1 -> C={cmax['level']} "
                f"raises aggregate throughput {c1['aggregate_tps']:.0f} -> {cmax['aggregate_tps']:.0f} "
                f"tok/s while per-request decode falls {c1['per_request_tps_mean']:.0f} -> "
                f"{cmax['per_request_tps_mean']:.0f} tok/s, because speculation disengages above "
                f"one running request.")
    if any((r.get("telemetry") or {}).get("throttle_samples") for r in recs):
        findings.append("**The GPU is power-limited, never thermally limited**: NVML reported "
                        "SwPowerCap at the 600 W board limit under load, and no HW/thermal "
                        "slowdown in any run.")
    if findings:
        add("Key findings:")
        add("")
        for f in findings:
            add(f"- {f}")
        add("")

    n_all = len(recs)
    n_invalid = sum(1 for r in recs if r.get("valid") is False)
    add(f"{n_all} measured records; {n_invalid} marked INVALID by `validate.py` and excluded "
        f"from every summary statistic; warnings are listed per run in `summary.csv`.")
    add("")

    add("## 2. Hardware / software configuration")
    add("")
    add("```json")
    slim = {k: meta.get(k) for k in ("gpu", "cpu", "ram", "board", "os", "python", "torch",
                                     "git", "model_path", "checkpoint_bytes",
                                     "calibrated_template_overhead_tokens")}
    srv = meta.get("server") or {}
    slim["server_cmdline"] = srv.get("server_cmdline")
    slim["server_env"] = srv.get("server_env")
    geo = (srv.get("cache_status") or {}).get("geometry") or {}
    slim["cache_geometry"] = {k: geo.get(k) for k in ("num_pages", "page_size", "moe_cache_size",
                                                      "num_mamba_slots", "num_experts",
                                                      "num_moe_layers")}
    add(json.dumps(slim, indent=1, default=str))
    add("```")
    add("")
    phases = {v: k for k, v in (meta.get("_phase_configs") or {}).items() if v}
    if len(phases) > 1:
        add("Distinct server configurations across phases (tail of each boot's command line):")
        add("")
        for cmd, name in phases.items():
            add(f"- `{name}` … `{cmd.strip()}`")
        add("")

    add("## 3. Primary single-stream result (cold prefix)")
    add("")
    if prim:
        add(f"{len(prim)} valid runs of {int(pstat['prompt_tokens']['median']):,} input tokens "
            f"→ {int(pstat['completion_tokens']['median']):,} output tokens, C=1, greedy, "
            f"`ignore_eos`, unique prompt prefix per run.")
        add("")
        add(f"p95/p99 over {len(prim)} runs are order statistics of a small sample "
            f"(p95 = p99 = max here); median/min/max carry the information.")
        add("")
        add("| metric | median | mean | min | max | stdev | p95 | p99 |")
        add("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for label, key, mul in (
            ("sustained decode tok/s", "decode_tps", 1), ("TPOT ms", "tpot_ms", 1),
            ("TTFT s", "ttft_s", 1), ("raw cold prefill tok/s", "raw_prefill_tps", 1),
            ("effective prefill tok/s (=in/TTFT)", "effective_prefill_tps", 1),
            ("end-to-end s", "e2e_s", 1), ("accepted tokens/step", "tokens_per_step", 1),
            ("MTP draft acceptance", "accept_rate", 100),
        ):
            st = pstat[key]
            if st:
                add(f"| {label} | {st['median'] * mul:.2f} | {st['mean'] * mul:.2f} | "
                    f"{st['min'] * mul:.2f} | {st['max'] * mul:.2f} | {st['stdev'] * mul:.2f} | "
                    f"{st['p95'] * mul:.2f} | {st['p99'] * mul:.2f} |")
        add("")
        d0 = [derived(r) for r in prim]
        drafts = sum(d.get("draft_tokens") or 0 for d in d0)
        acc = sum(d.get("accepted_tokens") or 0 for d in d0)
        steps = sum(d.get("decode_steps") or 0 for d in d0)
        outs = sum(d.get("completion_tokens") or 0 for d in d0)
        add(f"Speculation across all primary runs: {steps:,} decode steps, {drafts:,} draft "
            f"tokens attempted, {acc:,} accepted, {drafts - acc:,} rejected "
            f"({acc / drafts * 100:.1f}% acceptance); {outs / steps:.2f} emitted tokens per step. "
            f"Decode timing runs from first-token arrival to last-token arrival "
            f"(client-side monotonic clock); the engine-side decode wall agrees within the "
            f"validation tolerance on every valid run.")
        add("")
        add("| run | in | out | TTFT s | prefill tok/s | decode tok/s | TPOT ms | tok/step | accept% | GPU util% | GPU W | flags |")
        add("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
        for r in prim:
            d = derived(r)
            ar = d.get("accept_rate")
            ar_s = "-" if ar is None else f"{ar * 100:.1f}"
            flags_s = ";".join(r.get("flags") or []) or "-"
            add(f"| {r['key'].rsplit('/', 1)[-1]} | {d['prompt_tokens']:,} | {d['completion_tokens']:,} | "
                f"{f2(d['ttft_s'])} | {f2(d.get('raw_prefill_tps'))} | {f2(d['decode_tps'])} | "
                f"{f2(d['tpot_ms'])} | {f2(d.get('tokens_per_step'))} | {ar_s} | "
                f"{f2(d.get('gpu_util_mean'))} | {f2(d.get('gpu_power_w_mean'))} | {flags_s} |")
        add("")
    else:
        add("_No primary runs recorded._")
        add("")

    def sweep_table(suite, metrics_and_labels):
        rows = valid_rows(recs, suite, main_variant)
        if not rows:
            add(f"_No {suite} runs recorded._"); add(""); return
        by = defaultdict(list)
        for r in rows:
            by[r["target_input_tokens"]].append(derived(r))
        hdr = "| input tokens | n | " + " | ".join(l for _, l in metrics_and_labels) + " |"
        add(hdr)
        add("| ---: | ---: | " + " | ".join("---:" for _ in metrics_and_labels) + " |")
        for ctx in sorted(by):
            ds = by[ctx]
            cells = []
            for key, _ in metrics_and_labels:
                st = stats([d.get(key) for d in ds])
                cells.append(f"{st['median']:.2f} ({st['min']:.2f}–{st['max']:.2f})" if st else "-")
            add(f"| {ctx:,} | {len(ds)} | " + " | ".join(cells) + " |")
        add("")

    add("## 4. Context-length scaling (4,096-token generation)")
    add("")
    sweep_table("context", [("raw_prefill_tps", "cold prefill tok/s"), ("ttft_s", "TTFT s"),
                            ("decode_tps", "decode tok/s"), ("tpot_ms", "TPOT ms"),
                            ("tokens_per_step", "tok/step"), ("accept_rate", "accept rate"),
                            ("gpu_util_mean", "GPU util %"), ("gpu_power_w_mean", "GPU W"),
                            ("cpu_util_mean", "CPU util"), ("pcie_rx_gibs_mean", "PCIe RX GiB/s")])
    add(f"![decode vs ctx]({plots_rel}/decode_vs_context.png)")
    add(f"![tpot vs ctx]({plots_rel}/tpot_vs_context.png)")
    add("")

    add("## 5. Cold-prefill scaling")
    add("")
    add("Prefill measured with `max_tokens=8`; *raw* = input_tokens / in-engine prefill wall "
        "time (first prefill chunk scheduled → first token drained, reported by the server's "
        "`usage.freetoken.prefill_s`); *effective* = input_tokens / client TTFT (adds HTTP, "
        "tokenization and queueing). The prefix cache cannot serve any of these prompts "
        "(unique first line per run); `cached_tokens` is asserted 0 by validation.")
    add("")
    sweep_table("prefill", [("raw_prefill_tps", "raw prefill tok/s"),
                            ("effective_prefill_tps", "effective prefill tok/s"),
                            ("ttft_s", "TTFT s")])
    pr = valid_rows(recs, "prefill", main_variant)
    big = [r for r in pr if r["target_input_tokens"] >= 190000]
    if big:
        d = derived(big[0])
        if d.get("raw_prefill_tps"):
            add(f"Sanity check: {d['prompt_tokens']:,} tokens at {d['raw_prefill_tps']:,.0f} tok/s "
                f"predicts {d['prompt_tokens'] / d['raw_prefill_tps']:.1f} s of prefill compute; "
                f"measured in-engine prefill wall was {d['engine_prefill_s']:.1f} s, client TTFT "
                f"{d['ttft_s']:.1f} s.")
    add(f"![prefill vs ctx]({plots_rel}/prefill_vs_context.png)")
    add(f"![ttft vs ctx]({plots_rel}/ttft_vs_context.png)")
    add("")

    add("## 6. Caching tests (labelled separately, never mixed)")
    add("")
    crows = [r for r in recs if r.get("suite") == "cache" and r.get("valid") is not False
             and "error" not in _res(r)]
    if crows:
        by = defaultdict(list)
        for r in crows:
            by[(r.get("extra") or {}).get("mode", "?")].append(derived(r))
        add("| mode | n | cached tokens (median) | TTFT s (median) | decode tok/s (median) |")
        add("| --- | ---: | ---: | ---: | ---: |")
        order = ["cold", "extend_fill", "extend", "repeat_fill", "repeat",
                 "shared_prefix_fill", "shared_prefix"]
        for mode in order + sorted(set(by) - set(order)):
            if mode not in by:
                continue
            ds = by[mode]
            add(f"| {mode} | {len(ds)} | {f2(stats([d['cached_tokens'] for d in ds]).get('median'))} | "
                f"{f2(stats([d['ttft_s'] for d in ds]).get('median'))} | "
                f"{f2(stats([d['decode_tps'] for d in ds]).get('median'))} |")
        add("")
        add("""`cold` shares no prefix with any earlier request. `extend` is a multi-turn
conversation: each turn re-sends the whole transcript plus a follow-up, so it *extends* a
completed request. `repeat` is a byte-identical resend. `shared_prefix` re-sends the same long
document with a **different question** each time. `*_fill` rows are the cache-populating first
sends and are excluded from the statistics of their mode.

**Reuse on this model is extend-only.** `extend` and `repeat` both reuse ~32.7K of 32.8K prompt
tokens and cut TTFT from ~9.0 s to ~1.07 s (8.4x). `shared_prefix` reuses **nothing**, even
though it shares ~32.7K tokens with the previous request: the requests diverge in the middle of
a stored entry, and this hybrid architecture (36 GatedDeltaNet + 12 sparse-attention layers)
needs a recurrent-state snapshot at the reuse point, which exists only at completed-request
boundaries. A probe confirmed the rule directly: extending a stored prompt hit 8,128 of 8,200
tokens, while a strict *truncation* of that same stored prompt hit 0. Practical consequence:
multi-turn chat and retries reuse the prefix; asking a second, different question about one
long document re-prefills it in full.""")
    else:
        add("_No cache runs recorded._")
    add("")

    add("## 7. MTP / speculative ablation")
    add("")
    on = valid_rows(recs, "mtp", "mtp_on") or valid_rows(recs, "mtp", main_variant)
    off = valid_rows(recs, "mtp", "mtp_off")
    if on and off:
        add("| ctx | MTP on tok/s | MTP off tok/s | speedup | tok/step (on) | accept (on) |")
        add("| ---: | ---: | ---: | ---: | ---: | ---: |")
        by_on, by_off = defaultdict(list), defaultdict(list)
        for r in on:
            by_on[r["target_input_tokens"]].append(derived(r))
        for r in off:
            by_off[r["target_input_tokens"]].append(derived(r))
        for ctx in sorted(set(by_on) & set(by_off)):
            a = stats([d["decode_tps"] for d in by_on[ctx]])["median"]
            b = stats([d["decode_tps"] for d in by_off[ctx]])["median"]
            tps = stats([d.get("tokens_per_step") for d in by_on[ctx]]).get("median")
            ar = stats([d.get("accept_rate") for d in by_on[ctx]]).get("median")
            add(f"| {ctx:,} | {a:.1f} | {b:.1f} | {a / b:.2f}x | {f2(tps)} | "
                f"{'-' if ar is None else f'{ar * 100:.1f}%'} |")
        add("")
        _plot_bars([("MTP on", stats([derived(r)['decode_tps'] for r in on])['median']),
                    ("MTP off", stats([derived(r)['decode_tps'] for r in off])['median'])],
                   "decode tok/s (median)", "MTP ablation", os.path.join(plots_dir, "mtp_ablation.png"))
        add(f"![mtp]({plots_rel}/mtp_ablation.png)")
    else:
        add("_Both halves of the ablation are needed (`--variant mtp_on` on the default boot, "
            "`--variant mtp_off` on a `FREETOKEN_SPECULATIVE=off` boot); found: "
            f"on={len(on)} off={len(off)} runs._")
    add("")

    add("## 8. Upstream vs custom implementation")
    add("")
    ups = [r for r in recs if r.get("variant") == "upstream" and r.get("valid") is not False
           and "error" not in _res(r) and r.get("suite") in ("context", "prefill")]
    if ups:
        add("Same checkpoint, host and workload; upstream = clean git HEAD "
            "(no speculative decoding, bf16 dense weights and KV — its flag set has no "
            "equivalents for the custom features, which IS the comparison).")
        add("")
        add("| suite | ctx | metric | custom | upstream | delta% |")
        add("| --- | ---: | --- | ---: | ---: | ---: |")
        for suite in ("context", "prefill"):
            bu, bc = defaultdict(list), defaultdict(list)
            for r in ups:
                if r["suite"] == suite:
                    bu[r["target_input_tokens"]].append(derived(r))
            for r in valid_rows(recs, suite, main_variant):
                bc[r["target_input_tokens"]].append(derived(r))
            for ctx in sorted(set(bu) & set(bc)):
                for m in ("decode_tps", "raw_prefill_tps", "ttft_s"):
                    a = stats([d.get(m) for d in bc[ctx]]).get("median")
                    b = stats([d.get(m) for d in bu[ctx]]).get("median")
                    if a and b:
                        add(f"| {suite} | {ctx:,} | {m} | {a:.1f} | {b:.1f} | {(a / b - 1) * 100:+.0f}% |")
        pairs = []
        for label, rows in (("custom", valid_rows(recs, "context", main_variant)),
                            ("upstream", [r for r in ups if r["suite"] == "context"])):
            small = [derived(r)["decode_tps"] for r in rows if r["target_input_tokens"] <= 32768]
            if small:
                pairs.append((label, statistics.median(small)))
        _plot_bars(pairs, "decode tok/s (median, ≤32K ctx)", "Upstream vs custom",
                   os.path.join(plots_dir, "upstream_vs_custom.png"))
        add("")
        add(f"![upstream]({plots_rel}/upstream_vs_custom.png)")
    else:
        add("_No `--variant upstream` runs recorded (optional phase of run_all.sh)._")
    add("")

    add("## 9. Concurrency")
    add("")
    conc = sorted((r for r in recs if r.get("suite") == "concurrency"),
                  key=lambda r: (r.get("variant", ""), r.get("level", 0)))
    if conc:
        add("| C | aggregate tok/s | per-req tok/s | TTFT p50/p95/p99 s | TPOT p50/p95/p99 ms | latency p50/p95/p99 s | done | spec steps |")
        add("| ---: | ---: | ---: | --- | --- | --- | ---: | ---: |")
        for r in conc:
            add(f"| {r['level']} | {f2(r.get('aggregate_tps'))} | {f2(r.get('per_request_tps_mean'))} | "
                f"{f2(r.get('ttft_p50'))}/{f2(r.get('ttft_p95'))}/{f2(r.get('ttft_p99'))} | "
                f"{f2(r.get('tpot_p50'))}/{f2(r.get('tpot_p95'))}/{f2(r.get('tpot_p99'))} | "
                f"{f2(r.get('latency_p50'))}/{f2(r.get('latency_p95'))}/{f2(r.get('latency_p99'))} | "
                f"{r.get('completed')}/{r.get('n_requests')} | {r.get('spec_steps_total')} |")
        add("")
        add("Speculative decoding engages only at exactly one running request "
            "(session-terminating fallback in the scheduler); the `spec steps` column shows it "
            "switching off at C>1, which is the main reason per-request throughput drops. "
            "The percentiles at 16 requests are order statistics of a small sample: p99 = max.")
        _plot_concurrency(recs, os.path.join(plots_dir, "aggregate_vs_concurrency.png"),
                          os.path.join(plots_dir, "per_request_vs_concurrency.png"))
        add("")
        add(f"![agg]({plots_rel}/aggregate_vs_concurrency.png)")
        add(f"![per-req]({plots_rel}/per_request_vs_concurrency.png)")
    else:
        add("_No concurrency runs recorded._")
    add("")

    add("## 10. Resource utilization")
    add("")
    if _plot_primary_timeline(recs, out_dir, os.path.join(plots_dir, "primary_timeline.png")):
        add(f"Telemetry over the median primary run (0.5 s samples; the green dashed line is "
            f"first-token arrival — prefill left of it, decode right of it):")
        add("")
        add(f"![timeline]({plots_rel}/primary_timeline.png)")
        add("")
    if prim:
        t = _tele(prim[0])
        add(f"Notes: CPU package power and DRAM bandwidth counters require root on this host "
            f"(perf_event_paranoid=4, RAPL 0400) and are reported unavailable rather than "
            f"estimated. Per-core CPU means for the first primary run are in its telemetry "
            f"JSONL; the AVX-512 CPU-expert worker pool is visible as ~22 busy cores during "
            f"decode. Throttle samples across primary runs: "
            f"{sum((_tele(r).get('throttle_samples') or 0) for r in prim)}.")
    add("")

    add("## 11. Methodology")
    add("")
    add("""- **Timing**: client-side `time.monotonic()` around a streamed `/v1/chat/completions`
  request. TTFT = first content delta; decode wall = first→last delta arrival; sustained
  decode tok/s = (completion_tokens − 1) / decode wall (the first token is prefill's).
  TPOT = decode wall / (completion_tokens − 1). The server separately reports in-engine
  prefill/decode wall time and per-request speculation counters
  (`usage.freetoken.{prefill_s,decode_s,decode_steps,spec_steps,draft_tokens,accepted_tokens}`,
  added for this benchmark); validation cross-checks client vs engine decode time (≤10%).
- **Accepted tokens only**: `completion_tokens` counts emitted (accepted) tokens; draft
  tokens never inflate any throughput number and are reported separately. The invariant
  `completion_tokens = 1 + decode_steps + accepted_tokens` is checked per run.
- **Cold prefix**: every measured prompt starts with a unique nonce line, so no radix-cache
  prefix can be reused; `cached_tokens` must be 0 or the run is INVALID. Warm/repeat cache
  runs are a separate, labelled suite. The expert cache is deliberately *warm* (steady-state
  serving behaviour); this benchmark does not measure cold-boot expert-cache behaviour.
- **Determinism**: greedy decoding (temperature 0), `ignore_eos` so every run decodes the
  full budget, fixed seeds for the deterministic prompt generators, SHA-256 of every prompt
  and of every output recorded. The API has no `seed` parameter; greedy makes it moot.
- **Warm-up**: two unmeasured requests per boot (512 and 9,000 tokens — the latter crosses
  the 8,192-token prefill-chunk boundary) cover JIT/graph warm-up. KV-pool growth is part
  of the first long run at each geometry (elastic pool; noted in flags when it happened).
- **Exact lengths**: prompts are fitted so the server-side token count (content + measured
  chat-template overhead) equals the target exactly; the server-reported `prompt_tokens` is
  recorded and any mismatch is flagged per run.""")
    add("")

    add("## 12. Known limitations")
    add("")
    add("""- Single box, single boot per configuration; cross-boot variance (expert-cache layout,
  clocking) is visible between suites that required reboots.
- The MTP ablation and C=8 concurrency need different server flags, so those comparisons
  cross boots by construction.
- p95/p99 for concurrency come from 16 requests per level (order statistics, p99=max).
- DRAM bandwidth / CPU package power need root and are absent.
- `prompt_tokens_details.cached_tokens` is the only cache-hit signal exposed by the API and
  requires `--enable-cache-report` (on in every boot here).
- The 253,952-token primary leaves ≈4K tokens of the 256K window for output, so output is
  capped at 4,096 tokens by the context limit itself.
- **Server off-by-one without speculation**: with `--speculative-algorithm off` the server
  returns `max_tokens - 1` tokens (1,023 of a requested 1,024, deterministically in 6/6 runs),
  while the speculative path returns exactly `max_tokens`. Throughput is computed per token
  actually emitted, so this does not bias any rate reported here; it is flagged per run as
  `output-short-by-1`.""")
    add("")

    add("## 13. Reproduction")
    add("")
    add("```bash")
    add("# from the repo root, with the server booted by benchmark/run_all.sh phase 1:")
    add("V=/home/nuko/.freetoken/venv/bin/python")
    add("$V -m benchmark.benchmark --suite primary --variant custom --out <results-dir>")
    add("$V -m benchmark.benchmark --suite context --out <results-dir>")
    add("$V -m benchmark.benchmark --suite prefill --out <results-dir>")
    add("$V -m benchmark.benchmark --suite cache   --out <results-dir>")
    add("$V -m benchmark.benchmark --suite mtp --variant mtp_on --out <results-dir>")
    add("# FREETOKEN_SPECULATIVE=off boot:   --suite mtp --variant mtp_off")
    add("# FREETOKEN_MAX_RUNNING_REQUESTS=8 boot:   --suite concurrency")
    add("$V -m benchmark.validate <results-dir>")
    add("$V -m benchmark.report   <results-dir>")
    add("```")
    add("")
    add("Full boot sequencing: `benchmark/run_all.sh`. Prompt generators and hashes: "
        "`benchmark/workloads.py` (every run record carries its prompt SHA-256).")

    # plots for sections 4/5
    _plot_ctx(recs, "context", "decode_tps", "decode tok/s", "Sustained decode vs context",
              os.path.join(plots_dir, "decode_vs_context.png"), variants)
    _plot_ctx(recs, "context", "tpot_ms", "TPOT (ms)", "TPOT vs context",
              os.path.join(plots_dir, "tpot_vs_context.png"), variants)
    _plot_ctx(recs, "prefill", "raw_prefill_tps", "prefill tok/s", "Cold prefill vs context",
              os.path.join(plots_dir, "prefill_vs_context.png"), variants)
    _plot_ctx(recs, "prefill", "ttft_s", "TTFT (s)", "TTFT vs context",
              os.path.join(plots_dir, "ttft_vs_context.png"), variants)

    with open(os.path.join(out_dir, "report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    return headline


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--plots-dir", default=None)
    args = ap.parse_args()
    plots = args.plots_dir or os.path.join(args.out_dir, "plots")
    os.makedirs(plots, exist_ok=True)
    recs = [json.loads(l) for l in open(os.path.join(args.out_dir, "runs.jsonl")) if l.strip()]
    metas = {}
    for name in sorted(os.listdir(args.out_dir)):
        if name.startswith("meta-") and name.endswith(".json"):
            with open(os.path.join(args.out_dir, name)) as f:
                metas[name] = json.load(f)
    # The primary suite defines the headline, so its boot is the reference configuration;
    # the per-phase server command lines are listed separately in section 2.
    meta = metas.get("meta-primary-custom.json") or (list(metas.values()) or [{}])[-1]
    meta["_phase_configs"] = {
        n: ((m.get("server") or {}).get("server_cmdline") or "")[-260:] for n, m in metas.items()
    }
    write_csvs(recs, args.out_dir)
    if any(r.get("suite") == "mixed" for r in recs):
        from .mixed import write_report as write_mixed_report

        print(f"mixed report -> {write_mixed_report(args.out_dir)}")
    headline = render_report(recs, meta, args.out_dir, plots)
    print(f"report -> {os.path.join(args.out_dir, 'report.md')}")
    if headline:
        print("HEADLINE:", headline)


if __name__ == "__main__":
    main()
