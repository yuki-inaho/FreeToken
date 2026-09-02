"""Reproducible single-GPU benchmark suite for a running FreeToken server.

Modules:
    workloads   deterministic prompts with exact token counts (synthetic + realistic mixed)
    telemetry   background GPU/CPU/RAM/server sampler (NVML + /proc + /v1/stats)
    benchmark   the measurement driver (primary, context sweep, prefill sweep, cache tests,
                concurrency, MTP ablation); writes JSONL records + telemetry per run
    validate    automatic sanity checks that mark suspicious runs INVALID
    compare     variant-vs-variant tables (MTP on/off, upstream vs custom)
    report      CSV summaries, Markdown report, plots
"""
