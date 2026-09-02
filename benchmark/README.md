# FreeToken benchmark suite

Reproducible, publication-grade measurement of a running FreeToken server
(Qwen3.8-Flash-Next on a single RTX 5090 here, but nothing is model-specific).
The design goals are measurement accuracy, reproducibility and transparency —
not the biggest number: cold prefixes are guaranteed and validated, speculative
draft tokens are never counted as output, decode timing starts at the first
token and spans the whole generation, and every prompt is deterministic and
hash-recorded.

## Layout

    benchmark.py    measurement driver (suites: primary/context/prefill/cache/mtp/concurrency/smoke)
    workloads.py    deterministic prompts with exact server-side token counts (synthetic + realistic)
    telemetry.py    0.5 s GPU/CPU/RAM/PCIe/server sampler -> per-run JSONL + summaries
    validate.py     hard INVALID checks + soft warnings, stamped into runs.jsonl
    compare.py      variant-vs-variant tables (mtp_on/mtp_off, custom/upstream)
    report.py       summary.csv, summary_agg.csv, report.md, plots/
    run_all.sh      boots the right server per phase and runs everything end to end
    results/        one directory per suite execution (runs.jsonl, meta-*.json, telemetry/, report.md)

## Quick start

    # full suite (hours; boots servers, requires an idle desktop / no other GPU clients):
    bash benchmark/run_all.sh

    # or by hand against an already-running server:
    V=/home/nuko/.freetoken/venv/bin/python
    $V -m benchmark.benchmark --suite smoke --out benchmark/results/dev --variant custom
    $V -m benchmark.validate benchmark/results/dev
    $V -m benchmark.report   benchmark/results/dev

Run modules from the repo root (`python -m benchmark....`). The driver needs the
model checkpoint path only for its tokenizer (`--model-path`).

## Measurement conventions

* **TTFT / decode wall**: client-side monotonic clock on the SSE stream; sustained
  decode tok/s = (completion_tokens − 1) / (last − first token arrival); TPOT is its inverse.
* **Engine cross-check**: the server reports `usage.freetoken`
  (`prefill_s`, `decode_s`, `decode_steps`, `spec_steps`, `draft_tokens`, `accepted_tokens`)
  — per-request counters added to FreeToken for this benchmark. Validation requires
  client and engine decode walls to agree within 10% and
  `completion_tokens = 1 + decode_steps + accepted_tokens` (± adaptive-k slack).
* **Cold prefix**: unique nonce line at the top of every measured prompt; any
  `cached_tokens > 0` on a cold run marks it INVALID. Warm/repeat prefix runs are a
  separate labelled suite. The **expert cache stays warm** on purpose (steady-state serving).
* **Exact lengths**: prompts are token-fitted against the checkpoint tokenizer plus the
  measured chat-template overhead; the server-reported `prompt_tokens` is recorded and
  any off-target run is flagged.
* **Full-length decode**: `ignore_eos: true` (documented server extension), greedy
  (temperature 0), no stop strings; the API has no seed parameter — greedy makes runs
  deterministic up to kernel scheduling.

## Boots required for the full matrix (run_all.sh does this)

| phase | launcher env | suites |
| --- | --- | --- |
| 1 | default `scripts/serve-qwen38.sh` | primary, context, prefill, cache, mtp(--variant mtp_on), concurrency C=1 row |
| 2 | `FREETOKEN_SPECULATIVE=off` | mtp (--variant mtp_off) |
| 3 | `FREETOKEN_MAX_RUNNING_REQUESTS=8` | concurrency C=1,2,4,8 (--variant concurrency8) |
| 4 (optional) | clean git HEAD worktree (upstream) | context/prefill subset (--variant upstream) |

Between phases the old server is SIGTERMed and VRAM is awaited < 2 GiB.
