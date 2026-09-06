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
    mixed.py        balanced mixed matrix checks and sampling-separated rates
    compare_decode.py exact prompt-paired comparisons with geometry and follow-up flags
    run_decode_followup.sh fresh baseline/execution-only repeats for selected cells
    profile_decode.py CPU-only Chrome trace accounting across devices and streams
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

## Mixed decode optimization and validation

`mixed` measures code, English and Japanese independently at 4K, 32K, 128K and
253,952-token inputs. Each generation has 4,096 output tokens. Greedy uses
temperature 0 / top-k 1 / top-p 1; sampled uses temperature 1 / top-k 20 / top-p
0.95. The reported overall rate is total emitted decode tokens divided by total
decode seconds, with separate scores for the two sampling modes.

```bash
# Short, independent feature ablations (default port 1920, separate server boots).
bash benchmark/run_decode.sh decode-ablation

# Final configuration: ten runs per cell, five in each of two server boots.
bash benchmark/run_decode.sh decode-qualification combined 1,2 5 \
    4096,32768,131072,253952 4096

# Against an existing server; run this once per boot with different --boot-id.
/home/nuko/.freetoken/venv/bin/python -m benchmark.benchmark --suite mixed \
    --base http://127.0.0.1:1920/v1 --out benchmark/results/mixed \
    --variant combined --boot-id 1 --runs 5 --server-log /path/to/server.log
/home/nuko/.freetoken/venv/bin/python -m benchmark.mixed benchmark/results/mixed
```

The orchestrator stops only processes it starts and refuses to interrupt an
existing server. It snapshots `/home/nuko/ft/python` into a space-free temporary
directory before each boot: TVM-FFI resolves symlinks and otherwise breaks its
ninja source paths when the actual checkout contains spaces. Override
`FREETOKEN_DECODE_PYTHONPATH` for another installation.
`FREETOKEN_DECODE_BASELINE_PYTHON` can point at an exact pre-edit
source snapshot. Otherwise the baseline disables the following switches:

| Variant | Enabled switch(es) |
| --- | --- |
| admission | `FREETOKEN_HYBRID_ADMISSION_CREDIT` |
| shared | `FREETOKEN_HYBRID_SHARED_OVERLAP` |
| gdn | `FREETOKEN_GDN_STRIDED_NORM` |
| qsa | `FREETOKEN_QSA_PARALLEL_HEADS`, `FREETOKEN_QSA_DIRECT_PERMUTE` |
| combined | all five |

Switches are read at import; changing them requires a new server/graph capture.
Kernel microbenchmarks are in `benchmark/kernels/decode_cleanup.py` and require
the CUDA 13 environment used by the launcher. Do not run GPU microbenchmarks
concurrently with a server benchmark.

Each mixed record includes actual server instance identity, pool geometry,
sampling settings, prompt/output hashes, engine step/acceptance counters and raw
GPU/CPU/PCIe telemetry. With `--server-log` and `FREETOKEN_MOE_STATS_LOG=200`,
it also retains the existing rounded MoE diagnostic windows. Their first window
can include the previous request's tail; CPU-expert and fetched-byte summaries
are labelled estimates. GPU telemetry summaries span prefill and decode; raw
timestamps permit phase-specific analysis.

`mixed-report.md` and `mixed-summary.csv` separate every language/context and
sampling mode. The **Full matrix complete** column requires the full balanced
matrix, exact token counts, valid accounting, unique repetitions, and two distinct server instance
IDs. Partial results are exploratory even when an observed rate exceeds 220.
A complete matrix and the **220 tok/s rate** column cover measurement validity
and throughput only. Quality and regression gates are assessed separately and
must also pass before overall validation is complete.
Prompt bytes match across variants with the same boot/repeat labels; prefixes
remain unique within each boot. Never combine shorter ablation requests with
full-length qualification requests under one variant/boot identity.

For fresh diagnostic runs, add `--capture-output` to `python -m benchmark.benchmark`.
It defaults to off. Successful measured requests then save complete output text,
content/reasoning channels and raw choice deltas in `OUT/outputs/<key-hash>-<unique-id>.json`.
Each JSONL result references its artifact through `output_artifact`, relative to
`OUT`; concurrent requests each get their own reference. The artifact's `text`
matches the existing `output_sha256` definition (content or reasoning per delta),
while the separate channels and deltas retain both fields when they arrive together.
Artifacts are written after timing ends. Calibration and warm-up requests remain
unrecorded. Both modes of the follow-up helper enable capture automatically.

For the unchanged historical quality gates against the running candidate:

```bash
bash benchmark/run_decode_quality.sh benchmark/results/decode-quality
```

This uses the existing lm-eval environment: GSM8K `gsm8k_cot_llama` (150,
eight-shot chat turns), the checked-in custom MMLU-chat task (10 questions per
subject, 570 total), then the existing five-depth 200K needle check. Native
results and `summary.json` retain the exact metric names. Historical reference
scores are GSM8K 133/150 and MMLU 498/570; the task definitions must match before
comparing those values. Admission can move a route between the existing CPU
W4A8 and GPU W4A16 implementations, so full-service outputs need not be identical
despite unchanged expert selection and configured precision.

## Matched comparisons and selected-cell follow-ups

Run the CPU-only comparison after collecting the relevant records:

```bash
/home/nuko/.freetoken/venv/bin/python -m benchmark.compare_decode \
    benchmark/results/decode-round6-qualification
```

This writes `decode-comparison.json` and `decode-comparison.md` beside the input
JSONL. It pairs boot label, sampling mode, content, context and repetition, then
requires matching prompt SHA-256, equal output budgets and distinct actual server
instances. Duplicates, invalid rows and unmatched requests remain visible. Rates
use total emitted tokens divided by total decode time over the matched subset;
extra candidate repetitions cannot improve the paired aggregate. Per-cell and
per-boot ratios expose slower cells even when the aggregate improves. The tool
also records step latency, accepted tokens per step and known geometry changes;
missing geometry remains unavailable. These are descriptive comparisons, and
the command has no performance pass/fail exit gate.

After the main benchmark has stopped, the follow-up helper can collect five
fresh control repetitions in each of two boots for selected cells. Its cell
syntax is `sampling:kind:context`, with multiple cells separated by commas. The
following example selects only Japanese greedy at 253,952 input tokens:

```bash
# Validate the completed references and print a plan; no writes or server activity.
bash benchmark/run_decode_followup.sh --plan-only \
    benchmark/results/decode-followup-baseline \
    benchmark/results/decode-round6-qualification 'greedy:japanese:253952' 1,2

# Fresh baseline: original source snapshot and all five optimization controls off.
bash benchmark/run_decode_followup.sh \
    benchmark/results/decode-followup-baseline \
    benchmark/results/decode-round6-qualification 'greedy:japanese:253952' 1,2

# Plan an execution-only control using current source, with admission disabled.
FREETOKEN_FOLLOWUP_MODE=execution-only \
bash benchmark/run_decode_followup.sh --plan-only \
    benchmark/results/decode-followup-execution \
    benchmark/results/decode-round6-qualification 'greedy:japanese:253952' 1,2

FREETOKEN_FOLLOWUP_MODE=execution-only \
bash benchmark/run_decode_followup.sh \
    benchmark/results/decode-followup-execution \
    benchmark/results/decode-round6-qualification 'greedy:japanese:253952' 1,2
```

Use a fresh output directory for each execution; the helper refuses an existing
directory or occupied port and stops only the process group it starts. It loads
one control server per boot and visits selected cells in ascending context order.
The default port is 1920 (`FREETOKEN_PORT` overrides it). Both modes retain the
same single-request profile: MTP two to three drafts, hybrid experts, FP8 dense
weights/KV and float16 GDN state.

| Mode | Default variant | Runtime source | Optimization controls |
| --- | --- | --- | --- |
| `baseline` (default) | `baseline-followup` | `/tmp/ft-decode-baseline-lNjUhP/python` | All five off |
| `execution-only` | `execution-only-followup` | This repository's `python/` | Admission off; shared overlap, GDN strided norm and both QSA controls on |

Override source paths with `FREETOKEN_DECODE_BASELINE_PYTHON` or, for
execution-only, `FREETOKEN_DECODE_PYTHONPATH`. `FREETOKEN_FOLLOWUP_VARIANT` sets a
distinct label. Each boot stages the selected source under a space-free temporary
directory. `selection.json` retains selected reference keys/hashes, source path,
exact runtime controls, capture setting and limitations; the recorded controls
also drive the launch environment. Server logs retain each staged source path,
and client metadata records actual server IDs, command lines and environments.

Each cell has its own directory, for example
`OUT/cells/greedy-japanese-253952/`, containing `runs.jsonl`, per-boot metadata,
telemetry and complete `outputs/*.json` captures. A result's `output_artifact`
is relative to that cell directory. The helper automatically compares these new
records with the original combined records and writes `OUT/decode-comparison.*`.
Original measurements remain untouched. To repeat that comparison explicitly:

```bash
/home/nuko/.freetoken/venv/bin/python -m benchmark.compare_decode \
    benchmark/results/decode-round6-qualification \
    benchmark/results/decode-followup-execution/cells/greedy-japanese-253952 \
    --baseline execution-only-followup --candidate combined \
    --out-prefix benchmark/results/decode-followup-execution/decode-comparison
```

The helper requires all five eligible combined reference repetitions for each
requested boot/cell, with the expected nonces and no stored token-fit correction.
Fresh calibration must still produce the same prompt hashes; exclusions cannot
be repaired by relabeling a different repetition. Client warm-up runs before the
first cell of each boot by default. `FREETOKEN_FOLLOWUP_WARMUP=0` skips that warm-up
and records the choice; startup calibration and server-internal warm-up remain.

Selected-cell order changes expert-cache history relative to the full matrix,
and available GPU memory can change KV/expert-cache geometry. Inspect before/after
geometry and acceptance alongside throughput. These follow-ups reuse candidate
observations that helped select the cells, so they provide conditional evidence;
five repeats across two boots still give only two independent boot groups.
An execution-only difference can be attributed to admission only when its staged
runtime otherwise matches the combined runtime. Independent confirmation calls
for fresh measurements of both configurations with the same workload order.

## Fresh decode traces

Collect traces in a separate diagnostic run, after other benchmarks have stopped.
The existing profiler uses a server-wide verify-step counter: `skip:steps` counts
from executor creation across requests. For the current launcher and its standard
warm-up sequence, `180:6` places a six-step window safely inside the first long
code request rather than the warm-up requests. Use a fresh trace directory/run
label, and confirm start/end request boundaries in the server log:

```bash
FREETOKEN_SPEC_PROFILE=180:6 \
FREETOKEN_SPEC_PROFILE_DIR=/tmp/decode-profile-code \
bash benchmark/run_decode.sh decode-profile-code combined profile 1 4096 4096

# One fresh server boot above produces one spec-trace-<pid>.json in this directory.
/home/nuko/.freetoken/venv/bin/python -m benchmark.profile_decode \
    /tmp/decode-profile-code/spec-trace-*.json \
    --label combined-code-4096-fresh --steps 6 \
    --out-prefix benchmark/results/decode-profile-code/trace-activity
```

Changing warm-up, earlier requests or output length changes the counter's
relationship to request boundaries. A dedicated single-target client can use
`--no-warmup` with a separately chosen capture window. Supply `--steps` from the
actual capture configuration/log; the analyzer does not infer layer counts.
Profiling starts without an initial device synchronization and drains at the end,
so boundary normalization is approximate. Profiled requests must remain separate
from unprofiled throughput claims.

`profile_decode.py` is CPU-only. It reads every recorded `kernel`, `gpu_memcpy`
and `gpu_memset` complete event across all streams, separating devices. Its JSON
retains the trace hash, full per-name inventory, per-stream totals, duration sums,
activity unions, observed span/gaps and bucket-only activity. Duration sums
double-count concurrent work; per-stream times cannot simply be added to obtain
wall time. Union activity measures recorded activity, not occupancy or a critical
path, and bucket-only time is not guaranteed recoverable time.

Buckets use kernel names. The transfer bucket includes expert gathers and device
copies as well as host transfers, so its duration alone cannot establish PCIe
traffic or bandwidth. Host CUDA launch/synchronization calls have separate
per-thread unions to avoid adding nested driver/runtime calls. Their inclusive
durations can include GPU backpressure and are not exposed CPU overhead. Gaps
may include dependencies or unrecorded operations. Interpret traces alongside
source/control metadata, request counters, output captures and fresh unprofiled
timing; the analyzer does not infer acceptance or throughput from trace durations.
