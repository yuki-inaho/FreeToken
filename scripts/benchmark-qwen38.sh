#!/usr/bin/env bash

set -euo pipefail

readonly FT_VENV="${FREETOKEN_VENV:-${FREETOKEN_HOME:-$HOME/.freetoken}/venv}"
readonly PYTHON="${FREETOKEN_PYTHON:-$FT_VENV/bin/python}"
readonly API_BASE="${FREETOKEN_API_BASE:-http://127.0.0.1:1919/v1}"
readonly MODEL_NAME="${FREETOKEN_MODEL_NAME:-Qwen3.8-Flash-Next-NVFP4}"
readonly OUTPUT_TOKENS="${FREETOKEN_BENCH_TOKENS:-1024}"
readonly RUNS="${FREETOKEN_BENCH_RUNS:-2}"
readonly PROMPT_TOKENS="${FREETOKEN_BENCH_PROMPT_TOKENS:-0}"

if [[ ! -x "$PYTHON" ]]; then
  printf 'FreeToken Python is not installed at %s\n' "$PYTHON" >&2
  exit 1
fi

if [[ ! "$OUTPUT_TOKENS" =~ ^[1-9][0-9]*$ || ! "$RUNS" =~ ^[1-9][0-9]*$ ]]; then
  printf 'FREETOKEN_BENCH_TOKENS and FREETOKEN_BENCH_RUNS must be positive integers.\n' >&2
  exit 1
fi

if [[ ! "$PROMPT_TOKENS" =~ ^[0-9]+$ ]]; then
  printf 'FREETOKEN_BENCH_PROMPT_TOKENS must be a non-negative integer.\n' >&2
  exit 1
fi

export FREETOKEN_BENCH_API="$API_BASE/chat/completions"
export FREETOKEN_BENCH_MODEL="$MODEL_NAME"
export FREETOKEN_BENCH_OUTPUT_TOKENS="$OUTPUT_TOKENS"
export FREETOKEN_BENCH_RUN_COUNT="$RUNS"
export FREETOKEN_BENCH_PROMPT_TOKEN_TARGET="$PROMPT_TOKENS"

"$PYTHON" - <<'PY'
import json
import os
import time
import urllib.error
import urllib.request

url = os.environ["FREETOKEN_BENCH_API"]
model = os.environ["FREETOKEN_BENCH_MODEL"]
output_tokens = int(os.environ["FREETOKEN_BENCH_OUTPUT_TOKENS"])
runs = int(os.environ["FREETOKEN_BENCH_RUN_COUNT"])
prompt_target = int(os.environ["FREETOKEN_BENCH_PROMPT_TOKEN_TARGET"])

question = (
    "Explain mixture-of-experts inference in depth, including routing, "
    "expert caching, quantization, GPU memory, PCIe bandwidth, scheduling, "
    "and performance tradeoffs. Write at least 2,000 words."
)

# One labeled paragraph of filler measures about 82 tokens with the Qwen
# tokenizer. The actual prompt size is reported from the server's usage
# block, so the heuristic only needs to land near the target.
filler_paragraph = (
    "Datacenter inference balances latency, throughput, memory capacity, and "
    "interconnect bandwidth. Batch schedulers interleave prefill and decode "
    "phases while paged caches recycle memory across requests. Quantized "
    "weights travel across PCIe into device buffers, kernels overlap copies "
    "with compute, and routers spread tokens over experts to keep utilization "
    "high without starving the cache hierarchy of reusable working sets. "
)

def build_messages(run: int) -> list[dict]:
    if prompt_target <= 0:
        return [{"role": "user", "content": question}]
    paragraphs = max(1, prompt_target // 82)
    context = "\n\n".join(
        f"[run {run} passage {i}] {filler_paragraph}" for i in range(paragraphs)
    )
    content = (
        "Here is background reading; the question follows after it.\n\n"
        f"{context}\n\nNow, referring to the reading only where helpful: {question}"
    )
    return [{"role": "user", "content": content}]

target_note = f", ~{prompt_target} prompt tokens" if prompt_target > 0 else ""
print(f"Benchmarking {model}: {output_tokens} requested output tokens{target_note}, {runs} runs")
for run in range(1, runs + 1):
    payload = {
        "model": model,
        # A run-unique prefix defeats radix prefix caching so every run
        # measures a genuine prefill at the requested depth.
        "messages": build_messages(run),
        "temperature": 0,
        "max_tokens": output_tokens,
        # Force a full-length decode so runs remain comparable even if the model
        # happens to emit EOS early. This affects the benchmark only.
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    first_token_at = None
    usage = None
    try:
        with urllib.request.urlopen(request, timeout=1800) as response:
            for raw_line in response:
                line = raw_line.decode().strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                event = json.loads(line[6:])
                if event.get("error"):
                    raise SystemExit(f"FreeToken returned an error: {event['error']}")
                if first_token_at is None:
                    for choice in event.get("choices") or []:
                        if (choice.get("delta") or {}).get("content"):
                            first_token_at = time.perf_counter()
                            break
                if event.get("usage"):
                    usage = event["usage"]
    except urllib.error.HTTPError as error:
        raise SystemExit(
            f"FreeToken rejected the request ({error.code}): {error.read().decode(errors='replace')[:500]}"
        ) from error
    except urllib.error.URLError as error:
        raise SystemExit(f"Could not reach FreeToken at {url}: {error}") from error

    wall_seconds = time.perf_counter() - started
    if usage is None or "response_time" not in usage:
        raise SystemExit(
            f"FreeToken did not return the requested streaming usage block (got: {usage})"
        )
    completion_tokens = usage["completion_tokens"]
    wall_tps = completion_tokens / wall_seconds
    line = (
        f"run {run}: {completion_tokens} tokens, "
        f"response={usage['response_time']:.3f}s/{usage['response_token/s']:.2f} tok/s, "
        f"wall={wall_seconds:.3f}s/{wall_tps:.2f} tok/s"
    )
    if first_token_at is not None:
        ttft = first_token_at - started
        prompt_tokens = usage.get("prompt_tokens", 0)
        decode_seconds = wall_seconds - ttft
        line += f", prompt={prompt_tokens} tok, ttft={ttft:.3f}s"
        if prompt_tokens and ttft > 0:
            line += f" ({prompt_tokens / ttft:.0f} tok/s prefill)"
        if decode_seconds > 0:
            line += f", decode={completion_tokens / decode_seconds:.2f} tok/s"
    print(line)
PY
