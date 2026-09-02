#!/usr/bin/env python3
"""Per-content decode throughput against a running FreeToken server.

Streams one long completion per content type (the MoE benchmark question, English prose,
code, Japanese) with ``ignore_eos`` so every run decodes the full budget, and reports the
decode rate after the first token (wall clock, the number the user experiences) next to the
server's own ``response_token/s``. Optionally snapshots ``/v1/stats`` around each run.

    scripts/bench-decode.py --model DeepSeek-V4-Flash-0731 --tokens 512 --temperature 1.0
    scripts/bench-decode.py --kinds bench,code --runs 2

Env: FREETOKEN_API_BASE (default http://127.0.0.1:1919/v1).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

PROMPTS = {
    "bench": (
        "Explain mixture-of-experts inference in depth, including routing, expert caching, "
        "quantization, GPU memory, PCIe bandwidth, scheduling, and performance tradeoffs. "
        "Write at least 2,000 words."
    ),
    "english": (
        "Write a long, reflective essay about how coastal towns adapt to changing weather, "
        "with concrete examples, anecdotes, and a clear narrative arc. At least 1,500 words."
    ),
    "code": (
        "Write a complete, well-documented Python module implementing an LRU cache with TTL "
        "support, thread safety, statistics, and a comprehensive pytest test suite. Include "
        "type hints and docstrings everywhere. Then add a CLI demo."
    ),
    "japanese": (
        "日本の四季と食文化の関係について、具体的な料理や行事を挙げながら、"
        "できるだけ詳しく長文で説明してください。2000字以上でお願いします。"
    ),
}


def stats(base: str) -> dict | None:
    try:
        with urllib.request.urlopen(f"{base}/stats", timeout=10) as r:
            return json.loads(r.read().decode())
    except Exception:
        return None


def run_one(base: str, model: str, kind: str, tokens: int, temperature: float, tag: str):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": f"[{tag}] {PROMPTS[kind]}"}],
        "temperature": temperature,
        "max_tokens": tokens,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    first = None
    usage = None
    n_chunks = 0
    with urllib.request.urlopen(req, timeout=3600) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            ev = json.loads(line[6:])
            if ev.get("error"):
                raise SystemExit(f"server error: {ev['error']}")
            for ch in ev.get("choices") or []:
                if (ch.get("delta") or {}).get("content"):
                    n_chunks += 1
                    if first is None:
                        first = time.perf_counter()
            if ev.get("usage"):
                usage = ev["usage"]
    wall = time.perf_counter() - started
    if usage is None:
        raise SystemExit("no usage block in the stream")
    comp = usage["completion_tokens"]
    ttft = (first - started) if first else float("nan")
    decode = (comp - 1) / (wall - ttft) if first and wall > ttft else float("nan")
    return dict(
        kind=kind, prompt_tokens=usage.get("prompt_tokens"), completion_tokens=comp,
        ttft=ttft, wall=wall, decode_tps=decode,
        server_tps=usage.get("response_token/s"), chunks=n_chunks,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("FREETOKEN_MODEL_NAME", "DeepSeek-V4-Flash-0731"))
    ap.add_argument("--base", default=os.environ.get("FREETOKEN_API_BASE", "http://127.0.0.1:1919/v1"))
    ap.add_argument("--kinds", default="bench,english,code,japanese")
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--stats", action="store_true", help="print /v1/stats keys mentioning cache/moe after each run")
    args = ap.parse_args()
    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]
    print(f"model={args.model} tokens={args.tokens} temperature={args.temperature} runs={args.runs}")
    rows = []
    for r in range(args.runs):
        for kind in kinds:
            res = run_one(args.base, args.model, kind, args.tokens, args.temperature, f"run{r}")
            rows.append(res)
            print(
                f"{kind:9s} run{r}: prompt={res['prompt_tokens']} out={res['completion_tokens']} "
                f"ttft={res['ttft']:.2f}s decode={res['decode_tps']:.1f} tok/s "
                f"(server {res['server_tps']:.1f} tok/s over the whole response)",
                flush=True,
            )
            if args.stats:
                s = stats(args.base)
                if s:
                    keep = {k: v for k, v in s.items() if any(w in k.lower() for w in ("cache", "moe", "hit", "expert", "spec"))}
                    print("   stats:", json.dumps(keep)[:600])
    if len(rows) > 1:
        by = {}
        for res in rows:
            by.setdefault(res["kind"], []).append(res["decode_tps"])
        print("summary decode tok/s:", ", ".join(f"{k}={sum(v)/len(v):.1f}" for k, v in by.items()))


if __name__ == "__main__":
    sys.exit(main())
