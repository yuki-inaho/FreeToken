#!/usr/bin/env python3
"""API 経由でデコード tok/s を計測する (pixi run bench)。

環境変数:
  FT_URL           既定 http://127.0.0.1:1919/v1/chat/completions
  FT_MODEL         既定 DeepSeek-V4-Flash-0731
  FT_BENCH_RUNS    既定 3
  FT_BENCH_TOKENS  既定 128 (生成上限)
  FT_BENCH_PROMPT  既定の短文プロンプト

出力は「wall 時間ベースの tok/s」(プロンプト処理込み・短プロンプトなので影響は小)。
厳密なデコード速度はサーバーログの decode 行、または長い生成で近似すること。
"""
import json
import os
import statistics
import time
import urllib.request

URL = os.environ.get("FT_URL", "http://127.0.0.1:1919/v1/chat/completions")
MODEL = os.environ.get("FT_MODEL", "DeepSeek-V4-Flash-0731")
RUNS = int(os.environ.get("FT_BENCH_RUNS", "3"))
TOKENS = int(os.environ.get("FT_BENCH_TOKENS", "128"))
PROMPT = os.environ.get("FT_BENCH_PROMPT", "Write a short paragraph about GPUs.")


def one_run(i: int) -> float:
    body = json.dumps(
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": TOKENS,
            "temperature": 0,
        }
    ).encode()
    req = urllib.request.Request(
        URL, data=body, headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=3600) as resp:
        r = json.load(resp)
    dt = time.time() - t0
    n = r.get("usage", {}).get("completion_tokens", 0)
    rate = n / dt if dt > 0 else 0.0
    print(f"run {i + 1}: {n} tok in {dt:.1f}s -> {rate:.2f} tok/s (wall)")
    return rate


rates = [one_run(i) for i in range(RUNS)]
print(f"median: {statistics.median(rates):.2f} tok/s (n={RUNS}, {TOKENS} max tok, greedy)")
