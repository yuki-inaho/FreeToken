#!/usr/bin/env python3
"""Needle-in-a-haystack retrieval against a running FreeToken server.

Builds a synthetic document of roughly ``--tokens`` tokens (Qwen-style ~4 chars/token filler
of numbered passages), hides one fact ("The secret code word is X") at each requested depth
fraction, and asks the model to repeat the code word. Prints, per depth, the actual prompt
token count (from the server's usage block), TTFT, whether the answer contains the word, and
the decode rate. Use it to compare KV configurations (bf16 vs fp8, 128K vs 256K) at depth:

    scripts/needle-check.py --tokens 200000 --depths 0.1,0.3,0.5,0.7,0.9
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
import urllib.request

FILLER = (
    "The harbour town kept its records in a long ledger: tides, cargo, the names of "
    "captains, and the price of rope. Each season the clerk added a page, and each page "
    "repeated the same careful columns, so that anyone reading later could follow the years "
    "without a guide. "
)
WORDS = ["lantern", "granite", "saffron", "meridian", "cobalt", "juniper", "velvet", "quartz"]


def build_prompt(target_tokens: int, depth: float, word: str, seed: int) -> str:
    rng = random.Random(seed)
    # ~4 chars per token; the server reports the true count, so this only needs to land near.
    n_para = max(1, int(target_tokens * 4 / len(FILLER)))
    paras = [f"[{i}] {FILLER}" for i in range(n_para)]
    needle = f"[note] The secret code word for this document is {word}. Remember it."
    paras.insert(int(depth * n_para), needle)
    rng.shuffle(WORDS)
    return (
        "Read the document below carefully. At the end you will be asked one question.\n\n"
        + "\n".join(paras)
        + "\n\nQuestion: What is the secret code word stated in the document? "
        "Answer with the single word only."
    )


def ask(base: str, model: str, prompt: str) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 24,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        f"{base}/chat/completions", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    first = None
    text = []
    usage = None
    with urllib.request.urlopen(req, timeout=3600) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            ev = json.loads(line[6:])
            if ev.get("error"):
                raise SystemExit(f"server error: {ev['error']}")
            for ch in ev.get("choices") or []:
                piece = (ch.get("delta") or {}).get("content")
                if piece:
                    text.append(piece)
                    if first is None:
                        first = time.perf_counter()
            if ev.get("usage"):
                usage = ev["usage"]
    wall = time.perf_counter() - started
    ttft = (first - started) if first else float("nan")
    comp = (usage or {}).get("completion_tokens", 0)
    decode = (comp - 1) / (wall - ttft) if first and comp > 1 and wall > ttft else float("nan")
    return dict(text="".join(text), prompt_tokens=(usage or {}).get("prompt_tokens"), ttft=ttft, decode=decode)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("FREETOKEN_MODEL_NAME", "Qwen3.8-Flash-Next-NVFP4"))
    ap.add_argument("--base", default=os.environ.get("FREETOKEN_API_BASE", "http://127.0.0.1:1919/v1"))
    ap.add_argument("--tokens", type=int, default=100000)
    ap.add_argument("--depths", default="0.1,0.3,0.5,0.7,0.9")
    args = ap.parse_args()
    depths = [float(d) for d in args.depths.split(",")]
    hits = 0
    for i, depth in enumerate(depths):
        word = WORDS[i % len(WORDS)]
        res = ask(args.base, args.model, build_prompt(args.tokens, depth, word, seed=i))
        ok = word in res["text"].lower()
        hits += ok
        print(
            f"depth {depth:.1f}: prompt={res['prompt_tokens']} tokens ttft={res['ttft']:.1f}s "
            f"decode={res['decode']:.1f} tok/s -> {'OK ' if ok else 'MISS'} ({res['text'].strip()[:40]!r}, want {word!r})",
            flush=True,
        )
    print(f"retrieved {hits}/{len(depths)}")


if __name__ == "__main__":
    main()
