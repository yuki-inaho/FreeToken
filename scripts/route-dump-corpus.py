#!/usr/bin/env python3
"""Assemble and capture a route-dump corpus for the expert substitution table.

The table is conditioned on the tokens that route to each expert, so the corpus decides what it
generalises to. The existing M1 dumps are ~10K tokens from one session; this builds a wider one.

Three modes:

  ``build``    assemble chunks from real local text -- this repository's Python, C++ and
               Markdown (code and technical prose), and the MMLU / IFEval questions already in
               the HuggingFace cache (academic prose, maths, instruction-following).
  ``generate`` ask the running server for long outputs in the styles the local corpus lacks
               (Japanese above all, plus creative prose and chain-of-thought maths) and append
               them. Model-generated text is the right proxy here: it is what decode actually
               routes over, and the M1 harness used replayed generation for the same reason.
  ``capture``  send every chunk to a server booted with FREETOKEN_ROUTE_DUMP and
               FREETOKEN_ROUTE_DUMP_X=1 as a one-token completion, so each chunk is dumped as
               exactly one prefill forward.

Capture the dumps with residency routing OFF: the dumper records the raw router logits, but
``block_input`` at layer L reflects whatever the layers below it computed, so a substituting arm
would poison the corpus with its own perturbation.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# ~4 chars/token for English and code; the cap stays under FREETOKEN_ROUTE_DUMP_MAX_T so every
# chunk is dumped as one forward rather than skipped.
CHARS_PER_TOKEN = 3.6


def _chunk(text: str, target_tokens: int) -> list[str]:
    n = int(target_tokens * CHARS_PER_TOKEN)
    out, buf = [], []
    size = 0
    for line in text.splitlines(keepends=True):
        buf.append(line)
        size += len(line)
        if size >= n:
            out.append("".join(buf))
            buf, size = [], 0
    if size > n // 4:
        out.append("".join(buf))
    return out


def from_repo(target_tokens: int) -> list[str]:
    """Real code and technical prose: this repository."""
    pats = ["python/freetoken/**/*.py", "python/freetoken/kernel/csrc/**/*.cpp",
            "python/freetoken/kernel/csrc/**/*.h", "python/freetoken/kernel/csrc/**/*.cuh",
            "scripts/*.py", "scripts/*.sh", "tests/**/*.py",
            "*.md", "docs/*.md", "python/freetoken/tier3/*.md", "benchmark/*.md"]
    chunks = []
    for pat in pats:
        for p in sorted(glob.glob(os.path.join(REPO, pat), recursive=True)):
            try:
                t = open(p, encoding="utf-8", errors="ignore").read()
            except OSError:
                continue
            if len(t) < 800:
                continue
            chunks += _chunk(t, target_tokens)
    return chunks


def from_hf_datasets(target_tokens: int, per_subject: int = 12) -> list[str]:
    """Academic prose, maths and instruction-following from the local HF arrow cache.

    Read directly with pyarrow the way scripts/quality-gate.py does, so this needs no network
    and no ``datasets`` package.
    """
    try:
        import pyarrow.ipc as ipc
    except ImportError:
        print("  (no pyarrow in this interpreter; skipping the HF datasets)")
        return []
    root = os.path.expanduser("~/.cache/huggingface/datasets")
    texts: list[str] = []
    for arrow in sorted(glob.glob(f"{root}/**/*.arrow", recursive=True)):
        try:
            with open(arrow, "rb") as f:
                tbl = ipc.open_stream(f).read_all()
        except Exception:
            continue
        cols = {c.lower(): c for c in tbl.column_names}
        col = next((cols[k] for k in ("question", "prompt", "text", "sentence") if k in cols), None)
        if col is None:
            continue
        vals = tbl.column(col).to_pylist()[:per_subject]
        choices = tbl.column(cols["choices"]).to_pylist()[:per_subject] if "choices" in cols else None
        for i, q in enumerate(vals):
            if not isinstance(q, str) or len(q) < 40:
                continue
            body = q
            if choices and isinstance(choices[i], list):
                body += "\n" + "\n".join(f"{chr(65+j)}. {c}" for j, c in enumerate(choices[i]))
            texts.append(body)
    random.Random(0).shuffle(texts)
    # pack the short questions into chunks so each is one worthwhile prefill
    return _chunk("\n\n".join(texts), target_tokens)


GEN_PROMPTS = [
    ("japanese", "日本の四季と食文化の関係について、具体的な料理や年中行事を挙げながら、"
                 "できるだけ詳しく長文で説明してください。3000字以上でお願いします。"),
    ("japanese", "機械学習における注意機構の仕組みを、数式を交えて日本語で詳しく解説してください。"
                 "初学者にも分かるように、具体例を多く含めて3000字以上で書いてください。"),
    ("japanese", "京都の町家建築の歴史と構造について、専門的な用語を使いながら詳述してください。3000字以上。"),
    ("creative", "Write a long, atmospheric short story about a lighthouse keeper who begins "
                 "receiving letters addressed to someone who died a century ago. At least 2,500 words."),
    ("creative", "Write a detailed, opinionated essay on why most software rewrites fail, with "
                 "concrete historical examples and counterexamples. At least 2,500 words."),
    ("math", "Work through, step by step and in full detail, a proof that the harmonic series "
             "diverges, then three different proofs of the irrationality of sqrt(2), then a "
             "careful derivation of the closed form for the sum of the first n cubes."),
    ("math", "Solve and fully explain fifteen varied quantitative reasoning problems: rates, "
             "combinatorics, probability, geometry and number theory. Show every step."),
    ("chat", "I'm debugging a slow database query that only degrades under production load. "
             "Walk me through your whole diagnostic process in conversational detail, asking "
             "and then answering the questions you would ask me."),
    ("code", "Write a complete, heavily commented Rust implementation of a lock-free MPMC queue, "
             "then a C++ equivalent, then explain the memory-ordering differences between them."),
    ("code", "Write a full TypeScript React application for a kanban board with drag and drop, "
             "including state management, tests and CSS. Explain each file."),
]


def post(base: str, payload: dict, timeout: float = 1800.0) -> dict:
    req = urllib.request.Request(
        f"{base}/chat/completions", method="POST",
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["build", "generate", "capture"], required=True)
    ap.add_argument("--corpus", required=True, help="the corpus JSON (built, then captured)")
    ap.add_argument("--base", default="http://127.0.0.1:1919/v1")
    ap.add_argument("--model", default="Qwen3.8-Flash-Next-NVFP4")
    ap.add_argument("--chunk-tokens", type=int, default=2600,
                    help="target tokens per chunk; must stay under FREETOKEN_ROUTE_DUMP_MAX_T")
    ap.add_argument("--target-tokens", type=int, default=56000, help="corpus size to keep")
    ap.add_argument("--gen-tokens", type=int, default=1400, help="tokens per generated sample")
    args = ap.parse_args()

    if args.mode == "build":
        repo = from_repo(args.chunk_tokens)
        hf = from_hf_datasets(args.chunk_tokens)
        print(f"repo chunks {len(repo)}, dataset chunks {len(hf)}")
        rng = random.Random(1234)
        rng.shuffle(repo)
        # keep the mix roughly even between real code/prose and question-style text
        keep_each = max(1, int(args.target_tokens / args.chunk_tokens / 2))
        chunks = repo[:keep_each] + hf[:keep_each]
        rng.shuffle(chunks)
        json.dump(chunks, open(args.corpus, "w"))
        est = sum(len(c) for c in chunks) / CHARS_PER_TOKEN
        print(f"wrote {len(chunks)} chunks, ~{est:,.0f} tokens -> {args.corpus}")
        return

    if args.mode == "generate":
        chunks = json.load(open(args.corpus)) if os.path.exists(args.corpus) else []
        for i, (kind, prompt) in enumerate(GEN_PROMPTS):
            out = post(args.base, {"model": args.model,
                                   "messages": [{"role": "user", "content": prompt}],
                                   "temperature": 0.8, "top_p": 0.95,
                                   "max_tokens": args.gen_tokens})
            text = out["choices"][0]["message"]["content"] or ""
            chunks += _chunk(text, args.chunk_tokens)
            print(f"  [{i+1}/{len(GEN_PROMPTS)}] {kind}: {len(text)} chars", flush=True)
        json.dump(chunks, open(args.corpus, "w"))
        print(f"corpus now {len(chunks)} chunks -> {args.corpus}")
        return

    chunks = json.load(open(args.corpus))
    print(f"capturing {len(chunks)} chunks (one prefill dump each)")
    for i, c in enumerate(chunks):
        # max_tokens 1: the point is the prefill forward, which is what the dumper records
        req = urllib.request.Request(
            f"{args.base}/completions", method="POST",
            data=json.dumps({"model": args.model, "prompt": c, "max_tokens": 1,
                             "temperature": 0.0}).encode(),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=1800) as r:
                json.loads(r.read())
        except Exception as e:  # noqa: BLE001 -- one bad chunk must not lose the run
            print(f"  [{i+1}/{len(chunks)}] FAILED: {e!r}", flush=True)
            continue
        if (i + 1) % 5 == 0 or i == 0:
            print(f"  [{i+1}/{len(chunks)}]", flush=True)
    print("capture done")


if __name__ == "__main__":
    main()
