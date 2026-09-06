#!/usr/bin/env python3
"""How often does residency substitution change the token the model emits?

The N x N substitutability matrix says a substituted expert computes something almost
orthogonal to the one it replaced, and that the MoE block's output moves 14-31 %. What that
means downstream is a separate question, and the GSM8K/MMLU gates cannot answer it: they scored
byte-identically with and without the feature, which only says the questions sit far from the
decision boundary.

This measures the damage rate directly, under TEACHER FORCING so the two arms always see the
same context. For each of a reference model's own greedy continuations, replay every prefix on
the arm under test and ask for one token. The disagreement rate is the discrete analogue of a
KL divergence -- P(argmax moves) -- and unlike free-running divergence it does not compound, so
it is a per-token damage rate rather than a story about how fast two texts drift apart.

Uses /v1/completions (raw prefix continuation) so the prefix is exact. Note that on the NVMe
expert tier EVERY prefill re-materialises every expert of every layer, so a one-token request
costs the same ~4.6 s as a long one and the prefix cache does not help -- which is why this
samples positions along each continuation rather than teacher-forcing all of them.

  # arm 1, the reference (routing off): record its own greedy continuations
  scripts/tier3-substitution-divergence.py --mode record --out ref.json
  # arm 2, the arm under test: replay those prefixes and count disagreements
  scripts/tier3-substitution-divergence.py --mode replay --ref ref.json --out test.json
"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request

PROMPTS = [
    "The following is a proof that there are infinitely many prime numbers.",
    "def quicksort(arr):\n    \"\"\"Sort a list in place using the Lomuto partition scheme.\"\"\"\n",
    "In 1919, the Treaty of Versailles was signed. Its economic consequences were",
    "The mitochondrion is often described as the powerhouse of the cell because",
    "Q: A train leaves Boston at 60 mph and another leaves New York at 40 mph.\nA: Let me work",
    "Dear Dr. Menzies,\n\nThank you for your careful review of our manuscript. In response",
    "The difference between a monad and an applicative functor is best explained by",
    "import numpy as np\n\ndef kalman_filter(z, Q, R):\n",
    "Climate models disagree most about cloud feedback because",
    "The Japanese tea ceremony, or chanoyu, developed from",
    "SELECT customer_id, SUM(total) FROM orders WHERE",
    "A skeptic might object that consciousness cannot be reduced to computation. The strongest",
]


def post(base: str, payload: dict, timeout: float = 600.0) -> dict:
    req = urllib.request.Request(
        f"{base}/completions", method="POST",
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def one_token(base: str, model: str, prompt: str) -> str:
    """The single greedy continuation token for this exact prefix."""
    out = post(base, {"model": model, "prompt": prompt, "max_tokens": 1,
                      "temperature": 0.0, "stream": False})
    return out["choices"][0]["text"]


def stream_tokens(base: str, model: str, prompt: str, n: int) -> list[str]:
    """One greedy continuation, returned token by token.

    Streaming is what makes the token boundaries visible: generating n tokens in one call and
    splitting the text afterwards would guess at them.
    """
    req = urllib.request.Request(
        f"{base}/completions", method="POST",
        data=json.dumps({"model": model, "prompt": prompt, "max_tokens": n,
                         "temperature": 0.0, "stream": True,
                         "ignore_eos": True}).encode(),
        headers={"Content-Type": "application/json"},
    )
    toks = []
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            try:
                ch = json.loads(body)["choices"][0]
            except Exception:
                continue
            t = ch.get("text", "")
            if t:
                toks.append(t)
    return toks


def schedule(n: int, k: int) -> list[int]:
    """``k`` positions spread over a continuation of ``n`` tokens."""
    if n <= k:
        return list(range(n))
    step = n / k
    return sorted({min(n - 1, int(i * step)) for i in range(k)})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["record", "replay"], required=True)
    ap.add_argument("--base", default="http://127.0.0.1:1919/v1")
    ap.add_argument("--model", default="Qwen3.8-Flash-Next-NVFP4")
    ap.add_argument("--steps", type=int, default=96, help="tokens per reference continuation")
    ap.add_argument("--positions", type=int, default=12,
                    help="positions sampled per prompt on the arm under test")
    ap.add_argument("--prompts", type=int, default=len(PROMPTS))
    ap.add_argument("--ref", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    prompts = PROMPTS[: args.prompts]

    if args.mode == "record":
        rec = []
        for pi, p in enumerate(prompts):
            toks = stream_tokens(args.base, args.model, p, args.steps)
            rec.append({"prompt": p, "tokens": toks})
            print(f"  [{pi + 1}/{len(prompts)}] {len(toks)} tokens", flush=True)
        json.dump(rec, open(args.out, "w"))
        print(f"recorded {sum(len(r['tokens']) for r in rec)} reference tokens -> {args.out}")
        return

    ref = json.load(open(args.ref))
    total = agree = 0
    per_prompt = []
    for pi, r in enumerate(ref):
        toks = r["tokens"]
        pos = schedule(len(toks), args.positions)
        n_dis = 0
        for p in pos:
            # teacher forcing: the arm under test sees exactly the reference's own prefix
            got = one_token(args.base, args.model, r["prompt"] + "".join(toks[:p]))
            total += 1
            if got == toks[p]:
                agree += 1
            else:
                n_dis += 1
        per_prompt.append({"disagreements": n_dis, "of": len(pos)})
        print(f"  [{pi + 1}/{len(ref)}] {n_dis}/{len(pos)} sampled positions differ", flush=True)
    rate = 1.0 - agree / max(1, total)
    out = {"positions": total, "disagreements": total - agree, "rate": rate,
           "per_prompt": per_prompt}
    json.dump(out, open(args.out, "w"))
    print(f"\nper-token disagreement rate: {rate:.3%}  ({total - agree} of {total} positions)")


if __name__ == "__main__":
    main()
