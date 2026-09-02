"""GSM8K-150 (8-shot CoT, greedy) and MMLU-chat (N/subject) against a running server.

Replicates the methodology BENCH.md records for the project's quality gates. It is NOT lm_eval,
so absolute scores need not match the recorded 88.67 / 87.37 exactly -- what it is for is the
A/B: the identical harness against two server configurations, where the DELTA is the signal.
Reads the HF dataset cache directly (pyarrow), so it needs no network and no `datasets`.
"""
import argparse, concurrent.futures as cf, glob, json, os, re, urllib.request
import pyarrow.ipc as ipc

CACHE = os.path.expanduser("~/.cache/huggingface/datasets")


def _arrow(path):
    with open(path, "rb") as f:
        return ipc.open_stream(f).read_all().to_pylist()


def gsm8k(split):
    p = glob.glob(f"{CACHE}/openai___gsm8k/main/*/*/gsm8k-{split}.arrow")
    return _arrow(sorted(p)[0])


def mmlu_subjects():
    out = {}
    for d in sorted(glob.glob(f"{CACHE}/cais___mmlu/*/*/*/mmlu-test.arrow")):
        out[d.split("/cais___mmlu/")[1].split("/")[0]] = d
    return out


def ask(base, model, messages, max_tokens, timeout=600):
    body = json.dumps({"model": model, "messages": messages, "max_tokens": max_tokens,
                       "temperature": 0, "top_p": 1}).encode()
    req = urllib.request.Request(base + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)["choices"][0]["message"]["content"]


_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def last_number(s):
    m = _NUM.findall(s.replace("$", ""))
    return m[-1].replace(",", "").rstrip(".") if m else None


def run_gsm8k(base, model, n, shots):
    train, test = gsm8k("train"), gsm8k("test")[:n]
    fewshot = "".join(f"Question: {r['question']}\nAnswer: {r['answer']}\n\n"
                      for r in train[:shots])
    def one(r):
        prompt = (fewshot + f"Question: {r['question']}\nAnswer:")
        out = ask(base, model, [{"role": "user", "content": prompt}], 512)
        gold = r["answer"].split("####")[-1].strip().replace(",", "")
        got = last_number(out.split("####")[-1] if "####" in out else out)
        return got == gold
    with cf.ThreadPoolExecutor(4) as ex:
        hits = list(ex.map(one, test))
    return sum(hits), len(hits)


LETTERS = "ABCD"


def run_mmlu(base, model, per_subject):
    subs = mmlu_subjects()
    items = []
    for name, path in subs.items():
        for r in _arrow(path)[:per_subject]:
            items.append((name, r))
    def one(it):
        _, r = it
        ch = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(r["choices"]))
        prompt = (f"{r['question']}\n{ch}\n\n"
                  "Answer with the single letter of the correct choice and nothing else.")
        out = ask(base, model, [{"role": "user", "content": prompt}], 8).strip()
        m = re.search(r"[ABCD]", out.upper())
        return m is not None and LETTERS.index(m.group()) == int(r["answer"])
    with cf.ThreadPoolExecutor(8) as ex:
        hits = list(ex.map(one, items))
    return sum(hits), len(hits), len(subs)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:1919/v1")
    ap.add_argument("--model", default="Qwen3.8-Flash-Next-NVFP4")
    ap.add_argument("--gsm8k", type=int, default=150)
    ap.add_argument("--shots", type=int, default=8)
    ap.add_argument("--mmlu-per-subject", type=int, default=10)
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    gh, gn = run_gsm8k(a.base, a.model, a.gsm8k, a.shots)
    mh, mn, ns = run_mmlu(a.base, a.model, a.mmlu_per_subject)
    print(f"  {a.label:>10}  GSM8K-{gn} ({a.shots}-shot CoT, greedy): {100*gh/gn:5.2f}%  "
          f"({gh}/{gn})   MMLU-chat {mn}q/{ns} subjects: {100*mh/mn:5.2f}%  ({mh}/{mn})")
