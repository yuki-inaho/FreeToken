"""Deterministic benchmark prompts with exact token counts.

Two workload families, both seeded and hash-recorded so anyone can rebuild the exact bytes:

* ``synthetic``  -- numbered passages of template sentences with seeded word substitution.
  Natural-looking (no repeated single token, no pathological sequence) but content-free; used
  for exact-length performance tests.
* ``realistic``  -- a mixed document: prose sections, Python code blocks, Markdown tables and
  bullet lists, JSON configuration fragments and log excerpts, cycled deterministically.

``fit_prompt`` grows or trims a document until the *server-side* prompt length equals the
requested token count exactly: text tokens (measured with the checkpoint's tokenizer) plus the
chat-template overhead measured against the live server (``calibrate_overhead`` in
``benchmark.py``). Every prompt starts with a run-unique nonce line so no two measured requests
share a radix-cache prefix (the "cold prefix" guarantee), and every prompt's SHA-256 is recorded.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

# --------------------------------------------------------------------------------------------
# vocabulary (kept small on purpose: the goal is realistic token statistics, not literature)
# --------------------------------------------------------------------------------------------

_NOUNS = (
    "harbour", "ledger", "scheduler", "bandwidth", "cache", "granite", "meridian", "cargo",
    "compiler", "lantern", "protocol", "dataset", "reservoir", "orchard", "kernel", "invoice",
    "turbine", "archive", "council", "manifest", "pipeline", "corridor", "transcript", "budget",
    "quarry", "beacon", "registry", "glacier", "voltage", "harvest", "tenant", "sensor",
)
_ADJS = (
    "careful", "quiet", "measured", "brittle", "coastal", "redundant", "sparse", "durable",
    "ordinary", "narrow", "steady", "verbose", "nominal", "elastic", "stale", "pinned",
    "distant", "granular", "idle", "seasonal", "mutable", "linear", "modest", "shared",
)
_VERBS = (
    "records", "balances", "streams", "recycles", "measures", "interleaves", "restores",
    "allocates", "compares", "drains", "schedules", "observes", "reports", "repeats",
    "estimates", "overlaps", "queues", "validates", "trims", "flags", "pins", "tracks",
)
_TAILS = (
    "so that anyone reading later could follow the years without a guide",
    "while paged caches recycle memory across requests",
    "and each page repeated the same careful columns",
    "without starving the cache hierarchy of reusable working sets",
    "before the next season's totals were copied into the summary",
    "which the committee reviewed twice a year",
    "as the interconnect settled into its steady state",
    "long after the original authors had moved on",
)
_IDENTS = (
    "buffer", "stride", "offset", "window", "budget", "slot", "page", "token", "batch", "route",
    "expert", "score", "index", "chunk", "state", "header", "record", "entry", "limit", "cursor",
)


class _Rng:
    """Thin seeded RNG wrapper so generators stay deterministic per (family, seed)."""

    def __init__(self, seed: int) -> None:
        self.r = random.Random(seed)

    def pick(self, seq):
        return seq[self.r.randrange(len(seq))]

    def ints(self, n: int, lo: int, hi: int) -> list[int]:
        return [self.r.randint(lo, hi) for _ in range(n)]


def _sentence(rng: _Rng) -> str:
    a, b = rng.pick(_NOUNS), rng.pick(_NOUNS)
    return (
        f"The {rng.pick(_ADJS)} {a} {rng.pick(_VERBS)} the {rng.pick(_ADJS)} {b} "
        f"{rng.pick(_TAILS)}, {rng.pick(_VERBS).rstrip('s')}ing {rng.r.randint(2, 900)} "
        f"{rng.pick(_NOUNS)}s per {rng.pick(('hour', 'season', 'request', 'page', 'step'))}."
    )


# --------------------------------------------------------------------------------------------
# block generators
# --------------------------------------------------------------------------------------------

def synthetic_blocks(seed: int) -> Iterator[str]:
    """Numbered passages of 4-7 template sentences."""
    rng = _Rng(seed)
    i = 0
    while True:
        n = rng.r.randint(4, 7)
        yield f"[passage {i}] " + " ".join(_sentence(rng) for _ in range(n))
        i += 1


def _prose_block(rng: _Rng, i: int) -> str:
    paras = []
    for _ in range(rng.r.randint(2, 4)):
        paras.append(" ".join(_sentence(rng) for _ in range(rng.r.randint(3, 6))))
    return f"## Section {i}: {rng.pick(_ADJS).title()} {rng.pick(_NOUNS).title()}\n\n" + "\n\n".join(paras)


def _code_block(rng: _Rng, i: int) -> str:
    cls = f"{rng.pick(_NOUNS).title()}{rng.pick(('Cache', 'Planner', 'Router', 'Index', 'Pool'))}"
    f1, f2, f3 = rng.pick(_IDENTS), rng.pick(_IDENTS), rng.pick(_IDENTS)
    cap = rng.r.randint(16, 4096)
    lines = [
        "```python",
        f"# module {i}: {cls.lower()}.py",
        "from __future__ import annotations",
        "",
        "import threading",
        "from dataclasses import dataclass, field",
        "",
        "",
        "@dataclass",
        f"class {cls}:",
        f'    """Bounded {f1} store with LRU eviction and per-key statistics."""',
        "",
        f"    capacity: int = {cap}",
        f"    _{f2}: dict[str, int] = field(default_factory=dict)",
        "    _lock: threading.Lock = field(default_factory=threading.Lock)",
        "",
        f"    def put(self, key: str, {f3}: int) -> None:",
        "        with self._lock:",
        f"            if key in self._{f2}:",
        f"                del self._{f2}[key]",
        f"            elif len(self._{f2}) >= self.capacity:",
        f"                oldest = next(iter(self._{f2}))",
        f"                del self._{f2}[oldest]",
        f"            self._{f2}[key] = {f3}",
        "",
        "    def get(self, key: str, default: int | None = None) -> int | None:",
        "        with self._lock:",
        f"            if key not in self._{f2}:",
        "                return default",
        f"            value = self._{f2}.pop(key)",
        f"            self._{f2}[key] = value",
        "            return value",
        "",
        "",
        f"def plan_{f1}(items: list[int], budget: int = {rng.r.randint(8, 512)}) -> list[list[int]]:",
        '    """Greedy first-fit packing of *items* into bins of at most *budget*."""',
        "    bins: list[list[int]] = []",
        "    for item in sorted(items, reverse=True):",
        "        for b in bins:",
        "            if sum(b) + item <= budget:",
        "                b.append(item)",
        "                break",
        "        else:",
        "            bins.append([item])",
        "    return bins",
        "```",
    ]
    return "\n".join(lines)


def _table_block(rng: _Rng, i: int) -> str:
    rows = [f"| Table {i}: {rng.pick(_NOUNS)} utilisation | requests | p50 ms | p95 ms | misses |",
            "| --- | ---: | ---: | ---: | ---: |"]
    for _ in range(rng.r.randint(5, 9)):
        r = rng.ints(4, 1, 9000)
        rows.append(f"| {rng.pick(_ADJS)} {rng.pick(_NOUNS)} | {r[0]} | {r[1] / 10:.1f} | {r[2] / 10:.1f} | {r[3]} |")
    bullets = "\n".join(f"- {_sentence(rng)}" for _ in range(rng.r.randint(3, 6)))
    return "\n".join(rows) + "\n\nObservations:\n\n" + bullets


def _json_block(rng: _Rng, i: int) -> str:
    keys = [rng.pick(_IDENTS) for _ in range(6)]
    body = ",\n".join(
        f'    "{k}_{j}": {rng.r.randint(0, 65536) if j % 2 else json_str(rng)}' for j, k in enumerate(keys)
    )
    return f"```json\n{{\n    \"service\": \"{rng.pick(_NOUNS)}-{i}\",\n{body}\n}}\n```"


def json_str(rng: _Rng) -> str:
    return f'"{rng.pick(_ADJS)}-{rng.pick(_NOUNS)}"'


def _log_block(rng: _Rng, i: int) -> str:
    lines = []
    t = rng.r.randint(0, 86399)
    for _ in range(rng.r.randint(6, 12)):
        t += rng.r.randint(1, 40)
        hh, mm, ss = t // 3600 % 24, t // 60 % 60, t % 60
        lvl = rng.pick(("INFO", "INFO", "INFO", "WARN", "DEBUG"))
        lines.append(
            f"2026-03-{1 + i % 28:02d} {hh:02d}:{mm:02d}:{ss:02d} {lvl:5s} {rng.pick(_NOUNS)}: "
            f"{rng.pick(_VERBS)} {rng.r.randint(1, 4096)} {rng.pick(_IDENTS)}s in {rng.r.randint(1, 900)} ms"
        )
    return "```text\n" + "\n".join(lines) + "\n```"


def realistic_blocks(seed: int) -> Iterator[str]:
    """Mixed prose / code / table / JSON / log document, deterministic per seed."""
    rng = _Rng(seed)
    kinds: list[Callable[[_Rng, int], str]] = [_prose_block, _code_block, _prose_block, _table_block,
                                               _prose_block, _json_block, _code_block, _log_block]
    i = 0
    while True:
        yield kinds[i % len(kinds)](rng, i)
        i += 1


FAMILIES: dict[str, Callable[[int], Iterator[str]]] = {
    "synthetic": synthetic_blocks,
    "realistic": realistic_blocks,
}

# --------------------------------------------------------------------------------------------
# prompt assembly with exact token fitting
# --------------------------------------------------------------------------------------------

TASK = (
    "Now write a comprehensive, very long technical report about the document above: summarise "
    "every section in order, list every table and code listing with a short critique, propose an "
    "implementation plan with complete Python code examples, and finish with an FAQ. Do not stop "
    "early; the report should be at least 6,000 words."
)

# words that tokenize to exactly one token when preceded by a space (verified at runtime,
# see ``TokenFitter._filler``); used to pad a document by an exact number of tokens
_FILLER_CANDIDATES = ("and", "the", "data", "system", "record", "page", "note", "value", "item", "step")


@dataclass
class Prompt:
    family: str
    seed: int
    nonce: str
    target_tokens: int
    text: str
    text_tokens: int          # tokens of ``text`` alone (checkpoint tokenizer, no specials)
    overhead_tokens: int      # chat-template overhead assumed at build time
    sha256: str
    fit_iterations: int
    task: str = TASK
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def expected_prompt_tokens(self) -> int:
        return self.text_tokens + self.overhead_tokens

    def messages(self) -> list[dict[str, str]]:
        return [{"role": "user", "content": self.text}]

    def record(self) -> dict[str, Any]:
        return {
            "family": self.family, "seed": self.seed, "nonce": self.nonce,
            "target_tokens": self.target_tokens, "text_tokens": self.text_tokens,
            "overhead_tokens": self.overhead_tokens, "expected_prompt_tokens": self.expected_prompt_tokens,
            "sha256": self.sha256, "chars": len(self.text), "fit_iterations": self.fit_iterations,
        }


class TokenFitter:
    """Builds prompts whose server-side token count equals a target exactly.

    ``count`` must count tokens the way the server does for message content (no special
    tokens). ``overhead`` is the number of tokens the server adds around the content (chat
    template, generation prompt); measure it with ``benchmark.calibrate_overhead``.
    """

    def __init__(self, count: Callable[[str], int], overhead: int) -> None:
        self.count = count
        self.overhead = overhead
        self._filler = [w for w in _FILLER_CANDIDATES if count(" " + w) == 1 and count("x " + w) == 2]
        if not self._filler:
            raise RuntimeError("no single-token filler word found for this tokenizer")

    def build(self, family: str, target_tokens: int, seed: int, nonce: str, task: str = TASK,
              max_iter: int = 8) -> Prompt:
        """A prompt whose server-side token count is exactly ``target_tokens``.

        Strategy: assemble whole content blocks until the body is just *under* the budget
        (block token counts are measured individually, so this stays linear), then close the
        gap with single-token filler words appended after a fixed anchor line. The anchor
        isolates the fillers from whatever the last content block ends with, so every filler
        contributes exactly one token and the final adjustment converges instead of
        oscillating between "one token too long" and "one token too short".
        """
        if family not in FAMILIES:
            raise ValueError(f"unknown workload family {family!r}; choose from {sorted(FAMILIES)}")
        header = (
            f"[benchmark nonce {nonce}] Read the following document carefully; a task follows it.\n\n"
        )
        footer = "\n\n" + task
        anchor = "\n\nAppendix (padding to an exact token count):"
        fixed = self.count(header) + self.count(footer) + self.count(anchor)
        budget = target_tokens - self.overhead - fixed
        if budget < 8:
            raise ValueError(f"target_tokens={target_tokens} is too small for the fixed sections")

        # Deliberately undershoot by a small margin so the fillers only ever have to ADD
        # tokens; the exact join/boundary cost is then measured once on the full text.
        margin = 48
        gen = FAMILIES[family](seed)
        blocks: list[str] = []
        used = 0
        while used < budget - margin:
            b = next(gen)
            n = self.count(b) + 1  # +1 for the "\n\n" join
            if used + n > budget - margin:
                # Trim this block by whole lines (then words) so it lands under the target.
                lines = b.split("\n")
                while lines and used + self.count("\n".join(lines)) + 1 > budget - margin:
                    lines.pop()
                b = "\n".join(lines)
                if not b.strip():
                    break
                blocks.append(b)
                used += self.count(b) + 1
                break
            blocks.append(b)
            used += n
        if not blocks:
            blocks = [next(gen)]

        rng = _Rng(seed ^ 0x5F3759DF)
        pad = 0
        iters = 0
        while True:
            iters += 1
            tail = "".join(" " + rng.pick(self._filler) for _ in range(pad)) if pad else ""
            text = header + "\n\n".join(blocks) + anchor + tail + footer
            n = self.count(text)
            delta = target_tokens - self.overhead - n
            if delta == 0:
                break
            if iters > max_iter:
                raise RuntimeError(
                    f"could not fit {family} prompt to {target_tokens} tokens (off by {delta})"
                )
            if delta > 0:
                pad += delta
                rng = _Rng(seed ^ 0x5F3759DF)  # rebuild the same tail deterministically
                continue
            # Overshot (the block trim left us long): drop a line from the last block and
            # let the fillers take up the slack again.
            lines = blocks[-1].split("\n")
            if len(lines) > 1:
                blocks[-1] = "\n".join(lines[:-1])
            else:
                words = blocks[-1].split(" ")
                blocks[-1] = " ".join(words[: max(1, len(words) + delta - 4)])
            pad = 0
            rng = _Rng(seed ^ 0x5F3759DF)
        return Prompt(
            family=family, seed=seed, nonce=nonce, target_tokens=target_tokens, text=text,
            text_tokens=n, overhead_tokens=self.overhead,
            sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(), fit_iterations=iters, task=task,
            extra={"filler_tokens": pad, "blocks": len(blocks)},
        )


def load_tokenizer(model_path: str):
    """The checkpoint's own tokenizer (the server tokenizes with the same files)."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_path)


def make_counter(tokenizer) -> Callable[[str], int]:
    def count(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    return count


if __name__ == "__main__":  # quick self-test: python -m benchmark.workloads <model_path>
    import sys

    tok = load_tokenizer(sys.argv[1])
    fitter = TokenFitter(make_counter(tok), overhead=int(sys.argv[2]) if len(sys.argv) > 2 else 0)
    for fam in FAMILIES:
        for n in (4096, 32768):
            p = fitter.build(fam, n, seed=1, nonce="selftest")
            print(fam, n, p.text_tokens + p.overhead_tokens, p.sha256[:12], "iters", p.fit_iterations, "chars", len(p.text))
