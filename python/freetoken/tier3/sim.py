"""Offline two-level (VRAM / RAM) expert-cache simulator over recorded routing.

Answers the question the whole NVMe-tier design turns on: for a given (GPU slots, host slots)
geometry, how many bytes per decode step actually reach the disk? Replays a compacted route
trace (``scripts/tier3-compact-routes.py``) rather than the raw dumps, so a full geometry sweep
costs seconds.

Tier semantics deliberately mirror the serving code:

* **L1 (VRAM)** is one unified pool over the flat ``layer * num_experts + expert`` id space
  with global timestamp-LRU eviction -- ``OffloadMoeCache.id_of_slot`` / ``usage``.
* **L2 (RAM)** is a *per-layer* pool (see the plan: per-layer pools are what keep the H2D copy
  kernel unchanged), and it only ever sees the **L1 miss stream** -- a GPU hit never touches
  host memory. That filtering is why the tier-2 policy question is not the same as the tier-1
  one.
"""

from __future__ import annotations

import heapq
import math
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import torch


class GpuLru:
    """Timestamp-LRU over the flat (layer, expert) id space -- the serving cache's semantics
    (unified pool, global argmin eviction, per-call clock). Lazy-deletion heap, as in
    scripts/moe-prefetch-harness.py's LRUSim so the two agree by construction."""

    def __init__(self, slots: int) -> None:
        self.slots = slots
        self.last: dict[int, int] = {}
        self.heap: list[tuple[int, int]] = []
        self.clock = 0
        self.hits = 0
        self.misses = 0

    def access(self, ids: Iterable[int]) -> list[int]:
        self.clock += 1
        missed = []
        for i in ids:
            if i in self.last:
                self.hits += 1
            else:
                self.misses += 1
                missed.append(i)
                while len(self.last) >= self.slots:
                    ts, victim = heapq.heappop(self.heap)
                    if self.last.get(victim) == ts:
                        del self.last[victim]
            self.last[i] = self.clock
            heapq.heappush(self.heap, (self.clock, i))
        return missed


class HostPool:
    """One layer's tier-2 pool. Policies:

    ``lru``     plain recency over the L1-miss stream.
    ``lfu``     frequency over the observed stream, ties broken by recency.
    ``hotset``  a static core (``hot_frac`` of the pool) pinned from an offline frequency
                profile, with the remainder run as LRU. This is the shape the plan proposes
                if the routing turns out to be skewed.
    """

    def __init__(self, slots: int, policy: str = "lru",
                 hot: Sequence[int] = (), hot_frac: float = 0.7) -> None:
        self.slots = slots
        self.policy = policy
        self.hits = 0
        self.misses = 0
        self._lru: OrderedDict[int, None] = OrderedDict()
        self._freq: Counter[int] = Counter()
        self._pinned: set[int] = set()
        if policy == "hotset" and slots > 0:
            self._pinned = set(list(hot)[: int(slots * hot_frac)])
        self._dyn_slots = max(0, slots - len(self._pinned))

    def access(self, expert: int) -> bool:
        """Touch one expert; returns True on a hit (no NVMe read needed)."""
        self._freq[expert] += 1
        if expert in self._pinned:
            self.hits += 1
            return True
        if expert in self._lru:
            self.hits += 1
            self._lru.move_to_end(expert)
            return True
        self.misses += 1
        self._admit(expert)
        return False

    def _admit(self, expert: int) -> None:
        if self._dyn_slots <= 0:
            return
        while len(self._lru) >= self._dyn_slots:
            if self.policy == "lfu":
                victim = min(self._lru, key=lambda e: (self._freq[e], e))
            else:
                victim = next(iter(self._lru))
            del self._lru[victim]
        self._lru[expert] = None


@dataclass
class SimResult:
    gpu_slots: int
    host_slots: int
    policy: str
    steps: int = 0
    touches: int = 0
    gpu_hits: int = 0
    gpu_misses: int = 0
    host_hits: int = 0
    host_misses: int = 0
    reads_per_layer: list[int] = field(default_factory=list)
    bytes_per_expert: int = 2_764_800

    @property
    def gpu_miss_rate(self) -> float:
        return self.gpu_misses / max(1, self.touches)

    @property
    def host_hit_rate(self) -> float:
        return self.host_hits / max(1, self.gpu_misses)

    @property
    def reads_per_step(self) -> float:
        return self.host_misses / max(1, self.steps)

    @property
    def nvme_bytes_per_step(self) -> float:
        return self.reads_per_step * self.bytes_per_expert

    def tok_s_ceiling(self, bw_bytes_s: float, accepted_per_step: float = 1.0) -> float:
        b = self.nvme_bytes_per_step
        return math.inf if b <= 0 else accepted_per_step * bw_bytes_s / b

    def inflight(self, lookahead: int) -> tuple[float, float]:
        """(p50, p99) reads issuable within a ``lookahead``-layer window -- the achievable NVMe
        queue depth. A design that misses ~1 expert per layer can only ever run at QD~2 unless
        it reads ahead, and low-QD bandwidth is a fraction of the parallel-read ceiling."""
        r = self.reads_per_layer
        if not r or lookahead <= 0:
            return (0.0, 0.0)
        win = [sum(r[i:i + lookahead]) for i in range(0, max(1, len(r) - lookahead + 1))]
        win.sort()
        return (win[len(win) // 2], win[min(len(win) - 1, int(0.99 * len(win)))])


def load_trace(path: str) -> tuple[list[torch.Tensor], int]:
    """Compacted trace -> (per-chunk ``[L, T, K]`` int64 id tensors, top_k)."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    return [c["ids"].long() for c in blob["chunks"]], int(blob["top_k"])


def simulate(
    chunks: Sequence[torch.Tensor],
    *,
    gpu_slots: int,
    host_slots: int,
    num_experts: int = 512,
    policy: str = "lru",
    verify_rows: int = 3,
    verify_stride: int = 2,
    skip_steps: int = 32,
    bytes_per_expert: int = 2_764_800,
    hot: dict[int, Sequence[int]] | None = None,
) -> SimResult:
    """Replay the trace as decode steps.

    ``verify_rows`` / ``verify_stride`` model MTP speculative decode: each step evaluates a
    k+1-row verify batch and the stream advances by the accepted-token count (defaults match
    scripts/moe-prefetch-harness.py: 3 rows, ~2 accepted). ``skip_steps`` drops each chunk's
    cold front so the measurement is of the warm cache.
    """
    num_layers = int(chunks[0].shape[0])
    gpu = GpuLru(gpu_slots)
    pools = [
        HostPool(host_slots, policy, (hot or {}).get(l, ()) if hot else ())
        for l in range(num_layers)
    ]
    res = SimResult(gpu_slots, host_slots, policy, bytes_per_expert=bytes_per_expert)

    for ids in chunks:
        L, T, _ = ids.shape
        for si, t in enumerate(range(0, T - verify_rows + 1, verify_stride)):
            rows = ids[:, t:t + verify_rows, :]  # [L, rows, K]
            warm = si >= skip_steps
            if warm:
                res.steps += 1
            for layer in range(L):
                active = torch.unique(rows[layer]).tolist()
                base = layer * num_experts
                missed = gpu.access(base + e for e in active)
                if warm:
                    res.touches += len(active)
                    res.gpu_hits += len(active) - len(missed)
                    res.gpu_misses += len(missed)
                nread = 0
                for flat in missed:
                    if not pools[layer].access(flat - base):
                        nread += 1
                if warm:
                    res.host_hits += len(missed) - nread
                    res.host_misses += nread
                    res.reads_per_layer.append(nread)
    return res


def oracle_host_hit(chunks: Sequence[torch.Tensor], host_slots: int,
                    num_experts: int = 512) -> float:
    """Upper bound on the tier-2 hit rate: keep the globally most-frequent ``host_slots``
    experts per layer. If LRU is already close to this, there is no policy work worth doing."""
    num_layers = int(chunks[0].shape[0])
    freq = [Counter() for _ in range(num_layers)]
    for ids in chunks:
        for layer in range(num_layers):
            freq[layer].update(ids[layer].reshape(-1).tolist())
    hit = tot = 0
    for layer in range(num_layers):
        counts = sorted(freq[layer].values(), reverse=True)
        hit += sum(counts[:host_slots])
        tot += sum(counts)
    return hit / max(1, tot)


def frequency_hotset(chunks: Sequence[torch.Tensor], num_experts: int = 512) -> dict[int, list[int]]:
    """Per-layer expert ids ordered by observed routing frequency (the offline hot-set profile)."""
    num_layers = int(chunks[0].shape[0])
    out = {}
    for layer in range(num_layers):
        c: Counter[int] = Counter()
        for ids in chunks:
            c.update(ids[layer].reshape(-1).tolist())
        out[layer] = [e for e, _ in c.most_common()]
    return out


def prepare_steps(
    chunks: Sequence[torch.Tensor],
    *,
    verify_rows: int = 3,
    verify_stride: int = 2,
    skip_steps: int = 32,
) -> list[list[list[int]]]:
    """Precompute every step's per-layer DISTINCT active expert list.

    The dedup (``torch.unique`` per layer per step) dominates a replay, and it does not depend
    on the cache geometry -- so do it once and let a whole sweep reuse it.
    """
    steps: list[list[list[int]]] = []
    for ids in chunks:
        _, T, _ = ids.shape
        for si, t in enumerate(range(0, T - verify_rows + 1, verify_stride)):
            if si < skip_steps:
                continue
            rows = ids[:, t:t + verify_rows, :]
            steps.append([torch.unique(rows[l]).tolist() for l in range(int(ids.shape[0]))])
    return steps


def simulate_steps(
    steps: Sequence[Sequence[Sequence[int]]],
    *,
    gpu_slots: int,
    host_slots: int,
    num_experts: int = 512,
    policy: str = "lru",
    bytes_per_expert: int = 2_764_800,
    hot: dict[int, Sequence[int]] | None = None,
) -> SimResult:
    """Replay precomputed steps from :func:`prepare_steps`. Same semantics as :func:`simulate`."""
    num_layers = len(steps[0])
    gpu = GpuLru(gpu_slots)
    pools = [
        HostPool(host_slots, policy, (hot or {}).get(l, ()) if hot else ())
        for l in range(num_layers)
    ]
    res = SimResult(gpu_slots, host_slots, policy, bytes_per_expert=bytes_per_expert)
    for step in steps:
        res.steps += 1
        for layer, active in enumerate(step):
            base = layer * num_experts
            missed = gpu.access(base + e for e in active)
            res.touches += len(active)
            res.gpu_hits += len(active) - len(missed)
            res.gpu_misses += len(missed)
            pool = pools[layer]
            nread = 0
            for flat in missed:
                if not pool.access(flat - base):
                    nread += 1
            res.host_hits += len(missed) - nread
            res.host_misses += nread
            res.reads_per_layer.append(nread)
    return res
