"""Speculative MoE expert prefetch (eager-window MTP-lookahead variant).

At the end of speculative step t (spec.py's window 3, after the draft/chain passes) the
NEXT step's verify tokens are fully determined and device-resident: the bonus candidate
and the chained drafts d1..dk were all scattered into the token pool. This module
predicts which experts those rows will route to at every layer -- reusing the model's
OWN router gates on cheap proxy hidden states -- admits the best predictions into the
offload cache (``prefetch_admit``: LRU-victim slots, budget-capped, prefill-buffer
slots excluded), and launches the H2D gather on a dedicated side stream so the bytes
move while the drain/staging host work runs and the PCIe link is otherwise idle.

Authority and correctness:
- The real router path is byte-identical with prefetch on: predictions never touch
  routing tensors; a correct prediction simply turns the next step's demand ``ensure``
  into a hit whose bytes already landed, and a wrong one ages out of the LRU.
- Every consumer joins ``ev_done`` before its first demand ensure (spec verify joins
  before the graph-A replay; ``Engine.forward_batch`` joins for non-spec/prefill
  batches), so a demand HIT on a half-copied slot is impossible.
- Speculative bandwidth is structurally capped: at most ``budget`` experts per step,
  copied only between the admission event and the next join.

Proxy recipe (zero-training, all enqueue-only):
- token-embedding proxy: rmsnorm(embed(tok)) for every row -- strongest at early
  layers, where routing is heavily token-identity driven;
- MTP-hidden proxy: uniform-mix (per-stream rmsnorm, mean over the hyper-connection
  streams) of the MTP head's 4-stream output for the step's first rows -- an
  approximation of the future token's residual, strongest at late layers.
Per layer, ``FREETOKEN_PREFETCH_R1_CROSSOVER`` (default 12) picks which proxy feeds
that layer's gate; scripts/moe-prefetch-harness.py measures both and sets the default.
"""

from __future__ import annotations

import os
from typing import Callable

import torch

from freetoken.utils import init_logger

logger = init_logger(__name__)


def _env_floats(name: str, default: tuple[float, ...]) -> tuple[float, ...]:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return tuple(float(x) for x in raw.split(","))
    except ValueError:
        logger.warning(f"ignoring malformed {name}={raw!r}")
        return default


class ExpertPrefetcher:
    """Owns the predictor tensors, the prefetch side stream, and the event fences."""

    def __init__(
        self,
        cache,
        model,
        *,
        num_rows: int,
        budget: int,
        topk: int,
        device: torch.device,
    ) -> None:
        self.cache = cache
        self.budget = int(budget)
        self.topk = int(topk)
        self.num_rows = int(num_rows)  # k + 1 verify rows
        self.device = device
        self.eps = 1e-6
        self._embed: Callable[[torch.Tensor], torch.Tensor] = model.model.embed_tokens.forward
        self.hc_count = model.model.hc_count

        # One-time stacked gate copy [Lg, H, E] (the per-layer weights are separate
        # allocations; bmm needs contiguity). Lg = 48 main layers + the MTP head's own
        # MoE as bank layer 48. ~129 MB bf16 -- accounted in the cache auto-size budget.
        gates = [layer.mlp.gate.weight for layer in model.model.layers.op_list]
        mtp = getattr(model, "mtp", None)
        if mtp is not None and cache.num_layers > len(gates):
            gates.append(mtp.layers.op_list[0].mlp.gate.weight)
        assert len(gates) == cache.num_layers, (
            f"gate stack {len(gates)} != cache bank layers {cache.num_layers}"
        )
        self._gates = torch.stack(gates).transpose(1, 2).contiguous()  # [Lg, H, E]
        self.num_layers = len(gates)
        self.num_experts = cache.num_experts

        # Layers that must never be admitted: CPU-decode / unpinned layers (their host
        # banks have no device alias -- the copy cannot serve them; admitting one would
        # publish a resident slot with garbage bytes).
        excluded = set(getattr(cache, "cpu_layer_ids", None) or ())
        excluded |= set(getattr(cache, "_unpinned_layers", frozenset()))
        admit_ok = torch.ones(self.num_layers, dtype=torch.bool, device=device)
        for layer in excluded:
            if 0 <= int(layer) < self.num_layers:
                admit_ok[int(layer)] = False
        self._admit_ok = admit_ok
        crossover = int(os.getenv("FREETOKEN_PREFETCH_R1_CROSSOVER", "12"))
        self._use_r1 = torch.arange(self.num_layers, device=device) >= crossover
        self._layer_base = (
            torch.arange(self.num_layers, device=device, dtype=torch.int64)
            * self.num_experts
        )
        conf = _env_floats("FREETOKEN_PREFETCH_ROW_CONF", (1.0, 0.85, 0.6))
        conf = (conf + (0.5,) * self.num_rows)[: self.num_rows]
        self._row_conf = torch.tensor(conf, dtype=torch.float32, device=device)
        self._row_offsets = torch.arange(-1, self.num_rows - 1, device=device)
        # Candidate list handed to the admission kernel: enough headroom past the budget
        # that resident predictions (skipped for free) don't starve the admit count.
        self.admit_cap = min(
            max(4 * self.budget, self.budget + 8), self.num_layers * self.num_rows * self.topk
        )

        self.stream = torch.cuda.Stream(device=device)
        self._ev_admit = torch.cuda.Event()
        self._ev_done = torch.cuda.Event()
        self._outstanding = False
        self._need_score = False

    # ---- per-step hooks (all enqueue-only; see engine/spec.py verify_step) -----------

    def join(self) -> None:
        """Make the current (compute) stream wait for the in-flight prefetch copy.

        Must run before the first demand ensure of any forward that could route to a
        prefetched expert -- spec verify calls it before the graph-A replay, and
        Engine.forward_batch calls it for every other batch (prefill, non-spec decode),
        so a demand HIT can never read a half-copied slot (invariant I3)."""
        if self._outstanding:
            torch.cuda.current_stream().wait_event(self._ev_done)
            self._outstanding = False

    def used_check(self) -> None:
        """Score the previous step's plan (device-side counters). Call after join and
        before the next ``run`` rewrites plan-2 -- spec window 2 is both."""
        if self._need_score:
            self.cache.prefetch_used_check()
            self._need_score = False

    def run(
        self,
        pool_row: torch.Tensor,
        new_device_len: torch.Tensor,
        sel: torch.Tensor,
        r1_rows: torch.Tensor | None,
        chain_r1_out: torch.Tensor | None,
    ) -> None:
        """Predict + admit + launch the copy for the NEXT step's verify rows.

        ``pool_row``: this request's token-pool row; ``new_device_len``: [1] device
        tensor (next cached_len + 1); ``sel``: [1] int64, the accepted row n-1;
        ``r1_rows``: the draft pass's 4-stream outputs (row ``sel`` is the premix proxy
        for the next step's first row); ``chain_r1_out``: graph C's chained output (the
        proxy for the second row), or None outside graph mode."""
        row_cap = pool_row.shape[0] - 1
        idx = torch.clamp(new_device_len + self._row_offsets, max=row_cap)
        toks = pool_row.index_select(0, idx)
        emb = self._embed(toks).float()  # [R, H]
        embu = emb * torch.rsqrt(emb.pow(2).mean(-1, keepdim=True) + self.eps)

        r1_full = embu
        if r1_rows is not None:
            rows = [r1_rows.index_select(0, sel)]
            if chain_r1_out is not None and self.num_rows >= 3:
                rows.append(chain_r1_out[:1])
            r1 = torch.cat(rows) if len(rows) > 1 else rows[0]
            r1u = self._uniform_mix(r1.float())  # [n_r1, H]
            r1_full = embu.clone()
            r1_full[: r1u.shape[0]] = r1u

        dtype = self._gates.dtype
        proxy = torch.where(
            self._use_r1[:, None, None], r1_full.to(dtype), embu.to(dtype)
        )  # [Lg, R, H]
        logits = torch.bmm(proxy, self._gates)  # [Lg, R, E]
        probs = logits.float().softmax(-1) * self._row_conf[None, :, None]
        probs = torch.where(
            self._admit_ok[:, None, None], probs, probs.new_full((), -1.0)
        )
        topv, topi = torch.topk(probs, self.topk, dim=-1)  # [Lg, R, k']
        flat = (self._layer_base[:, None, None] + topi).reshape(-1)
        score = topv.reshape(-1)
        order = torch.argsort(score, descending=True)[: self.admit_cap]
        cand = flat.index_select(0, order).to(torch.int32)
        # Excluded-layer sentinels carry score -1; the admission kernel skips -1 ids.
        cand = torch.where(
            score.index_select(0, order) > 0, cand, cand.new_full((), -1)
        )

        self.cache.prefetch_admit(cand, self.budget)
        self._ev_admit.record()
        self.stream.wait_event(self._ev_admit)
        with torch.cuda.stream(self.stream):
            self.cache.prefetch_copy()
        self._ev_done.record(self.stream)
        self._outstanding = True
        self._need_score = True

    def _uniform_mix(self, r: torch.Tensor) -> torch.Tensor:
        """Cheap stand-in for the per-layer GatedResidual.mix: per-stream fp32 rmsnorm,
        mean over the streams (a uniform positive rescale keeps the gate's top-k set)."""
        xf = r.unflatten(-1, (self.hc_count, r.shape[-1] // self.hc_count))
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return xf.mean(-2)


class LayerLookahead:
    """In-graph cross-layer expert lookahead (``--moe-prefetch layer``).

    At hooked decode layer J, predict layer J+N's routed experts for the CURRENT rows by
    running layer J+N's own hyper-connection mix + router gate on layer J's residual --
    the M1 harness's xmix proxy (precision ~88/68/61% at N=1/2/3) -- admit the top
    predictions into the offload cache (ring plan ``target % P``) and copy their banks on
    a low-occupancy side stream that joins right before layer J+N's own hook. Everything
    is fixed-shape with device-side counts and static plan slices, so the whole
    choreography captures into the decode CUDA graphs; the per-plan event record/wait
    pairs become plain graph edges (record always precedes wait within one forward).

    The authoritative router is untouched (predictions never write routing tensors);
    correctness holds because a copy targeting layer T is joined at layer T's hook
    before T's demand ensure can hit the prefetched slot. All copies issued within a
    forward are also joined within it, so no cross-forward fence is needed.

    Cost knobs (the M1 histogram shows flat miss density across layers, so coverage
    scales with hooks and the per-hook launch cost is the whole game):
    The defaults ARE the measured live optimum (2026-09-01: 23 hooks x top-4 x xnorm
    = +1 tok/s over prefetch-off with ~15-20% miss coverage; denser hooks or deeper
    candidates measured net-negative):
    - FREETOKEN_PREFETCH_LOOKAHEAD: N (default 2 -- copy window ~2 layers of compute)
    - FREETOKEN_PREFETCH_LAYER_TOPK: predictions per row per hook (default 4)
    - FREETOKEN_PREFETCH_HOOK_STRIDE: hook every k-th layer (default 2)
    - FREETOKEN_PREFETCH_LAYER_BUDGET: max experts admitted per hook (default 6) --
      caps the worst-case join stall (a cold hook admitting 12 experts is ~1.2 ms of
      copy against a ~0.3 ms two-layer window)
    - FREETOKEN_PREFETCH_TARGETS: which layers get prefetched -- "miss" (default:
      stride-many layers picked by the M1 harness's per-layer miss density, heaviest
      first; measured +0.5 tok/s over the even-layer spread), "stride" (every
      stride-th layer), or an explicit comma list of target layer ids
    - FREETOKEN_PREFETCH_PROXY:
        mix     true per-layer mix + gate (torch chain, ~34 us/hook under graph replay)
        xnorm   target's (1+w) grouped norm + stream mean + gate -- drops the sigmoid
                gating chain (~3 GEMM-free nodes cheaper)
        uniform plain rmsnorm + stream mean + gate (weakest signal, cheapest)
        fused   kernel/triton/moe_lookahead.py 4-kernel pipeline -- numerically exact vs
                mix and 3x faster EAGER, but measured slower than mix under graph replay
                on this stack (53 vs 34 us/hook): experimental, not the default
    """

    def __init__(self, cache, model, *, device: torch.device) -> None:
        self.cache = cache
        self.device = device
        self.layers = model.model.layers.op_list
        self.hc_count = model.model.hc_count
        self.num_experts = cache.num_experts
        self.eps = 1e-6
        self.lookahead = max(1, int(os.getenv("FREETOKEN_PREFETCH_LOOKAHEAD", "2")))
        self.topk = max(1, int(os.getenv("FREETOKEN_PREFETCH_LAYER_TOPK", "4")))
        stride = max(1, int(os.getenv("FREETOKEN_PREFETCH_HOOK_STRIDE", "2")))
        self.proxy = os.getenv("FREETOKEN_PREFETCH_PROXY", "xnorm").strip().lower()

        self.call_budget = max(1, int(os.getenv("FREETOKEN_PREFETCH_LAYER_BUDGET", "6")))
        if self.proxy == "fused":
            # The fused Triton proxy reads the HC mix weights and the router gate raw
            # (bf16 layout); --dense-quant fp8's "hc" part turns those into fp8 + scale.
            probe = self.layers[0].mlp_hyper_connection.input_mix_weight_up.weight
            gate_w = self.layers[0].mlp.gate.weight
            if probe.dtype != torch.bfloat16 or gate_w.dtype != torch.bfloat16:
                raise ValueError(
                    "FREETOKEN_PREFETCH_PROXY=fused needs bf16 HC/gate weights; with "
                    "--dense-quant fp8 use the xnorm/mix proxy or drop 'hc' from "
                    "FREETOKEN_DENSE_FP8_PARTS"
                )
        excluded = set(getattr(cache, "cpu_layer_ids", None) or ())
        excluded |= set(getattr(cache, "_unpinned_layers", frozenset()))
        num_main = len(self.layers)
        n = self.lookahead
        spec = os.getenv("FREETOKEN_PREFETCH_TARGETS", "miss").strip().lower()
        if spec == "miss":
            # qwen4_exp per-layer decode miss density, heaviest first (M1 harness,
            # 2026-09-01, "/media/nuko/New Volume/route-dumps-m1"); same target COUNT
            # as the stride default, but spent where the misses are.
            order = [0, 1, 2, 22, 34, 38, 6, 7, 18, 28, 12, 5, 21, 42, 29, 13, 26, 37,
                     10, 33, 35, 30, 45, 46, 41, 44, 4, 36, 20, 27, 8, 24, 19, 14, 17,
                     47, 43, 23, 11, 25, 40, 3, 9, 32, 39, 31, 16, 15]
            want = len(range(0, num_main - n, stride))
            targets = [t for t in order if n <= t < num_main][:want]
        elif spec in ("", "stride"):
            targets = [j + n for j in range(0, num_main - n, stride)]
        else:
            targets = [int(x) for x in spec.split(",")]
        self._issue: dict[int, int] = {}  # issue layer J -> target layer J+N
        for target in sorted(set(targets)):
            if n <= target < num_main and target not in excluded:
                self._issue[target - n] = target
        # plan id per target; ring P = N+1 guarantees a plan is reused only after its
        # previous target's join already ran (join-before-issue, see the plan's I2).
        self.ring = self.lookahead + 1
        assert cache.prefetch_ring >= self.ring, (
            f"init_prefetch_plan ring {cache.prefetch_ring} < required {self.ring}"
        )
        self._join_plan = {t: t % self.ring for t in self._issue.values()}
        self.stream = torch.cuda.Stream(device=device)
        self._ev_adm = [torch.cuda.Event() for _ in range(self.ring)]
        self._ev_copy = [torch.cuda.Event() for _ in range(self.ring)]
        self._scratch = None
        if self.proxy == "fused":
            from freetoken.kernel.triton.moe_lookahead import LookaheadScratch

            self._scratch = LookaheadScratch(device, self.hc_count)
        logger.info_rank0(
            f"Cross-layer expert lookahead armed: N={self.lookahead}, "
            f"top-{self.topk}/row, {len(self._issue)} hooked layers "
            f"(targets={spec or 'stride'}, stride {stride}), proxy={self.proxy}, "
            f"per-hook budget {self.call_budget}"
        )

    def hook(self, layer_id: int, hidden: torch.Tensor, batch) -> None:
        """Called at the mlp seam of every decoder layer (decode only). Captured."""
        if not batch.is_decode:
            return
        plan = self._join_plan.get(layer_id)
        if plan is not None:
            # Join THIS layer's prefetch copy before its demand ensure can hit the slot.
            torch.cuda.current_stream().wait_event(self._ev_copy[plan])
        target = self._issue.get(layer_id)
        if target is None:
            return
        tgt = self.layers[target]
        if self.proxy == "fused":
            from freetoken.kernel.triton.moe_lookahead import lookahead_route_split

            hcm = tgt.mlp_hyper_connection
            cand = torch.empty(
                hidden.shape[0] * self.topk, dtype=torch.int32, device=hidden.device
            )
            lookahead_route_split(
                hidden,
                hcm.hc_norm.weight,
                hcm.input_mix_weight_down_block_inject.weight,
                hcm.input_mix_weight_up.weight,
                tgt.mlp.gate.weight,
                self._scratch,
                cand,
                hc=self.hc_count,
                eps=hcm.hc_norm.eps,
                topk=self.topk,
                flat_base=target * self.num_experts,
            )
        else:
            if self.proxy == "uniform":
                xf = hidden.float().unflatten(
                    -1, (self.hc_count, hidden.shape[-1] // self.hc_count)
                )
                xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
                mixed = xf.mean(-2).to(hidden.dtype)
            elif self.proxy == "xnorm":
                # The target's (1+w) grouped norm + stream mean: keeps the per-layer norm
                # weights (most of mix's routing signal) without the sigmoid gating chain.
                rn = tgt.mlp_hyper_connection.hc_norm.forward(hidden)
                mixed = rn.unflatten(
                    -1, (self.hc_count, hidden.shape[-1] // self.hc_count)
                ).mean(-2)
            else:
                mixed = tgt.mlp_hyper_connection.mix(hidden)[0]
            logits = tgt.mlp.gate.forward(mixed)  # [rows, E]
            topi = torch.topk(logits, self.topk, dim=-1).indices  # [rows, k]
            # Row-major flatten = row 0 (the most confident verify row) first; the
            # admission kernel's map lookup dedups and skips residents -- no sort needed.
            cand = (topi.reshape(-1) + target * self.num_experts).to(torch.int32)
        p = target % self.ring
        self.cache.prefetch_admit(cand, min(cand.numel(), self.call_budget), plan=p)
        self._ev_adm[p].record()
        self.stream.wait_event(self._ev_adm[p])
        with torch.cuda.stream(self.stream):
            self.cache.prefetch_copy(plan=p)
        self._ev_copy[p].record(self.stream)


def attach_moe_lookahead(model, lookahead: LayerLookahead | None) -> None:
    """Wire the lookahead into the model's own decoder layers (the MTP head's standalone
    layer is deliberately not walked -- its experts are graph-B territory)."""
    for layer in model.model.layers.op_list:
        layer._moe_lookahead = lookahead


__all__ = ["ExpertPrefetcher", "LayerLookahead", "attach_moe_lookahead"]
