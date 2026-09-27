"""DeepSeek-V4.1 MoE: bias-steered sqrtsoftplus router, FP8 (32-block) shared expert, and
the offloaded FP4 routed experts on the shared slot cache.

Differences from the 0731 module: no hash-routed layers (``tid2eid`` is gone), the router
carries ``gate_temp`` / ``norm_topk_prob`` (reference defaults), the dense FP8 linears use
32x32 block scales, and the expert counts come from ``args.get_moe_config(layer_id)`` so the
DSpark stages (128 experts, top-3) build the same module with their own geometry.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from freetoken.kernel.triton.dsv4.bf16_linear import bf16_linear_fp32
from freetoken.kernel.triton.dsv4.fp8_linear import act_quant_fp8_roundtrip
from freetoken.kernel.triton.dsv4.swiglu import fused_swiglu
from freetoken.layers import OffloadMoELayer

from .args import DeepseekV41Args
from .layers import Linear


class Gate(nn.Module):
    """Reference ``Gate``: ``scores = softplus(x @ W^T / gate_temp).sqrt()`` (or softmax /
    sigmoid), selection by ``topk(scores + bias)`` while the routing weights come from the
    UNBIASED scores, renormalised with the training-time ``1e-20`` and scaled by
    ``route_scale``. ``bias_vl`` (image tokens) is not loaded: the port is text-only."""

    def __init__(self, layer_id: int, args: DeepseekV41Args):
        super().__init__()
        n_routed, n_activated = args.get_moe_config(layer_id)
        self.topk = n_activated
        self.score_func = args.score_func
        self.gate_temp = float(args.gate_temp)
        self.norm_topk_prob = bool(args.norm_topk_prob)
        self.route_scale = float(args.route_scale)
        self.weight = nn.Parameter(
            torch.empty(n_routed, args.dim, dtype=torch.bfloat16), requires_grad=False
        )
        self.bias = nn.Parameter(torch.empty(n_routed, dtype=torch.float32), requires_grad=False)

    def scores(self, x: torch.Tensor) -> torch.Tensor:
        """Unbiased routing scores ``[rows, n_routed]`` fp32 (the tempered router logits through
        the score function); ``forward`` selects on ``scores + bias`` and weights with these."""
        scores = bf16_linear_fp32(x, self.weight)
        if self.gate_temp != 1.0:
            scores = scores / self.gate_temp
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1)
        elif self.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:
            scores = F.softplus(scores).sqrt()
        return scores

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.scores(x)
        indices = (scores + self.bias).topk(self.topk, dim=-1)[1]
        weights = scores.gather(1, indices)
        if self.norm_topk_prob and self.topk > 1:
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        weights = weights * self.route_scale
        return weights, indices


class Expert(nn.Module):
    """The dense SwiGLU shared expert (FP8 weights, 32x32 block scales). The clamps come
    from training (``swiglu_limit``): up clamped both sides, gate clamped from above."""

    def __init__(self, dim: int, inter_dim: int, swiglu_limit: float, block: int):
        super().__init__()
        self.w1 = Linear(dim, inter_dim, kind="fp8", block=block)
        self.w2 = Linear(inter_dim, dim, kind="fp8", block=block)
        self.w3 = Linear(dim, inter_dim, kind="fp8", block=block)
        self.swiglu_limit = swiglu_limit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = fused_swiglu(self.w1(x), self.w3(x), self.swiglu_limit, x.dtype)
        return self.w2(h)


class DSV41OffloadMoELayer(OffloadMoELayer):
    """Routed FP4 experts on the shared offload cache with explicit expert geometry (the
    main tower's 384/top-6 or a DSpark stage's 128/top-3). The prefill crossover mirrors the
    0731 ``DSV4OffloadMoELayer``: below ``E`` routes the decode-style slot path moves fewer
    bytes than streaming the whole layer."""

    def __init__(self, layer_id: int, args: DeepseekV41Args, num_experts: int, top_k: int):
        super().__init__(
            layer_id=layer_id,
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=args.dim,
            intermediate_size=args.moe_inter_dim,
            renormalize=True,
            activation="silu",
        )
        self.swiglu_limit = args.swiglu_limit

    def _prefill_routed(self, hidden_states, topk_weights, topk_ids):
        cache = self.offload_cache
        assert cache is not None
        # A narrow DSpark stage layer (128 rows on the 384-wide cache) has no whole-layer
        # rows: the per-route path is the only one defined for it, whatever m * top_k is.
        if cache.is_full_layer(self.layer_id) and (
            hidden_states.shape[0] * self.top_k >= self.num_experts
            or cache.is_unpinned_layer(self.layer_id)
        ):
            return super()._prefill_routed(hidden_states, topk_weights, topk_ids)
        cache.ensure_experts(self.layer_id, topk_ids)
        cache.copy_missing()
        if cache.collect_stats:
            cache.record_decode_stats(self.layer_id)
        return self._expert_gemm(
            cache, hidden_states, topk_weights, topk_ids,
            views=cache.bank_views(), n=None, alphas=cache.alphas_for_slots(self.layer_id),
            is_prefill=True,
        )


class MoE(nn.Module):
    """Sparse MoE block: router -> offloaded FP4 routed experts + the dense shared expert.

    ``quant_act`` applies the reference's FP8 (per-32, ue8m0) activation round trip before
    the routed experts (the reference feeds fp8 activations to ``fp4_gemm``); the production
    path leaves the bf16 activations to the expert kernels' own scheme, as 0731 does, and the
    parity tests turn it on.
    """

    def __init__(self, layer_id: int, args: DeepseekV41Args, *, quant_act: bool = False):
        super().__init__()
        self.dim = args.dim
        self.quant_act = quant_act
        n_routed, n_activated = args.get_moe_config(layer_id)
        self.gate = Gate(layer_id, args)
        self.shared_experts = Expert(
            args.dim, args.moe_inter_dim, args.swiglu_limit, args.fp8_block_size
        )
        self.experts = DSV41OffloadMoELayer(layer_id, args, n_routed, n_activated)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x)
        # The shared expert goes first so its GEMM is on the stream before routed_forward
        # blocks on the CPU pool (hybrid decode) -- same ordering as the 0731 module.
        shared = self.shared_experts(x)
        xr = act_quant_fp8_roundtrip(x, 32) if self.quant_act else x
        routed = self.experts.routed_forward(
            xr, weights.float().contiguous(), indices.to(torch.int32).contiguous()
        )
        return (routed + shared).view(shape)


__all__ = ["Gate", "Expert", "DSV41OffloadMoELayer", "MoE"]
