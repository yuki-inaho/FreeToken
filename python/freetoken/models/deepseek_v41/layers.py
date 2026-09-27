"""Shared DeepSeek-V4.1 primitives.

V4.1 reuses the 0731 leaf modules (``Linear`` with a configurable FP8 block, ``RMSNorm``)
and adds the Single-Pass mHC helpers: the reference splits the hyper-connection mix into
``hc_mixes`` (the coefficient projection + Sinkhorn, computed from the residual) and
``hc_pre_apply`` (collapsing the streams with a *previously computed* ``pre``), because a
sublayer consumes the mixing coefficients its predecessor produced (tech report 2.4.1).
The 0731 ``Block.hc_pre`` fused the two; keeping them apart is what lets ``pre_mix``
thread through the block chain.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from freetoken.kernel.triton.dsv4.hc import hc_post_combine, hc_pre_combine
from freetoken.kernel.triton.dsv4.sinkhorn import hc_split_sinkhorn
from freetoken.models.deepseek_v4.layers import FP8, Linear, RMSNorm, get_window_topk_idxs

__all__ = [
    "FP8",
    "Linear",
    "RMSNorm",
    "get_window_topk_idxs",
    "hc_mixes",
    "hc_pre_apply",
    "hc_post",
    "make_identity_pre_mix",
]


def hc_mixes(
    x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor,
    *, hc_mult: int, sinkhorn_iters: int, hc_eps: float, norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The mHC coefficient predictor: ``x`` ``[b, s, hc, dim]`` -> ``(pre [M, hc], post [M, hc],
    comb [M, hc, hc])`` with ``M = b*s``. The projection runs over the flattened ``hc*dim``
    stream and is RMS-normalised with one statistic per token (the norm weights are folded
    into ``hc_fn`` offline), then split and Sinkhorn-balanced exactly as the reference."""
    shape = x.size()
    xf = x.flatten(2).float()
    rsqrt = torch.rsqrt(xf.square().mean(-1, keepdim=True) + norm_eps)
    mixes = F.linear(xf, hc_fn) * rsqrt
    pre, post, comb = hc_split_sinkhorn(
        mixes.view(-1, mixes.size(-1)), hc_scale, hc_base, hc_mult, sinkhorn_iters, hc_eps
    )
    M = shape[0] * shape[1]
    return pre.view(M, hc_mult), post.view(M, hc_mult), comb.view(M, hc_mult, hc_mult)


def hc_pre_apply(x: torch.Tensor, pre: torch.Tensor, *, hc_mult: int, dim: int) -> torch.Tensor:
    """Collapse the ``hc_mult`` streams of ``x`` ``[b, s, hc, dim]`` with ``pre`` ``[M, hc]``
    (fp32 weights) -> ``[b, s, dim]`` in ``x``'s dtype."""
    shape, dtype = x.size(), x.dtype
    M = shape[0] * shape[1]
    y = hc_pre_combine(x.reshape(M, hc_mult, dim).float(), pre, dtype)
    return y.view(*shape[:2], dim)


def hc_post(
    x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor,
    *, hc_mult: int, dim: int,
) -> torch.Tensor:
    """Expand a sublayer output ``x`` ``[b, s, dim]`` back to the streams and mix the residual
    in through the doubly-stochastic ``comb``: ``[b, s, hc, dim]``."""
    shape = residual.size()
    M = shape[0] * shape[1]
    y = hc_post_combine(x.reshape(M, dim), residual.reshape(M, hc_mult, dim), post, comb)
    return y.view(shape)


def make_identity_pre_mix(M: int, hc_mult: int, device: torch.device) -> torch.Tensor:
    """The first block's input mix: a one-hot on stream 0 (the reference
    ``make_identity_pre_mix``), ``[M, hc]`` fp32."""
    pre = torch.zeros(M, hc_mult, dtype=torch.float32, device=device)
    pre[:, 0] = 1.0
    return pre
