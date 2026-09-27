"""The Engram conditional-memory module (layers 1 and 14): gated n-gram lookups written into
the 4-stream residual before the block.

Reference ``Engram.forward`` (inference/model.py): the ``n_hash_cols = (max_ngram_size-1) *
n_heads = 24`` fetched rows (256 dims each, dequantized) go through ``wkv`` (fp8, 32-block)
into one key per hyper-connection stream plus a shared value; the gate is a normalised dot
product of each stream against its key, put through a signed square root and a sigmoid;
``h += gate * value``. The table lookup itself (203 GB of FP8 rows served from NVMe) lives in
:mod:`freetoken.engram`; this module receives the rows already dequantized to bf16
``[T, n_hash_cols * head_dim]`` in the reference column order.
"""

from __future__ import annotations

import torch
from torch import nn

from .args import DeepseekV41Args
from .layers import Linear


class Engram(nn.Module):
    def __init__(self, args: DeepseekV41Args, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.dim = args.dim
        self.hc_mult = args.hc_mult
        self.eps = args.norm_eps
        self.clamp_value = 1e-6
        self.n_hash_cols = (args.engram_max_ngram_size - 1) * args.engram_n_heads
        self.head_dim = args.engram_head_dim
        self.wkv = Linear(
            self.n_hash_cols * self.head_dim, args.dim * (args.hc_mult + 1),
            kind="fp8", block=args.fp8_block_size,
        )
        self.q_weight = nn.Parameter(
            torch.ones(args.hc_mult, args.dim, dtype=torch.bfloat16), requires_grad=False
        )
        self.k_weight = nn.Parameter(
            torch.ones(args.hc_mult, args.dim, dtype=torch.bfloat16), requires_grad=False
        )

    @property
    def feat_dim(self) -> int:
        return self.n_hash_cols * self.head_dim

    def forward(self, h: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        """``h`` ``[1, T, hc, dim]`` residual streams, ``feats`` ``[T, n_hash_cols*head_dim]``
        bf16 (the dequantized table rows) -> the updated streams (same shape/dtype)."""
        shape, dtype = h.size(), h.dtype
        T = shape[0] * shape[1]
        hc, dim = self.hc_mult, self.dim
        kv = self.wkv(feats.view(T, -1))  # [T, (hc+1)*dim] bf16
        key = kv[:, : hc * dim].float().view(T, hc, dim)
        value = kv[:, hc * dim :].float()  # [T, dim]
        weight = self.q_weight.float() * self.k_weight.float()  # [hc, dim]: only used as a product
        hf = h.reshape(T, hc, dim).float()
        rstd = torch.rsqrt(hf.square().mean(-1) + self.eps) * torch.rsqrt(key.square().mean(-1) + self.eps)
        dot = (hf * weight * key).sum(-1) * rstd * dim ** -0.5  # [T, hc]
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(self.clamp_value).sqrt(), dot))
        out = hf + gate.unsqueeze(-1) * value.unsqueeze(1)
        return out.to(dtype).view(shape)


__all__ = ["Engram"]
