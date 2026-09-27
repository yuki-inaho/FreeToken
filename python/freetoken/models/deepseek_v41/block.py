"""The DeepSeek-V4.1 Transformer block: Single-Pass mHC around attention + MoE.

Reference ``Block.forward`` (inference/model.py):

    residual = x
    attn_pre, attn_post, attn_comb = hc_mixes(x, hc_attn_*)     # from THIS residual
    x = hc_pre(x, pre_mix)                                        # pre_mix: the PREVIOUS sublayer's
    x = attn(attn_norm(x)); x = hc_post(x, residual, attn_post, attn_comb)
    residual = x
    ffn_pre, ffn_post, ffn_comb = hc_mixes(x, hc_ffn_*)
    x = hc_pre(x, attn_pre)                                       # attention's pre feeds the FFN
    x = ffn(ffn_norm(x));  x = hc_post(x, residual, ffn_post, ffn_comb)
    return x, ffn_pre                                             # feeds the next block

So every sublayer's input mix was predicted by the sublayer before it (the "single pass":
the coefficient prediction no longer gates the input mixing of the same sublayer). The
first block receives a one-hot ``pre_mix`` and the model's head collapses the final stream
with the last block's ``ffn_pre``. There are no ``hc_head_*`` parameters in V4.1.

The attention call is injected (``attn_call``) so one block serves prefill (ragged
segments), decode (rows + snapshot) and the DSpark draft block without the block knowing
the addressing; the block only owns the residual-stream algebra.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import nn

from .args import DeepseekV41Args
from .layers import RMSNorm, hc_mixes, hc_post, hc_pre_apply

AttnCall = Callable[[torch.Tensor], torch.Tensor]
FfnCall = Callable[[torch.Tensor], torch.Tensor]


class Block(nn.Module):
    """One V4.1 block. ``attn`` / ``ffn`` are built by the caller (the main tower and the
    DSpark stages differ in both), the block owns the norms and the mHC parameters."""

    def __init__(self, layer_id: int, args: DeepseekV41Args, attn: nn.Module, ffn: nn.Module):
        super().__init__()
        self.layer_id = layer_id
        self.dim = args.dim
        self.norm_eps = args.norm_eps
        self.hc_mult = hc_mult = args.hc_mult
        self.hc_sinkhorn_iters = args.hc_sinkhorn_iters
        self.hc_eps = args.hc_eps
        self.attn = attn
        self.ffn = ffn
        self.attn_norm = RMSNorm(args.dim, self.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, self.norm_eps)
        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * args.dim
        f32 = torch.float32
        self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=f32), requires_grad=False)
        self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=f32), requires_grad=False)
        self.hc_attn_base = nn.Parameter(torch.empty(mix_hc, dtype=f32), requires_grad=False)
        self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc, dtype=f32), requires_grad=False)
        self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=f32), requires_grad=False)
        self.hc_ffn_scale = nn.Parameter(torch.empty(3, dtype=f32), requires_grad=False)

    # ----- mHC algebra -----------------------------------------------------------------
    def _mixes(self, x: torch.Tensor, which: str):
        if which == "attn":
            fn, scale, base = self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
        else:
            fn, scale, base = self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
        return hc_mixes(
            x, fn, scale, base, hc_mult=self.hc_mult, sinkhorn_iters=self.hc_sinkhorn_iters,
            hc_eps=self.hc_eps, norm_eps=self.norm_eps,
        )

    def pre_apply(self, x: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
        return hc_pre_apply(x, pre, hc_mult=self.hc_mult, dim=self.dim)

    def _post(self, x, residual, post, comb):
        return hc_post(x, residual, post, comb, hc_mult=self.hc_mult, dim=self.dim)

    def attn_input(self, x: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
        """The normalised attention input for stream ``x`` under ``pre_mix`` -- what the
        CED prefill feeds layer 20's compressor from the encoder output without running the
        block (the decoder's global KV is projected from exactly this tensor)."""
        return self.attn_norm(self.pre_apply(x, pre_mix))

    # ----- the block --------------------------------------------------------------------
    def forward(
        self, x: torch.Tensor, pre_mix: torch.Tensor, attn_call: AttnCall,
        ffn_call: FfnCall | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``x`` ``[b, s, hc, dim]`` (the residual streams), ``pre_mix`` ``[b*s, hc]`` fp32 (the
        previous sublayer's input mix). Returns the new streams and this block's ``ffn_pre``
        (the next block's ``pre_mix``)."""
        residual = x
        attn_pre, attn_post, attn_comb = self._mixes(x, "attn")
        y = self.attn_norm(self.pre_apply(x, pre_mix))
        y = attn_call(y)
        x = self._post(y, residual, attn_post, attn_comb)

        residual = x
        ffn_pre, ffn_post, ffn_comb = self._mixes(x, "ffn")
        y = self.ffn_norm(self.pre_apply(x, attn_pre))
        y = self.ffn(y) if ffn_call is None else ffn_call(y)
        x = self._post(y, residual, ffn_post, ffn_comb)
        return x, ffn_pre


__all__ = ["Block", "AttnCall", "FfnCall"]
