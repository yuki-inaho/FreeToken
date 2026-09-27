"""The CSA2 KV compressor (kv-source layers only): ratio 2 across the encoder, ratio 1 in the
decoder.

Reference ``Compressor`` (inference/model.py): a ratio-1 layer projects every token
(``norm(wkv(x))``, bf16, no gate, no state); a ratio-``r`` layer pools ``r`` consecutive
tokens with a learned softmax gate in fp32 (``wkv``/``wgate`` promoted to fp32), carrying a
partial group across decode steps. V4.1 dropped the 0731 overlap and absolute positional
embedding, so the carry is just the ``r`` slots of the group in progress.

This module returns the PRE-RoPE latent: the owner's indexer projects its index keys from
it, and only afterwards does the attention layer rotate it (compressed rope regime at the
group's first position), round it to E2M1 with one E4M3 scale per 16 channels and scatter it
into the owner's ``cmp_pool``. Addressing (pool rows, the per-page carry ring, scratch rows)
belongs to the attention backend, exactly as for the 0731 compressor; a page (128) is a
multiple of every ratio, so a group never straddles a page and a radix re-prefill at a page
boundary starts from the reset carry.
"""

from __future__ import annotations

import torch
from torch import nn

from freetoken.core import get_global_ctx
from freetoken.kernel.triton.dsv4.compress import gated_pool

from .args import DeepseekV41Args
from .layers import Linear, RMSNorm

TIER = "attn"


class Compressor(nn.Module):
    def __init__(self, args: DeepseekV41Args, layer_id: int):
        super().__init__()
        ratio = args.compress_ratios[layer_id]
        assert ratio >= 1, f"layer {layer_id} is not a compressing layer (ratio {ratio})"
        self.layer_id = layer_id
        self.tier = TIER
        self.dim = args.dim
        self.head_dim = args.head_dim
        self.compress_ratio = ratio
        self.gated = ratio > 1
        # ratio 1: a plain projection in the checkpoint's bf16; ratio>1: fp32 like the
        # reference (the softmax pooling runs in fp32, the weights are promoted to match).
        self.wkv = Linear(args.dim, args.head_dim, kind="fp32" if self.gated else "bf16")
        if self.gated:
            self.wgate = Linear(args.dim, args.head_dim, kind="fp32")
        else:
            self.register_module("wgate", None)
        self.norm = RMSNorm(args.head_dim, args.norm_eps)
        # Ring geometry (ratio>1 only): one carry block of ``ratio`` rows per window page,
        # each row ``kv | score`` (2 * head_dim fp32).
        self.ring_size = ratio if self.gated else 0
        self.item_size = args.head_dim
        self.P = args.window_size
        self._device: torch.device | None = None

    # ----- backend handles (read live, never cached: a runtime rebuild swaps the pool) ---
    @property
    def attn(self):
        return get_global_ctx().attn_backend

    def bind_paged(self, pool, device: torch.device) -> None:
        self.P = pool.P
        self._device = device

    def reset(self) -> None:  # the register carry is per call; nothing persists in the module
        return None

    def group_positions(self, pos: torch.Tensor) -> torch.Tensor:
        """RoPE position of the group a token at ``pos`` completes: its first token."""
        return (pos + 1 - self.compress_ratio).clamp_min(0)

    # ----- prefill / page-aligned extend -----------------------------------------------
    def prefill(
        self, x: torch.Tensor, start_pos: int, window_slots: torch.Tensor, ti: int
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Compress the tokens ``[start_pos, start_pos + n)`` of request ``ti``.

        ``x`` is ``[1, n, dim]`` (this layer's attention input), ``window_slots`` the tokens'
        window slots (for the carry ring), ``start_pos`` 0 or page-aligned (a radix hit /
        chunk continuation; the carry at a page boundary is the reset state). Returns the
        pre-RoPE latent ``[1, g, head_dim]`` (None when no group completed) and the absolute
        first positions of those ``g`` groups; the caller finishes and scatters it.
        """
        bsz, n, _ = x.size()
        assert bsz == 1
        ratio = self.compress_ratio
        dtype = x.dtype
        device = x.device
        if not self.gated:
            latent = self.norm(self.wkv(x))  # [1, n, head_dim] bf16
            starts = torch.arange(start_pos, start_pos + n, device=device)
            return latent, starts
        assert start_pos % self.P == 0, "a compressor extend must start on a window page"
        xf = x.float()
        kv = self.wkv(xf)          # [1, n, head_dim] fp32
        score = self.wgate(xf)
        remainder = n % ratio
        cutoff = n - remainder
        # Every page boundary inside (start_pos, end] gets the reset carry (a page is a
        # multiple of the ratio, so no group is in flight there); the unaligned tail page gets
        # the in-progress group so the next decode step can finish it.
        end = start_pos + n
        self.attn.write_boundary_carries(
            layer_id=self.layer_id, tier=self.tier, ratio=ratio, overlap=False,
            ring_size=self.ring_size, ape=None, kv=kv, score=score, lo=start_pos, hi=end,
            window_slots=window_slots,
        )
        if end % self.P != 0:
            ks = kv.new_zeros(ratio, self.item_size)
            ss = kv.new_full((ratio, self.item_size), float("-inf"))
            if remainder:
                ks[:remainder] = kv[0, cutoff:]
                ss[:remainder] = score[0, cutoff:]
            self.attn.write_carry(
                self.layer_id, self.tier, int(window_slots[-1].item()), self.ring_size,
                torch.cat([ks, ss], dim=-1),
            )
        if cutoff == 0:
            return None, torch.empty(0, dtype=torch.int64, device=device)
        kvg = kv[:, :cutoff].unflatten(1, (-1, ratio))          # [1, g, ratio, head_dim]
        scg = score[:, :cutoff].unflatten(1, (-1, ratio))
        pooled = (kvg * scg.softmax(dim=2)).sum(dim=2)           # [1, g, head_dim] fp32
        latent = self.norm(pooled.to(dtype))
        starts = torch.arange(start_pos, start_pos + cutoff, ratio, device=device)
        return latent, starts

    # ----- decode ---------------------------------------------------------------------
    def decode_step(
        self, x: torch.Tensor, pos: torch.Tensor, prev_window_slots: torch.Tensor,
        window_slots: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Batched single-token update. Returns the pre-RoPE latent ``[B, 1, head_dim]`` and
        ``completed`` ``[B]`` (whether this token closed a group; at ratio 1 always True).
        The rolling carry lives per row in the paged ring: read from the previous token's
        page, advanced, written to this token's page (see the 0731 compressor)."""
        B = x.size(0)
        ratio = self.compress_ratio
        dtype = x.dtype
        x2 = x.view(B, -1)
        if not self.gated:
            latent = self.norm(self.wkv(x2.view(B, 1, -1)))
            return latent, torch.ones(B, dtype=torch.bool, device=x.device)
        item = self.item_size
        xf = x2.float()
        kv = self.wkv(xf).view(B, 1, item)
        score = self.wgate(xf).view(B, 1, item)
        idx_mod = pos % ratio
        completed = (pos + 1) % ratio == 0
        block = self.attn.read_carry_blocks(
            self.layer_id, self.tier, prev_window_slots, self.ring_size
        )  # [B, ring, 2*item]
        ks = block[..., :item].clone()
        ss = block[..., item:].clone()
        ks.scatter_(1, idx_mod.view(B, 1, 1).expand(B, 1, item), kv)
        ss.scatter_(1, idx_mod.view(B, 1, 1).expand(B, 1, item), score)
        pooled = gated_pool(ks, ss, dtype)  # [B, 1, item]
        self.attn.write_carry_blocks(
            self.layer_id, self.tier, window_slots, self.ring_size, torch.cat([ks, ss], dim=-1)
        )
        return self.norm(pooled), completed

    def verify_rows(
        self, x: torch.Tensor, pos: torch.Tensor, prev_window_slot0: torch.Tensor,
        window_slots: torch.Tensor, intermediates: torch.Tensor,
        boundary_row: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Speculative verify: ``k1`` CONSECUTIVE tokens of one request folded sequentially
        through one carry read at row 0's previous page; every row's post-fold carry goes to
        ``intermediates`` ``[k1, ring, 2*item]`` for :meth:`commit_carry`. Returns the latent
        ``[k1, 1, head_dim]`` and ``completed`` ``[k1]``."""
        k1 = x.size(0)
        ratio = self.compress_ratio
        dtype = x.dtype
        x2 = x.view(k1, -1)
        if not self.gated:
            latent = self.norm(self.wkv(x2.view(k1, 1, -1)))
            return latent, torch.ones(k1, dtype=torch.bool, device=x.device)
        item = self.item_size
        kv = self.wkv(x2.float()).view(k1, 1, item)
        score = self.wgate(x2.float()).view(k1, 1, item)
        idx_mod = pos % ratio
        completed = (pos + 1) % ratio == 0
        block = self.attn.read_carry_blocks(
            self.layer_id, self.tier, prev_window_slot0.view(1), self.ring_size
        )
        ks = block[..., :item].clone()
        ss = block[..., item:].clone()
        outs = []
        for i in range(k1):
            slot = idx_mod[i : i + 1].view(1, 1, 1).expand(1, 1, item)
            ks.scatter_(1, slot, kv[i : i + 1])
            ss.scatter_(1, slot, score[i : i + 1])
            outs.append(gated_pool(ks, ss, dtype))
            intermediates[i].copy_(torch.cat([ks[0], ss[0]], dim=-1))
        if boundary_row is not None and boundary_row >= 1:
            self.attn.write_carry_blocks(
                self.layer_id, self.tier, window_slots[boundary_row - 1 : boundary_row],
                self.ring_size, intermediates[boundary_row - 1 : boundary_row],
            )
        self.attn.write_carry_blocks(
            self.layer_id, self.tier, window_slots[-1:], self.ring_size,
            torch.cat([ks, ss], dim=-1),
        )
        return self.norm(torch.cat(outs, dim=0)), completed

    def commit_carry(
        self, intermediates: torch.Tensor, num_accepted: torch.Tensor,
        window_slots: torch.Tensor, boundary_row: int | None,
    ) -> None:
        """Re-point the ring to the carry after the accepted prefix (same contract as the
        0731 compressor: idempotent, device-side)."""
        if not self.gated:
            return
        sel = (num_accepted - 1).clamp_min(0).view(1)
        self.attn.write_carry_blocks(
            self.layer_id, self.tier, window_slots.index_select(0, sel), self.ring_size,
            intermediates.index_select(0, sel),
        )
        if boundary_row is not None and boundary_row >= 1:
            sel2 = torch.minimum(sel, torch.full_like(sel, boundary_row - 1))
            self.attn.write_carry_blocks(
                self.layer_id, self.tier, window_slots.index_select(0, sel2), self.ring_size,
                intermediates.index_select(0, sel2),
            )


__all__ = ["Compressor", "TIER"]
