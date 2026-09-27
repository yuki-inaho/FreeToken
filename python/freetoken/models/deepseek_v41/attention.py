"""DeepSeek-V4.1 attention: MLA over a 128-token sliding window plus CSA2 compressed KV.

Per layer (reference ``Attention``): ``qr = q_norm(wq_a(x))``, ``q = wq_b(qr)`` [64 x 512]
rotated on its last 64 dims (no per-head norm -- that was 0731), window KV
``kv_norm(wkv(x))`` rotated and FP8-rounded over the WHOLE 512-vector with 32-blocks, then a
sparse attention over ``[window picks | compressed picks]`` with a per-head sink, the inverse
rotation on the output, the grouped ``wo_a`` einsum and ``wo_b``.

CSA2 roles come from the args (``args.mode(L)``):

* ``window``   -- layers 0/1 and the DSpark stages: window picks only.
* ``full``     -- a kv-source: runs the compressor on ITS input, publishes index keys from
                  the pre-RoPE latent, rotates + FP4(E4M3/16)-rounds the latent into its own
                  ``cmp_pool``, scores with its own indexer and publishes the picks.
* ``reindex``  -- scores the OWNER's keys with its own query (inside the candidate pool
                  when it sits after the candidate source) and publishes fresh picks.
* ``reuse``    -- attends with the last published picks; no compressor, no indexer.

Addressing is the attention backend's (window slots, compressed rows via the owner map,
the decode snapshot); the layer computes projections and picks. Prefill supports two window
geometries: ``causal`` (the 0731 cold/extend windows) and ``replay`` (the CED decoder replay:
a query at ``t`` sees window keys in ``[max(s, t-127), t]`` for a replay starting at ``s``,
and never the slots before ``s``).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from freetoken.core import get_global_ctx
from freetoken.kernel.triton.dsv4.fp8_linear import act_quant_fp8_inplace
from freetoken.models.deepseek_v4.ops import apply_rotary_emb, apply_rotary_emb_decode, get_freqs_cis

from .args import DeepseekV41Args
from .compress import Compressor
from .indexer import Indexer
from .layers import Linear, RMSNorm, get_window_topk_idxs
from .runtime import SharedAttentionRuntime

Segment = tuple  # (offset, n, table_idx, start_pos)


def _fp4_e4m3_inplace(x: torch.Tensor, group: int) -> torch.Tensor:
    from freetoken.kernel.triton.dsv4.fp8_linear import fp4_act_quant_e4m3_inplace

    return fp4_act_quant_e4m3_inplace(x, group)


class Attention(nn.Module):
    def __init__(self, layer_id: int, args: DeepseekV41Args):
        super().__init__()
        self.layer_id = layer_id
        self.args = args
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.q_lora_rank = args.q_lora_rank
        self.o_lora_rank = args.o_lora_rank
        self.head_dim = args.head_dim
        self.rope_head_dim = args.rope_head_dim
        self.n_groups = args.o_groups
        self.window_size = args.window_size
        self.compress_ratio = args.compress_ratios[layer_id]
        self.mode = args.mode(layer_id)
        self.cmp_owner = args.cmp_owner(layer_id)
        self.eps = args.norm_eps
        self.fp8_block = args.fp8_block_size
        self.cmp_fp4_group = args.cmp_fp4_group
        blk = self.fp8_block

        self.attn_sink = nn.Parameter(torch.empty(self.n_heads, dtype=torch.float32), requires_grad=False)
        self.wq_a = Linear(self.dim, self.q_lora_rank, kind="fp8", block=blk)
        self.q_norm = RMSNorm(self.q_lora_rank, self.eps)
        self.wq_b = Linear(self.q_lora_rank, self.n_heads * self.head_dim, kind="fp8", block=blk)
        self.wkv = Linear(self.dim, self.head_dim, kind="fp8", block=blk)
        self.kv_norm = RMSNorm(self.head_dim, self.eps)
        # wo_a: FP8 in the checkpoint, dequantized to bf16 at load (the reference's bf16 grouped einsum).
        wo_a_rows = self.n_groups * args.o_lora_rank
        wo_a_k = self.n_heads * self.head_dim // self.n_groups
        self.wo_a = nn.Parameter(torch.empty(wo_a_rows, wo_a_k, dtype=torch.bfloat16), requires_grad=False)
        self.wo_b = Linear(self.n_groups * args.o_lora_rank, self.dim, kind="fp8", block=blk)
        self.softmax_scale = self.head_dim ** -0.5

        self.compressor = Compressor(args, layer_id) if args.is_kv_source(layer_id) else None
        self.indexer = (
            Indexer(args, layer_id)
            if (args.is_index_source(layer_id) or args.is_kv_source(layer_id)) else None
        )
        if self.compress_ratio:
            original_seq_len, rope_theta = args.original_seq_len, args.compress_rope_theta
        else:
            original_seq_len, rope_theta = 0, args.rope_theta
        self._freqs_params = (
            self.rope_head_dim, args.max_seq_len, original_seq_len,
            rope_theta, args.rope_factor, args.beta_fast, args.beta_slow,
        )
        self.freqs_cis: torch.Tensor | None = None

    # ----- backend / pool handles (read live; a runtime rebuild swaps the pool) -----------
    @property
    def attn(self):
        return get_global_ctx().attn_backend

    def bind(self, pool, device: torch.device) -> None:
        self.P = pool.P
        self.freqs_cis = get_freqs_cis(*self._freqs_params, device)
        if self.compressor is not None:
            self.compressor.bind_paged(pool, device)
        if self.indexer is not None:
            self.indexer.bind_paged(pool, self.freqs_cis, device)

    def reset(self) -> None:
        if self.compressor is not None:
            self.compressor.reset()

    # ----- projections ------------------------------------------------------------------
    def _qkv(self, x: torch.Tensor, freqs: torch.Tensor, decode: bool):
        rd = self.rope_head_dim
        rope = apply_rotary_emb_decode if decode else apply_rotary_emb
        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr).unflatten(-1, (self.n_heads, self.head_dim))
        rope(q[..., -rd:], freqs)
        kv = self.kv_norm(self.wkv(x))
        rope(kv[..., -rd:], freqs)
        act_quant_fp8_inplace(kv, self.fp8_block)  # whole vector, rope tail included
        return qr, q, kv

    def _wo(self, o: torch.Tensor, bsz: int, seqlen: int) -> torch.Tensor:
        o = o.reshape(bsz, seqlen, self.n_groups, -1)
        wo_a = self.wo_a.view(self.n_groups, self.o_lora_rank, -1)
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a).flatten(2)
        return self.wo_b(o)

    def _finalize_latent(self, latent: torch.Tensor, freqs: torch.Tensor, decode: bool) -> torch.Tensor:
        """Pre-RoPE latent -> what the compressed pool stores: rotated at the group's first
        position (compressed rope regime) and rounded to E2M1 with one E4M3 scale per
        ``cmp_fp4_group`` channels (in place, bf16 storage)."""
        rd = self.rope_head_dim
        (apply_rotary_emb_decode if decode else apply_rotary_emb)(latent[..., -rd:], freqs)
        _fp4_e4m3_inplace(latent, self.cmp_fp4_group)
        return latent

    # ----- shared-KV production (kv-source layers) --------------------------------------
    def produce_shared_kv(self, x_seg: torch.Tensor, start_pos: int, ti: int, window_slots: torch.Tensor) -> None:
        """Run this kv-source's compressor over ``x_seg`` ``[1, n, dim]`` (this layer's
        normalised attention input for ``[start_pos, start_pos+n)`` of request ``ti``),
        publish the index keys and the compressed KV. Used inside the layer's own prefill
        and, hoisted, by the CED prefill for layer ``decoder_start`` over the encoder output."""
        assert self.compressor is not None and self.indexer is not None
        latent, starts = self.compressor.prefill(x_seg, start_pos, window_slots, ti)
        if latent is None:
            return
        self.indexer.publish_k_prefill(latent, starts, ti)
        latent = self._finalize_latent(latent, self.freqs_cis.index_select(0, starts), decode=False)
        rows = self.attn.compress_rows_of(ti, starts, self.compress_ratio)
        self.attn.scatter_compressed(self.layer_id, "attn", rows, latent[0])

    def _produce_decode(self, x, pos, rows, prev_window_slots, window_slots, spec) -> None:
        assert self.compressor is not None and self.indexer is not None
        B = x.size(0)
        if spec is None:
            latent, completed = self.compressor.decode_step(x, pos, prev_window_slots, window_slots)
        else:
            latent, completed = self.compressor.verify_rows(
                x.view(B, -1), pos, prev_window_slots[:1], window_slots,
                # ratio-1 (non-gated) kv sources have no carry (the scratch allocates them for
                # ratio > 1 only) and verify_rows returns before reading it; the first real
                # DSpark verify on the 5090 died here on KeyError (20, "attn").
                spec.scratch.carry.get((self.layer_id, "attn")), boundary_row=spec.scratch.boundary_row,
            )
        group_pos = self.compressor.group_positions(pos)
        self.indexer.publish_k_decode(latent, pos, rows, completed, group_pos)
        latent = self._finalize_latent(latent, self.freqs_cis.index_select(0, group_pos), decode=True)
        dst = self.attn.decode_compress_rows(rows, pos, self.compress_ratio, self.layer_id, "attn", completed)
        self.attn.scatter_compressed(self.layer_id, "attn", dst, latent.view(B, -1))

    # ----- prefill ----------------------------------------------------------------------
    def _window_picks(self, ti: int, start_pos: int, n: int, window_mode: str, device):
        """Per-query GLOBAL window slots ``[1, n, w]`` for one segment plus the segment's own
        slots ``[n]`` (for ``store_window`` / the carry ring)."""
        win = self.window_size
        end = start_pos + n
        slots = self.attn.window_slots_of(ti, start_pos, end)
        if window_mode == "replay" or start_pos == 0:
            # cold geometry over this segment only: query i sees [max(0, i-127), i] of it
            cols = get_window_topk_idxs(win, 1, n, 0).to(device)
            return self.attn.win_cols_to_global(cols, slots), slots
        w_lo = max(0, start_pos - win + 1)
        ws_pool = self.attn.window_slots_of(ti, w_lo, end)
        abs_p = start_pos + torch.arange(n, device=device).unsqueeze(1)
        cand = (abs_p - win + 1).clamp(min=w_lo) + torch.arange(win, device=device)
        cols = torch.where(cand > abs_p, -1, cand - w_lo).unsqueeze(0)
        return self.attn.win_cols_to_global(cols, ws_pool), slots

    def forward_ragged(
        self, x: torch.Tensor, segments: list[Segment], flat_positions: torch.Tensor,
        runtime: SharedAttentionRuntime, window_mode: str = "causal",
    ) -> torch.Tensor:
        """Ragged prefill over the concatenated segments of ``x`` ``[1, T, dim]``."""
        win, ratio = self.window_size, self.compress_ratio
        device = x.device
        _, T, _ = x.size()
        if len(segments) == 1:
            s0 = segments[0][3]
            freqs = self.freqs_cis[s0:s0 + T]
        else:
            freqs = self.freqs_cis.index_select(0, flat_positions)
        qr, q, kv = self._qkv(x, freqs, decode=False)

        win_parts, cmp_parts = [], []
        publish = self.mode in ("full", "reindex")
        max_c = 0
        for off, n, ti, start_pos in segments:
            win_global, slots = self._window_picks(ti, start_pos, n, window_mode, device)
            self.attn.store_window(kv[0, off:off + n], self.layer_id, slots)
            win_parts.append(win_global)
            if self.mode == "window":
                continue
            if self.mode == "full" and runtime.produce_kv:
                self.produce_shared_kv(x[:, off:off + n], start_pos, ti, slots)
            if publish:
                blocks = self.indexer.score_prefill(
                    x[:, off:off + n], qr[:, off:off + n], start_pos, ti, runtime
                )  # [1, n, k] block ids / -1
                cg = self.attn.blocks_to_global(blocks, ratio, ti=ti)
                cmp_parts.append(cg)
                max_c = max(max_c, cg.shape[-1])

        if self.mode == "reuse":
            topk_idxs, n_window, _ = runtime.picks()
            o = self.attn.attend(
                q, self.layer_id, topk_idxs, n_window, self.attn_sink, self.softmax_scale,
                has_compression=True,
            )
        else:
            n_window = win if len(segments) > 1 else win_parts[0].shape[-1]
            flat = []
            for i in range(len(segments)):
                wg = win_parts[i]
                if wg.shape[-1] < n_window:
                    wg = F.pad(wg, (0, n_window - wg.shape[-1]), value=-1)
                if publish:
                    cg = cmp_parts[i]
                    if cg.shape[-1] < max_c:
                        cg = F.pad(cg, (0, max_c - cg.shape[-1]), value=-1)
                    flat.append(torch.cat([wg, cg], dim=-1))
                else:
                    flat.append(wg)
            topk_idxs = (flat[0] if len(flat) == 1 else torch.cat(flat, dim=1)).int()
            o = self.attn.attend(
                q, self.layer_id, topk_idxs, n_window, self.attn_sink, self.softmax_scale,
                has_compression=publish,
            )
            if publish:
                runtime.publish_picks(topk_idxs, n_window, None)
        apply_rotary_emb(o[..., -self.rope_head_dim:], freqs, True)
        return self._wo(o, 1, T)

    # ----- decode / verify ---------------------------------------------------------------
    def decode_step(
        self, x: torch.Tensor, pos: torch.Tensor, rows: torch.Tensor, cmp_stage_cap: int,
        wctx, runtime: SharedAttentionRuntime, spec=None,
    ) -> torch.Tensor:
        """Batched single-token attention (eager / graph). See the 0731 layer for the
        addressing contract (``rows`` index the decode snapshot; ``wctx`` is the hoisted
        window context). ``spec`` marks a speculative verify (k+1 rows of one request)."""
        B = x.size(0)
        win, ratio, rd = self.window_size, self.compress_ratio, self.rope_head_dim
        if wctx is None:
            wctx = get_global_ctx().batch.attn_metadata.window_ctx(pos, rows)
        window_slots, prev_window_slots, window_slots_topk = wctx
        freqs_t = self.freqs_cis.index_select(0, pos)
        qr, q, kv = self._qkv(x, freqs_t, decode=True)
        self.attn.store_window(kv.view(B, -1), self.layer_id, window_slots)

        if self.mode == "window":
            topk_idxs, n_window, cmp_counts = window_slots_topk.int(), win, None
        elif self.mode == "reuse":
            topk_idxs, n_window, cmp_counts = runtime.picks()
        else:
            if self.mode == "full" and runtime.produce_kv:
                self._produce_decode(x, pos, rows, prev_window_slots, window_slots, spec)
            n_cmp_stage = (cmp_stage_cap + 1) // ratio
            valid = (pos + 1) // ratio
            blocks = self.indexer.score_decode(x, qr, pos, n_cmp_stage, runtime)  # [B, 1, k]
            compress_idxs = self.attn.blocks_to_global(blocks, ratio, rows=rows)
            topk_idxs = torch.cat([window_slots_topk, compress_idxs], dim=-1).int()
            n_window = win
            cmp_counts = valid.clamp(max=compress_idxs.shape[-1]).to(torch.int32).view(B, 1)
            runtime.publish_picks(topk_idxs, n_window, cmp_counts)
        o = self.attn.attend(
            q, self.layer_id, topk_idxs, n_window, self.attn_sink, self.softmax_scale,
            cmp_counts=cmp_counts, has_compression=self.mode != "window",
        )
        apply_rotary_emb_decode(o[..., -rd:], freqs_t, True)
        return self._wo(o, B, 1)


__all__ = ["Attention", "Segment"]
