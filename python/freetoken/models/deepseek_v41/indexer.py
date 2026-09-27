"""The CSA2 Lightning Indexer with the Hierarchical Sparse Indexer's candidate pool.

Roles (tech report 2.3): a kv-source layer ("Full") owns the index KEYS -- projected from
its pre-RoPE compressed latent (``k_norm(wk(latent))``, rotated at the group's first
position, MXFP4-rounded) and stored in the owner's ``idx_pool`` at the same rows as the
compressed KV. Every index source (Full and "Reindex") runs its own QUERY (``wq_b(qr)``
per head, rotated, MXFP4-rounded) and gate (``weights_proj(x)``), scores the shared keys
(``sum_h relu(q_h . k) * w_h``) and publishes ``index_topk`` picks; "Reuse" layers take the
last published picks and have no indexer at all.

The decoder's first Full layer (the candidate source) additionally keeps the best
``candidate_topk_blocks`` blocks of ``candidate_block_size`` positions (block score = max
over its positions, the block holding the newest position pinned); later Reindex layers
score only inside that pool. v1 scores every position and masks the rest (the reference
semantics); scoring only the gathered candidates is the follow-up kernel.

Addressing (pool rows, the decode snapshot) stays in the attention backend; the backend
resolves this layer's ``idx`` tier to its OWNER's pool, so a Reindex layer reads the keys
the Full layer before it published.
"""

from __future__ import annotations

import torch
from torch import nn

from freetoken.core import get_global_ctx
from freetoken.kernel.triton.dsv4.fp8_linear import fp4_act_quant_inplace

from .args import DeepseekV41Args
from .layers import Linear, RMSNorm
from .runtime import SharedAttentionRuntime

# Query rows scored per launch: bounds the [S, n_positions] fp32 score slab (a ratio-1
# decoder layer at 128K context is 512 KB per query row).
_SCORE_SLAB_BYTES = 256 << 20


def _helpers():
    # The pure selection helpers live next to the backend's indexer addressing.
    from freetoken.attention import dsv4_indexer as h

    return h


class Indexer(nn.Module):
    def __init__(self, args: DeepseekV41Args, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.compress_ratio = args.compress_ratios[layer_id]
        assert self.compress_ratio >= 1
        self.owns_k = args.is_kv_source(layer_id)
        self.is_index_source = args.is_index_source(layer_id)
        self.is_candidate_source = layer_id == args.candidate_source_layer
        self.uses_candidates = args.uses_candidates(layer_id)
        self.candidate_topk_blocks = args.candidate_topk_blocks
        self.candidate_block_size = args.candidate_block_size
        self.dim = args.dim
        self.n_heads = args.index_n_heads
        self.head_dim = args.index_head_dim
        self.rope_head_dim = args.rope_head_dim
        self.index_topk = args.index_topk
        self.softmax_scale = self.head_dim ** -0.5
        if self.is_index_source:
            self.wq_b = Linear(
                args.q_lora_rank, self.n_heads * self.head_dim, kind="fp8",
                block=args.fp8_block_size,
            )
            self.weights_proj = Linear(args.dim, self.n_heads, kind="bf16")
        else:
            self.register_module("wq_b", None)
            self.register_module("weights_proj", None)
        if self.owns_k:
            self.wk = Linear(args.head_dim, self.head_dim, kind="bf16")
            self.k_norm = RMSNorm(self.head_dim, args.norm_eps)
        else:
            self.register_module("wk", None)
            self.register_module("k_norm", None)
        self.freqs_cis: torch.Tensor | None = None  # the layer's (compressed-regime) rope table

    @property
    def attn(self):
        return get_global_ctx().attn_backend

    def bind_paged(self, pool, freqs_cis: torch.Tensor, device: torch.device) -> None:
        self.freqs_cis = freqs_cis
        self._device = device

    # ----- index keys (kv-source layers) -----------------------------------------------
    def _keys_from_latent(self, latent: torch.Tensor, freqs: torch.Tensor, decode: bool) -> torch.Tensor:
        from freetoken.models.deepseek_v4.ops import apply_rotary_emb, apply_rotary_emb_decode

        rd = self.rope_head_dim
        k = self.k_norm(self.wk(latent))  # [.., g, D] bf16
        if decode:
            apply_rotary_emb_decode(k[..., -rd:], freqs)
        else:
            apply_rotary_emb(k[..., -rd:], freqs)
        fp4_act_quant_inplace(k, 32)
        return k

    def publish_k_prefill(self, latent: torch.Tensor, starts: torch.Tensor, ti: int) -> None:
        """``latent`` ``[1, g, head_dim]`` pre-RoPE, ``starts`` ``[g]`` absolute first
        positions of the groups -> index keys into the owner's idx pool (this layer)."""
        assert self.owns_k and self.freqs_cis is not None
        if latent is None or starts.numel() == 0:
            return
        k = self._keys_from_latent(latent, self.freqs_cis.index_select(0, starts), decode=False)
        rows = self.attn.compress_rows_of(ti, starts, self.compress_ratio)
        self.attn.scatter_compressed(self.layer_id, "idx", rows, k[0])

    def publish_k_decode(
        self, latent: torch.Tensor, pos: torch.Tensor, rows: torch.Tensor,
        completed: torch.Tensor, group_pos: torch.Tensor,
    ) -> None:
        """Decode/verify twin: ``latent`` ``[B, 1, head_dim]``; rows whose group did not
        complete land on their scratch row (a discarded write) -- graph-safe like the cmp
        scatter."""
        assert self.owns_k and self.freqs_cis is not None
        B = latent.size(0)
        k = self._keys_from_latent(latent, self.freqs_cis.index_select(0, group_pos), decode=True)
        dst = self.attn.decode_compress_rows(
            rows, pos, self.compress_ratio, self.layer_id, "idx", completed
        )
        self.attn.scatter_compressed(self.layer_id, "idx", dst, k.view(B, -1))

    # ----- queries + selection (index-source layers) ------------------------------------
    def _query(self, qr: torch.Tensor, freqs: torch.Tensor, decode: bool) -> torch.Tensor:
        from freetoken.models.deepseek_v4.ops import apply_rotary_emb, apply_rotary_emb_decode

        rd = self.rope_head_dim
        q = self.wq_b(qr).unflatten(-1, (self.n_heads, self.head_dim))
        if decode:
            apply_rotary_emb_decode(q[..., -rd:], freqs)
        else:
            apply_rotary_emb(q[..., -rd:], freqs)
        fp4_act_quant_inplace(q, 32)
        return q

    def _gate(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights_proj(x) * (self.softmax_scale * self.n_heads ** -0.5)

    def score_prefill(
        self, x: torch.Tensor, qr: torch.Tensor, start_pos: int, ti: int,
        runtime: SharedAttentionRuntime,
    ) -> torch.Tensor:
        """Picks for the queries ``[start_pos, start_pos + n)`` of request ``ti``:
        ``[1, n, min(index_topk, n_positions)]`` block indices (``-1`` past the visible
        count). ``x`` ``[1, n, dim]`` is the attention input, ``qr`` the normalised low-rank
        query. Also builds (candidate source) or applies (Reindex) the candidate pool."""
        assert self.is_index_source and self.freqs_cis is not None
        h = _helpers()
        _, n, _ = x.size()
        ratio = self.compress_ratio
        device = x.device
        end_pos = start_pos + n
        n_pos = end_pos // ratio
        q = self._query(qr, self.freqs_cis[start_pos:end_pos], decode=False)  # [1, n, H, D]
        weights = self._gate(x)  # [1, n, H]
        topk = min(self.index_topk, n_pos)
        if n_pos == 0:
            return torch.empty(1, n, 0, dtype=torch.int32, device=device)
        keys = self.attn.indexer_keys(ti, n_pos, ratio, self.layer_id, 1)  # [1, n_pos, D]
        sub = max(1, min(n, _SCORE_SLAB_BYTES // max(1, 4 * n_pos)))
        picks, cands = [], []
        for s0 in range(0, n, sub):
            s1 = min(n, s0 + sub)
            scores = self.attn.indexer_prefill_logits(q[:, s0:s1], keys, weights[:, s0:s1])
            # positions visible to query p: [0, (p+1)//ratio)
            live = ((start_pos + torch.arange(s0 + 1, s1 + 1, device=device)) // ratio).view(1, -1, 1)
            scores.masked_fill_(torch.arange(n_pos, device=device).view(1, 1, -1) >= live, float("-inf"))
            if self.is_candidate_source:
                blocks = h.select_candidate_block_ids(
                    scores, live, self.candidate_topk_blocks, self.candidate_block_size,
                )  # [1, S, topk_blocks] block ids, -1 padded
                cands.append(blocks)
            elif self.uses_candidates:
                assert runtime.cand_blocks is not None, (
                    f"layer {self.layer_id} re-indexes inside the candidate pool but no "
                    "candidate source ran before it"
                )
                mask = h.candidate_mask_from_blocks(
                    runtime.cand_blocks[:, s0:s1], n_pos, self.candidate_block_size
                )
                scores.masked_fill_(~mask, float("-inf"))
            picks.append(h.indexer_select_prefill(scores, live, topk))
        if self.is_candidate_source:
            runtime.cand_blocks = torch.cat(cands, dim=1)
            runtime.cand_n_positions = n_pos
        return torch.cat(picks, dim=1).to(torch.int32)

    def score_decode(
        self, x: torch.Tensor, qr: torch.Tensor, pos: torch.Tensor, n_stage: int,
        runtime: SharedAttentionRuntime,
    ) -> torch.Tensor:
        """Batched decode/verify picks ``[B, 1, min(index_topk, n_stage)]`` (block indices,
        ``-1`` past each row's visible count). Scores the first ``n_stage`` positions off the
        decode snapshot with the work bounded by the live count in device memory."""
        assert self.is_index_source and self.freqs_cis is not None
        h = _helpers()
        B = x.size(0)
        ratio = self.compress_ratio
        q = self._query(qr, self.freqs_cis.index_select(0, pos), decode=True)  # [B, 1, H, D]
        weights = self._gate(x)  # [B, 1, H]
        valid = (pos + 1) // ratio  # [B]
        scores = self.attn.indexer_decode_scores(
            q.reshape(B, self.n_heads, self.head_dim), weights.reshape(B, self.n_heads),
            valid, n_stage, ratio, self.layer_id,
        ).float()  # [B, n_stage], -inf past valid
        if self.is_candidate_source:
            runtime.cand_blocks = h.select_candidate_block_ids(
                scores, valid.view(B, 1), self.candidate_topk_blocks, self.candidate_block_size,
            )
            runtime.cand_n_positions = n_stage
        elif self.uses_candidates:
            assert runtime.cand_blocks is not None
            mask = h.candidate_mask_from_blocks(
                runtime.cand_blocks, n_stage, self.candidate_block_size
            )
            scores = scores.masked_fill(~mask, float("-inf"))
        picks = h.indexer_select_prefill(scores.view(B, 1, n_stage), valid.view(B, 1, 1),
                                         min(self.index_topk, n_stage))
        return picks.to(torch.int32)


__all__ = ["Indexer"]
