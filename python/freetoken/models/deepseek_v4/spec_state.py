"""Per-step speculative-decoding state for DeepSeek-V4 (the model-family half of the
speculative executor's scratch; the GDN/PLE family lives in ``engine/spec.py``).

A verify step runs ``k+1`` consecutive rows of one request through the decode path. The only
per-request state that cannot simply be overwritten on re-issue is the compressors' rolling
carry (one ring block per window page, read-modify-write per token), so ``Compressor.verify_rows``
folds the rows sequentially and stashes every row's post-fold carry here; ``commit`` re-points
the ring to the accepted prefix (device-side, idempotent). Everything else the verify writes
(window KV, compressed / indexer rows) is addressed by position and bounded by position on
read, so a rejected row's writes are never observed and are overwritten when the position is
re-issued.

The DSpark drafter's inputs live here too: the target's layer-40/41/42 stream means for the
verify rows (``main_hidden_rows``) and the block drafter's own KV scratch (``block_kv``), plus
the draft distributions ``chain_accept`` consumes (``draft_probs``, the algorithm-neutral
drafter -> acceptor interface).
"""

from __future__ import annotations

import torch

from .args import DeepseekV4Args


class DSV4SpecScratch:
    def __init__(
        self, args: DeepseekV4Args, num_draft_tokens: int, vocab_size: int,
        device: torch.device, dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.args = args
        self.device = device
        self.k = num_draft_tokens
        self.k1 = num_draft_tokens + 1
        k1 = self.k1
        # Carry intermediates per (layer, tier): [k1, ring_size, 2 * item] fp32.
        self.carry: dict[tuple[int, str], torch.Tensor] = {}
        for L, ratio in enumerate(tuple(args.compress_ratios)[: args.n_layers]):
            if ratio == 0:
                continue
            coff = 2 if ratio == 4 else 1
            self.carry[(L, "attn")] = torch.zeros(
                k1, coff * ratio, 2 * coff * args.head_dim, dtype=torch.float32, device=device
            )
            if ratio == 4:
                self.carry[(L, "idx")] = torch.zeros(
                    k1, 8, 2 * 2 * args.index_head_dim, dtype=torch.float32, device=device
                )
        n_targets = len(args.dspark_target_layer_ids)
        self.main_hidden_rows = torch.zeros(
            k1, max(n_targets, 1) * args.dim, dtype=dtype, device=device
        )
        self.block = args.dspark_block_size
        self.block_kv = [
            torch.zeros(max(self.block, 1), args.head_dim, dtype=dtype, device=device)
            for _ in range(args.n_mtp_layers)
        ]
        self.draft_probs = torch.zeros(
            max(num_draft_tokens, 1), vocab_size, dtype=torch.float32, device=device
        )
        # Per-step host facts set by the executor at staging (``set_step``) and the model's
        # verify branch (``window_slots``): the commit's addressing.
        self.boundary_row: int | None = None
        self.window_slots: torch.Tensor | None = None

    def set_step(self, cached_len: int, num_input_tokens: int) -> None:
        """Host-side per-step facts: the verify rows are positions ``cached_len ..
        cached_len + num_input_tokens - 1``; ``boundary_row`` is the first row on a new window
        page (None when the rows stay inside one page)."""
        P = self.args.window_size
        self.boundary_row = None
        for j in range(1, num_input_tokens):
            if (cached_len + j) % P == 0:
                self.boundary_row = j
                break
        self.window_slots = None

    def commit(self, transformer, num_accepted: torch.Tensor) -> None:
        """Re-point every compressor ring to the carry after the first ``num_accepted`` (``[1]``
        device scalar >= 1) verify rows. Device-side and idempotent (a drain-side truncation
        re-runs it with a smaller count)."""
        assert self.window_slots is not None, "commit before the verify forward staged its rows"
        for L, block in enumerate(transformer.layers):
            attn = block.attn
            if attn.compressor is None:
                continue
            attn.compressor.commit_carry(
                self.carry[(L, "attn")], num_accepted, self.window_slots, self.boundary_row
            )
            if attn.indexer is not None:
                attn.indexer.compressor.commit_carry(
                    self.carry[(L, "idx")], num_accepted, self.window_slots, self.boundary_row
                )


__all__ = ["DSV4SpecScratch"]
