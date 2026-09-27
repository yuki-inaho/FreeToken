"""Per-step speculative-decoding state for DeepSeek-V4.1 (the DSV4 family's half of the
speculative executor's scratch; see the 0731 module for the rationale).

The only per-request state a verify cannot simply overwrite is the ratio-2 compressors'
rolling carry (one ring block per window page, read-modify-write per token) at the encoder
kv-source layers; the ratio-1 decoder compressor is stateless. ``commit`` re-points those
rings to the accepted prefix. The DSpark drafter's inputs live here too (the target-layer
stream means for the verify rows, the block drafter's KV scratch, the draft distributions).
"""

from __future__ import annotations

import torch

from .args import DeepseekV41Args


class DSV41SpecScratch:
    def __init__(
        self, args: DeepseekV41Args, num_draft_tokens: int, vocab_size: int,
        device: torch.device, dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.args = args
        self.device = device
        self.k = num_draft_tokens
        self.k1 = num_draft_tokens + 1
        k1 = self.k1
        # Carry intermediates per gated kv-source layer: [k1, ratio, 2 * head_dim] fp32.
        self.carry: dict[tuple[int, str], torch.Tensor] = {}
        for L in args.kv_source_layers:
            ratio = args.compress_ratios[L]
            if ratio <= 1:
                continue
            self.carry[(L, "attn")] = torch.zeros(
                k1, ratio, 2 * args.head_dim, dtype=torch.float32, device=device
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
        self.boundary_row: int | None = None
        self.window_slots: torch.Tensor | None = None

    def set_step(self, cached_len: int, num_input_tokens: int) -> None:
        P = self.args.window_size
        self.boundary_row = None
        for j in range(1, num_input_tokens):
            if (cached_len + j) % P == 0:
                self.boundary_row = j
                break
        self.window_slots = None

    def commit(self, transformer, num_accepted: torch.Tensor) -> None:
        assert self.window_slots is not None, "commit before the verify forward staged its rows"
        for L in self.args.kv_source_layers:
            key = (L, "attn")
            if key not in self.carry:
                continue
            transformer.layers[L].attn.compressor.commit_carry(
                self.carry[key], num_accepted, self.window_slots, self.boundary_row
            )


__all__ = ["DSV41SpecScratch"]
