"""Per-forward state that CSA2 layers hand down the stack instead of recomputing.

The reference keeps one ``SharedAttentionRuntime`` per process: layers run in order and every
producer writes before its consumers read, so a single slot per item suffices. FreeToken
creates one per forward (prefill, decode or DSpark draft) and threads it through the layers:

* ``topk_idxs`` / ``n_window`` / ``cmp_counts`` -- the COMPLETE sparse-attention pick list
  (global window slots then global compressed rows) published by the last Full/Reindex
  layer; Reuse layers attend with it unchanged. Sharing the window half is valid because
  every main layer's window ring uses the same slots for the same positions (the pool
  addresses window KV by position, not by layer), and the compressed half addresses the
  owner's pool, which the Reuse layers read through the same owner.
* ``cand_blocks`` -- the candidate pool built by the candidate-source layer (the decoder's
  first Full layer), consumed by the Reindex layers' indexers.
* ``latent`` -- the producing layer's pre-RoPE compressed latent, handed to its own indexer
  (which projects the index keys from it) before the layer rotates/quantizes it for the KV.
* ``produce_kv`` -- False during the CED decoder replay: layer 20's compressor and index keys
  were already produced over the whole chunk from the encoder output, so the replayed rows
  must not re-produce (and must not re-scatter) them.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SharedAttentionRuntime:
    produce_kv: bool = True
    topk_idxs: torch.Tensor | None = None       # [b, s, n_window + n_cmp] int32 global slots
    n_window: int = 0
    cmp_counts: torch.Tensor | None = None      # decode only: [B, 1] int32 live compressed cols
    cand_blocks: torch.Tensor | None = None     # [.., topk_blocks] int32 block ids (-1 padded)
    cand_n_positions: int = 0                   # the position width the candidate ids index
    latent: torch.Tensor | None = None          # producer-local: pre-RoPE latent [1, g, head_dim]
    latent_positions: torch.Tensor | None = None

    def publish_picks(
        self, topk_idxs: torch.Tensor, n_window: int, cmp_counts: torch.Tensor | None
    ) -> None:
        self.topk_idxs = topk_idxs
        self.n_window = n_window
        self.cmp_counts = cmp_counts

    def picks(self) -> tuple[torch.Tensor, int, torch.Tensor | None]:
        assert self.topk_idxs is not None, (
            "a Reuse-mode layer ran before any Full/Reindex layer published its top-k picks"
        )
        return self.topk_idxs, self.n_window, self.cmp_counts


__all__ = ["SharedAttentionRuntime"]
