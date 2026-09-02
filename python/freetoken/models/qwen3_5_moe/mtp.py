"""Qwen3.5 / Qwen3.8 dense MTP (multi-token-prediction) draft head.

The checkpoint ships one speculative head under the ``mtp.`` prefix: two pre-fc norms, an
``fc`` ``[hidden, 2*hidden]`` projection over the concatenation, one full-attention decoder
layer (``mtp.layers.0``, the same block as :class:`Qwen3_5DecoderLayer`) and a final norm.
Embeddings and ``lm_head`` are shared with the main model. The wiring is the documented
Qwen3-Next one (vLLM ``qwen3_next_mtp``): the head consumes the main tower's FINAL normed
hidden state ``h_i`` plus the embedding of the next token and predicts the token after that::

    x      = fc(cat(pre_fc_norm_embedding(embed(t_{i+1})), pre_fc_norm_hidden(h_i)))
    out    = norm(decoder_layer(x))                     # [T, hidden]
    logits = lm_head(out)

(embedding first in the concat -- the DeepSeek-V3 MTP order; the harness scores the other
order at chance.)

For chained drafts (k >= 2) ``out`` itself stands in for the next position's ``h`` (the
executor's ``r1_out`` seam), exactly as multi-step MTP does in vLLM.

The head's attention layer reads its batch through the global forward context like every
other layer; :meth:`forward` swaps the context batch to the caller's (a verify batch, or a
shadow prefill batch the speculative executor builds for its own passes) for its duration.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from typing import TYPE_CHECKING

import torch
from freetoken.core import get_global_ctx
from freetoken.layers import BaseOP, LinearReplicated, OPList

from .model import GemmaRMSNorm, Qwen3_5DecoderLayer

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


@contextmanager
def _ctx_batch(batch):
    """Run the body with ``batch`` as the global context batch (restored afterwards)."""
    ctx = get_global_ctx()
    saved = ctx._batch
    ctx._batch = batch
    try:
        yield
    finally:
        ctx._batch = saved


class Qwen3_5MTPHead(BaseOP):
    """The draft head. State-dict prefix ``mtp.`` matches the checkpoint keys (after the
    loader's q|k|v and gate|up fusions)."""

    def __init__(self, config: ModelConfig) -> None:
        h = config.hidden_size
        eps = config.rms_norm_eps
        # The MTP layer sits after the last decoder layer; its id keys the KV pool row and
        # must resolve as full attention (extend_config_for_speculative adds it).
        self._layer_id = config.num_layers
        assert not config.is_linear_layer(self._layer_id), (
            "the MTP layer id must be registered in the full-attention group "
            "(EngineConfig applies extend_config_for_speculative)"
        )
        self.pre_fc_norm_hidden = GemmaRMSNorm(h, eps=eps)
        self.pre_fc_norm_embedding = GemmaRMSNorm(h, eps=eps)
        self.fc = LinearReplicated(2 * h, h, has_bias=False)
        # The head is bf16 end to end in every checkpoint seen (its tensors sit in the
        # quantization ignore list): hide the main tower's quant flags from the layer.
        layer_config = replace(
            config,
            attn_quant="none",
            dense_quant="none",
            expert_quant="none",
            lm_head_quant="none",
            fp8_dense_layers=frozenset(),
        )
        self.layers = OPList([Qwen3_5DecoderLayer(layer_config, self._layer_id)])
        self.norm = GemmaRMSNorm(h, eps=eps)

    @property
    def layer_id(self) -> int:
        return self._layer_id

    def forward(
        self,
        premix_r: torch.Tensor,
        token_embeds: torch.Tensor,
        batch,
        r1_out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the draft hidden state ``[T, hidden]``; the caller applies the shared lm_head.

        ``premix_r`` is the main tower's final normed hidden for each row, ``token_embeds``
        the embedding of each row's NEXT token. ``r1_out`` receives the head's own output
        rows (the premix of a chained draft row).
        """
        # Embedding FIRST, hidden second (DeepSeek-V3-MTP order). Settled empirically by
        # scripts/mtp-dense-harness.py on this checkpoint: 56.8% teacher-forced next-next-
        # token agreement vs 0% for hidden|embed; (1+w) pre-fc norms, post-norm hidden.
        x = self.fc.forward(
            torch.cat(
                [
                    self.pre_fc_norm_embedding.forward(token_embeds),
                    self.pre_fc_norm_hidden.forward(premix_r),
                ],
                dim=-1,
            )
        )
        with _ctx_batch(batch):
            hidden, residual = self.layers.op_list[0].forward(x, None)
        out, _ = self.norm.forward_add_residual(hidden, residual)
        if r1_out is not None:
            r1_out[: out.shape[0]].copy_(out)
        return out


__all__ = ["Qwen3_5MTPHead"]
