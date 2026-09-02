"""Qwen3.8-Flash-Next MTP (multi-token-prediction) draft head.

The checkpoint ships one speculative head under the ``mtp.`` prefix (31 bf16 tensors, all in the
quantization ignore list): two pre-fc norms, two ``[hidden, hidden]`` fc projections, one decoder
layer whose state-dict subtree is byte-identical to :class:`Qwen4ExpDecoderLayer` (QSA attention +
indexer + MoE + shared expert, over the ``hc_count`` residual streams), and a top-level
``hyper_connection_mixer``. Embeddings and ``lm_head`` are shared with the main model
(``mtp_use_dedicated_embeddings`` is false and the checkpoint has no ``mtp.lm_head``).

The head consumes the main tower's PRE-mixer residual ``R [T, hc_count*hidden]`` (the tensor
``Qwen4ExpModel.forward`` feeds its own mixer) plus the embedding of each row's next token, and
predicts the token after that::

    R0     = merge(pre_fc_norm_hidden(R), fc, pre_fc_norm_embedding(embed(next_tok)))
    R1     = Qwen4ExpDecoderLayer(full_attention, own bf16 MoE)(R0)      # mtp.layers.0
    hidden = hyper_connection_mixer.mix(R1)[0]                            # [T, hidden]
    logits = lm_head(hidden)                                              # shared head

``merge`` is the one part with no reference implementation anywhere (HF transformers drops the
head; vLLM's Qwen3-Next MTP confirms only the concat-fc skeleton without hyper-connection
streams), so the exact stream wiring is selectable via ``FREETOKEN_MTP_WIRING`` and settled
empirically by ``scripts/mtp-wiring-harness.py`` (teacher-forcing top-1 agreement is ~55-75% for
the correct wiring and near-zero otherwise). The default is the harness winner.

The MTP MoE experts are the stacked bf16 tensors ``mtp.layers.0.mlp.experts.{gate_up_proj,
down_proj}``. Under an offload MoE backend (i.e. whenever this model actually serves) they are
NVFP4-quantized at load and appended to the unified expert offload cache as one extra bank layer
(``build_mtp_offload_bank_layer`` + the engine's ``append_spec_expert_bank`` hook), costing cache
slots instead of resident VRAM; the layer is then a plain ``OffloadMoELayer``. Under non-offload
backends (tests, toy configs) or ``FREETOKEN_MTP_EXPERTS={fp8,bf16}`` they are served resident
through :class:`_ResidentExperts`, which never touches the global ``ctx.moe_backend``.
"""

from __future__ import annotations

import os
from dataclasses import replace
from typing import TYPE_CHECKING

import torch
from freetoken.layers import BaseOP, LinearReplicated, OPList
from freetoken.layers.moe import MoELayer
from freetoken.moe.fused import fused_topk

from .hc import GatedResidual, GroupedPlusOneRMSNorm

if TYPE_CHECKING:
    from freetoken.core import Batch
    from freetoken.models.config import ModelConfig

# Candidate stream wirings for merging the 4-stream R with the next-token embedding. All are
# shape-consistent with the checkpoint (pre_fc_norm_hidden [hc*hidden] grouped, fc_hidden
# [hidden, hidden]); scripts/mtp-wiring-harness.py settled the real one against the live
# checkpoint (2026-08-30): ``per_stream`` scores 52.5% teacher-forced next-next-token top-1 on
# held text with ~0% echo, while every alternative (plain instead of (1+w) norms, swapped fc
# roles, collapse/residual/stream-substitution wirings) collapses to chance or degenerates to
# echoing the main model's own next-token prediction.
#
#   per_stream:    fc_hidden applied per stream (weight shared), embedding broadcast-added
#   mean_collapse: streams mean-collapsed after the grouped norm, fc'd, then re-broadcast
#   sum_collapse:  as mean_collapse with a sum instead of the mean
MTP_WIRINGS = ("per_stream", "mean_collapse", "sum_collapse")


def _wiring_from_env() -> str:
    wiring = os.getenv("FREETOKEN_MTP_WIRING", MTP_WIRINGS[0]).strip()
    if wiring not in MTP_WIRINGS:
        raise ValueError(f"FREETOKEN_MTP_WIRING must be one of {MTP_WIRINGS}, got {wiring!r}")
    return wiring


def mtp_decoder_config(config: ModelConfig, *, offload_experts: bool = False) -> ModelConfig:
    """The decoder-layer config for the MTP layer: bf16 dense/attention throughout.

    The main tower's quant flags describe the NVFP4 offload experts; the MTP layer is in the
    checkpoint's ignore list end to end, so hide them from the layer constructors. The expert
    home is two-mode: ``offload_experts`` keeps the real ``moe_backend`` so ``make_moe_layer``
    builds an :class:`~freetoken.layers.moe.OffloadMoELayer` (the draft experts join the
    unified cache as an extra bank layer -- see ``build_mtp_offload_bank_layer`` and the
    engine's ``append_spec_expert_bank`` hook); otherwise ``moe_backend="fused"`` only steers
    the class choice and the built experts are replaced with :class:`_ResidentExperts`. The
    full-attention group is extended with the MTP layer id so ``is_linear_layer`` and the
    group lookups resolve it (the engine-side pools get the same extension when speculative
    decoding is wired up).
    """
    mtp_layer_id = config.num_layers
    groups = tuple(
        replace(g, layer_ids=g.layer_ids + (mtp_layer_id,))
        if g.name == "full" and mtp_layer_id not in g.layer_ids
        else g
        for g in config.attention_groups
    )
    # --dense-quant fp8 "mtp" part: let the draft layer's own attention/HC/shared linears
    # take the load-time FP8 flags (its experts never do); off by default -- the head only
    # steers proposals, but its acceptance rate is what pays for the spec step.
    keep = "mtp" in getattr(config.qwen4_args, "dense_fp8_parts", ())
    return replace(
        config,
        expert_quant="none",
        dense_quant=config.dense_quant if keep else "none",
        attn_quant=config.attn_quant if keep else "none",
        moe_backend=config.moe_backend if offload_experts else "fused",
        attention_groups=groups,
    )


class _ResidentExperts(MoELayer):
    """Resident bf16 experts routed without the global MoE backend.

    ``MoELayer.forward``'s bf16 branch dispatches to ``ctx.moe_backend``, which is the offload
    backend when this model serves -- wrong for resident tensors. Route + GEMM directly instead;
    ``routed_forward`` is the ctx-free resident path.
    """

    def forward(
        self, hidden_states: torch.Tensor, router_logits: torch.Tensor | None = None,
        epilogue=None,  # MoeCombineEpilogue: not fused here (result stays None)
    ) -> torch.Tensor:
        topk_weights, topk_ids = fused_topk(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=self.top_k,
            renormalize=self.renormalize,
        )
        return self.routed_forward(hidden_states, topk_weights, topk_ids)


class Qwen4ExpMTPHead(BaseOP):
    """The draft head. State-dict prefix ``mtp.`` matches the checkpoint keys verbatim
    (after the loader's standard qkv / hyper-connection / shared-expert fusions)."""

    def __init__(self, config: ModelConfig, wiring: str | None = None) -> None:
        args = config.qwen4_args
        if args.mtp_num_layers != 1 or tuple(args.mtp_layer_types) != ("full_attention",):
            raise ValueError(
                "Qwen4ExpMTPHead supports exactly one full_attention MTP layer, got "
                f"{args.mtp_num_layers} layer(s) of {args.mtp_layer_types}"
            )
        if args.mtp_use_dedicated_embeddings:
            raise ValueError("dedicated MTP embeddings are not supported (none shipped)")
        self._hc_count = args.hc_count
        self._hidden = args.hidden_size
        self._wiring = wiring if wiring is not None else _wiring_from_env()
        # The MTP layer sits after the last decoder layer; the id keys the attention backend's
        # sparse-layer slot and the KV pool row, both sized num_layers+1 when the head is on.
        self._layer_id = config.num_layers

        width = args.ple_state_width
        self.pre_fc_norm_hidden = GroupedPlusOneRMSNorm(width, config.rms_norm_eps, args.hc_count)
        self.pre_fc_norm_embedding = GroupedPlusOneRMSNorm(args.hidden_size, config.rms_norm_eps, 1)
        self.fc_hidden = LinearReplicated(args.hidden_size, args.hidden_size, has_bias=False)
        self.fc_embedding = LinearReplicated(args.hidden_size, args.hidden_size, has_bias=False)

        from freetoken.layers.moe import is_offload_moe_backend

        from .model import Qwen4ExpDecoderLayer
        from .weight import mtp_expert_format

        # offload (default): the draft experts live in the unified NVFP4 offload cache as
        # bank layer ``num_layers`` -- the decoder layer builds its natural OffloadMoELayer
        # and the engine's append_spec_expert_bank hook supplies the quantized bank. Without
        # an offload backend (tests, toy configs) there is no cache to join, so fall back to
        # resident fp8 experts routed off the global ctx.
        fmt = mtp_expert_format()
        self.offload_experts = fmt == "offload" and is_offload_moe_backend(config.moe_backend)
        layer_config = mtp_decoder_config(config, offload_experts=self.offload_experts)
        layer = Qwen4ExpDecoderLayer(layer_config, self._layer_id)
        assert layer.ple is None, "the MTP layer must not carry a PLE block"
        if not self.offload_experts:
            layer.mlp.experts = _ResidentExperts(
                num_experts=config.num_experts,
                top_k=config.num_experts_per_tok,
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size,
                renormalize=config.norm_topk_prob,
                weight_format="fp8_block" if fmt == "offload" else fmt,
            )
        self.layers = OPList([layer])
        # Built from the MTP layer config: the head's dense linears stay bf16 under
        # --dense-quant fp8 unless the "mtp" part is selected (the loader emits mtp.* raw).
        self.hyper_connection_mixer = GatedResidual(layer_config, use_combine=False)

    @property
    def layer_id(self) -> int:
        return self._layer_id

    def build_input(self, premix_r: torch.Tensor, token_embeds: torch.Tensor) -> torch.Tensor:
        """Merge the main tower's pre-mixer residual with the next-token embedding.

        ``premix_r`` is ``[T, hc_count*hidden]``, ``token_embeds`` is ``[T, hidden]`` (the
        embedding of each row's NEXT token); the result feeds ``mtp.layers.0``.
        """
        t = premix_r.shape[0]
        hc, hidden = self._hc_count, self._hidden
        rn = self.pre_fc_norm_hidden.forward(premix_r)
        en = self.pre_fc_norm_embedding.forward(token_embeds)
        fe = self.fc_embedding.forward(en)
        if self._wiring == "per_stream":
            fh = self.fc_hidden.forward(rn.reshape(t * hc, hidden)).reshape(t, hc, hidden)
            return (fh + fe.unsqueeze(1)).reshape(t, hc * hidden)
        collapsed = rn.reshape(t, hc, hidden)
        collapsed = collapsed.mean(dim=1) if self._wiring == "mean_collapse" else collapsed.sum(dim=1)
        merged = self.fc_hidden.forward(collapsed) + fe
        return merged.repeat(1, hc)

    def forward(
        self,
        premix_r: torch.Tensor,
        token_embeds: torch.Tensor,
        batch: Batch,
        r1_out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the draft hidden state ``[T, hidden]``; the caller applies the shared lm_head.

        ``r1_out``: optional [>=T, hc_count*hidden] buffer that receives the decoder layer's
        4-stream residual output (pre-mixer). Row i stands in for position i+1's premix when
        the speculative executor chains a second draft; the copy is capture-safe.
        """
        hidden = self.build_input(premix_r, token_embeds)
        hidden = self.layers.op_list[0].forward(hidden, batch)
        if r1_out is not None:
            r1_out[: hidden.shape[0]].copy_(hidden)
        return self.hyper_connection_mixer.mix(hidden)[0]


__all__ = ["MTP_WIRINGS", "Qwen4ExpMTPHead", "mtp_decoder_config"]
