from __future__ import annotations

import os

from dataclasses import replace
from typing import TYPE_CHECKING

import torch
from freetoken.kernel.triton.moe_finalize import MoeCombineEpilogue
from freetoken.kernel.triton.moe_shared_gate import shared_gate_mul_add, shared_gate_sigmoid

# Decode: finish the MoE block (routed sum + shared gate + hyper-connection combine) in the
# expert kernel's sum-reduce launch instead of three kernels. FREETOKEN_MOE_FUSED_FINALIZE=0
# restores the separate ops (numerically: two fewer bf16 roundings on the fused path).
_FUSED_FINALIZE = os.getenv("FREETOKEN_MOE_FUSED_FINALIZE", "1") != "0"
from freetoken.layers.moe import make_moe_layer
from freetoken.models.qwen3_5_moe.moe import Qwen3_5MoE, _SharedExpert

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class Qwen4ExpMoE(Qwen3_5MoE):
    """Qwen3_5MoE with the shared-expert gate on triton instead of gemv + sigmoid + mul + add.

    Same weights, same state dict. The gate reduction stays ahead of the routed experts, which may write into ``hidden_states`` in place.
    """

    def __init__(self, config: ModelConfig, layer_id: int | None = None) -> None:
        if getattr(config, "expert_quant", "none") != "fp8_block":
            super().__init__(config, layer_id=layer_id)
        else:
            # Qwen3.8's block-fp8 checkpoint quantizes only the routed experts; the shared
            # expert stays bf16, so hide expert_quant from _SharedExpert's fp8 branch and
            # rebuild the routed experts with the fp8_block bank layout.
            super().__init__(replace(config, expert_quant="none"), layer_id=layer_id)
            self.experts = make_moe_layer(
                config,
                layer_id=layer_id,
                renormalize=config.norm_topk_prob,
                weight_format="fp8_block",
            )
        # --dense-quant fp8 "shared" part: per-channel FP8 shared expert (quantized at load);
        # _SharedExpert already carries the Fp8PerTensor* branch. Router gates stay bf16.
        args = getattr(config, "qwen4_args", None)
        if args is not None and args.part_quant("shared", getattr(config, "dense_quant", "none")) == "fp8_pertensor":
            self.shared_expert = _SharedExpert(
                replace(config, expert_quant="none"),
                config.hidden_size,
                config.shared_expert_intermediate_size,
                fp8_pertensor=True,
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate.forward(hidden_states)
        shared = self.shared_expert.forward(hidden_states)
        gate = shared_gate_sigmoid(hidden_states, self.shared_expert_gate.weight.view(-1))
        routed = self.experts.forward(hidden_states=hidden_states, router_logits=router_logits)
        return shared_gate_mul_add(routed, shared, gate).view(num_tokens, hidden_dim)

    def forward_combine(
        self, hidden_states: torch.Tensor, residual: torch.Tensor, inject: torch.Tensor | None, hc_layer
    ) -> torch.Tensor:
        """The MoE block followed by ``hc_layer.combine(residual, y, inject)``. On the
        NVFP4 offload decode path the routed sum-reduce, the shared-expert gate mul-add and
        the combine run as ONE kernel (``MoeCombineEpilogue``; the CPU partial of the hybrid
        backend folds in too); anywhere else this is exactly ``combine(forward())``."""
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate.forward(hidden_states)
        shared = self.shared_expert.forward(hidden_states)
        gate_w = self.shared_expert_gate.weight.view(-1)
        ep = None
        if _FUSED_FINALIZE and inject is not None and hidden_states.is_cuda and residual.stride(1) == 1:
            # the gate sigmoid is produced by the expert path (fused into the hybrid pre-quant
            # launch when possible; else computed there before the experts touch hidden)
            ep = MoeCombineEpilogue(shared, None, residual, inject, hc_layer.hc_count, gate_weight=gate_w)
        else:
            gate = shared_gate_sigmoid(hidden_states, gate_w)
        routed = self.experts.forward(hidden_states=hidden_states, router_logits=router_logits, epilogue=ep)
        if ep is not None and ep.result is not None:
            return ep.result
        if ep is not None:
            # fallback path took no epilogue: the gate must still come from the pre-expert
            # hidden state (the routed experts may have written hidden_states in place)
            gate = ep.gate if ep.gate is not None else shared_gate_sigmoid(hidden_states, gate_w)
        y = shared_gate_mul_add(routed, shared, gate).view(num_tokens, hidden_dim)
        return hc_layer.combine(residual, y, inject)


__all__ = ["Qwen4ExpMoE"]
