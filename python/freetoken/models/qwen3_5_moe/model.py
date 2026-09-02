from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch
from freetoken.core import get_global_ctx
from freetoken.layers import (
    BaseOP,
    GemmaRMSNorm,
    OPList,
    ParallelLMHead,
    VocabParallelEmbedding,
)
from freetoken.models.blocks import BaseLLMModel
from freetoken.utils import nvtx_annotate

from .attention import Qwen3_5Attention
from .gdn import Qwen3_5GatedDeltaNet
from .moe import Qwen3_5DenseMLP, Qwen3_5MoE

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class Qwen3_5DecoderLayer(BaseOP):
    """Pre-norm hybrid block: ``x = x + mixer(input_norm(x)); x = x + moe(post_norm(x))``,
    where the mixer is a GatedDeltaNet (linear layers) or gated attention (full layers).
    All norms are Gemma-style (1+weight)."""

    def __init__(self, config: ModelConfig, layer_id: int):
        self._layer_id = layer_id
        self._is_linear = config.is_linear_layer(layer_id)
        if self._is_linear:
            g = config.linear_attention_group()
            assert g is not None
            self.linear_attn = Qwen3_5GatedDeltaNet(
                hidden_size=config.hidden_size,
                num_k_heads=g.num_key_heads,
                num_v_heads=g.num_value_heads,
                head_k_dim=g.key_head_dim,
                head_v_dim=g.value_head_dim,
                conv_kernel_size=g.conv_kernel_dim,
                rms_norm_eps=config.rms_norm_eps,
                layer_id=layer_id,
                expert_quant=config.expert_quant,
                attn_quant=config.attn_quant,
            )
        else:
            self.self_attn = Qwen3_5Attention(config, layer_id)
        # Dense variants (num_experts==0, e.g. Qwen3.6-27B) use a plain SwiGLU MLP instead of
        # the routed MoE block; both expose ``forward(hidden)->hidden`` and the same key prefix.
        self.mlp = (
            Qwen3_5MoE(config, layer_id) if config.moe_enabled else Qwen3_5DenseMLP(config, layer_id)
        )
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(self, hidden: torch.Tensor, residual: torch.Tensor | None):
        # Residual-stream form: fuse each residual-add into the next RMSNorm
        # (GemmaRMSNorm.forward_add_residual) so add + norm are one kernel per sublayer.
        if residual is None:
            residual = hidden
            hidden = self.input_layernorm.forward(hidden)
        else:
            hidden, residual = self.input_layernorm.forward_add_residual(hidden, residual)
        hidden = self.linear_attn.forward(hidden) if self._is_linear else self.self_attn.forward(hidden)
        hidden, residual = self.post_attention_layernorm.forward_add_residual(hidden, residual)
        hidden = self.mlp.forward(hidden)
        return hidden, residual


class Qwen3_5Model(BaseOP):
    def __init__(self, config: ModelConfig):
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = OPList(
            [Qwen3_5DecoderLayer(config, layer_id) for layer_id in range(config.num_layers)]
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # The MTP draft head consumes this forward's final normed hidden rows; keeping the
        # reference alive between forwards is opt-in (set when the head is built).
        # Graph-captured decode forwards stash per captured batch size instead: the static
        # intermediate's address is stable per graph.
        self._keep_premix = False
        self._premix_hidden: torch.Tensor | None = None
        self._premix_graph: dict[int, torch.Tensor] = {}

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens.forward(input_ids)
        residual: torch.Tensor | None = None
        for layer in self.layers.op_list:
            x, residual = layer.forward(x, residual)
        x, pre_norm = self.norm.forward_add_residual(x, residual)
        if self._keep_premix:
            # The MTP head consumes the final NORMED hidden (scripts/mtp-dense-harness.py:
            # 56.8% teacher-forced agreement vs 54.3% for the pre-norm residual).
            if x.is_cuda and torch.cuda.is_current_stream_capturing():
                self._premix_graph[x.shape[0]] = x
            else:
                self._premix_hidden = x
            spec = getattr(get_global_ctx().batch, "spec", None)
            if spec is not None:
                spec.scratch.premix_rows = x  # the draft pass reads the verify rows
            dump_dir = os.getenv("FREETOKEN_MTP_DEBUG_DUMP")
            if dump_dir and not (x.is_cuda and torch.cuda.is_current_stream_capturing()):
                # Debug seam for the offline MTP wiring harness: one file per forward with
                # the rows' ids, positions, pre-norm residual and post-norm hidden.
                os.makedirs(dump_dir, exist_ok=True)
                batch = get_global_ctx().batch
                seq = getattr(self, "_dump_seq", 0)
                self._dump_seq = seq + 1
                torch.save(
                    {
                        "input_ids": input_ids.detach().cpu(),
                        "positions": batch.positions.detach().cpu(),
                        "pre_norm": pre_norm.detach().cpu(),
                        "post_norm": x.detach().cpu(),
                    },
                    os.path.join(dump_dir, f"rows_{seq:06d}.pt"),
                )
        return x


class Qwen3_5MoEForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self.model = Qwen3_5Model(config)
        if getattr(config, "lm_head_quant", "none") == "nvfp4":
            # checkpoint stores the (untied) lm_head as NVFP4: keep it native (W4A16) -- the
            # bf16 dequant of this ~1 GB matrix was the single largest decode kernel.
            from freetoken.kernel.triton.nvfp4_linear import Nvfp4LMHead

            assert not config.tie_word_embeddings, "NVFP4 lm_head assumes untied embeddings"
            self.lm_head = Nvfp4LMHead(
                num_embeddings=config.vocab_size, embedding_dim=config.hidden_size
            )
        else:
            self.lm_head = ParallelLMHead(
                num_embeddings=config.vocab_size,
                embedding_dim=config.hidden_size,
                tie_word_embeddings=config.tie_word_embeddings,
                tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
            )
        from freetoken.models.config import mtp_load_enabled

        if mtp_load_enabled():
            from .mtp import Qwen3_5MTPHead

            self.mtp = Qwen3_5MTPHead(config)
            self.model._keep_premix = True
            from freetoken.models.config import draft_lm_head_enabled

            if (
                draft_lm_head_enabled()
                and getattr(config, "lm_head_quant", "none") == "none"
                and not config.tie_word_embeddings
            ):
                # NVFP4 copy of the bf16 lm_head for the speculative DRAFT passes (the
                # loader quantizes it at load): the draft and chained-draft logits read
                # ~0.7 GB instead of ~2.5 GB each step; verification keeps the bf16 head.
                from freetoken.kernel.triton.nvfp4_linear import Nvfp4LMHead

                self.draft_lm_head = Nvfp4LMHead(
                    num_embeddings=config.vocab_size, embedding_dim=config.hidden_size
                )
        super().__init__()

    def forward(self) -> torch.Tensor:
        output = self.model.forward(get_global_ctx().batch.input_ids)
        return self.lm_head.forward(output)


__all__ = ["Qwen3_5MoEForCausalLM"]
