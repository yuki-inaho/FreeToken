"""Qwen3.8-Flash-Next decoder stack (text-only).

The residual state is ``R [T, hc_count*hidden]`` end to end: the embedding is repeated over the
``hc_count`` streams, every layer mixes them down to one ``[T, hidden]`` block input and injects
its output back, and the top-level mixer collapses them once before ``lm_head``. There is no
input/post layernorm and no final ``model.norm`` -- the hyper-connection norms are the only ones.

Layer contract (frozen): ``forward(R [T, hc*hidden], batch) -> R' [T, hc*hidden]`` with an
immediate combine::

    R  = R + ple(R, batch)                 # zero-based layer 1 only
    x, s = attn_hc.mix(R); y = (GDN | QSA)(x); R = attn_hc.combine(R, y, s)
    x, s = mlp_hc.mix(R);  y = MoE(x);        R = mlp_hc.combine(R, y, s)
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, List

import torch
from freetoken.core import get_global_ctx
from freetoken.distributed import get_tp_info
from freetoken.layers import BaseOP, OPList, ParallelLMHead, VocabParallelEmbedding
from freetoken.layers.embedding import HostUVAEmbedding
from freetoken.models.blocks import BaseLLMModel
from freetoken.utils import nvtx_annotate

from .attention import Qwen4ExpAttention
from .hc import GatedResidual
from .moe import Qwen4ExpMoE
from .ple import PLELayer

if TYPE_CHECKING:
    from freetoken.core import Batch
    from freetoken.models.config import ModelConfig


def build_linear_mixer(config: ModelConfig, layer_id: int) -> BaseOP:
    """GDN mixer of a linear_attention layer (Qwen3.5's GDN with a configurable output gate)."""
    from .gdn import Qwen4ExpGatedDeltaNet

    g = config.linear_attention_group()
    return Qwen4ExpGatedDeltaNet(
        hidden_size=config.hidden_size,
        num_k_heads=g.num_key_heads,
        num_v_heads=g.num_value_heads,
        head_k_dim=g.key_head_dim,
        head_v_dim=g.value_head_dim,
        conv_kernel_size=g.conv_kernel_dim,
        rms_norm_eps=config.rms_norm_eps,
        layer_id=layer_id,
        output_gate=g.output_gate,
        # Qwen3.8's block-fp8 checkpoint keeps the GDN projections bf16 (only the routed
        # experts are quantized), so do not let expert_quant flip them to Fp8Block.
        expert_quant="none" if config.expert_quant == "fp8_block" else config.expert_quant,
        # --dense-quant fp8: the GDN projections take the fp8 flag only under the "gdn" part.
        attn_quant=config.qwen4_args.part_quant("gdn", config.attn_quant),
    )


class _RouteDumper:
    """FREETOKEN_ROUTE_DUMP harness seam (scripts/moe-prefetch-harness.py): per prefill
    forward, buffer every main-tower layer's router logits -- plus, under
    FREETOKEN_ROUTE_DUMP_R=1, the pre-mlp-mix residual R and the mixed block input --
    then flush one ``route_NNNNNN.pt`` per forward. Teacher-forced prefill computes the
    same activations/routing a decode over the same text would, so these dumps are the
    ground truth for offline expert-prefetch predictor evaluation. Eager prefill only,
    synchronous per-layer D2H: never set in normal serving."""

    def __init__(self, dump_dir: str) -> None:
        self.dump_dir = dump_dir
        self.with_r = (
            os.getenv("FREETOKEN_ROUTE_DUMP_R", "0").strip().lower()
            not in {"", "0", "false", "no", "off"}
        )
        # Forwards larger than this are skipped whole -- notably the boot-time 8192-token
        # memory-profile prefill, which would otherwise dump gigabytes of dummy activations.
        self.max_t = int(os.getenv("FREETOKEN_ROUTE_DUMP_MAX_T", "4096"))
        self.seq = 0
        self._buf: dict[int, dict[str, torch.Tensor]] = {}

    def add(
        self,
        layer_id: int,
        router_logits: torch.Tensor,
        hidden: torch.Tensor,
        block_input: torch.Tensor,
    ) -> None:
        if router_logits.shape[0] > self.max_t:
            return
        entry = {"logits": router_logits.detach().to("cpu", torch.bfloat16)}
        if self.with_r:
            entry["r"] = hidden.detach().to("cpu", torch.bfloat16)
            entry["block_input"] = block_input.detach().to("cpu", torch.bfloat16)
        self._buf[layer_id] = entry

    def flush(self, input_ids: torch.Tensor, positions: torch.Tensor) -> None:
        if not self._buf:
            return
        os.makedirs(self.dump_dir, exist_ok=True)
        path = f"{self.dump_dir}/route_{self.seq:06d}.pt"
        self.seq += 1
        torch.save(
            {"input_ids": input_ids.cpu(), "positions": positions.cpu(), "layers": self._buf},
            path,
        )
        self._buf = {}


class Qwen4ExpDecoderLayer(BaseOP):
    """One decoder layer over the hyper-connection streams (see the module docstring for the flow)."""

    def __init__(self, config: ModelConfig, layer_id: int) -> None:
        self._layer_id = layer_id
        # Harness seam: Qwen4ExpModel attaches its _RouteDumper to its OWN layers only, so
        # the MTP head's standalone layer (id num_layers) never dumps through this path.
        self._route_dump: _RouteDumper | None = None
        # In-graph cross-layer expert lookahead (--moe-prefetch layer); attached by the
        # engine via freetoken.moe.prefetch.attach_moe_lookahead before graph capture.
        self._moe_lookahead = None
        self._is_linear = config.is_linear_layer(layer_id)
        if self._is_linear:
            self.linear_attn = build_linear_mixer(config, layer_id)
        else:
            self.self_attn = Qwen4ExpAttention(config, layer_id)
        self.mlp = Qwen4ExpMoE(config, layer_id)
        self.attn_hyper_connection = GatedResidual(config)
        self.mlp_hyper_connection = GatedResidual(config)
        self.ple = (
            PLELayer(config, layer_id) if layer_id in config.qwen4_args.ple_layer_ids else None
        )

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(self, hidden: torch.Tensor, batch: Batch) -> torch.Tensor:
        if self.ple is not None:
            hidden = hidden + self.ple.forward(hidden, batch)
        block_input, inject = self.attn_hyper_connection.mix(hidden)
        if self._is_linear:
            block_output = self.linear_attn.forward(block_input)
        else:
            block_output = self.self_attn.forward(block_input, batch)
        hidden = self.attn_hyper_connection.combine(hidden, block_output, inject)
        if self._moe_lookahead is not None:
            # Join this layer's prefetched-expert copy, then speculatively route a
            # LATER layer from the current residual (captured; decode only).
            self._moe_lookahead.hook(self._layer_id, hidden, batch)
        block_input, inject = self.mlp_hyper_connection.mix(hidden)
        if self._route_dump is not None and batch.is_prefill:
            # Before mlp.forward: the routed experts may write block_input in place. The
            # recomputed gate GEMM is bitwise the router's own (same op, input, weights).
            self._route_dump.add(
                self._layer_id, self.mlp.gate.forward(block_input), hidden, block_input
            )
        # MoE block + combine; the decode path fuses the block's tail into one kernel.
        return self.mlp.forward_combine(block_input, hidden, inject, self.mlp_hyper_connection)


class Qwen4ExpModel(BaseOP):
    def __init__(self, config: ModelConfig) -> None:
        self.hc_count = config.qwen4_args.hc_count
        # FREETOKEN_EMBED_HOST=1 (default): the bf16 embedding table stays in pinned host
        # memory and rows are gathered over UVA (decode reads <= 4 rows/step); its 1.27 GB of
        # VRAM becomes ~460 expert slots. =0 keeps the table resident.
        embed_cls = VocabParallelEmbedding
        if (
            os.getenv("FREETOKEN_EMBED_HOST", "1") != "0"
            and get_tp_info().size == 1
            and not config.tie_word_embeddings
        ):
            embed_cls = HostUVAEmbedding
        self.embed_tokens = embed_cls(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = OPList(
            [Qwen4ExpDecoderLayer(config, layer_id) for layer_id in range(config.num_layers)]
        )
        self.hyper_connection_mixer = GatedResidual(config, use_combine=False)
        # plain tuple (not an OP child), so it never shows up in the state dict
        self._ple = tuple(layer.ple for layer in self.layers.op_list if layer.ple is not None)
        # The MTP draft head reads the PRE-mixer residual; keeping the reference alive between
        # forwards costs a [T, hc*hidden] tensor, so it is opt-in. Graph-captured decode
        # forwards stash per captured batch size instead: the static intermediate's address is
        # stable per graph, and a later capture must not clobber a smaller graph's stash.
        self._keep_premix = False
        self._premix_hidden: torch.Tensor | None = None
        self._premix_graph: dict[int, torch.Tensor] = {}
        # Debug seam for scripts/mtp-wiring-harness.py: dump (input_ids, positions, premix R)
        # per forward. Never set in normal serving.
        self._premix_dump_dir = os.getenv("FREETOKEN_MTP_DEBUG_DUMP") or None
        self._premix_dump_seq = 0
        # Harness seam for scripts/moe-prefetch-harness.py (see _RouteDumper).
        route_dump_dir = os.getenv("FREETOKEN_ROUTE_DUMP") or None
        self._route_dumper = _RouteDumper(route_dump_dir) if route_dump_dir else None
        if self._route_dumper is not None:
            for layer in self.layers.op_list:
                layer._route_dump = self._route_dumper

    @property
    def ple_layers(self) -> List[PLELayer]:
        """The PLE layers in decoder order -- the seam the loader attaches table backends to."""
        return list(self._ple)

    def forward(self, input_ids: torch.Tensor, batch: Batch) -> torch.Tensor:
        hidden = self.embed_tokens.forward(input_ids).repeat(1, self.hc_count)
        meta = None
        if self._ple:
            from .ple import build_ple_metadata, commit_ngram_context

            meta = build_ple_metadata(batch, self._ple[0].args, input_ids.device)
            for ple in self._ple:  # gather the pinned-host PLE rows while the early layers run
                ple.start_prefetch(batch, meta)
        for layer in self.layers.op_list:
            hidden = layer.forward(hidden, batch)
        if meta is not None:
            # single writer: the layers only read the context, so a second PLE layer's
            # prefetch sees the un-rolled window
            commit_ngram_context(meta, getattr(batch, "fla_metadata", None))
        if self._keep_premix:
            if hidden.is_cuda and torch.cuda.is_current_stream_capturing():
                self._premix_graph[hidden.shape[0]] = hidden
            else:
                self._premix_hidden = hidden
        spec = getattr(batch, "spec", None)
        if spec is not None:
            spec.scratch.premix_rows = hidden  # the MTP draft pass reads the verify rows' R
        if self._premix_dump_dir is not None:
            self._dump_premix(input_ids, batch, hidden)
        if self._route_dumper is not None and batch.is_prefill and not (
            input_ids.is_cuda and torch.cuda.is_current_stream_capturing()
        ):
            self._route_dumper.flush(input_ids, batch.positions)
        return self.hyper_connection_mixer.mix(hidden)[0]

    def _dump_premix(self, input_ids: torch.Tensor, batch: Batch, hidden: torch.Tensor) -> None:
        if input_ids.is_cuda and torch.cuda.is_current_stream_capturing():
            return  # CUDA-graph capture replays this forward; a D2H copy here would abort it
        os.makedirs(self._premix_dump_dir, exist_ok=True)
        path = f"{self._premix_dump_dir}/premix_{self._premix_dump_seq:06d}.pt"
        self._premix_dump_seq += 1
        torch.save(
            {
                "input_ids": input_ids.cpu(),
                "positions": batch.positions.cpu(),
                "premix": hidden.detach().to("cpu", torch.bfloat16),
            },
            path,
        )


class Qwen4ExpForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        self.model = Qwen4ExpModel(config)
        lm_head_quant = config.qwen4_args.part_quant(
            "lm_head", getattr(config, "lm_head_quant", "none")
        )
        if (
            os.getenv("FREETOKEN_LM_HEAD_NVFP4", "0") == "1"
            and lm_head_quant in ("none", "fp8_pertensor")
            and not config.tie_word_embeddings
        ):
            # NVFP4 TARGET head quantized at load (the draft head then is the same tensor):
            # graph A's per-step head read halves again (636 -> 318 MB) and the separate
            # draft copy disappears (+~130 slots). Changes the emitted logits slightly ->
            # quality-gated, opt-in.
            lm_head_quant = "nvfp4"
        if lm_head_quant == "nvfp4":
            from freetoken.kernel.triton.nvfp4_linear import Nvfp4LMHead

            assert not config.tie_word_embeddings, "NVFP4 lm_head assumes untied embeddings"
            self.lm_head = Nvfp4LMHead(
                num_embeddings=config.vocab_size, embedding_dim=config.hidden_size
            )
        elif lm_head_quant == "fp8_pertensor":
            # --dense-quant fp8: per-row fp8 head quantized at load (halves the 3x-per-step
            # full-vocab read under speculative decoding; the loader emits weight_scale).
            from freetoken.layers.embedding import Fp8LMHead, HostUVAEmbedding

            assert not config.tie_word_embeddings, "FP8 lm_head assumes untied embeddings"
            self.lm_head = Fp8LMHead(
                num_embeddings=config.vocab_size, embedding_dim=config.hidden_size
            )
        else:
            self.lm_head = ParallelLMHead(
                num_embeddings=config.vocab_size,
                embedding_dim=config.hidden_size,
                tie_word_embeddings=config.tie_word_embeddings,
                tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
            )
        from freetoken.models.config import draft_lm_head_enabled

        if (
            draft_lm_head_enabled()
            and lm_head_quant != "nvfp4"
            and not config.tie_word_embeddings
        ):
            # NVFP4 copy of the head for the speculative DRAFT passes only (graphs B/C read
            # it once each per step instead of the fp8/bf16 target head): ~0.35 GB VRAM for
            # ~2x fewer draft-head bytes; verification keeps the target head, so the emitted
            # distribution is unchanged. The loader quantizes it at load
            # (weight.py _emit_dense); FREETOKEN_DRAFT_LM_HEAD=0 disables.
            from freetoken.kernel.triton.nvfp4_linear import Nvfp4LMHead

            self.draft_lm_head = Nvfp4LMHead(
                num_embeddings=config.vocab_size, embedding_dim=config.hidden_size
            )
        from freetoken.models.config import mtp_load_enabled

        if mtp_load_enabled() and config.qwen4_args.mtp_num_layers:
            from .mtp import Qwen4ExpMTPHead

            self.mtp = Qwen4ExpMTPHead(config)
            self.model._keep_premix = True
        super().__init__()

    def append_spec_expert_bank(self, banks, engine_config, device: torch.device) -> int:
        """Engine hook (offload-cache construction): when the MTP draft experts serve from
        the unified offload cache, quantize them into its NVFP4 format and append one bank
        layer (id ``num_moe_layers``, i.e. right after the main tower's). Returns the number
        of bank layers appended so the engine sizes the cache and its walk assert to match."""
        mtp = getattr(self, "mtp", None)
        if mtp is None or not mtp.offload_experts:
            return 0
        if banks.quant_format != "nvfp4":
            raise ValueError(
                "MTP offload experts join the expert cache as an extra NVFP4 bank layer and "
                f"require the plain nvfp4 format, got {banks.quant_format!r}; set "
                "FREETOKEN_MTP_EXPERTS=fp8 to keep them resident instead"
            )
        from .weight import build_mtp_offload_bank_layer

        layer_banks = build_mtp_offload_bank_layer(
            engine_config.model_path,
            self._config,
            device,
            dummy=engine_config.use_dummy_weight,
        )
        for name, tensor in layer_banks.items():
            banks.sources[name].append(tensor)
        if banks.layer_residency is not None:
            from freetoken.moe.host_banks import HostResidency

            banks.layer_residency.append(HostResidency.PINNED.value)
        return 1

    def load_host_tables(self, engine_config) -> int:
        """Attach the PLE n-gram table (pinned checkpoint bank, or zeros for dummy weights); returns the pinned host bytes the engine reserves from its pin budget."""
        ple_layers = self.model.ple_layers
        if not ple_layers:
            return 0
        from .ple import PinnedUVATable, ZeroTable, derive_ngram_hash_constants

        if getattr(engine_config, "use_dummy_weight", False):
            # Dummy fill leaves the int64 hash buffers garbage (a zero vocab size divides by
            # zero in the hash), so re-derive the real constants and read a zero table.
            for ple in ple_layers:
                args = ple.args
                mult, sizes, offsets = derive_ngram_hash_constants(
                    vocab_size=self._config.vocab_size,
                    ngram_size=args.ngram_size,
                    num_ngram_heads=args.num_ngram_heads,
                    ngram_vocab_size_base=args.ngram_vocab_size_base,
                    ple_layer_index=ple.ple_index,
                )
                emb = ple.ple_embedding
                emb.layer_multipliers.copy_(torch.tensor(mult, dtype=torch.int64))
                emb.ngram_heads_vocab_sizes.copy_(torch.tensor(sizes, dtype=torch.int64))
                emb.ngram_heads_offsets.copy_(torch.tensor(offsets, dtype=torch.int64))
                emb.attach_table(ZeroTable(offsets[-1] + sizes[-1], args.ngram_head_dim))
            return 0

        from .weight import load_ple_table

        table = load_ple_table(engine_config.model_path, self._config.qwen4_args)
        self._ple_table = table  # owns the pinned HostBank; keep it alive
        for ple in ple_layers:
            ple.ple_embedding.attach_table(
                PinnedUVATable(table.bank.tensor, float(table.weight_scale))
            )
        return table.bank.nbytes

    def forward(self) -> torch.Tensor:
        batch = get_global_ctx().batch
        return self.lm_head.forward(self.model.forward(batch.input_ids, batch))


__all__ = ["Qwen4ExpDecoderLayer", "Qwen4ExpForCausalLM", "Qwen4ExpModel", "build_linear_mixer"]
