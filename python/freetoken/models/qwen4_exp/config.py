from __future__ import annotations

from dataclasses import dataclass, replace
from fnmatch import fnmatch
from typing import Any, Tuple

import torch

from freetoken.models.config import (
    FullAttentionGroupConfig,
    LinearGatedDeltaGroupConfig,
    ModelConfig,
    RotaryConfig,
    SlotStateSpec,
)


@dataclass(frozen=True)
class Qwen4ExpArgs:
    """Qwen3.8-Flash-Next geometry beyond the generic ModelConfig fields (ModelConfig.qwen4_args)."""

    hidden_size: int
    # Hyper-connections: every layer reads/writes hc_count residual streams [T, hc_count*hidden].
    hc_count: int
    hc_lowrank: int
    # PLE n-gram embedding; layer ids are zero-based decoder layers.
    ple_layer_ids: Tuple[int, ...]
    ple_embed_dim: int
    ple_conv_kernel_size: int
    ngram_size: int
    heads_per_ngram: int
    ngram_vocab_size_base: int
    make_ngram_vocab_size_divisible_by: int
    split_ngram_parts: int
    # n-gram hash windows never cross this token (the eos id); they restart after it.
    ngram_boundary_token_id: int
    # QSA indexer scoring geometry (the slab/ratio geometry lives on the attention group).
    index_n_heads: int
    index_kv_heads: int
    index_head_dim: int
    index_budget: int
    index_ratio: int
    # MTP (multi-token-prediction) draft head, from ``text_config.mtp``. Zero layers means the
    # checkpoint ships no head. The head is only built/loaded when mtp_load_enabled().
    mtp_num_layers: int = 0
    mtp_layer_types: Tuple[str, ...] = ()
    mtp_use_dedicated_embeddings: bool = False
    # Load-time FP8 of the checkpoint's bf16 dense linears (--dense-quant fp8): the selected
    # parts (see models.config.DENSE_FP8_ALL_PARTS). Empty = the checkpoint's own dtypes.
    dense_fp8_parts: Tuple[str, ...] = ()

    def part_quant(self, part: str, quant: str) -> str:
        """The quant mode a dense part is built/loaded with. ``quant`` is the checkpoint-
        derived flag (attn_quant / dense_quant / lm_head_quant); under load-time FP8 only
        the selected parts take it, the rest stay bf16."""
        if not self.dense_fp8_parts:
            return quant
        return quant if part in self.dense_fp8_parts else "none"

    @property
    def index_topk_blocks(self) -> int:
        return self.index_budget // self.index_ratio

    @property
    def num_ngram_heads(self) -> int:
        # one head group per n-gram order 2..ngram_size (Qwen3.8: 8 x 2-gram + 8 x 3-gram)
        return (self.ngram_size - 1) * self.heads_per_ngram

    @property
    def ngram_head_dim(self) -> int:
        return self.ple_embed_dim // self.num_ngram_heads

    @property
    def ple_conv_dilation(self) -> int:
        # HF Qwen4ExpTextPLELayer sets the depthwise conv dilation to ngram_size
        return self.ngram_size

    @property
    def ple_conv_state_len(self) -> int:
        return (self.ple_conv_kernel_size - 1) * self.ple_conv_dilation

    @property
    def ple_state_width(self) -> int:
        return self.hc_count * self.hidden_size


PLE_CONV_STATE = "ple_conv"
PLE_NGRAM_STATE = "ple_ngram_ctx"


def ple_slot_states(args: Qwen4ExpArgs) -> Tuple[SlotStateSpec, ...]:
    """Per-request PLE state riding the linear-state slots (see LinearStatePool.slot_states)."""
    if not args.ple_layer_ids:
        return ()
    return (
        # dilated-conv left context; replicated (not TP-sharded), model dtype
        SlotStateSpec(
            name=PLE_CONV_STATE,
            shape=(args.ple_state_width, args.ple_conv_state_len),
            layer_ids=args.ple_layer_ids,
        ),
        # last ngram_size-1 token ids, shared by every PLE layer; eos = hash boundary
        SlotStateSpec(
            name=PLE_NGRAM_STATE,
            shape=(args.ngram_size - 1,),
            dtype=torch.int32,
            fill_value=float(args.ngram_boundary_token_id),
        ),
    )


# Load-time FP8 targets by state-dict suffix, per part (the constructors build the matching
# Fp8PerTensor* module for exactly these). ``mtp.*`` names are targets only under the ``mtp``
# part; ``mtp.fc_*`` never.
_DENSE_FP8_SUFFIXES: dict[str, tuple[str, ...]] = {
    "gdn": (".linear_attn.in_proj_qkvz.weight", ".linear_attn.out_proj.weight"),
    "attn": (".self_attn.qkv_proj.weight", ".self_attn.o_proj.weight"),
    "index": (".self_attn.indexer.index_qk_proj.weight",),
    "hc": (
        ".input_mix_weight_down_block_inject.weight",
        ".input_mix_weight_up.weight",
        ".input_mix_weight_down.weight",
    ),
    "shared": (".mlp.shared_expert.gate_up_proj.weight", ".mlp.shared_expert.down_proj.weight"),
    "ple": (".ple.key_proj.weight", ".ple.value_proj.weight"),
}


def dense_fp8_part_of(name: str, parts) -> str | None:
    """Which FP8 part (if any) the state-dict tensor ``name`` belongs to under ``parts``."""
    if not parts:
        return None
    if name == "lm_head.weight":
        return "lm_head" if "lm_head" in parts else None
    if name.startswith("mtp."):
        if "mtp" not in parts or name.startswith("mtp.fc_"):
            return None
    for part, suffixes in _DENSE_FP8_SUFFIXES.items():
        if part in parts and name.endswith(suffixes):
            return part
    return None


def _quant_get(hf_config: Any):
    quant = getattr(hf_config, "quantization_config", None)
    if quant is None:
        return None
    return quant.get if isinstance(quant, dict) else (lambda k, d=None: getattr(quant, k, d))


def _ignored(patterns, module_name: str) -> bool:
    return any(fnmatch(module_name, pat) for pat in patterns)


def _layer_types(text: Any) -> list[str]:
    layer_types = getattr(text, "layer_types", None)
    if layer_types is not None:
        # HF Qwen4ExpTextConfig rewrites full_attention to qwen_sparse_attention in __post_init__.
        return [
            "full_attention" if t == "qwen_sparse_attention" else t for t in layer_types
        ]
    # Fall back to full_attention_interval: every Nth layer (1-indexed) is full.
    interval = int(getattr(text, "full_attention_interval", 4))
    n = int(text.num_hidden_layers)
    return [
        "full_attention" if (i + 1) % interval == 0 else "linear_attention"
        for i in range(n)
    ]


def parse_config(hf_config: Any) -> ModelConfig:
    text = getattr(hf_config, "text_config", hf_config)

    head_dim = (
        getattr(text, "head_dim", None)
        or text.hidden_size // text.num_attention_heads
    )
    num_kv_heads = getattr(text, "num_key_value_heads", text.num_attention_heads)

    rope_params = getattr(text, "rope_parameters", None) or {}
    rope_theta = rope_params.get("rope_theta", getattr(text, "rope_theta", None))
    partial = (
        rope_params.get("partial_rotary_factor")
        or getattr(text, "partial_rotary_factor", None)
        or 1.0
    )
    # int(), not round(): HF configuration_qwen4_exp truncates head_dim * partial.
    rotary_dim = int(head_dim * partial)

    # Text-only serving with the default rope type: the mRoPE sections reduce to standard
    # partial rope, and the unhashable ``mrope_section`` list must not reach get_rope's
    # cache key.
    rope_type = rope_params.get("rope_type", "default")
    rope_scaling = (
        None
        if rope_type in (None, "default")
        else {k: v for k, v in rope_params.items() if not isinstance(v, (list, dict))}
    )

    get = _quant_get(hf_config)
    if get is None:
        expert_quant = attn_quant = dense_quant = lm_head_quant = "none"
    else:
        algo = str(get("quant_algo") or get("quant_method") or "").lower()
        block = get("weight_block_size")
        if algo == "fp8" and block:
            # Official FP8 build (DeepSeek-V3-style block-fp8): only the routed experts
            # are quantized (fp8-e4m3 weights + per-block weight_scale_inv); attention,
            # GDN, the shared expert, HC, PLE and lm_head stay bf16.
            bs = tuple(int(x) for x in block)
            assert bs == (128, 128), f"only 128x128 block-fp8 is supported, got {bs}"
            expert_quant = "fp8_block"
            attn_quant = dense_quant = lm_head_quant = "none"
        else:
            is_fp4 = "fp4" in algo
            ignore = list(get("ignore") or [])

            # The RadixArk NVFP4 build quantizes only the routed experts; attention/GDN,
            # the shared expert, HC, PLE and lm_head all sit in the modelopt ignore list
            # and stay bf16. Derive every flag from that list instead of assuming the split.
            def _quant(probe: str) -> str:
                return "nvfp4" if is_fp4 and not _ignored(ignore, probe) else "none"

            prefix = "model.language_model.layers.0"
            expert_quant = _quant(f"{prefix}.mlp.experts.0.gate_proj")
            dense_quant = _quant(f"{prefix}.mlp.shared_expert.gate_proj")
            attn_quant = _quant(f"{prefix}.self_attn.q_proj")
            lm_head_quant = _quant("lm_head")

    # --dense-quant fp8: upgrade the bf16 dense parts to per-channel FP8 quantized at load.
    # Only ever upgrades "none" (a checkpoint-native quant is never overridden); the part
    # set refines which modules take the flag (Qwen4ExpArgs.part_quant).
    from freetoken.models.config import dense_fp8_parts

    fp8_parts = dense_fp8_parts()
    if fp8_parts:
        if attn_quant == "none" and fp8_parts & {"gdn", "attn", "index"}:
            attn_quant = "fp8_pertensor"
        if dense_quant == "none" and fp8_parts & {"hc", "shared", "ple"}:
            dense_quant = "fp8_pertensor"
        if lm_head_quant == "none" and "lm_head" in fp8_parts:
            lm_head_quant = "fp8_pertensor"

    layer_types = _layer_types(text)
    full_ids = tuple(i for i, t in enumerate(layer_types) if t == "full_attention")
    linear_ids = tuple(i for i, t in enumerate(layer_types) if t == "linear_attention")

    # HF stores ple_layer_ids one-indexed (validated upstream as [1, num_layers]).
    ple_layer_ids = tuple(int(i) - 1 for i in (getattr(text, "ple_layer_ids", None) or ()))
    for lid in ple_layer_ids:
        if layer_types[lid] != "linear_attention":
            raise ValueError(f"PLE must sit on a linear_attention layer, got layer {lid}")

    full_rotary = RotaryConfig(
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        max_position=text.max_position_embeddings,
        base=rope_theta,
        scaling=rope_scaling,
    )
    full_group = FullAttentionGroupConfig(
        name="full",
        layer_ids=full_ids,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        rotary_config=full_rotary,
        index_head_dim=int(text.indexer_head_dim),
        num_index_layers=len(full_ids),
        index_ratio=int(text.indexer_compress_ratio),
    )
    linear_group = LinearGatedDeltaGroupConfig(
        name="linear",
        layer_ids=linear_ids,
        num_key_heads=text.linear_num_key_heads,
        num_value_heads=text.linear_num_value_heads,
        key_head_dim=text.linear_key_head_dim,
        value_head_dim=text.linear_value_head_dim,
        conv_kernel_dim=text.linear_conv_kernel_dim,
        # HF resolves a null output_gate_type to hidden_act; mirror that instead of
        # stringifying None.
        output_gate=str(getattr(text, "output_gate_type", None) or text.hidden_act),
    )
    # Order groups by their first layer id for deterministic iteration.
    groups = tuple(
        sorted(
            (full_group, linear_group),
            key=lambda g: g.layer_ids[0] if g.layer_ids else 1 << 30,
        )
    )

    num_experts = int(getattr(text, "num_experts", 0) or 0)

    # HF accepts int | list here and uses the first entry (modeling_qwen4_exp Qwen4ExpTextNGramEmbedding)
    eos_token_id = text.eos_token_id
    if isinstance(eos_token_id, (list, tuple)):
        eos_token_id = eos_token_id[0]

    # ``text.mtp`` is a plain dict in shipped configs; tolerate an object for typed configs.
    mtp = getattr(text, "mtp", None)
    mtp_get = (mtp.get if isinstance(mtp, dict) else lambda k, d=None: getattr(mtp, k, d)) if mtp else (lambda k, d=None: d)
    mtp_num_layers = int(
        mtp_get("num_hidden_layers", getattr(text, "mtp_num_hidden_layers", 0) or 0) or 0
    )
    mtp_layer_types = tuple(mtp_get("layer_types", ()) or ())

    qwen4_args = Qwen4ExpArgs(
        hidden_size=text.hidden_size,
        hc_count=int(text.hc_count),
        hc_lowrank=int(text.hc_lowrank),
        ple_layer_ids=ple_layer_ids,
        ple_embed_dim=int(text.ple_embed_dim),
        ple_conv_kernel_size=int(text.ple_conv_kernel_size),
        ngram_size=int(text.ngram_size),
        heads_per_ngram=int(text.heads_per_ngram),
        ngram_vocab_size_base=int(text.ngram_vocab_size_base),
        make_ngram_vocab_size_divisible_by=int(text.make_ngram_vocab_size_divisible_by),
        split_ngram_parts=int(text.split_ngram_parts),
        ngram_boundary_token_id=int(eos_token_id),
        index_n_heads=int(text.indexer_n_heads),
        index_kv_heads=int(text.indexer_kv_heads),
        index_head_dim=int(text.indexer_head_dim),
        index_budget=int(text.indexer_budget),
        index_ratio=int(text.indexer_compress_ratio),
        mtp_num_layers=mtp_num_layers,
        mtp_layer_types=mtp_layer_types,
        mtp_use_dedicated_embeddings=bool(getattr(text, "mtp_use_dedicated_embeddings", False)),
        dense_fp8_parts=tuple(sorted(fp8_parts)),
    )

    return ModelConfig(
        num_layers=text.num_hidden_layers,
        num_qo_heads=text.num_attention_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        hidden_size=text.hidden_size,
        vocab_size=text.vocab_size,
        intermediate_size=getattr(text, "intermediate_size", 0) or 0,
        hidden_act=text.hidden_act,
        rms_norm_eps=text.rms_norm_eps,
        tie_word_embeddings=bool(getattr(text, "tie_word_embeddings", False)),
        rotary_config=full_rotary,
        num_experts=num_experts,
        num_experts_per_tok=int(getattr(text, "num_experts_per_tok", 0) or 0),
        moe_intermediate_size=int(getattr(text, "moe_intermediate_size", 0) or 0),
        shared_expert_intermediate_size=int(
            getattr(text, "shared_expert_intermediate_size", 0) or 0
        ),
        # Absent from the shipped config; HF Qwen4ExpTextConfig defaults it True and the
        # Qwen3_5MoE block renormalizes unconditionally -- keep the two in agreement.
        norm_topk_prob=bool(getattr(text, "norm_topk_prob", True)),
        moe_enabled=num_experts > 0,
        use_qk_norm=True,
        model_type=getattr(hf_config, "model_type", "qwen4_exp"),
        architectures=getattr(hf_config, "architectures", ["Qwen4ExpForConditionalGeneration"]),
        vision_config=None,  # served text-only
        image_token_id=getattr(hf_config, "image_token_id", None),
        attention_groups=groups,
        expert_quant=expert_quant,
        attn_quant=attn_quant,
        dense_quant=dense_quant,
        lm_head_quant=lm_head_quant,
        qwen4_args=qwen4_args,
        slot_states=ple_slot_states(qwen4_args),
    )


def extend_config_for_speculative(config: ModelConfig, num_draft_tokens: int) -> ModelConfig:
    """Register the MTP draft layer for speculative decoding.

    The MTP layer (id ``num_layers``) joins the full-attention group so the QSA backend, the
    KV pool (13th slab + index tier) and ``is_linear_layer`` all resolve it; ``num_layers``
    itself stays the main tower's count (it drives decoder construction and the MoE-offload
    layer count). ``num_speculative_tokens`` sizes the QSA pending ring in both the pool
    factory and the KV cost model.
    """
    if not config.qwen4_args.mtp_num_layers:
        raise ValueError("this checkpoint ships no MTP head; speculative mtp is unavailable")
    if num_draft_tokens < 1:
        raise ValueError(f"num_draft_tokens must be >= 1, got {num_draft_tokens}")
    mtp_layer_id = config.num_layers
    groups = tuple(
        replace(
            g,
            layer_ids=g.layer_ids + (mtp_layer_id,),
            num_index_layers=g.num_index_layers + 1,
        )
        if isinstance(g, FullAttentionGroupConfig) and mtp_layer_id not in g.layer_ids
        else g
        for g in config.attention_groups
    )
    return replace(
        config,
        attention_groups=groups,
        num_extra_kv_layers=1,
        num_speculative_tokens=num_draft_tokens,
    )


__all__ = [
    "PLE_CONV_STATE",
    "PLE_NGRAM_STATE",
    "Qwen4ExpArgs",
    "extend_config_for_speculative",
    "parse_config",
    "ple_slot_states",
]
