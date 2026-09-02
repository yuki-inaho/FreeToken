"""Engine-facing config for DeepSeek-V4-Flash.

``parse_config`` maps the standard transformer fields the engine needs (layer
count, hidden size, vocab, MoE expert counts, etc.) into :class:`ModelConfig`,
and carries the full :class:`DeepseekV4Args` in ``ModelConfig.dsv4_args`` for the
DSV4-specific machinery (MLA sparse attention, CSA/HCA compressors, Lightning
Indexer, manifold-constrained Hyper-Connections, hash routing).

The transformers ``AutoConfig`` for ``deepseek_v4`` drops a few keys (e.g.
``compress_ratios``), so we recover the authoritative args from the checkpoint's
``inference/config.json`` via :func:`load_args`, keyed off ``hf_config._name_or_path``.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from freetoken.models.config import DSV4AttentionGroupConfig, ModelConfig, RotaryConfig

from .args import load_args


def parse_config(hf_config: Any) -> ModelConfig:
    model_path = getattr(hf_config, "_name_or_path", None) or getattr(
        hf_config, "name_or_path", None
    )
    if not model_path:
        raise ValueError(
            "DeepSeek-V4 parse_config needs the checkpoint path (hf_config._name_or_path)"
        )
    args = load_args(model_path, max_batch_size=1)

    rope_scaling = {
        "rope_type": "yarn",
        "factor": args.rope_factor,
        "beta_fast": args.beta_fast,
        "beta_slow": args.beta_slow,
        "original_max_position_embeddings": args.original_seq_len,
    }

    # Serving ceiling: the HF top-level config's max_position_embeddings (1M on V4-Flash).
    # args.original_seq_len is yarn's PRE-scaling length (64k); the compressed tiers serve
    # original_seq_len * rope_factor positions, so fall back to that product when a checkpoint
    # ships only the inference/ dialect.
    max_position = int(getattr(hf_config, "max_position_embeddings", 0) or 0)
    if max_position <= 0:
        max_position = int(args.original_seq_len * args.rope_factor)

    return ModelConfig(
        num_layers=args.n_layers,
        num_qo_heads=args.n_heads,
        num_kv_heads=1,  # MLA: a single shared latent KV head (K == V)
        head_dim=args.head_dim,
        hidden_size=args.dim,
        vocab_size=args.vocab_size,
        intermediate_size=args.moe_inter_dim,
        hidden_act="silu",
        rms_norm_eps=args.norm_eps,
        tie_word_embeddings=False,
        rotary_config=RotaryConfig(
            head_dim=args.head_dim,
            rotary_dim=args.rope_head_dim,
            max_position=max_position,
            base=args.rope_theta,
            scaling=rope_scaling,
        ),
        num_experts=args.n_routed_experts,
        num_experts_per_tok=args.n_activated_experts,
        moe_intermediate_size=args.moe_inter_dim,
        norm_topk_prob=True,
        model_type="deepseek_v4",
        architectures=["DeepseekV4ForCausalLM"],
        moe_enabled=True,
        expert_quant="ds_fp4",
        # NB: DSV4 has MoE on every layer (no dense-replace) and reads its routing /
        # shared-expert / scaling config from dsv4_args, so the generic DeepSeek-family
        # MoE ModelConfig fields (first_k_dense_replace / n_shared_experts /
        # routed_scaling_factor) are intentionally not set here.
        attn_sm_scale=args.head_dim**-0.5,
        dsv4_args=args,
        # Declares attn_type=DSV4 for the backend capability matrix (auto resolves to
        # dsv4_sparse, anything else is rejected at config time). Sizing stays on
        # dsv4_args; nothing generic prices this group.
        attention_groups=(
            DSV4AttentionGroupConfig(
                name="dsv4",
                layer_ids=tuple(range(args.n_layers)),
                num_kv_heads=1,  # MLA-style shared latent (K == V)
                head_dim=args.head_dim,
                sliding_window=args.window_size,
            ),
        ),
    )


def dspark_build_enabled(args) -> bool:
    """Whether the DSpark drafter is built and loaded: opt-in via the speculative-decoding
    env toggle (``FREETOKEN_LOAD_MTP``, set by ``--speculative-algorithm``) AND shipped by the
    checkpoint. Gates model construction and the weight loader identically."""
    from freetoken.models.config import mtp_load_enabled

    return mtp_load_enabled() and bool(args.dspark_enabled)


def extend_config_for_speculative(config: ModelConfig, num_draft_tokens: int) -> ModelConfig:
    """Register the DSpark drafter for speculative decoding.

    The ``n_mtp_layers`` DSpark stages (ids ``n_layers..``) keep a window-only KV over the
    main sequence, so they join the KV pool as extra WINDOW-tier layers
    (``dsv4_args.n_extra_window_layers`` / ``num_extra_kv_layers``); the compressed and
    indexer tiers are untouched. The drafter emits a fixed block of ``dspark_block_size``
    tokens per pass, so ``num_speculative_tokens`` (per-row scratch, verify graph widths)
    is the block size; ``num_draft_tokens`` may only trim how many of them are verified.
    """
    args = config.dsv4_args
    if not args.dspark_enabled:
        raise ValueError(
            "this checkpoint ships no DSpark drafter (dspark_block_size / n_mtp_layers / "
            "dspark_target_layer_ids in inference/config.json); speculative decoding is "
            "unavailable"
        )
    block = args.dspark_block_size
    if not 1 <= num_draft_tokens <= block:
        raise ValueError(
            f"--speculative-num-draft-tokens must be in [1, {block}] for DSpark (the drafter "
            f"emits a fixed block of {block} tokens), got {num_draft_tokens}"
        )
    n_stages = args.n_mtp_layers
    # The stages are window-only (ratio 0) layers; the 0731 checkpoint lists their ratios
    # after the main tower's, older/partial args may not.
    ratios = tuple(args.compress_ratios)
    n_total = args.n_layers + n_stages
    if len(ratios) < n_total:
        ratios = ratios + (0,) * (n_total - len(ratios))
    if any(ratios[args.n_layers : n_total]):
        raise ValueError(
            f"DSpark stages must be window-only layers, got compress_ratios "
            f"{ratios[args.n_layers:n_total]} for layers {args.n_layers}..{n_total - 1}"
        )
    for t in args.dspark_target_layer_ids:
        if not 0 <= t < args.n_layers:
            raise ValueError(f"dspark_target_layer_ids entry {t} outside [0, {args.n_layers})")
    args.compress_ratios = ratios
    args.n_extra_window_layers = n_stages
    return replace(
        config,
        num_extra_kv_layers=n_stages,
        num_speculative_tokens=block,
    )


__all__ = ["dspark_build_enabled", "extend_config_for_speculative", "parse_config"]
