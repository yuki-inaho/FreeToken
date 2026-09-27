"""Weight loading for DeepSeek-V4.1-Flash (engine path).

  - :func:`resident_tensor_plan` is the pure decision: for every tensor name in the
    checkpoint index, what happens to it (yielded to the model, consumed by the expert bank
    loaders, or a documented skip). Everything else is an error -- the **consumed-set
    invariant**: a renamed or new tensor can never be silently dropped. The plan is torch-
    and disk-free so it is unit-tested on the real V4.1 name list without the checkpoint.
  - :func:`iter_weights` drives the 0731 :class:`_ShardReader` off that plan; ``wo_a`` is
    dequantized to bf16 (the reference's bf16 einsum), everything else is yielded in
    checkpoint dtype (fp8 + e8m0 preserved).
  - The routed FP4 experts have the 0731 ``ds_fp4`` bank layout byte for byte (per-32 e8m0
    scales, packed E2M1), so the 0731 bank builders are reused unchanged with H=5120,
    I=2304; only the DSpark stage banks change width (128 experts per stage).
"""

from __future__ import annotations

import re
from typing import Iterable, Iterator, NamedTuple

import torch

from freetoken.models.deepseek_v4.weight import (  # noqa: F401  (re-exported bank builders)
    _ShardReader,
    _dequant_fp8_block as _dequant_fp8_block_v4,
    _dsfp4_bank_specs,
    _place_dsfp4_parsed,
    _weight_map,
    dummy_dsfp4_expert_sources,
    is_expert_tensor,
    load_dsfp4_expert_sources,
    load_dsfp4_expert_sources_parallel,
)
from freetoken.models.deepseek_v4.weight import (
    load_dspark_expert_bank_layers as _load_dspark_expert_bank_layers_v4,
)

from .args import DeepseekV41Args, load_args

# Plan kinds.
PASSTHROUGH = "passthrough"  # yielded as-is under the same name (checkpoint dtype)
DEQUANT_WO_A = "dequant_wo_a"  # (weight, scale) -> one bf16 tensor
EXPERT = "expert"  # routed FP4 expert: consumed by the bank loaders, never yielded
SKIP = "skip"  # documented skip class (vision, image tokens, gate.bias_vl, Engram tables)


class ResidentPlan(NamedTuple):
    """``yielded``: ``(param_name, source_names, kind)`` for every tensor the engine consumes
    (``kind`` in {passthrough, dequant_wo_a, expert}); ``skipped``: names in a documented skip
    class; ``leftover``: names nothing accounts for -- must be empty."""

    yielded: list[tuple[str, tuple[str, ...], str]]
    skipped: list[str]
    leftover: list[str]


def _dequant_fp8_block(
    weight: torch.Tensor, scale: torch.Tensor, block: int | None = None
) -> torch.Tensor:
    """Block-scaled FP8 -> bf16 with the block inferred from the shapes (32 on V4.1; the
    0731 128 still resolves, so the helper is checkpoint-agnostic)."""
    return _dequant_fp8_block_v4(weight, scale, block=block)


# --------------------------------------------------------------------------------------
# The plan.
# --------------------------------------------------------------------------------------
_TOP_SKIP_PREFIXES = ("vision.", "aligner.", "image_")
_EXPERT_TAIL_RE = re.compile(r"^ffn\.experts\.(?P<expert>\d+)\.(w1|w2|w3)\.(weight|scale)$")
_BLOCK_RE = re.compile(r"^(?P<prefix>layers\.(?P<layer>\d+)|mtp\.(?P<stage>\d+))\.(?P<rest>.+)$")

# Tensors every block (main tower or DSpark stage) carries, relative to the block prefix.
_BLOCK_COMMON = frozenset(
    [f"attn.{p}.{k}" for p in ("wq_a", "wq_b", "wkv", "wo_b") for k in ("weight", "scale")]
    + ["attn.q_norm.weight", "attn.kv_norm.weight", "attn.attn_sink"]
    + ["attn_norm.weight", "ffn_norm.weight", "ffn.gate.weight", "ffn.gate.bias"]
    + [f"ffn.shared_experts.{p}.{k}" for p in ("w1", "w2", "w3") for k in ("weight", "scale")]
    + [f"hc_{s}_{k}" for s in ("attn", "ffn") for k in ("fn", "base", "scale")]
)
_BLOCK_SKIP = frozenset(["ffn.gate.bias_vl"])
# Present only where the layer produces compressed KV (kv source): the compressor and the
# indexer's key side. ``wgate`` ships at ratio 2 only (softmax-gated pooling); ratio 1 is
# a plain norm(wkv(x)), so its presence is decided by the index, not by us.
_KV_SOURCE_ONLY = frozenset(
    ["attn.compressor.norm.weight", "attn.compressor.wkv.weight", "attn.compressor.wgate.weight",
     "attn.indexer.wk.weight", "attn.indexer.k_norm.weight"]
)
# Present where the layer runs its own indexer query (kv sources and reindex layers).
_INDEX_SOURCE_ONLY = frozenset(
    ["attn.indexer.wq_b.weight", "attn.indexer.wq_b.scale", "attn.indexer.weights_proj.weight"]
)
_ENGRAM_RESIDENT = frozenset(
    ["engram.wkv.weight", "engram.wkv.scale", "engram.q_weight", "engram.k_weight"]
)
_ENGRAM_TABLE = frozenset(["engram.embed.weight", "engram.embed.scale"])
_STAGE0_ONLY = frozenset(["main_proj.weight", "main_proj.scale", "main_norm.weight"])
_LAST_STAGE_ONLY = frozenset(
    ["norm.weight", "markov_head.embed.weight", "markov_head.head.weight",
     "confidence_head.proj.weight", "confidence_head.proj.scale"]
)


def _classify(name: str, args: DeepseekV41Args, names: frozenset[str], dspark: bool) -> str:
    """Kind of one index name (PASSTHROUGH / DEQUANT_WO_A / EXPERT / SKIP / None = leftover).
    ``wo_a.scale`` is folded into its weight's dequant entry; ``wo_a.weight`` needs the
    scale present (V4.1 ships every dense linear as fp8 + e8m0)."""
    if name in ("embed.weight", "norm.weight", "head.weight"):
        return PASSTHROUGH
    if name.startswith(_TOP_SKIP_PREFIXES):
        return SKIP
    m = _BLOCK_RE.match(name)
    if m is None:
        return None
    rest = m.group("rest")
    if m.group("layer") is not None:
        L = int(m.group("layer"))
        if L >= args.n_layers:
            return None
        is_stage = False
    else:
        if not dspark:
            return SKIP  # the whole drafter, experts included, when it is not built
        s = int(m.group("stage"))
        if s >= args.n_mtp_layers:
            return None
        L = args.n_layers + s
        is_stage = True

    if rest in _BLOCK_COMMON:
        return PASSTHROUGH
    if rest in _BLOCK_SKIP:
        return SKIP
    if rest == "attn.wo_a.weight":
        return DEQUANT_WO_A if f"{m.group('prefix')}.attn.wo_a.scale" in names else None
    if rest == "attn.wo_a.scale":
        return PASSTHROUGH if f"{m.group('prefix')}.attn.wo_a.weight" in names else None
    em = _EXPERT_TAIL_RE.match(rest)
    if em is not None:
        return EXPERT if int(em.group("expert")) < args.get_moe_config(L)[0] else None
    if is_stage:
        if rest in _STAGE0_ONLY:
            return PASSTHROUGH if s == 0 else None
        if rest in _LAST_STAGE_ONLY:
            return PASSTHROUGH if s == args.n_mtp_layers - 1 else None
        return None
    if rest in _KV_SOURCE_ONLY:
        return PASSTHROUGH if args.is_kv_source(L) else None
    if rest in _INDEX_SOURCE_ONLY:
        return PASSTHROUGH if args.is_index_source(L) else None
    if rest in _ENGRAM_RESIDENT:
        return PASSTHROUGH if L in args.engram_layer_ids else None
    if rest in _ENGRAM_TABLE:
        return SKIP if L in args.engram_layer_ids else None
    return None


def resident_tensor_plan(
    index_names: Iterable[str], args: DeepseekV41Args, *, dspark: bool | None = None
) -> ResidentPlan:
    """Decide the fate of every checkpoint tensor (see the module docstring).

    ``dspark`` (default: :func:`dspark_build_enabled`) says whether the drafter is built;
    when it is not, every ``mtp.*`` tensor is a skip. Skip classes: ``vision.*``,
    ``aligner.*``, ``image_*``, ``*.ffn.gate.bias_vl`` (the VL gate bias) and the Engram
    tables ``layers.L.engram.embed.*`` (served from disk by :mod:`freetoken.engram`).
    Parameter naming follows the checkpoint except ``head.weight`` -> ``head`` and
    ``attn.wo_a.{weight,scale}`` -> ``attn.wo_a`` (bf16)."""
    if dspark is None:
        from .config import dspark_build_enabled

        dspark = dspark_build_enabled(args)
    names = frozenset(index_names)
    yielded: list[tuple[str, tuple[str, ...], str]] = []
    skipped: list[str] = []
    leftover: list[str] = []
    for name in index_names:
        kind = _classify(name, args, names, dspark)
        if kind is None:
            leftover.append(name)
        elif kind == SKIP:
            skipped.append(name)
        elif kind == DEQUANT_WO_A:
            base = name[: -len(".weight")]
            yielded.append((base, (name, base + ".scale"), DEQUANT_WO_A))
        elif kind == PASSTHROUGH and name.endswith(".wo_a.scale"):
            continue  # folded into the weight's dequant entry
        elif kind == PASSTHROUGH and name == "head.weight":
            yielded.append(("head", (name,), PASSTHROUGH))
        else:
            yielded.append((name, (name,), kind))
    return ResidentPlan(yielded, skipped, leftover)


def iter_weights(
    model_path: str,
    device,
    *,
    include_moe_experts: bool = False,
    include_non_moe: bool = True,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Stream resident (non-expert) weights as ``(name, tensor)`` keyed to engine params.

    Routed FP4 experts come from the offload cache, so ``include_moe_experts`` must be
    False (V4.1 only runs on the offload / hybrid MoE backends). Raises if the index holds
    a tensor the plan cannot account for (consumed-set invariant).
    """
    if include_moe_experts:
        raise ValueError(
            "DeepSeek-V4.1 routed experts are served from the offload cache; "
            "run with --moe-backend offload/hybrid (include_moe_experts must be False)."
        )
    if not include_non_moe:
        return

    args = load_args(model_path, max_batch_size=1)
    weight_map = _weight_map(model_path)
    plan = resident_tensor_plan(list(weight_map), args)
    if plan.leftover:
        shown = ", ".join(plan.leftover[:12])
        more = f" (+{len(plan.leftover) - 12} more)" if len(plan.leftover) > 12 else ""
        raise RuntimeError(
            f"DeepSeek-V4.1 checkpoint has {len(plan.leftover)} tensor(s) the loader does not "
            f"account for: {shown}{more}. Either the checkpoint layout changed or "
            f"inference/config.json disagrees with it; refusing to drop them silently."
        )
    reader = _ShardReader(model_path, weight_map, device)
    try:
        for param, sources, kind in plan.yielded:
            if kind == EXPERT:
                continue
            if kind == DEQUANT_WO_A:
                yield param, _dequant_fp8_block(reader.get(sources[0]), reader.get(sources[1]))
            else:
                yield param, reader.get(sources[0])
    finally:
        reader.close()


# --------------------------------------------------------------------------------------
# DSpark stage banks (128 experts per stage).
# --------------------------------------------------------------------------------------
def load_dspark_expert_bank_layers(
    model_path: str | None, args: DeepseekV41Args, *, dummy: bool = False
) -> dict[str, list[torch.Tensor]]:
    """Pinned ds_fp4 banks for the DSpark stages, ``dspark_n_routed_experts`` rows each (128:
    the stage MoE is narrower than the tower's 384 and the banks are NOT zero-padded -- that
    would pin 13 GiB of nothing; the offload cache records the per-layer row count instead).
    See the 0731 builder for the pinning contract."""
    return _load_dspark_expert_bank_layers_v4(
        model_path, args, dummy=dummy, n_experts=args.get_moe_config(args.n_layers)[0]
    )


def dspark_bank_bytes_estimate(args, *, built: bool | None = None) -> int:
    """Pinned bytes the DSpark stage banks will take (0 when the drafter is not built).
    Duck-typed on the args (0731 args lack ``dspark_n_routed_experts`` -> tower width) so
    the engine's host-RAM preflight can price either checkpoint."""
    from freetoken.moe.offload_cache import _BANK_BYTES_PER_EXPERT

    if built is None:
        from .config import dspark_build_enabled

        built = dspark_build_enabled(args)
    if not built:
        return 0
    E = getattr(args, "dspark_n_routed_experts", 0) or args.n_routed_experts
    per_expert = _BANK_BYTES_PER_EXPERT["ds_fp4"](args.dim, args.moe_inter_dim)
    return int(args.n_mtp_layers) * int(E) * int(per_expert)


__all__ = [
    "ResidentPlan",
    "resident_tensor_plan",
    "iter_weights",
    "load_dsfp4_expert_sources",
    "load_dsfp4_expert_sources_parallel",
    "dummy_dsfp4_expert_sources",
    "load_dspark_expert_bank_layers",
    "dspark_bank_bytes_estimate",
    "is_expert_tensor",
]
