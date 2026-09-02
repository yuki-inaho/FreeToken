"""Weight loading for DeepSeek-V4-Flash (engine path).

  - :func:`iter_weights` streams resident (non-expert) tensors keyed to engine param
    names; model's ``load_state_dict`` casts each. ``wo_a`` dequantized to bf16 to match
    the reference bf16 einsum.
  - :func:`load_dsfp4_expert_sources` packs routed FP4 experts into pinned CPU banks for
    the offload cache. DeepSeek FP4: e8m0 per-32 block scale, no global scale.
"""

from __future__ import annotations

import collections
import json
import os
import re
from typing import Iterator

import safetensors
import torch
from tqdm import tqdm

from freetoken.models.loader import drop_page_cache

from .args import DeepseekV4Args, load_args


class _ShardReader:
    def __init__(self, folder: str, weight_map: dict, device):
        self._folder = folder
        self._weight_map = weight_map
        self._device = str(device)
        self._handles: dict[str, object] = {}

    def has(self, name: str) -> bool:
        return name in self._weight_map

    def get(self, name: str) -> torch.Tensor:
        shard = self._weight_map[name]
        handle = self._handles.get(shard)
        if handle is None:
            handle = safetensors.safe_open(
                os.path.join(self._folder, shard), framework="pt", device=self._device
            ).__enter__()
            self._handles[shard] = handle
        return handle.get_tensor(name)

    def close(self) -> None:
        for shard, handle in self._handles.items():
            try:
                handle.__exit__(None, None, None)
            except Exception:
                pass
            drop_page_cache(os.path.join(self._folder, shard))
        self._handles.clear()


def _weight_map(model_path: str) -> dict:
    with open(os.path.join(model_path, "model.safetensors.index.json")) as f:
        return json.load(f)["weight_map"]


def _dequant_fp8_block(weight: torch.Tensor, scale: torch.Tensor, block: int = 128) -> torch.Tensor:
    """Dequantize 128x128 block-scaled FP8 (e4m3) to bf16.

    scale is e8m0 exponent codes, ``value = 2^(code-127)`` (Triton FP8 GEMM convention).
    Used for ``wo_a`` to match the reference's bf16 einsum.
    """
    n, k = weight.shape
    codes = scale.view(torch.uint8).to(torch.float32)
    s = torch.exp2(codes - 127.0)
    s = s.repeat_interleave(block, dim=0).repeat_interleave(block, dim=1)[:n, :k]
    return (weight.to(torch.float32) * s).to(torch.bfloat16)


def iter_weights(
    model_path: str,
    device,
    *,
    include_moe_experts: bool = True,
    include_non_moe: bool = True,
):
    """Stream resident (non-expert) weights as ``(name, tensor)`` keyed to engine params.

    Routed FP4 experts come from the offload cache, so ``include_moe_experts`` must be
    False (DeepSeek-V4 only runs ``--moe-backend offload``). Tensors yielded in checkpoint
    dtype (fp8 + e8m0 preserved); ``wo_a`` dequantized to bf16 to match the reference einsum.
    """
    if include_moe_experts:
        raise ValueError(
            "DeepSeek-V4 routed experts are served from the offload cache; "
            "run with --moe-backend offload (include_moe_experts must be False)."
        )
    if not include_non_moe:
        return

    args = load_args(model_path, max_batch_size=1)
    reader = _ShardReader(model_path, _weight_map(model_path), device)

    def get(name: str) -> torch.Tensor:
        return reader.get(name)

    def linear(prefix: str):
        yield f"{prefix}.weight", get(f"{prefix}.weight")
        if reader.has(f"{prefix}.scale"):
            yield f"{prefix}.scale", get(f"{prefix}.scale")

    def block(prefix: str, *, ratio: int, hash_gate: bool):
        # One decoder block's resident tensors (main tower ``layers.L`` or DSpark ``mtp.s``;
        # the two share the block layout, DSpark stages being window-only bias-routed blocks).
        a = f"{prefix}.attn"
        yield from linear(f"{a}.wq_a")
        yield f"{a}.q_norm.weight", get(f"{a}.q_norm.weight")
        yield from linear(f"{a}.wq_b")
        yield from linear(f"{a}.wkv")
        yield f"{a}.kv_norm.weight", get(f"{a}.kv_norm.weight")
        # wo_a: FP8 in the checkpoint, dequantized to bf16 (reference bf16 einsum).
        yield f"{a}.wo_a", _dequant_fp8_block(
            get(f"{a}.wo_a.weight"), get(f"{a}.wo_a.scale")
        )
        yield from linear(f"{a}.wo_b")
        yield f"{a}.attn_sink", get(f"{a}.attn_sink")

        if ratio:
            c = f"{a}.compressor"
            yield f"{c}.ape", get(f"{c}.ape")
            yield f"{c}.wkv.weight", get(f"{c}.wkv.weight")
            yield f"{c}.wgate.weight", get(f"{c}.wgate.weight")
            yield f"{c}.norm.weight", get(f"{c}.norm.weight")
            if ratio == 4:
                idx = f"{a}.indexer"
                yield from linear(f"{idx}.wq_b")
                yield f"{idx}.weights_proj.weight", get(f"{idx}.weights_proj.weight")
                ic = f"{idx}.compressor"
                yield f"{ic}.ape", get(f"{ic}.ape")
                yield f"{ic}.wkv.weight", get(f"{ic}.wkv.weight")
                yield f"{ic}.wgate.weight", get(f"{ic}.wgate.weight")
                yield f"{ic}.norm.weight", get(f"{ic}.norm.weight")

        yield f"{prefix}.attn_norm.weight", get(f"{prefix}.attn_norm.weight")
        yield f"{prefix}.ffn_norm.weight", get(f"{prefix}.ffn_norm.weight")

        g = f"{prefix}.ffn.gate"
        yield f"{g}.weight", get(f"{g}.weight")
        if hash_gate:
            yield f"{g}.tid2eid", get(f"{g}.tid2eid")
        else:
            yield f"{g}.bias", get(f"{g}.bias")
        for proj in ("w1", "w2", "w3"):
            yield from linear(f"{prefix}.ffn.shared_experts.{proj}")

        for nm in (
            "hc_attn_fn", "hc_ffn_fn", "hc_attn_base",
            "hc_ffn_base", "hc_attn_scale", "hc_ffn_scale",
        ):
            yield f"{prefix}.{nm}", get(f"{prefix}.{nm}")

    try:
        yield "embed.weight", get("embed.weight")
        yield "norm.weight", get("norm.weight")
        yield "head", get("head.weight")
        for nm in ("hc_head_fn", "hc_head_base", "hc_head_scale"):
            yield nm, get(nm)

        for L in range(args.n_layers):
            yield from block(
                f"layers.{L}", ratio=args.compress_ratios[L], hash_gate=L < args.n_hash_layers
            )

        from .config import dspark_build_enabled

        if dspark_build_enabled(args):
            # DSpark drafter stages (speculative decoding): the block tensors plus stage 0's
            # context projection and the last stage's head / Markov / confidence weights.
            for s in range(args.n_mtp_layers):
                p = f"mtp.{s}"
                yield from block(p, ratio=0, hash_gate=False)
                if s == 0:
                    yield from linear(f"{p}.main_proj")
                    yield f"{p}.main_norm.weight", get(f"{p}.main_norm.weight")
                if s == args.n_mtp_layers - 1:
                    yield f"{p}.norm.weight", get(f"{p}.norm.weight")
                    for nm in ("hc_head_fn", "hc_head_base", "hc_head_scale"):
                        yield f"{p}.{nm}", get(f"{p}.{nm}")
                    for nm in ("markov_w1", "markov_w2"):
                        yield f"{p}.markov_head.{nm}.weight", get(f"{p}.markov_head.{nm}.weight")
                    yield from linear(f"{p}.confidence_head.proj")
    finally:
        reader.close()


# --------------------------------------------------------------------------------------
# Routed FP4 expert pinned banks.
# --------------------------------------------------------------------------------------
_EXPERT_RE = re.compile(
    r"^layers\.(?P<layer>\d+)\.ffn\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>w1|w2|w3)\.(?P<kind>weight|scale)$"
)


def load_dsfp4_expert_sources(
    model_path: str, args: DeepseekV4Args, *, layer_sink=None
) -> dict[str, list[torch.Tensor]]:
    """Build pinned CPU DeepSeek-FP4 banks for the routed experts.

    4 banks, one tensor per layer (independent allocations): ``gate_up_packed/scale``
    ``[E, 2I, H//2]`` uint8 / ``[..., H//32]`` e8m0, ``down_packed/scale`` ``[E, H, I//2]``
    / ``[..., I//32]``.

    ``layer_sink=None`` (serving): pin each layer as its writes complete, via an
    internally-owned :class:`PinPipeline`. ``layer_sink`` given (converter): the
    completion tracker fires into it instead -- nothing here is pinned, and the sink
    may release banks it has written out, so the returned tensors are only valid
    until then (the caller owns that tradeoff).
    """
    from freetoken.moe.host_banks import LayerCompletionTracker, PinPipeline, alloc_layer_banks

    folder = model_path
    weight_map = _weight_map(folder)
    L, E = args.n_layers, args.n_routed_experts
    H, I = args.dim, args.moe_inter_dim

    for shard in sorted(set(weight_map.values())):
        drop_page_cache(os.path.join(folder, shard))

    shards: dict[str, list[tuple[str, re.Match]]] = collections.defaultdict(list)
    for name, shard in weight_map.items():
        m = _EXPERT_RE.match(name)
        if m is None:
            continue
        if int(m.group("layer")) >= L:  # skip the MTP layer (index L)
            continue
        shards[shard].append((name, m))

    e8m0 = torch.float8_e8m0fnu
    specs = {  # alloc UNPINNED, fill, then pin-after-fill (skips slow cudaHostAlloc zero-fill)
        "gate_up_packed": ((E, 2 * I, H // 2), torch.uint8),
        "gate_up_scale": ((E, 2 * I, H // 32), e8m0),
        "down_packed": ((E, H, I // 2), torch.uint8),
        "down_scale": ((E, H, I // 32), e8m0),
    }
    hb = alloc_layer_banks(specs, L)
    banks = {name: [b.tensor for b in hb[name]] for name in specs}

    def _load(sink) -> int:
        tracker = LayerCompletionTracker(E * 6, hb, sink)  # {w1,w2,w3} x {weight,scale} x experts
        placed = 0
        for shard in tqdm(sorted(shards), desc="Loading DSV4 FP4 experts"):
            path = os.path.join(folder, shard)
            with safetensors.safe_open(path, framework="pt", device="cpu") as f:
                for name, m in shards[shard]:
                    layer = _place_dsfp4(banks, name, f.get_tensor(name), I)
                    tracker.note(layer)
                    placed += 1
            drop_page_cache(path)
        return placed

    if layer_sink is not None:
        placed = _load(layer_sink)
    else:
        with PinPipeline() as pins:
            placed = _load(pins)

    expected = L * E * 6  # {w1,w2,w3} x {weight, scale}
    assert placed == expected, f"loaded {placed} expert tensors, expected {expected}"
    return banks


def dummy_dsfp4_expert_sources(args: DeepseekV4Args) -> dict[str, list[torch.Tensor]]:
    """Fabricate the 4 ds_fp4 banks for --dummy-weight (no checkpoint on disk)."""
    from freetoken.moe.host_banks import alloc_layer_banks, pin_banks

    L, E = args.n_layers, args.n_routed_experts
    H, I = args.dim, args.moe_inter_dim
    e8m0 = torch.float8_e8m0fnu
    specs = {
        "gate_up_packed": ((E, 2 * I, H // 2), torch.uint8),
        "gate_up_scale": ((E, 2 * I, H // 32), e8m0),
        "down_packed": ((E, H, I // 2), torch.uint8),
        "down_scale": ((E, H, I // 32), e8m0),
    }
    hb = alloc_layer_banks(specs, L)
    banks = {name: [b.tensor for b in hb[name]] for name in specs}
    for t in banks["gate_up_packed"]:  # packed e2m1; scales stay 0 (valid e8m0)
        t.random_(0, 256)
    for t in banks["down_packed"]:
        t.random_(0, 256)
    pin_banks(hb)
    return banks


def is_expert_tensor(name: str) -> bool:
    """Predicate for the common parallel reader: is this a routed-expert tensor?"""
    return _EXPERT_RE.match(name) is not None


def _dsfp4_bank_specs(E: int, H: int, I: int) -> dict:
    """The 4 DeepSeek-FP4 bank row shapes: packed e2m1 codes + e8m0 per-32 block scales."""
    e8m0 = torch.float8_e8m0fnu
    return {
        "gate_up_packed": ((E, 2 * I, H // 2), torch.uint8),
        "gate_up_scale": ((E, 2 * I, H // 32), e8m0),
        "down_packed": ((E, H, I // 2), torch.uint8),
        "down_scale": ((E, H, I // 32), e8m0),
    }


# DSpark drafter experts: ``mtp.{stage}.ffn.experts.{e}.w{1,2,3}.{weight,scale}`` (stage ->
# bank layer ``n_layers + stage`` in the offload cache).
_DSPARK_EXPERT_RE = re.compile(
    r"^mtp\.(?P<layer>\d+)\.ffn\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>w1|w2|w3)\.(?P<kind>weight|scale)$"
)


def load_dspark_expert_bank_layers(
    model_path: str | None, args: DeepseekV4Args, *, dummy: bool = False
) -> dict[str, list[torch.Tensor]]:
    """Pinned ds_fp4 banks for the DSpark stages' routed experts: the same 4 banks as
    :func:`load_dsfp4_expert_sources`, one tensor per STAGE (appended to the main tower's
    bank lists by ``append_spec_expert_bank``). ``dummy`` fabricates them (no checkpoint).
    Always pinned: the stages run every speculative step, so they never take the CPU/locked
    residency of the main tower's split plan (whose layer ids would not even index them)."""
    from freetoken.moe.host_banks import (
        HostResidency, LayerCompletionTracker, PinPipeline, alloc_layer_banks, pin_banks,
    )

    S, E = args.n_mtp_layers, args.n_routed_experts
    H, I = args.dim, args.moe_inter_dim
    specs = _dsfp4_bank_specs(E, H, I)
    hb = alloc_layer_banks(specs, S)
    banks = {name: [b.tensor for b in hb[name]] for name in specs}
    if dummy:
        for t in banks["gate_up_packed"]:  # packed e2m1; scales stay 0 (valid e8m0)
            t.random_(0, 256)
        for t in banks["down_packed"]:
            t.random_(0, 256)
        pin_banks(hb)
        return banks

    assert model_path, "DSpark expert banks need the checkpoint path"
    weight_map = _weight_map(model_path)
    shards: dict[str, list[tuple[str, re.Match]]] = collections.defaultdict(list)
    for name, shard in weight_map.items():
        m = _DSPARK_EXPERT_RE.match(name)
        if m is None or int(m.group("layer")) >= S:
            continue
        shards[shard].append((name, m))

    with PinPipeline() as pins:
        def on_stage(stage_id: int, stage_banks: dict) -> None:
            for bank in stage_banks.values():
                pins.submit(bank, HostResidency.PINNED.value)

        tracker = LayerCompletionTracker(E * 6, hb, on_stage)
        placed = 0
        for shard in tqdm(sorted(shards), desc="Loading DSpark FP4 experts"):
            path = os.path.join(model_path, shard)
            with safetensors.safe_open(path, framework="pt", device="cpu") as f:
                for name, m in shards[shard]:
                    _place_dsfp4_parsed(
                        banks, int(m.group("layer")), int(m.group("expert")),
                        m.group("proj"), m.group("kind"), f.get_tensor(name), I,
                    )
                    tracker.note(int(m.group("layer")))
                    placed += 1
            drop_page_cache(path)
    expected = S * E * 6
    assert placed == expected, f"loaded {placed} DSpark expert tensors, expected {expected}"
    return banks


def _place_dsfp4(banks: dict, name: str, t: torch.Tensor, I: int) -> int:
    """Copy one expert tensor into its layer/expert slot (shared by serial + parallel
    readers): w1->gate_up[:,:I], w3->gate_up[:,I:], w2->down. Returns the layer index."""
    m = _EXPERT_RE.match(name)
    layer, expert = int(m.group("layer")), int(m.group("expert"))
    return _place_dsfp4_parsed(banks, layer, expert, m.group("proj"), m.group("kind"), t, I)


def _place_dsfp4_parsed(
    banks: dict, layer: int, expert: int, proj: str, kind: str, t: torch.Tensor, I: int
) -> int:
    """:func:`_place_dsfp4` on an already-parsed (layer, expert, proj, kind)."""
    if kind == "weight":
        t = t.view(torch.uint8)
        if proj == "w1":
            banks["gate_up_packed"][layer][expert, :I] = t
        elif proj == "w3":
            banks["gate_up_packed"][layer][expert, I:] = t
        else:  # w2 -> down
            banks["down_packed"][layer][expert] = t
    else:  # scale (e8m0)
        if proj == "w1":
            banks["gate_up_scale"][layer][expert, :I] = t
        elif proj == "w3":
            banks["gate_up_scale"][layer][expert, I:] = t
        else:
            banks["down_scale"][layer][expert] = t
    return layer


def load_dsfp4_expert_sources_parallel(
    model_path: str, args: DeepseekV4Args, *, workers: int = 8, chunk: int = 8 << 20,
    layer_sink=None,
) -> dict[str, list[torch.Tensor]]:
    """parallel path: same banks as load_dsfp4_expert_sources, filled from the common
    chunked multi-threaded O_DIRECT reader instead of serial per-shard safe_open.
    ``layer_sink``: see :func:`load_dsfp4_expert_sources`."""
    from freetoken.models.weight import iter_expert_tensors_parallel
    from freetoken.moe.host_banks import LayerCompletionTracker, PinPipeline, alloc_layer_banks

    L, E = args.n_layers, args.n_routed_experts
    H, I = args.dim, args.moe_inter_dim
    e8m0 = torch.float8_e8m0fnu
    specs = {
        "gate_up_packed": ((E, 2 * I, H // 2), torch.uint8),
        "gate_up_scale": ((E, 2 * I, H // 32), e8m0),
        "down_packed": ((E, H, I // 2), torch.uint8),
        "down_scale": ((E, H, I // 32), e8m0),
    }
    hb = alloc_layer_banks(specs, L)  # lazy host banks (unpinned)
    banks = {name: [b.tensor for b in hb[name]] for name in specs}

    def _is_expert(name: str) -> bool:
        m = _EXPERT_RE.match(name)
        return m is not None and int(m.group("layer")) < L  # skip the MTP layer (index L)

    def _load(sink) -> int:
        tracker = LayerCompletionTracker(E * 6, hb, sink)
        placed = 0
        for name, t in iter_expert_tensors_parallel(model_path, _is_expert, workers=workers, chunk=chunk):
            layer = _place_dsfp4(banks, name, t, I)
            tracker.note(layer)
            placed += 1
        return placed

    if layer_sink is not None:
        placed = _load(layer_sink)
    else:
        with PinPipeline() as pins:
            placed = _load(pins)

    expected = L * E * 6
    assert placed == expected, f"loaded {placed} expert tensors, expected {expected}"
    return banks


__all__ = [
    "iter_weights",
    "load_dsfp4_expert_sources",
    "load_dsfp4_expert_sources_parallel",
    "load_dspark_expert_bank_layers",
    "is_expert_tensor",
]
