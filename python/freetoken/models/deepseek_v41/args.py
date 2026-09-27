"""DeepSeek-V4.1-Flash hyperparameters.

Field names mirror the authors' ``inference/config.json`` (consumed by the reference
``ModelArgs``) so the port stays 1:1 with the reference; the defaults are the released
V4.1-Flash values. ``load_args`` reads that file from the checkpoint directory and, unlike
the 0731 loader, REFUSES unknown keys: V4.1 moved the architecture through several keyed
knobs (CSA2 source layers, the candidate pool, Engram, the DSpark stage MoE), and a key
silently dropped would be a topology change silently ignored. Keys the text-only port has
no use for (the vision tower, the image token, the sampling temperature) are warned about
and dropped instead.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, fields
from typing import ClassVar, Literal, Tuple

from freetoken.utils import init_logger

logger = init_logger(__name__)

# Keys the reference ModelArgs carries that this text-only port never reads. They are
# dropped with a warning rather than rejected so a checkpoint that ships them loads.
_IGNORED_KEY_PREFIXES = ("vision_",)
_IGNORED_KEYS = frozenset({"image_token_id", "temperature"})


@dataclass
class DeepseekV41Args:
    # ----- runtime -----
    max_batch_size: int = 1
    max_seq_len: int = 4096
    dtype: Literal["bf16", "fp8"] = "fp8"
    scale_fmt: Literal[None, "ue8m0"] = "ue8m0"
    expert_dtype: Literal[None, "fp4"] = "fp4"
    scale_dtype: Literal["fp32", "fp8"] = "fp8"

    # ----- shape -----
    vocab_size: int = 129280
    dim: int = 5120
    moe_inter_dim: int = 2304
    n_layers: int = 40
    n_mtp_layers: int = 3
    n_heads: int = 64

    # ----- moe -----
    n_routed_experts: int = 384
    n_shared_experts: int = 1
    n_activated_experts: int = 6
    score_func: Literal["softmax", "sigmoid", "sqrtsoftplus"] = "sqrtsoftplus"
    # Reference defaults that the shipped JSON does not spell out.
    gate_temp: float = 1.0
    norm_topk_prob: bool = True
    route_scale: float = 1.5
    swiglu_limit: float = 10.0

    # ----- mla -----
    q_lora_rank: int = 1280
    head_dim: int = 512
    rope_head_dim: int = 64
    norm_eps: float = 1e-20
    o_groups: int = 8
    o_lora_rank: int = 1024
    window_size: int = 128
    # One entry per layer, DSpark stages included: 0 = sliding window only, r = KV
    # compressed r-to-1 (V4.1: ratio 2 across the encoder, 1 across the decoder).
    compress_ratios: Tuple[int, ...] = (0, 0) + (2,) * 18 + (1,) * 20 + (0, 0, 0)
    # CSA2 ownership: layers sharing a ratio share one compressed KV (produced by the latest
    # kv-source at or before them) and one indexer K; index sources additionally run their own
    # indexer query (Reindex), everything else reuses the latest top-k (Reuse).
    kv_source_layers: Tuple[int, ...] = (2, 8, 14, 20)
    index_source_layers: Tuple[int, ...] = (2, 8, 14, 20, 24, 28, 32, 36)

    # ----- rope / yarn -----
    compress_rope_theta: float = 160000.0
    original_seq_len: int = 65536
    rope_theta: float = 10000.0
    rope_factor: float = 16
    beta_fast: int = 32
    beta_slow: int = 1

    # ----- lightning indexer + hierarchical candidate pool -----
    index_n_heads: int = 32
    index_head_dim: int = 128
    index_topk: int = 512
    # candidate_source_layer < 0 turns the pool off (the other two are then unused).
    candidate_source_layer: int = 20
    candidate_topk_blocks: int = 2048
    candidate_block_size: int = 8

    # ----- hyper-connections -----
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6

    # ----- engram (n-gram hash tables added to the residual before a few blocks) -----
    engram_layer_ids: Tuple[int, ...] = (1, 14)
    engram_num_embeddings: Tuple[int, ...] = (384006168, 384016682)  # table rows per layer
    engram_max_ngram_size: int = 4
    engram_vocab_size: int = 16000000  # prime-bucket search start per (n-gram size, head)
    engram_n_heads: int = 8
    engram_head_dim: int = 256
    engram_pad_id: int = 2
    engram_compressed_vocab_size: int = 99092

    # ----- DSpark block drafter (``mtp.{0..n_mtp_layers-1}.*``) -----
    dspark_block_size: int = 5
    dspark_noise_token_id: int = 0
    dspark_target_layer_ids: Tuple[int, ...] = (37, 38, 39)
    dspark_markov_rank: int = 256
    # Stage MoE geometry (0 = same as the main tower).
    dspark_n_routed_experts: int = 128
    dspark_n_activated_experts: int = 3

    # ----- runtime (set by extend_config_for_speculative) -----
    # Window-only KV layers past ``n_layers``: the DSpark stages' draft KV over the main
    # sequence. Sizes the window tier ONLY.
    n_extra_window_layers: int = 0

    # ----- fixed by the checkpoint format (not JSON keys) -----
    fp8_block_size: ClassVar[int] = 32  # dense FP8 weight blocks (and per-32 activations)
    fp4_block_size: ClassVar[int] = 32  # e8m0 per-32 groups: routed experts, indexer K/Q
    cmp_fp4_group: ClassVar[int] = 16  # e4m3-scaled E2M1 groups of the compressed KV

    _TUPLE_FIELDS: ClassVar[tuple[str, ...]] = (
        "compress_ratios",
        "kv_source_layers",
        "index_source_layers",
        "engram_layer_ids",
        "engram_num_embeddings",
        "dspark_target_layer_ids",
    )

    def __post_init__(self) -> None:
        for name in self._TUPLE_FIELDS:
            value = getattr(self, name)
            if not isinstance(value, tuple):
                setattr(self, name, tuple(int(v) for v in value))
        self._validate()

    # ------------------------------------------------------------------ validation
    def _validate(self) -> None:
        L = self.n_layers
        ratios = self.compress_ratios
        if len(ratios) < L:
            raise ValueError(
                f"compress_ratios has {len(ratios)} entries for n_layers={L} "
                f"(one per main-tower layer, DSpark stages optional)"
            )
        for name in ("kv_source_layers", "index_source_layers"):
            bad = [l for l in getattr(self, name) if not 0 <= l < L]
            if bad:
                raise ValueError(f"{name} entries {bad} outside [0, {L})")
        kv, idx = set(self.kv_source_layers), set(self.index_source_layers)
        if not kv <= idx:
            raise ValueError(
                f"kv_source_layers must be a subset of index_source_layers (a layer producing "
                f"compressed KV also produces its indexer K): {sorted(kv - idx)} are not"
            )
        for l in sorted(idx):
            if ratios[l] <= 0:
                raise ValueError(
                    f"layer {l} is a kv/index source but has compress_ratio {ratios[l]} "
                    f"(nothing to compress or index)"
                )
        for l in range(L):
            r = ratios[l]
            if r < 0:
                raise ValueError(f"compress_ratios[{l}] = {r} < 0")
            if r == 0:
                continue
            if self.window_size % r:
                raise ValueError(
                    f"window_size {self.window_size} must be a multiple of layer {l}'s "
                    f"compress_ratio {r} (page-aligned compressed groups)"
                )
            owner = self._latest_at_or_before(self.kv_source_layers, l)
            if owner is None:
                raise ValueError(
                    f"layer {l} has compress_ratio {r} but no kv_source layer at or before it"
                )
            if ratios[owner] != r:
                raise ValueError(
                    f"layer {l} (ratio {r}) would reuse the compressed KV of kv-source {owner} "
                    f"(ratio {ratios[owner]}); a consumer must match its owner's ratio"
                )
        cand = self.candidate_source_layer
        if cand >= 0:
            if cand not in idx:
                raise ValueError(
                    f"candidate_source_layer {cand} must be an index_source_layer "
                    f"{sorted(idx)} (the pool is picked from that layer's indexer scores)"
                )
            if self.candidate_topk_blocks <= 0 or self.candidate_block_size <= 0:
                raise ValueError(
                    "candidate_topk_blocks and candidate_block_size must be positive when "
                    f"candidate_source_layer={cand} enables the candidate pool"
                )
        bad = [t for t in self.dspark_target_layer_ids if not 0 <= t < L]
        if bad:
            raise ValueError(f"dspark_target_layer_ids entries {bad} outside [0, {L})")
        bad = [l for l in self.engram_layer_ids if not 0 <= l < L]
        if bad:
            raise ValueError(f"engram_layer_ids entries {bad} outside [0, {L})")
        if len(self.engram_num_embeddings) != len(self.engram_layer_ids):
            raise ValueError(
                f"engram_num_embeddings ({len(self.engram_num_embeddings)} entries) must list one "
                f"table size per engram layer ({len(self.engram_layer_ids)})"
            )
        if self.dspark_n_activated_experts > (self.dspark_n_routed_experts or self.n_routed_experts):
            raise ValueError(
                f"dspark_n_activated_experts {self.dspark_n_activated_experts} > stage experts "
                f"{self.dspark_n_routed_experts or self.n_routed_experts}"
            )
        if self.n_activated_experts > self.n_routed_experts:
            raise ValueError(
                f"n_activated_experts {self.n_activated_experts} > n_routed_experts "
                f"{self.n_routed_experts}"
            )

    @staticmethod
    def _latest_at_or_before(layers: Tuple[int, ...], layer_id: int) -> int | None:
        best = None
        for l in layers:
            if l <= layer_id and (best is None or l > best):
                best = l
        return best

    # ------------------------------------------------------------------ helpers
    def get_moe_config(self, layer_id: int) -> tuple[int, int]:
        """(routed, activated) expert counts: the main tower's for ``layer_id < n_layers``,
        the DSpark stage geometry for the stages (ids ``n_layers..``)."""
        if layer_id < self.n_layers:
            return self.n_routed_experts, self.n_activated_experts
        return (
            self.dspark_n_routed_experts or self.n_routed_experts,
            self.dspark_n_activated_experts or self.n_activated_experts,
        )

    def is_kv_source(self, layer_id: int) -> bool:
        return layer_id in self.kv_source_layers

    def is_index_source(self, layer_id: int) -> bool:
        return layer_id in self.index_source_layers

    def cmp_owner(self, layer_id: int) -> int | None:
        """The kv-source layer whose compressed KV (and indexer K) ``layer_id`` attends over:
        the latest kv-source at or before it. ``None`` for window-only layers."""
        if layer_id >= self.n_layers or self.compress_ratios[layer_id] == 0:
            return None
        return self._latest_at_or_before(self.kv_source_layers, layer_id)

    def idx_owner(self, layer_id: int) -> int | None:
        """The index-source layer whose top-k selection ``layer_id`` uses: itself for a
        source, else the latest index source at or before it. ``None`` for window-only layers."""
        if layer_id >= self.n_layers or self.compress_ratios[layer_id] == 0:
            return None
        return self._latest_at_or_before(self.index_source_layers, layer_id)

    def mode(self, layer_id: int) -> str:
        """CSA2 role of a main-tower layer: ``window`` (no compressed tier), ``full``
        (compresses its own KV + indexes it), ``reindex`` (reuses the compressed KV but runs
        its own indexer query) or ``reuse`` (reuses the latest top-k indices)."""
        if layer_id >= self.n_layers or self.compress_ratios[layer_id] == 0:
            return "window"
        if layer_id in self.kv_source_layers:
            return "full"
        if layer_id in self.index_source_layers:
            return "reindex"
        return "reuse"

    def decoder_start(self) -> int:
        """First decoder layer of the Causal Encoder-Decoder split: the last kv-source, whose
        compressed KV every later layer reads. Asserts the CED invariants the prefill relies
        on: the decoder runs on ONE ratio, no kv-source follows, and the candidate pool is
        picked at the decoder start (so the hoisted layer-20 production also seeds it)."""
        if not self.kv_source_layers:
            raise ValueError("no kv_source_layers: no encoder/decoder split")
        start = max(self.kv_source_layers)
        ratios = set(self.compress_ratios[start : self.n_layers])
        if len(ratios) != 1:
            raise ValueError(
                f"decoder layers {start}..{self.n_layers - 1} must share one compress_ratio, "
                f"got {sorted(ratios)}"
            )
        if self.candidate_source_layer != start:
            raise ValueError(
                f"candidate_source_layer {self.candidate_source_layer} must be the decoder "
                f"start {start} (the last kv_source layer)"
            )
        return start

    def uses_candidates(self, layer_id: int) -> bool:
        """Whether this index source scores only inside the candidate pool (the Reindex
        layers after the candidate source)."""
        return (
            self.candidate_source_layer >= 0
            and layer_id > self.candidate_source_layer
            and layer_id in self.index_source_layers
        )

    @property
    def nope_head_dim(self) -> int:
        return self.head_dim - self.rope_head_dim

    @property
    def dspark_enabled(self) -> bool:
        """Whether the checkpoint ships a DSpark drafter (block size, stages and target layers)."""
        return (
            self.dspark_block_size > 0
            and self.n_mtp_layers > 0
            and len(self.dspark_target_layer_ids) > 0
        )

    @property
    def n_window_layers(self) -> int:
        """Layers with a window-tier KV pool: the main tower plus any DSpark stages."""
        return self.n_layers + self.n_extra_window_layers

    @property
    def has_engram(self) -> bool:
        return len(self.engram_layer_ids) > 0


def _config_path(model_path: str) -> str:
    """Locate the authors' ModelArgs JSON inside the checkpoint directory."""
    candidates = [
        os.path.join(model_path, "inference", "config.json"),
        os.path.join(model_path, "model_args.json"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"No DeepSeek-V4.1 ModelArgs JSON found under {model_path} "
        f"(looked for inference/config.json)"
    )


def _is_ignored_key(key: str) -> bool:
    return key in _IGNORED_KEYS or key.startswith(_IGNORED_KEY_PREFIXES)


def parse_raw_args(raw: dict, *, strict: bool = True, **overrides) -> DeepseekV41Args:
    """Build :class:`DeepseekV41Args` from a decoded ``inference/config.json`` dict.

    Unknown keys raise ``ValueError`` (``strict=False`` downgrades that to a warning);
    the text-only ignore set (``vision_*``, ``image_token_id``, ``temperature``) warns.
    """
    valid = {f.name for f in fields(DeepseekV41Args)}
    ignored = sorted(k for k in raw if k not in valid and _is_ignored_key(k))
    unknown = sorted(k for k in raw if k not in valid and not _is_ignored_key(k))
    if ignored:
        logger.warning(
            "DeepSeek-V4.1 inference/config.json: ignoring text-only-irrelevant keys %s",
            ignored,
        )
    if unknown:
        msg = (
            f"DeepSeek-V4.1 inference/config.json has keys this port does not understand: "
            f"{unknown}. A new key is an architecture knob the port would silently ignore; "
            f"add it to DeepseekV41Args (or load with strict=False to proceed anyway)."
        )
        if strict:
            raise ValueError(msg)
        logger.warning(msg)
    kwargs = {k: v for k, v in raw.items() if k in valid}
    kwargs.update(overrides)
    return DeepseekV41Args(**kwargs)


def load_args(model_path: str, *, strict: bool = True, **overrides) -> DeepseekV41Args:
    """Build :class:`DeepseekV41Args` from the checkpoint's ``inference/config.json``.

    ``overrides`` (e.g. ``max_seq_len``, ``max_batch_size``) take precedence over the file,
    letting the runner size the per-request caches. See :func:`parse_raw_args` for the
    unknown-key policy.
    """
    with open(_config_path(model_path)) as f:
        raw = json.load(f)
    return parse_raw_args(raw, strict=strict, **overrides)


__all__ = ["DeepseekV41Args", "load_args", "parse_raw_args"]
