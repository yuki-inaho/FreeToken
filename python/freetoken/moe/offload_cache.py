from __future__ import annotations

from typing import NamedTuple

import math
import os
from dataclasses import dataclass
from typing import Iterator

import torch
from flashlib.kernels.slot_cache import N_STATS, Stat

# Fuse the per-bank expert copies into a single multi-bank launch (one per copy_missing
# instead of one per bank). Set FREETOKEN_FUSED_COPY=0 to force the legacy per-bank path
# (kept for A/B profiling). Falls back to per-bank automatically if a bank's row bytes or
# base address are not 16-byte aligned.
_FUSED_COPY = os.getenv("FREETOKEN_FUSED_COPY", "1").strip().lower() not in {"0", "false", "no", "off"}

# cudaMemcpyBatchAsync silently degrades to a SYNCHRONOUS copy when a batch mixes
# large entries with sub-~256KB entries on registered host memory (H100 + CUDA 13.0,
# empirically bisected: a single 5-22KB entry beside one large entry blocks the
# calling thread for the full transfer; >=253KB entries never do). A synchronous
# call still moves bytes at full PCIe rate but stalls the host, which un-hides the
# GEMM under the copy in transition-zone workloads (gpt-oss 2048tok: -22% e2e).
# Banks whose rows are smaller than this ship as ONE whole-layer entry (their
# whole layer is tiny) and are excluded from the hit gather, so every per-run
# entry the batch sees is >= this size.
_SMALL_BANK_FEAT_BYTES = 256 * 1024


def moe_stats_log_interval() -> int:
    """FREETOKEN_MOE_STATS_LOG=N: log ``decode_miss_stats`` every N drained decode steps
    (0 = off). Setting it also force-enables ``collect_stats`` at engine init, so the env
    alone yields a live readout without --moe-collect-stats."""
    try:
        return max(0, int(os.getenv("FREETOKEN_MOE_STATS_LOG", "0") or "0"))
    except ValueError:
        return 0


from freetoken.utils import init_logger

logger = init_logger(__name__)

from contextlib import nullcontext as _nullcontext

# quant_format -> bank names, in registration order: the single place a format's bank
# layout is declared. The cache machinery (copy_missing, the prefill double buffers,
# bank_views) iterates banks in this order, the layers' kernel dispatch unpacks views
# in this order, and set_bank_sources validates against it.
_BANK_SCHEMAS: dict[str, tuple[str, ...]] = {
    # dense bf16 expert weights
    "bf16": ("gate_up", "down"),
    # DeepSeek-V3-style 128x128 block-fp8 experts (Qwen3.5-FP8): fp8-e4m3 weights +
    # bf16 per-block weight_scale_inv. gate_up [L*E, 2I, H] fp8 + gate_up_scale
    # [L*E, 2I//128, H//128] bf16; down [L*E, H, I] fp8 + down_scale [L*E, H//128, I//128].
    # Half the host/cache footprint of bf16; the grouped GEMM (kernel/triton/fp8_blockscale_moe)
    # reads the routed fp8 rows directly and dequantizes in the K-loop (no bf16 materialization).
    "fp8_block": ("gate_up", "gate_up_scale", "down", "down_scale"),
    # native GGUF Q4_0 experts: packed block bytes per output row, dequantized inside
    # the borrowed ggml MoE kernels. gate_up [L*E, 2I, H//32*18], down [L*E, H, I//32*18].
    "q4_0": ("gate_up", "down"),
    # native ModelOpt rows for the Triton inline-dequant kernels: packed e2m1 codes +
    # fp8-e4m3 per-16 block scales + per-output-row fp16 globals (w1/w3 carry distinct
    # globals, and folding them into the e4m3 block scales would underflow)
    "nvfp4": (
        "gate_up_packed",
        "gate_up_scale",
        "gate_up_global",
        "down_packed",
        "down_scale",
        "down_global",
    ),
    # pre-tiled layouts for the borrowed kernels; the globals are folded into the
    # block scales at repack time and collapse to [L*E] GPU-resident alpha vectors
    # (set_alphas), so they are not banks
    "nvfp4_marlin": ("gate_up_packed", "gate_up_scale", "down_packed", "down_scale"),
    "nvfp4_b12x": ("gate_up_packed", "gate_up_scale", "down_packed", "down_scale"),
    # gpt-oss mxfp4, transposed split-K layout (N innermost): per-expert blocks_t
    # [K//2, N] (uint8), scales_t [K//32, N] (uint8 e8m0), bias [N]. No folded alphas
    # (scales are a bank); split-K GEMV decode + transposed _t grouped prefill.
    "mxfp4_triton": (
        "gate_up_blocks",
        "gate_up_scales",
        "gate_up_bias",
        "down_blocks",
        "down_scales",
        "down_bias",
    ),
    # DeepSeek-V4 FP4: packed e2m1 codes + e8m0 per-32 block scales, no global scale
    # (4 banks). Read by DeepSeek-V4's own DS-FP4 grouped GEMV kernels via bank_views().
    "ds_fp4": ("gate_up_packed", "gate_up_scale", "down_packed", "down_scale"),
}

def fp8_block_scale_pad(rows: int, cols: int) -> int:
    """Trailing scale-bank dim padded so per-expert row bytes are 16B-aligned (fused copy)."""
    while (rows * cols * 2) % 16:
        cols += 1
    return cols


# bytes per (expert, layer) as f(hidden, moe_intermediate), from the bank shapes above; keep in sync with _BANK_SCHEMAS
# keyed by the config-time format tag (expert_quant / moe_weight_format), not quant_format: "mxfp4" sizes the mxfp4_triton banks, "nvfp4" also covers its repacked variants
_BANK_BYTES_PER_EXPERT = {
    "bf16": lambda H, I: 3 * I * H * 2,
    "fp8_block": lambda H, I: 3 * I * H + (
        (2 * I // 128) * fp8_block_scale_pad(2 * I // 128, H // 128)
        + (H // 128) * fp8_block_scale_pad(H // 128, I // 128)
    ) * 2,
    "q4_0": lambda H, I: 2 * I * (H // 32) * 18 + H * (I // 32) * 18,
    "nvfp4": lambda H, I: 2 * I * (H // 2 + H // 16 + 2) + H * (I // 2 + I // 16 + 2),
    "mxfp4": lambda H, I: 2 * I * (H // 2 + H // 32 + 2) + H * (I // 2 + I // 32 + 2),
    "ds_fp4": lambda H, I: 2 * I * (H // 2 + H // 32) + H * (I // 2 + I // 32),
}

# vLLM's marlin grouped-GEMM hands the full [cache_size] slot cache as its expert
# dimension; moe_align_block_size requires round_up(experts, 32) < 1024, i.e. <= 992.
MARLIN_MAX_CACHE_SIZE = 992



# decode routes per step (bs * top_k) are far below this; the ensure kernel's BLOCK_A must fit
ROUTING_PACK_COLS = 256
# Rows of the pinned routing pack: [cpu ids, fp32 router weight bits, tier-3 host slots].
# cpu_executor._io_for mirrors this exactly; decode_submit compares the two shapes and falls
# back to per-row copies if they differ, so the two MUST be changed together.
ROUTING_PACK_ROWS = 3


class HybridRoutes(NamedTuple):
    """The hybrid ensure kernel's route classes for one layer step (see
    ``OffloadMoeCache.ensure_experts_hybrid``)."""

    cpu_ids: torch.Tensor
    hit_ids: torch.Tensor | None
    fetched_ids: torch.Tensor | None
    uniq_slots: torch.Tensor | None
    uniq_hit: torch.Tensor | None
    uniq_fetched: torch.Tensor | None
    pairs: torch.Tensor | None


@dataclass
class OffloadMoeCache:
    num_layers: int
    num_experts: int
    cache_size: int
    device: torch.device
    cache_policy: str = "lru"
    prefill_overlap: bool = False
    # Prefill hit/miss split: experts already resident in the slot cache (slots
    # >= 2 * num_experts) are gathered device-side into the double buffer instead
    # of re-crossing PCIe; only the misses are H2D'd (one cudaMemcpyBatchAsync of
    # coalesced runs). Requires prefill_overlap, cache_size > 2 * num_experts and
    # the fused copy plan; silently falls back to the full-layer copy otherwise.
    prefill_hit_d2d: bool = False
    # "bf16" (default, dense expert weights) or one of the NVFP4 bank layouts:
    # "nvfp4" (native ModelOpt rows, FreeToken Triton kernels), "nvfp4_marlin"
    # (Marlin-tiled, vLLM W4A16 GEMM, sm_80-99) or "nvfp4_b12x" (flashinfer SM12x
    # W4A16); or "mxfp4_triton" (gpt-oss transposed split-K GEMV decode + _t grouped
    # prefill). The format names its bank layout (_BANK_SCHEMAS) and which kernels
    # may read the banks; the cache machinery itself is layout-agnostic.
    quant_format: str = "bf16"
    # Decode mode + bank layout; per-layer CPU routing is cpu_layer_ids. "gpu":
    # GPU-tiled banks, all decode on GPU (stream misses over PCIe into the slot
    # cache, GEMM on GPU). "cpu": native (CPU-readable) banks + a CPU executor;
    # decode computes experts on the CPU (the slot cache only backs the prefill
    # double buffer). "hybrid": native banks + a CPU executor + a full slot cache;
    # each layer fetches a capped subset of its misses over PCIe (``hybrid_max_fetch``
    # / ``hybrid_fetch_fraction`` below; the GPU computes those plus the hits) and the
    # CPU absorbs the overflow misses, then the partials merge. The CPU executor is
    # attached (set_cpu_executor) for cpu/hybrid, set whenever >=1 layer decodes on the CPU.
    decode_target: str = "gpu"
    # hybrid only: max experts fetched over PCIe per (layer, decode step); the rest
    # of that step's misses are computed on the CPU. 0 -> never fetch (CPU does every
    # miss, the GPU cache stays cold); large -> behaves like pure offload.
    hybrid_max_fetch: int = 1
    # hybrid only: when > 0, replaces the fixed cap with a per-step fraction -- fetch
    # ~fraction * misses experts over PCIe (rounded to whichever integer balances the
    # overlap best), the CPU computes the rest. The engine sets it to the benched
    # pcie_bw / cpu_bw ratio so the PCIe fetch and the CPU overflow GEMV take equal
    # time (perfect overlap): fetched : cpu = pcie : cpu - pcie.
    hybrid_fetch_fraction: float = 0.0

    def __post_init__(self) -> None:
        policy_ids = {"lru": 0}
        assert self.cache_policy in policy_ids
        assert self.decode_target in ("gpu", "cpu", "hybrid"), self.decode_target
        assert self.quant_format in _BANK_SCHEMAS, f"unknown quant_format {self.quant_format!r}"
        # Attached by the engine for decode_target == "cpu" (CpuMoeExecutor); None
        # for the GPU decode path.
        self.cpu_executor = None
        # Tier-3 (NVMe) support. host_slots is the number of expert rows each layer's HOST bank
        # actually holds: == num_experts (full residency, today's behaviour and the default), or
        # fewer when freetoken.tier3 is paging experts in from disk. The GPU copy path addresses
        # a host row as base[layer, bank] + row * feat_bytes, so a smaller per-layer bank needs
        # no kernel change -- only the meaning of `row` changes, from expert id to host slot,
        # and the ensure kernel writes that translation into src_indices.
        self.host_slots = self.num_experts
        self.host_tier = None  # tier3.Tier3Handle when the NVMe tier is active
        # --moe-residency-routing: max routing weight one substitution may displace. 0 = off.
        # Only ever consulted when host_tier is set (see OffloadMoELayer._maybe_residency_route).
        self.residency_route_tau = 0.0
        self.residency_route_max_subs = 1
        self.residency_route_total = 0.0
        # The measured policy (ft moe subst): per-expert output norms [L, E] on device, the
        # per-layer ||y||-estimator calibration as a python list, and the perturbation budget.
        # With a norm table and eps > 0 the layer DROPS host-missing experts instead of
        # substituting them; see kernel/triton/moe_residency_route.py.
        self.residency_route_norm = None
        self.residency_route_ycal = None
        self.residency_route_eps = 0.0
        self.residency_route_eps_total = 0.0
        self.residency_route_drop_renorm = False
        # [4] int64: substitutions, host-missing routes, rows substituted, rows that hit the cap
        self.residency_route_stats = None
        # tier3.PrefillStager: prefill needs a full [num_experts, ...] layer at expert-id
        # positions, which a host_slots-wide pool cannot provide; the stager assembles one.
        self._tier3_stager = None
        # Prefill layers whose NVMe reads are in flight: the resident half of the buffer is
        # gathered immediately, the disk half only once the reads land (see _tier3_finish_fill).
        self._tier3_pending_fill: dict[int, int] = {}
        # Device mirror of host_slot_for_id. The ensure kernel used to read the pinned map
        # directly over UVA -- a 512-lane uncoalesced host gather per layer per step, measured
        # at ~35% of decode throughput even when nothing was missing. The mirror is refreshed
        # with one 2 KB H2D per layer right after the resolver runs, so the kernel reads device
        # memory. It cannot go stale in a way that matters: a layer's residency only changes in
        # its own resolve(), and the refresh is ordered after that and before the next ensure.
        self.host_slot_dev = None
        # MoE layer ids whose decode runs on the CPU executor; the rest use the GPU
        # offload/PCIe path. Set by the engine after construction (empty = all-GPU,
        # all layers = the plain --moe-backend cpu case).
        self.cpu_layer_ids: frozenset = frozenset()
        # num_experts floor + nvfp4_marlin slot cap, shared with the runtime-rebuild path.
        self.validate_rebuild(self.cache_size)
        assert not self.prefill_overlap or self.cache_size >= 2 * self.num_experts, (
            "Prefill overlap borrows two full expert-layer buffers from the unified MoE "
            "cache, so cache_size must be at least 2 * num_experts "
            "(raise moe_cache_size or disable moe_prefill_overlap)"
        )
        self.cache_policy_id = policy_ids[self.cache_policy]
        self.slot_for_id = torch.full(
            (self.num_layers, self.num_experts),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        # Reverse map, in the flat id space flashlib's slot_cache works in:
        # id == layer_id * num_experts + expert, so one array replaces the (layer,
        # expert) pair and evicting a slot needs no decode.
        self.id_of_slot = torch.full(
            (self.cache_size,),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        self.usage = torch.zeros((self.cache_size,), dtype=torch.int64, device=self.device)
        self.step = torch.zeros((), dtype=torch.int64, device=self.device)
        self.active_mask = torch.zeros((self.num_experts,), dtype=torch.int32, device=self.device)
        # lru_ensure validates these against plan = min(batch * top_k, cache_size), so num_experts elements would under-size them
        plan_slots = max(self.num_experts, self.cache_size)
        self.evict_slots = torch.empty((plan_slots,), dtype=torch.int32, device=self.device)
        self.src_indices = torch.empty((plan_slots,), dtype=torch.int32, device=self.device)
        # hybrid only: the CPU executor's route list, written by the ensure kernel in the
        # same launch that rewrites expert_ids (raw expert id where the slot is -1, else
        # -1); sliced to the step's [bs * top_k] and viewed like expert_ids.
        # hybrid only: the CPU route list + the routes' router weights (fp32 bits) side by
        # side, so the executor ships both to the pool in ONE D2H memcpy (rows: ids, weights).
        # Row 2 (tier 3): the HOST SLOT each GPU-fetched route is about to read over UVA, -1
        # elsewhere. The hybrid path runs the CPU pass for a layer concurrently with that same
        # layer's PCIe gather, so without this the tier-3 installer could recycle a slot whose
        # bytes are mid-flight. It rides the same single D2H as the ids and weights.
        self.routing_pack = torch.empty((ROUTING_PACK_ROWS, ROUTING_PACK_COLS),
                                        dtype=torch.int32, device=self.device)
        self.cpu_ids = self.routing_pack[0]
        self.cpu_w = self.routing_pack[1]
        self.gpu_host_slots = self.routing_pack[2]
        # hybrid only: the GPU routes split by residency (hit = resident before this step,
        # fetched = given a slot this step), for the demand-copy overlap.
        self.hit_ids = torch.empty((plan_slots,), dtype=torch.int32, device=self.device)
        self.fetched_ids = torch.empty((plan_slots,), dtype=torch.int32, device=self.device)
        # hybrid only: the GPU routes grouped by expert for the dedup GEMV (see
        # offload_kernels.ensure_experts_hybrid): u-th distinct expert's slot (all / hit /
        # fetched) and its route indices, one row of DEDUP_MAX_TOKENS per expert.
        dedup_cap = max(16, 1 << (plan_slots - 1).bit_length())  # >= the kernel's BLOCK_A
        self.uniq_slots = torch.empty((dedup_cap,), dtype=torch.int32, device=self.device)
        self.uniq_hit = torch.empty((dedup_cap,), dtype=torch.int32, device=self.device)
        self.uniq_fetched = torch.empty((dedup_cap,), dtype=torch.int32, device=self.device)
        self.uniq_pairs = torch.empty((dedup_cap * 4,), dtype=torch.int32, device=self.device)
        self.num_indices = torch.zeros((1,), dtype=torch.int64, device=self.device)
        # hybrid only: full missing count BEFORE the per-step fetch cap (num_indices holds
        # the capped count that copy_missing actually fetches). The difference is what the
        # CPU computes this step. Written by the hybrid ensure kernel.
        self.num_missing_full = torch.zeros((1,), dtype=torch.int64, device=self.device)
        # hybrid only: per-(layer, expert) last-active decode step (LRU on the expert), -1
        # if never active. The hybrid ensure kernel reads it to pick which capped misses to
        # fetch (most-recently active first) and bumps it for every active expert.
        self.expert_recency = torch.full(
            (self.num_layers, self.num_experts), -1, dtype=torch.int64, device=self.device
        )
        # Q16 admission credit for recurring misses that the balanced hybrid split
        # otherwise rounds down to zero forever. One independently bounded credit
        # per layer; hit-only visits pause it and statistics resets preserve it.
        self.hybrid_admission_credit = torch.zeros(
            (self.num_layers,), dtype=torch.int32, device=self.device
        )
        # Host source banks (one [num_experts, ...] tensor per layer, so layers can
        # carry independent host attributes -- see layer_residency) and their GPU
        # slot caches, keyed by the format's bank schema (attached by
        # set_bank_sources). The GPU slot cache stays one unified pool per bank.
        self.bank_schema = _BANK_SCHEMAS[self.quant_format]
        self.bank_sources: dict[str, list[torch.Tensor]] = {}
        self.bank_caches: dict[str, torch.Tensor] = {}
        # per-layer host residency: the GPU movement paths require "pinned"; LOCKED/PAGEABLE layers decode on the CPU executor and prefill via copy_missing's pageable branch
        # _unpinned_layers is the derived id set the hot paths test against
        self.layer_residency: list[str] = []
        self._unpinned_layers: frozenset = frozenset()
        # marlin/b12x per-expert global scales ([L*E], GPU resident, see set_alphas).
        self.gate_up_alpha: torch.Tensor | None = None
        self.down_alpha: torch.Tensor | None = None
        # Opt-in decode miss-rate instrumentation. Accumulated on-device (no per-step host
        # sync); read via ``decode_miss_stats``. Graph-safe: the ``+=`` is captured into the
        # decode graph and re-executes with each replay's REAL routing (record_decode_stats
        # must be enabled before capture — see engine graph setup). The only graph artifact
        # is a one-off warm-up increment at capture time (<0.1% over a session).
        self.collect_stats = False
        # [num_layers, N_STATS] -- ensure_experts passes lru_stats[layer_id] straight to
        # the kernel, which accumulates in the same launch. The stat_* tensors below stay
        # for the hybrid path, whose kernel is still ours.
        self.lru_stats = torch.zeros(
            (self.num_layers, N_STATS), dtype=torch.int64, device=self.device
        )
        self.stat_missing = torch.zeros((), dtype=torch.int64, device=self.device)
        self.stat_active = torch.zeros((), dtype=torch.int64, device=self.device)
        self.stat_calls = torch.zeros((), dtype=torch.int64, device=self.device)
        # hybrid only: experts actually fetched over PCIe (<= stat_missing). The CPU
        # computes stat_missing - stat_fetched of them.
        self.stat_fetched = torch.zeros((), dtype=torch.int64, device=self.device)
        # Per-layer counterparts of the scalars above (indexed by MoE-layer id). Same
        # device-side accumulation (graph-safe: layer_id is a static index per graph node),
        # so one req's per-layer miss rate is readable via decode_miss_stats_per_layer().
        self.stat_missing_layer = torch.zeros(self.num_layers, dtype=torch.int64, device=self.device)
        self.stat_active_layer = torch.zeros(self.num_layers, dtype=torch.int64, device=self.device)
        self.stat_fetched_layer = torch.zeros(self.num_layers, dtype=torch.int64, device=self.device)
        self.stat_steps_layer = torch.zeros(self.num_layers, dtype=torch.int64, device=self.device)
        # Opt-in decode routing histogram (per layer, per expert) for cache-skew
        # analysis. Accumulated in ``ensure_experts`` from the raw expert ids before the
        # kernel rewrites them to slots. Only accurate with CUDA graphs disabled (the
        # captured graph would not re-run this host-side scatter on replay).
        self.collect_decode_freq = False
        self.decode_freq = torch.zeros(
            (self.num_layers, self.num_experts), dtype=torch.int64, device=self.device
        )
        # (per-layer sources, cache) per bank, in schema order. Every piece of cache
        # machinery that moves bank bytes (copy_missing, the prefill double buffers,
        # bank_views) iterates this list, so the slot cache is bank-count agnostic.
        self.banks: list[tuple[list[torch.Tensor], torch.Tensor]] = []
        # Fused multi-bank copy descriptor (built by set_bank_sources/_build_copy_plan).
        # Source pointers are per layer (_copy_src_ptrs[layer_id] -> [num_banks] device
        # tensor); dst/feat are layer-invariant.
        self._copy_fused_ok = False
        self._copy_dst_ptrs: torch.Tensor | None = None
        self._copy_src_ptrs: list[torch.Tensor] | None = None
        self._copy_feat_bytes: torch.Tensor | None = None
        # The layer whose misses ensure_experts/materialize_layer staged last; consumed
        # by copy_missing to pick the per-layer source (part of the same pending-copy
        # state as evict_slots/src_indices/num_indices).
        # _pending_whole_layer records WHICH staged it: the pageable branch is only sound after materialize_layer
        self._pending_src_layer: int | None = None
        self._pending_whole_layer = False
        # Speculative-prefetch copy plan ("plan-2"): its own dst/src/count so a prefetch
        # can be in flight on a side stream while the demand plan above is rewritten by
        # graph replays. src_indices2 carries FLAT ids (layer * num_experts + expert) --
        # the layer rides in the id, so there is no _pending_src_layer analogue. Allocated
        # by init_prefetch_plan (engine init, prefetch enabled only); None means disabled.
        self.max_prefetch = 0
        self.evict_slots2: torch.Tensor | None = None
        self.src_indices2: torch.Tensor | None = None
        self.num_indices2: torch.Tensor | None = None
        self.pf_stamp: torch.Tensor | None = None
        self.pf_issued: torch.Tensor | None = None
        self.pf_used: torch.Tensor | None = None
        self.pf_evicted_unused: torch.Tensor | None = None
        self.pf_steps: torch.Tensor | None = None
        # Per-bank [2, num_experts, ...] double-buffer views over the slot cache's
        # first 2 * num_experts slots (set up when prefill_overlap is enabled).
        self.prefill_bank_buffers: list[torch.Tensor] = []
        self.prefill_copy_stream: torch.cuda.Stream | None = None
        self.prefill_begin_event: torch.cuda.Event | None = None
        self.prefill_ready_events: list[torch.cuda.Event] = []
        self.prefill_release_events: list[torch.cuda.Event] = []
        self._prefill_buffer_layer: list[int | None] = [None, None]
        self._prefill_buffer_released: list[bool] = [True, True]
        self._prefill_buffer_has_release_event: list[bool] = [False, False]
        # hit-D2D split state: pinned begin-of-chunk snapshot of slot_for_id (the
        # classification input; frozen for the chunk -- no decode runs inside one,
        # and buffer invalidation only clears slot < 2E entries, which classify as
        # miss regardless), the lazily resolved batch-memcpy entry point (False =
        # unavailable), and row counters for cache reports.
        self._prefill_slot_snapshot: torch.Tensor | None = None
        self._prefill_snapshot_np = None
        self._prefill_hit_d2d_active = False
        self._hit_d2d_fallback_logged = False
        self._batch_memcpy = None
        self.prefill_hit_rows = 0
        self.prefill_total_rows = 0

    def set_bank_sources(
        self,
        sources: dict[str, list[torch.Tensor]],
        layer_residency: list[str] | None = None,
    ) -> None:
        """Attach the host (CPU pinned) expert source banks and allocate a GPU slot
        cache per bank, following the format's bank schema.

        Every bank is a list of ``num_layers`` tensors, one ``[num_experts, ...]``
        per layer (independent allocations, so each layer can carry its own host
        attributes); each slot cache mirrors the bank's row shape and dtype as one
        unified GPU pool. The row layouts are produced by the weight loaders /
        repackers (see ``_BANK_SCHEMAS`` and :mod:`freetoken.moe.nvfp4_backends`)
        -- the cache machinery is layout-agnostic and just moves rows.

        ``layer_residency`` labels each layer with a ``HostResidency`` value (default: all pinned).
        Non-pinned (LOCKED/PAGEABLE) layers have no device address: they must already be routed to the CPU executor (``cpu_layer_ids``, set BEFORE this call), the copy plan skips their rows, and their only movement is ``copy_missing``'s whole-layer pageable prefill branch -- which is why prefill overlap is incompatible with them.
        """
        from freetoken.moe.host_banks import HostResidency

        assert set(sources) == set(self.bank_schema), (
            f"banks {sorted(sources)} do not match the {self.quant_format!r} "
            f"schema {self.bank_schema}"
        )
        residency = layer_residency or [HostResidency.PINNED.value] * self.num_layers
        assert len(residency) == self.num_layers, (len(residency), self.num_layers)
        unpinned = frozenset(
            i for i, r in enumerate(residency) if r != HostResidency.PINNED.value
        )
        if unpinned:
            if not unpinned <= self.cpu_layer_ids:
                raise ValueError(
                    f"non-pinned layers {sorted(unpinned - self.cpu_layer_ids)} are not in "
                    f"cpu_layer_ids: a layer without a device address can only decode on "
                    f"the CPU executor (set cache.cpu_layer_ids before set_bank_sources)"
                )
            if self.prefill_overlap:
                raise ValueError(
                    "prefill overlap DMAs from registered banks; it must be disabled "
                    "when any layer is LOCKED/PAGEABLE (the engine does this)"
                )
        self._unpinned_layers = unpinned
        self.layer_residency = list(residency)
        for name in self.bank_schema:
            per_layer = sources[name]
            assert len(per_layer) == self.num_layers, (name, len(per_layer))
            head = per_layer[0]
            for layer_id, source in enumerate(per_layer):
                assert source.is_contiguous(), f"bank {name!r} layer {layer_id} must be contiguous"
                # Row COUNT may differ per layer; only the row SHAPE has to match, and that
                # is all the copy plan and the slot caches ever use (feat bytes come from
                # shape[1:]). Tier 3 relies on this: its paged layers are host_slots rows wide
                # while an appended MTP draft layer stays full num_experts and resident.
                assert source.size(0) in (self.host_slots, self.num_experts), (
                    name, layer_id, source.shape, self.host_slots, self.num_experts,
                )
                assert source.shape[1:] == head.shape[1:] and source.dtype == head.dtype, (
                    name, layer_id, source.shape, source.dtype,
                )
            self.bank_sources[name] = list(per_layer)
            self.bank_caches[name] = torch.empty(
                (self.cache_size, *head.shape[1:]),
                dtype=head.dtype,
                device=self.device,
            )
        self.banks = [(self.bank_sources[n], self.bank_caches[n]) for n in self.bank_schema]
        self._build_copy_plan()
        if self.prefill_overlap:
            self._init_prefill_overlap_buffers()

    def _build_copy_plan(self) -> None:
        """Precompute the fused multi-bank copy descriptor (base addrs + per-row bytes).

        Built once here (and on :meth:`rebuild`, which reallocates the slot caches);
        the addresses are fixed for the cache's lifetime so the descriptor tensors are
        CUDA-graph safe. Disabled (-> per-bank fallback) if any bank's row bytes or base
        address is not 16-byte aligned, or via FREETOKEN_FUSED_COPY=0.
        """
        self._copy_fused_ok = False
        self._copy_dst_ptrs = None
        self._copy_src_ptrs = None
        self._copy_feat_bytes = None
        self._src_ptr_table = None
        self._copy_dst_ptrs_host: list[int] = []
        self._copy_src_ptrs_host: list[list[int]] = []
        self._copy_feat_bytes_host: list[int] = []
        self._gather_bank_ids: list[int] = []
        self._gather_dst_ptrs: torch.Tensor | None = None
        self._gather_feat_bytes: torch.Tensor | None = None
        if not _FUSED_COPY or self.device.type != "cuda" or not self.banks:
            return
        from freetoken.kernel.pinned import device_ptr

        dst_ptrs, feats = [], []
        layer_src_ptrs = [[] for _ in range(self.num_layers)]
        for per_layer, cache in self.banks:
            feat = math.prod(per_layer[0].shape[1:]) * per_layer[0].element_size()
            if feat % 16 != 0 or cache.data_ptr() % 16 != 0:
                return  # leave fused disabled; copy_missing uses the per-bank path
            for layer_id, source in enumerate(per_layer):
                if layer_id in self._unpinned_layers:
                    # unregistered layer: no device alias exists, and the row is never consumed (CPU decode; pageable prefill)
                    # a 0 placeholder keeps the descriptor shape
                    layer_src_ptrs[layer_id].append(0)
                    continue
                # The kernel dereferences these on the GPU, so store each host bank's
                # device alias (== data_ptr() under UVA identity; differs on
                # Windows/WDDM).
                src_dev = device_ptr(source)
                if src_dev % 16 != 0:
                    return
                layer_src_ptrs[layer_id].append(src_dev)
            dst_ptrs.append(cache.data_ptr())
            feats.append(feat)
        self._copy_dst_ptrs = torch.tensor(dst_ptrs, dtype=torch.int64, device=self.device)
        self._copy_src_ptrs = [
            torch.tensor(ptrs, dtype=torch.int64, device=self.device)
            for ptrs in layer_src_ptrs
        ]
        self._copy_feat_bytes = torch.tensor(feats, dtype=torch.int64, device=self.device)
        # Prefetch ("plan-2") source table: [num_layers * num_banks] flattened, indexed by
        # layer * num_banks + bank inside fast_index_copy_multi_layered. Rebuilt here with
        # the rest of the descriptor, so rebuild() keeps it fresh for free. Unpinned
        # layers keep their 0 placeholders (never admitted; the kernel skips 0 bases).
        self._src_ptr_table = torch.stack(self._copy_src_ptrs).reshape(-1).contiguous()
        self._copy_dst_ptrs_host = dst_ptrs
        self._copy_src_ptrs_host = layer_src_ptrs
        self._copy_feat_bytes_host = feats
        # hit-D2D gather serves only the big banks; small banks are whole-layer
        # H2D entries (see _SMALL_BANK_FEAT_BYTES), so their rows never need D2D.
        self._gather_bank_ids = [i for i, f in enumerate(feats) if f >= _SMALL_BANK_FEAT_BYTES]
        if len(self._gather_bank_ids) == len(feats):
            self._gather_dst_ptrs = self._copy_dst_ptrs
            self._gather_feat_bytes = self._copy_feat_bytes
        elif self._gather_bank_ids:
            self._gather_dst_ptrs = self._copy_dst_ptrs[self._gather_bank_ids].contiguous()
            self._gather_feat_bytes = self._copy_feat_bytes[self._gather_bank_ids].contiguous()
        self._copy_fused_ok = True

    def validate_rebuild(self, cache_size: int) -> None:
        """Pure geometry validation of a rebuild target (no GPU side effects).

        Raises ``ValueError`` if ``cache_size`` is below the ``num_experts`` floor or
        above the marlin slot cap. Called by :meth:`rebuild` and by the engine's
        pre-teardown check, so an invalid target rejects with the old cache intact
        (no destructive free first).
        """
        if cache_size < self.num_experts:
            raise ValueError(f"cache_size {cache_size} < num_experts {self.num_experts}")
        if self.quant_format == "nvfp4_marlin" and cache_size > MARLIN_MAX_CACHE_SIZE:
            raise ValueError(
                f"moe_cache_size={cache_size} exceeds the marlin backend's slot limit of "
                f"{MARLIN_MAX_CACHE_SIZE} (vLLM moe_align_block_size caps padded experts at "
                "1024); reduce moe_cache_size or force --nvfp4-backend triton"
            )

    def rebuild(self, cache_size: int) -> None:
        """Resize the GPU slot cache + bookkeeping to ``cache_size`` IN PLACE.

        Keeps the CPU/pinned ``bank_sources`` and the GPU-resident alphas; never
        reloads banks. Tears down prefill-overlap buffers first (their views alias
        the old ``bank_caches``), frees the old GPU tensors, then reallocates. Slots
        cold-start after rebuild. Object identity is preserved so attached layers and
        ``ctx.moe_offload_cache`` stay valid.
        """
        assert self.bank_sources, "set_bank_sources must run before rebuild"
        self.validate_rebuild(cache_size)
        self._tier3_invalidate()
        # 1. Tear down prefill-overlap (its buffer views alias the old bank_caches).
        self.prefill_bank_buffers = []
        self.prefill_copy_stream = None
        self.prefill_begin_event = None
        self.prefill_ready_events = []
        self.prefill_release_events = []
        self._prefill_buffer_layer = [None, None]
        self._prefill_buffer_released = [True, True]
        self._prefill_buffer_has_release_event = [False, False]
        # 2. Drop old GPU tensors (free-before-alloc).
        self.banks = []
        self.bank_caches = {}
        self.cache_size = cache_size
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()
        # 3. Reallocate the slot cache from the retained host sources.
        for name in self.bank_schema:
            head = self.bank_sources[name][0]
            self.bank_caches[name] = torch.empty(
                (cache_size, *head.shape[1:]), dtype=head.dtype, device=self.device
            )
        self.banks = [(self.bank_sources[n], self.bank_caches[n]) for n in self.bank_schema]
        self._build_copy_plan()  # slot caches were reallocated -> refresh fused-copy addrs
        # 4. Reallocate cache_size-shaped bookkeeping; reset the slot map (cold start).
        self.slot_for_id.fill_(-1)
        self.id_of_slot = torch.full((cache_size,), -1, dtype=torch.int32, device=self.device)
        self.usage = torch.zeros((cache_size,), dtype=torch.int64, device=self.device)
        plan_slots = max(self.num_experts, cache_size)
        self.evict_slots = torch.empty((plan_slots,), dtype=torch.int32, device=self.device)
        self.src_indices = torch.empty((plan_slots,), dtype=torch.int32, device=self.device)
        # hybrid only: the CPU executor's route list, written by the ensure kernel in the
        # same launch that rewrites expert_ids (raw expert id where the slot is -1, else
        # -1); sliced to the step's [bs * top_k] and viewed like expert_ids.
        # hybrid only: the CPU route list + the routes' router weights (fp32 bits) side by
        # side, so the executor ships both to the pool in ONE D2H memcpy (rows: ids, weights).
        # ROUTING_PACK_ROWS, not a literal: the pack carries the tier-3 host-slot row too, and
        # cpu_executor shape-checks it against its pinned mirror and SILENTLY falls back to
        # per-row copies when they disagree -- so a rebuild that narrowed it would drop the
        # exclusion list without a word, and leave gpu_host_slots pointing at the freed tensor.
        self.routing_pack = torch.empty((ROUTING_PACK_ROWS, ROUTING_PACK_COLS),
                                        dtype=torch.int32, device=self.device)
        self.cpu_ids = self.routing_pack[0]
        self.cpu_w = self.routing_pack[1]
        self.gpu_host_slots = self.routing_pack[2]
        # hybrid only: the GPU routes split by residency (hit = resident before this step,
        # fetched = given a slot this step), for the demand-copy overlap.
        self.hit_ids = torch.empty((plan_slots,), dtype=torch.int32, device=self.device)
        self.fetched_ids = torch.empty((plan_slots,), dtype=torch.int32, device=self.device)
        # hybrid only: the GPU routes grouped by expert for the dedup GEMV (see
        # offload_kernels.ensure_experts_hybrid): u-th distinct expert's slot (all / hit /
        # fetched) and its route indices, one row of DEDUP_MAX_TOKENS per expert.
        dedup_cap = max(16, 1 << (plan_slots - 1).bit_length())  # >= the kernel's BLOCK_A
        self.uniq_slots = torch.empty((dedup_cap,), dtype=torch.int32, device=self.device)
        self.uniq_hit = torch.empty((dedup_cap,), dtype=torch.int32, device=self.device)
        self.uniq_fetched = torch.empty((dedup_cap,), dtype=torch.int32, device=self.device)
        self.uniq_pairs = torch.empty((dedup_cap * 4,), dtype=torch.int32, device=self.device)
        self.step.zero_()
        self.active_mask.zero_()
        self.num_indices.zero_()
        self.num_missing_full.zero_()
        self.expert_recency.fill_(-1)
        self.hybrid_admission_credit.zero_()
        self.stat_missing.zero_()
        self.stat_active.zero_()
        self.stat_calls.zero_()
        self.stat_fetched.zero_()
        self.stat_missing_layer.zero_()
        # a rebuild is a cold start for the cache; carrying pre-rebuild hit/miss counts over would skew every post-rebuild stats report
        self.lru_stats.zero_()
        self.stat_active_layer.zero_()
        self.stat_fetched_layer.zero_()
        self.stat_steps_layer.zero_()
        self.decode_freq.zero_()
        self.prefill_hit_rows = 0
        self.prefill_total_rows = 0
        # Plan-2 buffers are budget-shaped (not cache_size-shaped): keep, zero the state.
        if self.num_indices2 is not None:
            self.num_indices2.zero_()
            self.pf_stamp.zero_()
            self.pf_issued.zero_()
            self.pf_used.zero_()
            self.pf_evicted_unused.zero_()
            self.pf_steps.zero_()
        self._hit_d2d_fallback_logged = False  # geometry changed; re-log if still unusable
        # 5. Re-evaluate prefill overlap against the new size.
        if self.prefill_overlap and cache_size < 2 * self.num_experts:
            logger.warning(
                f"Disabling MoE prefill overlap on rebuild: cache_size {cache_size} "
                f"< 2*num_experts {2 * self.num_experts}."
            )
            self.prefill_overlap = False
        if self.prefill_overlap:
            self._init_prefill_overlap_buffers()

    def set_alphas(
        self, gate_up_alpha: torch.Tensor | None, down_alpha: torch.Tensor | None
    ) -> None:
        """Attach the marlin/b12x per-expert global scales (``[L*E]``, GPU resident).

        These are kernel-preprocessed scalars, far too small to bother offloading;
        the forward path looks them up per slot with :meth:`alphas_for_slots` /
        :meth:`alphas_for_layer` (pure device-side lookups, CUDA-graph safe).
        ``(None, None)`` is a no-op so callers can pass a format's (possibly
        absent) alphas through unconditionally.
        """
        if gate_up_alpha is None and down_alpha is None:
            return
        assert gate_up_alpha is not None and down_alpha is not None
        total = self.num_layers * self.num_experts
        assert gate_up_alpha.shape == down_alpha.shape == (total,)
        self.gate_up_alpha = gate_up_alpha.to(self.device)
        self.down_alpha = down_alpha.to(self.device)

    def set_cpu_executor(self, executor) -> None:
        """Attach the CPU MoE executor (``decode_target`` in {"cpu", "hybrid"}).

        The executor owns the persistent worker pool, the pinned activation/result
        IO buffers, and the ``cudaLaunchHostFunc`` submit/sync plumbing. It reads
        experts straight from this cache's host ``bank_sources`` (no extra copy).
        """
        assert self.decode_target in ("cpu", "hybrid"), (
            "set_cpu_executor requires decode_target in {'cpu','hybrid'}"
        )
        self.cpu_executor = executor

    def is_cpu_layer(self, layer_id: int) -> bool:
        """Whether ``layer_id`` decodes on the CPU executor (vs the GPU offload path)."""
        return layer_id in self.cpu_layer_ids

    def is_unpinned_layer(self, layer_id: int) -> bool:
        """Whether ``layer_id``'s host banks have no device address (LOCKED/PAGEABLE): the GPU slot-gather paths cannot serve it.
        ``copy_missing`` takes the whole-layer pageable branch, which presumes materialize's position == expert id (never ``ensure_experts``'s LRU slot remap)."""
        return layer_id in self._unpinned_layers

    def alphas_for_slots(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Per-slot global scales for a decode call, or ``None`` when the format
        keeps no GPU-resident alphas (bf16 / triton-nvfp4). Slots of other layers
        yield garbage values, but only slots routed to -- and those belong to
        ``layer_id`` -- are ever read by the grouped GEMM."""
        if self.gate_up_alpha is None:
            return None
        idx = layer_id * self.num_experts + (
            self.id_of_slot.clamp(min=0).long() % self.num_experts
        )
        return self.gate_up_alpha[idx], self.down_alpha[idx]

    def alphas_for_layer(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Global scales for a full-layer prefill (overlap or materialize), where
        position == expert id (contiguous slices, no gather); ``None`` when the
        format keeps no GPU-resident alphas."""
        if self.gate_up_alpha is None:
            return None
        lo = layer_id * self.num_experts
        hi = lo + self.num_experts
        return self.gate_up_alpha[lo:hi], self.down_alpha[lo:hi]

    def bank_views(self, n: int | None = None) -> tuple[torch.Tensor, ...]:
        """Per-bank cache views in registration order: the full ``[S]`` slot cache
        (decode), or its first ``n`` slots (materialized layer)."""
        assert self.banks, "set_bank_sources must register the banks first"
        if n is None:
            return tuple(cache for _, cache in self.banks)
        return tuple(cache[:n] for _, cache in self.banks)

    def _init_prefill_overlap_buffers(self) -> None:
        assert self.banks, "set_bank_sources must register the banks first"
        self._prefill_buffer_layer = [None, None]
        self._prefill_buffer_released = [True, True]
        self._prefill_buffer_has_release_event = [False, False]
        # The double buffers borrow the slot cache's first 2 * num_experts slots
        # (one full expert layer per buffer), one view per registered bank.
        self.prefill_bank_buffers = [
            cache[: 2 * self.num_experts].view(2, self.num_experts, *cache.shape[1:])
            for _, cache in self.banks
        ]
        if self.device.type == "cuda":
            self.prefill_copy_stream = torch.cuda.Stream(device=self.device)
            self.prefill_ready_events = [torch.cuda.Event() for _ in range(2)]
            self.prefill_release_events = [torch.cuda.Event() for _ in range(2)]
            self.prefill_begin_event = torch.cuda.Event()
        if self.prefill_hit_d2d and self.device.type == "cuda":
            self._prefill_slot_snapshot = torch.empty(
                (self.num_layers, self.num_experts), dtype=torch.int32, pin_memory=True
            )
            self._prefill_snapshot_np = self._prefill_slot_snapshot.numpy()
            self._prefill_hit_dst = torch.empty(
                (self.num_experts,), dtype=torch.int32, device=self.device
            )
            self._prefill_hit_src = torch.empty(
                (self.num_experts,), dtype=torch.int32, device=self.device
            )
            self._prefill_hit_num = torch.zeros((1,), dtype=torch.int64, device=self.device)

    def _invalidate_prefill_buffer(self, buffer_id: int) -> None:
        slot_start = buffer_id * self.num_experts
        slot_end = slot_start + self.num_experts
        old_ids = self.id_of_slot[slot_start:slot_end]
        self.slot_for_id.view(-1)[old_ids[old_ids >= 0].long()] = -1
        old_ids.fill_(-1)
        # usage=0 makes these slots the oldest, so the argmin(usage) victim selection in
        # ensure_experts evicts them first.
        self.usage[slot_start:slot_end].zero_()

    def begin_prefill(self) -> None:
        if not self.prefill_overlap:
            return
        self._prefill_buffer_layer = [None, None]
        self._prefill_buffer_released = [True, True]
        if self.prefill_copy_stream is not None:
            # Fence this prefill's copy-stream work behind everything already enqueued
            # on the compute stream. The release/ready events only order against the
            # *previous prefill*; under overlap scheduling a new prefill can be enqueued
            # while the preceding decode batch is still running, and that decode may
            # have loaded experts into the slots the buffers borrow -- without this
            # fence the first prefetch would stomp bytes a running GEMM is reading.
            self.prefill_begin_event.record(torch.cuda.current_stream(self.device))
            self.prefill_copy_stream.wait_event(self.prefill_begin_event)
        self._prefill_hit_d2d_active = self.prefill_hit_d2d and self._hit_d2d_usable()
        if self._prefill_hit_d2d_active:
            # The copy stream is fenced behind the previous decode, so the snapshot
            # observes its final slot map; one host sync per chunk, then per-layer
            # classification is pure host math.
            with torch.cuda.stream(self.prefill_copy_stream):
                self._prefill_slot_snapshot.copy_(self.slot_for_id, non_blocking=True)
            self.prefill_copy_stream.synchronize()

    def prefetch_prefill_layer(self, layer_id: int) -> None:
        if not self.prefill_overlap or layer_id >= self.num_layers:
            return
        if layer_id < 0:
            raise ValueError(f"Invalid prefill layer id: {layer_id}")

        assert self.banks and self.prefill_bank_buffers

        buffer_id = layer_id % 2
        if self._prefill_buffer_layer[buffer_id] == layer_id:
            return
        if self._prefill_buffer_layer[buffer_id] is not None:
            assert self._prefill_buffer_released[buffer_id], (
                "Prefill overlap buffer is being reused before release"
            )

        def copy() -> None:
            self._invalidate_prefill_buffer(buffer_id)
            if self._tier3_stager is not None:
                self._tier3_prefill_fill(layer_id, buffer_id)
                return
            for src, buffer in zip(self._prefill_sources(layer_id), self.prefill_bank_buffers):
                buffer[buffer_id].copy_(src, non_blocking=True)

        if self._prefill_hit_d2d_active:
            self._prefetch_split(layer_id, buffer_id)
        elif self.prefill_copy_stream is None:
            copy()
        else:
            with torch.cuda.stream(self.prefill_copy_stream):
                if self._prefill_buffer_has_release_event[buffer_id]:
                    self.prefill_copy_stream.wait_event(self.prefill_release_events[buffer_id])
                copy()
                self.prefill_ready_events[buffer_id].record(self.prefill_copy_stream)

        self._prefill_buffer_layer[buffer_id] = layer_id
        self._prefill_buffer_released[buffer_id] = False

    def _tier3_invalidate(self) -> None:
        """Drop tier-3 state that records the addresses of buffers a rebuild is about to free.

        ``_tier3_buffer_ptrs`` and ``_materialize_dst_ptrs`` cache DEVICE ADDRESSES of the
        prefill overlap buffers and the slot caches; rebuild reallocates both, so keeping them
        hands the gather kernel dangling pointers -- an illegal memory access surfacing at
        whatever CUDA call comes next, which is what took a live server down after an elastic-KV
        grow. Any deferred prefill gather is dropped for the same reason: its destination is
        gone.
        """
        if self._tier3_stager is None:
            return
        self._tier3_buf_ptrs = None
        self._tier3_mat_ptrs = None
        self._tier3_pending_fill.clear()
        self._tier3_stager.discard_inflight()

    def refresh_host_map(self) -> None:
        """Publish the residency map to the device mirror the ensure kernel reads.

        Once per STEP, not once per layer: a per-layer refresh meant 48 extra 2 KiB stream
        operations, and it bought nothing in staleness terms. A layer's residency only changes
        in its own resolve(), so a whole-map copy at the top of a step shows every layer
        exactly the state its previous step left -- which is what the per-layer version did too.
        """
        if self.host_slot_dev is not None:
            self.host_slot_dev.copy_(self.host_slot_for_id, non_blocking=True)

    def _tier3_finish_fill(self, layer_id: int) -> None:
        """Join a deferred prefill read and gather the disk half into its buffer."""
        buffer_id = self._tier3_pending_fill.pop(layer_id, None)
        if buffer_id is None:
            return
        from freetoken.kernel.fast_index_copy import fast_index_copy_multi_jit

        stager = self._tier3_stager
        _, n_miss = stager.join(layer_id)
        ix = stager.idx(layer_id)
        stream = self.prefill_copy_stream
        ctx = torch.cuda.stream(stream) if stream is not None else _nullcontext()
        with ctx:
            if n_miss:
                fast_index_copy_multi_jit(
                    self._tier3_buffer_ptrs(buffer_id), stager.stage_ptrs(layer_id),
                    self._copy_feat_bytes, ix["miss_dst"][:n_miss],
                    ix["miss_src"][:n_miss], ix["miss_num"],
                )
            stager.mark_gathered(layer_id, stream)
            if stream is not None and self.prefill_ready_events:
                self.prefill_ready_events[buffer_id].record(stream)

    def _tier3_prefill_fill(self, layer_id: int, buffer_id: int) -> None:
        """Fill one prefill buffer under tier 3, without touching the CPU for resident experts.

        Two gathers, both ``fast_index_copy_multi_jit``:

        * pool-resident experts move host pool row -> buffer row at their expert id, one launch
          straight over PCIe. This is the bulk (~70% of the layer) and it never enters host
          compute -- the earlier staging version copied it on the CPU and cost ~30 s a chunk,
          because torch's intra-op pool is pinned to one thread here (the cores belong to the
          MoE workers).
        * the rest are read from NVMe into a compact staging bank and scattered from there.
        """
        from freetoken.kernel.fast_index_copy import fast_index_copy_multi_jit

        stager = self._tier3_stager
        # Start the disk reads and return: they are joined in wait_prefill_layer, one layer
        # later, so they overlap the previous layer's GEMMs instead of blocking on them.
        n_res, n_miss = stager.begin(layer_id)
        ix = stager.idx(layer_id)
        if n_res:
            fast_index_copy_multi_jit(
                self._tier3_buffer_ptrs(buffer_id), self._copy_src_ptrs[layer_id],
                self._copy_feat_bytes, ix["res_dst"][:n_res],
                ix["res_src"][:n_res], ix["res_num"],
            )
        self._tier3_pending_fill[layer_id] = buffer_id

    def _materialize_dst_ptrs(self) -> torch.Tensor:
        """Device table of the slot caches' base addresses (materialize_layer writes slots
        0..num_experts-1, so expert id == destination row)."""
        t = getattr(self, "_tier3_mat_ptrs", None)
        if t is None:
            t = self._tier3_mat_ptrs = self._copy_dst_ptrs
        return t

    def _tier3_gather_into(self, layer_id: int, dst_ptrs: torch.Tensor) -> None:
        """Synchronous whole-layer fill (materialize_layer; no double buffer to overlap with)."""
        from freetoken.kernel.fast_index_copy import fast_index_copy_multi_jit

        stager = self._tier3_stager
        n_res, n_miss = stager.plan(layer_id)
        ix = stager.idx(layer_id)
        if n_res:
            fast_index_copy_multi_jit(
                dst_ptrs, self._copy_src_ptrs[layer_id], self._copy_feat_bytes,
                ix["res_dst"][:n_res], ix["res_src"][:n_res], ix["res_num"],
            )
        if n_miss:
            stager.read_missing(layer_id)
            fast_index_copy_multi_jit(
                dst_ptrs, stager.stage_ptrs(layer_id), self._copy_feat_bytes,
                ix["miss_dst"][:n_miss], ix["miss_src"][:n_miss], ix["miss_num"],
            )

    def _tier3_buffer_ptrs(self, buffer_id: int) -> torch.Tensor:
        """Device table of this prefill buffer's per-bank base addresses (built once)."""
        cache = getattr(self, "_tier3_buf_ptrs", None)
        if cache is None:
            cache = self._tier3_buf_ptrs = {}   # reset by _tier3_invalidate on rebuild
        t = cache.get(buffer_id)
        if t is None:
            t = cache[buffer_id] = torch.tensor(
                [b[buffer_id].data_ptr() for b in self.prefill_bank_buffers],
                dtype=torch.int64, device=self.device)
        return t

    def _prefill_sources(self, layer_id: int) -> list[torch.Tensor]:
        """Per-bank whole-layer sources for a prefill copy, in bank registration order.

        Tier 3 does not come through here: its pool is host-slot indexed, so it fills the
        buffer with two index gathers instead (see _tier3_prefill_fill).
        """
        return [per_layer[layer_id] for per_layer, _ in self.banks]

    def _hit_d2d_usable(self) -> bool:
        """Whether the hit-D2D split can serve this prefill; logs the first fallback.

        The flag is an auto-fallback optional: any unusable condition must degrade
        to the legacy full-layer copy AND say so once in the server log, so a
        configuration that silently runs the legacy path is visible.
        """
        from freetoken.kernel.fast_index_copy import _skip_fast_index_copy_enabled

        if self._tier3_stager is not None:
            reason = ("the NVMe tier is active: the hit gather addresses host banks by expert "
                      "id, but the tier-3 pool is host-slot indexed")
        elif self._prefill_slot_snapshot is None or self.prefill_copy_stream is None:
            reason = "prefill overlap buffers are not initialized for this device"
        elif _skip_fast_index_copy_enabled():
            reason = "FREETOKEN_SKIP_FAST_INDEX_COPY is set (the hit gather would be a no-op)"
        elif not self._copy_fused_ok:
            reason = "the fused copy plan is unavailable (bank alignment or FREETOKEN_FUSED_COPY=0)"
        elif self.cache_size <= 2 * self.num_experts:
            reason = (
                f"cache_size {self.cache_size} leaves no hit region "
                f"(needs > {2 * self.num_experts} slots)"
            )
        elif not self._resolve_batch_memcpy():
            reason = "cudaMemcpyBatchAsync is unavailable"  # resolve logged the specifics
        else:
            return True
        if not self._hit_d2d_fallback_logged:
            logger.warning(
                f"MoE prefill hit-D2D requested but unavailable ({reason}); "
                "falling back to full-layer copies"
            )
            self._hit_d2d_fallback_logged = True
        return False

    def _resolve_batch_memcpy(self) -> bool:
        if self._batch_memcpy is None:
            try:
                from freetoken.kernel.batch_memcpy import load_batch_memcpy

                self._batch_memcpy = load_batch_memcpy()
            except Exception as exc:  # noqa: BLE001 -- any build/runtime gap => legacy path
                logger.warning(f"MoE prefill hit-D2D disabled ({exc}); using full-layer copies")
                self._batch_memcpy = False
        return self._batch_memcpy is not False

    def _prefetch_split(self, layer_id: int, buffer_id: int) -> None:
        """Hit/miss-split prefetch of one expert layer into the double buffer.

        Resident experts are gathered cache -> buffer on the CURRENT stream, fully
        device-side: a one-launch compaction reads the LIVE slot_for_id row into
        fixed-shape gather indices (no host round trip), then fast_index_copy_multi
        moves the rows. Serializing the gather before this layer's GEMMs costs its
        plain duration instead of nondeterministic SM contention. Misses cross
        PCIe as ONE cudaMemcpyBatchAsync of coalesced expert-id runs on the copy
        stream, under the existing release/ready event discipline; its host-built
        run list comes from the begin-of-chunk snapshot because the batch API
        takes HOST pointer arrays. Live-vs-snapshot cannot disagree: the only
        chunk-internal writer (buffer invalidation) rewrites slots already below
        the 2E threshold, and slots < 2E (including -1) are misses on both sides
        -- the buffers own those slots, so their bytes are volatile within the
        chunk. Hit and miss row sets are disjoint, so the streams need no
        ordering against each other.
        """
        import numpy as np

        from freetoken.kernel.fast_index_copy import fast_index_copy_multi_jit
        from freetoken.moe.offload_kernels import prefill_hit_compact

        E = self.num_experts
        snap = self._prefill_snapshot_np[layer_id]
        hit_mask = snap >= 2 * E
        self.prefill_hit_rows += int(hit_mask.sum())
        self.prefill_total_rows += E
        if self._gather_dst_ptrs is not None:
            prefill_hit_compact(self, layer_id, buffer_id)
            # blocks_per_bank=64 vs the PCIe-tuned default of 8: HBM D2D needs the
            # wider grid (~22 GB/s per 1024-thread block on H100).
            fast_index_copy_multi_jit(
                self._gather_dst_ptrs,
                self._gather_dst_ptrs,
                self._gather_feat_bytes,
                self._prefill_hit_dst,
                self._prefill_hit_src,
                self._prefill_hit_num,
                blocks_per_bank=64,
            )
        miss = np.nonzero(~hit_mask)[0]
        with torch.cuda.stream(self.prefill_copy_stream):
            if self._prefill_buffer_has_release_event[buffer_id]:
                self.prefill_copy_stream.wait_event(self.prefill_release_events[buffer_id])
            self._invalidate_prefill_buffer(buffer_id)
            if miss.size:
                run_starts = np.concatenate(([0], np.nonzero(np.diff(miss) != 1)[0] + 1))
                starts = miss[run_starts]
                lengths = np.diff(np.concatenate((run_starts, [miss.size])))
            dst, src, nbytes = [], [], []
            for b, feat in enumerate(self._copy_feat_bytes_host):
                if feat < _SMALL_BANK_FEAT_BYTES:
                    # Whole layer as one entry, EVEN with zero misses: it keeps every
                    # batch entry above the driver's async floor and covers the hit
                    # rows the gather skips for these banks.
                    dst.append(self._copy_dst_ptrs_host[b] + buffer_id * E * feat)
                    src.append(self._copy_src_ptrs_host[layer_id][b])
                    nbytes.append(E * feat)
                elif miss.size:
                    dst.extend(self._copy_dst_ptrs_host[b] + (buffer_id * E + starts) * feat)
                    src.extend(self._copy_src_ptrs_host[layer_id][b] + starts * feat)
                    nbytes.extend(lengths * feat)
            if dst:
                self._batch_memcpy(
                    torch.tensor(dst, dtype=torch.int64),
                    torch.tensor(src, dtype=torch.int64),
                    torch.tensor(nbytes, dtype=torch.int64),
                    torch.cuda.current_stream(self.device).cuda_stream,
                )
            self.prefill_ready_events[buffer_id].record(self.prefill_copy_stream)

    def wait_prefill_layer(self, layer_id: int) -> tuple[torch.Tensor, ...]:
        """Full-layer ``[num_experts, ...]`` bank views for ``layer_id``, one per
        registered bank in registration order: bf16 ``(gate_up, down)``; nvfp4
        marlin/b12x ``(gate_up_packed, gate_up_scale, down_packed, down_scale)``;
        nvfp4 native adds the two global banks after each scale bank."""
        assert self.prefill_overlap
        assert self.prefill_bank_buffers
        self.prefetch_prefill_layer(layer_id)
        if self._tier3_stager is not None:
            self._tier3_finish_fill(layer_id)
        buffer_id = layer_id % 2
        assert self._prefill_buffer_layer[buffer_id] == layer_id
        if self.prefill_ready_events:
            torch.cuda.current_stream(self.device).wait_event(self.prefill_ready_events[buffer_id])
        return tuple(buffer[buffer_id] for buffer in self.prefill_bank_buffers)

    def release_prefill_layer(self, layer_id: int) -> None:
        if not self.prefill_overlap:
            return
        buffer_id = layer_id % 2
        if self._prefill_buffer_layer[buffer_id] != layer_id:
            return
        if self.prefill_release_events:
            self.prefill_release_events[buffer_id].record(torch.cuda.current_stream(self.device))
            self._prefill_buffer_has_release_event[buffer_id] = True
        self._prefill_buffer_released[buffer_id] = True

    def ensure_experts(self, layer_id: int, expert_ids: torch.Tensor) -> None:
        from freetoken.moe.offload_kernels import ensure_experts

        if self.collect_decode_freq:
            # ``expert_ids`` still holds raw expert ids here (the kernel rewrites them to
            # slot ids in place), so snapshot the routing histogram before that happens.
            ids = expert_ids.reshape(-1).long()
            self.decode_freq[layer_id].scatter_add_(0, ids, torch.ones_like(ids))
        self._pending_src_layer = layer_id
        self._pending_whole_layer = False
        ensure_experts(self, layer_id, expert_ids)

    def ensure_experts_hybrid(
        self, layer_id: int, expert_ids: torch.Tensor, split: bool = False, dedup: bool = False,
        weights: torch.Tensor | None = None,
    ):
        """Capped-fetch LRU for the hybrid backend.

        Like :meth:`ensure_experts` but assigns slots to (and schedules copies for) at
        most ``hybrid_max_fetch`` -- or ``~hybrid_fetch_fraction * misses`` when the
        fraction is set -- of this step's missing experts; the overflow misses are
        left non-resident and ``expert_ids`` is rewritten to their cache slot (hit or
        freshly fetched) or ``-1`` (overflow -> compute on the CPU). ``num_indices`` holds
        the capped fetch count (for ``copy_missing``); ``num_missing_full`` the pre-cap
        miss count (for stats). All device-side / fixed-shape, so it is CUDA-graph safe.

        Returns the CPU route list (same shape as ``expert_ids``: the raw expert id
        where ``expert_ids`` is now ``-1``, else ``-1``), produced by the same launch; the
        hybrid miss statistics are accumulated in-kernel when ``collect_stats`` is set.
        With ``split`` and/or ``dedup`` a :class:`HybridRoutes` instead: the GPU routes by
        residency (``hit_ids`` = slot if resident before this step, ``fetched_ids`` = slot if
        assigned this step and written by the pending ``copy_missing``, -1 elsewhere) and
        the expert-dedup tables (``uniq_slots`` / ``uniq_hit`` / ``uniq_fetched`` [BLOCK_A],
        ``pairs`` [BLOCK_A, 4] route indices per distinct expert, -1 padded)."""
        from freetoken.moe.offload_kernels import ensure_experts_hybrid

        if self.collect_decode_freq:
            ids = expert_ids.reshape(-1).long()
            self.decode_freq[layer_id].scatter_add_(0, ids, torch.ones_like(ids))
        self._pending_src_layer = layer_id
        self._pending_whole_layer = False
        n = expert_ids.numel()
        assert n <= ROUTING_PACK_COLS, (n, ROUTING_PACK_COLS)
        cpu_ids = self.cpu_ids[:n].view(expert_ids.shape)
        cpu_w = self.cpu_w[:n].view(expert_ids.shape) if weights is not None else None
        hit_ids = self.hit_ids[:n].view(expert_ids.shape) if split else None
        fetched_ids = self.fetched_ids[:n].view(expert_ids.shape) if split else None
        tables = None
        if dedup:
            import triton

            from freetoken.moe.offload_kernels import DEDUP_MAX_TOKENS

            block_a = max(16, triton.next_power_of_2(n))
            tables = (
                self.uniq_slots[:block_a],
                self.uniq_hit[:block_a],
                self.uniq_fetched[:block_a],
                self.uniq_pairs[: block_a * DEDUP_MAX_TOKENS].view(block_a, DEDUP_MAX_TOKENS),
            )
        ensure_experts_hybrid(
            self, layer_id, expert_ids, self.hybrid_max_fetch, self.hybrid_fetch_fraction,
            cpu_ids=cpu_ids, hit_ids=hit_ids, fetched_ids=fetched_ids, dedup=tables,
            weights=weights, cpu_w=cpu_w,
        )
        if split or dedup:
            return HybridRoutes(
                cpu_ids, hit_ids, fetched_ids,
                *(tables if tables is not None else (None, None, None, None)),
            )
        return cpu_ids

    # ----- demand copy on a side stream (hybrid decode overlap) -----------------------

    def _demand_side_stream(self):
        """High-priority side stream + fork/join events for the demand gather, created once
        (eager warmup precedes graph capture; creating them under capture is not allowed)."""
        s = getattr(self, "_demand_stream", None)
        if s is None:
            self._demand_stream = s = torch.cuda.Stream(self.device, priority=-1)
            self._demand_ev_fork = torch.cuda.Event()
            self._demand_ev_copy = torch.cuda.Event()
        return s

    def copy_missing_async(self) -> None:
        """``copy_missing`` on the demand side stream, forked from the current stream after
        the ensure kernel wrote the plan; ``join_demand_copy`` waits for it. Captured into
        the decode graph as ordinary event edges (the layer-lookahead pattern)."""
        side = self._demand_side_stream()
        cur = torch.cuda.current_stream(self.device)
        self._demand_ev_fork.record(cur)
        side.wait_event(self._demand_ev_fork)
        with torch.cuda.stream(side):
            self.copy_missing()
        self._demand_ev_copy.record(side)

    def join_demand_copy(self) -> None:
        torch.cuda.current_stream(self.device).wait_event(self._demand_ev_copy)

    def materialize_layer(self, layer_id: int) -> None:
        """Bring a whole layer into the first ``num_experts`` slots (prefill, no overlap).

        Under tier 3 the fused whole-layer copy cannot be used: it addresses host rows 0..E-1,
        but the pool holds only ``host_slots`` of them. Stage the layer and copy from there --
        the slot maps are rewritten to the identity the same way either path.
        """
        from freetoken.moe.offload_kernels import materialize_layer

        self._pending_src_layer = layer_id
        self._pending_whole_layer = True
        materialize_layer(self, layer_id)
        if self._tier3_stager is not None:
            # The kernel staged a plan addressing host rows 0..E-1, which the pool does not
            # have; replace it with the two tier-3 gathers straight into the slot cache.
            self.num_indices.zero_()
            self._tier3_gather_into(layer_id, self._materialize_dst_ptrs())

    def reset(self) -> None:
        from freetoken.moe.offload_kernels import reset_cache

        reset_cache(self)
        # Per-expert recency is not cache_size-shaped, so reset_cache leaves it alone; wipe
        # it here so a new sequence starts with cold hybrid fetch priorities.
        self.expert_recency.fill_(-1)
        self.hybrid_admission_credit.zero_()
        if self.num_indices2 is not None:
            self.num_indices2.zero_()

    def reset_stats(self) -> None:
        self.prefill_hit_rows = 0
        self.prefill_total_rows = 0
        self.lru_stats.zero_()
        self.stat_missing.zero_()
        self.stat_active.zero_()
        self.stat_calls.zero_()
        self.stat_fetched.zero_()
        self.stat_missing_layer.zero_()
        self.stat_active_layer.zero_()
        self.stat_fetched_layer.zero_()
        self.stat_steps_layer.zero_()

    @property
    def prefetch_reserve_lo(self) -> int:
        """Lowest slot a prefetch admission may claim: the prefill double buffer owns
        slots [0, 2E) when prefill_overlap is on (their usage is deliberately zeroed by
        _invalidate_prefill_buffer, which would otherwise make them argmin favorites)."""
        return 2 * self.num_experts if self.prefill_overlap else 0

    def init_prefetch_plan(self, max_prefetch: int, ring: int = 1) -> None:
        """Allocate the speculative-prefetch copy plans ("plan-2") and counters.

        Called once at engine init when --moe-prefetch is enabled, after
        ``set_bank_sources`` (the layered copy needs the fused descriptor's source
        pointers). ``ring`` > 1 gives the in-graph lookahead one independent
        (dst/src/count) plan per in-flight layer; the eager variant uses ring 1.
        Buffers are budget-shaped (not cache_size-shaped), so ``rebuild`` keeps them
        and only zeroes the counts."""
        assert self.bank_sources, "set_bank_sources must run before init_prefetch_plan"
        if self.device.type == "cuda":  # CPU-device tests exercise the admission logic only
            assert self._copy_fused_ok and self._src_ptr_table is not None, (
                "expert prefetch requires the fused copy plan (16B-aligned banks, "
                "FREETOKEN_FUSED_COPY on)"
            )
        assert max_prefetch > 0 and ring > 0
        self.max_prefetch = int(max_prefetch)
        self.prefetch_ring = int(ring)
        n = max_prefetch * ring
        self.evict_slots2 = torch.zeros((n,), dtype=torch.int32, device=self.device)
        self.src_indices2 = torch.full((n,), -1, dtype=torch.int32, device=self.device)
        self.num_indices2 = torch.zeros((ring,), dtype=torch.int64, device=self.device)
        self.pf_stamp = torch.zeros((ring,), dtype=torch.int64, device=self.device)
        self.pf_issued = torch.zeros((1,), dtype=torch.int64, device=self.device)
        self.pf_used = torch.zeros((1,), dtype=torch.int64, device=self.device)
        self.pf_evicted_unused = torch.zeros((1,), dtype=torch.int64, device=self.device)
        self.pf_steps = torch.zeros((1,), dtype=torch.int64, device=self.device)

    def prefetch_admit(self, cand: torch.Tensor, budget: int, plan: int = 0) -> None:
        """Admit up to ``budget`` predicted experts into ring slot ``plan`` (see
        offload_kernels.prefetch_admit)."""
        from freetoken.moe.offload_kernels import prefetch_admit

        prefetch_admit(self, cand, min(int(budget), self.max_prefetch), plan)

    def prefetch_copy(self, plan: int = 0) -> None:
        """Launch ring slot ``plan``'s H2D gather on the CURRENT stream (the caller
        picks the prefetch side stream and fences with events). One multi-bank launch
        over flat expert ids; device-side count, so an empty plan costs one no-op
        launch. Fixed tensor slices per ``plan`` -> CUDA-graph capturable."""
        from freetoken.kernel.fast_index_copy import fast_index_copy_multi_layered_jit

        assert self.num_indices2 is not None, "init_prefetch_plan must run first"
        b = self.max_prefetch
        fast_index_copy_multi_layered_jit(
            self._copy_dst_ptrs,
            self._src_ptr_table,
            self._copy_feat_bytes,
            self.evict_slots2[plan * b : (plan + 1) * b],
            self.src_indices2[plan * b : (plan + 1) * b],
            self.num_indices2[plan : plan + 1],
            num_experts=self.num_experts,
        )

    def prefetch_used_check(self) -> None:
        """Score the previous step's prefetch plan (ring slot 0; the eager variant's),
        device-side, no host sync.

        A plan row was USED if its slot still maps to the prefetched expert and a later
        demand call bumped the slot's usage past the admission stamp; EVICTED-UNUSED if
        the slot was re-assigned before any use. Rows neither used nor evicted yet (a
        prediction one step early) count in neither -- pf_used is a lower bound. Must run
        before the next admission rewrites the plan."""
        assert self.num_indices2 is not None
        b = self.max_prefetch
        n = self.num_indices2[0:1]
        idx = torch.arange(b, device=self.device)
        valid = idx < n
        evict = self.evict_slots2[:b]
        src = self.src_indices2[:b]
        slots = evict.long()
        cur = self.id_of_slot[slots]
        if self.host_tier is not None:
            # src2 holds the copy-source row (a HOST SLOT under tier 3) while id_of_slot keeps
            # the flat expert id, so the comparison needs the same translation the admission
            # made. Reading today's map means a row whose host slot was reassigned since
            # admission scores as unused -- conservative, and pf_used is a lower bound anyway.
            hmap = self.host_slot_for_id.view(-1).to(cur.device)
            hrow = hmap[cur.clamp(min=0)]
            cur = (cur // self.num_experts) * self.num_experts + hrow
        used = valid & (cur == src) & (self.usage[slots] > self.pf_stamp[0:1])
        self.pf_used += used.sum()
        self.pf_evicted_unused += (valid & (cur != src)).sum()
        self.pf_steps += 1

    def record_decode_stats(self, layer_id: int) -> None:
        """No-op: ``ensure_experts`` accumulates into ``lru_stats`` inside its own launch.

        Kept so the hybrid and non-hybrid call sites stay symmetric. The previous version
        was eight torch ops per layer per step, all captured into the decode graph.
        """

    def record_decode_stats_hybrid(self, layer_id: int) -> None:
        """No-op: ``ensure_experts_hybrid`` accumulates the hybrid stats (full miss count
        pre-cap, PCIe-fetched count, active count; scalar + per-layer) inside its own
        launch when ``collect_stats`` is set. Kept for call-site symmetry; the previous
        version was eleven torch ops per layer per step captured into the decode graph."""
        assert 0 <= layer_id < self.num_layers, f"layer_id {layer_id} out of range [0, {self.num_layers})"

    def decode_miss_stats(self) -> dict:
        if self.decode_target == "hybrid":
            active = int(self.stat_active.item())
            missing = int(self.stat_missing.item())
            calls = int(self.stat_calls.item())
        else:
            active, missing, calls = (int(x) for x in self.lru_stats.sum(0))
        fetched = int(self.stat_fetched.item())
        out = {
            "layer_calls": calls,
            "active_per_layer": (active / calls) if calls else 0.0,
            "missing_per_layer": (missing / calls) if calls else 0.0,
            "miss_rate": (missing / active) if active else 0.0,
            # hybrid: how the misses split between PCIe fetch (GPU) and CPU compute.
            "fetched_per_layer": (fetched / calls) if calls else 0.0,
            "cpu_per_layer": ((missing - fetched) / calls) if calls else 0.0,
            "fetch_rate": (fetched / missing) if missing else 0.0,
            # prefill hit-D2D split: expert rows served from the cache (D2D) vs all
            # rows prefetched into the double buffer since the last reset.
            "prefill_hit_rows": self.prefill_hit_rows,
            "prefill_rows": self.prefill_total_rows,
        }
        if self.num_indices2 is not None:
            issued = int(self.pf_issued.item())
            used = int(self.pf_used.item())
            steps = int(self.pf_steps.item())
            out.update(
                prefetch_issued=issued,
                prefetch_used=used,
                prefetch_evicted_unused=int(self.pf_evicted_unused.item()),
                prefetch_use_rate=(used / issued) if issued else 0.0,
                prefetch_issued_per_step=(issued / steps) if steps else 0.0,
            )
        return out

    def decode_miss_stats_per_layer(self) -> dict:
        """Per-MoE-layer realized decode stats for one (reset_stats-delimited) window.

        Requires ``collect_stats`` and the call sites passing ``layer_id``. Returns python
        lists indexed by MoE-layer id: missing/active experts per step and the realized
        miss_rate (missing/active) -- i.e. how cacheable each layer's routing actually was
        under the running LRU. Reads device tensors once (no per-step host sync)."""
        if self.decode_target == "hybrid":
            steps = self.stat_steps_layer.tolist()
            missing = self.stat_missing_layer.tolist()
            active = self.stat_active_layer.tolist()
        else:
            cols = self.lru_stats.t().tolist()
            active, missing, steps = cols[Stat.ACTIVE], cols[Stat.MISS], cols[Stat.CALLS]
        fetched = self.stat_fetched_layer.tolist()
        per_layer = []
        for L in range(self.num_layers):
            s, m, a, f = steps[L], missing[L], active[L], fetched[L]
            per_layer.append({
                "layer": L,
                "steps": s,
                "active_per_step": (a / s) if s else 0.0,
                "missing_per_step": (m / s) if s else 0.0,
                "miss_rate": (m / a) if a else 0.0,
                "fetched_per_step": (f / s) if s else 0.0,
            })
        return {"per_layer": per_layer}

    def decode_routing_stats(self) -> dict:
        """Per-layer decode routing concentration, for cache-skew analysis.

        Uses the histogram from ``collect_decode_freq``. The ``oracle_hit`` is the best a
        per-layer LRU holding ``cache_size/num_layers`` slots could achieve on the observed
        (stationary) routing distribution -- i.e. an upper bound on hit rate that depends
        purely on how skewed routing is, independent of any LRU/LFU dynamics.
        """
        freq = self.decode_freq.float()
        total = freq.sum(dim=1)
        valid = total > 0
        if int(valid.sum()) == 0:
            return {}
        slots_per_layer = self.cache_size / self.num_layers
        C = max(1, int(round(slots_per_layer)))
        sorted_f, _ = torch.sort(freq, dim=1, descending=True)
        oracle_hit = (sorted_f[:, :C].sum(dim=1)[valid] / total[valid]).mean().item()
        ws = (freq > 0).sum(dim=1).float()
        cdf = torch.cumsum(sorted_f, dim=1) / total.clamp(min=1).unsqueeze(1)
        cover90 = ((cdf < 0.9).sum(dim=1).float() + 1)[valid]
        p = freq / total.clamp(min=1).unsqueeze(1)
        ent = -(p * p.clamp(min=1e-12).log()).sum(dim=1)[valid]
        norm_ent = (ent / torch.log(torch.tensor(float(self.num_experts)))).mean().item()
        return {
            "slots_per_layer": slots_per_layer,
            "working_set_mean": ws[valid].mean().item(),
            "working_set_max": int(ws[valid].max().item()),
            "experts_for_90pct": cover90.mean().item(),
            "oracle_hit_at_slots": oracle_hit,
            "norm_entropy": norm_ent,
        }

    def copy_missing(self) -> None:
        assert self.banks, "set_bank_sources must register the banks first"
        layer_id = self._pending_src_layer
        assert layer_id is not None, "no staged misses (ensure_experts/materialize_layer first)"
        if layer_id in self._unpinned_layers:
            if not self._pending_whole_layer:
                raise RuntimeError(
                    f"layer {layer_id} is unpinned: its only copy is the whole-layer "
                    f"pageable materialize (position == expert id); ensure_experts's "
                    f"LRU slot remap cannot be honored without a device alias"
                )
            # the only copy a non-pinned layer ever needs is the non-overlap prefill materialize, which schedules the whole layer into slots [0, num_experts) with position == expert id -- a plain synchronous pageable H2D copy
            # never CUDA-graph captured: prefill is not captured, and decode never reaches this branch (it routes to the CPU executor)
            for per_layer, cache in self.banks:
                cache[: self.num_experts].copy_(per_layer[layer_id])
            return
        if self._copy_fused_ok:
            from freetoken.kernel.fast_index_copy import fast_index_copy_multi_jit

            # One launch copies the missing rows for every bank (instead of one launch per
            # bank). evict_slots/src_indices/num_indices are shared across banks;
            # src_indices holds layer-local expert rows, resolved against this layer's
            # source pointers (layer_id is a static int per captured graph node).
            fast_index_copy_multi_jit(
                self._copy_dst_ptrs,
                self._copy_src_ptrs[layer_id],
                self._copy_feat_bytes,
                self.evict_slots,
                self.src_indices,
                self.num_indices,
            )
            return

        from freetoken.kernel import fast_index_copy_jit

        for per_layer, cache in self.banks:
            fast_index_copy_jit(
                cache,
                self.evict_slots,
                per_layer[layer_id],
                self.src_indices,
                self.num_indices,
            )


def iter_offload_moe_layers(model) -> Iterator:
    from freetoken.layers import BaseOP, OffloadMoELayer

    # A model whose MoE blocks are bespoke nn.Modules (not OffloadMoELayer) declares its
    # offload layers explicitly via this hook (e.g. DeepSeek-V4-Flash); attach_offload_moe_cache
    # then sets .offload_cache on each yielded layer just like the OffloadMoELayer walk.
    hook = getattr(model, "_iter_offload_moe_layers", None)
    if hook is not None:
        yield from hook()
        return

    if isinstance(model, OffloadMoELayer):
        yield model

    if not isinstance(model, BaseOP):
        return

    for value in model.__dict__.values():
        if isinstance(value, BaseOP):
            yield from iter_offload_moe_layers(value)
        elif isinstance(value, (list, tuple)):
            for item in value:
                yield from iter_offload_moe_layers(item)


def attach_offload_moe_cache(model, cache: OffloadMoeCache) -> list:
    layers = list(iter_offload_moe_layers(model))
    for layer in layers:
        layer.offload_cache = cache
    return layers
