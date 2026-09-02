"""Pure host-RAM budget policy for the NVMe expert tier.

Sibling of :mod:`freetoken.engine.cache_budget`, which splits *VRAM* between the GPU expert
slot cache and the KV pool. This module splits *host RAM* between the tier-2 expert pool, the
always-resident small banks, the PLE row cache and the NVMe staging ring -- and carries the
arithmetic that says whether the resulting geometry can hit a tok/s target at all.

Like its sibling it has no torch/GPU side effects: everything here is integer/float arithmetic
over already-measured quantities, so it is unit-testable without a device.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

# Mirrors offload_cache._SMALL_BANK_FEAT_BYTES. A bank whose per-expert row is smaller than
# this ships as one whole-layer entry in the GPU copy path and is far too small to be worth an
# individual NVMe request.
SMALL_BANK_FEAT_BYTES = 256 * 1024

# O_DIRECT block / page size. A bank row is only readable straight into a pinned host slot
# without a bounce buffer when its byte length is a whole number of blocks (the FTW entry start
# and the HostBank allocation are both already block-aligned).
ALIGN = 4096


@dataclass(frozen=True)
class BankSplit:
    """Which banks tier 3 pages, and which of those need a bounced read.

    **Every** bank is paged. That is forced by the movement path, not chosen: the fused
    multi-bank H2D gather (``fast_index_copy_multi``) applies ONE row index across all banks of
    a layer, so a bank left at full ``num_experts`` width would be read at the *host slot* of
    the expert being fetched and return some other expert's scales. Keeping a bank resident
    would mean giving that kernel a per-bank row table -- a real kernel change, for a
    second-order gain.

    The earlier measurement stands and is why this costs little: paging the scales as well as
    the packed weights leaves traffic flat (21.2 vs 21.0 MB/step) and *improves* residency
    (81.8% vs 79.5%, since the 7.2 GiB of resident scales go back into the pool); it only
    raises the request count, 17.1 -> 30.7 IOs/step.

    ``unaligned`` names the banks whose per-expert row is not a whole 4096-byte block -- NVFP4's
    two tiny ``*_global`` banks (2,560 and 5,120 B). Those cannot be read straight into a
    pinned slot and go through an aligned-window bounce; they are 0.3% of the bytes.
    """

    fetched: tuple[str, ...]
    unaligned: tuple[str, ...]
    fetched_bytes_per_expert: int

    @property
    def resident(self) -> tuple[str, ...]:
        """No bank stays resident; kept so callers reading the old field get an empty tuple
        rather than a stale answer."""
        return ()

    @property
    def resident_bytes_per_expert(self) -> int:
        return 0

    @property
    def total_bytes_per_expert(self) -> int:
        return self.fetched_bytes_per_expert


def host_bank_split(feat_bytes: Mapping[str, int]) -> BankSplit:
    """Classify a quant format's banks for tier 3. All are paged; see :class:`BankSplit`.

    ``feat_bytes`` maps bank name -> bytes for ONE expert row, i.e.
    ``prod(shape[1:]) * itemsize`` of a ``bank_sources[name][layer]`` tensor.
    """
    names = tuple(feat_bytes)  # preserve the schema's registration order
    return BankSplit(
        fetched=names,
        unaligned=tuple(n for n in names if feat_bytes[n] % ALIGN != 0),
        fetched_bytes_per_expert=sum(feat_bytes.values()),
    )


def net_host_budget_bytes(mem_total: int, reserve_bytes: int, pinned_other_bytes: int) -> int:
    """Host bytes the expert tier may spend: RAM minus the OS/process reserve and minus host
    allocations made outside the expert banks (the PLE table today, via
    ``Engine._host_tables_bytes``). Clamped at 0 so a hostile configuration plans an empty
    tier instead of a negative one."""
    return max(0, mem_total - reserve_bytes - pinned_other_bytes)


@dataclass(frozen=True)
class HostTierPlan:
    """The resolved host-RAM geometry. ``enabled`` False means the whole expert corpus fits in
    RAM and the tier is bypassed entirely -- today's code path, byte for byte."""

    enabled: bool
    host_slots: int  # per layer; == num_experts when disabled
    num_layers: int
    num_experts: int
    split: BankSplit
    pool_bytes: int  # the paged tier-2 pool
    resident_bytes: int  # small banks, every expert, always in RAM
    stage_bytes: int  # NVMe -> host staging ring (prefill)
    ple_cache_bytes: int
    reason: str

    @property
    def residency(self) -> float:
        """Fraction of the expert corpus held in RAM (1.0 when the tier is disabled)."""
        return self.host_slots / self.num_experts

    @property
    def total_bytes(self) -> int:
        return self.pool_bytes + self.resident_bytes + self.stage_bytes + self.ple_cache_bytes


class HostTierInfeasible(ValueError):
    """The host budget cannot fund even the minimum viable tier-2 pool."""


def plan_host_tier(
    *,
    host_budget_bytes: int,
    num_layers: int,
    num_experts: int,
    split: BankSplit,
    stage_experts: int = 64,
    ple_cache_bytes: int = 0,
    min_host_slots: int = 0,
) -> HostTierPlan:
    """Resolve the host-RAM budget into a per-layer tier-2 slot count.

    Allocation order, most-constrained first: the PLE row cache and the always-resident small
    banks are fixed costs, the staging ring is a small fixed cost, and whatever remains divides
    evenly across layers to become the paged pool.

    ``min_host_slots`` is a floor the caller derives from the GPU cache size -- the RAM tier
    must strictly dominate the VRAM tier, or every GPU eviction lands on disk. Raises
    :class:`HostTierInfeasible` (with the RAM figure that *would* work) rather than returning a
    geometry that cannot serve.
    """
    assert num_layers > 0 and num_experts > 0
    assert split.fetched_bytes_per_expert > 0, "no NVMe-pageable bank in this schema"

    resident_bytes = num_layers * num_experts * split.resident_bytes_per_expert
    stage_bytes = stage_experts * split.fetched_bytes_per_expert
    fixed = resident_bytes + stage_bytes + ple_cache_bytes
    per_layer_slot_bytes = num_layers * split.fetched_bytes_per_expert

    remaining = host_budget_bytes - fixed
    raw = remaining // per_layer_slot_bytes if remaining > 0 else 0
    host_slots = min(int(raw), num_experts)

    if host_slots >= num_experts:
        # The whole corpus fits: bypass the tier so the serving path is unchanged.
        return HostTierPlan(
            enabled=False, host_slots=num_experts, num_layers=num_layers,
            num_experts=num_experts, split=split,
            pool_bytes=num_layers * num_experts * split.fetched_bytes_per_expert,
            resident_bytes=resident_bytes, stage_bytes=0, ple_cache_bytes=ple_cache_bytes,
            reason="expert corpus fits in host RAM; NVMe tier disabled",
        )

    if host_slots < min_host_slots:
        need = fixed + min_host_slots * per_layer_slot_bytes
        raise HostTierInfeasible(
            f"host tier needs {need / 2**30:.1f} GiB to fund the {min_host_slots}-slot/layer "
            f"floor (fixed costs {fixed / 2**30:.1f} GiB + "
            f"{min_host_slots * per_layer_slot_bytes / 2**30:.1f} GiB of pool), but the budget "
            f"is {host_budget_bytes / 2**30:.1f} GiB -- raise host RAM, lower "
            f"--ple-cache-gb, or shrink the GPU expert cache"
        )

    return HostTierPlan(
        enabled=True, host_slots=host_slots, num_layers=num_layers, num_experts=num_experts,
        split=split, pool_bytes=host_slots * per_layer_slot_bytes,
        resident_bytes=resident_bytes, stage_bytes=stage_bytes, ple_cache_bytes=ple_cache_bytes,
        reason=(f"{host_slots}/{num_experts} experts per layer resident "
                f"({100 * host_slots / num_experts:.1f}%); remainder streams from NVMe"),
    )


# ----------------------------------------------------------------------------------------
# The governing law: how fast can this geometry possibly decode?
# ----------------------------------------------------------------------------------------


def tier3_step_bytes(
    *,
    num_layers: int,
    touches_per_layer: float,
    gpu_miss_rate: float,
    host_hit_rate: float,
    fetched_bytes_per_expert: int,
) -> float:
    """Bytes one decode step must read from NVMe.

    ``touches_per_layer`` is the count of DISTINCT experts a layer routes to in one step
    (~21.2 for Qwen3.8-Flash-Next at batch 1 with a 3-row MTP verify -- the dedup already
    happened). Only GPU misses reach RAM, and only RAM misses reach disk.
    """
    missed = num_layers * touches_per_layer * gpu_miss_rate
    return missed * (1.0 - host_hit_rate) * fetched_bytes_per_expert


def tier3_tok_s_ceiling(
    step_bytes: float, nvme_bw_bytes_s: float, accepted_tokens_per_step: float = 1.0
) -> float:
    """Upper bound on decode tok/s if NVMe bandwidth were the *only* cost.

    A ceiling, not a prediction: it ignores compute, PCIe and the CPU GEMV, all of which
    overlap with the reads. Its job is to fail a hopeless configuration in arithmetic.
    """
    if step_bytes <= 0:
        return math.inf
    return accepted_tokens_per_step * nvme_bw_bytes_s / step_bytes


def tier3_required_host_hit(
    *,
    target_tok_s: float,
    num_layers: int,
    touches_per_layer: float,
    gpu_miss_rate: float,
    fetched_bytes_per_expert: int,
    nvme_bw_bytes_s: float,
    accepted_tokens_per_step: float = 1.0,
) -> float:
    """The tier-2 hit rate needed to reach ``target_tok_s`` -- the law inverted.

    This is what turns "it runs at 2 tok/s" into a number the operator can act on: feed the
    result back through :func:`plan_host_tier` to get the RAM figure that would work.

    The result lies in ``(-inf, 1)``: at most 100% of misses can be absorbed by RAM. A value
    <= 0 means the target holds even with no tier-2 hits at all; a value approaching 1 means it
    is unreachable on this drive at any RAM size, since even a near-perfect tier-2 leaves more
    traffic than the bandwidth budget allows.
    """
    missed = num_layers * touches_per_layer * gpu_miss_rate
    budget_bytes = accepted_tokens_per_step * nvme_bw_bytes_s / max(target_tok_s, 1e-9)
    return 1.0 - budget_bytes / (missed * fetched_bytes_per_expert)


# Empirical GPU miss-rate curve for Qwen3.8-Flash-Next, fitted to three same-config live
# measurements (5,006 slots -> 0.17, 6,528 -> 0.10, 8,030 -> 0.062; residual < 0.003).
# `c` is slots per bank layer. Extrapolating far below the measured range is exactly what the
# 16 GiB profile does, so treat values under ~5,000 slots as an estimate to be replaced by the
# offline simulator, not as ground truth.
_MISS_A = 0.882
_MISS_SCALE = 61.2


def gpu_miss_rate_estimate(slots: int, num_bank_layers: int) -> float:
    """Fitted decode miss rate for a given GPU slot count. See the note above on extrapolation."""
    assert num_bank_layers > 0
    return _MISS_A * math.exp(-(slots / num_bank_layers) / _MISS_SCALE)
