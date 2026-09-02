from __future__ import annotations

import math

import pytest

from freetoken.tier3.budget import (
    ALIGN, BankSplit, HostTierInfeasible, gpu_miss_rate_estimate, host_bank_split,
    net_host_budget_bytes, plan_host_tier, tier3_required_host_hit, tier3_step_bytes,
    tier3_tok_s_ceiling,
)

# Qwen3.8-Flash-Next NVFP4, H=2560 I=640 -- the real per-expert row bytes per bank.
QWEN_NVFP4 = {
    "gate_up_packed": 1_638_400,
    "gate_up_scale": 204_800,
    "gate_up_global": 2_560,
    "down_packed": 819_200,
    "down_scale": 102_400,
    "down_global": 5_120,
}
QWEN_LAYERS, QWEN_EXPERTS = 48, 512
GIB = 2 ** 30


# ---------------------------------------------------------------- bank split

def test_every_bank_is_paged():
    """All banks are paged, because the fused H2D gather applies one row index across the banks
    of a layer -- a bank left at full width would be read at the fetched expert's host slot and
    return another expert's scales."""
    s = host_bank_split(QWEN_NVFP4)
    assert s.fetched == tuple(QWEN_NVFP4)
    assert s.resident == ()
    assert s.fetched_bytes_per_expert == sum(QWEN_NVFP4.values()) == 2_772_480
    assert s.total_bytes_per_expert == 2_772_480


def test_unaligned_banks_are_flagged_for_the_bounce_path():
    """Aligned rows DMA straight into the pinned slot; the rest need an aligned-window read plus
    a copy, so the reader has to be told which is which."""
    s = host_bank_split(QWEN_NVFP4)
    assert s.unaligned == ("gate_up_global", "down_global")
    assert all(QWEN_NVFP4[n] % ALIGN != 0 for n in s.unaligned)
    assert all(QWEN_NVFP4[n] % ALIGN == 0 for n in set(s.fetched) - set(s.unaligned))
    # they are a rounding error in bytes -- 0.3% of an expert
    frac = sum(QWEN_NVFP4[n] for n in s.unaligned) / s.fetched_bytes_per_expert
    assert frac < 0.005


def test_paging_everything_costs_more_requests_not_more_bytes():
    """The trade this design accepts: 6 reads per fetched expert instead of 2, for the same
    bytes -- and the 7.2 GiB the scales would have occupied goes back into the pool."""
    s = host_bank_split(QWEN_NVFP4)
    assert len(s.fetched) == 6
    packed_only = QWEN_NVFP4["gate_up_packed"] + QWEN_NVFP4["down_packed"]
    assert s.fetched_bytes_per_expert / packed_only < 1.13   # ~12% more bytes
    reclaimed = QWEN_LAYERS * QWEN_EXPERTS * (s.fetched_bytes_per_expert - packed_only)
    assert 7.0 < reclaimed / GIB < 7.5


# ---------------------------------------------------------------- budget

def test_net_budget_subtracts_reserve_and_other_pinned():
    assert net_host_budget_bytes(64 * GIB, 8 * GIB, 4 * GIB) == 52 * GIB


def test_net_budget_never_negative():
    assert net_host_budget_bytes(8 * GIB, 16 * GIB, 4 * GIB) == 0


# ---------------------------------------------------------------- plan

def _plan(budget_gib: float, **kw):
    return plan_host_tier(
        host_budget_bytes=int(budget_gib * GIB), num_layers=QWEN_LAYERS,
        num_experts=QWEN_EXPERTS, split=host_bank_split(QWEN_NVFP4), **kw,
    )


def test_corpus_fits_disables_the_tier():
    """The zero-overhead default: on a big-RAM box nothing changes. This is the regression
    guard for 'the existing serving path is untouched'."""
    plan = _plan(300)
    assert plan.enabled is False
    assert plan.host_slots == QWEN_EXPERTS
    assert plan.residency == 1.0
    assert plan.stage_bytes == 0


def test_64gib_box_plans_the_measured_operating_point():
    # 64 GiB minus an 8 GiB OS/process reserve, with a 4 GiB PLE row cache
    plan = _plan(56, ple_cache_bytes=4 * GIB)
    assert plan.enabled is True
    # Phase A measured 97%+ RAM hit at ~407 slots/layer for this split; land in that region
    assert 380 <= plan.host_slots <= 440, plan.host_slots
    assert 0.72 < plan.residency < 0.88
    assert plan.total_bytes <= 56 * GIB


def test_plan_never_exceeds_its_budget():
    for gib in (20, 30, 40, 50, 56, 60):
        plan = _plan(gib, ple_cache_bytes=4 * GIB)
        assert plan.total_bytes <= gib * GIB, (gib, plan)


def test_ple_cache_trades_directly_against_expert_slots():
    small = _plan(56, ple_cache_bytes=2 * GIB)
    big = _plan(56, ple_cache_bytes=16 * GIB)
    assert big.host_slots < small.host_slots


def test_infeasible_budget_names_the_ram_figure():
    with pytest.raises(HostTierInfeasible, match=r"GiB"):
        _plan(6, min_host_slots=256)


def test_min_host_slots_floor_is_enforced_not_rounded_down():
    plan = _plan(56, ple_cache_bytes=4 * GIB, min_host_slots=128)
    assert plan.host_slots >= 128


# ---------------------------------------------------------------- the law

def _law_kw(host_hit: float):
    return dict(num_layers=QWEN_LAYERS, touches_per_layer=22.54, gpu_miss_rate=0.289,
                host_hit_rate=host_hit, fetched_bytes_per_expert=2_764_800)


def test_step_bytes_matches_the_measured_sweep():
    """The simulator reported 13.6 reads/step = 37.6 MB/step at 95.6% RAM hit; the closed
    form must agree with the replay it is meant to summarise."""
    b = tier3_step_bytes(**_law_kw(0.956))
    assert 30e6 < b < 45e6, b
    assert math.isclose(b / 2_764_800, 13.6, rel_tol=0.10)


def test_step_bytes_is_zero_when_ram_never_misses():
    assert tier3_step_bytes(**_law_kw(1.0)) == 0.0


def test_ceiling_is_infinite_without_disk_traffic():
    assert tier3_tok_s_ceiling(0.0, 5.9e9) == math.inf


def test_ceiling_is_monotonic_in_bandwidth_and_hit_rate():
    slow = tier3_tok_s_ceiling(tier3_step_bytes(**_law_kw(0.956)), 2.58e9)
    fast = tier3_tok_s_ceiling(tier3_step_bytes(**_law_kw(0.956)), 5.90e9)
    assert fast > slow
    worse = tier3_tok_s_ceiling(tier3_step_bytes(**_law_kw(0.50)), 2.58e9)
    assert worse < slow


def test_required_host_hit_inverts_the_ceiling_exactly():
    target = 60.0
    need = tier3_required_host_hit(
        target_tok_s=target, num_layers=QWEN_LAYERS, touches_per_layer=22.54,
        gpu_miss_rate=0.289, fetched_bytes_per_expert=2_764_800,
        nvme_bw_bytes_s=2.58e9, accepted_tokens_per_step=2.4,
    )
    got = tier3_tok_s_ceiling(tier3_step_bytes(**_law_kw(need)), 2.58e9, 2.4)
    assert math.isclose(got, target, rel_tol=1e-6)


def _need(target: float) -> float:
    return tier3_required_host_hit(
        target_tok_s=target, num_layers=QWEN_LAYERS, touches_per_layer=22.54,
        gpu_miss_rate=0.289, fetched_bytes_per_expert=2_764_800, nvme_bw_bytes_s=2.58e9,
    )


def test_required_host_hit_asymptotes_to_one_never_exceeds_it():
    """At most 100% of misses can be absorbed by RAM, so an absurd target drives the
    requirement toward 1.0 rather than past it -- 'unreachable' reads as 'needs ~100%'."""
    assert 0.999 < _need(500_000.0) < 1.0


def test_required_host_hit_is_non_positive_when_the_target_is_easy():
    # slow enough that raw NVMe bandwidth covers every miss with no tier-2 help at all
    assert _need(1.0) <= 0.0


def test_required_host_hit_is_monotonic_in_target():
    assert _need(10.0) < _need(60.0) < _need(200.0)


# ---------------------------------------------------------------- miss curve

def test_miss_curve_reproduces_its_three_calibration_points():
    for slots, want in ((5006, 0.17), (6528, 0.10), (8030, 0.062)):
        assert abs(gpu_miss_rate_estimate(slots, 49) - want) < 0.02, slots


def test_miss_curve_is_decreasing():
    xs = [gpu_miss_rate_estimate(s, 48) for s in (1650, 2500, 3000, 3500, 8030)]
    assert xs == sorted(xs, reverse=True)
