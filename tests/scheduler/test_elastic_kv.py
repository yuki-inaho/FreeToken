"""Elastic KV pool planner (pure arithmetic)."""
from __future__ import annotations

from freetoken.scheduler.elastic_kv import (
    GEN_SLACK_MAX,
    GEN_SLACK_MIN,
    SLOT_MARGIN,
    needed_tokens,
    plan_moe_for_pages,
    plan_target_tokens,
)

BASE, CAP = 65536, 262144


def test_needed_tokens_bounds_the_generation_allowance():
    assert needed_tokens(1000, 10, CAP) == 1000 + GEN_SLACK_MIN
    assert needed_tokens(1000, 1_000_000, CAP) == 1000 + GEN_SLACK_MAX
    assert needed_tokens(260_000, 8192, CAP) == CAP  # never beyond the cap


def test_plan_target_tokens_doubles_from_base_and_caps():
    assert plan_target_tokens(1000, BASE, CAP) == BASE
    assert plan_target_tokens(BASE, BASE, CAP) == BASE
    assert plan_target_tokens(BASE + 1, BASE, CAP) == 2 * BASE
    assert plan_target_tokens(130_000, BASE, CAP) == 2 * BASE      # 131072 covers it
    assert plan_target_tokens(140_000, BASE, CAP) == 4 * BASE == CAP
    assert plan_target_tokens(250_000, BASE, CAP) == CAP
    assert plan_target_tokens(10 ** 9, BASE, CAP) == CAP  # clamped, terminates


def test_plan_moe_for_pages_exchanges_bytes_one_for_one():
    kv_per_page, slot = 64 * 14_336, 2_764_800   # 64-token page of ~14 KB/token; NVFP4 expert
    cur_pages, cur_moe = 1024, 8000
    grow = plan_moe_for_pages(cur_pages, cur_moe, 4096, kv_per_page, slot)
    freed = (4096 - 1024) * kv_per_page
    assert grow == cur_moe - (-(-freed // slot)) - SLOT_MARGIN
    assert (cur_moe - grow - SLOT_MARGIN) * slot >= freed  # the slots given up cover the KV growth
    shrink = plan_moe_for_pages(4096, grow, cur_pages, kv_per_page, slot)
    assert grow < shrink <= cur_moe  # shrinking back reclaims (almost) all of them
    assert plan_moe_for_pages(cur_pages, cur_moe, cur_pages, kv_per_page, slot) == cur_moe - SLOT_MARGIN
    assert plan_moe_for_pages(cur_pages, cur_moe, 4096, kv_per_page, 0) == cur_moe  # unknown unit: no-op


# ---- scheduler decision logic (method bound to a stub; no engine / GPU) ----------------------
import time
import types
from types import SimpleNamespace


def _stub_scheduler(monkeypatch, *, num_pages=1024, base_pages=1024, moe=8000, idle_s=300.0, cap=262144):
    import freetoken.scheduler.scheduler as sched_mod
    from freetoken.scheduler.scheduler import Scheduler

    monkeypatch.setattr(
        "freetoken.kvcache.cache_status.compute_cache_unit_bytes",
        lambda engine: {"kv_bytes_per_token": 14_336, "moe_bytes_per_expert": 2_764_800},
    )
    self = SimpleNamespace(
        config=SimpleNamespace(
            kv_elastic_max_tokens=cap, kv_elastic_idle_s=idle_s, page_size=64,
            tp_info=SimpleNamespace(size=1),
        ),
        engine=SimpleNamespace(
            num_pages=num_pages, moe_offload_cache=SimpleNamespace(cache_size=moe), max_seq_len=num_pages * 64,
        ),
        _pending_rebuild=None, _parked_msgs=[], _elastic_attempted=set(), _elastic_seq=0,
        _elastic_base_pages=base_pages, _elastic_last_activity=time.monotonic(), _elastic_backoff_until=0.0,
    )
    self._elastic_kv_enabled = types.MethodType(Scheduler._elastic_kv_enabled, self)
    self._elastic_kv_maybe_resize = types.MethodType(Scheduler._elastic_kv_maybe_resize, self)
    return self


def _msg(uid, n, max_tokens=1024):
    return SimpleNamespace(uid=uid, input_ids=list(range(n)), sampling_params=SimpleNamespace(max_tokens=max_tokens))


def test_elastic_fits_no_action(monkeypatch):
    s = _stub_scheduler(monkeypatch)
    assert s._elastic_kv_maybe_resize(_msg(1, 1000), 1000) is False
    assert s._pending_rebuild is None and not s._parked_msgs


def test_elastic_grow_parks_once(monkeypatch):
    s = _stub_scheduler(monkeypatch)
    m = _msg(2, 130_000)
    assert s._elastic_kv_maybe_resize(m, 130_000) is True
    rb = s._pending_rebuild
    assert rb is not None and rb.request_id == "auto:1"
    assert rb.num_pages == 131072 // 64            # doubled from the 64K base to 128K
    freed = (rb.num_pages - 1024) * 14_336 * 64
    assert rb.moe_cache_size <= 8000 - freed // 2_764_800
    assert s._parked_msgs == [m] and 2 in s._elastic_attempted
    # re-processing the same request after the rebuild must not park it again
    s._pending_rebuild = None
    assert s._elastic_kv_maybe_resize(m, 130_000) is False
    # a different request arriving while a rebuild is queued rides along
    s._pending_rebuild = rb
    assert s._elastic_kv_maybe_resize(_msg(3, 10), 10) is True
    assert len(s._parked_msgs) == 2


def test_elastic_grow_caps_and_shrinks_after_idle(monkeypatch):
    s = _stub_scheduler(monkeypatch)
    assert s._elastic_kv_maybe_resize(_msg(4, 250_000), 250_000) is True
    assert s._pending_rebuild.num_pages == 262144 // 64
    # beyond the cap: not our problem (falls through to the too-long error)
    s2 = _stub_scheduler(monkeypatch)
    assert s2._elastic_kv_maybe_resize(_msg(5, 300_000), 300_000) is False
    # grown pool, small request, not idle long enough -> stay
    s3 = _stub_scheduler(monkeypatch, num_pages=4096, moe=7000)
    assert s3._elastic_kv_maybe_resize(_msg(6, 20), 20) is False
    # ... idle long enough -> shrink to the base, slots come back
    s3._elastic_last_activity = time.monotonic() - 1000
    assert s3._elastic_kv_maybe_resize(_msg(7, 20), 20) is True
    rb = s3._pending_rebuild
    assert rb.num_pages == 1024 and rb.moe_cache_size > 7000
    # backoff after a failed automatic resize blocks further attempts
    s4 = _stub_scheduler(monkeypatch)
    s4._elastic_backoff_until = time.monotonic() + 100
    assert s4._elastic_kv_maybe_resize(_msg(8, 130_000), 130_000) is False


# ---- stepped shrink -------------------------------------------------------------------------
#
# Growing walks a 64K/128K/256K ladder, but shrinking only ever went all the way to the base and
# only for a request that fits it. A server that grew for one long prompt and then sees a steady
# diet of mid-length ones therefore kept the whole 256K pool -- on the NVMe-tier profile that is
# ~590 expert slots held hostage, which is the decode rate.

def test_plan_shrink_tokens_lands_on_the_grow_ladder():
    from freetoken.scheduler.elastic_kv import plan_shrink_tokens

    # a 90K request in a 256K pool: down to the 128K step, not to the base and not to 90K
    assert plan_shrink_tokens(90_000, CAP, BASE, CAP) == 2 * BASE
    assert plan_shrink_tokens(1000, CAP, BASE, CAP) == BASE
    # already at (or below) the right step: nothing to do
    assert plan_shrink_tokens(90_000, 2 * BASE, BASE, CAP) is None
    assert plan_shrink_tokens(200_000, CAP, BASE, CAP) is None
    assert plan_shrink_tokens(1000, BASE, BASE, CAP) is None
    # every answer is a step the grow path can also produce, so an oscillating workload
    # does not rebuild on every request
    for needed in (10, 60_000, 70_000, 130_000, 200_000):
        got = plan_shrink_tokens(needed, CAP, BASE, CAP)
        assert got is None or got == plan_target_tokens(needed, BASE, CAP)


def test_elastic_shrinks_by_steps_for_a_midsize_request(monkeypatch):
    s = _stub_scheduler(monkeypatch, num_pages=4096, base_pages=1024, moe=6500)  # 256K pool
    s._elastic_last_activity = time.monotonic() - 1000
    assert s._elastic_kv_maybe_resize(_msg(20, 90_000), 90_000) is True
    rb = s._pending_rebuild
    assert rb.num_pages == 131072 // 64, "should step down to 128K, not stay at 256K"
    assert rb.moe_cache_size > 6500, "the reclaimed KV bytes must come back as expert slots"


def test_stepped_shrink_still_needs_the_idle_window(monkeypatch):
    s = _stub_scheduler(monkeypatch, num_pages=4096, base_pages=1024, moe=6500)
    assert s._elastic_kv_maybe_resize(_msg(21, 90_000), 90_000) is False
    assert s._pending_rebuild is None


def test_stepped_shrink_can_be_turned_off(monkeypatch):
    import freetoken.scheduler.scheduler as sched_mod

    monkeypatch.setattr(sched_mod, "_ELASTIC_STEPPED", False)
    s = _stub_scheduler(monkeypatch, num_pages=4096, base_pages=1024, moe=6500)
    s._elastic_last_activity = time.monotonic() - 1000
    assert s._elastic_kv_maybe_resize(_msg(22, 90_000), 90_000) is False, "old policy: base only"
    assert s._elastic_kv_maybe_resize(_msg(23, 20), 20) is True
    assert s._pending_rebuild.num_pages == 1024
