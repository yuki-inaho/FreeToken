from __future__ import annotations

import pytest

from freetoken.tier3.host_pool import NOT_RESIDENT, Fetch, HostExpertPool


def _pool(layers=2, experts=32, slots=4):
    return HostExpertPool(layers, experts, slots)


def _fetch(pool, layer, experts, exclude=()):
    """begin + commit, i.e. a completed fetch round."""
    fs = pool.begin_fetch(layer, experts, exclude=exclude)
    pool.commit_fetch(fs)
    return fs


# ---------------------------------------------------------------- basics

def test_everything_starts_non_resident():
    p = _pool()
    assert not p.is_resident(0, 0)
    assert p.slot_of(0, 0) == NOT_RESIDENT


def test_first_touch_misses_then_hits():
    p = _pool()
    assert len(_fetch(p, 0, [3])) == 1
    assert p.is_resident(0, 3)
    assert _fetch(p, 0, [3]) == []          # now a hit
    assert (p.hits, p.misses) == (1, 1)


def test_duplicate_experts_in_one_call_fetch_once():
    p = _pool()
    fs = p.begin_fetch(0, [5])
    p.commit_fetch(fs)
    fs2 = p.begin_fetch(0, [5, 5, 5])
    assert fs2 == []


def test_layers_are_independent_pools():
    p = _pool()
    _fetch(p, 0, [1])
    assert p.is_resident(0, 1) and not p.is_resident(1, 1)
    _fetch(p, 1, [1])
    # same expert id, different layer -> different slot bookkeeping, both resident
    assert p.is_resident(0, 1) and p.is_resident(1, 1)


def test_free_slots_are_used_before_any_eviction():
    p = _pool(slots=4)
    fs = _fetch(p, 0, [0, 1, 2, 3])
    assert sorted(f.slot for f in fs) == [0, 1, 2, 3]
    assert all(f.evicted == NOT_RESIDENT for f in fs)


# ---------------------------------------------------------------- LRU identity

def test_evicts_the_least_recently_used_expert():
    p = _pool(slots=4)
    _fetch(p, 0, [10, 11, 12, 13])
    _fetch(p, 0, [11, 12, 13])          # 10 is now the LRU
    fs = _fetch(p, 0, [14])
    assert len(fs) == 1 and fs[0].evicted == 10
    assert not p.is_resident(0, 10) and p.is_resident(0, 14)


def test_a_hit_refreshes_recency_and_spares_the_expert():
    p = _pool(slots=4)
    _fetch(p, 0, [10, 11, 12, 13])
    _fetch(p, 0, [10])                  # touch 10 -> 11 becomes LRU
    fs = _fetch(p, 0, [14])
    assert fs[0].evicted == 11
    assert p.is_resident(0, 10)


def test_victim_tie_break_is_the_lowest_slot():
    """Deterministic tie-break: the C++ mirror has to reproduce this bit-for-bit."""
    p = _pool(experts=32, slots=4)
    _fetch(p, 0, [10, 11, 12, 13])
    p.usage[0].fill_(7)                 # force an exact tie across every slot
    fs = _fetch(p, 0, [14])
    assert fs[0].slot == 0
    assert fs[0].evicted == int(10)     # whichever expert happened to hold slot 0


# ---------------------------------------------------------------- I2 exclusion

def test_excluded_slots_are_never_evicted():
    p = _pool(slots=4)
    _fetch(p, 0, [10, 11, 12, 13])
    keep = p.slot_of(0, 10)             # pretend a live H2D is reading 10's slot
    fs = _fetch(p, 0, [14], exclude=[keep])
    assert fs[0].slot != keep
    assert p.is_resident(0, 10), "an excluded slot was recycled under a live read"


def test_exhausting_the_pool_with_exclusions_raises_rather_than_corrupting():
    p = _pool(slots=4)
    _fetch(p, 0, [10, 11, 12, 13])
    all_slots = [p.slot_of(0, e) for e in (10, 11, 12, 13)]
    with pytest.raises(RuntimeError, match="undersized"):
        p.begin_fetch(0, [14], exclude=all_slots)


def test_two_misses_in_one_call_cannot_claim_the_same_slot():
    p = _pool(slots=4)
    _fetch(p, 0, [10, 11, 12, 13])
    fs = p.begin_fetch(0, [14, 15])
    assert fs[0].slot != fs[1].slot


# ---------------------------------------------------------------- I1 publish ordering

def test_mapping_is_cleared_before_the_read_and_set_after():
    p = _pool(slots=4)
    _fetch(p, 0, [10, 11, 12, 13])
    fs = p.begin_fetch(0, [14])         # bytes have NOT moved yet
    ev = fs[0].evicted
    # mid-flight, a reader must see NEITHER expert as resident -- not the evicted one (its
    # bytes are about to be overwritten) and not the incoming one (its bytes are not there)
    assert not p.is_resident(0, ev), "evicted expert still mapped while its slot is overwritten"
    assert not p.is_resident(0, 14), "incoming expert published before its bytes landed"
    p.commit_fetch(fs)
    assert p.is_resident(0, 14) and not p.is_resident(0, ev)


def test_aborted_fetch_leaves_the_slot_unmapped_and_reusable():
    p = _pool(slots=4)
    fs = p.begin_fetch(0, [10])
    p.abort_fetch(fs)
    assert not p.is_resident(0, 10)
    again = p.begin_fetch(0, [11])
    assert again[0].slot == fs[0].slot   # the slot came back to the free list


def test_reverse_map_stays_consistent_with_the_forward_map():
    p = _pool(slots=4)
    _fetch(p, 0, [10, 11, 12, 13])
    _fetch(p, 0, [14, 15])
    for e in range(p.num_experts):
        s = p.slot_of(0, e)
        if s != NOT_RESIDENT:
            assert int(p.id_of_host_slot[0, s]) == e
    resident = [int(x) for x in p.id_of_host_slot[0] if int(x) != NOT_RESIDENT]
    assert len(resident) == len(set(resident)) <= p.host_slots
