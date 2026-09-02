"""--linear-state-cache: explicit GDN snapshot-cache slot count vs the ratio-derived default."""
from types import SimpleNamespace

from freetoken.kvcache.linear_state_pool import _linear_pool_min_slots, _linear_pool_num_slots


def _cfg(explicit=-1, ratio=2.0, mr=1, cache_type="hybrid_radix"):
    return SimpleNamespace(max_running_req=mr, cache_type=cache_type,
                           linear_state_cache_ratio=ratio, linear_state_cache=explicit)


def test_ratio_default_keeps_the_four_slot_floor():
    assert _linear_pool_num_slots(_cfg()) == 4 * 1 + 4 + 1          # 9 slots at mr=1
    assert _linear_pool_num_slots(_cfg(ratio=8.0)) == 4 + 8 + 1


def test_explicit_cache_count_overrides_the_floor():
    assert _linear_pool_num_slots(_cfg(explicit=1)) == 4 + 1 + 1     # 6 slots
    assert _linear_pool_num_slots(_cfg(explicit=0)) == 4 + 0 + 1     # working set only
    assert _linear_pool_num_slots(_cfg(explicit=0)) == _linear_pool_min_slots(_cfg())
    assert _linear_pool_num_slots(_cfg(explicit=6, mr=2)) == 8 + 6 + 1


def test_naive_cache_type_ignores_the_knob():
    assert _linear_pool_num_slots(_cfg(explicit=1, cache_type="naive")) == 2
