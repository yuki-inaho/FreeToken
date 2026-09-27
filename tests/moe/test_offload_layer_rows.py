"""OffloadMoeCache.layer_rows: an appended bank layer narrower than the cache width
(DeepSeek-V4.1's 128-expert DSpark stages on the 384-wide tower cache) is accepted by
set_bank_sources, recorded per layer, and refused by every whole-layer path with a
message. The cache is built on the CPU device (the GPU-only pieces -- the fused copy plan,
the slot gathers -- are skipped there); the bookkeeping under test is device-agnostic."""

from __future__ import annotations

import pytest
import torch

from freetoken.moe.offload_cache import OffloadMoeCache


def _cache(num_layers=4, num_experts=8, cache_size=10, **kw):
    return OffloadMoeCache(
        num_layers=num_layers, num_experts=num_experts, cache_size=cache_size,
        device=torch.device("cpu"), **kw,
    )


def _sources(rows_per_layer):
    return {
        "gate_up": [torch.randn(r, 4, 6) for r in rows_per_layer],
        "down": [torch.randn(r, 6, 4) for r in rows_per_layer],
    }


def test_narrow_appended_layer_is_recorded():
    cache = _cache()
    cache.set_bank_sources(_sources([8, 8, 8, 3]))
    assert cache.layer_rows == [8, 8, 8, 3]
    assert cache.host_slots == 8  # untouched (tier-3 meaning)


def test_all_full_layers_record_the_width():
    cache = _cache()
    cache.set_bank_sources(_sources([8, 8, 8, 8]))
    assert cache.layer_rows == [8] * 4


@pytest.mark.parametrize("rows", [0, 9])
def test_rows_outside_zero_to_num_experts_rejected(rows):
    cache = _cache()
    with pytest.raises(AssertionError):
        cache.set_bank_sources(_sources([8, 8, 8, rows]))


def test_banks_of_one_layer_must_agree_on_rows():
    cache = _cache()
    src = _sources([8, 8, 8, 3])
    src["down"][3] = torch.randn(4, 6, 4)
    with pytest.raises(AssertionError, match="has 4 rows"):
        cache.set_bank_sources(src)


def test_whole_layer_paths_refuse_a_narrow_layer():
    cache = _cache()
    cache.set_bank_sources(_sources([8, 8, 8, 3]))
    # full layers are fine
    cache._require_full_layer(0, "x")
    assert len(cache._prefill_sources(2)) == 2
    with pytest.raises(RuntimeError, match=r"layer 3 holds 3 expert rows, not the cache width 8"):
        cache._prefill_sources(3)
    with pytest.raises(RuntimeError, match="materialize_layer"):
        cache.materialize_layer(3)
    # the pageable whole-layer copy (unpinned layer) also refuses
    cache._unpinned_layers = frozenset({3})
    cache._pending_src_layer = 3
    cache._pending_whole_layer = True
    with pytest.raises(RuntimeError, match="copy_missing"):
        cache.copy_missing()


def test_is_full_layer():
    cache = _cache()
    cache.set_bank_sources(_sources([8, 8, 8, 3]))
    assert [cache.is_full_layer(L) for L in range(4)] == [True, True, True, False]


def test_prefill_prefetch_skips_a_narrow_layer():
    """The overlap choreography prefetches "this layer" and "the next": the last full-width
    layer's next layer is a narrow stage (V4.1: 40 after 39), which prefills through
    ensure_experts on its own -- the prefetch must be a silent no-op there, never a copy and
    never an error (the first real prompt on the 5090 died on the latter)."""
    cache = _cache(cache_size=16, prefill_overlap=True)
    cache.set_bank_sources(_sources([8, 8, 8, 3]))
    cache.prefetch_prefill_layer(2)  # full layer: fills buffer 0
    before = list(cache._prefill_buffer_layer)
    cache.prefetch_prefill_layer(3)  # the narrow "next layer": no-op
    assert list(cache._prefill_buffer_layer) == before
    # the whole-layer source lookup itself still refuses a narrow layer
    with pytest.raises(RuntimeError, match="prefill whole-layer prefetch"):
        cache._prefill_sources(3)
