"""A cache rebuild (elastic-KV grow) must not leave tier 3 holding freed addresses.

Both failures here took a live server down after an elastic-KV grow, and neither is visible
from the rebuild site:

* the prefill-buffer and slot-cache pointer tables are cached device addresses of buffers
  ``rebuild`` frees and reallocates -- reusing them is an illegal memory access reported at
  whatever CUDA call comes next;
* ``rebuild`` reallocated the routing pack at a hardcoded 2 rows, dropping the tier-3 host-slot
  row. That one is worse than a crash: ``cpu_executor.decode_submit`` shape-checks the pack
  against its pinned mirror and *silently* falls back to per-row copies, so the eviction
  exclusion list would just stop arriving.
"""

from __future__ import annotations

import pytest
import torch

from freetoken.moe.offload_cache import ROUTING_PACK_COLS, ROUTING_PACK_ROWS, OffloadMoeCache

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

L, E = 2, 16


def _cache(device="cuda", cache_size=20):
    c = OffloadMoeCache(num_layers=L, num_experts=E, cache_size=cache_size,
                        device=torch.device(device), quant_format="bf16",
                        decode_target="hybrid", hybrid_max_fetch=E)
    src = {n: [torch.zeros(E, 4, 4) for _ in range(L)] for n in ("gate_up", "down")}
    c.set_bank_sources(src)
    return c


def test_routing_pack_keeps_its_width_across_rebuild():
    c = _cache(device="cpu")
    assert c.routing_pack.shape == (ROUTING_PACK_ROWS, ROUTING_PACK_COLS)
    c.rebuild(24)
    assert c.routing_pack.shape == (ROUTING_PACK_ROWS, ROUTING_PACK_COLS), (
        "rebuild narrowed the routing pack; the executor would silently drop the tier-3 row")


def test_routing_pack_views_follow_the_new_storage():
    c = _cache(device="cpu")
    c.rebuild(24)
    for name, row in (("cpu_ids", 0), ("cpu_w", 1), ("gpu_host_slots", 2)):
        view = getattr(c, name)
        assert view.data_ptr() == c.routing_pack[row].data_ptr(), (
            f"{name} still aliases the pre-rebuild pack")


def test_rebuild_drops_cached_tier3_pointer_tables():
    c = _cache(device="cpu")

    class _Stager:                       # stands in for PrefillStager
        discarded = False

        def discard_inflight(self):
            self.discarded = True

    c._tier3_stager = _Stager()
    c._tier3_buf_ptrs = {0: object()}
    c._tier3_mat_ptrs = object()
    c._tier3_pending_fill[3] = 1

    c.rebuild(24)

    assert c._tier3_buf_ptrs is None, "prefill buffer addresses survived a rebuild"
    assert c._tier3_mat_ptrs is None, "slot-cache addresses survived a rebuild"
    assert not c._tier3_pending_fill, "a deferred gather still targets a freed buffer"
    assert c._tier3_stager.discarded, "in-flight prefill reads were not abandoned"


def test_rebuild_without_tier3_is_untouched():
    """The invalidation must be inert when the tier is off."""
    c = _cache(device="cpu")
    assert c._tier3_stager is None
    c.rebuild(24)          # must not raise
    assert c.cache_size == 24


@cuda
def test_pointer_tables_are_rebuilt_and_usable_after_a_rebuild():
    """The lazy caches must repopulate from the NEW buffers rather than stay None."""
    c = _cache(device="cuda")
    before = c._tier3_buffer_ptrs(0).clone() if c.prefill_bank_buffers else None
    c.rebuild(24)
    if c.prefill_bank_buffers:
        after = c._tier3_buffer_ptrs(0)
        assert after.numel() == len(c.prefill_bank_buffers)
        if before is not None:
            assert not torch.equal(before, after), "addresses did not change across a rebuild"
