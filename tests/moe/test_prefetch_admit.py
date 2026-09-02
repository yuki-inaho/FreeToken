"""Speculative expert-prefetch substrate: admission kernel, used-check, layered copy.

CPU tests exercise the reference mirror (`_prefetch_admit_cpu`) and the plan/counter
plumbing; GPU tests check the Triton kernel against the mirror bit-for-bit and the
layered multi-bank copy against a torch reference.
"""

import pytest
import torch

from freetoken.distributed import set_tp_info, try_get_tp_info


def _init_tp():
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)


def _make_cache(device, num_layers=3, num_experts=8, cache_size=12, max_prefetch=6,
                prefill_overlap=False):
    from freetoken.moe.offload_cache import OffloadMoeCache

    _init_tp()
    cache = OffloadMoeCache(
        num_layers=num_layers,
        num_experts=num_experts,
        cache_size=cache_size,
        device=torch.device(device),
        prefill_overlap=prefill_overlap,
    )
    torch.manual_seed(0)

    def _bank(*shape):
        t = torch.randn(*shape)
        return t.pin_memory() if device == "cuda" else t

    cache.set_bank_sources(
        {
            "gate_up": [_bank(num_experts, 4, 8) for _ in range(num_layers)],
            "down": [_bank(num_experts, 8, 2) for _ in range(num_layers)],
        }
    )
    cache.init_prefetch_plan(max_prefetch)
    return cache


def _snapshot(cache):
    return {
        "slot_for_id": cache.slot_for_id.clone(),
        "id_of_slot": cache.id_of_slot.clone(),
        "usage": cache.usage.clone(),
        "evict2": cache.evict_slots2.clone(),
        "src2": cache.src_indices2.clone(),
        "num2": cache.num_indices2.clone(),
        "issued": cache.pf_issued.clone(),
    }


def test_admit_budget_resident_skip_and_maps():
    cache = _make_cache("cpu")
    E = cache.num_experts
    # Pre-populate: expert (0, 3) resident in slot 5.
    cache.slot_for_id[0, 3] = 5
    cache.id_of_slot[5] = 0 * E + 3
    cache.usage[5] = 7
    cache.step.fill_(9)

    cand = torch.tensor([0 * E + 3, 1 * E + 2, -1, 2 * E + 7, 1 * E + 2, 0 * E + 1],
                        dtype=torch.int32)
    cache.prefetch_admit(cand, budget=2)

    # Resident (0,3) skipped; -1 skipped; duplicate (1,2) deduped via the map; budget=2
    # admits (1,2) and (2,7) only -- (0,1) is beyond the budget.
    assert int(cache.num_indices2.item()) == 2
    assert cache.src_indices2[:2].tolist() == [1 * E + 2, 2 * E + 7]
    for i in range(2):
        slot = int(cache.evict_slots2[i].item())
        flat = int(cache.src_indices2[i].item())
        assert int(cache.slot_for_id.view(-1)[flat].item()) == slot
        assert int(cache.id_of_slot[slot].item()) == flat
        assert int(cache.usage[slot].item()) == 9  # stamped with the current step
    assert int(cache.slot_for_id[0, 1].item()) == -1  # not admitted
    assert int(cache.pf_stamp.item()) == 9
    assert int(cache.pf_issued.item()) == 2
    assert int(cache.step.item()) == 9  # admission never advances the clock


def test_admit_respects_prefill_reserve():
    cache = _make_cache("cpu", num_layers=1, num_experts=4, cache_size=10,
                        prefill_overlap=True)
    assert cache.prefetch_reserve_lo == 8
    cand = torch.tensor([0, 1, 2], dtype=torch.int32)
    cache.prefetch_admit(cand, budget=3)
    n = int(cache.num_indices2.item())
    assert n == 3
    assert all(int(s) >= 8 for s in cache.evict_slots2[:n].tolist())


def test_admit_evicts_lru_victim_and_disowns():
    cache = _make_cache("cpu")
    E = cache.num_experts
    # Fill two slots with different recency; the older one must be the victim.
    for slot, (layer, e, ts) in {0: (2, 1, 3), 1: (2, 2, 1)}.items():
        cache.slot_for_id[layer, e] = slot
        cache.id_of_slot[slot] = layer * E + e
        cache.usage[slot] = ts
    # Make every empty slot non-preferable by loading it up.
    for slot in range(2, cache.cache_size):
        cache.usage[slot] = 100
    cache.prefetch_admit(torch.tensor([0 * E + 0], dtype=torch.int32), budget=1)
    assert int(cache.evict_slots2[0].item()) == 1  # usage 1 < 3 < 100
    assert int(cache.slot_for_id[2, 2].item()) == -1  # old owner disowned
    assert int(cache.slot_for_id[0, 0].item()) == 1


def test_used_check_counters():
    cache = _make_cache("cpu")
    E = cache.num_experts
    cache.step.fill_(5)
    cache.prefetch_admit(torch.tensor([0 * E + 1, 1 * E + 1], dtype=torch.int32), budget=2)
    s0, s1 = (int(x) for x in cache.evict_slots2[:2].tolist())
    # Slot s0: a later demand call hit it (usage bumped past the stamp).
    cache.usage[s0] = 6
    # Slot s1: evicted and re-assigned before use.
    cache.id_of_slot[s1] = 2 * E + 5
    cache.prefetch_used_check()
    assert int(cache.pf_used.item()) == 1
    assert int(cache.pf_evicted_unused.item()) == 1
    stats = cache.decode_miss_stats()
    assert stats["prefetch_issued"] == 2
    assert stats["prefetch_used"] == 1
    assert stats["prefetch_evicted_unused"] == 1


def test_rebuild_clears_plan():
    cache = _make_cache("cpu")
    cache.prefetch_admit(torch.tensor([1], dtype=torch.int32), budget=1)
    assert int(cache.num_indices2.item()) == 1
    cache.rebuild(cache.cache_size)
    assert int(cache.num_indices2.item()) == 0
    assert int(cache.pf_issued.item()) == 0
    assert cache.max_prefetch == 6  # buffers survive the rebuild


def test_ring_plans_are_independent():
    from freetoken.moe.offload_cache import OffloadMoeCache

    _init_tp()
    cache = OffloadMoeCache(
        num_layers=2, num_experts=8, cache_size=12, device=torch.device("cpu")
    )
    cache.set_bank_sources(
        {"gate_up": [torch.randn(8, 4, 8) for _ in range(2)],
         "down": [torch.randn(8, 8, 2) for _ in range(2)]}
    )
    cache.init_prefetch_plan(4, ring=3)
    cache.prefetch_admit(torch.tensor([1, 2], dtype=torch.int32), budget=2, plan=0)
    cache.prefetch_admit(torch.tensor([8 + 5], dtype=torch.int32), budget=1, plan=2)
    assert cache.num_indices2.tolist() == [2, 0, 1]
    assert cache.src_indices2[0:2].tolist() == [1, 2]  # plan 0's slice
    assert cache.src_indices2[8:9].tolist() == [13]  # plan 2's slice (offset 2*4)
    assert int(cache.pf_issued.item()) == 3
    # Maps are shared: plan 2's expert reads as resident for a later plan-0 admission.
    cache.prefetch_admit(torch.tensor([8 + 5, 3], dtype=torch.int32), budget=2, plan=0)
    assert int(cache.num_indices2[0].item()) == 1
    assert cache.src_indices2[0:1].tolist() == [3]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_reset_clears_plan_gpu():
    cache = _make_cache("cuda")
    cache.prefetch_admit(torch.tensor([1], dtype=torch.int32, device="cuda"), budget=1)
    assert int(cache.num_indices2.item()) == 1
    cache.reset()  # reset_cache is a GPU kernel; CPU-device caches never call it
    assert int(cache.num_indices2.item()) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_admit_gpu_matches_cpu_reference():
    torch.manual_seed(1)
    E, L, S = 16, 4, 24
    caches = {dev: _make_cache(dev, num_layers=L, num_experts=E, cache_size=S,
                               max_prefetch=8) for dev in ("cpu", "cuda")}
    # Same warm state on both: some resident ids with assorted usage.
    resident = [(0, 2, 4), (1, 5, 11), (2, 9, 2), (3, 1, 7), (1, 2, 30)]
    for slot, (layer, e, ts) in enumerate(resident):
        for cache in caches.values():
            cache.slot_for_id[layer, e] = slot
            cache.id_of_slot[slot] = layer * E + e
            cache.usage[slot] = ts
    for cache in caches.values():
        cache.step.fill_(31)

    cand = torch.tensor(
        [1 * E + 5, 0 * E + 7, 2 * E + 9, -1, 3 * E + 15, 3 * E + 15, 0 * E + 0,
         2 * E + 1, 1 * E + 14],
        dtype=torch.int32,
    )
    caches["cpu"].prefetch_admit(cand.clone(), budget=5)
    caches["cuda"].prefetch_admit(cand.clone().cuda(), budget=5)

    n = int(caches["cpu"].num_indices2.item())
    assert n == int(caches["cuda"].num_indices2.item())
    for name in ("slot_for_id", "id_of_slot", "usage"):
        assert torch.equal(getattr(caches["cpu"], name),
                           getattr(caches["cuda"], name).cpu()), name
    assert torch.equal(caches["cpu"].evict_slots2[:n],
                       caches["cuda"].evict_slots2[:n].cpu())
    assert torch.equal(caches["cpu"].src_indices2[:n],
                       caches["cuda"].src_indices2[:n].cpu())
    assert int(caches["cpu"].pf_issued.item()) == int(caches["cuda"].pf_issued.item())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_layered_copy_matches_torch_reference():
    from freetoken.kernel.fast_index_copy import fast_index_copy_multi_layered_jit
    from freetoken.kernel.pinned import device_ptr

    torch.manual_seed(2)
    L, E, S = 3, 8, 12
    feats = [(4, 8), (8, 2)]  # per-bank row shapes; bytes multiples of 16 (fp32)
    sources = [
        [torch.randn(E, *shape, dtype=torch.float32).pin_memory() for _ in range(L)]
        for shape in feats
    ]
    caches = [
        torch.zeros(S, *shape, dtype=torch.float32, device="cuda") for shape in feats
    ]
    dst_ptrs = torch.tensor([c.data_ptr() for c in caches], dtype=torch.int64, device="cuda")
    feat_bytes = torch.tensor(
        [shape[0] * shape[1] * 4 for shape in feats], dtype=torch.int64, device="cuda"
    )
    table = torch.tensor(
        [[device_ptr(sources[b][layer]) for b in range(len(feats))] for layer in range(L)],
        dtype=torch.int64, device="cuda",
    ).reshape(-1)

    dst_slots = torch.tensor([3, 7, 0], dtype=torch.int32, device="cuda")
    src_flat = torch.tensor([0 * E + 2, 2 * E + 5, 1 * E + 7], dtype=torch.int32, device="cuda")
    num = torch.tensor([2], dtype=torch.int64, device="cuda")  # 3rd row beyond the count

    fast_index_copy_multi_layered_jit(
        dst_ptrs, table, feat_bytes, dst_slots, src_flat, num, num_experts=E
    )
    torch.cuda.synchronize()
    for b in range(len(feats)):
        assert torch.equal(caches[b][3].cpu(), sources[b][0][2])
        assert torch.equal(caches[b][7].cpu(), sources[b][2][5])
        assert caches[b][0].abs().sum() == 0  # beyond the device-side count


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_prefetch_copy_end_to_end_demand_hit():
    """admit -> prefetch_copy -> the demand ensure sees a HIT and copies nothing."""
    from freetoken.moe.offload_kernels import ensure_experts

    torch.manual_seed(3)
    E, L, S = 8, 2, 12
    cache = _make_cache("cuda", num_layers=L, num_experts=E, cache_size=S, max_prefetch=4)
    assert cache._copy_fused_ok
    cand = torch.tensor([1 * E + 3, 0 * E + 6], dtype=torch.int32, device="cuda")
    cache.prefetch_admit(cand, budget=2)
    cache.prefetch_copy()
    torch.cuda.synchronize()
    slot = int(cache.slot_for_id[1, 3].item())
    assert slot >= 0
    assert torch.equal(cache.bank_caches["gate_up"][slot].cpu(),
                       cache.bank_sources["gate_up"][1][3])
    # Demand path: the prefetched expert is a hit (no new copy staged for it).
    ids = torch.tensor([3], dtype=torch.int32, device="cuda")  # layer-local id 3 of layer 1
    ensure_experts(cache, 1, ids)
    torch.cuda.synchronize()
    assert int(ids.item()) == slot  # rewritten to the prefetched slot
    assert int(cache.num_indices.item()) == 0  # nothing to copy
