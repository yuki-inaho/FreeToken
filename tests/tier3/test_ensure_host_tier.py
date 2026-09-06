"""The tier-3 indirection inside the hybrid ensure kernel: GPU path vs its CPU mirror.

Two properties matter and neither is visible from the miss-rate counters:

* an expert that is in neither VRAM nor the host pool must NOT be selected for a PCIe gather
  (there is nothing to gather from) -- it has to fall through to the CPU branch, which is where
  the NVMe read happens;
* ``src_indices`` must carry the expert's HOST SLOT, not its expert id, because that integer is
  what the copy kernel multiplies by ``feat_bytes`` to address the host bank.
"""

from __future__ import annotations

import pytest
import torch

from freetoken.moe.offload_cache import OffloadMoeCache

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

NUM_LAYERS, NUM_EXPERTS, CACHE_SIZE, TOP_K = 2, 16, 20, 6
HOST_SLOTS = 10


def _cache(device: str) -> OffloadMoeCache:
    return OffloadMoeCache(
        num_layers=NUM_LAYERS, num_experts=NUM_EXPERTS, cache_size=CACHE_SIZE,
        device=torch.device(device), quant_format="bf16", decode_target="hybrid",
        hybrid_max_fetch=NUM_EXPERTS,
    )


def _attach_host_map(cache: OffloadMoeCache, resident: dict[int, int]) -> None:
    """resident maps expert -> host slot for layer 0; everything else lives on NVMe."""
    m = torch.full((NUM_LAYERS, NUM_EXPERTS), -1, dtype=torch.int32)
    for e, slot in resident.items():
        m[0, e] = slot
    cache.host_slot_for_id = m.to(cache.device)
    cache.host_slots = HOST_SLOTS


def _resident_map(seed: int) -> dict[int, int]:
    g = torch.Generator().manual_seed(seed)
    experts = torch.randperm(NUM_EXPERTS, generator=g)[:HOST_SLOTS].tolist()
    return {e: i for i, e in enumerate(experts)}


@cuda
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_gpu_matches_cpu_mirror_with_a_partial_host_pool(seed):
    gpu, ref = _cache("cuda"), _cache("cpu")
    resident = _resident_map(seed)
    _attach_host_map(gpu, resident)
    _attach_host_map(ref, resident)
    gpu.collect_stats = ref.collect_stats = True

    rng = torch.Generator().manual_seed(seed + 100)
    for _ in range(48):
        ids = torch.randperm(NUM_EXPERTS, generator=rng)[:TOP_K].to(torch.int32)
        g, c = ids.clone().cuda(), ids.clone()
        gpu.ensure_experts_hybrid(0, g)
        ref.ensure_experts_hybrid(0, c)

        assert torch.equal(g.cpu(), c), "slot rewrite diverged"
        assert torch.equal(gpu.slot_for_id.cpu(), ref.slot_for_id.cpu())
        assert torch.equal(gpu.id_of_slot.cpu(), ref.id_of_slot.cpu())
        assert torch.equal(gpu.usage.cpu(), ref.usage.cpu())
        assert int(gpu.num_indices.item()) == int(ref.num_indices.item())
        assert int(gpu.num_missing_full.item()) == int(ref.num_missing_full.item())
        n = int(gpu.num_indices.item())
        assert torch.equal(gpu.src_indices[:n].cpu(), ref.src_indices[:n]), "host rows diverged"
        assert torch.equal(gpu.evict_slots[:n].cpu(), ref.evict_slots[:n])


@cuda
def test_only_host_resident_experts_are_ever_fetched():
    """The core tier-3 rule. An NVMe-only expert must never appear in src_indices, and must be
    left at slot -1 so the CPU branch picks it up."""
    gpu = _cache("cuda")
    resident = _resident_map(7)
    _attach_host_map(gpu, resident)
    host_map = gpu.host_slot_for_id.cpu()

    rng = torch.Generator().manual_seed(11)
    saw_nvme_only = 0
    for _ in range(64):
        ids = torch.randperm(NUM_EXPERTS, generator=rng)[:TOP_K].to(torch.int32)
        before = gpu.slot_for_id[0].cpu().clone()
        g = ids.clone().cuda()
        gpu.ensure_experts_hybrid(0, g)
        n = int(gpu.num_indices.item())
        rows = gpu.src_indices[:n].cpu().tolist()
        # every fetched row is a valid host slot, and belongs to a host-resident expert
        for row in rows:
            assert 0 <= row < HOST_SLOTS, row
            assert row in resident.values()
        newly = [e for e in ids.tolist()
                 if before[e] == -1 and int(gpu.slot_for_id[0, e].item()) != -1]
        for e in newly:
            assert int(host_map[0, e]) >= 0, f"expert {e} was fetched but is NVMe-only"
        for e in ids.tolist():
            if int(host_map[0, e]) < 0 and before[e] == -1:
                assert int(gpu.slot_for_id[0, e].item()) == -1, (
                    f"NVMe-only expert {e} was given a VRAM slot with no host bytes behind it")
                saw_nvme_only += 1
    assert saw_nvme_only > 0, "the fixture never exercised an NVMe-only route"


@cuda
def test_src_indices_are_host_slots_not_expert_ids():
    """With a deliberately permuted map, expert id and host slot disagree for every expert, so
    a path that forgot the indirection is caught."""
    gpu = _cache("cuda")
    resident = {e: (HOST_SLOTS - 1 - i) for i, e in enumerate(range(HOST_SLOTS))}
    _attach_host_map(gpu, resident)
    ids = torch.tensor(sorted(resident)[:TOP_K], dtype=torch.int32)
    gpu.ensure_experts_hybrid(0, ids.clone().cuda())
    n = int(gpu.num_indices.item())
    rows = gpu.src_indices[:n].cpu().tolist()
    assert n > 0
    assert rows == [resident[e] for e in _fetched_experts(gpu, n)]


def _fetched_experts(cache, n):
    return [int(cache.id_of_slot[int(s)].item()) % NUM_EXPERTS
            for s in cache.evict_slots[:n].cpu().tolist()]


@cuda
def test_live_gather_slots_are_published_for_the_installer():
    """Row 2 of the routing pack names the host slots under a live UVA gather, so the tier-3
    installer running concurrently on the CPU pool cannot recycle them."""
    gpu = _cache("cuda")
    _attach_host_map(gpu, _resident_map(5))
    ids = torch.randperm(NUM_EXPERTS)[:TOP_K].to(torch.int32)
    gpu.ensure_experts_hybrid(0, ids.clone().cuda())
    n = int(gpu.num_indices.item())
    published = gpu.gpu_host_slots[:n].cpu()
    assert torch.equal(published, gpu.src_indices[:n].cpu())
    # every lane the installer will scan past the fetched ones must read as "no live gather"
    tail = gpu.gpu_host_slots[n:TOP_K].cpu()
    assert tail.numel() == 0 or int(tail.max().item()) == -1


@cuda
def test_tier_off_is_byte_identical_to_before():
    """The zero-overhead default: with no host map the kernel must behave exactly as it did."""
    on, off = _cache("cuda"), _cache("cuda")
    _attach_host_map(on, {e: e for e in range(NUM_EXPERTS)})  # identity map == full residency
    on.host_slots = NUM_EXPERTS
    rng = torch.Generator().manual_seed(3)
    for _ in range(32):
        ids = torch.randperm(NUM_EXPERTS, generator=rng)[:TOP_K].to(torch.int32)
        a, b = ids.clone().cuda(), ids.clone().cuda()
        on.ensure_experts_hybrid(0, a)
        off.ensure_experts_hybrid(0, b)
        assert torch.equal(a, b)
        assert torch.equal(on.slot_for_id, off.slot_for_id)
        n = int(off.num_indices.item())
        assert torch.equal(on.src_indices[:n], off.src_indices[:n])


# ---- dropped routes (`-1` in topk_ids) ------------------------------------------------------
#
# The table-driven residency policy can DROP a routed expert instead of substituting one, by
# writing -1 into topk_ids and 0 into the weight. `-1` was already the "no expert" value
# everywhere on this path except the ensure step, which is the one place that turns an expert id
# into a slot: Phase 3's gather is masked by lane occupancy, not by the id, so `-1` read
# slot_for_id[base - 1] -- the previous layer's last expert, or out of bounds at layer 0.

@cuda
@pytest.mark.parametrize("seed", [0, 3])
def test_dropped_routes_are_never_fetched_or_activated(seed):
    gpu, ref = _cache("cuda"), _cache("cpu")
    resident = _resident_map(seed)
    _attach_host_map(gpu, resident)
    _attach_host_map(ref, resident)

    rng = torch.Generator().manual_seed(seed + 500)
    saw_drop = 0
    for _ in range(48):
        ids = torch.randperm(NUM_EXPERTS, generator=rng)[:TOP_K].to(torch.int32)
        # drop one or two routes, as the policy would
        n_drop = 1 + int(torch.randint(0, 2, (1,), generator=rng).item())
        dropped = sorted(torch.randperm(TOP_K, generator=rng)[:n_drop].tolist())
        kept = [int(ids[j]) for j in range(TOP_K) if j not in dropped]
        for j in dropped:
            ids[j] = -1
        saw_drop += n_drop

        g, c = ids.clone().cuda(), ids.clone()
        gpu.ensure_experts_hybrid(0, g)
        ref.ensure_experts_hybrid(0, c)

        # the mirror must agree, which is the property the whole tier-3 path is tested on
        assert torch.equal(g.cpu(), c), "slot rewrite diverged on a dropped route"
        assert torch.equal(gpu.slot_for_id.cpu(), ref.slot_for_id.cpu())
        assert torch.equal(gpu.id_of_slot.cpu(), ref.id_of_slot.cpu())
        assert torch.equal(gpu.usage.cpu(), ref.usage.cpu())

        # a dropped lane stays -1: no slot, so no GEMV
        for j in dropped:
            assert int(g[j].item()) == -1, "a dropped route was given a slot"
        # and it never became active, so it can never be selected for a PCIe gather
        act = gpu.active_mask.cpu()
        assert int(act[NUM_EXPERTS - 1]) == (1 if (NUM_EXPERTS - 1) in kept else 0), (
            "the LAST expert was marked active by a -1 route (negative index wrap)")
        n = int(gpu.num_indices.item())
        for row in gpu.src_indices[:n].cpu().tolist():
            assert 0 <= row < HOST_SLOTS, row
    assert saw_drop > 0


@cuda
def test_ensure_is_unchanged_when_no_route_is_dropped():
    """The regression guard on the Phase-3 edit: with no -1 present the rewrite must be exactly
    what it was, or every measurement taken before the drop policy existed is invalidated."""
    gpu, ref = _cache("cuda"), _cache("cpu")
    resident = _resident_map(2)
    _attach_host_map(gpu, resident)
    _attach_host_map(ref, resident)
    rng = torch.Generator().manual_seed(909)
    for _ in range(64):
        ids = torch.randperm(NUM_EXPERTS, generator=rng)[:TOP_K].to(torch.int32)
        g, c = ids.clone().cuda(), ids.clone()
        gpu.ensure_experts_hybrid(0, g)
        ref.ensure_experts_hybrid(0, c)
        assert torch.equal(g.cpu(), c)
        assert torch.equal(gpu.slot_for_id.cpu(), ref.slot_for_id.cpu())
        assert torch.equal(gpu.usage.cpu(), ref.usage.cpu())
