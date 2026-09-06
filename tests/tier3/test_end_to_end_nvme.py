"""End to end: a CPU MoE decode whose expert weights are streamed from an FTW file on disk.

The host pool holds fewer experts than the layer has, so most routes miss and the C++ resolver
has to fetch them over O_DIRECT before the workers run. The output must match a full-residency
executor exactly -- a wrong row, a torn publish or a bad LRU victim all show up as different
numbers, not as a crash.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch

from freetoken.checkpoint.ftw import FTWWriter, layer_bank_entry_name
from freetoken.tier3.budget import host_bank_split
from freetoken.tier3.wiring import enable_cpu_tier3, tier3_stats

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

# H and I are sized so a bank row clears host_bank_split's 256 KiB paging threshold:
# gate_up 2*I*H*2 = 524,288 B, down H*I*2 = 262,144 B, both whole 4096-blocks.
L, E, H, I, TOP_K = 3, 32, 512, 256, 3
# Far smaller than E so the pool thrashes, but still above one step's distinct working
# set (max bs * TOP_K = 9): below that the resolver runs out of evictable slots and
# drops an expert's contribution outright (see tier3_stats()['pool_exhausted']).
HOST_SLOTS = 12
LAYER = 1


def _weights(seed=0):
    torch.manual_seed(seed)
    gu = (torch.randn(L * E, 2 * I, H) * 0.1).to(torch.bfloat16)
    dn = (torch.randn(L * E, H, I) * 0.1).to(torch.bfloat16)
    return gu, dn


def _write_ftw(tmp_path, gu, dn, shard_limit=4 << 20) -> str:
    out = str(tmp_path / "ftw")
    os.makedirs(out, exist_ok=True)
    w = FTWWriter(out, shard_limit=shard_limit)
    w.add_tensor("model.norm.weight", torch.zeros(8, dtype=torch.bfloat16), kind="weight")
    for layer in range(L):
        w.add_tensor(layer_bank_entry_name("gate_up", layer),
                     gu[layer * E:(layer + 1) * E].contiguous(), kind="experts_bank")
        w.add_tensor(layer_bank_entry_name("down", layer),
                     dn[layer * E:(layer + 1) * E].contiguous(), kind="experts_bank")
    w.finalize({"quant_format": "bf16"})
    return out


def _cache(sources, num_experts):
    return SimpleNamespace(quant_format="bf16", bank_sources=sources, num_layers=L,
                           num_experts=num_experts, decode_target="cpu", cpu_executor=None)


def _executor(cache, bs, dev):
    from freetoken.moe.cpu_executor import CpuMoeExecutor

    return CpuMoeExecutor(cache, top_k=TOP_K, activation="silu",
                          apply_router_weight_on_input=False, num_threads=0,
                          max_tokens=bs, device=dev)


def _pool_banks(host_slots):
    from freetoken.kernel.pinned import alloc_pinned_tensor

    gu = alloc_pinned_tensor(L * host_slots, 2 * I, H, dtype=torch.bfloat16)
    dn = alloc_pinned_tensor(L * host_slots, H, I, dtype=torch.bfloat16)
    gu.zero_()
    dn.zero_()
    return {"gate_up": list(gu.split(host_slots)), "down": list(dn.split(host_slots))}, (gu, dn)


@cuda
@pytest.mark.parametrize("bs", [1, 3])
def test_decode_streamed_from_disk_matches_full_residency(tmp_path, bs):
    dev = torch.device("cuda")
    gu, dn = _weights(seed=bs)
    ftw = _write_ftw(tmp_path, gu, dn)

    # reference: everything resident
    from freetoken.kernel.pinned import alloc_pinned_tensor

    rgu = alloc_pinned_tensor(L * E, 2 * I, H, dtype=torch.bfloat16)
    rdn = alloc_pinned_tensor(L * E, H, I, dtype=torch.bfloat16)
    rgu.copy_(gu)
    rdn.copy_(dn)
    ref = _executor(_cache({"gate_up": list(rgu.split(E)), "down": list(rdn.split(E))}, E), bs, dev)

    # tier 3: a pool of HOST_SLOTS rows per layer, starting empty
    sources, _keep = _pool_banks(HOST_SLOTS)
    tier_exec = _executor(_cache(sources, E), bs, dev)
    split = host_bank_split({"gate_up": 2 * I * H * 2, "down": H * I * 2})
    assert split.fetched == ("gate_up", "down"), split
    handle = enable_cpu_tier3(tier_exec, sources, ftw, split, num_layers=L, num_experts=E,
                              host_slots=HOST_SLOTS, readers=4)

    g = torch.Generator().manual_seed(bs * 31)
    hidden = torch.randn(bs, H, generator=g).to(dev, torch.bfloat16)
    for step in range(12):
        ids = torch.stack([torch.randperm(E, generator=g)[:TOP_K] for _ in range(bs)])
        ids = ids.to(dev, torch.int32)
        w = torch.rand(bs, TOP_K, generator=g).to(dev, torch.float32)
        a = ref.decode(LAYER, hidden, w, ids.clone()).float()
        b = tier_exec.decode(LAYER, hidden, w, ids.clone()).float()
        torch.cuda.synchronize()
        torch.testing.assert_close(b, a, rtol=0, atol=0, msg=f"step {step} diverged")

    st = tier3_stats(tier_exec)
    assert st["nvme_fetches"] > 0, "nothing was ever streamed -- the test is vacuous"
    assert st["nvme_bytes"] == st["nvme_fetches"] * (2 * I * H * 2 + H * I * 2)
    # the pool is smaller than a step's working set across steps, so it must be reused AND evicted
    assert int(handle.host_slot_for_id[LAYER].ge(0).sum()) <= HOST_SLOTS
    assert st["pool_exhausted"] == 0, "the pool ran dry mid-step; results would be wrong"


@cuda
def test_residency_map_never_exceeds_the_pool(tmp_path):
    """Every published mapping points at a real slot, and no two experts of a layer claim the
    same one -- the invariant that a torn publish or a bad victim choice would break."""
    dev = torch.device("cuda")
    gu, dn = _weights(seed=7)
    ftw = _write_ftw(tmp_path, gu, dn)
    sources, _keep = _pool_banks(HOST_SLOTS)
    ex = _executor(_cache(sources, E), 2, dev)
    split = host_bank_split({"gate_up": 2 * I * H * 2, "down": H * I * 2})
    handle = enable_cpu_tier3(ex, sources, ftw, split, num_layers=L, num_experts=E,
                              host_slots=HOST_SLOTS, readers=4)

    g = torch.Generator().manual_seed(5)
    hidden = torch.randn(2, H, generator=g).to(dev, torch.bfloat16)
    for _ in range(20):
        ids = torch.stack([torch.randperm(E, generator=g)[:TOP_K] for _ in range(2)])
        w = torch.rand(2, TOP_K, generator=g).to(dev, torch.float32)
        ex.decode(LAYER, hidden, w, ids.to(dev, torch.int32))
        torch.cuda.synchronize()
        for layer in range(L):
            live = handle.host_slot_for_id[layer]
            slots = live[live >= 0].tolist()
            assert all(0 <= s < HOST_SLOTS for s in slots), slots
            assert len(slots) == len(set(slots)), f"two experts share a slot: {slots}"


@cuda
def test_an_undersized_pool_is_reported_not_hidden(tmp_path):
    """Negative control for the one silent-corruption path.

    If a step routes to more distinct experts than the pool has slots, the resolver cannot make
    them all resident and the compute skips the ones it could not place -- wrong numbers, no
    crash. That must be *visible*, so the counter has to fire and the output has to differ from
    the full-residency reference. The planner's min_host_slots floor is what keeps it
    unreachable in production.
    """
    from freetoken.kernel.pinned import alloc_pinned_tensor

    dev = torch.device("cuda")
    bs, tiny = 3, 4                      # 3 * TOP_K = 9 distinct routes into 4 slots
    gu, dn = _weights(seed=13)
    ftw = _write_ftw(tmp_path, gu, dn)

    rgu = alloc_pinned_tensor(L * E, 2 * I, H, dtype=torch.bfloat16)
    rdn = alloc_pinned_tensor(L * E, H, I, dtype=torch.bfloat16)
    rgu.copy_(gu)
    rdn.copy_(dn)
    ref = _executor(_cache({"gate_up": list(rgu.split(E)), "down": list(rdn.split(E))}, E), bs, dev)

    sources, _keep = _pool_banks(tiny)
    ex = _executor(_cache(sources, E), bs, dev)
    split = host_bank_split({"gate_up": 2 * I * H * 2, "down": H * I * 2})
    enable_cpu_tier3(ex, sources, ftw, split, num_layers=L, num_experts=E,
                     host_slots=tiny, readers=2)

    g = torch.Generator().manual_seed(99)
    hidden = torch.randn(bs, H, generator=g).to(dev, torch.bfloat16)
    ids = torch.stack([torch.randperm(E, generator=g)[:TOP_K] for _ in range(bs)])
    ids = ids.to(dev, torch.int32)
    w = torch.rand(bs, TOP_K, generator=g).to(dev, torch.float32)
    a = ref.decode(LAYER, hidden, w, ids.clone()).float()
    b = ex.decode(LAYER, hidden, w, ids.clone()).float()
    torch.cuda.synchronize()

    assert tier3_stats(ex)["pool_exhausted"] > 0, "an undersized pool went unreported"
    assert not torch.allclose(b, a), "expected dropped experts to change the output"


@cuda
def test_the_executor_owns_the_residency_map(tmp_path):
    """Lifetime guard. The C++ resolver holds raw pointers into the handle's pinned tensors and
    writes through them from the coordinator thread. If the handle were only owned by the
    caller, dropping it would free that memory under a live writer -- heap corruption, which is
    exactly how this was found. Enabling the tier must anchor it to the executor.
    """
    dev = torch.device("cuda")
    gu, dn = _weights(seed=21)
    ftw = _write_ftw(tmp_path, gu, dn)
    sources, _keep = _pool_banks(HOST_SLOTS)
    ex = _executor(_cache(sources, E), 1, dev)
    split = host_bank_split({"gate_up": 2 * I * H * 2, "down": H * I * 2})
    handle = enable_cpu_tier3(ex, sources, ftw, split, num_layers=L, num_experts=E,
                              host_slots=HOST_SLOTS, readers=2)
    assert getattr(ex, "_tier3_handle", None) is handle
    assert handle.host_slot_for_id.data_ptr() != 0

    # drop the caller's reference; the executor's must keep it alive
    del handle
    import gc

    gc.collect()
    g = torch.Generator().manual_seed(4)
    hidden = torch.randn(1, H, generator=g).to(dev, torch.bfloat16)
    ids = torch.randperm(E, generator=g)[:TOP_K].view(1, TOP_K).to(dev, torch.int32)
    w = torch.rand(1, TOP_K, generator=g).to(dev, torch.float32)
    ex.decode(LAYER, hidden, w, ids)
    torch.cuda.synchronize()
    assert tier3_stats(ex)["nvme_fetches"] > 0


@cuda
def test_a_routed_hit_is_never_evicted_by_a_miss_in_the_same_step(tmp_path):
    """Resolving a layer used to be one pass over the routed ids, so a miss at position i could
    evict the slot of a HIT at position j > i: the hit's LRU stamp was still its old one when
    the victim search ran.

    Two consequences. The loop re-read that expert from disk when it reached it -- a read the
    step had already paid for. And the slot it freed is one the GPU may be gathering from over
    UVA right now (the routing pack's exclusion list, which was meant to cover exactly this, is
    never populated, so the resolver always runs with n_excl == 0).

    Stamping and claiming every resident routed expert BEFORE choosing any victim fixes both,
    and the extra fetch is what makes it observable from here.
    """
    dev = torch.device("cuda")
    slots = 12
    gu, dn = _weights(seed=77)
    ftw = _write_ftw(tmp_path, gu, dn)
    sources, _keep = _pool_banks(slots)
    ex = _executor(_cache(sources, E), 1, dev)
    split = host_bank_split({"gate_up": 2 * I * H * 2, "down": H * I * 2})
    handle = enable_cpu_tier3(ex, sources, ftw, split, num_layers=L, num_experts=E,
                              host_slots=slots, readers=4)

    hidden = torch.randn(1, H).to(dev, torch.bfloat16)
    w = torch.full((1, TOP_K), 0.5, device=dev, dtype=torch.float32)

    def step(expert_ids):
        ex.decode(LAYER, hidden, w,
                  torch.tensor([expert_ids], device=dev, dtype=torch.int32))
        torch.cuda.synchronize()

    # Fill the pool exactly, oldest first: experts 0-2 end up the least recently used.
    for base in range(0, slots, TOP_K):
        step([base, base + 1, base + 2])
    assert int(handle.host_slot_for_id[LAYER].ge(0).sum()) == slots
    assert handle.host_slot_for_id[LAYER][0].item() >= 0, "expert 0 must be resident and LRU"

    # Two misses and one resident expert, the resident one LAST in the routed order: the misses
    # need victims and expert 0 is the least recently used slot they would pick.
    before = tier3_stats(ex)["nvme_fetches"]
    step([20, 21, 0])
    fetched = tier3_stats(ex)["nvme_fetches"] - before

    assert fetched == 2, (
        f"expected the two genuine misses only, got {fetched} -- a routed hit was evicted "
        "and re-read within its own step"
    )
    assert handle.host_slot_for_id[LAYER][0].item() >= 0, "the routed hit was evicted"
    for e in (20, 21, 0):
        assert handle.host_slot_for_id[LAYER][e].item() >= 0, f"expert {e} not resident after"
