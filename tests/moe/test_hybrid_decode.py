"""Hybrid decode choreography (NVFP4 experts): the split ensure outputs, the GEMV's slot -1
early exit + folded CPU addend, and the side-stream demand-copy overlap -- eager vs the
dequant reference, overlap on/off bit-identical, and under CUDA-graph capture + replays.
"""

from __future__ import annotations

import pytest
import torch

from freetoken.moe.offload_cache import OffloadMoeCache

from .test_nvfp4_backends import (
    E,
    H,
    I,
    L,
    S,
    TOPK,
    _assert_close,
    _make_native_sources,
    _ref_moe,
)

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _cache(device, cache_size=E, frac=0.5):
    """E slots shared by L=2 layers of E experts each: alternating layers keep evicting, so
    every step has hits, fetches and CPU (overflow) routes."""
    sources = _make_native_sources(device, seed=3)
    cache = OffloadMoeCache(
        num_layers=L, num_experts=E, cache_size=cache_size, device=device, quant_format="nvfp4",
        decode_target="hybrid", hybrid_max_fetch=E, hybrid_fetch_fraction=frac,
    )
    cache.set_bank_sources({name: sources[name] for name in cache.bank_schema})
    cache.reset()
    return cache, sources


def _gpu_part(sources, layer, hidden, w, raw_ids, gpu_ids):
    """Reference over the GPU-kept routes only (zero weight elsewhere)."""
    keep = gpu_ids >= 0
    w_ref = torch.where(keep, w, torch.zeros_like(w))
    return _ref_moe(sources, layer, hidden, w_ref, raw_ids)


@cuda
def test_split_ensure_classes_partition_the_routes():
    device = torch.device("cuda")
    cache, _ = _cache(device, frac=0.5)
    # fill the cache with layer 1's experts so layer 0's routes all miss
    warm = torch.arange(E, dtype=torch.int32, device=device).view(1, E)
    cache.ensure_experts_hybrid(1, warm, split=False)
    cache.copy_missing()
    ids = torch.tensor([[0, 1], [2, 3], [1, 5]], dtype=torch.int32, device=device)
    raw = ids.clone()
    r = cache.ensure_experts_hybrid(0, ids, split=True, dedup=True)
    torch.cuda.synchronize()
    # the outputs are views of the cache's shared buffers: snapshot before the next call
    cpu_ids, hit_ids, fetched_ids = r.cpu_ids.clone(), r.hit_ids.clone(), r.fetched_ids.clone()
    uniq_slots, uniq_hit, uniq_fetched, pairs = (
        r.uniq_slots.clone(), r.uniq_hit.clone(), r.uniq_fetched.clone(), r.pairs.clone())
    # dedup tables: every GPU route appears exactly once under its expert's entry
    slots_np = ids.cpu().view(-1).tolist()
    seen = {}
    for route, slot in enumerate(slots_np):
        if slot >= 0:
            seen.setdefault(slot, []).append(route)
    u_used = [u for u in range(uniq_slots.numel()) if int(uniq_slots[u]) >= 0]
    assert len(u_used) == len(seen), (u_used, seen)
    for u in u_used:
        slot = int(uniq_slots[u]); routes = sorted(int(p) for p in pairs[u].tolist() if p >= 0)
        assert routes == seen[slot], (u, slot, routes, seen[slot])
        # hit vs fetched classification matches the per-route split
        r0 = routes[0]
        if int(hit_ids.view(-1)[r0]) >= 0:
            assert int(uniq_hit[u]) == slot and int(uniq_fetched[u]) == -1
        else:
            assert int(uniq_fetched[u]) == slot and int(uniq_hit[u]) == -1
    # every route lands in exactly one class
    on_cpu, hit, fetched = cpu_ids >= 0, hit_ids >= 0, fetched_ids >= 0
    assert torch.equal((on_cpu.int() + hit.int() + fetched.int()), torch.ones_like(ids))
    assert torch.equal(torch.where(on_cpu, raw, torch.full_like(raw, -1)), cpu_ids)
    assert torch.equal(torch.where(hit | fetched, ids, torch.full_like(ids, -1)),
                       torch.maximum(hit_ids, fetched_ids))
    # first step: nothing was resident, so no hits; the fetched count matches the plan
    assert not hit.any()
    assert int(fetched.sum()) >= 1
    n_fetched_experts = int(cache.num_indices.item())
    assert n_fetched_experts == torch.unique(fetched_ids[fetched]).numel()
    # second step with the same routing: the fetched ones are now hits
    ids2 = raw.clone()
    r2 = cache.ensure_experts_hybrid(0, ids2, split=True)
    hit2 = r2.hit_ids
    torch.cuda.synchronize()
    assert torch.equal(hit2 >= 0, fetched)
    assert torch.equal(hit2[fetched], fetched_ids[fetched])


@cuda
@pytest.mark.parametrize("overlap", [False, True])
def test_hybrid_decode_layer_matches_reference(overlap, monkeypatch):
    """The full _decode_hybrid choreography through OffloadMoELayer (CPU executor + GPU
    slot cache) against the dequant reference, with and without the side-stream copy."""
    import freetoken.layers.moe as moe_mod
    from freetoken.layers.moe import OffloadMoELayer
    from freetoken.moe.cpu_executor import CpuMoeExecutor

    monkeypatch.setattr(moe_mod, "_HYBRID_COPY_OVERLAP", overlap)
    device = torch.device("cuda")
    cache, sources = _cache(device, frac=0.5)
    ex = CpuMoeExecutor(cache, top_k=TOPK, activation="silu", apply_router_weight_on_input=False,
                        num_threads=4, max_tokens=4, device=device)
    cache.set_cpu_executor(ex)
    layer = OffloadMoELayer.__new__(OffloadMoELayer)
    # minimal layer surface _decode_hybrid / _expert_gemm need
    layer.offload_cache = cache
    layer.activation = "silu"
    layer.apply_router_weight_on_input = False
    layer.top_k = TOPK
    torch.manual_seed(7)
    for step in range(6):
        layer.layer_id = step % L  # alternate layers so the shared slots keep missing
        hidden = torch.randn(3, H, dtype=torch.bfloat16, device=device) / 4
        w = torch.rand(3, TOPK, dtype=torch.float32, device=device)
        raw = torch.randint(0, E, (3, TOPK), dtype=torch.int32, device=device)
        ref = _ref_moe(sources, layer.layer_id, hidden, w, raw)
        out = layer._decode_hybrid(cache, hidden, w, raw.clone())
        torch.cuda.synchronize()
        _assert_close(out, ref)


@cuda
def test_hybrid_decode_overlap_is_bit_identical_and_graph_safe(monkeypatch):
    import freetoken.layers.moe as moe_mod
    from freetoken.layers.moe import OffloadMoELayer
    from freetoken.moe.cpu_executor import CpuMoeExecutor

    device = torch.device("cuda")
    torch.manual_seed(11)
    steps = [(torch.randn(3, H, dtype=torch.bfloat16, device=device) / 4,
              torch.rand(3, TOPK, dtype=torch.float32, device=device),
              torch.randint(0, E, (3, TOPK), dtype=torch.int32, device=device)) for _ in range(5)]

    def run(overlap: bool, graph: bool):
        monkeypatch.setattr(moe_mod, "_HYBRID_COPY_OVERLAP", overlap)
        cache, sources = _cache(device, frac=0.5)
        ex = CpuMoeExecutor(cache, top_k=TOPK, activation="silu", apply_router_weight_on_input=False,
                            num_threads=4, max_tokens=4, device=device)
        cache.set_cpu_executor(ex)
        # layer 1 warmed into the slots so layer 0 (the graph's layer) keeps missing
        warm = torch.arange(E, dtype=torch.int32, device=device).view(1, E)
        cache.ensure_experts_hybrid(1, warm, split=False)
        cache.copy_missing()
        layer = OffloadMoELayer.__new__(OffloadMoELayer)
        layer.layer_id = 0
        layer.offload_cache = cache
        layer.activation = "silu"
        layer.apply_router_weight_on_input = False
        layer.top_k = TOPK
        hidden = torch.empty(3, H, dtype=torch.bfloat16, device=device)
        w = torch.empty(3, TOPK, dtype=torch.float32, device=device)
        ids = torch.empty(3, TOPK, dtype=torch.int32, device=device)
        outs = []
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            # warmup (materializes the executor's pinned buffers / flag task + side stream)
            hidden.copy_(steps[0][0]); w.copy_(steps[0][1]); ids.copy_(steps[0][2])
            out = layer._decode_hybrid(cache, hidden, w, ids)
            torch.cuda.synchronize()
            if graph:
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, stream=stream):
                    out = layer._decode_hybrid(cache, hidden, w, ids)
                torch.cuda.synchronize()
            for h, ww, rr in steps[1:]:
                hidden.copy_(h); w.copy_(ww); ids.copy_(rr)
                if graph:
                    g.replay()
                else:
                    out = layer._decode_hybrid(cache, hidden, w, ids)
                torch.cuda.synchronize()
                outs.append(out.clone())
                ref = _ref_moe(sources, 0, h, ww, rr)
                _assert_close(out, ref)
        return outs

    a = run(False, False)
    b = run(True, False)
    c = run(True, True)
    for x, y, z in zip(a, b, c):
        assert torch.equal(x, y), "overlap on/off must be bit-identical"
        assert torch.equal(x, z), "graph replay must equal eager"


def _hybrid_layer(device):
    from freetoken.layers.moe import OffloadMoELayer
    from freetoken.moe.cpu_executor import CpuMoeExecutor

    cache, sources = _cache(device, frac=0.5)
    ex = CpuMoeExecutor(cache, top_k=TOPK, activation="silu", apply_router_weight_on_input=False,
                        num_threads=4, max_tokens=4, device=device)
    cache.set_cpu_executor(ex)
    layer = OffloadMoELayer.__new__(OffloadMoELayer)
    layer.offload_cache = cache
    layer.activation = "silu"
    layer.apply_router_weight_on_input = False
    layer.top_k = TOPK
    layer.tp_size = 1
    return layer, cache, sources


@cuda
def test_hybrid_decode_fused_act_matches_separate_act(monkeypatch):
    """gemm2 forming silu(gate)*up on load vs the separate act kernel: the activation is
    rounded to bf16 identically, so the outputs agree to bf16 rounding of the GEMV."""
    import freetoken.moe.fused_nvfp4 as fn

    device = torch.device("cuda")
    layer, cache, sources = _hybrid_layer(device)
    torch.manual_seed(11)
    for step in range(4):
        layer.layer_id = step % L
        hidden = torch.randn(3, H, dtype=torch.bfloat16, device=device) / 4
        w = torch.rand(3, TOPK, dtype=torch.float32, device=device)
        raw = torch.randint(0, E, (3, TOPK), dtype=torch.int32, device=device)
        # same LRU state for both calls -> identical CPU/GPU route split (the CPU path
        # quantizes activations to int8, so a different split is not an A/B)
        cache.reset()
        monkeypatch.setattr(fn, "_FUSED_ACT", False)
        ref = layer._decode_hybrid(cache, hidden, w, raw.clone())
        torch.cuda.synchronize()
        cache.reset()
        monkeypatch.setattr(fn, "_FUSED_ACT", True)
        out = layer._decode_hybrid(cache, hidden, w, raw.clone())
        torch.cuda.synchronize()
        torch.testing.assert_close(out.float(), ref.float(), rtol=1e-2, atol=1e-2)
        _assert_close(out, _ref_moe(sources, layer.layer_id, hidden, w, raw))


@cuda
def test_hybrid_decode_combine_epilogue_matches_separate_ops(monkeypatch):
    """MoeCombineEpilogue through _decode_hybrid: routed sum + CPU partial + shared gate +
    hyper-connection combine in one kernel vs the three separate ops on the routed output."""
    from freetoken.kernel.triton.hc import hc_combine
    from freetoken.kernel.triton.moe_finalize import MoeCombineEpilogue
    from freetoken.kernel.triton.moe_shared_gate import shared_gate_mul_add

    device = torch.device("cuda")
    layer, cache, sources = _hybrid_layer(device)
    hc = 4
    torch.manual_seed(5)
    for step in range(4):
        layer.layer_id = step % L
        hidden = torch.randn(3, H, dtype=torch.bfloat16, device=device) / 4
        w = torch.rand(3, TOPK, dtype=torch.float32, device=device)
        raw = torch.randint(0, E, (3, TOPK), dtype=torch.int32, device=device)
        shared = torch.randn(3, H, dtype=torch.bfloat16, device=device)
        gate = torch.rand(3, dtype=torch.float32, device=device)
        R = torch.randn(3, hc * H, dtype=torch.bfloat16, device=device)
        inj = torch.randn(3, hc, dtype=torch.bfloat16, device=device)
        cache.reset()
        routed = layer._decode_hybrid(cache, hidden, w, raw.clone())
        ref = hc_combine(R, shared_gate_mul_add(routed, shared, gate), inj, hc)
        ep = MoeCombineEpilogue(shared, gate, R, inj, hc)
        cache.reset()
        out = layer._decode_hybrid(cache, hidden, w, raw.clone(), ep)
        torch.cuda.synchronize()
        assert ep.result is out and out.shape == R.shape
        # the fused path skips two intermediate bf16 roundings: allow ~2 bf16 ulps
        torch.testing.assert_close(out.float(), ref.float(), rtol=2e-2, atol=5e-2)


@cuda
def test_hybrid_decode_epilogue_produces_gate_from_weight():
    """MoeCombineEpilogue given only gate_weight: _decode_hybrid produces the gate (fused
    into the pre-quant launch when available) before the experts run; same result as
    handing it a precomputed gate."""
    from freetoken.kernel.triton.moe_finalize import MoeCombineEpilogue
    from freetoken.kernel.triton.moe_shared_gate import shared_gate_sigmoid

    device = torch.device("cuda")
    layer, cache, sources = _hybrid_layer(device)
    hc = 4
    torch.manual_seed(9)
    for step in range(3):
        layer.layer_id = step % L
        hidden = torch.randn(3, H, dtype=torch.bfloat16, device=device) / 4
        w = torch.rand(3, TOPK, dtype=torch.float32, device=device)
        raw = torch.randint(0, E, (3, TOPK), dtype=torch.int32, device=device)
        shared = torch.randn(3, H, dtype=torch.bfloat16, device=device)
        gw = (torch.randn(H, device=device) * 0.05).to(torch.bfloat16)
        R = torch.randn(3, hc * H, dtype=torch.bfloat16, device=device)
        inj = torch.randn(3, hc, dtype=torch.bfloat16, device=device)
        cache.reset()
        ep_ref = MoeCombineEpilogue(shared, shared_gate_sigmoid(hidden, gw), R, inj, hc)
        ref = layer._decode_hybrid(cache, hidden, w, raw.clone(), ep_ref)
        cache.reset()
        ep = MoeCombineEpilogue(shared, None, R, inj, hc, gate_weight=gw)
        out = layer._decode_hybrid(cache, hidden, w, raw.clone(), ep)
        torch.cuda.synchronize()
        assert ep.gate is not None and ep.result is out
        torch.testing.assert_close(ep.gate, ep_ref.gate, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(out.float(), ref.float(), rtol=1e-2, atol=2e-2)


@cuda
def test_routing_pack_carries_ids_and_weights():
    """The ensure kernel's routing pack: row 0 = CPU route ids, row 1 = the router weights'
    fp32 bits (what the executor ships to the CPU pool in one memcpy)."""
    device = torch.device("cuda")
    layer, cache, sources = _hybrid_layer(device)
    torch.manual_seed(3)
    layer.layer_id = 0
    hidden = torch.randn(3, H, dtype=torch.bfloat16, device=device) / 4
    w = torch.rand(3, TOPK, dtype=torch.float32, device=device)
    raw = torch.randint(0, E, (3, TOPK), dtype=torch.int32, device=device)
    cache.reset()
    out = layer._decode_hybrid(cache, hidden, w, raw.clone())
    torch.cuda.synchronize()
    n = raw.numel()
    assert torch.equal(cache.routing_pack[1, :n].view(torch.float32), w.reshape(-1))
    ids = cache.routing_pack[0, :n]
    assert ((ids == -1) | ((ids >= 0) & (ids < E))).all()
    _assert_close(out, _ref_moe(sources, 0, hidden, w, raw))
