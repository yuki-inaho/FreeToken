"""Fused QSA decode index-write prologue vs the five separate kernels (T <= 4, one request)."""
from __future__ import annotations

import pytest
import torch

from .common import requires_cuda

DEV = "cuda"
# Cross the compression/ring boundary at the long input and output edges. The
# logical positions are large; physical K/V and compressed-cache slots stay small.
HIGH_POSITION_CASES = [
    pytest.param(4, 253950, id="high-253952"),
    pytest.param(4, 258045, id="high-258047"),
]


def _rope_cache(max_pos, rotary_dim, base=10000.0):
    half = rotary_dim // 2
    inv = 1.0 / (base ** (torch.arange(0, half, device=DEV).float() / half))
    t = torch.arange(max_pos, device=DEV).float()
    ang = torch.outer(t, inv)
    return torch.cat([ang.cos(), ang.sin()], dim=1).contiguous()  # [max_pos, rotary_dim]


@requires_cuda
@pytest.mark.parametrize("T,start", [(1, 7), (3, 5), (4, 12), (4, 21), *HIGH_POSITION_CASES])
def test_fused_index_write_matches_separate_kernels(T, start, monkeypatch):
    import freetoken.kernel.triton.qsa.fused_decode as fused
    from freetoken.kernel.triton.qsa import (
        qsa_compress_groups, qsa_index_norm_rope, qsa_index_write_fused, qsa_store_kv_fp8,
        qsa_store_rows,
    )

    monkeypatch.setattr(fused, "_DIRECT_PERMUTE", True)
    torch.manual_seed(T * 31 + start)
    kv_heads, head_dim, idx_dim, q_heads, ratio, cap, rotary = 2, 256, 128, 4, 4, 8, 64
    slots, ring_slots_n, cmp_rows_n = 512, 3, 256
    k = torch.randn(T, kv_heads * head_dim, device=DEV, dtype=torch.bfloat16)
    v = torch.randn(T, kv_heads * head_dim, device=DEV, dtype=torch.bfloat16)
    # Paged K/V slots need not equal logical RoPE positions. Retain the same
    # compression/ring phase without allocating a context-sized K/V slab.
    slot_start = start if start + T <= slots else 64 + start % cap
    out_loc = torch.arange(slot_start, slot_start + T, device=DEV, dtype=torch.int32)
    positions = torch.arange(start, start + T, device=DEV, dtype=torch.int32)
    raw_k = torch.randn(T, idx_dim, device=DEV, dtype=torch.bfloat16)
    idx_q = torch.randn(T, q_heads * idx_dim, device=DEV, dtype=torch.bfloat16)
    kw = torch.randn(idx_dim, device=DEV, dtype=torch.bfloat16) * 0.1
    qw = torch.randn(idx_dim, device=DEV, dtype=torch.bfloat16) * 0.1
    cos_sin = _rope_cache(max(1024, start + T), rotary)
    token_to_req = torch.zeros(T, device=DEV, dtype=torch.int32)
    cu_seqlens = torch.tensor([0, T], device=DEV, dtype=torch.int32)
    ring_slots = torch.tensor([1, 0, 2], device=DEV, dtype=torch.int32)  # request 0 -> ring slot 1
    # host-side plan (mirrors QSASparseAttnBackend._plan_index_writes for one request)
    closing = out_loc % ratio == ratio - 1
    scratch_base = cmp_rows_n - ring_slots_n
    cmp_rows = torch.where(closing, out_loc // ratio, torch.full_like(out_loc, scratch_base + 1)).to(torch.int32)
    ring_rows = (1 * cap + positions % cap).to(torch.int32)

    def fresh():
        kc = torch.zeros(slots, kv_heads, head_dim, device=DEV, dtype=torch.float8_e4m3fn)
        vc = torch.zeros_like(kc)
        ks = torch.zeros(slots, kv_heads, device=DEV); vs = torch.zeros_like(ks)
        ring = torch.randn(ring_slots_n, cap, idx_dim, device=DEV, dtype=torch.bfloat16)
        cmp = torch.zeros(cmp_rows_n, idx_dim, device=DEV, dtype=torch.bfloat16)
        qo = torch.zeros(T, q_heads, idx_dim, device=DEV, dtype=torch.bfloat16)
        return kc, vc, ks, vs, ring, cmp, qo

    torch.manual_seed(99)
    a = fresh()
    torch.manual_seed(99)
    b = fresh()
    torch.manual_seed(99)
    c = fresh()
    assert torch.equal(a[4], b[4])  # same ring contents
    # --- separate kernels
    kc, vc, ks, vs, ring, cmp, qo = a
    qsa_store_kv_fp8(k, v, out_loc, kc, vc, ks, vs)
    pooled = torch.empty(T, idx_dim, device=DEV, dtype=torch.bfloat16)
    first = torch.empty(T, device=DEV, dtype=torch.int32)
    qsa_compress_groups(raw_k, ring, ring_slots, token_to_req, cu_seqlens, positions, ratio, pooled, first)
    qsa_index_norm_rope(pooled, first, cos_sin, kw, 1e-6, cmp, dest_rows=cmp_rows)
    qsa_store_rows(ring, ring_rows, raw_k)
    qsa_index_norm_rope(idx_q.view(-1, idx_dim), positions, cos_sin, qw, 1e-6, qo.view(-1, idx_dim), heads=q_heads)
    # --- fused
    kc2, vc2, ks2, vs2, ring2, cmp2, qo2 = b

    def run_fused(buffers):
        kc, vc, ks, vs, ring, cmp, qo = buffers
        qsa_index_write_fused(
            k=k, v=v, out_loc=out_loc, k_cache=kc, v_cache=vc, k_scale=ks, v_scale=vs,
            raw_k=raw_k, idx_q=idx_q, q_out=qo, ring=ring, ring_slots=ring_slots,
            token_to_req=token_to_req, cu_seqlens=cu_seqlens, positions=positions,
            cmp_cache=cmp, cmp_rows=cmp_rows, ring_rows=ring_rows, cos_sin=cos_sin,
            k_norm_weight=kw, q_norm_weight=qw, eps=1e-6, ratio=ratio,
        )

    run_fused(b)
    torch.cuda.synchronize()
    assert torch.equal(kc.view(torch.uint8), kc2.view(torch.uint8)) and torch.equal(vc.view(torch.uint8), vc2.view(torch.uint8))
    assert torch.equal(ks, ks2) and torch.equal(vs, vs2)
    assert torch.equal(ring, ring2)
    # only the closing rows' slab entries are defined (several non-closing rows share the
    # scratch row and the last writer wins in BOTH kernels); bf16 1-ulp tolerance for the
    # fp32 reduction-order difference of the rms
    live_rows = (out_loc[closing] // ratio).long()
    torch.testing.assert_close(cmp2[live_rows].float(), cmp[live_rows].float(), rtol=1.6e-2, atol=2e-2)
    torch.testing.assert_close(qo2.float(), qo.float(), rtol=1.6e-2, atol=2e-2)
    # the closing rows landed in the slab, the others only in the scratch row
    if bool(closing.any()):
        assert (cmp2[out_loc[closing] // ratio].float().abs().sum(1) > 0).all()

    # The direct lane permutation must match the former matrix/reduction exactly.
    monkeypatch.setattr(fused, "_DIRECT_PERMUTE", False)
    kc3, vc3, ks3, vs3, ring3, cmp3, qo3 = c
    run_fused(c)
    torch.cuda.synchronize()
    graphs = []
    for direct, buffers in ((False, c), (True, b)):
        monkeypatch.setattr(fused, "_DIRECT_PERMUTE", direct)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run_fused(buffers)
        graphs.append(graph)
    for _ in range(5):
        k.normal_()
        v.normal_()
        raw_k.normal_()
        idx_q.normal_()
        for graph in graphs:
            graph.replay()
        for old, new in ((kc3.view(torch.uint8), kc2.view(torch.uint8)),
                         (vc3.view(torch.uint8), vc2.view(torch.uint8)),
                         (ks3, ks2), (vs3, vs2), (ring3, ring2),
                         (cmp3[live_rows], cmp2[live_rows]), (qo3, qo2)):
            torch.testing.assert_close(new, old, rtol=0, atol=0)


@requires_cuda
@pytest.mark.parametrize("T", [1, 4])
def test_fused_qk_norm_rope_matches_norm_plus_rotary(T):
    """One-launch q/k (1+w) RMSNorm + NeoX rope vs GemmaPlusOneRMSNorm.forward_inplace x2 +
    RotaryEmbedding.forward (flashinfer or the Triton fallback)."""
    from freetoken.kernel.triton.qsa import qsa_qk_norm_rope_fused
    from freetoken.layers.norm import GemmaPlusOneRMSNorm
    from freetoken.layers.rotary import get_rope
    from freetoken.utils import torch_dtype

    torch.manual_seed(T)
    hq, hk, d, rot = 24, 2, 256, 64
    with torch.device("meta"), torch_dtype(torch.bfloat16):
        qn = GemmaPlusOneRMSNorm(d, 1e-6); kn = GemmaPlusOneRMSNorm(d, 1e-6)
    qn.load_state_dict({"weight": (torch.randn(d, device=DEV) * 0.1).to(torch.bfloat16)})
    kn.load_state_dict({"weight": (torch.randn(d, device=DEV) * 0.1).to(torch.bfloat16)})
    with torch.device(DEV):
        rope = get_rope(head_dim=d, rotary_dim=rot, max_position=4096, base=10000.0)
    q0 = torch.randn(T, hq, d, device=DEV, dtype=torch.bfloat16)
    k0 = torch.randn(T, hk, d, device=DEV, dtype=torch.bfloat16)
    pos = torch.arange(100, 100 + T, device=DEV, dtype=torch.int32)
    q1, k1 = q0.clone(), k0.clone()
    qn.forward_inplace(q1); kn.forward_inplace(k1)
    rope.forward(pos, q1.view(-1, hq * d), k1.view(-1, hk * d))
    q2, k2 = q0.clone(), k0.clone()
    qsa_qk_norm_rope_fused(q2, k2, pos, rope._cos_sin_cache, qn.weight, kn.weight, qn.eps)
    torch.cuda.synchronize()
    torch.testing.assert_close(q2.float(), q1.float(), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(k2.float(), k1.float(), rtol=1e-2, atol=1e-2)


@requires_cuda
def test_plan_index_writes_kernel_matches_torch_plan():
    from freetoken.kernel.triton.qsa import qsa_plan_index_writes

    ratio, cap, scratch_base = 4, 8, 1000
    for start, T in [(0, 1), (5, 4), (62, 4), (7, 3)]:
        out_loc = torch.arange(start, start + T, device=DEV, dtype=torch.int32)
        positions = out_loc + 3
        token_to_req = torch.zeros(T, device=DEV, dtype=torch.int32)
        ring_slots = torch.tensor([2], device=DEV, dtype=torch.int32)
        cu = torch.tensor([0, T], device=DEV, dtype=torch.int32)
        cmp, ring = qsa_plan_index_writes(out_loc, positions, token_to_req, ring_slots, cu,
                                          ratio=ratio, ring_capacity=cap, scratch_base=scratch_base)
        # torch reference (QSASparseAttnBackend._plan_index_writes)
        ol = out_loc.long(); po = positions.long(); rows = torch.arange(T, device=DEV)
        slots = ring_slots.long()[token_to_req.long()]
        closing = ol % ratio == ratio - 1
        cmp_ref = torch.where(closing, ol // ratio, scratch_base + slots).int()
        ends = cu.long()[token_to_req.long() + 1]
        keep = rows >= ends - cap
        ring_ref = torch.where(keep, slots * cap + po % cap, torch.full_like(slots, -1)).int()
        assert torch.equal(cmp, cmp_ref) and torch.equal(ring, ring_ref), (start, T)


@requires_cuda
@pytest.mark.parametrize("T,start", [(1, 100), (2, 100), (3, 100), (4, 100), *HIGH_POSITION_CASES])
def test_parallel_qk_heads_match_serial_graph_replay(T, start, monkeypatch):
    import freetoken.kernel.triton.qsa.fused_decode as fused

    torch.manual_seed(T)
    q = torch.randn(T, 24, 256, device=DEV, dtype=torch.bfloat16)
    k = torch.randn(T, 2, 256, device=DEV, dtype=torch.bfloat16)
    qw = torch.randn(256, device=DEV, dtype=torch.bfloat16) * 0.1
    kw = torch.randn_like(qw) * 0.1
    pos = torch.arange(start, start + T, device=DEV, dtype=torch.int32)
    cos_sin = _rope_cache(max(1024, start + T), 64)
    graphs, outputs = [], []
    for parallel in (False, True):
        monkeypatch.setattr(fused, "_PARALLEL_HEADS", parallel)
        for _ in range(2):
            fused.qsa_qk_norm_rope_fused(q.clone(), k.clone(), pos, cos_sin, qw, kw, 1e-6)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            qo, ko = q.clone(), k.clone()
            fused.qsa_qk_norm_rope_fused(qo, ko, pos, cos_sin, qw, kw, 1e-6)
        graphs.append(graph)
        outputs.append((qo, ko))
    for _ in range(5):
        q.normal_()
        k.normal_()
        for graph in graphs:
            graph.replay()
        for old, new in zip(outputs[0], outputs[1]):
            torch.testing.assert_close(new, old, rtol=0, atol=0)
