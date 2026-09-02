"""Fused MoE finalize (sum-reduce + shared gate mul-add + hyper-connection combine)."""
from __future__ import annotations

import pytest
import torch

from freetoken.kernel.triton.hc import hc_combine
from freetoken.kernel.triton.moe_finalize import MoeCombineEpilogue, moe_finalize_combine
from freetoken.kernel.triton.moe_shared_gate import shared_gate_mul_add
from freetoken.kernel import moe_sum_reduce_triton

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
DEV = "cuda"


def _reference_fp32(ic3, ids, addend, shared, gate, residual, inject, hc):
    M, topk, H = ic3.shape
    x = ic3.float()
    if ids is not None:  # skipped rows may hold garbage/NaN: mask, don't multiply by 0
        x = torch.where((ids >= 0)[..., None], x, torch.zeros_like(x))
    y = x.sum(1)
    if addend is not None:
        y = y + addend.float()
    if shared is not None:
        y = y + gate[:, None] * shared.float()
    inj = 2.0 * torch.sigmoid(inject.float()[:, :hc] / hc)          # [M, hc]
    res = residual.float().view(M, hc, H)
    return (res + y[:, None, :] * inj[:, :, None]).view(M, hc * H)


def _reference_unfused(ic3, ids, addend, shared, gate, residual, inject, hc):
    """The three separate kernels the fused launch replaces (bf16 between them)."""
    M, topk, H = ic3.shape
    routed = torch.empty((M, H), dtype=ic3.dtype, device=ic3.device)
    moe_sum_reduce_triton(ic3, routed, addend=addend, route_ids=ids)
    y = shared_gate_mul_add(routed, shared, gate) if shared is not None else routed
    return hc_combine(residual, y, inject[:, :hc].contiguous(), hc)


@cuda
@pytest.mark.parametrize("M", [1, 3, 4])
@pytest.mark.parametrize("skip_neg,addend,shared", [(False, False, True), (True, True, True), (False, True, False)])
def test_moe_finalize_combine_matches_reference(M, skip_neg, addend, shared):
    torch.manual_seed(M + 10 * skip_neg + 100 * addend)
    topk, H, hc = 10, 2560, 4
    ic3 = torch.randn(M, topk, H, device=DEV, dtype=torch.bfloat16) * 0.1
    ids = None
    if skip_neg:
        ids = torch.randint(0, 4000, (M, topk), device=DEV, dtype=torch.int32)
        ids[:, ::3] = -1
        ic3[:, ::3] = float("nan")   # unwritten rows must not leak
    ad = (torch.randn(M, H, device=DEV, dtype=torch.bfloat16) * 0.1) if addend else None
    sh = (torch.randn(M, H, device=DEV, dtype=torch.bfloat16) * 0.5) if shared else None
    g = torch.rand(M, device=DEV, dtype=torch.float32) if shared else None
    R = torch.randn(M, hc * H, device=DEV, dtype=torch.bfloat16)
    inj = torch.randn(M, hc, device=DEV, dtype=torch.bfloat16) * 2
    out = moe_finalize_combine(ic3, ids, ad, sh, g, R, inj, hc)
    torch.cuda.synchronize()
    assert out.shape == R.shape and out.dtype == R.dtype
    ref32 = _reference_fp32(ic3, ids, ad, sh, g, R, inj, hc)
    assert torch.isfinite(out.float()).all()
    torch.testing.assert_close(out.float(), ref32, rtol=1e-2, atol=2e-2)
    unf = _reference_unfused(ic3, ids, ad, sh, g, R, inj, hc)
    torch.testing.assert_close(out.float(), unf.float(), rtol=1e-2, atol=3e-2)


@cuda
def test_moe_combine_epilogue_apply_sets_result():
    M, topk, H, hc = 3, 10, 2560, 4
    ic3 = torch.randn(M, topk, H, device=DEV, dtype=torch.bfloat16)
    R = torch.randn(M, hc * H, device=DEV, dtype=torch.bfloat16)
    inj = torch.randn(M, hc, device=DEV, dtype=torch.bfloat16)
    ep = MoeCombineEpilogue(None, None, R, inj, hc)
    assert ep.result is None
    out = ep.apply(ic3, None, None)
    assert out is ep.result and out.shape == R.shape
    torch.testing.assert_close(out.float(), _reference_fp32(ic3, None, None, None, None, R, inj, hc), rtol=1e-2, atol=2e-2)


@cuda
def test_moe_finalize_combine_under_cuda_graph():
    M, topk, H, hc = 3, 10, 2560, 4
    ic3 = torch.randn(M, topk, H, device=DEV, dtype=torch.bfloat16)
    sh = torch.randn(M, H, device=DEV, dtype=torch.bfloat16)
    g = torch.rand(M, device=DEV)
    R = torch.randn(M, hc * H, device=DEV, dtype=torch.bfloat16)
    inj = torch.randn(M, hc, device=DEV, dtype=torch.bfloat16)
    moe_finalize_combine(ic3, None, None, sh, g, R, inj, hc)
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=s):
            out = moe_finalize_combine(ic3, None, None, sh, g, R, inj, hc)
    torch.cuda.current_stream().wait_stream(s)
    for _ in range(3):
        ic3.copy_(torch.randn_like(ic3)); R.copy_(torch.randn_like(R))
        graph.replay(); torch.cuda.synchronize()
        torch.testing.assert_close(out.float(), _reference_fp32(ic3, None, None, sh, g, R, inj, hc), rtol=1e-2, atol=2e-2)
