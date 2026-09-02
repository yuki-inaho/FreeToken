"""NVFP4 decode expert GEMVs: arithmetic dequant bit-identity, and (later) the tensor-core
per-distinct-expert kernel."""
from __future__ import annotations

import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
DEV = "cuda"


def _banks(slots, n, k, seed):
    """Realistic NVFP4 banks: quantize random bf16 experts (finite scales/globals)."""
    from freetoken.models.qwen4_exp.weight import _quantize_nvfp4_stacked

    torch.manual_seed(seed)
    w = (torch.randn(slots, n, k, device=DEV) * 0.05).to(torch.bfloat16)
    p, s, g = _quantize_nvfp4_stacked(w)
    return p.contiguous(), s.contiguous(), g.contiguous()


@cuda
@pytest.mark.parametrize("M", [1, 3, 4])
def test_per_route_arith_dequant_bit_identical(M, monkeypatch):
    """ARITH_DEQUANT (pair bit trick, 2^14 folded into the block scale, one contiguous
    activation load per K-tile) reproduces the LUT-gather kernel bit for bit on gemm1
    (K=2560) and on gemm2 (K=640, fused silu*mul, router weights, slot -1 zeroing)."""
    import freetoken.moe.fused_nvfp4 as fn
    from freetoken.moe.fused_nvfp4 import _decode_gemm_marlin

    SLOTS, TOPK, H, I = 64, 10, 2560, 640
    gu = _banks(SLOTS, 2 * I, H, 1)
    dn = _banks(SLOTS, H, I, 2)
    torch.manual_seed(M)
    x = torch.randn(M, H, device=DEV, dtype=torch.bfloat16)
    w = torch.rand(M, TOPK, device=DEV)
    ids = torch.randint(0, SLOTS, (M, TOPK), device=DEV, dtype=torch.int32)
    ids[:, 3] = -1  # CPU routes
    ic1_ref = torch.full((M, TOPK, 2 * I), 7.0, device=DEV, dtype=torch.bfloat16)
    ic1 = ic1_ref.clone()
    ic3_ref = torch.empty((M, TOPK, H), device=DEV, dtype=torch.bfloat16)
    ic3 = torch.empty_like(ic3_ref)
    monkeypatch.setattr(fn, "_ARITH_DEQUANT", False)
    _decode_gemm_marlin(x, *gu, ic1_ref, w, ids, False, False)
    _decode_gemm_marlin(ic1_ref.view(-1, 2 * I), *dn, ic3_ref, w, ids, True, True, zero_on_skip=True, fused_act_off=I)
    monkeypatch.setattr(fn, "_ARITH_DEQUANT", True)
    _decode_gemm_marlin(x, *gu, ic1, w, ids, False, False)
    _decode_gemm_marlin(ic1.view(-1, 2 * I), *dn, ic3, w, ids, True, True, zero_on_skip=True, fused_act_off=I)
    torch.cuda.synchronize()
    assert torch.isfinite(ic3_ref.float()).all()
    assert torch.equal(ic1, ic1_ref), (ic1.float() - ic1_ref.float()).abs().max().item()
    assert torch.equal(ic3, ic3_ref), (ic3.float() - ic3_ref.float()).abs().max().item()
    assert torch.equal(ic3[:, 3], torch.zeros_like(ic3[:, 3]))  # skipped route zeroed
    # and both agree with a dequantized reference within bf16 tolerance
    from freetoken.kernel.triton.nvfp4_dequant import dequant_nvfp4

    slots = ids[0, :3].long()
    wd = dequant_nvfp4(gu[0], gu[1], gu[2], slots.to(torch.int32), dtype=torch.bfloat16)  # [3, 2I, H]
    ref = torch.einsum("h,rnh->rn", x[0].float(), wd.float())
    rel = ((ic1[0, :3].float() - ref).norm() / ref.norm()).item()
    assert rel < 2e-2, rel
