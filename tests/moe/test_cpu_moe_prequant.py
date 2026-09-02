"""GPU twin of the CPU executor's nvfp4 W4A8 input quantization (quant_i8_pg16): bit-exact
codes/scales against the C++ reference, and a decode step whose output is bit-identical
with the GPU pre-quant on and off."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _cpu_ref(ex, x_row: torch.Tensor):
    K = x_row.numel()
    q = torch.empty(K, dtype=torch.int8)
    s = torch.empty(K // 16, dtype=torch.float32)
    ex._ext.quant_i8_pg16_ref(x_row.contiguous().data_ptr(), K, q.data_ptr(), s.data_ptr())
    return q, s


@pytest.mark.parametrize("H", [64, 512, 2560])
def test_gpu_prequant_is_bit_exact(H):
    from freetoken.kernel.triton.cpu_moe_prequant import nvfp4_act_quant_i8_pg16
    from freetoken.moe.cpu_executor import CpuMoeExecutor

    from .test_cpu_moe import _make_nvfp4_cache

    cache = _make_nvfp4_cache(1, 4, H, 64)
    ex = CpuMoeExecutor(cache, top_k=2, activation="silu", apply_router_weight_on_input=False,
                        num_threads=2, max_tokens=4, device=torch.device("cuda"))
    if "vnni" not in str(ex.isa).lower():
        pytest.skip("VNNI tier not available on this CPU")
    torch.manual_seed(0)
    rows = [
        torch.randn(H) * 0.3,
        torch.randn(H) * 30.0,
        torch.zeros(H),
        torch.randn(H) * 1e-3,
    ]
    rows.append(rows[0].clone().index_fill_(0, torch.arange(0, H, 16), 0.0))  # zero-heavy groups
    x = torch.stack(rows).to(torch.bfloat16)
    q = torch.empty(x.shape, dtype=torch.int8, device="cuda")
    s = torch.empty(x.shape[0], H // 16, dtype=torch.float32, device="cuda")
    nvfp4_act_quant_i8_pg16(x.cuda(), q, s)
    torch.cuda.synchronize()
    for i in range(x.shape[0]):
        rq, rs = _cpu_ref(ex, x[i])
        assert torch.equal(q[i].cpu(), rq), f"row {i}: codes differ"
        assert torch.equal(s[i].cpu(), rs), f"row {i}: scales differ"


def test_decode_output_identical_with_gpu_prequant(monkeypatch):
    from freetoken.moe.cpu_executor import CpuMoeExecutor

    from .test_cpu_moe import _make_nvfp4_cache

    L, E, H, I, top_k, bs = 2, 8, 512, 256, 2, 3
    device = torch.device("cuda")
    torch.manual_seed(1)
    hidden = torch.randn(bs, H, device=device, dtype=torch.bfloat16)
    ids = torch.randint(0, E, (bs, top_k), device=device, dtype=torch.int32)
    w = torch.rand(bs, top_k, device=device, dtype=torch.float32)

    outs = {}
    for mode in ("1", "0"):
        monkeypatch.setenv("FREETOKEN_CPU_MOE_GPU_PREQUANT", mode)
        cache = _make_nvfp4_cache(L, E, H, I)
        ex = CpuMoeExecutor(cache, top_k=top_k, activation="silu", apply_router_weight_on_input=False,
                            num_threads=4, max_tokens=bs, device=device)
        if "vnni" not in str(ex.isa).lower():
            pytest.skip("VNNI tier not available on this CPU")
        assert ex._gpu_prequant_i8 == (mode == "1")
        outs[mode] = ex.decode(1, hidden, w, ids).clone()
        torch.cuda.synchronize()
    assert torch.equal(outs["1"], outs["0"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("T", [1, 3, 4])
def test_prequant_with_fused_shared_gate(T):
    """GATE variant (row-wide programs): q/s bit-identical to the plain launch, and the
    fused sigmoid(x . w) matches shared_gate_sigmoid to fp32 reduction-order noise."""
    from freetoken.kernel.triton.cpu_moe_prequant import nvfp4_act_quant_i8_pg16
    from freetoken.kernel.triton.moe_shared_gate import shared_gate_sigmoid

    H = 2560
    torch.manual_seed(T)
    x = (torch.randn(T, H, device="cuda") * 0.7).to(torch.bfloat16)
    w = (torch.randn(H, device="cuda") * 0.05).to(torch.bfloat16)
    q0 = torch.empty(T, H, dtype=torch.int8, device="cuda"); s0 = torch.empty(T, H // 16, device="cuda")
    q1 = torch.empty_like(q0); s1 = torch.empty_like(s0); g = torch.empty(T, device="cuda")
    nvfp4_act_quant_i8_pg16(x, q0, s0)
    nvfp4_act_quant_i8_pg16(x, q1, s1, w, g)
    torch.cuda.synchronize()
    assert torch.equal(q0, q1) and torch.equal(s0, s1)
    torch.testing.assert_close(g, shared_gate_sigmoid(x, w), rtol=1e-5, atol=1e-6)
