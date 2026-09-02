"""Per-tensor FP8 linear: the W8A16 kernels against the dequant reference, and the W8A8
(torch._scaled_mm) path against a W8A8 reference.

Which of the two runs is fixed by the deployment -- an ``input_scale`` in the checkpoint plus
sm_89+ -- and never by M, so each is checked against the reference matching its own contract:
W8A16 keeps the activation in bf16 and matches the exact dequant reference, W8A8 quantizes it
to fp8 and is held to a reference that applies the same quantization.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest
import torch

if not torch.cuda.is_available():  # pragma: no cover
    pytest.skip("CUDA required", allow_module_level=True)

from freetoken.kernel.triton.e4m3_compat import e4m3_native

DEV = "cuda"
FP8 = torch.float8_e4m3fn


def _quant_parts(part_rows: list[int], K: int, seed: int = 0):
    """A fused per-tensor-FP8 weight: each part quantized under its own scalar, exactly how
    modelopt stores q/k/v (and how the loader concatenates them into a per-row vector)."""
    torch.manual_seed(seed)
    N = sum(part_rows)
    wf = torch.randn(N, K, device=DEV) * 0.05
    rows, qs, scales = 0, [], []
    for p in part_rows:
        block = wf[rows : rows + p]
        s = block.abs().max() / 448.0
        qs.append((block / s).clamp(-448, 448).to(FP8))
        scales.append(s.expand(p))
        rows += p
    return torch.cat(qs, 0), torch.cat(scales).contiguous().float()


def _dequant(w8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return w8.to(torch.float32) * scale[:, None]


# 1 = split-K GEMV; 2..16 = decode batch (the CUDA-graph ladder); 64/300 = prefill.
@pytest.mark.parametrize("M", [1, 2, 4, 8, 16, 64, 300])
# (5120, 14336) and (5120, 16384) are Qwen3.5-27B's fused qkv_proj / in_proj_qkvz;
# (1024, 6144) is a standalone o_proj shape; 6112 leaves a k-mask tail.
@pytest.mark.parametrize("K,part_rows", [
    (5120, [12288, 1024, 1024]),
    (1024, [6144]),
    (6112, [512, 128]),
])
def test_w8a16_matches_dequant_reference(M: int, K: int, part_rows: list[int]):
    """Without an ``input_scale``: M=1 (and every M where row-wise ``_scaled_mm`` is
    unavailable) runs the W8A16 kernels, activation exact, against the dequant reference;
    on sm_89+ with row-wise ``_scaled_mm`` every M>1 runs per-token dynamic W8A8 and is held
    to the reference that applies the same activation quantization."""
    from freetoken.kernel.triton.fp8_pertensor_linear import (
        _ROWS_GEMV_MAX_M,
        dynamic_w8a8_ok,
        fp8_pertensor_linear,
    )

    w8, scale = _quant_parts(part_rows, K, seed=M)
    x = torch.randn(M, K, device=DEV, dtype=torch.bfloat16)
    y = fp8_pertensor_linear(x, w8, scale)
    if M > _ROWS_GEMV_MAX_M and dynamic_w8a8_ok():
        xs = (x.float().abs().amax(dim=1, keepdim=True) / 448.0).clamp_min(1e-12)
        xq = (x.float() / xs).clamp(-448, 448).to(FP8).to(torch.float32) * xs
        y_ref = xq @ _dequant(w8, scale).t()
        rel = ((y.float() - y_ref).norm() / y_ref.norm()).item()
        assert rel < 1e-2, rel
        return
    y_ref = (x.float() @ _dequant(w8, scale).t()).to(torch.bfloat16)
    rel = (y.float() - y_ref.float()).abs().max() / y_ref.float().abs().max().clamp(min=1e-6)
    assert rel.item() < 2e-2, rel.item()


@pytest.mark.skipif(not e4m3_native(), reason="torch._scaled_mm needs sm_89+")
@pytest.mark.parametrize("M", [1, 2, 4, 16, 64])
@pytest.mark.parametrize("part_rows,uniform", [
    ([12288, 1024, 1024], False),  # fused -> piecewise-constant scale -> row-wise
    ([6144], True),                # standalone -> one scalar -> tensor-wise
])
def test_w8a8_matches_w8a8_reference(M: int, part_rows: list[int], uniform: bool):
    from freetoken.kernel.triton.fp8_pertensor_linear import fp8_pertensor_linear

    K = 2048
    w8, scale = _quant_parts(part_rows, K, seed=M)
    x = torch.randn(M, K, device=DEV, dtype=torch.bfloat16)
    input_scale = (x.abs().max().float() / 448.0).reshape(())

    y = fp8_pertensor_linear(x, w8, scale, None, input_scale, uniform)

    xq = (x.float() / input_scale).clamp(-448, 448).to(FP8)
    y_ref = (xq.to(torch.float32) * input_scale) @ _dequant(w8, scale).t()
    rel = ((y.float() - y_ref).norm() / y_ref.norm()).item()
    assert rel < 1e-2, rel


@pytest.mark.skipif(not e4m3_native(), reason="torch._scaled_mm needs sm_89+")
def test_batch_size_does_not_change_the_numeric_scheme():
    """A deployment that can run W8A8 must run it at every M, so that a reply reproduces at
    bs=1 regardless of how many other requests shared its forward. Feeding the same row alone
    and as part of a batch must therefore agree bit-for-bit."""
    from freetoken.kernel.triton.fp8_pertensor_linear import fp8_pertensor_linear

    K, part_rows = 2048, [1024, 256]
    w8, scale = _quant_parts(part_rows, K)
    x = torch.randn(8, K, device=DEV, dtype=torch.bfloat16)
    input_scale = (x.abs().max().float() / 448.0).reshape(())

    batched = fp8_pertensor_linear(x, w8, scale, None, input_scale, False)
    alone = fp8_pertensor_linear(x[:1], w8, scale, None, input_scale, False)
    assert torch.equal(alone, batched[:1])


def test_layer_load_marks_uniform_scale_and_optional_input_scale():
    from freetoken.kernel.triton.fp8_pertensor_linear import (
        Fp8PerTensorColMerged,
        Fp8PerTensorLinear,
    )

    K = 512
    w8, scale = _quant_parts([256, 64, 64], K)
    merged = Fp8PerTensorColMerged(K, [256, 64, 64])
    merged.load_state_dict({
        "weight": w8, "weight_scale": scale, "input_scale": torch.tensor(0.01, device=DEV),
    })
    assert merged._uniform_scale is False
    assert merged.input_scale is not None

    single = Fp8PerTensorLinear(K, 384)
    flat = scale[:1].expand(384).contiguous()
    single.load_state_dict({"weight": w8, "weight_scale": flat})  # no input_scale
    assert single._uniform_scale is True
    assert single.input_scale is None

    # a reload must not trip over the input_scale it kept from the first load
    single.load_state_dict({"weight": w8, "weight_scale": flat})
    assert single.input_scale is None


@pytest.mark.skipif(not e4m3_native(), reason="torch._scaled_mm needs sm_89+")
@pytest.mark.parametrize("M", [1, 4, 64, 300])
def test_per_part_path_matches_rowwise(M: int, monkeypatch):
    """Where row-wise ``_scaled_mm`` is unsafe a fused projection runs one tensor-wise GEMM per
    part instead. Same scheme, so the two paths agree up to accumulation order (~7e-4)."""
    import freetoken.kernel.triton.fp8_pertensor_linear as mod

    K, part_rows = 2048, [1024, 256, 256]
    w8, scale = _quant_parts(part_rows, K, seed=M)
    x = torch.randn(M, K, device=DEV, dtype=torch.bfloat16)
    input_scale = (x.abs().max().float() / 448.0).reshape(())

    monkeypatch.setattr(mod, "rowwise_scaled_mm_ok", lambda: True)
    y_row = mod.fp8_pertensor_linear(x, w8, scale, None, input_scale, False)
    monkeypatch.setattr(mod, "rowwise_scaled_mm_ok", lambda: False)
    y_part = mod.fp8_pertensor_linear(x, w8, scale, None, input_scale, False)
    rel = ((y_part.float() - y_row.float()).norm() / y_row.float().norm()).item()
    assert rel < 2e-3, rel


@pytest.mark.skipif(not e4m3_native(), reason="torch._scaled_mm needs sm_89+")
def test_fused_layer_forward_on_a_side_stream_completes():
    """Regression for #182 / #72 / #220: on sm_89 with torch < 2.12 a fused FP8 projection's
    row-wise ``_scaled_mm`` issued from a non-default stream stalls the GPU (PyTorch's
    CUTLASS row-wise kernel ignored the current stream; fixed upstream in 2.12). The layer
    must take a path that completes on every supported build. Runs in a subprocess so a
    stall fails the test instead of hanging the session."""
    script = textwrap.dedent("""
        import os, time, torch
        from freetoken.kernel.triton.fp8_pertensor_linear import FP8, Fp8PerTensorColMerged

        torch.manual_seed(0)
        K, parts = 2048, [8192, 512, 512]  # a prefill-sized fused qkv
        w8 = (torch.randn(sum(parts), K, device="cuda") * 8).clamp(-448, 448).to(FP8)
        scale = torch.cat([torch.full((p,), 0.01 * (i + 1), device="cuda")
                           for i, p in enumerate(parts)])
        layer = Fp8PerTensorColMerged(K, parts)
        layer.load_state_dict({"weight": w8, "weight_scale": scale,
                               "input_scale": torch.tensor(0.02, device="cuda")})
        x = torch.randn(2010, K, device="cuda", dtype=torch.bfloat16)  # #182 shape
        torch.cuda.synchronize()

        stream = torch.cuda.Stream()
        events = []
        with torch.cuda.stream(stream):
            for _ in range(128):
                layer.forward(x)
                ev = torch.cuda.Event()
                ev.record(stream)
                events.append(ev)
        deadline = time.monotonic() + 30
        done = 0
        while time.monotonic() < deadline:
            while done < len(events) and events[done].query():
                done += 1
            if done == len(events):
                print("completed", done, flush=True)
                os._exit(0)
            time.sleep(0.05)
        print("stalled at", done, "of", len(events), flush=True)
        os._exit(124)  # a normal exit would wait on the stuck kernel
    """)
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, f"rc={proc.returncode}\n{proc.stdout}\n{proc.stderr[-2000:]}"


# --dense-quant fp8 (qwen4_exp): genuinely per-channel scales (N distinct values), the real
# Qwen3.8-Flash-Next shapes, M=1 (W8A16 GEMV) and the speculative-verify M=3/4 plus prefill.
_DENSE_SHAPES = [
    (2560, 16384),   # GDN in_proj_qkvz
    (6144, 2560),    # GDN out_proj / QSA o_proj
    (2560, 13312),   # QSA qkv_proj
    (10240, 336),    # HC down_block_inject (12 zero pad rows)
    (320, 10240),    # HC up
    (2560, 1280),    # shared expert gate_up
    (640, 2560),     # shared expert down
    (2560, 16384),   # lm_head slice (V=248320 is the same kernel path)
]


def test_quantize_fp8_per_row_roundtrip():
    from freetoken.kernel.triton.fp8_pertensor_linear import quantize_fp8_per_row

    torch.manual_seed(0)
    w = torch.randn(336, 10240, device=DEV, dtype=torch.bfloat16) * 0.05
    w[-12:] = 0  # the HC merged GEMM's zero pad rows
    q, s = quantize_fp8_per_row(w, chunk_rows=100)  # chunking must be exact per row
    assert q.dtype == FP8 and s.dtype == torch.float32 and s.shape == (336,)
    assert torch.isfinite(s).all() and (s > 0).all()
    assert torch.equal(q[-12:].to(torch.float32), torch.zeros(12, 10240, device=DEV))
    back = _dequant(q, s)
    amax = w.float().abs().amax(dim=1, keepdim=True)
    # e4m3: 3 mantissa bits -> |err| <= |w|/16 (+ a tiny absolute floor near zero)
    assert ((back - w.float()).abs() <= w.float().abs() / 16 + amax * 2 ** -9).all()
    q2, s2 = quantize_fp8_per_row(w)
    assert torch.equal(q.to(torch.float32), q2.to(torch.float32)) and torch.equal(s, s2)


@pytest.mark.parametrize("M", [1, 3, 4, 64])
@pytest.mark.parametrize("K,N", _DENSE_SHAPES)
def test_per_channel_scale_matches_reference(M: int, K: int, N: int):
    from freetoken.kernel.triton.fp8_pertensor_linear import (
        _ROWS_GEMV_MAX_M,
        dynamic_w8a8_ok,
        fp8_pertensor_linear,
        quantize_fp8_per_row,
    )

    torch.manual_seed(K + N + M)
    w = torch.randn(N, K, device=DEV, dtype=torch.bfloat16) * (0.02 + 0.2 * torch.rand(N, 1, device=DEV))
    w8, scale = quantize_fp8_per_row(w)
    assert torch.unique(scale).numel() > N // 2  # genuinely per channel
    x = torch.randn(M, K, device=DEV, dtype=torch.bfloat16)
    y = fp8_pertensor_linear(x, w8, scale)
    exact = x.float() @ _dequant(w8, scale).t()
    if M <= _ROWS_GEMV_MAX_M or not dynamic_w8a8_ok():
        # W8A16 (M=1 GEMV / decode rows GEMV): activation exact
        rel = ((y.float() - exact).norm() / exact.norm()).item()
        assert rel < 1e-2, rel
    else:
        # per-token dynamic W8A8 through row-wise _scaled_mm: hold to the matching reference
        xs = (x.float().abs().amax(dim=1, keepdim=True) / 448.0).clamp_min(1e-12)
        xq = (x.float() / xs).clamp(-448, 448).to(FP8).to(torch.float32) * xs
        ref = xq @ _dequant(w8, scale).t()
        rel = ((y.float() - ref).norm() / ref.norm()).item()
        assert rel < 1e-2, rel
        loose = ((y.float() - exact).norm() / exact.norm()).item()
        assert loose < 3e-2, loose


def test_per_channel_layer_skips_segment_walk():
    """A load-time-quantized layer has N distinct scales and no input_scale: load_state_dict
    must not build the O(N) segment list (only the static-input_scale path reads it)."""
    from freetoken.kernel.triton.fp8_pertensor_linear import Fp8PerTensorLinear, quantize_fp8_per_row
    from freetoken.utils import torch_dtype

    N, K = 4096, 512
    w8, scale = quantize_fp8_per_row(torch.randn(N, K, device=DEV, dtype=torch.bfloat16))
    with torch.device("meta"), torch_dtype(torch.bfloat16):
        layer = Fp8PerTensorLinear(K, N)
    layer.load_state_dict({"weight": w8, "weight_scale": scale})
    assert layer._scale_segments is None and layer._uniform_scale is False
    x = torch.randn(3, K, device=DEV, dtype=torch.bfloat16)
    y = layer.forward(x)
    assert y.shape == (3, N) and y.dtype == torch.bfloat16


@pytest.mark.parametrize("M", [1, 2, 3, 4])
@pytest.mark.parametrize("K,N", [(2560, 96), (2560, 512), (2560, 640), (2560, 2560), (6144, 2560), (10240, 336)])
def test_bf16_rows_gemv_matches_cublas(M: int, K: int, N: int):
    """The small-M bf16 rows GEMV (deep-K bf16 linears at decode) vs F.linear: both
    accumulate in fp32, so only bf16 output rounding separates them."""
    from freetoken.kernel.triton.fp8_pertensor_linear import bf16_rows_gemv, bf16_rows_gemv_applicable

    torch.manual_seed(M * 7 + N)
    w = torch.randn(N, K, device=DEV, dtype=torch.bfloat16) * 0.05
    x = torch.randn(M, K, device=DEV, dtype=torch.bfloat16)
    assert bf16_rows_gemv_applicable(x, w, None)
    y = bf16_rows_gemv(x, w)
    ref = torch.nn.functional.linear(x, w)
    rel = ((y.float() - ref.float()).norm() / ref.float().norm()).item()
    assert rel < 5e-3, rel
    exact = x.float() @ w.float().t()
    assert ((y.float() - exact).norm() / exact.norm()).item() < 5e-3


def test_bf16_rows_gemv_dispatch_in_linear_layer():
    from freetoken.layers import LinearReplicated
    from freetoken.utils import torch_dtype

    with torch.device("meta"), torch_dtype(torch.bfloat16):
        lin = LinearReplicated(2560, 96, has_bias=False)
        shallow = LinearReplicated(320, 4096, has_bias=False)
    w = torch.randn(96, 2560, device=DEV, dtype=torch.bfloat16) * 0.05
    lin.load_state_dict({"weight": w})
    shallow.load_state_dict({"weight": torch.randn(4096, 320, device=DEV, dtype=torch.bfloat16) * 0.05})
    x = torch.randn(3, 2560, device=DEV, dtype=torch.bfloat16)
    y = lin.forward(x)
    # fp32 accumulation both sides; bf16 output rounding may differ by one ulp on a few lanes
    torch.testing.assert_close(y.float(), torch.nn.functional.linear(x, w).float(), rtol=1e-2, atol=3e-2)
    # M > 4 and shallow K keep cuBLAS (identical results)
    x8 = torch.randn(8, 2560, device=DEV, dtype=torch.bfloat16)
    assert torch.equal(lin.forward(x8), torch.nn.functional.linear(x8, w))
    xs = torch.randn(3, 320, device=DEV, dtype=torch.bfloat16)
    assert torch.equal(shallow.forward(xs), torch.nn.functional.linear(xs, shallow.weight))


def _rows_case(kind: str, M: int, fp8: bool, seed: int):
    """Build (args, epilogue) for one rows-GEMV call; shapes mirror Qwen3.8's split-K users."""
    from freetoken.kernel.triton.fp8_pertensor_linear import quantize_fp8_per_row

    torch.manual_seed(seed)
    if kind == "plain_narrow":      # in_proj_ba-like: 6 tiles -> split_k 16
        N, K, epi = 96, 2560, None
    elif kind == "plain_wide":      # out_proj-like: 160 tiles -> split_k 8
        N, K, epi = 2560, 6144, None
    elif kind == "hc_down":         # [lowrank + hc + pad, width]: lora + inject logits
        lowrank, hcn, hc = 384, 4, 4
        N, K = lowrank + hcn + 12, 10240
        epi = ("hc_down", lowrank, hcn, hc)
    elif kind == "hc_gate_mix":     # [hc * hidden, lowrank]: 640 tiles -> split_k 2
        hc, hidden, K = 4, 2560, 384
        N = hc * hidden
        rn = torch.randn(M, N, device=DEV, dtype=torch.bfloat16)
        epi = ("hc_gate_mix", rn, hc, hidden)
    else:
        raise ValueError(kind)
    w = torch.randn(N, K, device=DEV, dtype=torch.bfloat16) * 0.05
    x = torch.randn(M, K, device=DEV, dtype=torch.bfloat16)
    if fp8:
        from freetoken.kernel.triton.fp8_pertensor_linear import e4m3_kernel_view

        q, scale = quantize_fp8_per_row(w)
        return (x, e4m3_kernel_view(q), scale, torch.bfloat16), epi
    return (x, w, None, torch.bfloat16), epi


def _as_tuple(r):
    return r if isinstance(r, tuple) else (r,)


@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("M", [1, 3, 4])
@pytest.mark.parametrize("kind", ["plain_narrow", "plain_wide", "hc_down", "hc_gate_mix"])
def test_rows_gemv_splitk_fixup_bit_identical(kind: str, M: int, fp8: bool, monkeypatch):
    """The in-kernel split-K fixup (last CTA per tile reduces) must reproduce the two-launch
    reduce path bit for bit: same partials, same summation order, same epilogue math."""
    import freetoken.kernel.triton.fp8_pertensor_linear as mod

    args, epi = _rows_case(kind, M, fp8, seed=11 * M + len(kind))
    monkeypatch.setattr(mod, "_ROWS_FIXUP", False)
    ref = _as_tuple(mod._gemv_rows(*args, epilogue=epi))
    monkeypatch.setattr(mod, "_ROWS_FIXUP", True)
    for _ in range(3):  # repeated launches: counters must be re-armed by the finishing CTA
        got = _as_tuple(mod._gemv_rows(*args, epilogue=epi))
        torch.cuda.synchronize()
        assert len(got) == len(ref)
        for g, r in zip(got, ref):
            if r is None:
                assert g is None
                continue
            assert g.shape == r.shape and g.dtype == r.dtype
            assert torch.equal(g, r), (kind, M, fp8, (g.float() - r.float()).abs().max().item())
    for st in mod._fixup_counters.values():
        assert int(st[0].abs().sum().item()) == 0, "tile counters not re-armed"


@pytest.mark.parametrize("kind", ["plain_wide", "hc_down", "hc_gate_mix"])
def test_rows_gemv_splitk_fixup_under_cuda_graph(kind: str, monkeypatch):
    """Captured once, replayed with changing inputs: the counters baked into the graph must
    keep working across replays and the results must match the eager two-launch path."""
    import freetoken.kernel.triton.fp8_pertensor_linear as mod

    M = 3
    args, epi = _rows_case(kind, M, True, seed=99)
    x = args[0]
    monkeypatch.setattr(mod, "_ROWS_FIXUP", True)
    mod._gemv_rows(*args, epilogue=epi)  # warm-up / compile outside capture
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            out = _as_tuple(mod._gemv_rows(*args, epilogue=epi))
    torch.cuda.current_stream().wait_stream(s)
    for rep in range(4):
        x.copy_(torch.randn_like(x) * (1 + rep))
        g.replay()
        torch.cuda.synchronize()
        monkeypatch.setattr(mod, "_ROWS_FIXUP", False)
        ref = _as_tuple(mod._gemv_rows(*args, epilogue=epi))
        monkeypatch.setattr(mod, "_ROWS_FIXUP", True)
        torch.cuda.synchronize()
        for o, r in zip(out, ref):
            if r is None:
                assert o is None
                continue
            assert torch.equal(o, r), (kind, rep, (o.float() - r.float()).abs().max().item())


@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("M", [1, 3])
def test_rows_gemv_split_epilogue(M: int, fp8: bool, monkeypatch):
    """("split", n1, n2): two contiguous outputs equal to the plain GEMV's column slices
    (bit-identical: same partials, same reduce order), fixup on and off."""
    import freetoken.kernel.triton.fp8_pertensor_linear as mod

    args, _ = _rows_case("plain_narrow", M, fp8, seed=5 + M)   # [96, 2560]: in_proj_ba-like
    ref = mod._gemv_rows(*args)
    for flag in (True, False):
        monkeypatch.setattr(mod, "_ROWS_FIXUP", flag)
        b, a = mod._gemv_rows(*args, epilogue=("split", 48, 48))
        torch.cuda.synchronize()
        assert b.is_contiguous() and a.is_contiguous() and b.shape == (M, 48) and a.shape == (M, 48)
        assert torch.equal(b, ref[:, :48]) and torch.equal(a, ref[:, 48:96]), (flag, fp8)


@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("M", [1, 3, 4])
def test_rows_gemv_qsa_qkv_epilogue(M: int, fp8: bool, monkeypatch):
    """("qsa_qkv", num_q, num_kv, hd): q / gate (de-interleaved per head) / k / v contiguous,
    bit-identical to the plain GEMV's slices, on the DIRECT (split_k == 1) geometry."""
    import freetoken.kernel.triton.fp8_pertensor_linear as mod
    from freetoken.kernel.triton.fp8_pertensor_linear import quantize_fp8_per_row
    from freetoken.kernel.triton.e4m3_compat import e4m3_kernel_view

    num_q, num_kv, hd, K = 32, 4, 128, 2560          # N = 2*4096 + 2*512 = 9216 -> 576 tiles (DIRECT)
    N = 2 * num_q * hd + 2 * num_kv * hd
    torch.manual_seed(M)
    w = torch.randn(N, K, device=DEV, dtype=torch.bfloat16) * 0.05
    x = torch.randn(M, K, device=DEV, dtype=torch.bfloat16)
    if fp8:
        q8, sc = quantize_fp8_per_row(w)
        args = (x, e4m3_kernel_view(q8), sc, torch.bfloat16)
    else:
        args = (x, w, None, torch.bfloat16)
    ref = mod._gemv_rows(*args)
    q, gate, k, v = mod._gemv_rows(*args, epilogue=("qsa_qkv", num_q, num_kv, hd))
    torch.cuda.synchronize()
    nq, nk = num_q * hd, num_kv * hd
    qg = ref[:, : 2 * nq].view(M, num_q, 2 * hd)
    assert all(t.is_contiguous() for t in (q, gate, k, v))
    assert torch.equal(q.view(M, num_q, hd), qg[..., :hd])
    assert torch.equal(gate.view(M, num_q, hd), qg[..., hd:])
    assert torch.equal(k, ref[:, 2 * nq: 2 * nq + nk])
    assert torch.equal(v, ref[:, 2 * nq + nk:])


@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("M", [1, 3, 4])
def test_rows_gemv_silu_mul_prologue(M: int, fp8: bool):
    """down(silu(g) * u) with the activation formed inside the GEMV vs the separate act
    kernel + plain GEMV (the activation is rounded to bf16 identically; exp may differ by an
    ulp before rounding)."""
    import freetoken.kernel.triton.fp8_pertensor_linear as mod
    from freetoken.kernel.triton.fp8_pertensor_linear import quantize_fp8_per_row
    from freetoken.kernel.triton.e4m3_compat import e4m3_kernel_view
    from freetoken.layers.activation import silu_and_mul

    inter, H = 1024, 2560
    torch.manual_seed(3 + M)
    w = torch.randn(H, inter, device=DEV, dtype=torch.bfloat16) * 0.05
    gu = torch.randn(M, 2 * inter, device=DEV, dtype=torch.bfloat16)
    if fp8:
        q8, sc = quantize_fp8_per_row(w)
        wk, scale = e4m3_kernel_view(q8), sc
    else:
        wk, scale = w, None
    ref = mod._gemv_rows(silu_and_mul(gu).contiguous(), wk, scale, torch.bfloat16)
    out = mod._gemv_rows(gu, wk, scale, torch.bfloat16, act=("silu_mul", inter))
    torch.cuda.synchronize()
    torch.testing.assert_close(out.float(), ref.float(), rtol=1e-2, atol=2e-2)


@pytest.mark.parametrize("M", [1, 3, 4])
@pytest.mark.parametrize("a_fp8,b_fp8", [(True, False), (True, True), (False, False)])
def test_rows_gemv_direct_pair_matches_separate(M: int, a_fp8: bool, b_fp8: bool):
    """A big DIRECT GEMV + its small partner in one launch == the two separate rows GEMVs
    (bit-identical: same tile math per segment)."""
    from types import SimpleNamespace

    import freetoken.kernel.triton.fp8_pertensor_linear as mod
    from freetoken.kernel.triton.fp8_pertensor_linear import quantize_fp8_per_row

    torch.manual_seed(M + 2 * a_fp8 + 4 * b_fp8)
    K, NA, NB = 2560, 16384, 96
    x = torch.randn(M, K, device=DEV, dtype=torch.bfloat16)

    def layer(N, fp8):
        w = torch.randn(N, K, device=DEV, dtype=torch.bfloat16) * 0.05
        if fp8:
            q, sc = quantize_fp8_per_row(w)
            return SimpleNamespace(weight=q, weight_scale=sc, bias=None, input_scale=None)
        return SimpleNamespace(weight=w, weight_scale=None, bias=None, input_scale=None)

    la, lb = layer(NA, a_fp8), layer(NB, b_fp8)
    res = mod.rows_gemv_direct_pair(la, x, None, lb, ("split", 48, 48))
    assert res is not None
    out_a, (b1, b2) = res
    ref_a = mod.rows_gemv_with_epilogue(la, x, None) if False else mod._gemv_rows(
        x, mod.e4m3_kernel_view(la.weight) if a_fp8 else la.weight, la.weight_scale, torch.bfloat16)
    ref_b = mod._gemv_rows(
        x, mod.e4m3_kernel_view(lb.weight) if b_fp8 else lb.weight, lb.weight_scale, torch.bfloat16)
    torch.cuda.synchronize()
    assert torch.equal(out_a, ref_a)
    assert torch.equal(b1, ref_b[:, :48]) and torch.equal(b2, ref_b[:, 48:96])
    # qsa_qkv epilogue on A with the partner
    num_q, num_kv, hd = 48, 4, 128          # N = 13312 -> 832 tiles: a DIRECT shape like qkv_proj
    la2 = layer(2 * num_q * hd + 2 * num_kv * hd, a_fp8)
    res = mod.rows_gemv_direct_pair(la2, x, ("qsa_qkv", num_q, num_kv, hd), lb, ("split", 64, 32))
    assert res is not None
    (q, gate, k, v), (b1, b2) = res
    ref = mod._gemv_rows(x, mod.e4m3_kernel_view(la2.weight) if a_fp8 else la2.weight, la2.weight_scale, torch.bfloat16)
    nq, nk = num_q * hd, num_kv * hd
    qg = ref[:, : 2 * nq].view(M, num_q, 2 * hd)
    assert torch.equal(q.view(M, num_q, hd), qg[..., :hd]) and torch.equal(gate.view(M, num_q, hd), qg[..., hd:])
    assert torch.equal(k, ref[:, 2 * nq: 2 * nq + nk]) and torch.equal(v, ref[:, 2 * nq + nk:])
    assert torch.equal(b1, ref_b[:, :64]) and torch.equal(b2, ref_b[:, 64:96])
