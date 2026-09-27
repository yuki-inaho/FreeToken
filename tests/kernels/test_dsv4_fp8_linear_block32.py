"""Block-32 FP8 linear and the e4m3-scaled FP4 round-trip against pure-torch references.

DeepSeek-V4.1 quantizes its dense layers with 32x32 weight blocks and per-32 activation
blocks (0731 used 128), and its compressed KV with an e2m1 round-trip whose per-16 scale is
an e4m3 value rather than a power of two. The Triton kernels take a ``block`` parameter for
the first and a new entry point for the second; the references here are written from the
reference ``inference/kernel.py`` (``act_quant_kernel(round_scale=True)``,
``fp8_gemm_kernel(block_size=32)``, ``fp4_quant_kernel(scale_dtype=FP8)``) in plain torch
so they share no code with the kernels:

- the activation scale's ``ceil(log2)`` comes from ``torch.frexp`` (the kernel uses IEEE
  bit tricks), the e4m3 cast is torch's ``.to(float8_e4m3fn)``;
- the weight is dequantized whole (``w * 2**(scale-127)`` expanded per block) and the
  matmul runs in float64, so the only error left in the comparison is the kernel's own
  fp32 accumulation order and the final bf16 rounding;
- the e2m1 rounding is a nearest-with-ties-to-even-index search over the 8-value grid,
  not the kernel's comparison ladder.

Block 128 is checked against the same reference so the 0731 path is proven unchanged by
the generalization, and N = 32 / 2304 exercise the tiles that the old ``N % 128`` assert
forbade.

``FREETOKEN_TEST_DEVICE=cpu`` runs the same tests under Triton's interpreter
(``TRITON_INTERPRET=1``, which needs ``FREETOKEN_FORCE_E4M3_EMU=1`` because the
interpreter has no fp8 type): the emulated-e4m3 branch of every kernel, checked without
a GPU.
"""

from __future__ import annotations

import math
import os

import pytest
import torch

from freetoken.kernel.triton.dsv4.fp8_linear import (
    block_fp8_linear,
    fp4_act_quant_e4m3_inplace,
)

DEV = os.environ.get("FREETOKEN_TEST_DEVICE", "cuda")
pytestmark = pytest.mark.skipif(DEV == "cuda" and not torch.cuda.is_available(), reason="needs CUDA")

FP8 = torch.float8_e4m3fn
FP8_MAX = 448.0


# ----------------------------------------------------------------------------------------
# References
# ----------------------------------------------------------------------------------------
def _ceil_log2_pow2(v: torch.Tensor) -> torch.Tensor:
    """``2 ** ceil(log2(v))`` for ``v > 0`` via frexp: ``v = m * 2**e`` with ``m`` in
    [0.5, 1), so ``ceil(log2 v)`` is ``e - 1`` exactly when ``m == 0.5`` and ``e``
    otherwise. Independent of the kernel's exponent-field arithmetic."""
    m, e = torch.frexp(v)
    e = torch.where(m == 0.5, e - 1, e)
    return torch.ldexp(torch.ones_like(v), e)


def ref_act_quant(x: torch.Tensor, block: int) -> torch.Tensor:
    """Reference ``act_quant(x, block, scale_fmt='ue8m0')`` dequantized back to fp32:
    per-``block`` ``s = 2**ceil(log2(max(amax, 1e-4) / 448))``, ``e4m3(clamp(x/s))``."""
    M, K = x.shape
    xb = x.float().reshape(M, K // block, block)
    amax = xb.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    # the reference multiplies by the fp32 reciprocal (``fast_round_scale``); keep the
    # constant in fp32 so an amax exactly on 448 * 2^k lands on the same side of the ceil
    s = _ceil_log2_pow2(amax * torch.tensor(1.0 / FP8_MAX, dtype=torch.float32, device=x.device))
    q = (xb / s).clamp(-FP8_MAX, FP8_MAX).to(FP8).float()
    return (q * s).reshape(M, K)


def ref_weight_dequant(w: torch.Tensor, scale_codes: torch.Tensor, block: int) -> torch.Tensor:
    """``w.float() * 2**(code-127)`` with the ``[ceil(N/b), K/b]`` codes expanded per
    ``block x block`` tile and cropped to ``[N, K]``."""
    N, K = w.shape
    s = torch.exp2(scale_codes.float() - 127.0)
    s = s.repeat_interleave(block, 0).repeat_interleave(block, 1)[:N, :K]
    return w.float() * s


def ref_linear(x: torch.Tensor, w: torch.Tensor, codes: torch.Tensor, block: int) -> torch.Tensor:
    a = ref_act_quant(x, block).double()
    wd = ref_weight_dequant(w, codes, block).double()
    return a @ wd.T


def make_weight(N: int, K: int, block: int, gen: torch.Generator):
    """Random fp8 weight + e8m0 block codes: quantize a normal matrix per ``block x
    block`` tile with a power-of-two scale (as the checkpoint's quantizer did), so
    the codes span a realistic range instead of a single value."""
    w = torch.randn(N, K, device=DEV, generator=gen)
    w = w * torch.exp2(torch.randint(-6, 6, (N, 1), device=DEV, generator=gen).float())
    nb, kb = -(-N // block), K // block
    pad = torch.zeros(nb * block, K, device=DEV)
    pad[:N] = w
    tiles = pad.reshape(nb, block, kb, block)
    amax = tiles.abs().amax(dim=(1, 3)).clamp_min(1e-4)            # [nb, kb]
    s = _ceil_log2_pow2(amax / FP8_MAX)
    codes = (torch.log2(s) + 127).round().to(torch.uint8)
    q = (tiles / s[:, None, :, None]).clamp(-FP8_MAX, FP8_MAX)
    w_fp8 = q.reshape(nb * block, K)[:N].to(FP8)
    return w_fp8.contiguous(), codes.view(torch.float8_e8m0fnu).contiguous()


# ----------------------------------------------------------------------------------------
# block_fp8_linear
# ----------------------------------------------------------------------------------------
SHAPES_32 = [
    (32, 64),        # single 32-N scale row, single 128-N tile (masked)
    (96, 128),       # N < 128, three scale rows in one tile
    (2304, 5120),    # V4.1 shared w1/w3 (decode config entry; 72 scale rows)
    (1280, 5120),    # V4.1 wq_a (decode config entry; K/32 = 160 blocks)
    (200, 160),      # N and N-tiles ragged, K/32 = 5 (odd block count -> split-K clamp)
]


@pytest.mark.parametrize("M", [1, 3, 7, 64])
@pytest.mark.parametrize("N,K", SHAPES_32)
def test_block32_matches_reference(M, N, K):
    gen = torch.Generator(device=DEV).manual_seed(N * 7919 + K * 31 + M)
    w, scale = make_weight(N, K, 32, gen)
    x = torch.randn(M, K, device=DEV, generator=gen).to(torch.bfloat16)
    x = x * torch.exp2(torch.randint(-4, 5, (M, 1), device=DEV, generator=gen).float()).bfloat16()
    got = block_fp8_linear(x, w, scale, block=32)
    assert got.shape == (M, N) and got.dtype == torch.bfloat16
    ref = ref_linear(x, w, scale.view(torch.uint8), 32)
    assert torch.isfinite(ref).all()
    torch.testing.assert_close(got.double(), ref, rtol=1e-2, atol=1e-2)


def test_block32_leading_dims_and_bias():
    gen = torch.Generator(device=DEV).manual_seed(11)
    N, K = 96, 128
    w, scale = make_weight(N, K, 32, gen)
    x = torch.randn(2, 5, K, device=DEV, generator=gen).to(torch.bfloat16)
    bias = torch.randn(N, device=DEV, generator=gen)
    got = block_fp8_linear(x, w, scale, bias, block=32)
    assert got.shape == (2, 5, N)
    ref = ref_linear(x.reshape(-1, K), w, scale.view(torch.uint8), 32) + bias.double()
    # the output is rounded to bf16 twice here (GEMM result, then ``+ bias``): up to
    # ~1 ulp = 0.78% just above a power of two, so allow two ulps
    torch.testing.assert_close(got.reshape(-1, N).double(), ref, rtol=2e-2, atol=2e-2)


def test_block32_rejects_wrong_scale_shape():
    gen = torch.Generator(device=DEV).manual_seed(5)
    w, scale = make_weight(64, 128, 32, gen)
    x = torch.randn(2, 128, device=DEV, generator=gen).bfloat16()
    with pytest.raises(AssertionError):
        block_fp8_linear(x, w, scale, block=128)  # 128-block scale would be [1, 1]


@pytest.mark.parametrize("M", [1, 3, 64])
@pytest.mark.parametrize("N,K", [(256, 512), (512, 4096), (1024, 4096)])
def test_block128_still_matches_reference(M, N, K):
    """The 0731 layout through the generalized kernels (default ``block=128``)."""
    gen = torch.Generator(device=DEV).manual_seed(N + K + M)
    w, scale = make_weight(N, K, 128, gen)
    x = torch.randn(M, K, device=DEV, generator=gen).to(torch.bfloat16)
    got = block_fp8_linear(x, w, scale)
    ref = ref_linear(x, w, scale.view(torch.uint8), 128)
    torch.testing.assert_close(got.double(), ref, rtol=1e-2, atol=1e-2)


# ----------------------------------------------------------------------------------------
# fp4_act_quant_e4m3_inplace
# ----------------------------------------------------------------------------------------
_E2M1_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def ref_round_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Nearest e2m1 magnitude, ties to the grid point with the even index (the even
    mantissa bit): a search over the 8 grid values, not a comparison ladder."""
    grid = _E2M1_GRID.to(x.device)
    a = x.abs()
    d = (a[..., None] - grid).abs()
    dmin = d.amin(-1, keepdim=True)
    idx = torch.arange(8, device=x.device)
    # among exact-minimum candidates prefer the even index; everything else is ranked out
    rank = torch.where(d == dmin, (idx % 2).expand_as(d), torch.full_like(d, 2, dtype=torch.long))
    pick = rank.argmin(-1)
    return torch.copysign(grid[pick], x)


def ref_fp4_e4m3_roundtrip(x: torch.Tensor, block: int) -> torch.Tensor:
    """Torch port of the reference ``fp4_quant_kernel(scale_dtype=FP8, inplace=True)``."""
    M, N = x.shape
    xb = x.float().reshape(M, N // block, block)
    amax = xb.abs().amax(-1, keepdim=True).clamp_min(6 * 2.0 ** -9)
    s = (amax / 6.0).clamp(max=FP8_MAX).to(FP8).float()   # satfinite e4m3 cast, RNE
    q = ref_round_e2m1((xb / s).clamp(-6.0, 6.0))
    return (q * s).to(torch.bfloat16).reshape(M, N)


def _fp4_inputs(gen):
    rows = []
    # random rows over a wide range of magnitudes (many distinct e4m3 scales)
    r = torch.randn(64, 512, device=DEV, generator=gen)
    r = r * torch.exp2(torch.randint(-12, 10, (64, 1), device=DEV, generator=gen).float())
    rows.append(r)
    # all-zero group -> the 6*2^-9 amax floor
    rows.append(torch.zeros(1, 512, device=DEV))
    # exact e2m1 midpoints with s == 1 (amax == 6): the tie-breaking probe
    ties = torch.tensor([6, 0.75, 1.75, 3.5, 0.25, 1.25, 2.5, 5.0,
                         -6, -0.75, -1.75, -3.5, -0.25, -1.25, -2.5, -5.0], device=DEV)
    rows.append(ties.repeat(1, 32))
    # a group whose amax/6 exceeds 448 -> the saturating scale cast
    big = torch.zeros(1, 512, device=DEV)
    big[0, ::16] = 4000.0
    big[0, 1::16] = -37.0
    rows.append(big)
    # scale exactly on an e4m3 rounding midpoint: amax/6 = 1.0625 (halfway 1.0 / 1.125)
    mid = torch.zeros(1, 512, device=DEV)
    mid[0, ::16] = 6.375
    mid[0, 2::16] = 1.5
    rows.append(mid)
    return torch.cat(rows).to(torch.bfloat16)


def test_fp4_e4m3_roundtrip_matches_reference_port():
    gen = torch.Generator(device=DEV).manual_seed(2024)
    x = _fp4_inputs(gen)
    ref = ref_fp4_e4m3_roundtrip(x, 16)
    got = fp4_act_quant_e4m3_inplace(x.clone(), 16)
    assert got.dtype == torch.bfloat16
    mism = (got != ref).nonzero()
    assert mism.numel() == 0, f"{mism.shape[0]} mismatches, first {mism[:8].tolist()}"


def test_fp4_e4m3_roundtrip_is_in_place_and_strided():
    """The compressed KV is a slice of a wider latent: the kernel must write through
    the strided view rather than a contiguous copy."""
    gen = torch.Generator(device=DEV).manual_seed(7)
    full = torch.randn(9, 96, device=DEV, generator=gen).to(torch.bfloat16)
    view = full[:, 16:80]
    ref = ref_fp4_e4m3_roundtrip(view.clone(), 16)
    ret = fp4_act_quant_e4m3_inplace(view, 16)
    assert ret.data_ptr() == view.data_ptr()
    assert torch.equal(full[:, 16:80], ref)
    assert torch.isfinite(full).all()


def test_fp4_e4m3_scale_floor_keeps_zero_group_finite():
    x = torch.zeros(3, 32, device=DEV, dtype=torch.bfloat16)
    got = fp4_act_quant_e4m3_inplace(x, 16)
    assert torch.equal(got, torch.zeros_like(got))
    assert math.isfinite(got.float().sum().item())
