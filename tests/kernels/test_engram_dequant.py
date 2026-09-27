"""Engram row dequant (e4m3 bytes x e8m0 block codes -> bf16) against a numpy LUT.

The Engram runtime hands the kernel raw bytes gathered off NVMe, so a wrong decode would
not crash -- it would feed plausible-looking garbage into the Engram gate. The reference
here decodes e4m3 from the bit fields by formula (sign / exponent / mantissa, subnormals
included) in numpy, checks that table against torch's own ``float8_e4m3fn`` view once, and
builds the expected output in float64 before a single bf16 rounding. The kernel must match
bit-for-bit: a value times a power of two is exact, so both sides round the same number.

NaN codes (``0x7F``/``0xFF``) are excluded from the random bytes on purpose (HANDOVER
section 8, item 2: a NaN reference compares nothing) and asserted absent; the emulated
pre-sm89 decode maps them to +-480 by design, so a real table must never contain them.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from freetoken.kernel.triton.dsv4.engram_dequant import engram_dequant

# FREETOKEN_TEST_DEVICE=cpu + TRITON_INTERPRET=1 + FREETOKEN_FORCE_E4M3_EMU=1 runs the
# emulated-decode branch under Triton's interpreter without a GPU.
DEV = os.environ.get("FREETOKEN_TEST_DEVICE", "cuda")
pytestmark = pytest.mark.skipif(DEV == "cuda" and not torch.cuda.is_available(), reason="needs CUDA")

E4M3_NAN_CODES = (0x7F, 0xFF)


def e4m3_lut() -> np.ndarray:
    """``float(e4m3(code))`` for all 256 codes from the bit fields: bias 7, 3 mantissa
    bits, subnormals at exponent 0, no infinities (the ``fn`` variant)."""
    codes = np.arange(256, dtype=np.uint32)
    sign = np.where(codes & 0x80, -1.0, 1.0)
    exp = (codes >> 3) & 0xF
    man = (codes & 0x7).astype(np.float64)
    val = np.where(exp == 0, man / 8.0 * 2.0 ** -6, (1.0 + man / 8.0) * 2.0 ** (exp.astype(np.float64) - 7))
    val = sign * val
    val[list(E4M3_NAN_CODES)] = np.nan
    return val


def test_lut_matches_torch_fp8_view():
    lut = e4m3_lut()
    t = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float().numpy()
    finite = ~np.isnan(lut)
    assert np.array_equal(lut[finite], t[finite])
    assert np.isnan(t[list(E4M3_NAN_CODES)]).all()


def random_rows(R: int, D: int, nb: int, seed: int, lo=100, hi=140):
    """Random e4m3 bytes without NaN codes and e8m0 codes in a realistic band (real
    tables sit around 2^-20 .. 2^13; the kernel's exp2 is not asked about 2^-127)."""
    rng = np.random.default_rng(seed)
    vals = rng.integers(0, 256, size=(R, D), dtype=np.uint8)
    for nan in E4M3_NAN_CODES:
        vals[vals == nan] = 0
    scales = rng.integers(lo, hi, size=(R, nb), dtype=np.uint8)
    assert not np.isin(vals, E4M3_NAN_CODES).any()
    return vals, scales


def reference(vals: np.ndarray, scales: np.ndarray, block: int) -> torch.Tensor:
    lut = e4m3_lut()
    v = lut[vals]                                                    # float64 [R, D]
    s = np.exp2(scales.astype(np.float64) - 127.0)
    s = np.repeat(s, block, axis=1)
    out = v * s
    assert np.isfinite(out).all()
    return torch.from_numpy(out).to(torch.bfloat16)


@pytest.mark.parametrize("R", [1, 33, 1000, 24 * 300])
def test_dequant_matches_lut_reference(R):
    D, block = 256, 32
    vals, scales = random_rows(R, D, D // block, seed=R)
    ref = reference(vals, scales, block)
    got = engram_dequant(torch.from_numpy(vals).to(DEV), torch.from_numpy(scales).to(DEV))
    assert got.dtype == torch.bfloat16 and got.shape == (R, D)
    mism = (got.cpu() != ref).nonzero()
    assert mism.numel() == 0, f"{mism.shape[0]} mismatches, first {mism[:8].tolist()}"


def test_dequant_keeps_leading_shape_and_fills_out_buffer():
    T, H, D = 5, 24, 256
    vals, scales = random_rows(T * H, D, 8, seed=99)
    v = torch.from_numpy(vals).to(DEV).reshape(T, H, D)
    s = torch.from_numpy(scales).to(DEV).reshape(T, H, 8)
    out = torch.full((T, H, D), float("nan"), dtype=torch.bfloat16, device=DEV)
    ret = engram_dequant(v, s, out)
    assert ret is out
    ref = reference(vals, scales, 32).reshape(T, H, D)
    assert torch.equal(out.cpu(), ref)


def test_dequant_generic_head_dim_and_block():
    R, D, block = 77, 128, 16
    vals, scales = random_rows(R, D, D // block, seed=3)
    got = engram_dequant(torch.from_numpy(vals).to(DEV), torch.from_numpy(scales).to(DEV),
                         head_dim=D, block=block)
    assert torch.equal(got.cpu(), reference(vals, scales, block))


def test_dequant_covers_every_finite_code():
    """Every e4m3 code (bar NaN) and every realistic e8m0 code appears at least once."""
    R, D = 256, 256
    vals = np.tile(np.arange(256, dtype=np.uint8)[:, None], (1, D))     # row r = code r
    vals[list(E4M3_NAN_CODES)] = 0
    scales = np.tile(np.arange(96, 104, dtype=np.uint8)[None, :], (R, 1))
    scales[:, 0] = np.arange(R, dtype=np.uint8) % 60 + 90               # 90..149 down column 0
    got = engram_dequant(torch.from_numpy(vals).to(DEV), torch.from_numpy(scales).to(DEV))
    assert torch.equal(got.cpu(), reference(vals, scales, 32))


def test_dequant_rejects_mismatched_scales():
    v = torch.zeros(4, 256, dtype=torch.uint8, device=DEV)
    with pytest.raises(AssertionError):
        engram_dequant(v, torch.zeros(4, 4, dtype=torch.uint8, device=DEV))
    with pytest.raises(AssertionError):
        engram_dequant(v, torch.zeros(3, 8, dtype=torch.uint8, device=DEV))
