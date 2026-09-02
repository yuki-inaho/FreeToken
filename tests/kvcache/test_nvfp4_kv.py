"""NVFP4 K/V numerics -- the quality probe that gates the real cache format.

These pin the quantiser's contract so a later kernel implementation has a reference to diff
against: same 16-element blocks, same e4m3 block scale, same per-row fp16 global, and the same
E2M1 grid the expert dequant kernel reads.
"""

from __future__ import annotations

import torch

from freetoken.kvcache.nvfp4_kv import (
    BLOCK, E2M1_MAX, _round_to_e2m1, fake_quant_kv, nvfp4_fake_quant,
)

GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _fp8_roundtrip(x: torch.Tensor) -> torch.Tensor:
    """What the shipping fp8 KV does: one scale per (token, head) row, e4m3 codes."""
    s = (x.abs().amax(-1, keepdim=True) / 448.0).clamp_min(1e-12)
    return (x / s).to(torch.float8_e4m3fn).float() * s


def _relerr(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b).norm() / a.norm()).item()


def test_rounding_lands_only_on_the_e2m1_grid():
    x = torch.linspace(-7.0, 7.0, 20001)
    assert sorted(_round_to_e2m1(x).abs().unique().tolist()) == list(GRID)


def test_rounding_uses_no_host_tensor():
    """It runs inside the captured decode graph, where a host->device copy is rejected.

    A ``torch.tensor(...)`` lookup table is exactly that copy -- it crashed a live boot with
    "Cannot copy between CPU and CUDA tensors during CUDA graph capture". Keep the
    implementation free of tensor literals.
    """
    import inspect

    src = inspect.getsource(_round_to_e2m1)
    assert "torch.tensor(" not in src, "a tensor literal here is a host copy under capture"


def test_roundtrip_is_deterministic_and_shape_preserving():
    torch.manual_seed(0)
    x = torch.randn(64, 256)
    a, b = nvfp4_fake_quant(x), nvfp4_fake_quant(x)
    assert torch.equal(a, b)
    assert a.shape == x.shape and a.dtype == x.dtype


def test_zero_rows_stay_zero():
    """Never-written slots read as a finite zero, as the fp8 path guarantees.

    Regression: the fp16 row global underflows to 0 for an all-zero row, and the block scale
    was then 0/0 = NaN -- which attention would have propagated silently.
    """
    x = torch.zeros(8, 256)
    out = nvfp4_fake_quant(x)
    assert torch.isfinite(out).all(), "zero rows produced NaN/Inf"
    assert torch.equal(out, x)


def test_mixed_zero_and_live_rows_are_both_finite():
    torch.manual_seed(9)
    x = torch.randn(16, 256)
    x[::2] = 0.0
    out = nvfp4_fake_quant(x)
    assert torch.isfinite(out).all()
    assert torch.equal(out[::2], torch.zeros_like(out[::2]))
    assert not torch.equal(out[1::2], torch.zeros_like(out[1::2]))


def test_tiny_but_nonzero_rows_stay_finite():
    """Rows far below the fp16 global's range flush to zero rather than producing NaN."""
    out = nvfp4_fake_quant(torch.full((4, 256), 1e-9))
    assert torch.isfinite(out).all()


def test_error_is_bounded_and_worse_than_fp8_by_a_known_factor():
    """~3.6-3.9x fp8's relative error. Pinned so a regression in the quantiser is visible;
    this ratio is what the needle/reasoning gates were run against."""
    torch.manual_seed(0)
    x = torch.randn(4096, 256)
    n, f = _relerr(x, nvfp4_fake_quant(x)), _relerr(x, _fp8_roundtrip(x))
    assert n < 0.12, f"NVFP4 relative error {n:.4f} is worse than the measured 0.095"
    assert 3.0 < n / f < 4.5, f"error ratio vs fp8 drifted to {n / f:.2f}x"


def test_outliers_do_not_blow_up_the_block():
    """One large value must not wipe out its block: the per-16 scale is why NVFP4 beats a
    single row scale at 4 bits."""
    torch.manual_seed(1)
    x = torch.randn(256, 256)
    x[:, 0] = 40.0
    out = nvfp4_fake_quant(x)
    assert torch.isfinite(out).all()
    # the untouched tail keeps roughly its usual accuracy
    assert _relerr(x[:, BLOCK:], out[:, BLOCK:]) < 0.15


def test_block_granularity_actually_helps():
    """A per-row-only 4-bit scale would be much worse; prove the block scale is doing work."""
    torch.manual_seed(2)
    x = torch.randn(512, 256) * torch.exp(torch.randn(512, 1) * 2)
    per_row = torch.sign(x) * _round_to_e2m1(
        (x / (x.abs().amax(-1, keepdim=True) / E2M1_MAX)).abs()
    ) * (x.abs().amax(-1, keepdim=True) / E2M1_MAX)
    assert _relerr(x, nvfp4_fake_quant(x)) < _relerr(x, per_row)


def test_fake_quant_kv_splits_heads_correctly():
    torch.manual_seed(3)
    head_dim, heads, T = 256, 2, 5
    k = torch.randn(T, heads * head_dim)
    v = torch.randn(T, heads * head_dim)
    kq, vq = fake_quant_kv(k, v, head_dim)
    assert kq.shape == k.shape and vq.shape == v.shape
    # each head is scaled independently: a huge head 1 must not degrade head 0
    k2 = k.clone()
    k2[:, head_dim:] *= 1000.0
    kq2, _ = fake_quant_kv(k2, v, head_dim)
    assert torch.allclose(kq[:, :head_dim], kq2[:, :head_dim], atol=0, rtol=0)
