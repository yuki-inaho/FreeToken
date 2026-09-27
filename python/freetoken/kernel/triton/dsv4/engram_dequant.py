"""Dequantize gathered Engram table rows (FP8 e4m3 values + e8m0 block scales) to BF16.

DeepSeek-V4.1's Engram tables ship as ``embed.weight`` float8_e4m3fn ``[rows, 256]`` and
``embed.scale`` float8_e8m0fnu ``[rows, 8]`` -- one power-of-two scale per 32 dims. The
Engram runtime gathers the hashed rows off NVMe as raw bytes (``uint8``, so the staging
ring never depends on torch's fp8 dtypes and the reads stay plain memcpys) and hands them
to this kernel, which computes

    ``out[r, d] = float(e4m3(vals[r, d])) * 2 ** (scales[r, d // block] - 127)``

in fp32 and rounds once to bf16. A value times a power of two is exact in fp32 (barring
underflow), so the result is the RNE-to-bf16 of the exact product -- what
``vals.view(float8_e4m3fn).float() * scales.view(float8_e8m0fnu).float()`` gives.

The e4m3 decode branches on :func:`e4m3_native_cx` like the other DSV4 kernels: on
sm_89+ the bytes are bitcast to ``float8e4nv`` (NaN codes ``0x7F``/``0xFF`` stay NaN),
before that they are expanded with integer ops (``e4m3_u8_to_f32``, which maps the NaN
codes to +-480 -- a checkpoint never stores NaN table entries).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from freetoken.kernel.triton.e4m3_compat import e4m3_native_cx, e4m3_u8_to_f32


@triton.jit
def _engram_dequant_kernel(
    v_ptr,            # [R, D] uint8 (e4m3 bits)
    s_ptr,            # [R, D // BLOCK] uint8 (e8m0 codes)
    o_ptr,            # [R, D] bf16
    R,
    stride_vr, stride_sr, stride_or,
    D: tl.constexpr, BLOCK: tl.constexpr, BLOCK_R: tl.constexpr,
):
    """``BLOCK_R`` rows x all ``D`` dims per program, one scale block per static-loop
    step so each row's 8 scale codes are read once and the multiply is a plain
    ``[BLOCK_R, BLOCK] * [BLOCK_R, 1]`` broadcast (no reshape across layouts)."""
    pid = tl.program_id(0)
    offs_r = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    r_mask = offs_r < R
    offs_d = tl.arange(0, BLOCK)
    for b in tl.static_range(D // BLOCK):
        cols = b * BLOCK + offs_d
        v = tl.load(v_ptr + offs_r[:, None] * stride_vr + cols[None, :],
                    mask=r_mask[:, None], other=0)
        if e4m3_native_cx():
            f = v.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        else:
            f = e4m3_u8_to_f32(v)
        code = tl.load(s_ptr + offs_r * stride_sr + b, mask=r_mask, other=127)
        scale = tl.exp2(code.to(tl.float32) - 127.0)            # [BLOCK_R]
        tl.store(o_ptr + offs_r[:, None] * stride_or + cols[None, :],
                 (f * scale[:, None]).to(tl.bfloat16), mask=r_mask[:, None])


def engram_dequant(
    vals_u8: torch.Tensor,
    scales_u8: torch.Tensor,
    out: torch.Tensor | None = None,
    head_dim: int = 256,
    block: int = 32,
) -> torch.Tensor:
    """``float(e4m3(vals)) * 2**(scale - 127)`` per ``block`` dims -> bf16.

    ``vals_u8``: ``[..., head_dim]`` uint8 (raw e4m3 bytes); ``scales_u8``:
    ``[..., head_dim // block]`` uint8 (raw e8m0 codes) with the same leading shape;
    ``out``: optional bf16 ``[..., head_dim]`` buffer (a static graph-captured buffer in
    decode). Returns ``out`` (allocated when None) shaped like ``vals_u8``.
    """
    assert vals_u8.dtype == torch.uint8 and scales_u8.dtype == torch.uint8, (
        vals_u8.dtype, scales_u8.dtype)
    assert head_dim % block == 0 and (block & (block - 1)) == 0, (head_dim, block)
    assert vals_u8.shape[-1] == head_dim, (vals_u8.shape, head_dim)
    nb = head_dim // block
    assert scales_u8.shape[-1] == nb, (scales_u8.shape, head_dim, block)
    assert vals_u8.shape[:-1] == scales_u8.shape[:-1], (vals_u8.shape, scales_u8.shape)
    v2 = vals_u8.reshape(-1, head_dim)
    s2 = scales_u8.reshape(-1, nb)
    assert v2.stride(1) == 1 and s2.stride(1) == 1, "rows must be contiguous"
    if out is None:
        out = torch.empty(vals_u8.shape, dtype=torch.bfloat16, device=vals_u8.device)
    assert out.dtype == torch.bfloat16 and out.shape == vals_u8.shape, (out.dtype, out.shape)
    o2 = out.reshape(-1, head_dim)
    assert o2.stride(1) == 1, "out rows must be contiguous"
    R = v2.shape[0]
    if R == 0:
        return out
    BLOCK_R = 32
    _engram_dequant_kernel[(triton.cdiv(R, BLOCK_R),)](
        v2, s2, o2, R, v2.stride(0), s2.stride(0), o2.stride(0),
        D=head_dim, BLOCK=block, BLOCK_R=BLOCK_R, num_warps=4,
    )
    return out


__all__ = ["engram_dequant"]
