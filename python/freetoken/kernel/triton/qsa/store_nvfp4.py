# SPDX-License-Identifier: Apache-2.0
"""Quantizing K/V store for the NVFP4 QSA pool.

Same shape as the fp8 store, one program per (token slot, kv head), but three tiers instead of
two -- and the middle one is why NVFP4 is not a drop-in for fp8 anywhere downstream:

* ``k_cache``/``v_cache`` ``[slots, heads, head_dim // 2]`` uint8 -- two E2M1 codes per byte,
  low nibble first, matching the expert banks (``triton/nvfp4_dequant.py``);
* ``k_blk``/``v_blk`` ``[slots, heads, head_dim // 16]`` e4m3 -- one scale per 16 elements;
* ``k_glb``/``v_glb`` ``[slots, heads]`` fp32 -- one global per row, the tier the fp8 pool
  already carries.

``value = E2M1[code] * block_scale * global``. The block scale varies ALONG head_dim, i.e.
inside the attention dot product's reduction, so unlike fp8's single row scale it cannot be
folded onto the scores afterwards -- the attend kernel has to dequantize before ``tl.dot``.

The global is fp32 here where the probe used fp16. That is strictly more accurate (the fp16
version underflowed to zero for near-zero rows) and the gates were passed with the worse of the
two. The zero-row guard is kept regardless: a never-written slot must dequantize to a finite 0,
exactly as the fp8 path promises, and ``0 / 0`` in the block scale would poison attention.

No LUT tensor anywhere: this runs under CUDA graph capture, where materialising one with
``torch.tensor`` is a host->device copy and capture rejects it outright.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

E2M1_MAX = 6.0
E4M3_MAX = 448.0
KV_BLOCK = 16  # elements per block scale

# Triton cannot read plain module globals from a @jit'ed function; these are the in-kernel
# mirrors of the constants above.
_E2M1_MAX = tl.constexpr(6.0)
_E4M3_MAX = tl.constexpr(448.0)


@triton.jit
def _e2m1_code(x):
    """Nearest E2M1 code (4 bits: sign, 2-bit exponent, 1-bit mantissa) for ``x``.

    A where-chain over the 8 magnitudes {0, .5, 1, 1.5, 2, 3, 4, 6} rather than a lookup
    table, so nothing touches the host under graph capture.
    """
    a = tl.abs(x)
    idx = tl.where(a < 0.25, 0,
          tl.where(a < 0.75, 1,
          tl.where(a < 1.25, 2,
          tl.where(a < 1.75, 3,
          tl.where(a < 2.50, 4,
          tl.where(a < 3.50, 5,
          tl.where(a < 5.00, 6, 7)))))))
    return (idx | tl.where(x < 0, 8, 0)).to(tl.uint8)


@triton.jit
def _quant_head_row(x_even, x_odd, in_range):
    """-> (packed bytes [NBLK, HALF], block scales [NBLK] fp32, global fp32) for one row."""
    amax_e = tl.max(tl.abs(x_even), axis=1)
    amax_o = tl.max(tl.abs(x_odd), axis=1)
    blk_absmax = tl.maximum(amax_e, amax_o)                 # [NBLK]
    g = tl.math.div_rn(tl.max(blk_absmax, axis=0), _E2M1_MAX * _E4M3_MAX)  # scalar

    # An all-zero row drives the global to 0 and the block scale to 0/0. Divide by a safe 1.0
    # and let the zero denominator carry the row to exact zeros (finite, as fp8 guarantees).
    safe_g = tl.where(g > 0, g, 1.0)
    # div_rn, not `/`: Triton lowers fp32 division to the approximate reciprocal, and 16-bit
    # inputs make block_absmax / row_absmax a small dyadic rational often enough that s lands
    # EXACTLY on an e4m3 midpoint (448 * 3/4 = 336, 448 * 27/28 = 432). One ulp there flips the
    # stored scale by a whole step, so an approximate divide would make the kernel disagree
    # with nvfp4_fake_quant on a few blocks per row -- equally accurate, but not reproducible,
    # and it would put the quality gates out of reach of the reference they were taken with.
    s = tl.minimum(tl.math.div_rn(blk_absmax / _E2M1_MAX, safe_g), _E4M3_MAX)
    s = s.to(tl.float8e4nv).to(tl.float32)                  # round-trip through the stored type
    denom = s[:, None] * g
    safe_denom = tl.where(denom > 0, denom, 1.0)

    lo = _e2m1_code(tl.where(in_range, tl.math.div_rn(x_even, safe_denom), 0.0)).to(tl.int32)
    hi = _e2m1_code(tl.where(in_range, tl.math.div_rn(x_odd, safe_denom), 0.0)).to(tl.int32)
    return (lo | (hi << 4)).to(tl.uint8), s, g


@triton.jit
def _qsa_store_kv_nvfp4_kernel(
    k_ptr, v_ptr, slots_ptr,
    k_cache_ptr, v_cache_ptr,      # [slots, heads, HEAD_DIM // 2] uint8
    k_blk_ptr, v_blk_ptr,          # [slots, heads, NBLK] e4m3
    k_glb_ptr, v_glb_ptr,          # [slots, heads] fp32
    stride_k_row, stride_v_row,
    stride_cache_token, stride_cache_head,
    stride_blk_token, stride_blk_head,
    stride_glb_token,
    num_slots,
    HEAD_DIM: tl.constexpr,
    NBLK: tl.constexpr,            # HEAD_DIM // KV_BLOCK
    HALF: tl.constexpr,            # KV_BLOCK // 2
) -> None:
    row = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    slot = tl.load(slots_ptr + row).to(tl.int64)
    valid = (slot >= 0) & (slot < num_slots)

    b = tl.arange(0, NBLK)[:, None]
    t = tl.arange(0, HALF)[None, :]
    d_even = b * (HALF * 2) + t * 2
    d_odd = d_even + 1
    in_range = valid & (d_odd < HEAD_DIM)

    base_k = row * stride_k_row + head * HEAD_DIM
    base_v = row * stride_v_row + head * HEAD_DIM
    ke = tl.load(k_ptr + base_k + d_even, mask=in_range, other=0.0).to(tl.float32)
    ko = tl.load(k_ptr + base_k + d_odd, mask=in_range, other=0.0).to(tl.float32)
    ve = tl.load(v_ptr + base_v + d_even, mask=in_range, other=0.0).to(tl.float32)
    vo = tl.load(v_ptr + base_v + d_odd, mask=in_range, other=0.0).to(tl.float32)

    kb, ks, kg = _quant_head_row(ke, ko, in_range)
    vb, vs, vg = _quant_head_row(ve, vo, in_range)

    safe_slot = tl.maximum(slot, 0)
    byte_idx = b * HALF + t
    dst = safe_slot * stride_cache_token + head * stride_cache_head + byte_idx
    tl.store(k_cache_ptr + dst, kb, mask=in_range)
    tl.store(v_cache_ptr + dst, vb, mask=in_range)

    blk_dst = safe_slot * stride_blk_token + head * stride_blk_head + tl.arange(0, NBLK)
    blk_mask = valid & (tl.arange(0, NBLK) < NBLK)
    tl.store(k_blk_ptr + blk_dst, ks.to(tl.float8e4nv), mask=blk_mask)
    tl.store(v_blk_ptr + blk_dst, vs.to(tl.float8e4nv), mask=blk_mask)

    glb_dst = safe_slot * stride_glb_token + head
    tl.store(k_glb_ptr + glb_dst, kg, mask=valid)
    tl.store(v_glb_ptr + glb_dst, vg, mask=valid)


def qsa_store_kv_nvfp4(
    k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor,
    k_cache: torch.Tensor, v_cache: torch.Tensor,
    k_blk: torch.Tensor, v_blk: torch.Tensor,
    k_glb: torch.Tensor, v_glb: torch.Tensor,
) -> None:
    """Quantize ``k``/``v`` ``[T, H*D]`` into the packed NVFP4 slabs at slots ``out_loc``.

    Negative or out-of-range slots are skipped, as in the fp8 store.
    """
    T = k.shape[0]
    if T == 0:
        return
    slots, heads, packed = k_cache.shape
    head_dim = packed * 2
    nblk = head_dim // KV_BLOCK
    assert k_cache.dtype == v_cache.dtype == torch.uint8, (k_cache.dtype, v_cache.dtype)
    assert v_cache.shape == k_cache.shape
    assert k_blk.shape == v_blk.shape == (slots, heads, nblk), (k_blk.shape, (slots, heads, nblk))
    assert k_blk.dtype == v_blk.dtype == torch.float8_e4m3fn
    assert k_glb.shape == v_glb.shape == (slots, heads)
    assert k_glb.dtype == v_glb.dtype == torch.float32
    assert k.shape == (T, heads * head_dim) and v.shape == k.shape, (k.shape, v.shape)
    assert k.stride(1) == 1 and v.stride(1) == 1
    assert k_cache.stride(2) == 1 and k_blk.stride(2) == 1 and k_glb.stride(1) == 1
    assert out_loc.shape == (T,) and out_loc.stride(0) == 1
    _qsa_store_kv_nvfp4_kernel[(T, heads)](
        k, v, out_loc, k_cache, v_cache, k_blk, v_blk, k_glb, v_glb,
        k.stride(0), v.stride(0),
        k_cache.stride(0), k_cache.stride(1),
        k_blk.stride(0), k_blk.stride(1),
        k_glb.stride(0),
        slots,
        HEAD_DIM=head_dim, NBLK=nblk, HALF=KV_BLOCK // 2,
    )
