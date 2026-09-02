# SPDX-License-Identifier: Apache-2.0
"""Quantizing K/V store for the FP8 QSA pool.

The pool's paged K/V slabs hold e4m3 codes with ONE fp32 scale per (token slot, kv head):
``scale = max(amax(row) / 448, 1e-12)``, ``code = round(x / scale)`` -- the per-token
"dynamic" fp8 recipe (``fp8_pertensor_linear._dynamic_quant_kernel``), applied per head so a
row's 256 elements share a scale. K rows arrive post-norm+rope and V rows raw, both in the
compute dtype; one program per (token, head) scatters the codes and the scale to the row's
slot. No host sync (CUDA-graph safe); a rejected speculative row is simply overwritten with
its codes AND its scale when the position is re-issued.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _qsa_store_kv_fp8_kernel(
    k_ptr,
    v_ptr,
    slots_ptr,
    k_cache_ptr,
    v_cache_ptr,
    k_scale_ptr,
    v_scale_ptr,
    stride_k_row,
    stride_v_row,
    stride_cache_token,
    stride_cache_head,
    stride_scale_token,
    num_slots,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    row = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    dims = tl.arange(0, BLOCK_D)
    in_dim = dims < HEAD_DIM
    slot = tl.load(slots_ptr + row).to(tl.int64)
    valid = (slot >= 0) & (slot < num_slots)
    mask = valid & in_dim
    k = tl.load(k_ptr + row * stride_k_row + head * HEAD_DIM + dims, mask=mask, other=0.0)
    v = tl.load(v_ptr + row * stride_v_row + head * HEAD_DIM + dims, mask=mask, other=0.0)
    k = k.to(tl.float32)
    v = v.to(tl.float32)
    k_scale = tl.maximum(tl.max(tl.abs(k), axis=0) / 448.0, 1e-12)
    v_scale = tl.maximum(tl.max(tl.abs(v), axis=0) / 448.0, 1e-12)
    kq = tl.minimum(tl.maximum(k / k_scale, -448.0), 448.0)
    vq = tl.minimum(tl.maximum(v / v_scale, -448.0), 448.0)
    safe_slot = tl.maximum(slot, 0)
    dst = safe_slot * stride_cache_token + head * stride_cache_head + dims
    tl.store(k_cache_ptr + dst, kq.to(tl.float8e4nv), mask=mask)
    tl.store(v_cache_ptr + dst, vq.to(tl.float8e4nv), mask=mask)
    scale_index = safe_slot * stride_scale_token + head
    tl.store(k_scale_ptr + scale_index, k_scale, mask=valid)
    tl.store(v_scale_ptr + scale_index, v_scale, mask=valid)


def qsa_store_kv_fp8(
    k: torch.Tensor,
    v: torch.Tensor,
    out_loc: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> None:
    """Quantize ``k``/``v`` ``[T, H*D]`` rows into the e4m3 slabs ``k_cache``/``v_cache``
    ``[slots, H, D]`` at token slots ``out_loc`` ``[T]``, writing their fp32 scales to
    ``k_scale``/``v_scale`` ``[slots, H]``. Negative or out-of-range slots are skipped."""
    T = k.shape[0]
    if T == 0:
        return
    slots, heads, head_dim = k_cache.shape
    assert v_cache.shape == k_cache.shape, (k_cache.shape, v_cache.shape)
    assert k_cache.dtype == v_cache.dtype == torch.float8_e4m3fn, (k_cache.dtype, v_cache.dtype)
    assert k.shape == (T, heads * head_dim) and v.shape == (T, heads * head_dim), (
        k.shape, v.shape, (T, heads * head_dim)
    )
    assert k.stride(1) == 1 and v.stride(1) == 1
    assert k_cache.stride(2) == 1 and v_cache.stride(2) == 1
    assert k_cache.stride(0) == v_cache.stride(0) and k_cache.stride(1) == v_cache.stride(1)
    assert k_scale.shape == (slots, heads) and v_scale.shape == (slots, heads), (
        k_scale.shape, v_scale.shape, (slots, heads)
    )
    assert k_scale.dtype == v_scale.dtype == torch.float32
    assert k_scale.stride(1) == 1 and v_scale.stride(1) == 1
    assert k_scale.stride(0) == v_scale.stride(0)
    assert out_loc.shape == (T,) and out_loc.stride(0) == 1
    _qsa_store_kv_fp8_kernel[(T, heads)](
        k,
        v,
        out_loc,
        k_cache,
        v_cache,
        k_scale,
        v_scale,
        k.stride(0),
        v.stride(0),
        k_cache.stride(0),
        k_cache.stride(1),
        k_scale.stride(0),
        slots,
        HEAD_DIM=head_dim,
        BLOCK_D=triton.next_power_of_2(head_dim),
        num_warps=4,
    )


__all__ = ["qsa_store_kv_fp8"]
