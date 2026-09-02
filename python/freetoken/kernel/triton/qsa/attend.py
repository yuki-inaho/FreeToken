# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from vLLM (vllm/models/qwen4_exp/nvidia/ops/qsa.py)
"""Sparse paged GQA over the QSA selection."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from freetoken.kernel.triton.qsa.store_nvfp4 import KV_BLOCK


# Reciprocal of the 2^-14 factor the bit-trick decoders below leave on their result. It is a
# power of two, so folding it into the fp32 row global is exact.
_E2M1_UNSCALE = tl.constexpr(16384.0)


@triton.jit
def _e2m1_pair_f16(v):
    """Two E2M1 codes at int32 bits [3:0] and [19:16] -> ``(lo, hi)`` fp16, each ``value *
    2^-14``. The caller restores the factor via the row global (see ``_E2M1_UNSCALE``).

    Same bit trick as ``kernel/triton/nvfp4_dequant._nvfp4_pair_f16``: the magnitude bits land
    in fp16's [11:9] and the sign in [15], and fp16's subnormal encoding puts the e == 0 codes
    on the same 2^-14 scale, so the {0, 0.5} case needs no branch at all. Three integer ops for
    two elements, against ~10 fp32 ops each for the arithmetic spelling this replaced -- which
    is what made the prefill-shaped attend 14x the fp8 one.
    """
    r = ((v << 9) & 0x0E000E00) | ((v & 0x00080008) << 12)
    lo = (r & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True)
    hi = (r >> 16).to(tl.uint16).to(tl.float16, bitcast=True)
    return lo, hi


@triton.jit
def _e2m1_one_f16(code):
    """One E2M1 code at bits [3:0] -> fp16 ``value * 2^-14``. See :func:`_e2m1_pair_f16`."""
    r = ((code << 9) & 0x0E00) | ((code & 0x0008) << 12)
    return (r & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True)


@triton.jit
def _qsa_sparse_paged_gqa_splitk_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    k_scale_ptr,
    v_scale_ptr,
    k_blk_ptr,
    v_blk_ptr,
    indices_ptr,
    block_table_ptr,
    token_to_req_ptr,
    partial_output_ptr,
    partial_lse_ptr,
    output_ptr,
    stride_q_row,
    stride_q_head,
    stride_k_block,
    stride_k_token,
    stride_k_head,
    stride_v_block,
    stride_v_token,
    stride_v_head,
    stride_scale_token,
    stride_blk_token,
    stride_blk_head,
    stride_indices_row,
    stride_table_req,
    stride_output_row,
    stride_output_head,
    num_rows,
    num_cache_blocks,
    num_requests,
    TOPK: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_QUERY_HEADS: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    NUM_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    KV_FP8: tl.constexpr,
    KV_NVFP4: tl.constexpr,
    KV_BLOCK: tl.constexpr,
) -> None:
    # row * stride can overflow int32 for large row counts.
    row = tl.program_id(0).to(tl.int64)
    kv_head = tl.program_id(1)
    split_id = tl.program_id(2)
    request = tl.load(token_to_req_ptr + row)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)

    head_offsets = tl.arange(0, BLOCK_M)
    dim_offsets = tl.arange(0, HEAD_DIM)
    column_offsets = tl.arange(0, BLOCK_N)
    first_head = kv_head * GROUP_SIZE
    query = tl.load(
        q_ptr
        + row * stride_q_row
        + (first_head + head_offsets[:, None]) * stride_q_head
        + dim_offsets[None, :],
        mask=head_offsets[:, None] < GROUP_SIZE,
        other=0.0,
    )

    max_value = tl.full((BLOCK_M,), -1.0e20, dtype=tl.float32)
    normalizer = tl.zeros((BLOCK_M,), dtype=tl.float32)
    accumulator = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    softmax_scale_log2: tl.constexpr = (HEAD_DIM**-0.5) * 1.4426950408889634

    # Dynamic bounds avoid padded main-loop iterations for uneven splits.
    split_tile_start = split_id * NUM_TILES // NUM_SPLITS
    split_tile_end = (split_id + 1) * NUM_TILES // NUM_SPLITS
    for tile in range(split_tile_start, split_tile_end):
        columns = tile * BLOCK_N + column_offsets
        logical_token = tl.load(
            indices_ptr + row * stride_indices_row + columns,
            mask=columns < TOPK,
            other=-1,
        )
        safe_token = tl.maximum(logical_token, 0)
        logical_page = safe_token // PAGE_SIZE
        page_offset = safe_token % PAGE_SIZE
        valid = (
            (request >= 0)
            & (request < num_requests)
            & (logical_token >= 0)
            & (logical_page < PAGE_TABLE_WIDTH)
        )
        physical_page = tl.load(
            block_table_ptr
            + safe_request.to(tl.int64) * stride_table_req
            + tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1),
            mask=valid,
            other=-1,
        )
        valid &= (physical_page >= 0) & (physical_page < num_cache_blocks)
        # physical_page * block stride can overflow int32 for large caches.
        safe_page = tl.maximum(physical_page, 0).to(tl.int64)
        if KV_NVFP4:
            # Two E2M1 codes per byte (low nibble first) plus one e4m3 scale per KV_BLOCK
            # elements. Unlike fp8's single row scale, the block scale varies ALONG head_dim --
            # inside the dot product's reduction -- so it cannot be folded onto the scores
            # afterwards. Dequantize into registers here and let tl.dot see plain values.
            #
            # Loads are issued at their NATURAL width and expanded in registers wherever that
            # is safe. Indexing everything by dim_offsets -- `// 2` for the byte, `// KV_BLOCK`
            # for the scale -- reads each byte twice and each block scale KV_BLOCK times, and
            # measured 3.8x the fp8 attend; at their own widths it is 1.8x.
            #
            # "Wherever safe" is doing real work here. K is built token-major and transposed
            # into the dot's A operand, and tl.interleave is correct there. Feeding an
            # interleaved block STRAIGHT into the dot's B operand (V) is not: on Triton 3.6 it
            # silently yields exactly 2x the right answer -- no error, no NaN, just a doubled
            # V. The tl.trans on the K side happens to force a re-layout that avoids it; a
            # double transpose on V does not (it folds away), and it is the layout rather than
            # the decode -- both the arithmetic and the bit-trick spelling reproduce it. So V
            # keeps the per-element byte index, and only its block scale is broadcast. If you
            # try to interleave V here, tests/kvcache/test_nvfp4_kv_kernel.py will catch it.
            slot_lin = safe_page * PAGE_SIZE + page_offset
            glb_off = slot_lin * stride_scale_token + kv_head
            blk_base = kv_head * stride_blk_head
            half = tl.arange(0, HEAD_DIM // 2)
            blk = tl.arange(0, HEAD_DIM // KV_BLOCK)

            # --- K: loaded token-major, transposed into the dot's [HEAD_DIM, BLOCK_N] operand
            kbytes = tl.load(
                k_cache_ptr
                + safe_page[:, None] * stride_k_block
                + page_offset[:, None] * stride_k_token
                + kv_head * stride_k_head
                + half[None, :],
                mask=valid[:, None], other=0,
            ).to(tl.int32)
            klo, khi = _e2m1_pair_f16((kbytes & 0xF) | ((kbytes & 0xF0) << 12))
            kblk = tl.load(
                k_blk_ptr + slot_lin[:, None] * stride_blk_token + blk_base + blk[None, :],
                mask=valid[:, None], other=0.0,
            ).to(tl.float16)
            kblk = tl.reshape(
                tl.broadcast_to(kblk[:, :, None], (BLOCK_N, HEAD_DIM // KV_BLOCK, KV_BLOCK)),
                (BLOCK_N, HEAD_DIM),
            )
            keys = tl.trans(tl.interleave(klo, khi) * kblk).to(query.dtype)
            kglb = tl.load(k_scale_ptr + glb_off, mask=valid, other=0.0) * _E2M1_UNSCALE

            # --- V: the dot's B operand, so the byte index stays per-element (see above)
            byte_off = tl.arange(0, HEAD_DIM) // 2
            is_lo = (tl.arange(0, HEAD_DIM) % 2) == 0
            vbytes = tl.load(
                v_cache_ptr
                + safe_page[:, None] * stride_v_block
                + page_offset[:, None] * stride_v_token
                + kv_head * stride_v_head
                + byte_off[None, :],
                mask=valid[:, None], other=0,
            ).to(tl.int32)
            vcode = tl.where(is_lo[None, :], vbytes & 0xF, (vbytes >> 4) & 0xF)
            vblk = tl.load(
                v_blk_ptr + slot_lin[:, None] * stride_blk_token + blk_base + blk[None, :],
                mask=valid[:, None], other=0.0,
            ).to(tl.float16)
            vblk = tl.reshape(
                tl.broadcast_to(vblk[:, :, None], (BLOCK_N, HEAD_DIM // KV_BLOCK, KV_BLOCK)),
                (BLOCK_N, HEAD_DIM),
            )
            values = (_e2m1_one_f16(vcode) * vblk).to(query.dtype)
            vglb = tl.load(v_scale_ptr + glb_off, mask=valid, other=0.0) * _E2M1_UNSCALE
        else:
            keys = tl.load(
                k_cache_ptr
                + safe_page[None, :] * stride_k_block
                + page_offset[None, :] * stride_k_token
                + kv_head * stride_k_head
                + dim_offsets[:, None],
                mask=valid[None, :],
                other=0.0,
            )
            values = tl.load(
                v_cache_ptr
                + safe_page[:, None] * stride_v_block
                + page_offset[:, None] * stride_v_token
                + kv_head * stride_v_head
                + dim_offsets[None, :],
                mask=valid[:, None],
                other=0.0,
            )
        if KV_FP8:
            # e4m3 codes + one fp32 scale per (token, kv head). The codes convert exactly to
            # the compute dtype; the scales fold in algebraically: K's on the scores (per
            # column), V's on the probabilities the second dot consumes (per column of P ==
            # per row of V). Masked gathers with other=0.0: an invalid column's probability
            # is 0 and 0 * garbage would be NaN.
            scale_index = (safe_page * PAGE_SIZE + page_offset) * stride_scale_token + kv_head
            k_scale = tl.load(k_scale_ptr + scale_index, mask=valid, other=0.0)
            v_scale = tl.load(v_scale_ptr + scale_index, mask=valid, other=0.0)
            keys = keys.to(query.dtype)
            values = values.to(query.dtype)
        scores = tl.dot(query, keys)
        if KV_FP8:
            scores *= k_scale[None, :]
        elif KV_NVFP4:
            # Only the BLOCK scale had to be applied before the dot; the row global is constant
            # along head_dim, so it folds onto the scores exactly as fp8's single scale does.
            scores *= kglb[None, :]
        # Scaling scores avoids re-quantizing a scaled query to BF16.
        scores *= softmax_scale_log2
        scores = tl.where(valid[None, :], scores, -1.0e20)
        next_max = tl.maximum(max_value, tl.max(scores, axis=1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.where(
            valid[None, :], tl.math.exp2(scores - next_max[:, None]), 0.0
        )
        if KV_FP8:
            # The denominator stays over the UNSCALED probabilities; only the numerator's
            # operand carries V's per-token scale.
            p_operand = (probabilities * v_scale[None, :]).to(query.dtype)
        elif KV_NVFP4:
            p_operand = (probabilities * vglb[None, :]).to(query.dtype)
        else:
            p_operand = probabilities.to(values.dtype)
        accumulator = tl.dot(
            p_operand,
            values,
            acc=accumulator * alpha[:, None],
        )
        normalizer = normalizer * alpha + tl.sum(probabilities, axis=1)
        max_value = next_max

    has_values = normalizer > 0
    normalized_output = tl.where(
        has_values[:, None],
        accumulator / tl.maximum(normalizer[:, None], 1.0e-20),
        0.0,
    )
    output_mask = head_offsets[:, None] < GROUP_SIZE
    if NUM_SPLITS == 1:
        tl.store(
            output_ptr
            + row * stride_output_row
            + (first_head + head_offsets[:, None]) * stride_output_head
            + dim_offsets[None, :],
            normalized_output,
            mask=output_mask,
        )
    else:
        partial_lse = tl.where(
            has_values,
            max_value + tl.math.log2(tl.maximum(normalizer, 1.0e-20)),
            -float("inf"),
        )
        tl.store(
            partial_output_ptr
            + (
                (split_id.to(tl.int64) * num_rows + row) * NUM_QUERY_HEADS
                + first_head
                + head_offsets[:, None]
            )
            * HEAD_DIM
            + dim_offsets[None, :],
            normalized_output,
            mask=output_mask,
        )
        tl.store(
            partial_lse_ptr
            + (split_id.to(tl.int64) * num_rows + row) * NUM_QUERY_HEADS
            + first_head
            + head_offsets,
            partial_lse,
            mask=head_offsets < GROUP_SIZE,
        )


@triton.jit
def _qsa_merge_splitk_kernel(
    partial_output_ptr,
    partial_lse_ptr,
    output_ptr,
    stride_output_row,
    stride_output_head,
    num_rows,
    HEAD_DIM: tl.constexpr,
    NUM_QUERY_HEADS: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_SPLITS: tl.constexpr,
) -> None:
    row = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    split_offsets = tl.arange(0, BLOCK_SPLITS)
    dim_offsets = tl.arange(0, HEAD_DIM)
    split_mask = split_offsets < NUM_SPLITS
    lse = tl.load(
        partial_lse_ptr + (split_offsets.to(tl.int64) * num_rows + row) * NUM_QUERY_HEADS + head,
        mask=split_mask,
        other=-float("inf"),
    )
    lse_max = tl.max(lse, axis=0)
    has_values = lse_max > -float("inf")
    shifted = tl.where(split_mask & has_values, lse - lse_max, -float("inf"))
    weights = tl.math.exp2(shifted)
    denominator = tl.sum(weights, axis=0)
    partial_output = tl.load(
        partial_output_ptr
        + ((split_offsets[:, None].to(tl.int64) * num_rows + row) * NUM_QUERY_HEADS + head)
        * HEAD_DIM
        + dim_offsets[None, :],
        mask=split_mask[:, None],
        other=0.0,
    )
    merged = tl.sum(partial_output * weights[:, None], axis=0)
    merged = tl.where(denominator > 0, merged / denominator, 0.0)
    tl.store(
        output_ptr + row * stride_output_row + head * stride_output_head + dim_offsets,
        merged,
    )


def qsa_sparse_paged_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    out: torch.Tensor | None = None,
    *,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
    k_blk: torch.Tensor | None = None,
    v_blk: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run sparse GQA directly over paged K/V caches, dequantized inside the kernel.

    Three storage formats, picked from ``k_cache.dtype``:

    * 16-bit -- the compute dtype, no scales.
    * ``float8_e4m3fn`` -- codes plus fp32 per-(token slot, kv head) scales
      ``k_scale``/``v_scale`` ``[pages * page_size, kv_heads]`` (the FP8 QSA pool).
    * ``uint8`` -- NVFP4: two E2M1 codes per byte (``head_dim // 2`` bytes per head row), e4m3
      per-16-element block scales ``k_blk``/``v_blk`` ``[slots, kv_heads, head_dim // 16]``,
      and the same fp32 row scales as fp8 reused as the per-row global.
    """

    if q.ndim != 3 or k_cache.ndim != 4 or v_cache.shape != k_cache.shape:
        raise ValueError("QSA sparse attention received invalid Q/K/V shapes")
    if logical_indices.ndim != 2 or logical_indices.shape[0] != q.shape[0]:
        raise ValueError("QSA indices must have one row per query")
    if token_to_req.shape != (q.shape[0],) or block_table.ndim != 2:
        raise ValueError("QSA sparse attention metadata has invalid shapes")
    if logical_indices.shape[1] <= 0:
        raise ValueError("QSA sparse attention requires a positive selection width")
    kv_nvfp4 = k_cache.dtype == torch.uint8
    stored_dim = k_cache.shape[3] * 2 if kv_nvfp4 else k_cache.shape[3]
    if q.shape[2] != stored_dim or q.shape[1] % k_cache.shape[2]:
        raise ValueError("QSA sparse attention requires valid grouped-query heads")
    head_dim = q.shape[2]
    assert head_dim >= 16 and (head_dim & (head_dim - 1)) == 0
    kv_fp8 = k_cache.dtype == torch.float8_e4m3fn
    assert k_cache.dtype == v_cache.dtype and (
        kv_fp8 or kv_nvfp4 or q.dtype == k_cache.dtype
    ), (q.dtype, k_cache.dtype, v_cache.dtype)
    stride_blk_token = stride_blk_head = 0
    if kv_nvfp4:
        num_slots = k_cache.shape[0] * k_cache.shape[1]
        n_blocks = head_dim // KV_BLOCK
        assert k_scale is not None and v_scale is not None, "NVFP4 K/V needs row globals"
        assert k_blk is not None and v_blk is not None, "NVFP4 K/V needs block scales"
        assert k_scale.shape == v_scale.shape == (num_slots, k_cache.shape[2])
        assert k_scale.dtype == v_scale.dtype == torch.float32
        assert k_scale.stride() == v_scale.stride() and k_scale.stride(1) == 1
        assert k_blk.shape == v_blk.shape == (num_slots, k_cache.shape[2], n_blocks)
        assert k_blk.dtype == v_blk.dtype == torch.float8_e4m3fn
        assert k_blk.stride() == v_blk.stride() and k_blk.stride(2) == 1
        stride_scale_token = k_scale.stride(0)
        stride_blk_token, stride_blk_head = k_blk.stride(0), k_blk.stride(1)
    elif kv_fp8:
        num_slots = k_cache.shape[0] * k_cache.shape[1]
        assert k_scale is not None and v_scale is not None, "fp8 K/V needs per-token scales"
        assert k_scale.shape == v_scale.shape == (num_slots, k_cache.shape[2]), (
            k_scale.shape, (num_slots, k_cache.shape[2])
        )
        assert k_scale.dtype == v_scale.dtype == torch.float32
        assert k_scale.stride() == v_scale.stride() and k_scale.stride(1) == 1
        stride_scale_token = k_scale.stride(0)
    else:
        assert k_scale is None and v_scale is None, "scales are for quantized K/V pools only"
        k_scale = v_scale = q  # unused pointers when the pool is not quantized
        stride_scale_token = 0
    if not kv_nvfp4:
        k_blk = v_blk = q  # unused pointer under KV_NVFP4=False
    assert logical_indices.dtype == block_table.dtype == torch.int32
    assert token_to_req.dtype == torch.int32
    assert q.stride(2) == k_cache.stride(3) == v_cache.stride(3) == 1
    assert logical_indices.stride(1) == block_table.stride(1) == 1
    assert token_to_req.stride(0) == 1
    if out is None:
        out = torch.empty_like(q)
    assert out.shape == q.shape and out.dtype == q.dtype and out.stride(2) == 1
    if not q.shape[0]:
        return out

    group_size = q.shape[1] // k_cache.shape[2]
    block_m = triton.next_power_of_2(group_size)
    base_programs = q.shape[0] * k_cache.shape[2]
    small_profile_limit = 8 if block_m <= 8 else 4

    # Tuned on GB300 for the Qwen-Air TP1, TP2, and TP4 attention shapes.
    # Narrow tiles favor decode; wide tiles improve throughput for prefill.
    if base_programs <= small_profile_limit:
        block_n, target_splits, partial_warps = 16, 64, 4
    elif base_programs < 32:
        block_n, target_splits, partial_warps = 16, 32, 4
    elif base_programs <= 256:
        block_n, target_splits, partial_warps = 64, 8, 2
    elif base_programs <= 512:
        block_n, target_splits, partial_warps = 64, 4, 2
    else:
        block_n, target_splits, partial_warps = 64, 1, 2

    num_tiles = triton.cdiv(logical_indices.shape[1], block_n)
    # Avoid empty splits when the selection width is smaller than the profile.
    max_useful_splits = 1 << (num_tiles.bit_length() - 1)
    num_splits = min(max_useful_splits, target_splits)

    # Split=1 writes output directly and compiles out all workspace accesses.
    if num_splits == 1:
        partial_output = out
        partial_lse = out
    else:
        # FP32 partials preserve accuracy when merging independently normalized
        # splits.
        partial_output = torch.empty(
            (num_splits, *q.shape), dtype=torch.float32, device=q.device
        )
        partial_lse = torch.empty(
            (num_splits, q.shape[0], q.shape[1]),
            dtype=torch.float32,
            device=q.device,
        )

    partial_grid = (q.shape[0], k_cache.shape[2], num_splits)
    _qsa_sparse_paged_gqa_splitk_kernel[partial_grid](
        q,
        k_cache,
        v_cache,
        k_scale,
        v_scale,
        k_blk,
        v_blk,
        logical_indices,
        block_table,
        token_to_req,
        partial_output,
        partial_lse,
        out,
        q.stride(0),
        q.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        stride_scale_token,
        stride_blk_token,
        stride_blk_head,
        logical_indices.stride(0),
        block_table.stride(0),
        out.stride(0),
        out.stride(1),
        q.shape[0],
        k_cache.shape[0],
        block_table.shape[0],
        TOPK=logical_indices.shape[1],
        PAGE_SIZE=k_cache.shape[1],
        PAGE_TABLE_WIDTH=block_table.shape[1],
        GROUP_SIZE=group_size,
        HEAD_DIM=q.shape[2],
        NUM_QUERY_HEADS=q.shape[1],
        NUM_SPLITS=num_splits,
        NUM_TILES=num_tiles,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        KV_FP8=kv_fp8,
        KV_NVFP4=kv_nvfp4,
        KV_BLOCK=KV_BLOCK,
        num_warps=partial_warps,
        num_stages=2,
    )
    if num_splits == 1:
        return out

    _qsa_merge_splitk_kernel[(q.shape[0], q.shape[1])](
        partial_output,
        partial_lse,
        out,
        out.stride(0),
        out.stride(1),
        q.shape[0],
        HEAD_DIM=q.shape[2],
        NUM_QUERY_HEADS=q.shape[1],
        NUM_SPLITS=num_splits,
        BLOCK_SPLITS=triton.next_power_of_2(num_splits),
        num_warps=2,
        num_stages=1,
    )
    return out


__all__ = ["qsa_sparse_paged_attention"]
