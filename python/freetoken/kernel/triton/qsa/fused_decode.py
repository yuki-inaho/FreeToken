"""Decode-time (T <= 4, one request) fusion of the QSA index-write prologue.

Per QSA layer and step the sparse-attention backend launches, on <= 4 rows:
``_qsa_store_kv_fp8_kernel`` (K/V rows -> fp8 slabs + scales), ``_compress_qsa_groups_kernel``
(pool the closing 4-token group from the pending ring + this step's raw index keys),
``_index_norm_rope_kernel`` (norm + partial NeoX rope of the pooled key -> compressed slab
row), ``_store_qsa_rows_kernel`` (raw index keys -> pending ring) and a second
``_index_norm_rope_kernel`` over the query-side index rows. Five launches of 1-8 CTAs each.
This kernel does all five in ONE launch with one program per token row.

Hazards: the compress step reads ring positions ``chunk_start-3 .. chunk_start-1`` and the
ring store writes ``chunk_start .. chunk_start+T-1``; with ``T + ratio - 1 <= ring_capacity``
(4 + 3 <= 8 here) the two windows are disjoint modulo the capacity, so programs never read a
row another program writes in the same launch. Rows of this step that belong to a group are
read from the raw keys, never from the ring (as in the unfused kernel). The math of each
stage is the unfused kernel's, so results match within fp32 reduction-order noise (the K/V
store and ring store are bit-identical).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _norm_rope_row(x, w_plus1, w_plus1_partner, x_partner, cos, sin, dims, eps,
                   HEAD_DIM: tl.constexpr, ROTARY_HALF: tl.constexpr):
    """One indexer row: zero-centered RMSNorm ((1+w) weight) then partial NeoX rope."""
    rrms = tl.rsqrt(tl.sum(x * x, axis=0) / HEAD_DIM + eps)
    y = x * rrms * w_plus1
    y_partner = x_partner * rrms * w_plus1_partner
    in_rotary = dims < 2 * ROTARY_HALF
    sign = tl.where(dims < ROTARY_HALF, -1.0, 1.0)
    return tl.where(in_rotary, y * cos + sign * y_partner * sin, y)


@triton.jit
def _qsa_index_write_kernel(
    # K/V store (fp8 slabs)
    k_ptr, v_ptr, out_loc_ptr, k_cache_ptr, v_cache_ptr, k_scale_ptr, v_scale_ptr,
    stride_k_row, stride_v_row, stride_cache_token, stride_cache_head, stride_scale_token,
    num_slots,
    # indexer keys / queries
    raw_k_ptr, stride_rawk_row,          # [T, IDX_DIM]
    idx_q_ptr, stride_idxq_row,          # [T, Q_HEADS * IDX_DIM]
    q_out_ptr, stride_qout_row,          # [T, Q_HEADS, IDX_DIM] contiguous heads
    ring_ptr, stride_ring_slot, stride_ring_row, num_ring_slots,
    ring_slots_ptr, token_to_req_ptr, cu_seqlens_ptr, positions_ptr, num_requests,
    cmp_ptr, stride_cmp_row, cmp_rows_ptr, ring_rows_ptr,
    cos_sin_ptr, stride_cos_sin_row, k_norm_w_ptr, q_norm_w_ptr, eps,
    num_rows,
    KV_HEADS: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_KV: tl.constexpr,
    Q_HEADS: tl.constexpr, IDX_DIM: tl.constexpr, BLOCK_I: tl.constexpr,
    RING_CAPACITY: tl.constexpr, RATIO: tl.constexpr, ROTARY_HALF: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    live = row < num_rows

    # ---- 1. K/V rows -> fp8 slabs + per-(token, head) scales (the store_fp8 kernel, per head)
    slot = tl.load(out_loc_ptr + row, mask=live, other=-1).to(tl.int64)
    valid = live & (slot >= 0) & (slot < num_slots)
    safe_slot = tl.maximum(slot, 0)
    dkv = tl.arange(0, BLOCK_KV)
    in_kv = dkv < HEAD_DIM
    for head in tl.static_range(KV_HEADS):
        m = valid & in_kv
        k = tl.load(k_ptr + row * stride_k_row + head * HEAD_DIM + dkv, mask=m, other=0.0).to(tl.float32)
        v = tl.load(v_ptr + row * stride_v_row + head * HEAD_DIM + dkv, mask=m, other=0.0).to(tl.float32)
        k_scale = tl.maximum(tl.max(tl.abs(k), axis=0) / 448.0, 1e-12)
        v_scale = tl.maximum(tl.max(tl.abs(v), axis=0) / 448.0, 1e-12)
        kq = tl.minimum(tl.maximum(k / k_scale, -448.0), 448.0)
        vq = tl.minimum(tl.maximum(v / v_scale, -448.0), 448.0)
        dst = safe_slot * stride_cache_token + head * stride_cache_head + dkv
        tl.store(k_cache_ptr + dst, kq.to(tl.float8e4nv), mask=m)
        tl.store(v_cache_ptr + dst, vq.to(tl.float8e4nv), mask=m)
        tl.store(k_scale_ptr + safe_slot * stride_scale_token + head, k_scale, mask=valid)
        tl.store(v_scale_ptr + safe_slot * stride_scale_token + head, v_scale, mask=valid)

    # ---- 2. compress this row's closing group (ring for older members, raw rows for this step's)
    dims = tl.arange(0, BLOCK_I)
    in_dim = dims < IDX_DIM
    request = tl.load(token_to_req_ptr + row, mask=live, other=-1)
    end_position = tl.load(positions_ptr + row, mask=live, other=0)
    valid_request = live & (request >= 0) & (request < num_requests)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    q_start = tl.load(cu_seqlens_ptr + safe_request, mask=valid_request, other=0)
    q_end = tl.load(cu_seqlens_ptr + safe_request + 1, mask=valid_request, other=0)
    chunk_start = end_position - (row - q_start)
    ring_slot = tl.load(ring_slots_ptr + safe_request, mask=valid_request, other=-1)
    valid_ring = (ring_slot >= 0) & (ring_slot < num_ring_slots)
    valid_row = (
        valid_request & (row >= q_start) & (row < q_end) & (end_position >= RATIO - 1)
    )
    acc = tl.zeros((BLOCK_I,), dtype=tl.float32)
    for g in tl.static_range(RATIO):
        position = end_position - (RATIO - 1 - g)
        use_raw = position >= chunk_start
        raw_row = q_start + position - chunk_start
        raw_vals = tl.load(
            raw_k_ptr + raw_row.to(tl.int64) * stride_rawk_row + dims,
            mask=valid_row & use_raw & (raw_row >= q_start) & (raw_row < q_end) & (raw_row < num_rows) & in_dim,
            other=0.0,
        ).to(tl.float32)
        ring_vals = tl.load(
            ring_ptr + tl.maximum(ring_slot, 0).to(tl.int64) * stride_ring_slot
            + (position % RING_CAPACITY) * stride_ring_row + dims,
            mask=valid_row & (~use_raw) & valid_ring & in_dim,
            other=0.0,
        ).to(tl.float32)
        acc += tl.where(use_raw, raw_vals, ring_vals)
    pooled = acc / RATIO
    first_position = tl.where(valid_row, end_position - RATIO + 1, 0)

    # ---- 3. norm + rope of the pooled key at the group's first position -> compressed slab row
    pair = dims % ROTARY_HALF
    partner = tl.where(dims < ROTARY_HALF, dims + ROTARY_HALF, dims - ROTARY_HALF)
    in_rot = dims < 2 * ROTARY_HALF
    partner = tl.where(in_rot, partner, dims)
    kw = tl.load(k_norm_w_ptr + dims, mask=in_dim, other=0.0).to(tl.float32) + 1.0
    kw_p = tl.load(k_norm_w_ptr + partner, mask=in_dim, other=0.0).to(tl.float32) + 1.0
    # the partner lanes of the pooled row: gather via a second masked reduction-free path --
    # pooled lives in registers, so rebuild the partner permutation with a tiny exchange:
    # NeoX halves swap = (dims + ROTARY_HALF) % (2*ROTARY_HALF) within the rotary span.
    pooled_partner = tl.sum(tl.where(dims[None, :] == partner[:, None], pooled[None, :], 0.0), axis=1)
    cbase = cos_sin_ptr + first_position.to(tl.int64) * stride_cos_sin_row
    cos = tl.load(cbase + pair, mask=live & in_rot, other=1.0)
    sin = tl.load(cbase + ROTARY_HALF + pair, mask=live & in_rot, other=0.0)
    cmp_val = _norm_rope_row(pooled, kw, kw_p, pooled_partner, cos, sin, dims, eps, IDX_DIM, ROTARY_HALF)
    dest = tl.load(cmp_rows_ptr + row, mask=live, other=-1)
    tl.store(
        cmp_ptr + tl.maximum(dest, 0).to(tl.int64) * stride_cmp_row + dims,
        cmp_val.to(cmp_ptr.dtype.element_ty),
        mask=live & (dest >= 0) & in_dim,
    )

    # ---- 4. raw index key -> pending ring row (after the compress read, disjoint window)
    rrow = tl.load(ring_rows_ptr + row, mask=live, other=-1)
    raw_self = tl.load(raw_k_ptr + row * stride_rawk_row + dims, mask=live & in_dim, other=0.0)
    rblock = tl.maximum(rrow, 0) // RING_CAPACITY
    rtok = tl.maximum(rrow, 0) % RING_CAPACITY
    tl.store(
        ring_ptr + rblock.to(tl.int64) * stride_ring_slot + rtok * stride_ring_row + dims,
        raw_self,
        mask=live & (rrow >= 0) & (rrow < num_ring_slots * RING_CAPACITY) & in_dim,
    )

    # ---- 5. query-side indexer rows: norm + rope at this row's position, all heads
    qw = tl.load(q_norm_w_ptr + dims, mask=in_dim, other=0.0).to(tl.float32) + 1.0
    qw_p = tl.load(q_norm_w_ptr + partner, mask=in_dim, other=0.0).to(tl.float32) + 1.0
    qbase = cos_sin_ptr + end_position.to(tl.int64) * stride_cos_sin_row
    qcos = tl.load(qbase + pair, mask=live & in_rot, other=1.0)
    qsin = tl.load(qbase + ROTARY_HALF + pair, mask=live & in_rot, other=0.0)
    for h in tl.static_range(Q_HEADS):
        xb = idx_q_ptr + row * stride_idxq_row + h * IDX_DIM
        x = tl.load(xb + dims, mask=live & in_dim, other=0.0).to(tl.float32)
        xp = tl.load(xb + partner, mask=live & in_dim, other=0.0).to(tl.float32)
        yq = _norm_rope_row(x, qw, qw_p, xp, qcos, qsin, dims, eps, IDX_DIM, ROTARY_HALF)
        tl.store(
            q_out_ptr + row * stride_qout_row + h * IDX_DIM + dims,
            yq.to(q_out_ptr.dtype.element_ty),
            mask=live & in_dim,
        )


def qsa_index_write_fused(
    *, k, v, out_loc, k_cache, v_cache, k_scale, v_scale,
    raw_k, idx_q, q_out, ring, ring_slots, token_to_req, cu_seqlens, positions,
    cmp_cache, cmp_rows, ring_rows, cos_sin, k_norm_weight, q_norm_weight, eps, ratio,
) -> None:
    """One launch for the QSA decode index-write prologue (see module docstring)."""
    T = k.shape[0]
    if T == 0:
        return
    slots, kv_heads, head_dim = k_cache.shape
    q_heads = q_out.shape[1]
    idx_dim = raw_k.shape[1]
    rotary_dim = cos_sin.shape[1]
    assert T <= 4 and T + ratio - 1 <= ring.shape[1], (T, ratio, ring.shape)
    assert k_cache.dtype == torch.float8_e4m3fn and v_cache.dtype == torch.float8_e4m3fn
    assert k.shape == (T, kv_heads * head_dim) and v.shape == k.shape
    assert q_out.shape == (T, q_heads, idx_dim) and idx_q.shape == (T, q_heads * idx_dim)
    assert ring.ndim == 3 and ring.shape[2] == idx_dim and ring.stride(2) == 1
    assert cmp_cache.stride(1) == 1 and cos_sin.is_contiguous()
    assert k_scale.shape == (slots, kv_heads) and k_scale.dtype == torch.float32
    _qsa_index_write_kernel[(T,)](
        k, v, out_loc, k_cache, v_cache, k_scale, v_scale,
        k.stride(0), v.stride(0), k_cache.stride(0), k_cache.stride(1), k_scale.stride(0), slots,
        raw_k, raw_k.stride(0),
        idx_q, idx_q.stride(0),
        q_out, q_out.stride(0),
        ring, ring.stride(0), ring.stride(1), ring.shape[0],
        ring_slots, token_to_req, cu_seqlens, positions, cu_seqlens.shape[0] - 1,
        cmp_cache, cmp_cache.stride(0), cmp_rows, ring_rows,
        cos_sin, cos_sin.stride(0), k_norm_weight, q_norm_weight, float(eps),
        T,
        KV_HEADS=kv_heads, HEAD_DIM=head_dim, BLOCK_KV=triton.next_power_of_2(head_dim),
        Q_HEADS=q_heads, IDX_DIM=idx_dim, BLOCK_I=triton.next_power_of_2(idx_dim),
        RING_CAPACITY=ring.shape[1], RATIO=ratio, ROTARY_HALF=rotary_dim // 2,
        num_warps=4,
    )


@triton.jit
def _qk_norm_rope_kernel(
    q_ptr, k_ptr, positions_ptr, cos_sin_ptr, qw_ptr, kw_ptr, eps,
    stride_q_row, stride_k_row, stride_cos_sin_row, num_rows,
    Q_HEADS: tl.constexpr, KV_HEADS: tl.constexpr, HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr, ROTARY_HALF: tl.constexpr,
):
    """Attention-side per-head (1+w) RMSNorm + partial NeoX rope of q [T, Q_HEADS, D] and
    k [T, KV_HEADS, D], IN PLACE, one program per token row (replaces q_norm + k_norm + rope)."""
    row = tl.program_id(0).to(tl.int64)
    live = row < num_rows
    dims = tl.arange(0, BLOCK_D)
    in_dim = dims < HEAD_DIM
    in_rot = dims < 2 * ROTARY_HALF
    pair = dims % ROTARY_HALF
    partner = tl.where(dims < ROTARY_HALF, dims + ROTARY_HALF, dims - ROTARY_HALF)
    partner = tl.where(in_rot, partner, dims)
    position = tl.load(positions_ptr + row, mask=live, other=0).to(tl.int64)
    cbase = cos_sin_ptr + position * stride_cos_sin_row
    cos = tl.load(cbase + pair, mask=live & in_rot, other=1.0)
    sin = tl.load(cbase + ROTARY_HALF + pair, mask=live & in_rot, other=0.0)
    qw = tl.load(qw_ptr + dims, mask=in_dim, other=0.0).to(tl.float32) + 1.0
    qw_p = tl.load(qw_ptr + partner, mask=in_dim, other=0.0).to(tl.float32) + 1.0
    kw = tl.load(kw_ptr + dims, mask=in_dim, other=0.0).to(tl.float32) + 1.0
    kw_p = tl.load(kw_ptr + partner, mask=in_dim, other=0.0).to(tl.float32) + 1.0
    m = live & in_dim
    for h in tl.static_range(Q_HEADS):
        base = q_ptr + row * stride_q_row + h * HEAD_DIM
        x = tl.load(base + dims, mask=m, other=0.0).to(tl.float32)
        xp = tl.load(base + partner, mask=m, other=0.0).to(tl.float32)
        y = _norm_rope_row(x, qw, qw_p, xp, cos, sin, dims, eps, HEAD_DIM, ROTARY_HALF)
        tl.store(base + dims, y.to(q_ptr.dtype.element_ty), mask=m)
    for h in tl.static_range(KV_HEADS):
        base = k_ptr + row * stride_k_row + h * HEAD_DIM
        x = tl.load(base + dims, mask=m, other=0.0).to(tl.float32)
        xp = tl.load(base + partner, mask=m, other=0.0).to(tl.float32)
        y = _norm_rope_row(x, kw, kw_p, xp, cos, sin, dims, eps, HEAD_DIM, ROTARY_HALF)
        tl.store(base + dims, y.to(k_ptr.dtype.element_ty), mask=m)


def qsa_qk_norm_rope_fused(q, k, positions, cos_sin, q_norm_weight, k_norm_weight, eps) -> None:
    """In-place gemma (1+w) RMSNorm per head + NeoX rope on ``q`` [T, HQ, D] and ``k`` [T, HK, D]
    (contiguous, T <= 4): one launch instead of q_norm + k_norm + rope."""
    T, hq, d = q.shape
    hk = k.shape[1]
    assert q.is_contiguous() and k.is_contiguous() and k.shape == (T, hk, d)
    assert cos_sin.is_contiguous() and cos_sin.shape[1] % 2 == 0 and cos_sin.shape[1] <= d
    if T == 0:
        return
    _qk_norm_rope_kernel[(T,)](
        q, k, positions, cos_sin, q_norm_weight, k_norm_weight, float(eps),
        q.stride(0), k.stride(0), cos_sin.stride(0), T,
        Q_HEADS=hq, KV_HEADS=hk, HEAD_DIM=d, BLOCK_D=triton.next_power_of_2(d),
        ROTARY_HALF=cos_sin.shape[1] // 2, num_warps=4,
    )


@triton.jit
def _plan_index_writes_kernel(
    out_loc_ptr, positions_ptr, token_to_req_ptr, ring_slots_ptr, cu_seqlens_ptr,
    cmp_rows_ptr, ring_rows_ptr, scratch_base, num_rows,
    RATIO: tl.constexpr, RING_CAPACITY: tl.constexpr, BLOCK: tl.constexpr,
):
    rows = tl.arange(0, BLOCK)
    live = rows < num_rows
    out_loc = tl.load(out_loc_ptr + rows, mask=live, other=0).to(tl.int64)
    pos = tl.load(positions_ptr + rows, mask=live, other=0).to(tl.int64)
    req = tl.load(token_to_req_ptr + rows, mask=live, other=0).to(tl.int64)
    slot = tl.load(ring_slots_ptr + req, mask=live, other=0).to(tl.int64)
    ends = tl.load(cu_seqlens_ptr + req + 1, mask=live, other=0).to(tl.int64)
    closing = out_loc % RATIO == RATIO - 1
    cmp = tl.where(closing, out_loc // RATIO, scratch_base + slot)
    keep = rows.to(tl.int64) >= ends - RING_CAPACITY
    ring = tl.where(keep, slot * RING_CAPACITY + pos % RING_CAPACITY, -1)
    tl.store(cmp_rows_ptr + rows, cmp.to(tl.int32), mask=live)
    tl.store(ring_rows_ptr + rows, ring.to(tl.int32), mask=live)


def qsa_plan_index_writes(out_loc, positions, token_to_req, ring_slots, cu_seqlens, *,
                          ratio, ring_capacity, scratch_base):
    """Device plan of the compressed-slab row and ring row per token (one launch; the torch
    version is ~20 elementwise ops per step). Returns (cmp_rows, ring_rows) int32 [T]."""
    T = out_loc.shape[0]
    cmp_rows = torch.empty(T, dtype=torch.int32, device=out_loc.device)
    ring_rows = torch.empty(T, dtype=torch.int32, device=out_loc.device)
    if T:
        _plan_index_writes_kernel[(1,)](
            out_loc, positions, token_to_req, ring_slots, cu_seqlens, cmp_rows, ring_rows,
            int(scratch_base), T, RATIO=ratio, RING_CAPACITY=ring_capacity,
            BLOCK=max(16, triton.next_power_of_2(T)), num_warps=1,
        )
    return cmp_rows, ring_rows


__all__ = ["qsa_index_write_fused", "qsa_plan_index_writes", "qsa_qk_norm_rope_fused"]
