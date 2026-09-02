"""Inline-dequant NVFP4 fused-MoE Triton kernels.

These kernels read the NVFP4 expert cache (packed FP4 codes + fp8 block scales +
fp16 per-row global scale) *directly* and dequantize inside the GEMM K-loop, so the
grouped MoE GEMM never materializes a BF16 copy of the experts (no separate dequant
pass, no HBM round-trip of the 4x larger BF16 weights).

Layout, for an expert weight ``W[N, K]``:
  - ``packed[slot, n, k//2]`` uint8: two FP4 codes per byte, low nibble = even k.
  - ``scale[slot, n, k//16]`` fp8-e4m3: per 16-wide block scale.
  - ``global[slot, n]`` fp16: per-output-row scale (``weight_scale_2``).
  - ``W[n, k] = E2M1[code] * scale[n, k//16] * global[n]``.

The global scale is constant along K, so it is applied once after the K-loop.

Decode (M=1) is HBM-bandwidth bound. :func:`_decode_nvfp4_marlin_kernel` is the production
decode GEMV (int32 wide loads + deferred K reduction); :func:`_decode_nvfp4_moe_kernel` is
the original LUT-gather variant kept for A/B comparison. The marlin-style wide load lifts
MiniMax-M2 decode toward the RTX 5090 read-bandwidth ceiling (~87%); the residual gap is FP4
dequant ALU, which a swizzled-layout tensor-core path (marlin / flashinfer b12x) closes but
those need sm_80-99 / CUDA>=13 respectively.
"""

from __future__ import annotations

import functools

import torch
import triton
import triton.language as tl

from freetoken.kernel.triton.e4m3_compat import e4m3_native_cx, e4m3_u8_to_f32
from freetoken.kernel.triton.nvfp4_dequant import _nvfp4_pair_f32, _split8

_E2M1_VALUES = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
]


@functools.lru_cache(maxsize=None)
def _e2m1_lut(device_index: int) -> torch.Tensor:
    return torch.tensor(
        _E2M1_VALUES, dtype=torch.float32, device=torch.device("cuda", device_index)
    )


@triton.jit
def _decode_nvfp4_moe_kernel(
    a_ptr,             # [M, K] activations (compute dtype)
    packed_ptr,        # [S, N, K // 2] uint8
    scale_ptr,         # [S, N, K // 16] fp8-e4m3
    global_ptr,        # [S, N] fp16
    c_ptr,             # [M, TOP_K, N] output (compute dtype)
    topk_weights_ptr,  # [M, TOP_K] fp32
    topk_ids_ptr,      # [M, TOP_K] int32 -> cache slot
    lut_ptr,           # [16] fp32
    total_routes,
    N,
    K,
    stride_am, stride_ak,
    stride_pe, stride_pn, stride_pkb,
    stride_se, stride_sn, stride_sblk,
    stride_ge, stride_gn,
    stride_cm, stride_ck, stride_cn,
    stride_tw_m, stride_tw_k,
    stride_tid_m, stride_tid_k,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_KB: tl.constexpr,  # bytes processed per K-iter (covers 2*KB k-values)
    TOP_K: tl.constexpr,
    A_ROW_IS_ROUTE: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    compute_type: tl.constexpr,
    ZERO_ON_SKIP: tl.constexpr = False,
):
    """Original LUT-gather serial-K decode GEMV. Kept as the A/B baseline; the production
    decode path is :func:`_decode_nvfp4_marlin_kernel`."""
    route_id = tl.program_id(0)
    n_block_id = tl.program_id(1)
    token_id = route_id // TOP_K
    route_k = route_id - token_id * TOP_K

    offs_n = n_block_id * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    n_mask = offs_n < N

    slot = tl.load(topk_ids_ptr + token_id * stride_tid_m + route_k * stride_tid_k).to(tl.int64)
    # Routes the hybrid backend assigned to the CPU carry slot -1: nothing to compute here.
    # ZERO_ON_SKIP writes zeros so a following sum-reduce over all routes stays correct
    # (gemm2); gemm1 leaves its row untouched (only its own route's gemm2 would read it).
    if slot < 0:
        if ZERO_ON_SKIP:
            c_skip = c_ptr + token_id * stride_cm + route_k * stride_ck + offs_n * stride_cn
            tl.store(
                c_skip,
                tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32).to(compute_type),
                mask=(route_id < total_routes) & n_mask,
            )
        return
    a_row = route_id if A_ROW_IS_ROUTE else token_id
    a_base = a_ptr + a_row * stride_am

    offs_kb = tl.arange(0, BLOCK_SIZE_KB)
    K_BYTES = K // 2
    accumulator = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)

    packed_slot = packed_ptr + slot * stride_pe
    scale_slot = scale_ptr + slot * stride_se
    for kb_start in range(0, tl.cdiv(K_BYTES, BLOCK_SIZE_KB)):
        byte_idx = kb_start * BLOCK_SIZE_KB + offs_kb
        byte_mask = byte_idx < K_BYTES

        p_ptrs = packed_slot + offs_n[None, :] * stride_pn + byte_idx[:, None] * stride_pkb
        bytes_ = tl.load(
            p_ptrs, mask=byte_mask[:, None] & n_mask[None, :], other=0
        ).to(tl.int32)
        lo = bytes_ & 0xF
        hi = (bytes_ >> 4) & 0xF
        b_lo = tl.load(lut_ptr + lo)
        b_hi = tl.load(lut_ptr + hi)

        sblk = byte_idx // 8
        s_ptrs = scale_slot + offs_n[None, :] * stride_sn + sblk[:, None] * stride_sblk
        s_mask = byte_mask[:, None] & n_mask[None, :]
        if e4m3_native_cx():
            scale = tl.load(s_ptrs, mask=s_mask, other=0.0).to(tl.float32)
        else:
            scale = e4m3_u8_to_f32(tl.load(s_ptrs, mask=s_mask, other=0))
        b_lo = b_lo * scale
        b_hi = b_hi * scale

        a_lo = tl.load(a_base + (2 * byte_idx) * stride_ak, mask=byte_mask, other=0.0).to(tl.float32)
        a_hi = tl.load(a_base + (2 * byte_idx + 1) * stride_ak, mask=byte_mask, other=0.0).to(tl.float32)
        accumulator += tl.sum(a_lo[:, None] * b_lo, axis=0)
        accumulator += tl.sum(a_hi[:, None] * b_hi, axis=0)

    g = tl.load(global_ptr + slot * stride_ge + offs_n * stride_gn, mask=n_mask, other=0.0).to(tl.float32)
    accumulator = accumulator * g

    if MUL_ROUTED_WEIGHT:
        weight = tl.load(topk_weights_ptr + token_id * stride_tw_m + route_k * stride_tw_k)
        accumulator = accumulator * weight

    c_ptrs = c_ptr + token_id * stride_cm + route_k * stride_ck + offs_n * stride_cn
    tl.store(c_ptrs, accumulator.to(compute_type), mask=(route_id < total_routes) & n_mask)


@triton.jit
def _decode_nvfp4_marlin_kernel(
    a_ptr,             # [M, K] activations (compute dtype)
    packed_ptr,        # [S, N, K // 8] int32 (8 fp4 codes per word, nibble j -> k=8*w+j)
    scale_ptr,         # [S, N, K // 16] fp8-e4m3
    global_ptr,        # [S, N] fp16
    c_ptr,             # [M, TOP_K, N] output (compute dtype)
    topk_weights_ptr,  # [M, TOP_K] fp32
    topk_ids_ptr,      # [M, TOP_K] int32 -> cache slot
    lut_ptr,           # [16] fp32
    total_routes,
    N,
    K,
    stride_am, stride_ak,
    stride_pe, stride_pn, stride_pkw,
    stride_se, stride_sn, stride_sblk,
    stride_ge, stride_gn,
    stride_cm, stride_ck, stride_cn,
    stride_tw_m, stride_tw_k,
    stride_tid_m, stride_tid_k,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_KW: tl.constexpr,  # int32 words per K-iter (covers 8*KW k-values)
    TOP_K: tl.constexpr,
    A_ROW_IS_ROUTE: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    compute_type: tl.constexpr,
    ZERO_ON_SKIP: tl.constexpr = False,
    FUSE_SILU_MUL: tl.constexpr = False,
    act_off=0,
    ARITH_DEQUANT: tl.constexpr = True,
):
    """Marlin-style NVFP4 decode GEMV: wide int32 weight loads + deferred reduction.

    ``ARITH_DEQUANT``: decode the e2m1 codes with the branchless fp16-subnormal bit trick
    (``_nvfp4_pair_f32``: value * 2^-14, the 2^14 folded into the block scale -- exact, so the
    result is bit-identical to the LUT gather) and load each K-tile's activations once as a
    contiguous vector split into the 8 nibble slices in registers (``_split8``) instead of 8
    strided gathers per word. The LUT gather serialized the load/store unit (the dense head
    measured ~36% -> ~64% of the bandwidth roofline with the same change).

    ``FUSE_SILU_MUL`` (gemm2): ``a`` is gemm1's raw ``[route, 2I]`` gate|up output and the
    activation ``silu(gate) * up`` is formed on load (``act_off`` = I, the up half's column
    offset; rounded to the compute dtype exactly like the separate act kernel's output), so
    the act launch and its intermediate round trip disappear.

    Versus :func:`_decode_nvfp4_moe_kernel` (which loads the packed codes one byte at a
    time and runs a cross-lane reduction *every* K-iter) this:

      * Loads the FP4 codes as **int32 words** (8 codes / 4 bytes per element) so the HBM
        read issues wide coalesced transactions -- the single biggest lever for the
        gate/up GEMM, whose mem-only ceiling jumps from ~65% to ~85% of peak read BW.
      * **Defers** the K reduction: the ``[BLOCK_KW, BLOCK_N]`` partial accumulates across
        *all* K-iters and is reduced once at the end, removing the per-iter ``tl.sum``
        barrier so the weight loads of successive iters pipeline.

    The e2m1 codes share one fp8 block scale per 16 k-values (== 2 words), applied per
    word. Mirrors the fast kernel's route/tile mapping and epilogue so it is a drop-in
    decode GEMV (CUDA-graph safe: fixed shapes, no host sync)."""
    route_id = tl.program_id(0)
    n_block_id = tl.program_id(1)
    token_id = route_id // TOP_K
    route_k = route_id - token_id * TOP_K

    offs_n = n_block_id * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    n_mask = offs_n < N

    slot = tl.load(topk_ids_ptr + token_id * stride_tid_m + route_k * stride_tid_k).to(tl.int64)
    # Routes the hybrid backend assigned to the CPU carry slot -1: nothing to compute here.
    # ZERO_ON_SKIP writes zeros so a following sum-reduce over all routes stays correct
    # (gemm2); gemm1 leaves its row untouched (only its own route's gemm2 would read it).
    if slot < 0:
        if ZERO_ON_SKIP:
            c_skip = c_ptr + token_id * stride_cm + route_k * stride_ck + offs_n * stride_cn
            tl.store(
                c_skip,
                tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32).to(compute_type),
                mask=(route_id < total_routes) & n_mask,
            )
        return
    a_row = route_id if A_ROW_IS_ROUTE else token_id
    a_base = a_ptr + a_row * stride_am

    offs_kw = tl.arange(0, BLOCK_SIZE_KW)
    K_WORDS = K // 8
    partial = tl.zeros((BLOCK_SIZE_KW, BLOCK_SIZE_N), dtype=tl.float32)

    packed_slot = packed_ptr + slot * stride_pe
    scale_slot = scale_ptr + slot * stride_se
    for kw_start in range(0, tl.cdiv(K_WORDS, BLOCK_SIZE_KW)):
        widx = kw_start * BLOCK_SIZE_KW + offs_kw
        w_mask = widx < K_WORDS

        word = tl.load(
            packed_slot + offs_n[None, :] * stride_pn + widx[:, None] * stride_pkw,
            mask=w_mask[:, None] & n_mask[None, :], other=0,
        )
        # 8 codes/word fall in the same or adjacent 16-wide block -> one scale per word.
        s_ptrs = scale_slot + offs_n[None, :] * stride_sn + (widx[:, None] // 2) * stride_sblk
        s_mask = w_mask[:, None] & n_mask[None, :]
        if e4m3_native_cx():
            scale = tl.load(s_ptrs, mask=s_mask, other=0.0).to(tl.float32)
        else:
            scale = e4m3_u8_to_f32(tl.load(s_ptrs, mask=s_mask, other=0))

        kbase = 8 * widx
        acc_w = tl.zeros((BLOCK_SIZE_KW, BLOCK_SIZE_N), dtype=tl.float32)
        if ARITH_DEQUANT:
            # one contiguous activation tile per K-tile: k = 8*kw_start*BLOCK_KW + [0, 8*BLOCK_KW)
            offs_k = kw_start * (8 * BLOCK_SIZE_KW) + tl.arange(0, 8 * BLOCK_SIZE_KW)
            k_mask = offs_k < K
            if FUSE_SILU_MUL:
                g_v = tl.load(a_base + offs_k * stride_ak, mask=k_mask, other=0.0).to(tl.float32)
                u_v = tl.load(a_base + (act_off + offs_k) * stride_ak, mask=k_mask, other=0.0).to(tl.float32)
                a_v = ((g_v / (1.0 + tl.exp(-g_v))) * u_v).to(compute_type).to(tl.float32)
            else:
                a_v = tl.load(a_base + offs_k * stride_ak, mask=k_mask, other=0.0).to(tl.float32)
            a0, a1, a2, a3, a4, a5, a6, a7 = _split8(a_v, BLOCK_SIZE_KW)
            b_lo, b_hi = _nvfp4_pair_f32(word)        # nibbles 0 / 4
            acc_w += a0[:, None] * b_lo + a4[:, None] * b_hi
            b_lo, b_hi = _nvfp4_pair_f32(word >> 4)   # 1 / 5
            acc_w += a1[:, None] * b_lo + a5[:, None] * b_hi
            b_lo, b_hi = _nvfp4_pair_f32(word >> 8)   # 2 / 6
            acc_w += a2[:, None] * b_lo + a6[:, None] * b_hi
            b_lo, b_hi = _nvfp4_pair_f32(word >> 12)  # 3 / 7
            acc_w += a3[:, None] * b_lo + a7[:, None] * b_hi
            partial += acc_w * (scale * 16384.0)  # 2^14: exact, undoes the pair dequant's 2^-14
        else:
            for j in tl.static_range(8):
                code = (word >> (4 * j)) & 0xF
                b = tl.load(lut_ptr + code)
                if FUSE_SILU_MUL:
                    g_j = tl.load(a_base + (kbase + j) * stride_ak, mask=w_mask, other=0.0).to(tl.float32)
                    u_j = tl.load(a_base + (act_off + kbase + j) * stride_ak, mask=w_mask, other=0.0).to(tl.float32)
                    a_j = ((g_j / (1.0 + tl.exp(-g_j))) * u_j).to(compute_type).to(tl.float32)
                else:
                    a_j = tl.load(a_base + (kbase + j) * stride_ak, mask=w_mask, other=0.0).to(tl.float32)
                acc_w += a_j[:, None] * b
            partial += acc_w * scale

    accumulator = tl.sum(partial, axis=0)
    g = tl.load(global_ptr + slot * stride_ge + offs_n * stride_gn, mask=n_mask, other=0.0).to(tl.float32)
    accumulator = accumulator * g

    if MUL_ROUTED_WEIGHT:
        weight = tl.load(topk_weights_ptr + token_id * stride_tw_m + route_k * stride_tw_k)
        accumulator = accumulator * weight

    c_ptrs = c_ptr + token_id * stride_cm + route_k * stride_ck + offs_n * stride_cn
    tl.store(c_ptrs, accumulator.to(compute_type), mask=(route_id < total_routes) & n_mask)


@triton.jit
def _decode_nvfp4_marlin_dedup_kernel(
    a_ptr,             # [rows, K] activations (compute dtype); row = route or token
    packed_ptr,        # [S, N, K // 8] int32
    scale_ptr,         # [S, N, K // 16] fp8-e4m3
    global_ptr,        # [S, N] fp16
    c_ptr,             # [M, TOP_K, N] output (compute dtype)
    topk_weights_ptr,  # [M, TOP_K] fp32
    uniq_slots_ptr,    # [U] int32 cache slot of the u-th distinct expert (-1: none)
    pairs_ptr,         # [U, MAXB] int32 route indices (token * TOP_K + k) sharing it, -1 padded
    lut_ptr,           # [16] fp32
    N,
    K,
    stride_am, stride_ak,
    stride_pe, stride_pn, stride_pkw,
    stride_se, stride_sn, stride_sblk,
    stride_ge, stride_gn,
    stride_cm, stride_ck, stride_cn,
    stride_tw_m, stride_tw_k,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_KW: tl.constexpr,
    TOP_K: tl.constexpr,
    MAXB: tl.constexpr,
    A_ROW_IS_ROUTE: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    compute_type: tl.constexpr,
):
    """:func:`_decode_nvfp4_marlin_kernel` with one program per DISTINCT expert: the weight
    tile is loaded and dequantized once and applied to every route (<= MAXB, one per token)
    that selected the expert, so a speculative verify's 3-4 rows read each routed expert's
    bytes once instead of once per route (~30% fewer HBM bytes at k=2, and less dequant
    ALU). Same tile geometry and epilogue as the per-route kernel."""
    u = tl.program_id(0)
    n_block_id = tl.program_id(1)
    slot = tl.load(uniq_slots_ptr + u).to(tl.int64)
    if slot < 0:
        return
    offs_n = n_block_id * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    n_mask = offs_n < N

    p0 = tl.load(pairs_ptr + u * MAXB + 0)
    p1 = tl.load(pairs_ptr + u * MAXB + 1) if MAXB > 1 else -1
    p2 = tl.load(pairs_ptr + u * MAXB + 2) if MAXB > 2 else -1
    p3 = tl.load(pairs_ptr + u * MAXB + 3) if MAXB > 3 else -1
    v0 = p0 >= 0
    v1 = p1 >= 0
    v2 = p2 >= 0
    v3 = p3 >= 0
    t0 = tl.maximum(p0, 0) // TOP_K
    t1 = tl.maximum(p1, 0) // TOP_K
    t2 = tl.maximum(p2, 0) // TOP_K
    t3 = tl.maximum(p3, 0) // TOP_K
    if A_ROW_IS_ROUTE:
        r0 = tl.maximum(p0, 0)
        r1 = tl.maximum(p1, 0)
        r2 = tl.maximum(p2, 0)
        r3 = tl.maximum(p3, 0)
    else:
        r0 = t0
        r1 = t1
        r2 = t2
        r3 = t3

    offs_kw = tl.arange(0, BLOCK_SIZE_KW)
    K_WORDS = K // 8
    part0 = tl.zeros((BLOCK_SIZE_KW, BLOCK_SIZE_N), dtype=tl.float32)
    part1 = tl.zeros((BLOCK_SIZE_KW, BLOCK_SIZE_N), dtype=tl.float32)
    part2 = tl.zeros((BLOCK_SIZE_KW, BLOCK_SIZE_N), dtype=tl.float32)
    part3 = tl.zeros((BLOCK_SIZE_KW, BLOCK_SIZE_N), dtype=tl.float32)

    packed_slot = packed_ptr + slot * stride_pe
    scale_slot = scale_ptr + slot * stride_se
    for kw_start in range(0, tl.cdiv(K_WORDS, BLOCK_SIZE_KW)):
        widx = kw_start * BLOCK_SIZE_KW + offs_kw
        w_mask = widx < K_WORDS
        word = tl.load(
            packed_slot + offs_n[None, :] * stride_pn + widx[:, None] * stride_pkw,
            mask=w_mask[:, None] & n_mask[None, :], other=0,
        )
        s_ptrs = scale_slot + offs_n[None, :] * stride_sn + (widx[:, None] // 2) * stride_sblk
        s_mask = w_mask[:, None] & n_mask[None, :]
        if e4m3_native_cx():
            scale = tl.load(s_ptrs, mask=s_mask, other=0.0).to(tl.float32)
        else:
            scale = e4m3_u8_to_f32(tl.load(s_ptrs, mask=s_mask, other=0))
        kbase = 8 * widx
        acc0 = tl.zeros((BLOCK_SIZE_KW, BLOCK_SIZE_N), dtype=tl.float32)
        acc1 = tl.zeros((BLOCK_SIZE_KW, BLOCK_SIZE_N), dtype=tl.float32)
        acc2 = tl.zeros((BLOCK_SIZE_KW, BLOCK_SIZE_N), dtype=tl.float32)
        acc3 = tl.zeros((BLOCK_SIZE_KW, BLOCK_SIZE_N), dtype=tl.float32)
        for j in tl.static_range(8):
            code = (word >> (4 * j)) & 0xF
            b = tl.load(lut_ptr + code)
            koff = (kbase + j) * stride_ak
            a0 = tl.load(a_ptr + r0 * stride_am + koff, mask=w_mask & v0, other=0.0).to(tl.float32)
            acc0 += a0[:, None] * b
            if MAXB > 1:
                a1 = tl.load(a_ptr + r1 * stride_am + koff, mask=w_mask & v1, other=0.0).to(tl.float32)
                acc1 += a1[:, None] * b
            if MAXB > 2:
                a2 = tl.load(a_ptr + r2 * stride_am + koff, mask=w_mask & v2, other=0.0).to(tl.float32)
                acc2 += a2[:, None] * b
            if MAXB > 3:
                a3 = tl.load(a_ptr + r3 * stride_am + koff, mask=w_mask & v3, other=0.0).to(tl.float32)
                acc3 += a3[:, None] * b
        part0 += acc0 * scale
        if MAXB > 1:
            part1 += acc1 * scale
        if MAXB > 2:
            part2 += acc2 * scale
        if MAXB > 3:
            part3 += acc3 * scale

    g = tl.load(global_ptr + slot * stride_ge + offs_n * stride_gn, mask=n_mask, other=0.0).to(tl.float32)
    # epilogue per route: global scale, optional router weight, store c[token, k, n]
    o0 = tl.sum(part0, axis=0) * g
    k0 = tl.maximum(p0, 0) - t0 * TOP_K
    if MUL_ROUTED_WEIGHT:
        o0 = o0 * tl.load(topk_weights_ptr + t0 * stride_tw_m + k0 * stride_tw_k, mask=v0, other=0.0)
    tl.store(c_ptr + t0 * stride_cm + k0 * stride_ck + offs_n * stride_cn, o0.to(compute_type), mask=n_mask & v0)
    if MAXB > 1:
        o1 = tl.sum(part1, axis=0) * g
        k1 = tl.maximum(p1, 0) - t1 * TOP_K
        if MUL_ROUTED_WEIGHT:
            o1 = o1 * tl.load(topk_weights_ptr + t1 * stride_tw_m + k1 * stride_tw_k, mask=v1, other=0.0)
        tl.store(c_ptr + t1 * stride_cm + k1 * stride_ck + offs_n * stride_cn, o1.to(compute_type), mask=n_mask & v1)
    if MAXB > 2:
        o2 = tl.sum(part2, axis=0) * g
        k2 = tl.maximum(p2, 0) - t2 * TOP_K
        if MUL_ROUTED_WEIGHT:
            o2 = o2 * tl.load(topk_weights_ptr + t2 * stride_tw_m + k2 * stride_tw_k, mask=v2, other=0.0)
        tl.store(c_ptr + t2 * stride_cm + k2 * stride_ck + offs_n * stride_cn, o2.to(compute_type), mask=n_mask & v2)
    if MAXB > 3:
        o3 = tl.sum(part3, axis=0) * g
        k3 = tl.maximum(p3, 0) - t3 * TOP_K
        if MUL_ROUTED_WEIGHT:
            o3 = o3 * tl.load(topk_weights_ptr + t3 * stride_tw_m + k3 * stride_tw_k, mask=v3, other=0.0)
        tl.store(c_ptr + t3 * stride_cm + k3 * stride_ck + offs_n * stride_cn, o3.to(compute_type), mask=n_mask & v3)


@triton.jit
def _prefill_nvfp4_moe_kernel(
    a_ptr,             # [M, K] activations
    packed_ptr,        # [S, N, K // 2] uint8
    scale_ptr,         # [S, N, K // 16] fp8-e4m3
    global_ptr,        # [S, N] fp16
    c_ptr,             # [num_valid_tokens, N] output (flat over M*top_k)
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,    # cache slot per M-block
    num_tokens_post_padded_ptr,
    lut_ptr,
    N,
    K,
    EM,
    num_valid_tokens,
    stride_am, stride_ak,
    stride_pe, stride_pn, stride_pkb,
    stride_se, stride_sn, stride_sblk,
    stride_ge, stride_gn,
    stride_cm, stride_cn,
    stride_tw,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_KB: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_kb = tl.arange(0, BLOCK_SIZE_KB)
    a_ptrs_lo = a_ptr + (offs_token[:, None] // top_k * stride_am + (2 * offs_kb)[None, :] * stride_ak)
    a_ptrs_hi = a_ptr + (offs_token[:, None] // top_k * stride_am + (2 * offs_kb + 1)[None, :] * stride_ak)

    slot = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    packed_base = packed_ptr + slot * stride_pe + offs_bn[None, :] * stride_pn
    scale_base = scale_ptr + slot * stride_se + offs_bn[None, :] * stride_sn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    K_BYTES = K // 2
    for kb in range(0, tl.cdiv(K_BYTES, BLOCK_SIZE_KB)):
        byte_idx = kb * BLOCK_SIZE_KB + offs_kb
        byte_mask = byte_idx < K_BYTES

        p_ptrs = packed_base + byte_idx[:, None] * stride_pkb
        bytes_ = tl.load(p_ptrs, mask=byte_mask[:, None], other=0).to(tl.int32)
        lo = bytes_ & 0xF
        hi = (bytes_ >> 4) & 0xF
        sblk = byte_idx // 8
        s_ptrs = scale_base + sblk[:, None] * stride_sblk
        if e4m3_native_cx():
            scale = tl.load(s_ptrs, mask=byte_mask[:, None], other=0.0).to(tl.float32)
        else:
            scale = e4m3_u8_to_f32(tl.load(s_ptrs, mask=byte_mask[:, None], other=0))
        b_lo = tl.load(lut_ptr + lo) * scale  # [BLOCK_KB, BLOCK_N]
        b_hi = tl.load(lut_ptr + hi) * scale

        a_lo = tl.load(a_ptrs_lo, mask=token_mask[:, None] & byte_mask[None, :], other=0.0)
        a_hi = tl.load(a_ptrs_hi, mask=token_mask[:, None] & byte_mask[None, :], other=0.0)
        accumulator += tl.dot(a_lo, b_lo.to(a_lo.dtype))
        accumulator += tl.dot(a_hi, b_hi.to(a_hi.dtype))

        a_ptrs_lo += BLOCK_SIZE_KB * 2 * stride_ak
        a_ptrs_hi += BLOCK_SIZE_KB * 2 * stride_ak

    g = tl.load(global_ptr + slot * stride_ge + offs_bn * stride_gn).to(tl.float32)
    accumulator = accumulator * g[None, :]

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token * stride_tw, mask=token_mask, other=0)
        accumulator = accumulator * moe_weight[:, None]

    accumulator = accumulator.to(compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


__all__ = [
    "_decode_nvfp4_moe_kernel",
    "_decode_nvfp4_marlin_kernel",
    "_prefill_nvfp4_moe_kernel",
    "_e2m1_lut",
]
