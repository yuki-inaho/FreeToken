"""Per-tensor (per-output-row) FP8 W8A16 dense linear, shared across models.

The nvidia/modelopt ``MIXED_PRECISION`` checkpoints (Qwen3.5/3.6, ...) quantize the attention
and GatedDeltaNet projections to **per-tensor FP8** (``weight`` fp8-e4m3 + a single scalar
``weight_scale``; ``weight_bf16 = weight_fp8 * weight_scale``) while the routed experts are
NVFP4. FreeToken used to *dequantize these dense weights to bf16 at load* and run them through
cuBLAS bf16 GEMV -- which is the dominant decode cost (~54% of the step) because it reads
2x the bytes of the resident FP8 weight. This module keeps the weight FP8 and reads it
directly in a W8A16 kernel (bf16 activation, fp8 weight), halving that traffic.

Fused projections (q/k/v -> ``qkv_proj``; GDN qkv|z -> ``in_proj_qkvz``) concatenate parts
that each carry their *own* per-tensor scalar, so the scalar is broadcast to a per-output-row
vector ``weight_scale`` ``[N]`` (piecewise-constant after fusion) and applied per output row.
This is exactly equal to applying each part's scalar to its own rows.

Numerics: the kernel converts fp8->f32 and accumulates in fp32 (standard W8A16; strictly more
accurate than the previous ``weight.to(bf16) * scale`` materialization, which it replaces).
"""

from __future__ import annotations

import functools
import os
import re

import torch
import triton
import triton.language as tl
from freetoken.layers import BaseOP
from freetoken.layers.base import _concat_prefix

from freetoken.kernel.triton.e4m3_compat import (
    e4m3_kernel_view,
    e4m3_native,
    e4m3_native_cx,
    e4m3_u8_to_f32,
)

FP8 = torch.float8_e4m3fn
_TL_DTYPE = {torch.bfloat16: tl.bfloat16, torch.float16: tl.float16, torch.float32: tl.float32}

# Escape hatch: FREETOKEN_DEBUG_FP8_REF=1 swaps the triton kernels for a pure-torch dequant matmul
# (numeric reference / A-B debugging). Evaluated once; the kernels are the default.
_USE_REF = os.environ.get("FREETOKEN_DEBUG_FP8_REF") == "1"


# Row-wise _scaled_mm on sm_89 with torch < 2.12 launches its CUTLASS stream-K kernel off the
# current stream (pytorch/pytorch#177651, fixed by pytorch/pytorch@252bb4a; #182/#72/#220), and
# some builds (Windows) have no row-wise kernel (#227). Fallback: one tensor-wise GEMM per part.
def _torch_version() -> tuple[int, int]:
    m = re.match(r"(\d+)\.(\d+)", torch.__version__)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


@functools.cache
def rowwise_scaled_mm_ok() -> bool:
    """Whether row-wise ``torch._scaled_mm`` may be issued from a side stream on this GPU.
    Decided once per process, at load (never under graph capture). ``FREETOKEN_FP8_ROWWISE_MM=0/1``
    forces the answer."""
    forced = os.environ.get("FREETOKEN_FP8_ROWWISE_MM")
    if forced in ("0", "1"):
        return forced == "1"
    if not torch.cuda.is_available():
        return True
    from freetoken.gpu_select import assigned_visible_gpu

    idx = assigned_visible_gpu()
    dev = torch.device("cuda", torch.cuda.current_device() if idx is None else idx)
    if torch.cuda.get_device_capability(dev) == (8, 9) and _torch_version() < (2, 12):
        return False
    # Probe on the default stream (safe even where the launch ignores the current stream); a
    # build without the row-wise kernel raises here instead of at the first forward.
    try:
        with torch.cuda.device(dev), torch.cuda.stream(torch.cuda.default_stream(dev)):
            a = torch.zeros(16, 32, dtype=FP8, device=dev)
            b = torch.zeros(32, 32, dtype=FP8, device=dev)
            torch._scaled_mm(
                a, b.t(), scale_a=torch.ones(16, 1, device=dev),
                scale_b=torch.ones(1, 32, device=dev), out_dtype=torch.bfloat16,
            )
            torch.cuda.synchronize(dev)
    except RuntimeError:
        return False
    return True


def weight_scale_segments(weight_scale: torch.Tensor) -> list[tuple[int, int]]:
    """``[start, end)`` row ranges over which ``weight_scale`` is constant (the fused parts).
    Syncs; call at load."""
    s = weight_scale.detach().reshape(-1).float().cpu()
    change = (torch.nonzero(s[1:] != s[:-1]).flatten() + 1).tolist()
    bounds = [0, *change, s.numel()]
    return list(zip(bounds[:-1], bounds[1:]))


_MAX_SEGMENTS = 8  # q/k/v = 3, GDN qkv|z = 2; a genuine per-row scale stays W8A16 instead


def _segments_w8a8_ok(segments: list[tuple[int, int]]) -> bool:
    """cuBLASLt needs 16-row aligned fp8 operands; more parts than a fused projection has
    means a genuine per-row scale."""
    return 0 < len(segments) <= _MAX_SEGMENTS and all((e - s) % 16 == 0 for s, e in segments)


# ======================================================================================
# Decode (M==1) split-K GEMV: raw fp8 x bf16 reduction in fp32, per-row scale at reduce.
# ======================================================================================
@triton.jit
def _gemv_splitk_kernel(
    a_ptr, w_ptr, part_ptr, N, K, n_kb, kb_per,
    stride_ak, stride_wn, stride_wk, stride_pk, stride_pn,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Each (pid_n, pid_k) computes the partial sum over ``kb_per`` BLOCK_K chunks for a
    BLOCK_N slice of outputs. ``kb_per`` ceil-tiles K so K only needs to be a multiple of
    BLOCK_K is *not* required (loads are k-masked). Per-row scale is applied in the reduce."""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    kb_start = pid_k * kb_per
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for i in range(kb_per):
        kb = kb_start + i
        if kb < n_kb:
            offs_k = kb * BLOCK_K + tl.arange(0, BLOCK_K)
            k_mask = offs_k < K
            a = tl.load(a_ptr + offs_k * stride_ak, mask=k_mask, other=0.0).to(tl.float32)
            if e4m3_native_cx():
                w = tl.load(
                    w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk,
                    mask=n_mask[:, None] & k_mask[None, :], other=0.0,
                ).to(tl.float32)
            else:
                w = e4m3_u8_to_f32(tl.load(
                    w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk,
                    mask=n_mask[:, None] & k_mask[None, :], other=0,
                ))
            acc += tl.sum(w * a[None, :], axis=1)
    tl.store(part_ptr + pid_k * stride_pk + offs_n * stride_pn, acc, mask=n_mask)


@triton.jit
def _splitk_reduce_kernel(
    part_ptr, scale_ptr, out_ptr, N, SPLIT_K: tl.constexpr,
    stride_pk, stride_pn, BLOCK: tl.constexpr, OUT: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for k in tl.static_range(SPLIT_K):
        acc += tl.load(part_ptr + k * stride_pk + offs * stride_pn, mask=mask, other=0.0)
    scale = tl.load(scale_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + offs, (acc * scale).to(OUT), mask=mask)


@triton.jit
def _rows_act_load(row_ptr, offs_k, stride_ak, k_mask, ACT_SILU_MUL: tl.constexpr, act_off, OUT: tl.constexpr):
    """One activation row chunk as fp32; with ACT_SILU_MUL the row holds [gate | up] and
    silu(gate) * up is formed here (rounded to OUT, matching a separate act kernel)."""
    if ACT_SILU_MUL:
        g = tl.load(row_ptr + offs_k * stride_ak, mask=k_mask, other=0.0).to(tl.float32)
        u = tl.load(row_ptr + (act_off + offs_k) * stride_ak, mask=k_mask, other=0.0).to(tl.float32)
        return ((g / (1.0 + tl.exp(-g))) * u).to(OUT).to(tl.float32)
    return tl.load(row_ptr + offs_k * stride_ak, mask=k_mask, other=0.0).to(tl.float32)


@triton.jit
def _qsa_qkv_dest(offs_n, r, NQ2, HD, NK, M):
    """Column -> flat offset into the packed [q | gate | k | v] decode buffer of the QSA
    qkv projection (q/gate interleaved per head in the GEMM output: [head][q(HD) gate(HD)]).
    Regions are [M, NQ], [M, NQ], [M, NK], [M, NK] with NQ = NQ2 // 2."""
    NQ = NQ2 // 2
    in_qg = offs_n < NQ2
    head = offs_n // (2 * HD)
    w = offs_n - head * (2 * HD)
    is_gate = in_qg & (w >= HD)
    qg_off = r * NQ + head * HD + tl.where(is_gate, w - HD, w) + tl.where(is_gate, M * NQ, 0)
    in_k = (offs_n >= NQ2) & (offs_n < NQ2 + NK)
    k_off = 2 * M * NQ + r * NK + (offs_n - NQ2)
    v_off = 2 * M * NQ + M * NK + r * NK + (offs_n - NQ2 - NK)
    return tl.where(in_qg, qg_off, tl.where(in_k, k_off, v_off))


@triton.jit
def _rows_gemv_seg2(
    a_ptr, w_ptr, N, K, n_kb, stride_am, stride_ak, stride_wn, stride_wk,
    scale_ptr, out_ptr, s_ptr, stride_om, stride_sm, LORA, HCN, tile,
    ROWS: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    HAS_SCALE: tl.constexpr, OUT: tl.constexpr,
):
    """Second segment of a DIRECT rows-GEMV launch: one [BLOCK_N] tile of the small partner
    weight over the WHOLE K (no split), plain or two-way split epilogue (cols < LORA -> out,
    next HCN cols -> s)."""
    offs_n = tile * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    acc0 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    acc3 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for kb in range(n_kb):
        offs_k = kb * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K
        w = tl.load(
            w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk,
            mask=n_mask[:, None] & k_mask[None, :], other=0.0,
        ).to(tl.float32)
        a0 = tl.load(a_ptr + offs_k * stride_ak, mask=k_mask, other=0.0).to(tl.float32)
        acc0 += tl.sum(w * a0[None, :], axis=1)
        if ROWS > 1:
            a1 = tl.load(a_ptr + stride_am + offs_k * stride_ak, mask=k_mask, other=0.0).to(tl.float32)
            acc1 += tl.sum(w * a1[None, :], axis=1)
        if ROWS > 2:
            a2 = tl.load(a_ptr + 2 * stride_am + offs_k * stride_ak, mask=k_mask, other=0.0).to(tl.float32)
            acc2 += tl.sum(w * a2[None, :], axis=1)
        if ROWS > 3:
            a3 = tl.load(a_ptr + 3 * stride_am + offs_k * stride_ak, mask=k_mask, other=0.0).to(tl.float32)
            acc3 += tl.sum(w * a3[None, :], axis=1)
    if HAS_SCALE:
        sc = tl.load(scale_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
        acc0 = acc0 * sc
        acc1 = acc1 * sc
        acc2 = acc2 * sc
        acc3 = acc3 * sc
    for r in tl.static_range(ROWS):
        if r == 0:
            acc = acc0
        elif r == 1:
            acc = acc1
        elif r == 2:
            acc = acc2
        else:
            acc = acc3
        tl.store(out_ptr + r * stride_om + offs_n, acc.to(OUT), mask=n_mask & (offs_n < LORA))
        if HCN > 0:
            tl.store(s_ptr + r * stride_sm + (offs_n - LORA), acc.to(OUT),
                     mask=n_mask & (offs_n >= LORA) & (offs_n < LORA + HCN))


@triton.jit
def _gemv_rows_splitk_kernel(
    a_ptr, w_ptr, part_ptr, N, K, n_kb, kb_per,
    stride_am, stride_ak, stride_wn, stride_wk, stride_pk, stride_pm, stride_pn,
    ROWS: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    scale_ptr=None, DIRECT: tl.constexpr = False, HAS_SCALE: tl.constexpr = True,
    OUT: tl.constexpr = tl.bfloat16,
    # Serial split-K fixup (FIXUP): the last CTA to finish a tile reduces the SPLIT_K partials
    # itself -- same summation order as the reduce kernels, so the output is bit-identical --
    # and runs the epilogue (EPI 0 plain, 1 hc_down, 2 hc_gate_mix). cnt_ptr: int32 tile
    # counters, zero at launch and re-zeroed by the finishing CTA (graph-replay safe).
    cnt_ptr=None, out_ptr=None, s_ptr=None, rn_ptr=None,
    stride_om=0, stride_sm=0, stride_rn=0, LORA=0, HCN=0, HIDDEN=0,
    SPLIT_K: tl.constexpr = 1, FIXUP: tl.constexpr = False, EPI: tl.constexpr = 0,
    HC: tl.constexpr = 1,
    # ACT_SILU_MUL: the activation row is a raw [gate | up] pair (act_off = up offset) and
    # silu(gate) * up is formed on load, rounded to OUT like a separate act kernel would.
    ACT_SILU_MUL: tl.constexpr = False, act_off=0,
    # SEG2 (DIRECT launches only): tiles >= n_tiles_a run a SECOND, small weight w2 [N2, K]
    # (fp8 with scale2 or bf16) over the same activations, with a plain / two-way split
    # epilogue into out2 [M, LORA2] (+ s2 [M, HCN2]). Lets the tiny partner projection of a
    # big DIRECT GEMM (GDN in_proj_ba next to in_proj_qkvz, the QSA indexer next to
    # qkv_proj) ride in the same launch instead of costing its own latency-bound one.
    SEG2: tl.constexpr = False, w2_ptr=None, N2=0, n_tiles_a=0, stride_w2n=0, stride_w2k=0,
    scale2_ptr=None, HAS_SCALE2: tl.constexpr = False, out2_ptr=None, s2_ptr=None,
    stride_o2m=0, stride_s2m=0, LORA2=0, HCN2=0,
):
    """``_gemv_splitk_kernel`` for a handful of activation rows (decode batch / speculative
    verify, M <= ROWS): the fp8 weight tile is loaded and upcast ONCE per K chunk and applied
    to every row, so the weight bytes stream from HBM exactly once whatever M is. Rows past
    M are masked at load (zero partials). Same tile geometry as the M=1 kernel."""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    if SEG2:
        if pid_n >= n_tiles_a:
            _rows_gemv_seg2(
                a_ptr, w2_ptr, N2, K, n_kb, stride_am, stride_ak, stride_w2n, stride_w2k,
                scale2_ptr, out2_ptr, s2_ptr, stride_o2m, stride_s2m, LORA2, HCN2,
                pid_n - n_tiles_a, ROWS, BLOCK_N, BLOCK_K, HAS_SCALE2, OUT,
            )
            return
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    kb_start = pid_k * kb_per
    acc0 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    acc3 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for i in range(kb_per):
        kb = kb_start + i
        if kb < n_kb:
            offs_k = kb * BLOCK_K + tl.arange(0, BLOCK_K)
            k_mask = offs_k < K
            if e4m3_native_cx():
                w = tl.load(
                    w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk,
                    mask=n_mask[:, None] & k_mask[None, :], other=0.0,
                ).to(tl.float32)
            else:
                w = e4m3_u8_to_f32(tl.load(
                    w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk,
                    mask=n_mask[:, None] & k_mask[None, :], other=0,
                ))
            a0 = _rows_act_load(a_ptr, offs_k, stride_ak, k_mask, ACT_SILU_MUL, act_off, OUT)
            acc0 += tl.sum(w * a0[None, :], axis=1)
            if ROWS > 1:
                a1 = _rows_act_load(a_ptr + stride_am, offs_k, stride_ak, k_mask, ACT_SILU_MUL, act_off, OUT)
                acc1 += tl.sum(w * a1[None, :], axis=1)
            if ROWS > 2:
                a2 = _rows_act_load(a_ptr + 2 * stride_am, offs_k, stride_ak, k_mask, ACT_SILU_MUL, act_off, OUT)
                acc2 += tl.sum(w * a2[None, :], axis=1)
            if ROWS > 3:
                a3 = _rows_act_load(a_ptr + 3 * stride_am, offs_k, stride_ak, k_mask, ACT_SILU_MUL, act_off, OUT)
                acc3 += tl.sum(w * a3[None, :], axis=1)
    if DIRECT:
        # SPLIT_K == 1: no reduce pass -- scale here and write the output rows directly
        # (part_ptr is the [M, N] output, stride_pm its row stride).
        if HAS_SCALE:
            sc = tl.load(scale_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
            acc0 = acc0 * sc
            acc1 = acc1 * sc
            acc2 = acc2 * sc
            acc3 = acc3 * sc
        if EPI == 4:
            # QSA qkv projection: scatter into the packed [q | gate | k | v] buffer (out_ptr)
            tl.store(out_ptr + _qsa_qkv_dest(offs_n, 0, LORA, HIDDEN, HCN, ROWS), acc0.to(OUT), mask=n_mask)
            if ROWS > 1:
                tl.store(out_ptr + _qsa_qkv_dest(offs_n, 1, LORA, HIDDEN, HCN, ROWS), acc1.to(OUT), mask=n_mask)
            if ROWS > 2:
                tl.store(out_ptr + _qsa_qkv_dest(offs_n, 2, LORA, HIDDEN, HCN, ROWS), acc2.to(OUT), mask=n_mask)
            if ROWS > 3:
                tl.store(out_ptr + _qsa_qkv_dest(offs_n, 3, LORA, HIDDEN, HCN, ROWS), acc3.to(OUT), mask=n_mask)
        else:
            base = part_ptr + offs_n * stride_pn
            tl.store(base, acc0.to(OUT), mask=n_mask)
            if ROWS > 1:
                tl.store(base + stride_pm, acc1.to(OUT), mask=n_mask)
            if ROWS > 2:
                tl.store(base + 2 * stride_pm, acc2.to(OUT), mask=n_mask)
            if ROWS > 3:
                tl.store(base + 3 * stride_pm, acc3.to(OUT), mask=n_mask)
    else:
        base = part_ptr + pid_k * stride_pk + offs_n * stride_pn
        tl.store(base, acc0, mask=n_mask)
        if ROWS > 1:
            tl.store(base + stride_pm, acc1, mask=n_mask)
        if ROWS > 2:
            tl.store(base + 2 * stride_pm, acc2, mask=n_mask)
        if ROWS > 3:
            tl.store(base + 3 * stride_pm, acc3, mask=n_mask)
        if FIXUP:
            if EPI == 2:
                # gate-mix groups the HC tiles that share a d-block: tile h*(HIDDEN/BLOCK_N)+c
                cidx = pid_n % (HIDDEN // BLOCK_N)
                need = HC * SPLIT_K - 1
            else:
                cidx = pid_n
                need = SPLIT_K - 1
            # CTA-wide barrier then a gpu-scope release/acquire atomic by one thread: the
            # PTX model makes the release cumulative over the stores ordered by the barrier
            # (CUTLASS semaphore pattern), so the last CTA sees every partial.
            tl.debug_barrier()
            old = tl.atomic_add(cnt_ptr + cidx, 1, sem="acq_rel", scope="gpu")
            if old == need:
                tl.debug_barrier()
                tl.store(cnt_ptr + cidx, 0)
                if EPI == 2:
                    offs_d = cidx * BLOCK_N + tl.arange(0, BLOCK_N)
                    for r in tl.static_range(ROWS):
                        mixed = tl.zeros((BLOCK_N,), dtype=tl.float32)
                        for h in tl.static_range(HC):
                            cols = h * HIDDEN + offs_d
                            acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
                            for k in tl.static_range(SPLIT_K):
                                acc += tl.load(
                                    part_ptr + k * stride_pk + r * stride_pm + cols * stride_pn,
                                    cache_modifier=".cg",
                                )
                            if HAS_SCALE:
                                acc = acc * tl.load(scale_ptr + cols).to(tl.float32)
                            rn = tl.load(rn_ptr + r * stride_rn + cols).to(tl.float32)
                            mixed += tl.sigmoid(acc) * rn
                        mixed = mixed / HC
                        tl.store(out_ptr + r * stride_om + offs_d, mixed.to(OUT))
                else:
                    if HAS_SCALE:
                        sc = tl.load(scale_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
                    for r in tl.static_range(ROWS):
                        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
                        for k in tl.static_range(SPLIT_K):
                            acc += tl.load(
                                part_ptr + k * stride_pk + r * stride_pm + offs_n * stride_pn,
                                mask=n_mask, other=0.0, cache_modifier=".cg",
                            )
                        if HAS_SCALE:
                            acc = acc * sc
                        if EPI == 4:
                            tl.store(out_ptr + _qsa_qkv_dest(offs_n, r, LORA, HIDDEN, HCN, ROWS),
                                     acc.to(OUT), mask=n_mask)
                        elif EPI == 1 or EPI == 3:
                            # EPI 1 (hc_down): lora = silu(x / HC) on the first LORA columns;
                            # EPI 3 (split): the first LORA columns raw -> out, the next HCN -> s
                            if EPI == 1:
                                x = acc / HC
                                lora = x * tl.sigmoid(x)
                            else:
                                lora = acc
                            tl.store(out_ptr + r * stride_om + offs_n, lora.to(OUT),
                                     mask=n_mask & (offs_n < LORA))
                            tl.store(s_ptr + r * stride_sm + (offs_n - LORA), acc.to(OUT),
                                     mask=n_mask & (offs_n >= LORA) & (offs_n < LORA + HCN))
                        else:
                            tl.store(out_ptr + r * stride_om + offs_n, acc.to(OUT), mask=n_mask)


@triton.jit
def _splitk_rows_reduce_kernel(
    part_ptr, scale_ptr, out_ptr, N, SPLIT_K: tl.constexpr,
    stride_pk, stride_pm, stride_pn, stride_om, BLOCK: tl.constexpr, OUT: tl.constexpr,
    HAS_SCALE: tl.constexpr = True,
    s_ptr=None, stride_sm=0, LORA: tl.constexpr = 0, HCN: tl.constexpr = 0,
    HC: tl.constexpr = 1, HC_DOWN: tl.constexpr = False,
):
    row = tl.program_id(1)
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for k in tl.static_range(SPLIT_K):
        acc += tl.load(part_ptr + k * stride_pk + row * stride_pm + offs * stride_pn, mask=mask, other=0.0)
    if HAS_SCALE:
        scale = tl.load(scale_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        acc = acc * scale
    if HC_DOWN:
        # Hyper-connection down GEMM epilogue: columns [0, LORA) are the low-rank mix
        # (silu(x / HC), written contiguously to out), [LORA, LORA + HCN) the raw block-inject
        # logits (written to s); the pad columns are dropped.
        x = acc / HC
        lora = x * tl.sigmoid(x)
        tl.store(out_ptr + row * stride_om + offs, lora.to(OUT), mask=mask & (offs < LORA))
        tl.store(
            s_ptr + row * stride_sm + (offs - LORA), acc.to(OUT),
            mask=mask & (offs >= LORA) & (offs < LORA + HCN),
        )
    else:
        tl.store(out_ptr + row * stride_om + offs, acc.to(OUT), mask=mask)


@triton.jit
def _splitk_rows_gatemix_kernel(
    part_ptr, scale_ptr, rn_ptr, out_ptr, HIDDEN, SPLIT_K: tl.constexpr,
    stride_pk, stride_pm, stride_pn, stride_rn, stride_om,
    HC: tl.constexpr, BLOCK: tl.constexpr, OUT: tl.constexpr, HAS_SCALE: tl.constexpr,
):
    """Hyper-connection up GEMM epilogue: the GEMM's output is the gate over HC streams
    ([row, h * HIDDEN + d]); emit mixed[row, d] = mean_h sigmoid(gate[h, d]) * rn[h, d]
    straight from the split-K partials (replaces the reduce + hc_gate_mix launches)."""
    row = tl.program_id(1)
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < HIDDEN
    mixed = tl.zeros((BLOCK,), dtype=tl.float32)
    for h in tl.static_range(HC):
        cols = h * HIDDEN + offs
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        for k in tl.static_range(SPLIT_K):
            acc += tl.load(part_ptr + k * stride_pk + row * stride_pm + cols * stride_pn, mask=mask, other=0.0)
        if HAS_SCALE:
            acc = acc * tl.load(scale_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        rn = tl.load(rn_ptr + row * stride_rn + cols, mask=mask, other=0.0).to(tl.float32)
        mixed += tl.sigmoid(acc) * rn
    mixed = mixed / HC
    tl.store(out_ptr + row * stride_om + offs, mixed.to(OUT), mask=mask)


# Decode-sized batches (a speculative verify is 3-4 rows) take the rows GEMV: the M=1 tile
# geometry streams fp8 at ~1.4-1.6 TB/s on every shape, where the tile GEMMs (cuBLASLt
# row-wise fp8, the prefill W8A16 kernel) are latency-bound and slower than bf16 cuBLAS on
# the narrow/deep shapes (HC mixes, out_proj). Measured on the RTX 5090 (HBM regime, M=3).
_ROWS_GEMV_MAX_M = int(os.getenv("FREETOKEN_FP8_ROWS_GEMV_MAX_M", "4"))

# Split-K shapes reduce inside the GEMV launch (last CTA per tile, see FIXUP in the kernel)
# instead of a second reduce kernel: ~475 graph nodes fewer per verify step on Qwen3.8.
# FREETOKEN_ROWS_GEMV_FIXUP=0 restores the two-launch path (bit-identical output).
_ROWS_FIXUP = os.getenv("FREETOKEN_ROWS_GEMV_FIXUP", "1") != "0"
_FIXUP_COUNTERS = 8192
_FIXUP_RING = 4
_fixup_counters: dict = {}


def _fixup_counter_buf(device: torch.device, n: int) -> torch.Tensor:
    """Zeroed int32 tile counters for the fixup. Every launch leaves them at zero again, so
    launches on one stream share a buffer; a small ring keeps back-to-back launches on
    different streams apart."""
    assert n <= _FIXUP_COUNTERS, n
    key = (device.type, device.index)
    st = _fixup_counters.get(key)
    if st is None:
        st = [torch.zeros(_FIXUP_RING, _FIXUP_COUNTERS, dtype=torch.int32, device=device), 0]
        _fixup_counters[key] = st
    st[1] = (st[1] + 1) % _FIXUP_RING
    return st[0][st[1]]


# Offline tuning seam (see docs/decode-megakernel-evaluation.md). Empty in production: the
# heuristics below decide, and the only hot-path cost is one truthiness test on an empty dict
# plus a module-global read for num_warps. A shape sweep sets BLOCK_N / BLOCK_K / SPLIT_K here
# and _ROWS_WARPS below, so candidates run through this exact wrapper -- including every fused
# epilogue and the split-K fixup -- instead of hand-rebuilt kernel launches, which is how an
# earlier sweep silently measured nothing at all.
_ROWS_TUNE: dict = {}
_ROWS_WARPS = 1


def _gemv_rows(a: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor | None,
               out_dtype: torch.dtype, epilogue=None, act=None):
    """M<=4 split-K rows GEMV (W8A16, exact activations). ``a`` [M, K] bf16; ``weight``
    [N, K] fp8 (or bf16 with ``weight_scale=None``: the same memory-bound kernel serves the
    small-N bf16 projections cuBLAS handles poorly at M<=4); ``weight_scale`` [N] fp32
    -> [M, N]."""
    M = a.shape[0]
    N, K = weight.shape
    if act is None:
        assert a.shape[1] == K, (a.shape, weight.shape)
        act_kw = {}
    else:
        # ("silu_mul", up_off): a is the raw [M, >= up_off + K] gate|up row; silu(g)*u on load
        kind_a, up_off = act
        assert kind_a == "silu_mul" and a.shape[1] >= up_off + K, (a.shape, up_off, K)
        act_kw = dict(ACT_SILU_MUL=True, act_off=int(up_off))
    assert 1 <= M <= 4, M
    # 256-wide K chunks give the deep fp8 shapes a few % more bandwidth (in_proj_qkvz
    # 28.4 -> 27.5 us, qkv_proj 23.4 -> 22.5, RTX 5090 sweep); the small/bf16 ones prefer 128.
    BLOCK_K = 256 if (weight_scale is not None and K >= 2048 and N >= 1024) else 128
    BLOCK_N = 16
    if _ROWS_TUNE:  # offline sweep only
        BLOCK_K = _ROWS_TUNE.get("BLOCK_K", BLOCK_K)
        BLOCK_N = _ROWS_TUNE.get("BLOCK_N", BLOCK_N)
    n_kb = triton.cdiv(K, BLOCK_K)
    n_tiles = triton.cdiv(N, BLOCK_N)
    split_k = max(1, min(_ROWS_TUNE.get("SPLIT_K", 1536 // n_tiles) if _ROWS_TUNE
                         else 1536 // n_tiles, n_kb))
    split_k = 1 << (split_k.bit_length() - 1)  # pow2 -> stable reduction order
    kb_per = triton.cdiv(n_kb, split_k)
    compute = out_dtype if out_dtype in _TL_DTYPE else torch.bfloat16
    has_scale = weight_scale is not None
    scale_arg = a if weight_scale is None else weight_scale
    kind = None if epilogue is None else epilogue[0]
    if kind == "qsa_qkv":
        # ("qsa_qkv", num_q, num_kv, head_dim): one packed buffer holding q, gate, k, v as four
        # contiguous [M, *] tensors (q/gate de-interleaved per head), from the DIRECT store
        # or the fixup reduce -- no .contiguous() copies downstream.
        _, num_q, num_kv, hd = epilogue
        nq, nk = num_q * hd, num_kv * hd
        assert N == 2 * nq + 2 * nk, (N, num_q, num_kv, hd)
        buf = torch.empty((2 * M * nq + 2 * M * nk,), dtype=compute, device=a.device)
        if split_k == 1 or not _ROWS_FIXUP:
            if split_k != 1:
                # two-launch fallback: plain GEMM then slice (only when the fixup is disabled)
                out = _gemv_rows(a, weight, weight_scale, out_dtype)
                qg = out[:, : 2 * nq].view(M, num_q, 2 * hd)
                return (qg[..., :hd].contiguous(), qg[..., hd:].contiguous(),
                        out[:, 2 * nq: 2 * nq + nk].contiguous(), out[:, 2 * nq + nk:].contiguous())
            _gemv_rows_splitk_kernel[(n_tiles, 1)](
                a, weight, buf, N, K, n_kb, kb_per,
                a.stride(0), a.stride(1), weight.stride(0), weight.stride(1),
                0, 0, 1,
                ROWS=M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                scale_ptr=buf if weight_scale is None else weight_scale, DIRECT=True,
                HAS_SCALE=weight_scale is not None, OUT=_TL_DTYPE[compute], num_warps=_ROWS_WARPS,
                out_ptr=buf, LORA=2 * nq, HIDDEN=hd, HCN=nk, EPI=4, **act_kw,
            )
        else:
            part = torch.empty((split_k, M, N), dtype=torch.float32, device=a.device)
            _gemv_rows_splitk_kernel[(n_tiles, split_k)](
                a, weight, part, N, K, n_kb, kb_per,
                a.stride(0), a.stride(1), weight.stride(0), weight.stride(1),
                part.stride(0), part.stride(1), part.stride(2),
                ROWS=M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, num_warps=_ROWS_WARPS,
                scale_ptr=buf if weight_scale is None else weight_scale,
                HAS_SCALE=weight_scale is not None, OUT=_TL_DTYPE[compute],
                SPLIT_K=split_k, FIXUP=True, cnt_ptr=_fixup_counter_buf(a.device, n_tiles),
                out_ptr=buf, LORA=2 * nq, HIDDEN=hd, HCN=nk, EPI=4, **act_kw,
            )
        q = buf[: M * nq].view(M, nq)
        gate = buf[M * nq: 2 * M * nq].view(M, nq)
        k = buf[2 * M * nq: 2 * M * nq + M * nk].view(M, nk)
        v = buf[2 * M * nq + M * nk:].view(M, nk)
        return q, gate, k, v
    if split_k > 1 and _ROWS_FIXUP and (kind != "hc_gate_mix" or epilogue[3] % BLOCK_N == 0):
        part = torch.empty((split_k, M, N), dtype=torch.float32, device=a.device)
        common = dict(
            ROWS=M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, num_warps=_ROWS_WARPS,
            scale_ptr=scale_arg, HAS_SCALE=has_scale, OUT=_TL_DTYPE[compute],
            SPLIT_K=split_k, FIXUP=True, **act_kw,
        )
        args = (a, weight, part, N, K, n_kb, kb_per,
                a.stride(0), a.stride(1), weight.stride(0), weight.stride(1),
                part.stride(0), part.stride(1), part.stride(2))
        if kind is None:
            out = torch.empty((M, N), dtype=compute, device=a.device)
            _gemv_rows_splitk_kernel[(n_tiles, split_k)](
                *args, cnt_ptr=_fixup_counter_buf(a.device, n_tiles), out_ptr=out,
                stride_om=out.stride(0), EPI=0, **common,
            )
            return out
        if kind == "hc_down":
            _, lowrank, hcn, hc = epilogue
            lora = torch.empty((M, lowrank), dtype=compute, device=a.device)
            sgate = torch.empty((M, max(hcn, 1)), dtype=compute, device=a.device)
            _gemv_rows_splitk_kernel[(n_tiles, split_k)](
                *args, cnt_ptr=_fixup_counter_buf(a.device, n_tiles), out_ptr=lora, s_ptr=sgate,
                stride_om=lora.stride(0), stride_sm=sgate.stride(0), LORA=lowrank, HCN=hcn,
                EPI=1, HC=hc, **common,
            )
            return lora, (sgate if hcn > 0 else None)
        if kind == "split":
            # two contiguous outputs from one GEMM: [M, n_first] and [M, n_second]
            _, n_first, n_second = epilogue
            assert n_first + n_second <= N
            first = torch.empty((M, n_first), dtype=compute, device=a.device)
            second = torch.empty((M, n_second), dtype=compute, device=a.device)
            _gemv_rows_splitk_kernel[(n_tiles, split_k)](
                *args, cnt_ptr=_fixup_counter_buf(a.device, n_tiles), out_ptr=first, s_ptr=second,
                stride_om=first.stride(0), stride_sm=second.stride(0), LORA=n_first, HCN=n_second,
                EPI=3, **common,
            )
            return first, second
        if kind == "hc_gate_mix":
            _, rn, hc, hidden = epilogue
            assert N == hc * hidden and rn.shape == (M, N)
            mixed = torch.empty((M, hidden), dtype=compute, device=a.device)
            _gemv_rows_splitk_kernel[(n_tiles, split_k)](
                *args, cnt_ptr=_fixup_counter_buf(a.device, hidden // BLOCK_N), out_ptr=mixed,
                rn_ptr=rn, stride_om=mixed.stride(0), stride_rn=rn.stride(0), HIDDEN=hidden,
                EPI=2, HC=hc, **common,
            )
            return mixed
        raise ValueError(f"unknown rows-GEMV epilogue {kind!r}")
    if epilogue is not None:
        # Fused hyper-connection epilogues (see GatedResidual._mix_kernel): the split-K
        # partials feed the epilogue kernel directly instead of a plain reduce.
        part = torch.empty((split_k, M, N), dtype=torch.float32, device=a.device)
        _gemv_rows_splitk_kernel[(n_tiles, split_k)](
            a, weight, part, N, K, n_kb, kb_per,
            a.stride(0), a.stride(1), weight.stride(0), weight.stride(1),
            part.stride(0), part.stride(1), part.stride(2),
            ROWS=M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, num_warps=_ROWS_WARPS, **act_kw,
        )
        kind = epilogue[0]
        if kind == "split":
            _, n_first, n_second = epilogue
            out = torch.empty((M, N), dtype=compute, device=a.device)
            _splitk_rows_reduce_kernel[(triton.cdiv(N, 256), M)](
                part, part if weight_scale is None else weight_scale, out, N, split_k,
                part.stride(0), part.stride(1), part.stride(2), out.stride(0),
                BLOCK=256, OUT=_TL_DTYPE[compute], HAS_SCALE=weight_scale is not None, num_warps=2,
            )
            return out[:, :n_first].contiguous(), out[:, n_first:n_first + n_second].contiguous()
        if kind == "hc_down":
            _, lowrank, hcn, hc = epilogue
            lora = torch.empty((M, lowrank), dtype=compute, device=a.device)
            sgate = torch.empty((M, max(hcn, 1)), dtype=compute, device=a.device)
            _splitk_rows_reduce_kernel[(triton.cdiv(N, 256), M)](
                part, part if weight_scale is None else weight_scale, lora, N, split_k,
                part.stride(0), part.stride(1), part.stride(2), lora.stride(0),
                BLOCK=256, OUT=_TL_DTYPE[compute], HAS_SCALE=weight_scale is not None,
                s_ptr=sgate, stride_sm=sgate.stride(0), LORA=lowrank, HCN=hcn, HC=hc,
                HC_DOWN=True, num_warps=2,
            )
            return lora, (sgate if hcn > 0 else None)
        if kind == "hc_gate_mix":
            _, rn, hc, hidden = epilogue
            assert N == hc * hidden and rn.shape == (M, N)
            mixed = torch.empty((M, hidden), dtype=compute, device=a.device)
            _splitk_rows_gatemix_kernel[(triton.cdiv(hidden, 256), M)](
                part, part if weight_scale is None else weight_scale, rn, mixed, hidden, split_k,
                part.stride(0), part.stride(1), part.stride(2), rn.stride(0), mixed.stride(0),
                HC=hc, BLOCK=256, OUT=_TL_DTYPE[compute], HAS_SCALE=weight_scale is not None,
                num_warps=2,
            )
            return mixed
        raise ValueError(f"unknown rows-GEMV epilogue {kind!r}")
    out = torch.empty((M, N), dtype=compute, device=a.device)
    if split_k == 1:
        # wide-N shapes (in_qkvz, qkv_proj, lm_head): one pass, no reduce kernel
        _gemv_rows_splitk_kernel[(n_tiles, 1)](
            a, weight, out, N, K, n_kb, kb_per,
            a.stride(0), a.stride(1), weight.stride(0), weight.stride(1),
            0, out.stride(0), out.stride(1),
            ROWS=M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            scale_ptr=out if weight_scale is None else weight_scale, DIRECT=True,
            HAS_SCALE=weight_scale is not None, OUT=_TL_DTYPE[compute], num_warps=_ROWS_WARPS, **act_kw,
        )
        return out
    part = torch.empty((split_k, M, N), dtype=torch.float32, device=a.device)
    _gemv_rows_splitk_kernel[(n_tiles, split_k)](
        a, weight, part, N, K, n_kb, kb_per,
        a.stride(0), a.stride(1), weight.stride(0), weight.stride(1),
        part.stride(0), part.stride(1), part.stride(2),
        ROWS=M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, num_warps=_ROWS_WARPS, **act_kw,
    )
    _splitk_rows_reduce_kernel[(triton.cdiv(N, 256), M)](
        part, part if weight_scale is None else weight_scale, out, N, split_k,
        part.stride(0), part.stride(1), part.stride(2), out.stride(0),
        BLOCK=256, OUT=_TL_DTYPE[compute], HAS_SCALE=weight_scale is not None, num_warps=2,
    )
    return out


# bf16 linears at decode batch sizes (M <= 4): cuBLAS's small-M kernels are latency-bound on
# the narrow/deep shapes that remain bf16 in Qwen3.8-Flash-Next under --dense-quant fp8
# (GDN in_proj_ba [96, 2560] = 15.6 us, router gate [512, 2560], indexer, the MTP head's
# fc/qkv/o); the rows GEMV streams them at HBM speed (~2 us for in_proj_ba). Only deep-K
# shapes qualify (K >= 1024): on shallow-K / wide-N ones cuBLAS is already at bandwidth.
_BF16_ROWS_GEMV = os.getenv("FREETOKEN_BF16_ROWS_GEMV", "1") != "0"
_BF16_ROWS_MIN_K = 1024


def bf16_rows_gemv_applicable(x: torch.Tensor, weight: torch.Tensor, bias) -> bool:
    return (
        _BF16_ROWS_GEMV
        and bias is None
        and x.dim() == 2
        and 1 <= x.shape[0] <= _ROWS_GEMV_MAX_M
        and x.is_cuda
        and x.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and weight.shape[1] >= _BF16_ROWS_MIN_K
        and weight.is_contiguous()
    )


def bf16_rows_gemv(x: torch.Tensor, weight: torch.Tensor, epilogue=None):
    """``x @ weight.T`` for M <= 4 bf16 rows (fp32 accumulation, bf16 out)."""
    return _gemv_rows(x.contiguous(), weight, None, torch.bfloat16, epilogue=epilogue)


def rows_gemv_silu_mul(layer, gate_up: torch.Tensor):
    """``layer(silu(g) * u)`` for a decode-sized ``gate_up`` [M, 2I] (I = layer's K), the
    activation formed inside the rows GEMV (no act kernel / intermediate). None when the
    rows path does not apply (M > 4, CPU, bias, shallow-K bf16...)."""
    w = layer.weight
    if not (gate_up.is_cuda and gate_up.dim() == 2 and 1 <= gate_up.shape[0] <= _ROWS_GEMV_MAX_M
            and gate_up.dtype == torch.bfloat16 and gate_up.shape[1] == 2 * w.shape[1]):
        return None
    inter = w.shape[1]
    if w.dtype == FP8:
        if getattr(layer, "input_scale", None) is not None:
            return None
        return _gemv_rows(gate_up.contiguous(), e4m3_kernel_view(w), layer.weight_scale, gate_up.dtype,
                          act=("silu_mul", inter))
    if (_BF16_ROWS_GEMV and getattr(layer, "bias", None) is None and w.dtype == torch.bfloat16
            and inter >= _BF16_ROWS_MIN_K and w.is_contiguous()):
        return _gemv_rows(gate_up.contiguous(), w, None, torch.bfloat16, act=("silu_mul", inter))
    return None


def rows_gemv_direct_pair(layer_a, x: torch.Tensor, epilogue_a, layer_b, split_b):
    """One launch for a big DIRECT (split_k == 1) rows GEMV ``layer_a`` (plain or
    ``qsa_qkv`` epilogue) plus a small partner ``layer_b`` over the same rows with a
    ``("split", n1, n2)`` epilogue (n2 may be 0). Returns ``(out_a, (b1, b2))`` -- or None
    when the pair does not qualify (A not DIRECT, M > 4, biases, input scales...), in which
    case the caller runs the two GEMVs separately."""
    if not (x.is_cuda and x.dim() == 2 and 1 <= x.shape[0] <= _ROWS_GEMV_MAX_M and x.dtype == torch.bfloat16):
        return None
    wa, wb = layer_a.weight, layer_b.weight
    if getattr(layer_a, "input_scale", None) is not None or getattr(layer_b, "input_scale", None) is not None:
        return None
    if getattr(layer_a, "bias", None) is not None or getattr(layer_b, "bias", None) is not None:
        return None
    if not (wb.is_contiguous() and wa.shape[1] == wb.shape[1] == x.shape[1]):
        return None
    if wa.dtype != FP8 and not (_BF16_ROWS_GEMV and wa.dtype == torch.bfloat16):
        return None
    if wb.dtype not in (FP8, torch.bfloat16):
        return None
    M, K = x.shape
    N, N2 = wa.shape[0], wb.shape[0]
    BLOCK_N = 16
    BLOCK_K = 256 if (wa.dtype == FP8 and K >= 2048 and N >= 1024) else 128
    n_kb = triton.cdiv(K, BLOCK_K)
    n_tiles = triton.cdiv(N, BLOCK_N)
    split_k = max(1, min(1536 // n_tiles, n_kb))
    if split_k != 1:
        return None  # A is not a DIRECT shape
    _, n1, n2 = split_b
    if n1 + n2 > N2:
        return None
    a = x.contiguous()
    weight = e4m3_kernel_view(wa) if wa.dtype == FP8 else wa
    scale_a = layer_a.weight_scale if wa.dtype == FP8 else None
    w2 = e4m3_kernel_view(wb) if wb.dtype == FP8 else wb
    scale_b = layer_b.weight_scale if wb.dtype == FP8 else None
    compute = torch.bfloat16
    n_tiles_b = triton.cdiv(N2, BLOCK_N)
    b1 = torch.empty((M, n1), dtype=compute, device=a.device)
    b2 = torch.empty((M, max(n2, 1)), dtype=compute, device=a.device)
    seg2 = dict(
        SEG2=True, w2_ptr=w2, N2=N2, n_tiles_a=n_tiles, stride_w2n=w2.stride(0), stride_w2k=w2.stride(1),
        scale2_ptr=b1 if scale_b is None else scale_b, HAS_SCALE2=scale_b is not None,
        out2_ptr=b1, s2_ptr=b2, stride_o2m=b1.stride(0), stride_s2m=b2.stride(0), LORA2=n1, HCN2=n2,
    )
    kind = None if epilogue_a is None else epilogue_a[0]
    grid = (n_tiles + n_tiles_b, 1)
    if kind is None:
        out = torch.empty((M, N), dtype=compute, device=a.device)
        _gemv_rows_splitk_kernel[grid](
            a, weight, out, N, K, n_kb, n_kb,
            a.stride(0), a.stride(1), weight.stride(0), weight.stride(1),
            0, out.stride(0), out.stride(1),
            ROWS=M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            scale_ptr=out if scale_a is None else scale_a, DIRECT=True,
            HAS_SCALE=scale_a is not None, OUT=_TL_DTYPE[compute], num_warps=_ROWS_WARPS, **seg2,
        )
        return out, (b1, b2 if n2 else None)
    if kind == "qsa_qkv":
        _, num_q, num_kv, hd = epilogue_a
        nq, nk = num_q * hd, num_kv * hd
        assert N == 2 * nq + 2 * nk, (N, num_q, num_kv, hd)
        buf = torch.empty((2 * M * nq + 2 * M * nk,), dtype=compute, device=a.device)
        _gemv_rows_splitk_kernel[grid](
            a, weight, buf, N, K, n_kb, n_kb,
            a.stride(0), a.stride(1), weight.stride(0), weight.stride(1),
            0, 0, 1,
            ROWS=M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            scale_ptr=buf if scale_a is None else scale_a, DIRECT=True,
            HAS_SCALE=scale_a is not None, OUT=_TL_DTYPE[compute], num_warps=_ROWS_WARPS,
            out_ptr=buf, LORA=2 * nq, HIDDEN=hd, HCN=nk, EPI=4, **seg2,
        )
        q = buf[: M * nq].view(M, nq)
        gate = buf[M * nq: 2 * M * nq].view(M, nq)
        k = buf[2 * M * nq: 2 * M * nq + M * nk].view(M, nk)
        v = buf[2 * M * nq + M * nk:].view(M, nk)
        return (q, gate, k, v), (b1, b2 if n2 else None)
    return None


def rows_gemv_with_epilogue(layer, x: torch.Tensor, epilogue):
    """Run a linear layer (bf16 ``LinearReplicated`` or ``Fp8PerTensorLinear``) through the
    rows GEMV with a fused hyper-connection epilogue, or return None when the rows path does
    not apply (M > 4, CPU, bias, shallow K...) so the caller falls back to the plain ops."""
    w = layer.weight
    if not (x.is_cuda and x.dim() == 2 and 1 <= x.shape[0] <= _ROWS_GEMV_MAX_M and x.dtype == torch.bfloat16):
        return None
    if w.dtype == FP8:
        if getattr(layer, "input_scale", None) is not None:
            return None
        return _gemv_rows(x.contiguous(), e4m3_kernel_view(w), layer.weight_scale, x.dtype, epilogue=epilogue)
    if bf16_rows_gemv_applicable(x, w, getattr(layer, "bias", None)):
        return bf16_rows_gemv(x, w, epilogue=epilogue)
    return None


def _gemv(a: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor,
          out_dtype: torch.dtype) -> torch.Tensor:
    """M==1 split-K GEMV. ``a`` [K] bf16; ``weight`` [N, K] fp8; ``weight_scale`` [N] fp32."""
    N, K = weight.shape
    BLOCK_K = 128
    n_kb = triton.cdiv(K, BLOCK_K)
    BLOCK_N = 16
    n_tiles = triton.cdiv(N, BLOCK_N)
    split_k = max(1, min(1536 // n_tiles, n_kb))
    split_k = 1 << (split_k.bit_length() - 1)  # pow2 -> stable reduction order
    kb_per = triton.cdiv(n_kb, split_k)
    part = torch.empty((split_k, N), dtype=torch.float32, device=a.device)
    _gemv_splitk_kernel[(n_tiles, split_k)](
        a, weight, part, N, K, n_kb, kb_per,
        a.stride(0), weight.stride(0), weight.stride(1), part.stride(0), part.stride(1),
        BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, num_warps=_ROWS_WARPS,
    )
    out = torch.empty(N, dtype=out_dtype, device=a.device)
    _splitk_reduce_kernel[(triton.cdiv(N, 256),)](
        part, weight_scale, out, N, split_k, part.stride(0), part.stride(1),
        BLOCK=256, OUT=_TL_DTYPE[out_dtype if out_dtype in _TL_DTYPE else torch.bfloat16],
        num_warps=2,
    )
    return out


# ======================================================================================
# Prefill (M>1) W8A16 GEMM: fp8 weight read from HBM, upcast to bf16 in-register for the
# tensor-core dot (fp8 e4m3 -> bf16 is lossless), per-row scale applied after accumulation.
# ======================================================================================
@triton.jit
def _gemm_kernel(
    a_ptr, w_ptr, scale_ptr, c_ptr, M, N, K,
    stride_am, stride_ak, stride_wn, stride_wk, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    compute_type: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    m_mask = offs_m < M
    n_mask = offs_n < N
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(tl.cdiv(K, BLOCK_K)):
        k_rem = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=m_mask[:, None] & (offs_k[None, :] < k_rem), other=0.0)
        w_mask = n_mask[:, None] & (offs_k[None, :] < k_rem)
        if e4m3_native_cx():
            w = tl.load(w_ptrs, mask=w_mask, other=0.0).to(a.dtype)
        else:
            w = e4m3_u8_to_f32(tl.load(w_ptrs, mask=w_mask, other=0)).to(a.dtype)
        acc += tl.dot(a, tl.trans(w), out_dtype=tl.float32)
        a_ptrs += BLOCK_K * stride_ak
        w_ptrs += BLOCK_K * stride_wk
    scale = tl.load(scale_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    acc = acc * scale[None, :]
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(compute_type), mask=m_mask[:, None] & n_mask[None, :])


def _gemm(a: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor,
          out_dtype: torch.dtype) -> torch.Tensor:
    """M>1 W8A16 GEMM. ``a`` [M, K] bf16; ``weight`` [N, K] fp8; ``weight_scale`` [N] fp32."""
    M, K = a.shape
    N = weight.shape[0]
    compute = out_dtype if out_dtype in _TL_DTYPE else torch.bfloat16
    out = torch.empty((M, N), dtype=compute, device=a.device)
    BLOCK_M = 64 if M >= 64 else 32
    BLOCK_N, BLOCK_K = 128, 64
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _gemm_kernel[grid](
        a, weight, weight_scale, out, M, N, K,
        a.stride(0), a.stride(1), weight.stride(0), weight.stride(1), out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, compute_type=_TL_DTYPE[compute],
        num_warps=8 if M >= 64 else 4, num_stages=3,
    )
    return out


# ======================================================================================
# Batched decode (M > 1) W8A8: quantize the activation with the checkpoint's static
# per-tensor ``input_scale`` and hand both fp8 operands to cuBLASLt via torch._scaled_mm.
#
# ``_gemm`` above is a *prefill* kernel: grid is (cdiv(M, BLOCK_M), cdiv(N, 128)) with no
# split-K, so at decode batch sizes the M axis contributes a single block row and the whole
# GEMM runs on cdiv(N, 128) CTAs -- 8 of them for N=1024, on a 188-SM part. Its runtime then
# tracks K and ignores both N and M: latency-bound, not bandwidth-bound. Measured on
# Qwen3.5-27B's projections (RTX PRO 6000, 1461 GB/s copy roof): 594 GB/s flat across
# M=2..16, against 1334 GB/s for the M=1 GEMV over the same weights. Routing M>=2 to
# cuBLASLt restores ~1300 GB/s -- 10.15 ms -> 5.56 ms of per-decode-step GEMM.
#
# Requires sm_89+ (torch._scaled_mm's floor; Ampere has no FP8 tensor cores) *and* a
# checkpoint carrying ``input_scale``. Without either, ``_gemm`` still runs: glm_moe_dsa
# quantizes to fp8 at load with genuine per-row scales and has no activation scale at all.
# ======================================================================================
_E4M3_MAX = 448.0  # largest finite e4m3 magnitude


@triton.jit
def _static_quant_kernel(x_ptr, out_ptr, scale_ptr, n_elements, BLOCK: tl.constexpr):
    """bf16 activation -> fp8-e4m3 under one broadcast per-tensor scale."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    inv = 1.0 / tl.load(scale_ptr).to(tl.float32)
    v = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32) * inv
    v = tl.minimum(tl.maximum(v, -448.0), 448.0)
    tl.store(out_ptr + offs, v.to(tl.float8e4nv), mask=mask)


def _static_quant(a: torch.Tensor, input_scale: torch.Tensor) -> torch.Tensor:
    """One launch, no host sync: the scale is read from device memory inside the kernel, so
    this stays CUDA-graph safe (fixed shapes, no dependence on tensor *values* on the host)."""
    n = a.numel()
    out = torch.empty_like(a, dtype=FP8)
    BLOCK = 1024
    _static_quant_kernel[(triton.cdiv(n, BLOCK),)](
        a, out, input_scale, n, BLOCK=BLOCK, num_warps=4,
    )
    return out


@triton.jit
def _dynamic_quant_kernel(x_ptr, out_ptr, scale_ptr, K, stride_xm, BLOCK_K: tl.constexpr):
    """bf16 activation row -> fp8-e4m3 under its own per-token scale (amax / 448), the
    compressed-tensors ``input_activations: dynamic, strategy: token`` recipe."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_K)
    mask = offs < K
    x = tl.load(x_ptr + row * stride_xm + offs, mask=mask, other=0.0).to(tl.float32)
    amax = tl.max(tl.abs(x), axis=0)
    scale = tl.maximum(amax / 448.0, 1e-12)
    v = x / scale
    v = tl.minimum(tl.maximum(v, -448.0), 448.0)
    tl.store(out_ptr + row * K + offs, v.to(tl.float8e4nv), mask=mask)
    tl.store(scale_ptr + row, scale)


def _dynamic_quant(a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token fp8 quantization of ``a`` [M, K] -> (fp8 [M, K], fp32 scales [M, 1]). One
    launch, device-side scales: CUDA-graph safe."""
    M, K = a.shape
    out = torch.empty((M, K), dtype=FP8, device=a.device)
    scales = torch.empty((M, 1), dtype=torch.float32, device=a.device)
    _dynamic_quant_kernel[(M,)](
        a, out, scales, K, a.stride(0), BLOCK_K=triton.next_power_of_2(K), num_warps=8,
    )
    return out, scales


_DYNAMIC_W8A8 = os.environ.get("FREETOKEN_FP8_DYNAMIC_W8A8", "1") != "0"
_dynamic_ok: bool | None = None


def dynamic_w8a8_ok() -> bool:
    """Whether checkpoints WITHOUT a static ``input_scale`` run their M>1 GEMMs as dynamic
    per-token W8A8 through cuBLASLt (row-wise ``_scaled_mm``) instead of the latency-bound
    W8A16 tile GEMM. Decided once per process at load (never under graph capture)."""
    global _dynamic_ok
    if _dynamic_ok is None:
        _dynamic_ok = _DYNAMIC_W8A8 and e4m3_native() and rowwise_scaled_mm_ok()
    return _dynamic_ok


def _dynamic_scaled_mm(
    a: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor, out_dtype: torch.dtype
) -> torch.Tensor:
    """``a @ (weight_fp8 * weight_scale)^T`` with per-token dynamic activation scales."""
    qa, sa = _dynamic_quant(a)
    return torch._scaled_mm(
        qa, weight.t(), scale_a=sa, scale_b=weight_scale.reshape(1, -1).float(),
        out_dtype=out_dtype,
    )


def _scaled_mm(
    a: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor,
    input_scale: torch.Tensor, uniform_scale: bool, out_dtype: torch.dtype,
    scale_segments: list[tuple[int, int]] | None = None,
) -> torch.Tensor:
    """``a @ (weight_fp8 * weight_scale)^T`` as a W8A8 cuBLASLt GEMM.

    ``weight`` stays in the checkpoint-native row-major ``[N, K]``: ``.t()`` is exactly the
    column-major ``[K, N]`` operand cuBLASLt wants, so the transpose is a stride change and
    never a copy (unlike Marlin, or our own NVFP4 path's K-major repack).

    A genuine per-tensor weight (one scalar repeated down ``weight_scale``) takes the
    tensor-wise path. A fused projection, whose ``weight_scale`` is piecewise-constant
    because each part carries its own scalar, takes the row-wise path -- that keeps every
    part's scale exact, where vLLM/SGLang instead requantize the parts onto a shared maximum
    and eat the precision loss. Row-wise costs ~4% here (5.56 ms vs 5.39 ms per step).

    Where row-wise is unsafe (:func:`rowwise_scaled_mm_ok`) ``scale_segments`` is passed and
    each part runs its own tensor-wise GEMM over ``weight[s:e]`` (still stride-only), outputs
    concatenated: the same W8A8 scheme, not bit-identical (accumulation order differs)."""
    qa = _static_quant(a, input_scale)
    wt = weight.t()  # [N, K] row-major -> [K, N] column-major, stride-only
    sa = input_scale.reshape(())
    if uniform_scale:
        return torch._scaled_mm(
            qa, wt, scale_a=sa, scale_b=weight_scale[0].reshape(()), out_dtype=out_dtype,
        )
    if scale_segments is not None:
        return torch.cat([
            torch._scaled_mm(
                qa, weight[s:e].t(), scale_a=sa, scale_b=weight_scale[s].reshape(()),
                out_dtype=out_dtype,
            )
            for s, e in scale_segments
        ], dim=1)
    return torch._scaled_mm(
        qa, wt,
        scale_a=input_scale.reshape(1, 1).expand(a.shape[0], 1).contiguous(),
        scale_b=weight_scale.reshape(1, -1),
        out_dtype=out_dtype,
    )


def fp8_pertensor_linear(
    x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    input_scale: torch.Tensor | None = None,
    uniform_scale: bool = False,
    scale_segments: list[tuple[int, int]] | None = None,
) -> torch.Tensor:
    """``y = x @ (weight_fp8 * weight_scale)^T``. ``weight`` [N, K] fp8-e4m3, ``weight_scale``
    [N] fp32 (per output row). ``scale_segments``: the fused parts' row ranges, precomputed at
    load by the layer; derived here (with a sync) when omitted and needed.

    Whether the activation is quantized is a property of the *deployment*, never of the batch:
    with ``input_scale`` on sm_89+ every M runs W8A8, otherwise every M runs W8A16 (split-K
    GEMV at M=1, GEMM above it). Picking per-M would make a request's numerics depend on how
    many unrelated requests happened to be in flight, so a reply would not reproduce at bs=1 --
    and W8A16 at M=1 is worth only ~0.5% (5.53 ms vs 5.56 ms of per-step GEMM) anyway. vLLM and
    SGLang likewise run one scheme across all M on any GPU with FP8 tensor cores."""
    *lead, K = x.shape
    N = weight.shape[0]
    w8a8 = input_scale is not None and e4m3_native()
    segments = None
    if w8a8 and not uniform_scale and not rowwise_scaled_mm_ok():
        segments = scale_segments if scale_segments is not None else weight_scale_segments(weight_scale)
        if not _segments_w8a8_ok(segments):
            w8a8 = False  # W8A16 below is exact for any per-row scale and never calls _scaled_mm
    if _USE_REF:  # numeric-reference fallback (debug / A-B)
        w = weight.to(x.dtype) * weight_scale.to(x.dtype)[:, None]
        out = (x.reshape(-1, K) @ w.t()).reshape(*lead, N)
    elif w8a8:
        out = _scaled_mm(
            x.reshape(-1, K), weight, weight_scale, input_scale, uniform_scale, x.dtype,
            scale_segments=segments,
        ).reshape(*lead, N)
    elif input_scale is None and 1 < x.numel() // K <= _ROWS_GEMV_MAX_M:
        # Decode-sized batch (speculative verify rows): memory-bound rows GEMV, exact
        # activations -- faster than cuBLASLt's row-wise fp8 tile GEMM at these shapes.
        out = _gemv_rows(
            x.reshape(-1, K).contiguous(), e4m3_kernel_view(weight), weight_scale, x.dtype,
        ).reshape(*lead, N)
    elif input_scale is None and x.numel() // K > 1 and dynamic_w8a8_ok():
        # No calibrated activation scale (dynamic-quant checkpoints): per-token dynamic
        # W8A8 through cuBLASLt. The W8A16 tile GEMM below is latency-bound at small M
        # (a speculative verify's 3 rows run it ~2x slower than the M=1 GEMV).
        out = _dynamic_scaled_mm(x.reshape(-1, K), weight, weight_scale, x.dtype).reshape(*lead, N)
    elif x.numel() // K == 1:
        out = _gemv(x.reshape(K), e4m3_kernel_view(weight), weight_scale, x.dtype).reshape(*lead, N)
    else:
        out = _gemm(
            x.reshape(-1, K), e4m3_kernel_view(weight), weight_scale, x.dtype,
        ).reshape(*lead, N)
    if bias is not None:
        out = out + bias.to(out.dtype)
    return out


def fp8_rows_gemv(a: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor,
                  out_dtype: torch.dtype) -> torch.Tensor:
    """Public alias of the decode rows GEMV (M <= 4), for benchmarks and tests."""
    return _gemv_rows(a.contiguous(), e4m3_kernel_view(weight), weight_scale, out_dtype)


def quantize_fp8_per_row(
    w: torch.Tensor, chunk_rows: int = 16384
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-output-row fp8-e4m3 quantization of a ``[N, K]`` weight: ``w ~= q * scale[:, None]``
    with ``scale = amax_row / 448`` (e4m3 finite max). The layout :class:`Fp8PerTensorLinear`
    consumes (its ``weight_scale`` is per row already). Row-chunked so a 248K-row lm_head
    costs a ~170 MB fp32 temporary instead of the whole matrix; per-row scales make the
    chunking exact. All-zero rows quantize to zero codes with a finite scale."""
    assert w.dim() == 2, w.shape
    n = w.shape[0]
    q = torch.empty(w.shape, dtype=FP8, device=w.device)
    scale = torch.empty(n, dtype=torch.float32, device=w.device)
    for r0 in range(0, n, chunk_rows):
        blk = w[r0 : r0 + chunk_rows].float()
        s = (blk.abs().amax(dim=1) / 448.0).clamp_min(1e-12)
        q[r0 : r0 + chunk_rows] = (blk / s[:, None]).clamp(-448.0, 448.0).to(FP8)
        scale[r0 : r0 + chunk_rows] = s
    return q, scale


# ======================================================================================
# BaseOP linear layers (TP=1, replicated). Buffers: fp8 ``weight`` + fp32 ``weight_scale``.
# ======================================================================================
class Fp8PerTensorLinear(BaseOP):
    """Replicated per-tensor-FP8 linear: fp8-e4m3 ``weight`` ``[out, in]`` + per-row fp32
    ``weight_scale`` ``[out]`` (a genuine per-tensor weight stores the same scalar in every
    row; a fused projection stores each part's scalar across its own rows).

    ``input_scale`` (modelopt's calibrated per-tensor activation scale) is optional: when the
    checkpoint ships it, batched decode runs W8A8 through cuBLASLt; when it does not (models
    that quantize to fp8 at load, e.g. glm_moe_dsa), every batch size stays W8A16."""

    def __init__(self, in_features: int, out_features: int, has_bias: bool = False):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.empty(out_features, in_features, dtype=FP8)
        self.weight_scale = torch.empty(out_features, dtype=torch.float32)
        self.bias = torch.empty(out_features) if has_bias else None
        # Set from the state dict when present. Left as None (not a tensor) so BaseOP's
        # reflective state_dict/load_state_dict skip it entirely on checkpoints without one.
        self.input_scale: torch.Tensor | None = None
        self._uniform_scale = False
        self._scale_segments: list[tuple[int, int]] | None = None

    def load_state_dict(self, state_dict, *, prefix: str = "", _internal: bool = False) -> None:
        # Taken out before BaseOP's reflective pass (so it is not an "unexpected key") and
        # put back after (so that pass does not try to pop it a second time on a reload).
        input_scale = state_dict.pop(_concat_prefix(prefix, "input_scale"), None)
        self.input_scale = None
        super().load_state_dict(state_dict, prefix=prefix, _internal=_internal)
        self.input_scale = input_scale
        # Tensor-wise scaling needs one scalar for the whole weight; a fused projection is
        # only piecewise-constant, so decide once here rather than syncing on every forward.
        scale = self.weight_scale
        self._uniform_scale = bool((scale == scale[0]).all().item())
        # Segments for the per-part path; decide row-wise safety now, not under graph capture.
        # Only the static-input_scale path consults them (fp8_pertensor_linear), so a
        # genuinely per-channel scale (N distinct values, e.g. weights quantized at load)
        # skips the O(N) host walk entirely.
        self._scale_segments = (
            None
            if self._uniform_scale or self.input_scale is None
            else weight_scale_segments(scale)
        )
        if self.input_scale is not None and not self._uniform_scale:
            rowwise_scaled_mm_ok()
        if self.input_scale is None:
            dynamic_w8a8_ok()  # decide the dynamic W8A8 path now, never under capture

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return fp8_pertensor_linear(
            x, self.weight, self.weight_scale, self.bias,
            self.input_scale, self._uniform_scale, scale_segments=self._scale_segments,
        )


class Fp8PerTensorColMerged(Fp8PerTensorLinear):
    """Column-merged per-tensor-FP8 linear (drop-in for ``LinearColParallelMerged`` at TP=1):
    one fp8 weight concatenating several projections along the output dim; the caller splits
    the bf16 output by ``output_sizes`` as before."""

    def __init__(self, in_features: int, output_sizes: list[int], has_bias: bool = False):
        self.output_sizes = list(output_sizes)
        super().__init__(in_features, sum(output_sizes), has_bias)


__all__ = [
    "FP8",
    "Fp8PerTensorLinear",
    "Fp8PerTensorColMerged",
    "bf16_rows_gemv",
    "bf16_rows_gemv_applicable",
    "rows_gemv_with_epilogue",
    "fp8_pertensor_linear",
    "fp8_rows_gemv",
    "quantize_fp8_per_row",
    "rowwise_scaled_mm_ok",
    "weight_scale_segments",
]
