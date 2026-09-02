"""Fused cross-layer lookahead router (one launch per lookahead hook).

Computes, for every decode row, the xmix expert-prefetch proxy of moe/prefetch.py's
``LayerLookahead`` -- the target layer's own hyper-connection mix followed by its router
gate and a per-row top-k -- as ONE Triton kernel instead of the ~12 captured nodes of the
torch chain (grouped norm, down GEMM, silu, up GEMM, gate-mix, gate GEMM, topk, reshape,
cast). The math mirrors ``GatedResidual._mix_torch`` (models/qwen4_exp/hc.py):

    rn      = grouped_rmsnorm(x) * (1 + w_norm)          # per hc stream, fp32 stats
    lora    = rn @ W_down[:lowrank].T                    # merged weight's lora rows
    s       = silu(lora / hc)
    gate    = s @ W_up.T                                 # [width]
    mixed   = mean_i( sigmoid(gate_i) * rn_i )           # over the hc streams
    logits  = mixed @ W_gate.T                           # [num_experts]
    cand    = flat_base + top_k(logits)                  # int32 flat expert ids, row-major

Single program, rows padded to 16 so every reduction runs through tensor-core ``tl.dot``
(weights are consumed in place as bf16 -- no stacked copies, no extra VRAM). All shapes
are constexpr and the output buffer is fixed-shape, so the launch is CUDA-graph
capturable exactly like the torch chain it replaces. Intermediate precision follows the
torch reference: fp32 norm statistics and accumulators, bf16 operands into each GEMM.

qwen4_exp geometry gets specialized at first call (width 10240, hc 4, lowrank 320,
hidden 2560, experts 512); other geometries compile their own instance as long as
width == hc * hidden, K_TILE_B and H_TILE divide hidden, and lowrank % 16 == 0.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _lookahead_route_kernel(
    x_ptr,        # [R, width] bf16: residual rows at the ISSUE layer
    w_norm_ptr,   # [width] target hc_norm.weight (zero-centered; applied as 1 + w)
    w_down_ptr,   # [>=lowrank, width] target input_mix_weight_down_block_inject.weight
    w_up_ptr,     # [width, lowrank] target input_mix_weight_up.weight
    w_gate_ptr,   # [E, hidden] target mlp.gate.weight
    cand_ptr,     # [R * TOPK] int32 out: flat expert ids, row-major
    logits_ptr,   # [R, E] fp32 out (debug/testing), only written when HAS_LOGITS
    num_rows,
    stride_x,
    flat_base,
    eps,
    HC: tl.constexpr,
    HIDDEN: tl.constexpr,
    WIDTH: tl.constexpr,
    LOWRANK: tl.constexpr,
    LORA_PAD: tl.constexpr,   # next power of 2 >= LOWRANK (block shapes must be pow2)
    E: tl.constexpr,
    TOPK: tl.constexpr,
    R_PAD: tl.constexpr,      # 16: dot's minimum M
    K_TILE_A: tl.constexpr,   # width tile of the rstd pass; must divide HIDDEN
    K_TILE_B: tl.constexpr,   # width tile of the down GEMM; must divide HIDDEN
    H_TILE: tl.constexpr,     # hidden tile of the up/mix/gate phase; must divide HIDDEN
    HAS_LOGITS: tl.constexpr,
):
    rows = tl.arange(0, R_PAD)
    row_mask = rows < num_rows

    # ---- Phase A: per-(row, stream) inverse RMS -------------------------------------
    rstd = tl.zeros((R_PAD, HC), tl.float32)
    for i in tl.static_range(HC):
        ss = tl.zeros((R_PAD,), tl.float32)
        for t in tl.range(0, HIDDEN // K_TILE_A):
            cols = i * HIDDEN + t * K_TILE_A + tl.arange(0, K_TILE_A)
            x = tl.load(
                x_ptr + rows[:, None] * stride_x + cols[None, :],
                mask=row_mask[:, None],
                other=0.0,
            ).to(tl.float32)
            ss += tl.sum(x * x, axis=1)
        r_i = tl.rsqrt(ss / HIDDEN + eps)
        rstd = tl.where(tl.arange(0, HC)[None, :] == i, r_i[:, None], rstd)

    # ---- Phase B: lora = rn @ W_down[:lowrank].T; s = silu(lora / hc) ---------------
    lora_mask = tl.arange(0, LORA_PAD) < LOWRANK
    acc = tl.zeros((R_PAD, LORA_PAD), tl.float32)
    for t in tl.range(0, WIDTH // K_TILE_B):
        stream = t // (HIDDEN // K_TILE_B)
        rs = tl.sum(tl.where(tl.arange(0, HC)[None, :] == stream, rstd, 0.0), axis=1)
        cols = t * K_TILE_B + tl.arange(0, K_TILE_B)
        x = tl.load(
            x_ptr + rows[:, None] * stride_x + cols[None, :],
            mask=row_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        wn = tl.load(w_norm_ptr + cols).to(tl.float32)
        rn = (x * rs[:, None] * (1.0 + wn[None, :])).to(tl.bfloat16)
        wd = tl.load(  # [K_TILE_B, LORA_PAD] = W_down[:lowrank, cols].T, zero-padded
            w_down_ptr + tl.arange(0, LORA_PAD)[None, :] * WIDTH + cols[:, None],
            mask=lora_mask[None, :],
            other=0.0,
        )
        acc += tl.dot(rn, wd)
    v = acc / HC
    sb = (v * tl.sigmoid(v)).to(tl.bfloat16)  # silu, cast like the torch chain

    # ---- Phase C: up GEMM + sigmoid-stream-mix + gate GEMM, tiled over hidden -------
    logits = tl.zeros((R_PAD, E), tl.float32)
    for t in tl.range(0, HIDDEN // H_TILE):
        h = t * H_TILE + tl.arange(0, H_TILE)
        mixed = tl.zeros((R_PAD, H_TILE), tl.float32)
        for i in tl.static_range(HC):
            j = i * HIDDEN + h
            wu = tl.load(  # [LORA_PAD, H_TILE] = W_up[j, :].T, zero-padded rows
                w_up_ptr + j[None, :] * LOWRANK + tl.arange(0, LORA_PAD)[:, None],
                mask=lora_mask[:, None],
                other=0.0,
            )
            ga = tl.dot(sb, wu)  # [R_PAD, H_TILE] fp32
            rs = tl.sum(tl.where(tl.arange(0, HC)[None, :] == i, rstd, 0.0), axis=1)
            x = tl.load(
                x_ptr + rows[:, None] * stride_x + j[None, :],
                mask=row_mask[:, None],
                other=0.0,
            ).to(tl.float32)
            wn = tl.load(w_norm_ptr + j).to(tl.float32)
            mixed += tl.sigmoid(ga) * (x * rs[:, None] * (1.0 + wn[None, :]))
        mixed_b = (mixed / HC).to(tl.bfloat16)
        wg = tl.load(  # [H_TILE, E] = W_gate[:, h].T
            w_gate_ptr + tl.arange(0, E)[None, :] * HIDDEN + h[:, None]
        )
        logits += tl.dot(mixed_b, wg)

    if HAS_LOGITS:
        tl.store(
            logits_ptr + rows[:, None] * E + tl.arange(0, E)[None, :],
            logits,
            mask=row_mask[:, None],
        )

    # ---- Phase D: per-row top-k -> flat candidate ids -------------------------------
    for k in tl.static_range(TOPK):
        best = tl.argmax(logits, axis=1)  # [R_PAD]
        tl.store(
            cand_ptr + rows * TOPK + k,
            (flat_base + best).to(tl.int32),
            mask=row_mask,
        )
        logits = tl.where(
            tl.arange(0, E)[None, :] == best[:, None], float("-inf"), logits
        )


# ---------------------------------------------------------------------------------
# Split fused pipeline: the mono-kernel above is correct but single-program, so ONE
# SM must stream all ~16 MB of target weights (measured 604 us vs the torch chain's
# 33 us/hook under graph replay). These four kernels keep the fusion wins (no glue
# launches, no reshape/cast/topk nodes) while spreading the GEMM weight reads across
# ~40 programs each, with fp32 atomic reductions into small shared scratch buffers
# that every hook of the step reuses (hooks are stream-serialized, so one scratch
# set suffices and its addresses stay static for graph capture).
#   K0: per-(row, stream) rstd + zero the lora/logits accumulators
#   K1: lora += rn @ W_down[:lowrank].T          (grid: lora tiles x K splits)
#   K3: logits += mix_tile @ W_gate[:, tile].T   (grid: hidden tiles; silu+up+mix inside)
#   K4: per-row top-k -> flat candidate ids
# ---------------------------------------------------------------------------------


@triton.jit
def _la_rstd_zero_kernel(
    x_ptr, rstd_ptr, lora_ptr, logits_ptr,
    num_rows, stride_x, eps,
    HC: tl.constexpr, HIDDEN: tl.constexpr,
    LORA_PAD: tl.constexpr, E: tl.constexpr,
    R_PAD: tl.constexpr, K_TILE: tl.constexpr,
):
    rows = tl.arange(0, R_PAD)
    row_mask = rows < num_rows
    for i in tl.static_range(HC):
        ss = tl.zeros((R_PAD,), tl.float32)
        for t in tl.range(0, HIDDEN // K_TILE):
            cols = i * HIDDEN + t * K_TILE + tl.arange(0, K_TILE)
            x = tl.load(
                x_ptr + rows[:, None] * stride_x + cols[None, :],
                mask=row_mask[:, None], other=0.0,
            ).to(tl.float32)
            ss += tl.sum(x * x, axis=1)
        tl.store(rstd_ptr + rows * HC + i, tl.rsqrt(ss / HIDDEN + eps))
    z_l = tl.zeros((R_PAD, LORA_PAD), tl.float32)
    tl.store(lora_ptr + rows[:, None] * LORA_PAD + tl.arange(0, LORA_PAD)[None, :], z_l)
    z_e = tl.zeros((R_PAD, E), tl.float32)
    tl.store(logits_ptr + rows[:, None] * E + tl.arange(0, E)[None, :], z_e)


@triton.jit
def _la_down_kernel(
    x_ptr, w_norm_ptr, w_down_ptr, rstd_ptr, lora_ptr,
    num_rows, stride_x,
    HC: tl.constexpr, HIDDEN: tl.constexpr, WIDTH: tl.constexpr,
    LOWRANK: tl.constexpr, LORA_PAD: tl.constexpr,
    R_PAD: tl.constexpr, M_TILE: tl.constexpr, K_SPLIT: tl.constexpr,
    K_TILE: tl.constexpr,
):
    m = tl.program_id(0)
    ks = tl.program_id(1)
    rows = tl.arange(0, R_PAD)
    row_mask = rows < num_rows
    k0 = ks * K_SPLIT
    stream = k0 // HIDDEN  # K_SPLIT divides HIDDEN, so a split never crosses streams
    rs = tl.load(rstd_ptr + rows * HC + stream, mask=row_mask, other=0.0)
    m_offs = m * M_TILE + tl.arange(0, M_TILE)
    m_mask = m_offs < LOWRANK
    acc = tl.zeros((R_PAD, M_TILE), tl.float32)
    for t in tl.range(0, K_SPLIT // K_TILE):
        cols = k0 + t * K_TILE + tl.arange(0, K_TILE)
        x = tl.load(
            x_ptr + rows[:, None] * stride_x + cols[None, :],
            mask=row_mask[:, None], other=0.0,
        ).to(tl.float32)
        wn = tl.load(w_norm_ptr + cols).to(tl.float32)
        rn = (x * rs[:, None] * (1.0 + wn[None, :])).to(tl.bfloat16)
        wd = tl.load(
            w_down_ptr + m_offs[None, :] * WIDTH + cols[:, None],
            mask=m_mask[None, :], other=0.0,
        )
        acc += tl.dot(rn, wd)
    tl.atomic_add(
        lora_ptr + rows[:, None] * LORA_PAD + m_offs[None, :],
        acc,
        mask=row_mask[:, None] & m_mask[None, :],
    )


@triton.jit
def _la_up_mix_gate_kernel(
    x_ptr, w_norm_ptr, w_up_ptr, w_gate_ptr, rstd_ptr, lora_ptr, logits_ptr,
    num_rows, stride_x,
    HC: tl.constexpr, HIDDEN: tl.constexpr,
    LOWRANK: tl.constexpr, LORA_PAD: tl.constexpr, E: tl.constexpr,
    R_PAD: tl.constexpr, H_TILE: tl.constexpr,
):
    t = tl.program_id(0)
    rows = tl.arange(0, R_PAD)
    row_mask = rows < num_rows
    lora_offs = tl.arange(0, LORA_PAD)
    lora_mask = lora_offs < LOWRANK
    v = tl.load(
        lora_ptr + rows[:, None] * LORA_PAD + lora_offs[None, :],
        mask=row_mask[:, None] & lora_mask[None, :], other=0.0,
    ) / HC
    sb = (v * tl.sigmoid(v)).to(tl.bfloat16)  # silu of the split-K sum
    h = t * H_TILE + tl.arange(0, H_TILE)
    mixed = tl.zeros((R_PAD, H_TILE), tl.float32)
    for i in tl.static_range(HC):
        j = i * HIDDEN + h
        wu = tl.load(
            w_up_ptr + j[None, :] * LOWRANK + lora_offs[:, None],
            mask=lora_mask[:, None], other=0.0,
        )
        ga = tl.dot(sb, wu)
        rs = tl.load(rstd_ptr + rows * HC + i, mask=row_mask, other=0.0)
        x = tl.load(
            x_ptr + rows[:, None] * stride_x + j[None, :],
            mask=row_mask[:, None], other=0.0,
        ).to(tl.float32)
        wn = tl.load(w_norm_ptr + j).to(tl.float32)
        mixed += tl.sigmoid(ga) * (x * rs[:, None] * (1.0 + wn[None, :]))
    mixed_b = (mixed / HC).to(tl.bfloat16)
    wg = tl.load(w_gate_ptr + tl.arange(0, E)[None, :] * HIDDEN + h[:, None])
    part = tl.dot(mixed_b, wg)  # [R_PAD, E] fp32
    tl.atomic_add(
        logits_ptr + rows[:, None] * E + tl.arange(0, E)[None, :],
        part,
        mask=row_mask[:, None],
    )


@triton.jit
def _la_topk_kernel(
    logits_ptr, cand_ptr,
    num_rows, flat_base,
    E: tl.constexpr, TOPK: tl.constexpr, R_PAD: tl.constexpr,
):
    rows = tl.arange(0, R_PAD)
    row_mask = rows < num_rows
    logits = tl.load(
        logits_ptr + rows[:, None] * E + tl.arange(0, E)[None, :],
        mask=row_mask[:, None], other=float("-inf"),
    )
    for k in tl.static_range(TOPK):
        best = tl.argmax(logits, axis=1)
        tl.store(cand_ptr + rows * TOPK + k, (flat_base + best).to(tl.int32), mask=row_mask)
        logits = tl.where(tl.arange(0, E)[None, :] == best[:, None], float("-inf"), logits)


class LookaheadScratch:
    """Static accumulator buffers shared by every lookahead hook of a step (hooks are
    stream-serialized, and K0 re-zeroes before each hook's accumulation)."""

    def __init__(
        self, device: torch.device, hc: int, lora_pad: int = 512, e: int = 512
    ) -> None:
        # rstd rows are indexed as rows * hc + stream: keep it exactly [16, hc].
        self.rstd = torch.zeros(16, hc, dtype=torch.float32, device=device)
        self.lora = torch.zeros(16, lora_pad, dtype=torch.float32, device=device)
        self.logits = torch.zeros(16, e, dtype=torch.float32, device=device)


def lookahead_route_split(
    hidden: torch.Tensor,
    w_norm: torch.Tensor,
    w_down: torch.Tensor,
    w_up: torch.Tensor,
    w_gate: torch.Tensor,
    scratch: LookaheadScratch,
    cand: torch.Tensor,
    *,
    hc: int,
    eps: float,
    topk: int,
    flat_base: int,
) -> torch.Tensor:
    """4-launch fused proxy router; writes int32 flat candidate ids into ``cand``."""
    rows, width = hidden.shape
    e, hid = w_gate.shape
    lowrank = w_up.shape[1]
    lora_pad = scratch.lora.shape[1]
    assert width == hc * hid and rows <= 16
    m_tile = 64
    k_split = hid // 2  # 2 splits per stream; never crosses a stream boundary
    _la_rstd_zero_kernel[(1,)](
        hidden, scratch.rstd, scratch.lora, scratch.logits,
        rows, hidden.stride(0), eps,
        HC=hc, HIDDEN=hid, LORA_PAD=lora_pad, E=e, R_PAD=16, K_TILE=512,
        num_warps=8,
    )
    _la_down_kernel[((lowrank + m_tile - 1) // m_tile, width // k_split)](
        hidden, w_norm, w_down, scratch.rstd, scratch.lora,
        rows, hidden.stride(0),
        HC=hc, HIDDEN=hid, WIDTH=width, LOWRANK=lowrank, LORA_PAD=lora_pad,
        R_PAD=16, M_TILE=m_tile, K_SPLIT=k_split, K_TILE=128,
        num_warps=4, num_stages=2,
    )
    _la_up_mix_gate_kernel[(hid // 64,)](
        hidden, w_norm, w_up, w_gate, scratch.rstd, scratch.lora, scratch.logits,
        rows, hidden.stride(0),
        HC=hc, HIDDEN=hid, LOWRANK=lowrank, LORA_PAD=lora_pad, E=e,
        R_PAD=16, H_TILE=64,
        num_warps=4, num_stages=2,
    )
    _la_topk_kernel[(1,)](
        scratch.logits, cand, rows, flat_base,
        E=e, TOPK=topk, R_PAD=16,
        num_warps=4,
    )
    return cand


def lookahead_route(
    hidden: torch.Tensor,
    w_norm: torch.Tensor,
    w_down: torch.Tensor,
    w_up: torch.Tensor,
    w_gate: torch.Tensor,
    *,
    hc: int,
    eps: float,
    topk: int,
    flat_base: int,
    logits_out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the fused proxy router; returns int32 flat candidate ids [rows * topk]."""
    rows, width = hidden.shape
    e, hid = w_gate.shape
    lowrank = w_up.shape[1]
    assert width == hc * hid and rows <= 16, (rows, width, hc, hid)
    cand = torch.empty(rows * topk, dtype=torch.int32, device=hidden.device)
    _lookahead_route_kernel[(1,)](
        hidden,
        w_norm,
        w_down,
        w_up,
        w_gate,
        cand,
        logits_out if logits_out is not None else cand,
        rows,
        hidden.stride(0),
        flat_base,
        eps,
        HC=hc,
        HIDDEN=hid,
        WIDTH=width,
        LOWRANK=lowrank,
        LORA_PAD=max(16, 1 << (lowrank - 1).bit_length()),
        E=e,
        TOPK=topk,
        R_PAD=16,
        K_TILE_A=512,
        K_TILE_B=64,
        H_TILE=64,
        HAS_LOGITS=logits_out is not None,
        num_warps=8,
        num_stages=1,
    )
    return cand


__all__ = ["LookaheadScratch", "lookahead_route", "lookahead_route_split"]
