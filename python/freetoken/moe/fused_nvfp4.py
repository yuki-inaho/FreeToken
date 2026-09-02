"""Host orchestration for the inline-dequant NVFP4 fused-MoE path.

Mirrors :mod:`freetoken.moe.fused` (gemm1 -> act -> gemm2 -> sum-reduce) but the two
grouped GEMMs read the NVFP4 expert cache directly and dequantize inside the K-loop,
so no BF16 copy of the experts is ever materialized.
"""

from __future__ import annotations

import os

from typing import Any, Dict

import torch
import triton
import triton.language as tl

from freetoken.kernel import moe_sum_reduce_triton
from freetoken.kernel.triton.e4m3_compat import e4m3_kernel_view
from freetoken.kernel.triton.nvfp4_fused_moe import (
    _decode_nvfp4_marlin_dedup_kernel,
    _decode_nvfp4_marlin_kernel,
    _decode_nvfp4_moe_kernel,
    _e2m1_lut,
    _prefill_nvfp4_moe_kernel,
)
from freetoken.layers import (
    gelu_and_mul,
    gelu_tanh_and_mul,
    silu_and_mul,
    swigluoai_and_mul,
)
from freetoken.moe.fused import moe_align_block_size

_ACT = {"silu": silu_and_mul, "gelu": gelu_and_mul, "gelu_tanh": gelu_tanh_and_mul}


def _run_act(
    activation: str,
    gate_up: torch.Tensor,
    out: torch.Tensor,
    act_alpha: float,
    act_limit: float,
) -> None:
    """gemm1 -> gemm2 activation dispatch. ``swigluoai`` (MiniMax-M3, clamped
    gpt-oss swiglu over the banks' uninterleaved [gate; up] halves) carries the
    per-model ``act_alpha``/``act_limit`` scalars; the plain *_and_mul kinds
    ignore them."""
    if activation == "swigluoai":
        swigluoai_and_mul(gate_up, out, alpha=act_alpha, limit=act_limit)
        return
    _ACT[activation](gate_up, out)

# Decode is captured into a CUDA graph, so the config must be fixed (no triton.autotune,
# which benchmarks at run time). Tuned offline against the NVFP4 decode kernels.
# These drive the original LUT-gather decode (_decode_gemm), kept only for A/B.
_DECODE_BLOCK_N = 64
_DECODE_BLOCK_KB = 128
_DECODE_WARPS = 4

# Decode gemm2 forms the silu*mul activation on load (FUSE_SILU_MUL in the marlin GEMV) instead
# of a separate act kernel + intermediate; FREETOKEN_MOE_FUSED_ACT=0 restores the act launch.
_FUSED_ACT = os.getenv("FREETOKEN_MOE_FUSED_ACT", "1") != "0"
# Per-route decode GEMV: branchless e2m1 dequant + one contiguous activation load per K-tile
# (bit-identical to the LUT gather); FREETOKEN_MOE_ARITH_DEQUANT=0 restores the LUT path.
_ARITH_DEQUANT = os.getenv("FREETOKEN_MOE_ARITH_DEQUANT", "1") != "0"

# Marlin-style decode config (int32 wide loads + deferred reduction). Offline sweep over
# the qwen35/qwen3moe (I=512/768) decode shapes picked BLOCK_N=16, BLOCK_KW=16 (== 128
# k-values/iter), 4 warps -- the wide load lifts the gate/up GEMM ~43%->~51% of peak BW.
_DECODE_MARLIN_BLOCK_N = 16
_DECODE_MARLIN_BLOCK_KW = 16
_DECODE_MARLIN_WARPS = 4
# Deep-K variant: at K > 2048 (qwen4_exp gate_up, K=2560) a narrower N tile with the whole
# K strip in one program iteration measures ~13% faster (18.6 vs 21.0us); short-K shapes
# regress under it, so the split is by K, not by gemm position.
_DECODE_MARLIN_DEEPK_BLOCK_N = 8
_DECODE_MARLIN_DEEPK_BLOCK_KW = 128
_DECODE_MARLIN_DEEPK_THRESHOLD = 2048


def _tl_dtype(dt: torch.dtype):
    if dt == torch.bfloat16:
        return tl.bfloat16
    if dt == torch.float16:
        return tl.float16
    return tl.float32


def _decode_gemm(
    a: torch.Tensor,
    packed: torch.Tensor,
    scale: torch.Tensor,
    glob: torch.Tensor,
    c: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    mul_routed_weight: bool,
    a_row_is_route: bool,
    zero_on_skip: bool = False,
) -> None:
    M, top_k = topk_ids.shape
    N = packed.shape[1]
    K = packed.shape[2] * 2
    scale = e4m3_kernel_view(scale)
    total_routes = M * top_k
    grid = (total_routes, triton.cdiv(N, _DECODE_BLOCK_N))
    _decode_nvfp4_moe_kernel[grid](
        a, packed, scale, glob, c, topk_weights, topk_ids,
        _e2m1_lut(a.device.index),
        total_routes, N, K,
        a.stride(0), a.stride(1),
        packed.stride(0), packed.stride(1), packed.stride(2),
        scale.stride(0), scale.stride(1), scale.stride(2),
        glob.stride(0), glob.stride(1),
        c.stride(0), c.stride(1), c.stride(2),
        topk_weights.stride(0), topk_weights.stride(1),
        topk_ids.stride(0), topk_ids.stride(1),
        BLOCK_SIZE_N=_DECODE_BLOCK_N,
        BLOCK_SIZE_KB=_DECODE_BLOCK_KB,
        TOP_K=top_k,
        A_ROW_IS_ROUTE=a_row_is_route,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        compute_type=_tl_dtype(c.dtype),
        ZERO_ON_SKIP=zero_on_skip,
        num_warps=_DECODE_WARPS,
    )


def _decode_gemm_marlin(
    a: torch.Tensor,
    packed: torch.Tensor,
    scale: torch.Tensor,
    glob: torch.Tensor,
    c: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    mul_routed_weight: bool,
    a_row_is_route: bool,
    zero_on_skip: bool = False,
    fused_act_off: int | None = None,
) -> None:
    """Marlin-style decode GEMV: int32 wide loads + deferred reduction
    (:func:`_decode_nvfp4_marlin_kernel`). ``packed`` is the uint8 ``[S, N, K//2]`` bank;
    it is reinterpreted as int32 ``[S, N, K//8]`` (contiguous, K%8==0 for NVFP4)."""
    M, top_k = topk_ids.shape
    N = packed.shape[1]
    K = packed.shape[2] * 2
    packed_i32 = packed.view(torch.int32)  # [S, N, K // 8]
    scale = e4m3_kernel_view(scale)
    total_routes = M * top_k
    deep_k = K > _DECODE_MARLIN_DEEPK_THRESHOLD
    block_n = _DECODE_MARLIN_DEEPK_BLOCK_N if deep_k else _DECODE_MARLIN_BLOCK_N
    block_kw = _DECODE_MARLIN_DEEPK_BLOCK_KW if deep_k else _DECODE_MARLIN_BLOCK_KW
    grid = (total_routes, triton.cdiv(N, block_n))
    _decode_nvfp4_marlin_kernel[grid](
        a, packed_i32, scale, glob, c, topk_weights, topk_ids,
        _e2m1_lut(a.device.index),
        total_routes, N, K,
        a.stride(0), a.stride(1),
        packed_i32.stride(0), packed_i32.stride(1), packed_i32.stride(2),
        scale.stride(0), scale.stride(1), scale.stride(2),
        glob.stride(0), glob.stride(1),
        c.stride(0), c.stride(1), c.stride(2),
        topk_weights.stride(0), topk_weights.stride(1),
        topk_ids.stride(0), topk_ids.stride(1),
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_KW=block_kw,
        TOP_K=top_k,
        A_ROW_IS_ROUTE=a_row_is_route,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        compute_type=_tl_dtype(c.dtype),
        ZERO_ON_SKIP=zero_on_skip,
        FUSE_SILU_MUL=fused_act_off is not None,
        act_off=0 if fused_act_off is None else int(fused_act_off),
        ARITH_DEQUANT=_ARITH_DEQUANT,
        num_warps=_DECODE_MARLIN_WARPS,
    )


# Dedup variant (hybrid decode, M <= 4): programs per distinct expert. The deep-K gate_up
# tile keeps 4 route accumulators, so its K strip is halved vs the per-route config.
_DEDUP_DEEPK_BLOCK_KW = 64


def _decode_gemm_marlin_dedup(
    a: torch.Tensor,
    packed: torch.Tensor,
    scale: torch.Tensor,
    glob: torch.Tensor,
    c: torch.Tensor,
    topk_weights: torch.Tensor,
    uniq_slots: torch.Tensor,
    pairs: torch.Tensor,
    top_k: int,
    mul_routed_weight: bool,
    a_row_is_route: bool,
) -> None:
    """Marlin-style decode GEMV over DISTINCT experts (``uniq_slots`` [U], ``pairs`` [U, MAXB]
    from the hybrid ensure kernel); each expert's rows serve all its routes."""
    N = packed.shape[1]
    K = packed.shape[2] * 2
    packed_i32 = packed.view(torch.int32)
    scale = e4m3_kernel_view(scale)
    U, maxb = pairs.shape
    deep_k = K > _DECODE_MARLIN_DEEPK_THRESHOLD
    block_n = _DECODE_MARLIN_DEEPK_BLOCK_N if deep_k else _DECODE_MARLIN_BLOCK_N
    block_kw = _DEDUP_DEEPK_BLOCK_KW if deep_k else _DECODE_MARLIN_BLOCK_KW
    grid = (U, triton.cdiv(N, block_n))
    _decode_nvfp4_marlin_dedup_kernel[grid](
        a, packed_i32, scale, glob, c, topk_weights, uniq_slots, pairs,
        _e2m1_lut(a.device.index),
        N, K,
        a.stride(0), a.stride(1),
        packed_i32.stride(0), packed_i32.stride(1), packed_i32.stride(2),
        scale.stride(0), scale.stride(1), scale.stride(2),
        glob.stride(0), glob.stride(1),
        c.stride(0), c.stride(1), c.stride(2),
        topk_weights.stride(0), topk_weights.stride(1),
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_KW=block_kw,
        TOP_K=top_k,
        MAXB=maxb,
        A_ROW_IS_ROUTE=a_row_is_route,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        compute_type=_tl_dtype(c.dtype),
        num_warps=_DECODE_MARLIN_WARPS,
    )


def _fused_experts_decode_nvfp4(
    gemm_fn,
    hidden_states: torch.Tensor,
    gate_up_packed: torch.Tensor,
    gate_up_scale: torch.Tensor,
    gate_up_global: torch.Tensor,
    down_packed: torch.Tensor,
    down_scale: torch.Tensor,
    down_global: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str,
    apply_router_weight_on_input: bool,
    act_alpha: float = 1.702,
    act_limit: float = 7.0,
    finalize=None,
    split=None,
    dedup=None,
    epilogue=None,
) -> torch.Tensor:
    """Shared decode body (gemm1 -> act -> gemm2 -> sum-reduce); ``gemm_fn`` is either
    the marlin-style int32 GEMV (:func:`_decode_gemm_marlin`) or the original LUT-gather
    GEMV (:func:`_decode_gemm`), both with the same calling convention.

    Routes with ``topk_ids == -1`` are skipped (hybrid backend: the CPU computes them);
    ``finalize``, if given, is called right before the sum-reduce and returns a ``[M, H]``
    addend folded into it (the CPU partial) -- so the join with the CPU pool sits as late
    as possible and costs no extra kernel.

    ``split`` = ``(hit_ids, fetched_ids, join_fn)`` (hybrid demand-copy overlap): gemm1 runs
    first over the routes already resident (``hit_ids``), concurrently with the PCIe gather
    of this step's fetched experts on its side stream; ``join_fn()`` then waits for the
    gather and gemm1 covers the fetched routes (``fetched_ids``, disjoint rows of ``ic1``).
    One extra launch per layer, ~the copy's duration hidden behind the hit GEMV.

    ``dedup`` = ``(uniq_slots, uniq_hit, uniq_fetched, pairs, join_fn_or_None)``: the routes
    grouped by distinct expert (hybrid ensure kernel); both GEMVs run one program per
    expert, reading each routed expert once for all its rows, and the sum-reduce skips the
    CPU routes' (unwritten) rows via ``topk_ids``."""
    M, H = hidden_states.shape
    top_k = topk_ids.shape[1]
    two_i = gate_up_packed.shape[1]
    inter = two_i // 2
    dev, dt = hidden_states.device, hidden_states.dtype

    ic1 = torch.empty((M, top_k, two_i), device=dev, dtype=dt)
    if dedup is not None:
        uniq_all, uniq_hit, uniq_fetched, pairs, join_fn = dedup
        if join_fn is None:
            _decode_gemm_marlin_dedup(
                hidden_states, gate_up_packed, gate_up_scale, gate_up_global, ic1,
                topk_weights, uniq_all, pairs, top_k, apply_router_weight_on_input, False,
            )
        else:
            _decode_gemm_marlin_dedup(
                hidden_states, gate_up_packed, gate_up_scale, gate_up_global, ic1,
                topk_weights, uniq_hit, pairs, top_k, apply_router_weight_on_input, False,
            )
            join_fn()
            _decode_gemm_marlin_dedup(
                hidden_states, gate_up_packed, gate_up_scale, gate_up_global, ic1,
                topk_weights, uniq_fetched, pairs, top_k, apply_router_weight_on_input, False,
            )
    elif split is None:
        gemm_fn(
            hidden_states, gate_up_packed, gate_up_scale, gate_up_global,
            ic1, topk_weights, topk_ids, apply_router_weight_on_input, False,
        )
    else:
        hit_ids, fetched_ids, join_fn = split
        gemm_fn(
            hidden_states, gate_up_packed, gate_up_scale, gate_up_global,
            ic1, topk_weights, hit_ids, apply_router_weight_on_input, False,
        )
        join_fn()
        gemm_fn(
            hidden_states, gate_up_packed, gate_up_scale, gate_up_global,
            ic1, topk_weights, fetched_ids, apply_router_weight_on_input, False,
        )
    ic3 = torch.empty((M, top_k, H), device=dev, dtype=dt)
    fuse_act = (
        _FUSED_ACT and activation == "silu" and dedup is None and gemm_fn is _decode_gemm_marlin
    )
    if fuse_act:
        # gemm2 forms silu(gate) * up on load from gemm1's raw output: no act launch, no ic2.
        gemm_fn(
            ic1.view(-1, two_i), down_packed, down_scale, down_global,
            ic3, topk_weights, topk_ids, not apply_router_weight_on_input, True,
            zero_on_skip=True, fused_act_off=inter,
        )
    else:
        ic2 = torch.empty((M * top_k, inter), device=dev, dtype=dt)
        _run_act(activation, ic1.view(-1, two_i), ic2, act_alpha, act_limit)
        if dedup is not None:
            _decode_gemm_marlin_dedup(
                ic2, down_packed, down_scale, down_global, ic3,
                topk_weights, dedup[0], dedup[3], top_k, not apply_router_weight_on_input, True,
            )
        else:
            gemm_fn(
                ic2, down_packed, down_scale, down_global,
                ic3, topk_weights, topk_ids, not apply_router_weight_on_input, True,
                zero_on_skip=True,
            )
    addend = finalize() if finalize is not None else None
    route_ids = topk_ids.contiguous() if dedup is not None else None
    if epilogue is not None:
        # MoeCombineEpilogue: finish the whole block (routed sum + CPU partial + shared
        # gate + hyper-connection combine) in one kernel; returns the new residual.
        return epilogue.apply(ic3, route_ids, addend)
    out = torch.empty_like(hidden_states)
    moe_sum_reduce_triton(ic3, out, addend=addend, route_ids=route_ids)
    return out


def fused_experts_decode_nvfp4_marlin(
    hidden_states: torch.Tensor,
    gate_up_packed: torch.Tensor,
    gate_up_scale: torch.Tensor,
    gate_up_global: torch.Tensor,
    down_packed: torch.Tensor,
    down_scale: torch.Tensor,
    down_global: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    act_alpha: float = 1.702,
    act_limit: float = 7.0,
    finalize=None,
    split=None,
    dedup=None,
    epilogue=None,
) -> torch.Tensor:
    """Decode inline-NVFP4 MoE using the Marlin-style int32 wide-load GEMV."""
    return _fused_experts_decode_nvfp4(
        _decode_gemm_marlin,
        hidden_states, gate_up_packed, gate_up_scale, gate_up_global,
        down_packed, down_scale, down_global,
        topk_weights, topk_ids, activation, apply_router_weight_on_input,
        act_alpha, act_limit, finalize=finalize, split=split, dedup=dedup, epilogue=epilogue,
    )


def fused_experts_decode_nvfp4_serial(
    hidden_states: torch.Tensor,
    gate_up_packed: torch.Tensor,
    gate_up_scale: torch.Tensor,
    gate_up_global: torch.Tensor,
    down_packed: torch.Tensor,
    down_scale: torch.Tensor,
    down_global: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    act_alpha: float = 1.702,
    act_limit: float = 7.0,
) -> torch.Tensor:
    """Original LUT-gather decode (one program per route, full K reduction). Retained for
    A/B benchmarking against the marlin decode path; not on the production decode path."""
    return _fused_experts_decode_nvfp4(
        _decode_gemm,
        hidden_states, gate_up_packed, gate_up_scale, gate_up_global,
        down_packed, down_scale, down_global,
        topk_weights, topk_ids, activation, apply_router_weight_on_input,
        act_alpha, act_limit,
    )


def _prefill_config(M: int) -> Dict[str, int]:
    # ``BLOCK_SIZE_M`` is coupled to host-side ``moe_align_block_size`` (token padding),
    # so it cannot be picked by triton.autotune; these were chosen by an offline sweep
    # over (BLOCK_M, BLOCK_N, BLOCK_KB, num_warps, num_stages) for the MiniMax-M2 shapes.
    if M <= 64:
        return dict(BLOCK_SIZE_M=16, BLOCK_SIZE_N=64, BLOCK_SIZE_KB=32,
                    GROUP_SIZE_M=1, num_warps=8, num_stages=4)
    return dict(BLOCK_SIZE_M=32, BLOCK_SIZE_N=64, BLOCK_SIZE_KB=32,
                GROUP_SIZE_M=8, num_warps=8, num_stages=4)


def _prefill_gemm(
    a: torch.Tensor,
    packed: torch.Tensor,
    scale: torch.Tensor,
    glob: torch.Tensor,
    c: torch.Tensor,
    topk_weights_flat: torch.Tensor,
    sorted_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    num_valid_tokens: int,
    kernel_top_k: int,
    mul_routed_weight: bool,
    cfg: Dict[str, Any],
) -> None:
    N = packed.shape[1]
    K = packed.shape[2] * 2
    EM = sorted_ids.shape[0]
    scale = e4m3_kernel_view(scale)
    grid = lambda META: (  # noqa: E731
        triton.cdiv(EM, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    _prefill_nvfp4_moe_kernel[grid](
        a, packed, scale, glob, c, topk_weights_flat, sorted_ids, expert_ids,
        num_tokens_post_padded,
        _e2m1_lut(a.device.index),
        N, K, EM, num_valid_tokens,
        a.stride(0), a.stride(1),
        packed.stride(0), packed.stride(1), packed.stride(2),
        scale.stride(0), scale.stride(1), scale.stride(2),
        glob.stride(0), glob.stride(1),
        c.stride(1), c.stride(2),
        topk_weights_flat.stride(0),
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=kernel_top_k,
        compute_type=_tl_dtype(c.dtype),
        **cfg,
    )


def fused_experts_nvfp4(
    hidden_states: torch.Tensor,
    gate_up_packed: torch.Tensor,
    gate_up_scale: torch.Tensor,
    gate_up_global: torch.Tensor,
    down_packed: torch.Tensor,
    down_scale: torch.Tensor,
    down_global: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    act_alpha: float = 1.702,
    act_limit: float = 7.0,
) -> torch.Tensor:
    """Prefill inline-NVFP4 MoE. ``topk_ids`` index rows of the bank tensors in
    ``[0, num_experts)``: full-layer banks with position == expert id (the
    materialized ``[:E]`` slot view or the overlap double buffer), raw ids."""
    M, H = hidden_states.shape
    top_k = topk_ids.shape[1]
    two_i = gate_up_packed.shape[1]
    inter = two_i // 2
    dev, dt = hidden_states.device, hidden_states.dtype
    cfg = _prefill_config(M)

    sorted_ids, expert_ids, ntpp = moe_align_block_size(topk_ids, cfg["BLOCK_SIZE_M"], num_experts)
    tw = topk_weights.reshape(-1).contiguous()
    num_valid = topk_ids.numel()

    ic1 = torch.empty((M, top_k, two_i), device=dev, dtype=dt)
    _prefill_gemm(
        hidden_states, gate_up_packed, gate_up_scale, gate_up_global, ic1,
        tw, sorted_ids, expert_ids, ntpp, num_valid, top_k,
        apply_router_weight_on_input, cfg,
    )
    ic2 = torch.empty((M * top_k, inter), device=dev, dtype=dt)
    _run_act(activation, ic1.view(-1, two_i), ic2, act_alpha, act_limit)
    ic3 = torch.empty((M, top_k, H), device=dev, dtype=dt)
    _prefill_gemm(
        ic2, down_packed, down_scale, down_global, ic3,
        tw, sorted_ids, expert_ids, ntpp, num_valid, 1,
        not apply_router_weight_on_input, cfg,
    )
    out = torch.empty_like(hidden_states)
    moe_sum_reduce_triton(ic3, out)
    return out


__all__ = [
    "fused_experts_decode_nvfp4_marlin",
    "fused_experts_decode_nvfp4_serial",
    "fused_experts_nvfp4",
]
