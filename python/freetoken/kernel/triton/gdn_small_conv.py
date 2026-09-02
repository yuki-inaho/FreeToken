"""Depthwise causal conv for a handful of tokens (the speculative verify: T = k+1 <= 4).

The generic varlen conv (sgl_kernel / the tiled Triton fallback) is a prefill kernel: at
T=3 it costs ~22 us plus a channels-first transpose copy on the way in and three
non-contiguous reshape copies (q / k / v) on the way out -- ~30 us per GDN layer per step,
36 layers per step. This kernel does the T<=4 case in one launch that reads the request's
conv state window + the raw inputs once and writes q, k and v as three CONTIGUOUS [T, *]
tensors (so the fused recurrent kernel gets unit-stride views without copies), then
refreshes the conv state in place with the last (kernel-1) inputs -- the same state
convention as the varlen kernel (oldest first).

Semantics (matched to sgl_kernel's causal_conv1d_fwd with has_initial_state=True, silu):
    ext[c] = state[slot, c, 0:K-1] ++ x[0:T, c]
    out[t, c] = silu(sum_j w[c, j] * ext[c][t + j]),   silu(v) = v / (1 + exp(-v))
    state[slot, c, :] = ext[c][T : T + K - 1]
fp32 math, bf16 in/out (not bit-identical to the CUDA op's fast-math expf; the GDN verify
test holds both paths to the sequential reference at 2e-2).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _small_conv_kernel(
    x_ptr,        # [T, C] bf16 (row-major)
    w_ptr,        # [C, K] (any float dtype)
    state_ptr,    # [num_slots, C, K-1]
    slot_ptr,     # [1] int (the request's state slot)
    q_ptr, k_ptr, v_ptr,   # [T, KD], [T, KD], [T, VD] bf16 out (contiguous)
    C, KD, VD,
    stride_x_t, stride_state_slot, stride_state_c,
    T: tl.constexpr, K: tl.constexpr, BLOCK_C: tl.constexpr,
    # STASH: also record the speculative-verify rollback inputs -- the PRE-update state
    # window -> pre_ptr [C, K-1] and the raw inputs -> xin_ptr [C, XIN_STRIDE] (channels
    # first, columns 0..T-1) -- in place of SpecStateScratch.stash_gdn's three launches.
    pre_ptr=None, xin_ptr=None, xin_stride=0, STASH: tl.constexpr = False,
):
    pid = tl.program_id(0)
    offs_c = pid * BLOCK_C + tl.arange(0, BLOCK_C)
    cmask = offs_c < C
    slot = tl.load(slot_ptr).to(tl.int64)
    sbase = state_ptr + slot * stride_state_slot + offs_c * stride_state_c
    # ext window: K-1 state taps followed by the T new inputs (fp32)
    s0 = tl.load(sbase + 0, mask=cmask, other=0.0).to(tl.float32)
    s1 = tl.load(sbase + 1, mask=cmask, other=0.0).to(tl.float32)
    s2 = tl.load(sbase + 2, mask=cmask, other=0.0).to(tl.float32)
    x0 = tl.load(x_ptr + 0 * stride_x_t + offs_c, mask=cmask & (T > 0), other=0.0).to(tl.float32)
    x1 = tl.load(x_ptr + 1 * stride_x_t + offs_c, mask=cmask & (T > 1), other=0.0).to(tl.float32)
    x2 = tl.load(x_ptr + 2 * stride_x_t + offs_c, mask=cmask & (T > 2), other=0.0).to(tl.float32)
    x3 = tl.load(x_ptr + 3 * stride_x_t + offs_c, mask=cmask & (T > 3), other=0.0).to(tl.float32)
    if STASH:
        pre_dt: tl.constexpr = pre_ptr.dtype.element_ty
        tl.store(pre_ptr + offs_c * (K - 1) + 0, s0.to(pre_dt), mask=cmask)
        tl.store(pre_ptr + offs_c * (K - 1) + 1, s1.to(pre_dt), mask=cmask)
        tl.store(pre_ptr + offs_c * (K - 1) + 2, s2.to(pre_dt), mask=cmask)
        xin_dt: tl.constexpr = xin_ptr.dtype.element_ty
        tl.store(xin_ptr + offs_c * xin_stride + 0, x0.to(xin_dt), mask=cmask & (T > 0))
        if T > 1:
            tl.store(xin_ptr + offs_c * xin_stride + 1, x1.to(xin_dt), mask=cmask)
        if T > 2:
            tl.store(xin_ptr + offs_c * xin_stride + 2, x2.to(xin_dt), mask=cmask)
        if T > 3:
            tl.store(xin_ptr + offs_c * xin_stride + 3, x3.to(xin_dt), mask=cmask)
    w0 = tl.load(w_ptr + offs_c * K + 0, mask=cmask, other=0.0).to(tl.float32)
    w1 = tl.load(w_ptr + offs_c * K + 1, mask=cmask, other=0.0).to(tl.float32)
    w2 = tl.load(w_ptr + offs_c * K + 2, mask=cmask, other=0.0).to(tl.float32)
    w3 = tl.load(w_ptr + offs_c * K + 3, mask=cmask, other=0.0).to(tl.float32)
    # ext = [s0, s1, s2, x0, x1, x2, x3]; out[t] = w0*ext[t] + w1*ext[t+1] + w2*ext[t+2] + w3*ext[t+3]
    o0 = w0 * s0 + w1 * s1 + w2 * s2 + w3 * x0
    o1 = w0 * s1 + w1 * s2 + w2 * x0 + w3 * x1
    o2 = w0 * s2 + w1 * x0 + w2 * x1 + w3 * x2
    o3 = w0 * x0 + w1 * x1 + w2 * x2 + w3 * x3
    # silu, then scatter each channel to its q / k / v destination
    is_q = offs_c < KD
    is_k = (offs_c >= KD) & (offs_c < 2 * KD)
    is_v = offs_c >= 2 * KD
    for t in tl.static_range(T):
        if t == 0:
            o = o0
        elif t == 1:
            o = o1
        elif t == 2:
            o = o2
        else:
            o = o3
        y = (o / (1.0 + tl.exp(-o))).to(tl.bfloat16)
        tl.store(q_ptr + t * KD + offs_c, y, mask=cmask & is_q)
        tl.store(k_ptr + t * KD + (offs_c - KD), y, mask=cmask & is_k)
        tl.store(v_ptr + t * VD + (offs_c - 2 * KD), y, mask=cmask & is_v)
    # new state = last K-1 entries of ext
    if T == 1:
        n0 = s1
        n1 = s2
        n2 = x0
    elif T == 2:
        n0 = s2
        n1 = x0
        n2 = x1
    elif T == 3:
        n0 = x0
        n1 = x1
        n2 = x2
    else:
        n0 = x1
        n1 = x2
        n2 = x3
    tl.store(sbase + 0, n0.to(state_ptr.dtype.element_ty), mask=cmask)
    tl.store(sbase + 1, n1.to(state_ptr.dtype.element_ty), mask=cmask)
    tl.store(sbase + 2, n2.to(state_ptr.dtype.element_ty), mask=cmask)


def causal_conv1d_small_t(
    conv_in: torch.Tensor,     # [T, C] bf16, T <= 4
    conv_state: torch.Tensor,  # [num_slots, C, K-1], updated in place
    weight: torch.Tensor,      # [C, K]
    slot: torch.Tensor,        # [1] int32/int64 device
    key_dim: int,
    value_dim: int,
    stash: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """silu(causal conv) of T<=4 tokens for one request, split into contiguous q/k/v
    ([T, key_dim], [T, key_dim], [T, value_dim]); refreshes ``conv_state[slot]``.
    ``stash`` = (pre [C, K-1], xin [C, >=T]) additionally receives the pre-update state
    window and the raw inputs transposed (the speculative verify's rollback record)."""
    T, C = conv_in.shape
    K = weight.shape[1]
    assert 1 <= T <= 4 and K == 4 and conv_state.shape[-1] == 3, (T, K, conv_state.shape)
    assert C == 2 * key_dim + value_dim and conv_in.stride(1) == 1 and weight.is_contiguous()
    assert conv_state.stride(2) == 1
    if stash is not None:
        pre, xin = stash
        assert pre.shape == (C, K - 1) and pre.is_contiguous(), pre.shape
        assert xin.shape[0] == C and xin.shape[1] >= T and xin.stride(1) == 1, xin.shape
    else:
        pre = xin = conv_in
    q = torch.empty((T, key_dim), dtype=torch.bfloat16, device=conv_in.device)
    k = torch.empty((T, key_dim), dtype=torch.bfloat16, device=conv_in.device)
    v = torch.empty((T, value_dim), dtype=torch.bfloat16, device=conv_in.device)
    BLOCK_C = 1024
    _small_conv_kernel[(triton.cdiv(C, BLOCK_C),)](
        conv_in, weight, conv_state, slot, q, k, v,
        C, key_dim, value_dim,
        conv_in.stride(0), conv_state.stride(0), conv_state.stride(1),
        T=T, K=K, BLOCK_C=BLOCK_C, num_warps=4,
        pre_ptr=pre, xin_ptr=xin, xin_stride=0 if stash is None else xin.stride(0),
        STASH=stash is not None,
    )
    return q, k, v


__all__ = ["causal_conv1d_small_t"]
