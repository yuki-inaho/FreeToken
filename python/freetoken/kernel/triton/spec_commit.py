"""Speculative-verify state commit: copy each GDN layer's accepted per-token recurrent state
into the pool slot in one launch.

The eager form was ``gdn_states[:, 0].index_select(1, n-1)`` (113 MB read + 113 MB write at
Qwen3.8's 36 layers x [48, 128, 128] fp32) followed by ``recurrent_states.index_copy_`` (another
113 MB read + write): ~275 us per step. This kernel reads the accepted row once and writes the
slot once (~150 us), with ``n`` and the slot taken from device scalars (graph-safe).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _commit_gdn_states_kernel(
    src_ptr,        # [L, 1, K1, S] states (S = hv * k * v contiguous per token)
    dst_ptr,        # [L, SLOTS, S]
    n_ptr,          # [1] int: number of accepted input rows (>= 1); the row is n - 1
    slot_ptr,       # [1] int: destination slot
    S,
    stride_src_l, stride_src_t, stride_dst_l, stride_dst_slot,
    BLOCK: tl.constexpr,
):
    layer = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < S
    row = tl.load(n_ptr).to(tl.int64) - 1
    slot = tl.load(slot_ptr).to(tl.int64)
    src = src_ptr + layer.to(tl.int64) * stride_src_l + row * stride_src_t + offs
    dst = dst_ptr + layer.to(tl.int64) * stride_dst_l + slot * stride_dst_slot + offs
    x = tl.load(src, mask=mask, other=0.0)
    tl.store(dst, x.to(dst_ptr.dtype.element_ty), mask=mask)


def commit_gdn_states(
    gdn_states: torch.Tensor,        # [L, 1, K1, hv, k, v]
    recurrent_states: torch.Tensor,  # [L, SLOTS, hv, k, v]
    num_inputs: torch.Tensor,        # [1] device int
    slot: torch.Tensor,              # [1] device int
) -> None:
    L, one, k1 = gdn_states.shape[:3]
    assert one == 1 and recurrent_states.shape[0] == L
    S = gdn_states[0, 0, 0].numel()
    assert recurrent_states[0, 0].numel() == S
    assert gdn_states.stride(-1) == 1 and recurrent_states.stride(-1) == 1
    assert gdn_states[0, 0, 0].is_contiguous() and recurrent_states[0, 0].is_contiguous()
    BLOCK = 4096
    _commit_gdn_states_kernel[(L, triton.cdiv(S, BLOCK))](
        gdn_states, recurrent_states, num_inputs, slot, S,
        gdn_states.stride(0), gdn_states.stride(2),
        recurrent_states.stride(0), recurrent_states.stride(1),
        BLOCK=BLOCK, num_warps=4,
    )


__all__ = ["commit_gdn_states"]
