"""Fused MoE-block finalize for decode: routed sum-reduce (+ the hybrid CPU partial), the
shared-expert gate mul-add and the hyper-connection combine in one launch.

Replaces ``moe_sum_reduce`` -> ``shared_gate_mul_add`` -> ``hc_combine`` (three graph nodes
per MoE layer, each ~1-2 us of pure launch floor at decode sizes) with a single kernel that
also skips the two intermediate bf16 roundings:

    y[t]        = sum_k ic3[t, k] (+ addend[t]) + gate[t] * shared[t]        (fp32)
    R_out[t, h] = R[t, h] + y[t] * 2 * sigmoid(inject[t, h] / HC)             (per stream h)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import triton
import triton.language as tl


@triton.jit
def _moe_finalize_combine_kernel(
    ic3_ptr, stride_i0, stride_i1,
    ids_ptr,
    addend_ptr, stride_ad,
    shared_ptr, stride_sh, gate_ptr,
    res_ptr, stride_res,
    inj_ptr, stride_inj,
    out_ptr, stride_out,
    H, TOPK,
    HC: tl.constexpr, BLOCK: tl.constexpr,
    SKIP_NEG: tl.constexpr, HAS_ADDEND: tl.constexpr, HAS_SHARED: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < H
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    base = ic3_ptr + row * stride_i0 + offs
    for k in range(TOPK):
        if SKIP_NEG:
            # routes computed elsewhere (hybrid: on the CPU) left their rows unwritten
            live = tl.load(ids_ptr + row * TOPK + k) >= 0
            tmp = tl.load(base + k * stride_i1, mask=mask & live, other=0.0)
        else:
            tmp = tl.load(base + k * stride_i1, mask=mask, other=0.0)
        acc += tmp.to(tl.float32)
    if HAS_ADDEND:
        acc += tl.load(addend_ptr + row * stride_ad + offs, mask=mask, other=0.0).to(tl.float32)
    if HAS_SHARED:
        g = tl.load(gate_ptr + row).to(tl.float32)
        acc += g * tl.load(shared_ptr + row * stride_sh + offs, mask=mask, other=0.0).to(tl.float32)
    for h in tl.static_range(HC):
        inj = tl.load(inj_ptr + row * stride_inj + h).to(tl.float32)
        inj = 2.0 * tl.sigmoid(inj / HC)
        res = tl.load(res_ptr + row * stride_res + h * H + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + row * stride_out + h * H + offs, (res + acc * inj).to(out_ptr.dtype.element_ty), mask=mask)


def moe_finalize_combine(
    ic3: torch.Tensor,
    route_ids: torch.Tensor | None,
    addend: torch.Tensor | None,
    shared: torch.Tensor | None,
    gate: torch.Tensor | None,
    residual: torch.Tensor,
    inject: torch.Tensor,
    hc: int,
) -> torch.Tensor:
    """``ic3`` [M, topk, H] per-route expert outputs (router weights already applied);
    ``route_ids`` [M, topk] int32 or None (negative = skip the unwritten row); ``addend``
    [M, H] or None; ``shared`` [M, H] + ``gate`` [M] fp32 or both None; ``residual``
    [M, hc*H]; ``inject`` [M, hc] -> new residual [M, hc*H] (``residual``'s dtype)."""
    M, topk, H = ic3.shape
    assert ic3.stride(2) == 1
    assert residual.shape == (M, hc * H) and residual.stride(1) == 1
    assert inject.shape[0] == M and inject.shape[1] >= hc and inject.stride(1) == 1
    if route_ids is not None:
        assert route_ids.shape == (M, topk) and route_ids.is_contiguous()
    if addend is not None:
        assert addend.shape == (M, H) and addend.stride(1) == 1
    if shared is not None:
        assert shared.shape == (M, H) and shared.stride(1) == 1 and gate is not None
        assert gate.shape == (M,) and gate.dtype == torch.float32
    out = torch.empty_like(residual)
    BLOCK = 512
    dummy = ic3
    _moe_finalize_combine_kernel[(M, triton.cdiv(H, BLOCK))](
        ic3, ic3.stride(0), ic3.stride(1),
        dummy if route_ids is None else route_ids,
        dummy if addend is None else addend, 0 if addend is None else addend.stride(0),
        dummy if shared is None else shared, 0 if shared is None else shared.stride(0),
        dummy if gate is None else gate,
        residual, residual.stride(0),
        inject, inject.stride(0),
        out, out.stride(0),
        H, topk,
        HC=hc, BLOCK=BLOCK,
        SKIP_NEG=route_ids is not None, HAS_ADDEND=addend is not None, HAS_SHARED=shared is not None,
        num_warps=4,
    )
    return out


@dataclass
class MoeCombineEpilogue:
    """Handed by a model's MoE block to ``experts.forward(..., epilogue=)``: a decode path
    that supports it finishes the block with :func:`moe_finalize_combine` and sets
    ``result`` (the new residual); a path that does not leaves ``result`` None and returns
    the routed output as usual, so the caller applies the separate ops."""

    shared: torch.Tensor | None
    gate: torch.Tensor | None          # fp32 [M]; may be produced later from gate_weight
    residual: torch.Tensor
    inject: torch.Tensor
    hc: int
    result: torch.Tensor | None = None
    gate_weight: torch.Tensor | None = None   # [H] shared-gate weight, for a fused producer
    shared_fn: Callable[[], torch.Tensor] | None = None

    @property
    def has_shared(self) -> bool:
        return self.shared is not None or self.shared_fn is not None

    def materialize_shared(self) -> torch.Tensor | None:
        """Enqueue the deferred shared expert once, while its input is still intact.

        Hybrid decode calls this after CPU submission, before the routed expert kernels.
        The callback runs during graph capture; its kernels then replay with the graph.
        """
        if self.shared_fn is not None:
            assert self.shared is None, "shared expert was supplied twice"
            self.shared = self.shared_fn()
            self.shared_fn = None
        return self.shared

    def apply(self, ic3: torch.Tensor, route_ids: torch.Tensor | None, addend: torch.Tensor | None) -> torch.Tensor:
        assert self.shared_fn is None, "shared expert must run before routed kernels"
        assert self.shared is None or self.gate is not None, "shared gate not produced"
        self.result = moe_finalize_combine(
            ic3, route_ids, addend, self.shared, self.gate, self.residual, self.inject, self.hc
        )
        return self.result


__all__ = ["MoeCombineEpilogue", "moe_finalize_combine"]
