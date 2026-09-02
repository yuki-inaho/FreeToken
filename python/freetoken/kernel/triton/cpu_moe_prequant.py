"""GPU-side input quantization for the CPU MoE executor's NVFP4 W4A8 path.

The C++ executor's ``quant_i8_pg16`` turns a bf16 activation row into per-16-group int8 codes
(``[even(8) | odd(8)]`` per group, matching the e2m1 nibble order the VNNI dot consumes) plus
one fp32 scale per group. It ran single-threaded on the coordinator thread, on the decode
critical path, once per MoE layer per step (~30 us at bs=3, H=2560). This Triton kernel is
the bit-exact GPU twin: it runs before the D2H copy, so the CPU receives the int8 codes and
scales directly (fewer bytes over PCIe too: 1 B + 4 B/16 per element instead of 2 B).

Bit-exactness contract (tests/moe/test_cpu_moe_prequant.py): fp32 math throughout, IEEE
round-to-nearest division for ``amax / 127`` and ``1 / s`` (``div_rn``; Triton's default
fp32 ``/`` is the approximate ``div.full``), ``lround`` == round-half-away-from-zero
(libdevice ``round``), clamp to [-127, 127].
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _act_quant_i8_pg16_kernel(
    x_ptr,        # [T, H] bf16 (row-major, contiguous)
    q_ptr,        # [T, H] int8 out
    s_ptr,        # [T, H // 16] fp32 out
    H: tl.constexpr,
    BLOCK_G: tl.constexpr,   # groups of 16 per program
    gw_ptr=None,             # [H] shared-expert gate weight (GATE)
    gate_ptr=None,           # [T] fp32 out: sigmoid(x . gw) (GATE)
    GATE: tl.constexpr = False,
):
    row = tl.program_id(0)
    g0 = tl.program_id(1) * BLOCK_G
    groups = g0 + tl.arange(0, BLOCK_G)
    gmask = groups < H // 16
    j = tl.arange(0, 8)
    # even = x[16g + 2j], odd = x[16g + 2j + 1]  (the deinterleave)
    base = row * H + groups[:, None] * 16 + 2 * j[None, :]
    m2 = gmask[:, None] & (j[None, :] < 8)
    xe = tl.load(x_ptr + base, mask=m2, other=0.0).to(tl.float32)
    xo = tl.load(x_ptr + base + 1, mask=m2, other=0.0).to(tl.float32)
    if GATE:
        # the shared-expert gate reads the same row: fold sigmoid(x . w) in (one program
        # spans the whole row when GATE, so the reduction stays in-program / deterministic)
        gbase = groups[:, None] * 16 + 2 * j[None, :]
        we = tl.load(gw_ptr + gbase, mask=m2, other=0.0).to(tl.float32)
        wo = tl.load(gw_ptr + gbase + 1, mask=m2, other=0.0).to(tl.float32)
        dot = tl.sum(tl.sum(xe * we + xo * wo, axis=1), axis=0)
        tl.store(gate_ptr + row, tl.sigmoid(dot))
    amax = tl.maximum(tl.max(tl.abs(xe), axis=1), tl.max(tl.abs(xo), axis=1))
    s = tl.where(amax > 0.0, libdevice.div_rn(amax, 127.0), 1.0)
    inv = libdevice.div_rn(1.0, s)
    qe = libdevice.round(xe * inv[:, None])
    qo = libdevice.round(xo * inv[:, None])
    qe = tl.minimum(tl.maximum(qe, -127.0), 127.0).to(tl.int8)
    qo = tl.minimum(tl.maximum(qo, -127.0), 127.0).to(tl.int8)
    out_base = row * H + groups[:, None] * 16 + j[None, :]
    tl.store(q_ptr + out_base, qe, mask=m2)
    tl.store(q_ptr + out_base + 8, qo, mask=m2)
    tl.store(s_ptr + row * (H // 16) + groups, s, mask=gmask)


def nvfp4_act_quant_i8_pg16(
    x: torch.Tensor, q: torch.Tensor, s: torch.Tensor,
    gate_weight: torch.Tensor | None = None, gate_out: torch.Tensor | None = None,
) -> None:
    """``x`` [T, H] bf16 -> ``q`` [T, H] int8 (per-16 group ``even(8)|odd(8)`` layout) and
    ``s`` [T, H//16] fp32 group scales, bit-exact with the executor's ``quant_i8_pg16``.
    With ``gate_weight`` [H] / ``gate_out`` [T] fp32 the shared-expert gate
    ``sigmoid(x @ gate_weight)`` is produced in the same launch (row-wide programs).
    Fixed shapes, no host sync: CUDA-graph safe."""
    T, H = x.shape
    assert H % 16 == 0 and x.is_contiguous() and q.shape == (T, H) and s.shape == (T, H // 16)
    assert x.dtype == torch.bfloat16 and q.dtype == torch.int8 and s.dtype == torch.float32
    gate = gate_weight is not None
    if gate:
        assert gate_out is not None and gate_out.shape == (T,) and gate_out.dtype == torch.float32
        assert gate_weight.shape == (H,) and gate_weight.is_contiguous()
        BLOCK_G = triton.next_power_of_2(H // 16)   # whole row per program
        grid = (T, 1)
        _act_quant_i8_pg16_kernel[grid](
            x, q, s, H=H, BLOCK_G=BLOCK_G, gw_ptr=gate_weight, gate_ptr=gate_out, GATE=True,
            num_warps=8,
        )
        return
    BLOCK_G = 64
    grid = (T, triton.cdiv(H // 16, BLOCK_G))
    _act_quant_i8_pg16_kernel[grid](x, q, s, H=H, BLOCK_G=BLOCK_G, num_warps=4)


__all__ = ["nvfp4_act_quant_i8_pg16"]
