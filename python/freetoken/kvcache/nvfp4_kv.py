"""NVFP4 numerics for the QSA K/V cache -- quality probe before any kernel work.

The KV slab is the dominant VRAM cost at long context: 1.75 GiB at 128K in fp8, and halving it
again buys both expert-cache slots and KV read bandwidth (measured: bf16 -> fp8 was +19.6%
decode at 96K depth). NVFP4 is ~0.55x fp8 per token.

The performance case is good; the risk is **quality**, and it lands exactly where this would be
used -- 4-bit KV degrades long-context retrieval first. So this module exists to answer the
quality question *without* writing a storage format or an attention kernel: it applies the
NVFP4 quantize/dequantize round trip to K and V at store time and lets them land in the normal
slab. Numerically identical to a real NVFP4 KV cache; no memory saving, no kernel change. If the
needle and reasoning gates fail here, the kernel project is off.

Format matches the expert weights exactly (see kernel/triton/nvfp4_dequant.py), so a real
implementation can reuse the same LUT and dequant math:

    value = E2M1[code] * block_scale_e4m3 * row_global_fp16

with 16-element blocks along ``head_dim`` and one global scale per (token, head) -- the same
granularity the fp8 slab already carries in ``_kv_scale``, so the storage tier's shape is
precedented.
"""

from __future__ import annotations

import os

import torch

BLOCK = 16          # elements per block scale, as in the expert banks
E2M1_MAX = 6.0      # largest magnitude the 4-bit grid represents
E4M3_MAX = 448.0    # largest magnitude an e4m3 block scale represents

# The signed E2M1 grid (magnitudes; sign is carried separately).
_E2M1_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def fake_quant_enabled() -> bool:
    """``FREETOKEN_KV_FAKE_NVFP4=1``: round K/V through NVFP4 at store time.

    Pair it with ``--kv-cache-dtype bf16`` to measure NVFP4 error alone; with an fp8 slab the
    values are quantized twice and the result is not what a real NVFP4 cache would produce.
    """
    return os.environ.get("FREETOKEN_KV_FAKE_NVFP4", "0").strip().lower() in ("1", "true", "yes")


def _round_to_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Round to the nearest representable E2M1 magnitude, preserving sign.

    Written as a chain of scalar comparisons rather than a lookup table on purpose: this runs
    inside the captured decode graph, and materialising the grid with ``torch.tensor`` is a
    host->device copy, which capture rejects ("Cannot copy between CPU and CUDA tensors during
    CUDA graph capture"). Every operand here is a Python float, so nothing touches the host.
    """
    m = x.abs()
    # midpoints of the 8-point grid 0, 0.5, 1, 1.5, 2, 3, 4, 6
    q = torch.where(m < 0.25, 0.0,
        torch.where(m < 0.75, 0.5,
        torch.where(m < 1.25, 1.0,
        torch.where(m < 1.75, 1.5,
        torch.where(m < 2.50, 2.0,
        torch.where(m < 3.50, 3.0,
        torch.where(m < 5.00, 4.0, 6.0)))))))
    return torch.sign(x) * q


def nvfp4_fake_quant(x: torch.Tensor, block: int = BLOCK,
                     global_dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """Quantize/dequantize ``x`` ``[..., head_dim]`` through NVFP4 and return the result.

    One global scale per row (the last dim is one token-head's head_dim vector) and one e4m3
    scale per ``block`` elements, exactly as the dequant kernel reads them.

    ``global_dtype`` defaults to fp16, which is what the quality probe was gated with. The real
    store keeps the global in fp32 (the tier the fp8 pool already has), which is strictly more
    accurate -- fp16 underflows to zero for near-zero rows.
    """
    orig_dtype = x.dtype
    xf = x.float()
    *lead, d = xf.shape
    assert d % block == 0, f"head_dim {d} is not a multiple of the {block}-element block"
    xb = xf.reshape(*lead, d // block, block)

    # Row global, stored fp16 as the dequant kernel reads it.
    row_absmax = xf.abs().amax(dim=-1, keepdim=True)
    g = (row_absmax / (E2M1_MAX * E4M3_MAX)).to(global_dtype).float()
    gb = g.unsqueeze(-1)

    # An all-zero row (a never-written slot, a padded row) drives the global to zero, and the
    # block scale would then be 0/0 = NaN -- which would propagate through attention. The fp8
    # path guarantees such slots read as a finite zero, so this must too: divide by a safe 1.0
    # and let the zero denominator below carry the row to an exact zero.
    safe_gb = torch.where(gb > 0, gb, torch.ones_like(gb))

    # Per-block scale, stored e4m3.
    blk_absmax = xb.abs().amax(dim=-1, keepdim=True)
    s = ((blk_absmax / E2M1_MAX) / safe_gb).clamp(0.0, E4M3_MAX)
    s = s.to(torch.float8_e4m3fn).float()

    denom = s * gb
    safe_denom = torch.where(denom > 0, denom, torch.ones_like(denom))
    q = _round_to_e2m1((xb / safe_denom).clamp(-E2M1_MAX, E2M1_MAX))
    return (q * denom).reshape(*lead, d).to(orig_dtype)


def fake_quant_kv(k: torch.Tensor, v: torch.Tensor, head_dim: int
                  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Round ``[T, H*head_dim]`` K/V rows through NVFP4, per (token, head)."""
    def one(t: torch.Tensor) -> torch.Tensor:
        T = t.shape[0]
        return nvfp4_fake_quant(t.view(T, -1, head_dim)).view(T, -1)

    return one(k), one(v)


def dequant_packed(codes: torch.Tensor, blk: torch.Tensor, glb: torch.Tensor,
                   head_dim: int) -> torch.Tensor:
    """Reference dequant of the packed slab: ``[.., head_dim//2]`` uint8 codes (low nibble
    first), ``[.., head_dim//16]`` e4m3 block scales, ``[..]`` fp32 global -> float32.

    Mirrors what the attend kernel must do in registers; the tests diff the two.
    """
    lo = (codes & 0xF).to(torch.int32)
    hi = ((codes >> 4) & 0xF).to(torch.int32)
    idx = torch.stack((lo, hi), dim=-1).reshape(*codes.shape[:-1], head_dim)

    e = (idx >> 1) & 3
    m = (idx & 1).float()
    # E2M1: exponent 0 is subnormal (0, 0.5); otherwise (1 + m/2) * 2^(e-1)
    mag = torch.where(e == 0, 0.5 * m, (1.0 + 0.5 * m) * torch.pow(2.0, (e - 1).float()))
    sign = torch.where((idx & 8) != 0, -1.0, 1.0)
    vals = sign * mag

    scale = blk.float().repeat_interleave(BLOCK, dim=-1)
    return vals * scale * glb.float().unsqueeze(-1)
