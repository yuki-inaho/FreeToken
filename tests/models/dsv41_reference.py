"""Pure-torch CPU oracle for the DeepSeek-V4.1-Flash reference implementation.

This is a faithful port of the math in DeepSeek's ``inference/model.py`` + ``inference/kernel.py``
(HF repo ``deepseek-ai/DeepSeek-V4.1-Flash``) for arbitrary small ``ModelArgs``, with every
quantization round trip SIMULATED exactly in fp32:

* ``act_quant``: per-block (32) FP8 e4m3 with a ue8m0 power-of-two scale ``2^ceil(log2(amax/448))``
  (amax floored at 1e-4); ``inplace`` = quant + dequant back to bf16.
* ``fp4_act_quant``: per-block E2M1 with either an e8m0 scale (amax floored at ``6*2^-126``,
  ``2^ceil(log2(amax/6))``) or an E4M3 scale (amax floored at ``6*2^-9``, ``s = e4m3(amax/6)``).
* ``fp8_gemm`` / ``fp4_gemm``: block-scaled matmuls in fp32 with per-K-block scale application.
* ``sparse_attn``: gathered top-k attention (-1 masked) with the attention sink, ported including
  the kernel's 64-wide online-softmax blocking and the bf16 cast of the probabilities before the PV
  product (both switchable through :class:`Numerics`).
* ``hc_split_sinkhorn``: the exact iteration order of the tilelang kernel.

Everything runs in fp32 on CPU. Tensors that the reference materializes in bf16 (every Linear
output, RMSNorm output, RoPE'd tensor, quant round trip, attention output, ``hc_pre``/``hc_post``
output, MoE output, ...) are rounded to bf16 at exactly those points (``Numerics.bf16_activations``),
so a FreeToken port can be compared at tight tolerances. Storage tensors (bf16 / fp8 / e8m0 / packed
e2m1) are kept in the CHECKPOINT formats inside the ``state_dict`` and upcast at construction.

Determinism: nothing here touches torch's global RNG. :func:`make_random_checkpoint` draws every
tensor from a private ``torch.Generator`` seeded with ``seed`` (the same ``(args, seed)`` always
produces bit-identical tensors); ``ReferenceModel`` is deterministic (argmax drafting by default).
CPU BLAS reassociation can still differ between thread counts at the ~1 ulp fp32 level.

Extensions beyond the reference (documented where they live):
* ``forward`` accepts chunked prefill (``start_pos > 0`` with ``seqlen > 1``); the reference only
  supports ``start_pos == 0`` or single-token decode. The extension reproduces the single-shot
  prefill layout so chunked == single prefill up to fp32 reassociation.
* :func:`ced_prefill` is the Causal Encoder-Decoder prefill approximation (encoder layers over all
  tokens, hoisted decoder KV production, decoder replay over the last ``window_size`` tokens).
* ``forward_spec`` takes an explicit ``draft`` switch and a ``sampler`` callable (default argmax).

DEVIATION (reference bug, see ``Indexer.__call__``): an index-key owner always scores against its
own ``k_cache``. The reference rebinds ``shared_attn.index_k`` only when it publishes a latent, so
at decode steps where a ratio-2 owner's group is still filling it silently scores against the last
publisher's (ratio-1) keys, which makes its own decode inconsistent with its prefill.

TIE-BREAKING: the reference's ``topk`` (indexer scores, candidate blocks, gate) leaves ties
unspecified, and exact ties are common in the indexer (``relu`` zeros: with ``h`` index heads a
fraction ``2^-h`` of all scores is exactly 0 -- 25% in the 2-head toy, negligible with the real
32 heads). The oracle breaks every such tie toward the LOWER index (:func:`stable_topk_indices`, a
stable descending sort), which is what makes it self-consistent across prefill / decode / chunk
widths. A port that breaks ties differently will diverge on tied toy scores; compare the selected
indices directly, or use the same rule.

Import from tests as ``from .dsv41_reference import ...`` (inside ``tests/models``) or
``from ..dsv41_reference import ...`` (from a sub-package such as ``tests/models/deepseek_v41``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn.functional as F

# --------------------------------------------------------------------------------------------
# Constants (reference: model.py module globals / kernel.py)
# --------------------------------------------------------------------------------------------

FP8_MAX = 448.0
FP4_MAX = 6.0
FP8_BLOCK_SIZE = 32  # one fp8 scale per 32x32 weight block / 32 activations
FP4_BLOCK_SIZE = 32  # one fp4 scale per 32 elements along K
CMP_FP4_GROUP = 16  # compressed KV: E2M1 with one E4M3 scale per 16 channels
ATTN_SCORE_LOWER_BOUND = -1e30  # sparse_attn_kernel: finite init of the running max

_FP8_MAX_INV = torch.tensor(1.0 / FP8_MAX, dtype=torch.float32)
_FP4_MAX_INV = torch.tensor(1.0 / FP4_MAX, dtype=torch.float32)

E2M1_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
# nibble code -> value, sign in bit 3 (convert.py FP4_TABLE)
FP4_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


# --------------------------------------------------------------------------------------------
# Numerics knobs
# --------------------------------------------------------------------------------------------


@dataclass
class Numerics:
    """Where the oracle rounds like the bf16 reference.

    ``bf16_activations``: round to bf16 at every point where the reference materializes a bf16
    tensor (default True = faithful). False keeps the whole activation stream in fp32 (useful to
    separate quantization effects from bf16 rounding; prefill-vs-decode then agrees to ~1e-5).
    ``attn_probs_bf16``: cast the softmax numerators to bf16 before the PV product, as the
    tilelang ``sparse_attn_kernel`` does (``acc_s_cast``). ``attn_block``: the kernel's top-k
    blocking (64) that decides the running max each probability is rounded against.
    """

    bf16_activations: bool = True
    attn_probs_bf16: bool = True
    attn_block: int = 64

    def bf(self, x: torch.Tensor) -> torch.Tensor:
        if self.bf16_activations:
            return x.to(torch.bfloat16).to(torch.float32)
        return x


def bf16_round(x: torch.Tensor) -> torch.Tensor:
    """Round an fp32 tensor to the nearest bf16 value (returns fp32)."""
    return x.to(torch.bfloat16).to(torch.float32)


# --------------------------------------------------------------------------------------------
# Quantization primitives (kernel.py ports)
# --------------------------------------------------------------------------------------------


def pow2_ceil_log2(x: torch.Tensor) -> torch.Tensor:
    """``2^ceil(log2(x))`` for positive normal fp32 ``x`` via the kernel's IEEE bit trick
    (``fast_round_scale``)."""
    x = x.to(torch.float32).contiguous()
    bits = x.view(torch.int32)
    exp = ((bits >> 23) & 0xFF) - 127
    man = bits & 0x7FFFFF
    e = exp + (man != 0).to(torch.int32)
    return torch.ldexp(torch.ones_like(x), e)


def e8m0_scale(amax: torch.Tensor, max_value: float) -> torch.Tensor:
    """ue8m0 scale ``2^ceil(log2(amax * (1/max_value)))`` computed in fp32 like the kernel."""
    inv = torch.tensor(1.0 / max_value, dtype=torch.float32)
    return pow2_ceil_log2(amax.to(torch.float32) * inv)


def e4m3_round(x: torch.Tensor) -> torch.Tensor:
    """Round-to-nearest-even to FP8 e4m3 (saturating clamp at +-448 first; the kernel clamps
    before every cast). Returns fp32."""
    return x.to(torch.float32).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).to(torch.float32)


def round_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Round-to-nearest-even to E2M1 (values 0, .5, 1, 1.5, 2, 3, 4, 6 with sign), saturating at
    +-6. Ties go to the even code: 0.25->0, 0.75->1, 1.25->1, 1.75->2, 2.5->2, 3.5->4, 5->4."""
    x = x.to(torch.float32)
    a = x.abs().clamp(max=FP4_MAX)
    v = E2M1_VALUES
    lo = torch.bucketize(a, v, right=True) - 1
    hi = (lo + 1).clamp(max=7)
    d_lo = a - v[lo]
    d_hi = v[hi] - a
    pick_hi = (d_hi < d_lo) | ((d_hi == d_lo) & (hi % 2 == 0))
    val = torch.where(pick_hi, v[hi], v[lo])
    return torch.copysign(val, x)


def act_quant(x: torch.Tensor, block_size: int = FP8_BLOCK_SIZE) -> tuple[torch.Tensor, torch.Tensor]:
    """``act_quant(x, block, "ue8m0", e8m0, inplace=False)``: returns the fp8-valued codes
    (fp32 tensor) and the fp32 power-of-two scales ``[..., N // block]``."""
    n = x.size(-1)
    assert n % block_size == 0, (n, block_size)
    xb = x.to(torch.float32).unflatten(-1, (-1, block_size))
    amax = xb.abs().amax(dim=-1).clamp_min(1e-4)
    s = pow2_ceil_log2(amax * _FP8_MAX_INV)
    q = e4m3_round(xb / s.unsqueeze(-1))
    return q.flatten(-2), s


def act_quant_roundtrip(x: torch.Tensor, block_size: int = FP8_BLOCK_SIZE) -> torch.Tensor:
    """``act_quant(..., inplace=True)``: quant + dequant. The product ``fp8 * 2^k`` is exact in
    bf16, so the reference's bf16 store loses nothing; returns fp32."""
    q, s = act_quant(x, block_size)
    y = q.unflatten(-1, (-1, block_size)) * s.unsqueeze(-1)
    return bf16_round(y.flatten(-2))


def fp4_act_quant(
    x: torch.Tensor, block_size: int = FP4_BLOCK_SIZE, scale_dtype: str = "e8m0"
) -> tuple[torch.Tensor, torch.Tensor]:
    """``fp4_act_quant(x, block, inplace=False, scale_dtype)``: e2m1-valued codes (fp32) and
    scales. ``scale_dtype``: ``"e8m0"`` (indexer) or ``"e4m3"`` (compressed KV)."""
    assert scale_dtype in ("e8m0", "e4m3")
    n = x.size(-1)
    assert n % block_size == 0, (n, block_size)
    xb = x.to(torch.float32).unflatten(-1, (-1, block_size))
    amax = xb.abs().amax(dim=-1)
    if scale_dtype == "e4m3":
        amax = amax.clamp_min(6 * 2.0**-9)
        s = e4m3_round(amax / FP4_MAX)
    else:
        amax = amax.clamp_min(6 * 2.0**-126)
        s = pow2_ceil_log2(amax * _FP4_MAX_INV)
    q = round_e2m1((xb / s.unsqueeze(-1)).clamp(-FP4_MAX, FP4_MAX))
    return q.flatten(-2), s


def fp4_act_quant_roundtrip(
    x: torch.Tensor, block_size: int = FP4_BLOCK_SIZE, scale_dtype: str = "e8m0"
) -> torch.Tensor:
    """``fp4_act_quant(..., inplace=True)``: dequantized values, bf16-rounded like the kernel's
    bf16 store (exact: 2 significant bits x <= 4 significant bits fit in bf16)."""
    q, s = fp4_act_quant(x, block_size, scale_dtype)
    y = q.unflatten(-1, (-1, block_size)) * s.unsqueeze(-1)
    return bf16_round(y.flatten(-2))


def fp8_gemm(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    block_size: int = FP8_BLOCK_SIZE,
) -> torch.Tensor:
    """``C[..., N] = A[..., K] @ B[N, K]^T`` with per-block scaling in fp32.

    ``a``: fp8-valued activations (fp32) with ``a_s [..., K // block]``; ``b``: fp8-valued weights
    (fp32) with ``b_s [ceil(N / block), K // block]``. Per K block the raw fp32 product is scaled by
    ``a_s[m, k] * b_s[n // block, k]`` and accumulated (kernel ``C_local_accum``). Returns fp32
    (the caller rounds to bf16 where the reference stores bf16)."""
    k = a.size(-1)
    n = b.size(0)
    assert k % block_size == 0
    nk = k // block_size
    lead = a.shape[:-1]
    am = a.reshape(-1, nk, block_size).to(torch.float32)
    bm = b.reshape(n, nk, block_size).to(torch.float32)
    partial = torch.einsum("mkb,nkb->mkn", am, bm)  # [M, nk, N]
    bs_rows = b_s.to(torch.float32).repeat_interleave(block_size, dim=0)[:n]  # [N, nk]
    scaled = partial * a_s.reshape(-1, nk, 1).to(torch.float32) * bs_rows.t().unsqueeze(0)
    return scaled.sum(dim=1).reshape(*lead, n)


def fp4_gemm(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    act_block_size: int = FP8_BLOCK_SIZE,
) -> torch.Tensor:
    """``C = A_fp8[..., K] @ B_fp4[N, K]^T``: fp8-valued activations with per-``act_block_size``
    scales, e2m1-valued weights (fp32, already unpacked) with per-32 e8m0 scales ``b_s [N, K//32]``.
    Returns fp32."""
    k = a.size(-1)
    n = b.size(0)
    assert act_block_size % FP4_BLOCK_SIZE == 0 and k % act_block_size == 0
    nk = k // FP4_BLOCK_SIZE
    n_sub = act_block_size // FP4_BLOCK_SIZE
    lead = a.shape[:-1]
    am = a.reshape(-1, nk, FP4_BLOCK_SIZE).to(torch.float32)
    bm = b.reshape(n, nk, FP4_BLOCK_SIZE).to(torch.float32)
    partial = torch.einsum("mkb,nkb->mkn", am, bm)
    a_scale = a_s.reshape(-1, k // act_block_size).to(torch.float32).repeat_interleave(n_sub, dim=1)
    scaled = partial * a_scale.unsqueeze(-1) * b_s.to(torch.float32).t().unsqueeze(0)
    return scaled.sum(dim=1).reshape(*lead, n)


def dequant_fp8_block(codes: torch.Tensor, scales: torch.Tensor, block_size: int = FP8_BLOCK_SIZE) -> torch.Tensor:
    """Dequantize an fp8 weight with ``[ceil(N/b), ceil(K/b)]`` block scales to fp32 (exact)."""
    n, k = codes.shape
    w = codes.to(torch.float32)
    s = scales.to(torch.float32)
    s = s.repeat_interleave(block_size, dim=0)[:n].repeat_interleave(block_size, dim=1)[:, :k]
    return w * s


def unpack_e2m1(packed: torch.Tensor) -> torch.Tensor:
    """``[N, K//2]`` int8/uint8 packed e2m1 (low nibble = even k) -> fp32 ``[N, K]``."""
    u = packed.view(torch.uint8)
    lo = FP4_TABLE[(u & 0x0F).long()]
    hi = FP4_TABLE[((u >> 4) & 0x0F).long()]
    return torch.stack([lo, hi], dim=-1).flatten(-2)


def pack_e2m1(values: torch.Tensor) -> torch.Tensor:
    """fp32 ``[N, K]`` of e2m1-valued numbers -> int8 ``[N, K//2]`` (low nibble = even k), the
    checkpoint's ``I8`` view of ``float4_e2m1fn_x2``. Negative zero packs as +0."""
    assert values.size(-1) % 2 == 0
    a = values.abs()
    code = (a.unsqueeze(-1) == E2M1_VALUES).to(torch.int64).argmax(-1)
    assert torch.equal(E2M1_VALUES[code], a), "values are not e2m1-representable"
    code = code + 8 * ((values < 0) & (a > 0)).to(torch.int64)
    pairs = code.unflatten(-1, (-1, 2))
    byte = pairs[..., 0] | (pairs[..., 1] << 4)
    return byte.to(torch.uint8).view(torch.int8)


# --------------------------------------------------------------------------------------------
# Attention / hyper-connection kernels
# --------------------------------------------------------------------------------------------


def sparse_attn(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
    numerics: Numerics | None = None,
) -> torch.Tensor:
    """Port of ``sparse_attn_kernel``: ``q [b, m, h, d]``, ``kv [b, n, d]`` (K == V, MLA latent),
    ``attn_sink [h]`` fp32, ``topk_idxs [b, m, topk]`` int with ``-1`` = nothing. Online softmax
    over ``numerics.attn_block``-wide slices of the top-k row with a finite initial max; the sink
    only enters the denominator. Output bf16-rounded (fp32 tensor)."""
    num = numerics or Numerics()
    b, m, h, d = q.shape
    t = topk_idxs.size(-1)
    block = num.attn_block
    n_blocks = -(-t // block) if t > 0 else 0
    q = q.to(torch.float32)
    kv = kv.to(torch.float32)
    idx = topk_idxs.to(torch.int64)
    if n_blocks * block > t:
        idx = F.pad(idx, (0, n_blocks * block - t), value=-1)
    valid = idx != -1
    batch = torch.arange(b).view(b, 1, 1).expand_as(idx)
    gathered = kv[batch, idx.clamp_min(0)]  # [b, m, T, d]
    gathered = torch.where(valid.unsqueeze(-1), gathered, torch.zeros_like(gathered))
    scores = torch.einsum("bmhd,bmtd->bmht", q, gathered)
    scores = torch.where(valid.unsqueeze(2), scores, torch.full_like(scores, -math.inf))
    scores = scores * float(softmax_scale)

    scores_max = torch.full((b, m, h), ATTN_SCORE_LOWER_BOUND, dtype=torch.float32)
    sum_exp = torch.zeros(b, m, h, dtype=torch.float32)
    acc_o = torch.zeros(b, m, h, d, dtype=torch.float32)
    for blk in range(n_blocks):
        sl = slice(blk * block, (blk + 1) * block)
        s = scores[..., sl]
        prev = scores_max
        scores_max = torch.maximum(prev, s.amax(dim=-1))
        alpha = torch.exp(prev - scores_max)
        p = torch.exp(s - scores_max.unsqueeze(-1))
        sum_exp = sum_exp * alpha + p.sum(dim=-1)
        p_cast = bf16_round(p) if num.attn_probs_bf16 else p
        acc_o = acc_o * alpha.unsqueeze(-1) + torch.einsum("bmht,bmtd->bmhd", p_cast, gathered[:, :, sl])
    sum_exp = sum_exp + torch.exp(attn_sink.to(torch.float32).view(1, 1, h) - scores_max)
    o = acc_o / sum_exp.unsqueeze(-1)
    return num.bf(o)


def hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Port of ``hc_split_sinkhorn_kernel``: ``mixes [..., (2 + hc) * hc]`` fp32 ->
    ``pre [..., hc]``, ``post [..., hc]``, ``comb [..., hc, hc]``; comb = softmax(-1) + eps, then
    column normalization, then ``sinkhorn_iters - 1`` alternating row / column normalizations."""
    hc = hc_mult
    mixes = mixes.to(torch.float32)
    hc_scale = hc_scale.to(torch.float32)
    hc_base = hc_base.to(torch.float32)
    pre = torch.sigmoid(mixes[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(mixes[..., hc : 2 * hc] * hc_scale[1] + hc_base[hc : 2 * hc])
    comb = (mixes[..., 2 * hc :] * hc_scale[2] + hc_base[2 * hc :]).unflatten(-1, (hc, hc))
    row_max = comb.amax(dim=-1, keepdim=True)
    comb = torch.exp(comb - row_max)
    comb = comb / comb.sum(dim=-1, keepdim=True) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb


# --------------------------------------------------------------------------------------------
# RoPE (model.py precompute_freqs_cis / apply_rotary_emb)
# --------------------------------------------------------------------------------------------


def precompute_freqs_cis(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow) -> torch.Tensor:
    """Complex rotary frequencies ``[seqlen, dim // 2]`` with YaRN when ``original_seq_len > 0``."""
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:

        def corrected_dim(rotations):
            return dim * math.log(original_seq_len / (rotations * 2 * math.pi)) / (2 * math.log(base))

        low = max(math.floor(corrected_dim(beta_fast)), 0)
        high = min(math.ceil(corrected_dim(beta_slow)), dim - 1)
        ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / max(high - low, 1e-3)).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    freqs = torch.outer(torch.arange(seqlen), freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def rotate(x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """``apply_rotary_emb`` on the full last dim (adjacent pairs as complex numbers), fp32 in / out
    (the caller rounds). ``x``: ``[b, s, d]`` or ``[b, s, h, d]``; ``freqs_cis [s, d // 2]``."""
    xc = torch.view_as_complex(x.to(torch.float32).contiguous().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    if xc.ndim == 3:
        freqs_cis = freqs_cis.view(1, xc.size(1), xc.size(-1))
    else:
        freqs_cis = freqs_cis.view(1, xc.size(1), 1, xc.size(-1))
    return torch.view_as_real(xc * freqs_cis).flatten(-2)


def rope_tail(x: torch.Tensor, rd: int, freqs_cis: torch.Tensor, num: Numerics, inverse: bool = False) -> torch.Tensor:
    """``apply_rotary_emb(x[..., -rd:], freqs, inverse)``: rotates the last ``rd`` channels and
    stores them back through the reference's bf16 ``copy_``."""
    y = x.clone()
    y[..., -rd:] = num.bf(rotate(x[..., -rd:], freqs_cis, inverse))
    return y


# --------------------------------------------------------------------------------------------
# Index bookkeeping (model.py get_window_topk_idxs / get_dspark_topk_idxs / select_candidate_blocks)
# --------------------------------------------------------------------------------------------


def get_window_topk_idxs(window_size: int, bsz: int, seqlen: int, start_pos: int) -> torch.Tensor:
    """Reference ``get_window_topk_idxs``: prefill rows (``start_pos == 0``) are ascending positions
    then ``-1`` padding; a decode step lists the ring slots oldest first with ``-1`` where the ring
    is still filling."""
    if start_pos == 0:
        end = torch.arange(seqlen).unsqueeze(1)
        idxs = (end - window_size + 1).clamp(0) + torch.arange(min(seqlen, window_size))
        idxs = torch.where(idxs > end, -1, idxs)
    else:
        oldest = start_pos % window_size + 1
        idxs = torch.cat([torch.arange(oldest, window_size), torch.arange(oldest)])
        idxs = torch.where(idxs > start_pos, -1, idxs)
    return idxs.int().unsqueeze(0).expand(bsz, -1, -1).contiguous()


def chunk_window_topk_idxs(window_size: int, bsz: int, seqlen: int, start_pos: int) -> torch.Tensor:
    """ORACLE EXTENSION (chunked prefill, ``start_pos > 0`` and ``seqlen > 1``): rows index the
    concatenation ``[ring (window_size slots) | chunk rows]``. Row ``i`` (position ``p = start_pos
    + i``) lists positions ``max(0, p - window_size + 1) ..`` ascending, ``-1`` past ``p``, width
    ``min(window_size, start_pos + seqlen)``; positions before ``start_pos`` map to ring slot ``pos
    % window_size``, later ones to ``window_size + (pos - start_pos)``. Same row layout as a
    single-shot prefill of the whole prompt, so the block-wise online softmax sees the same order."""
    end_pos = start_pos + seqlen
    width = min(window_size, end_pos)
    p = torch.arange(start_pos, end_pos).unsqueeze(1)
    pos = (p - window_size + 1).clamp(0) + torch.arange(width)
    live = pos <= p
    idx = torch.where(pos < start_pos, pos % window_size, window_size + pos - start_pos)
    idx = torch.where(live, idx, torch.full_like(idx, -1))
    return idx.int().unsqueeze(0).expand(bsz, -1, -1).contiguous()


def get_dspark_topk_idxs(window_size: int, bsz: int, block_size: int, n_filled: int) -> torch.Tensor:
    """Reference ``get_dspark_topk_idxs`` with ``n_filled = min(window_size, start_pos + 1)`` ring
    slots (slot order) followed by every draft row of the block (bidirectional)."""
    matrix = torch.cat([torch.arange(min(window_size, n_filled)), window_size + torch.arange(block_size)])
    return matrix.int().view(1, 1, -1).expand(bsz, block_size, -1).contiguous()


def stable_topk_indices(x: torch.Tensor, k: int) -> torch.Tensor:
    """Indices of the ``k`` largest entries along the last dim, ties broken toward the LOWER index
    (stable descending sort). Deterministic and independent of the row width beyond the ties."""
    return torch.sort(x, dim=-1, descending=True, stable=True).indices[..., :k]


def select_candidate_blocks(
    logits: torch.Tensor,
    compress_lens: torch.Tensor | int,
    topk_blocks: int,
    block_size: int,
) -> torch.Tensor:
    """Level one of the two-level top-k (reference ``select_candidate_blocks``): keep the
    ``topk_blocks`` best-scoring ``block_size``-position blocks per query, the block holding the
    query's newest position pinned, blocks that scored ``-inf`` dropped. Bool mask like ``logits``."""
    width = logits.size(-1)
    scores = F.pad(logits, (0, -width % block_size), value=-math.inf)
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    num_blocks = scores.size(-1)
    last = (compress_lens - 1) // block_size
    scores = scores.masked_fill(torch.arange(num_blocks) == last, math.inf)
    top_idx = stable_topk_indices(scores, min(topk_blocks, num_blocks))
    top_val = scores.gather(-1, top_idx)
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, top_idx, top_val > -math.inf)
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


# --------------------------------------------------------------------------------------------
# ModelArgs
# --------------------------------------------------------------------------------------------


@dataclass
class ModelArgs:
    """Mirror of the reference ``ModelArgs`` (text-only: the vision fields are dropped). Field
    names are the ``inference/config.json`` keys; ``compress_ratios`` has one entry per layer,
    MTP layers included."""

    max_batch_size: int = 4
    max_seq_len: int = 4096
    temperature: float = 1.0
    dtype: str = "fp8"
    expert_dtype: str | None = "fp4"
    vocab_size: int = 129280
    dim: int = 1024
    moe_inter_dim: int = 1024
    n_layers: int = 5
    n_mtp_layers: int = 1
    n_heads: int = 16
    # moe
    n_routed_experts: int = 8
    n_shared_experts: int = 1
    n_activated_experts: int = 2
    score_func: str = "sqrtsoftplus"
    gate_temp: float = 1.0
    norm_topk_prob: bool = True
    route_scale: float = 1.0
    swiglu_limit: float = 0.0
    # attention
    q_lora_rank: int = 256
    head_dim: int = 128
    rope_head_dim: int = 32
    norm_eps: float = 1e-20
    o_groups: int = 8
    o_lora_rank: int = 256
    window_size: int = 128
    compress_ratios: tuple[int, ...] = (0, 2, 2, 1, 1, 0)
    kv_source_layers: tuple[int, ...] = (1, 3)
    index_source_layers: tuple[int, ...] = (1, 3)
    # rope
    compress_rope_theta: float = 40000.0
    original_seq_len: int = 0
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    # indexer
    index_n_heads: int = 16
    index_head_dim: int = 64
    index_topk: int = 64
    candidate_source_layer: int = -1
    candidate_topk_blocks: int = 0
    candidate_block_size: int = 0
    # hyper-connections
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    # engram (hash ids are supplied by the caller; the hashing itself is out of scope here)
    engram_layer_ids: tuple[int, ...] = ()
    engram_num_embeddings: tuple[int, ...] = ()
    engram_max_ngram_size: int = 1
    engram_vocab_size: int = 0
    engram_n_heads: int = 0
    engram_head_dim: int = 0
    engram_pad_id: int = 2
    engram_compressed_vocab_size: int = 0
    # dspark
    dspark_block_size: int = 0
    dspark_noise_token_id: int = 0
    dspark_target_layer_ids: tuple[int, ...] = ()
    dspark_markov_rank: int = 256
    dspark_n_routed_experts: int = 0
    dspark_n_activated_experts: int = 0

    def __post_init__(self):
        self.compress_ratios = tuple(self.compress_ratios)
        self.kv_source_layers = tuple(self.kv_source_layers)
        self.index_source_layers = tuple(self.index_source_layers)
        self.engram_layer_ids = tuple(self.engram_layer_ids)
        self.engram_num_embeddings = tuple(self.engram_num_embeddings)
        self.dspark_target_layer_ids = tuple(self.dspark_target_layer_ids)
        assert self.dtype == "fp8" and self.expert_dtype == "fp4", "the oracle models the shipped fp8/fp4 formats"
        assert len(self.compress_ratios) == self.n_layers + self.n_mtp_layers, (
            len(self.compress_ratios),
            self.n_layers + self.n_mtp_layers,
        )
        assert set(self.kv_source_layers) <= set(self.index_source_layers)
        assert self.n_shared_experts == 1
        for lid in range(self.n_layers + self.n_mtp_layers):
            ratio = self.compress_ratios[lid]
            if lid >= self.n_layers:
                assert ratio == 0, "DSpark stages are window-only"
            elif ratio:
                srcs = [s for s in self.kv_source_layers if s <= lid and self.compress_ratios[s] == ratio]
                assert srcs, f"layer {lid} (ratio {ratio}) has no preceding kv source of the same ratio"
                assert self.max_seq_len % ratio == 0
        if self.candidate_source_layer >= 0:
            assert self.candidate_source_layer in self.index_source_layers
            assert self.candidate_block_size > 0 and self.candidate_topk_blocks > 0
        assert len(self.engram_num_embeddings) == len(self.engram_layer_ids)
        assert self.head_dim % FP8_BLOCK_SIZE == 0 and self.head_dim % CMP_FP4_GROUP == 0
        assert self.index_head_dim % FP4_BLOCK_SIZE == 0
        for t in self.dspark_target_layer_ids:
            assert 0 <= t < self.n_layers

    def get_moe_config(self, layer_id: int) -> tuple[int, int]:
        if layer_id < self.n_layers:
            return self.n_routed_experts, self.n_activated_experts
        return (
            self.dspark_n_routed_experts or self.n_routed_experts,
            self.dspark_n_activated_experts or self.n_activated_experts,
        )

    @property
    def engram_n_hash_cols(self) -> int:
        return (self.engram_max_ngram_size - 1) * self.engram_n_heads

    def mode(self, layer_id: int) -> str:
        """``window`` (ratio 0), ``full`` (kv source), ``reindex`` (index source only), ``reuse``."""
        if self.compress_ratios[layer_id] == 0 or layer_id >= self.n_layers:
            return "window"
        if layer_id in self.kv_source_layers:
            return "full"
        if layer_id in self.index_source_layers:
            return "reindex"
        return "reuse"

    def decoder_start(self) -> int:
        """First decoder layer for :func:`ced_prefill`: the last kv source. Its ratio must be 1 and
        shared by every later backbone layer, and the candidate source (if any) must be it."""
        d = max(self.kv_source_layers)
        assert self.compress_ratios[d] == 1, "CED needs a ratio-1 decoder"
        for lid in range(d, self.n_layers):
            assert self.compress_ratios[lid] == 1, "one ratio across the decoder"
        assert self.candidate_source_layer in (-1, d)
        return d


# --------------------------------------------------------------------------------------------
# Random checkpoint in the STORED formats, keyed by the checkpoint tensor names
# --------------------------------------------------------------------------------------------


def _blocked_scales(g: torch.Generator, w: torch.Tensor, block: int, max_value: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-``block x block`` ue8m0 scales for a dense weight; returns (fp8 codes, e8m0 scales)."""
    n, k = w.shape
    nb, kb = -(-n // block), -(-k // block)
    wp = F.pad(w, (0, kb * block - k, 0, nb * block - n))
    blocks = wp.view(nb, block, kb, block)
    amax = blocks.abs().amax(dim=(1, 3)).clamp_min(1e-12)
    s = pow2_ceil_log2(amax / max_value)
    codes = e4m3_round(blocks / s[:, None, :, None]).view(nb * block, kb * block)[:n, :k]
    return codes.to(torch.float8_e4m3fn), s.to(torch.float8_e8m0fnu)


def _fp8_linear(g: torch.Generator, out_f: int, in_f: int, std: float | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    std = std if std is not None else in_f**-0.5
    w = torch.randn(out_f, in_f, generator=g, dtype=torch.float32) * std
    return _blocked_scales(g, w, FP8_BLOCK_SIZE, FP8_MAX)


def _fp4_linear(g: torch.Generator, out_f: int, in_f: int, std: float | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Packed e2m1 (int8 view, low nibble = even k) + e8m0 per-32 scales ``[out, in // 32]``."""
    assert in_f % FP4_BLOCK_SIZE == 0
    std = std if std is not None else in_f**-0.5
    w = torch.randn(out_f, in_f, generator=g, dtype=torch.float32) * std
    wb = w.unflatten(-1, (-1, FP4_BLOCK_SIZE))
    amax = wb.abs().amax(dim=-1).clamp_min(1e-12)
    s = pow2_ceil_log2(amax / FP4_MAX)
    codes = round_e2m1((wb / s.unsqueeze(-1)).clamp(-FP4_MAX, FP4_MAX)).flatten(-2)
    return pack_e2m1(codes), s.to(torch.float8_e8m0fnu)


def _bf16_randn(g: torch.Generator, shape, std: float) -> torch.Tensor:
    return (torch.randn(*shape, generator=g, dtype=torch.float32) * std).to(torch.bfloat16)


def _bf16_norm(g: torch.Generator, n: int) -> torch.Tensor:
    return (1.0 + 0.1 * torch.randn(n, generator=g, dtype=torch.float32)).to(torch.bfloat16)


def _f32_randn(g: torch.Generator, shape, std: float) -> torch.Tensor:
    return torch.randn(*shape, generator=g, dtype=torch.float32) * std


def make_random_checkpoint(args: ModelArgs, seed: int = 0) -> dict[str, torch.Tensor]:
    """Random weights in the checkpoint's stored formats and names.

    dtypes follow the V4.1 safetensors headers: fp8 e4m3 codes + ``float8_e8m0fnu`` ``[ceil(N/32),
    ceil(K/32)]`` scales for the dense Linear kinds (``wq_a``, ``wq_b``, ``wkv``, ``wo_a``,
    ``wo_b``, shared experts, ``indexer.wq_b``, ``engram.wkv``, ``mtp.0.main_proj``); packed e2m1
    ``int8 [N, K//2]`` + e8m0 ``[N, K//32]`` for routed experts; bf16 for ``embed``, ``head``,
    norms, ``gate.weight``, ``compressor.{wkv,wgate}``, ``indexer.{wk,weights_proj}``,
    ``engram.{q,k}_weight``, ``markov_head.{embed,head}``, ``confidence_head.proj``; fp32 for
    ``hc_*``, ``attn_sink``, ``gate.bias``. The engram table is ``engram.embed.weight`` fp8
    ``[rows, head_dim]`` + ``engram.embed.scale`` e8m0 ``[rows, head_dim // 32]``.
    Deterministic in ``(args, seed)``; never touches the global RNG."""
    g = torch.Generator().manual_seed(seed)
    sd: dict[str, torch.Tensor] = {}
    dim, hc = args.dim, args.hc_mult
    mix_hc = (2 + hc) * hc
    sd["embed.weight"] = _bf16_randn(g, (args.vocab_size, dim), 1.0)
    sd["head.weight"] = _bf16_randn(g, (args.vocab_size, dim), dim**-0.5)
    sd["norm.weight"] = _bf16_norm(g, dim)

    def put_linear(name: str, out_f: int, in_f: int, kind: str, std: float | None = None):
        if kind == "fp8":
            w, s = _fp8_linear(g, out_f, in_f, std)
            sd[name + ".weight"], sd[name + ".scale"] = w, s
        elif kind == "fp4":
            w, s = _fp4_linear(g, out_f, in_f, std)
            sd[name + ".weight"], sd[name + ".scale"] = w, s
        else:
            sd[name + ".weight"] = _bf16_randn(g, (out_f, in_f), std if std is not None else in_f**-0.5)

    def put_block(prefix: str, layer_id: int):
        n_exp, _ = args.get_moe_config(layer_id)
        # attention
        sd[f"{prefix}.attn.attn_sink"] = _f32_randn(g, (args.n_heads,), 1.0)
        sd[f"{prefix}.attn.q_norm.weight"] = _bf16_norm(g, args.q_lora_rank)
        sd[f"{prefix}.attn.kv_norm.weight"] = _bf16_norm(g, args.head_dim)
        put_linear(f"{prefix}.attn.wq_a", args.q_lora_rank, dim, "fp8")
        put_linear(f"{prefix}.attn.wq_b", args.n_heads * args.head_dim, args.q_lora_rank, "fp8")
        put_linear(f"{prefix}.attn.wkv", args.head_dim, dim, "fp8")
        put_linear(
            f"{prefix}.attn.wo_a", args.o_groups * args.o_lora_rank, args.n_heads * args.head_dim // args.o_groups, "fp8"
        )
        put_linear(f"{prefix}.attn.wo_b", dim, args.o_groups * args.o_lora_rank, "fp8")
        ratio = args.compress_ratios[layer_id]
        if layer_id < args.n_layers and layer_id in args.kv_source_layers:
            sd[f"{prefix}.attn.compressor.norm.weight"] = _bf16_norm(g, args.head_dim)
            put_linear(f"{prefix}.attn.compressor.wkv", args.head_dim, dim, "bf16")
            if ratio > 1:
                put_linear(f"{prefix}.attn.compressor.wgate", args.head_dim, dim, "bf16")
            put_linear(f"{prefix}.attn.indexer.wk", args.index_head_dim, args.head_dim, "bf16")
            sd[f"{prefix}.attn.indexer.k_norm.weight"] = _bf16_norm(g, args.index_head_dim)
        if layer_id < args.n_layers and layer_id in args.index_source_layers:
            put_linear(f"{prefix}.attn.indexer.wq_b", args.index_n_heads * args.index_head_dim, args.q_lora_rank, "fp8")
            put_linear(f"{prefix}.attn.indexer.weights_proj", args.index_n_heads, dim, "bf16")
        # block norms + hyper-connections
        sd[f"{prefix}.attn_norm.weight"] = _bf16_norm(g, dim)
        sd[f"{prefix}.ffn_norm.weight"] = _bf16_norm(g, dim)
        for sub in ("attn", "ffn"):
            sd[f"{prefix}.hc_{sub}_fn"] = _f32_randn(g, (mix_hc, hc * dim), (hc * dim) ** -0.5)
            base = torch.zeros(mix_hc, dtype=torch.float32)
            base[2 * hc :] = 2.0 * torch.eye(hc).flatten()
            sd[f"{prefix}.hc_{sub}_base"] = base + _f32_randn(g, (mix_hc,), 0.1)
            sd[f"{prefix}.hc_{sub}_scale"] = 0.5 + torch.rand(3, generator=g, dtype=torch.float32)
        # moe
        sd[f"{prefix}.ffn.gate.weight"] = _bf16_randn(g, (n_exp, dim), dim**-0.5)
        sd[f"{prefix}.ffn.gate.bias"] = _f32_randn(g, (n_exp,), 0.1)
        for e in range(n_exp):
            put_linear(f"{prefix}.ffn.experts.{e}.w1", args.moe_inter_dim, dim, "fp4")
            put_linear(f"{prefix}.ffn.experts.{e}.w2", dim, args.moe_inter_dim, "fp4")
            put_linear(f"{prefix}.ffn.experts.{e}.w3", args.moe_inter_dim, dim, "fp4")
        put_linear(f"{prefix}.ffn.shared_experts.w1", args.moe_inter_dim, dim, "fp8")
        put_linear(f"{prefix}.ffn.shared_experts.w2", dim, args.moe_inter_dim, "fp8")
        put_linear(f"{prefix}.ffn.shared_experts.w3", args.moe_inter_dim, dim, "fp8")

    for lid in range(args.n_layers):
        put_block(f"layers.{lid}", lid)
        if lid in args.engram_layer_ids:
            rows = args.engram_num_embeddings[args.engram_layer_ids.index(lid)]
            hd = args.engram_head_dim
            assert hd % FP8_BLOCK_SIZE == 0
            table = torch.randn(rows, hd, generator=g, dtype=torch.float32)
            tb = table.unflatten(-1, (-1, FP8_BLOCK_SIZE))
            s = pow2_ceil_log2(tb.abs().amax(-1).clamp_min(1e-12) / FP8_MAX)
            codes = e4m3_round(tb / s.unsqueeze(-1)).flatten(-2)
            sd[f"layers.{lid}.engram.embed.weight"] = codes.to(torch.float8_e4m3fn)
            sd[f"layers.{lid}.engram.embed.scale"] = s.to(torch.float8_e8m0fnu)
            put_linear(f"layers.{lid}.engram.wkv", dim * (hc + 1), args.engram_n_hash_cols * hd, "fp8")
            sd[f"layers.{lid}.engram.q_weight"] = (1.0 + 0.1 * torch.randn(hc, dim, generator=g)).to(torch.bfloat16)
            sd[f"layers.{lid}.engram.k_weight"] = (1.0 + 0.1 * torch.randn(hc, dim, generator=g)).to(torch.bfloat16)

    if args.dspark_block_size:
        for st in range(args.n_mtp_layers):
            prefix = f"mtp.{st}"
            put_block(prefix, args.n_layers + st)
            if st == 0:
                put_linear(f"{prefix}.main_proj", dim, dim * len(args.dspark_target_layer_ids), "fp8")
                sd[f"{prefix}.main_norm.weight"] = _bf16_norm(g, dim)
            if st == args.n_mtp_layers - 1:
                sd[f"{prefix}.norm.weight"] = _bf16_norm(g, dim)
                rank = args.dspark_markov_rank
                sd[f"{prefix}.markov_head.embed.weight"] = _bf16_randn(g, (args.vocab_size, rank), 1.0)
                sd[f"{prefix}.markov_head.head.weight"] = _bf16_randn(g, (args.vocab_size, rank), rank**-0.5)
                sd[f"{prefix}.confidence_head.proj.weight"] = _bf16_randn(g, (1, dim + rank), (dim + rank) ** -0.5)
    return sd


# --------------------------------------------------------------------------------------------
# Modules (plain classes over fp32 tensors; no autograd)
# --------------------------------------------------------------------------------------------


class Linear:
    """``kind``: ``fp8`` (act_quant + fp8_gemm -> bf16), ``fp4`` (act_quant + fp4_gemm -> bf16),
    ``bf16`` (``F.linear`` in bf16 -> bf16), ``fp32`` (``F.linear`` in fp32, no rounding)."""

    def __init__(self, sd: dict[str, torch.Tensor], name: str, kind: str, num: Numerics):
        self.kind = kind
        self.num = num
        w = sd[name + ".weight"]
        if kind == "fp8":
            self.w = w.to(torch.float32)
            self.s = sd[name + ".scale"].to(torch.float32)
        elif kind == "fp4":
            self.w = unpack_e2m1(w)
            self.s = sd[name + ".scale"].to(torch.float32)
        else:
            assert w.dtype == torch.bfloat16, (name, w.dtype)
            self.w = w.to(torch.float32)
            self.s = None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "fp8":
            q, s = act_quant(x, FP8_BLOCK_SIZE)
            return self.num.bf(fp8_gemm(q, s, self.w, self.s, FP8_BLOCK_SIZE))
        if self.kind == "fp4":
            q, s = act_quant(x, FP8_BLOCK_SIZE)
            return self.num.bf(fp4_gemm(q, s, self.w, self.s, FP8_BLOCK_SIZE))
        if self.kind == "bf16":
            return self.num.bf(F.linear(x, self.w))
        return F.linear(x.to(torch.float32), self.w)


class RMSNorm:
    def __init__(self, sd, name: str, eps: float, num: Numerics):
        self.w = sd[name + ".weight"].to(torch.float32)
        self.eps = eps
        self.num = num

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        var = x.square().mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return self.num.bf(self.w * x)


class SharedAttentionRuntime:
    """What attention layers hand down the stack (reference ``shared_attn``)."""

    def __init__(self):
        self.compress_kv: torch.Tensor | None = None
        self.index_k: torch.Tensor | None = None
        self.topk_idxs: torch.Tensor | None = None
        self.candidates: torch.Tensor | None = None


class Engram:
    """Reference ``Engram`` with the table taken from the state dict (fp8 codes + e8m0 scales)."""

    def __init__(self, args: ModelArgs, sd, prefix: str, num: Numerics):
        self.num = num
        self.dim = args.dim
        self.hc_mult = args.hc_mult
        self.clamp_value = 1e-6
        self.eps = args.norm_eps
        self.block_size = FP8_BLOCK_SIZE
        self.table = sd[f"{prefix}.embed.weight"].to(torch.float32)
        self.table_scale = sd[f"{prefix}.embed.scale"].to(torch.float32)
        self.wkv = Linear(sd, f"{prefix}.wkv", "fp8", num)
        self.q_weight = sd[f"{prefix}.q_weight"].to(torch.float32)
        self.k_weight = sd[f"{prefix}.k_weight"].to(torch.float32)

    def embed(self, indices: torch.Tensor) -> torch.Tensor:
        values = self.table[indices]
        scales = self.table_scale[indices]
        values = values.unflatten(-1, (-1, self.block_size)) * scales.unsqueeze(-1)
        return self.num.bf(values.flatten(-2))

    def __call__(self, x: torch.Tensor, hash_ids: torch.Tensor, token_mask: torch.Tensor | None = None) -> torch.Tensor:
        """``x [B, L, hc, dim]``, ``hash_ids [B, L, n_hash_cols]``."""
        kv = self.wkv(self.embed(hash_ids).flatten(-2))
        key, value = kv.split([self.hc_mult * self.dim, self.dim], dim=-1)
        key = key.unflatten(-1, (self.hc_mult, self.dim))
        weight = self.q_weight * self.k_weight
        h, eps = x.to(torch.float32), self.eps
        rstd = torch.rsqrt(h.square().mean(-1) + eps) * torch.rsqrt(key.square().mean(-1) + eps)
        dot = (h * weight * key).sum(-1) * rstd * self.dim**-0.5
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(self.clamp_value).sqrt(), dot))
        if token_mask is not None:
            gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0)
        return self.num.bf(h + gate.unsqueeze(-1) * value.unsqueeze(-2))


class Compressor:
    """Softmax-gated pooling of ``ratio`` tokens into one pre-RoPE latent (ratio 1: plain
    ``norm(wkv(x))``). Ratio > 1 pools in fp32 with the bf16 checkpoint weights promoted to fp32
    (as the reference does) and carries the partial group in ``kv_state`` / ``score_state``.
    Generalized to any ``(start_pos, seqlen)``: group boundaries are absolute."""

    def __init__(self, args: ModelArgs, sd, prefix: str, ratio: int, num: Numerics):
        self.ratio = ratio
        self.num = num
        self.norm = RMSNorm(sd, f"{prefix}.norm", args.norm_eps, num)
        if ratio == 1:
            self.wkv = Linear(sd, f"{prefix}.wkv", "bf16", num)
        else:
            self.wkv = Linear(sd, f"{prefix}.wkv", "fp32", num)
            self.wgate = Linear(sd, f"{prefix}.wgate", "fp32", num)
            shape = (args.max_batch_size, ratio, args.head_dim)
            self.kv_state = torch.zeros(shape, dtype=torch.float32)
            self.score_state = torch.full(shape, -math.inf, dtype=torch.float32)

    def reset(self):
        if self.ratio > 1:
            self.kv_state.zero_()
            self.score_state.fill_(-math.inf)

    def __call__(self, x: torch.Tensor, start_pos: int) -> torch.Tensor | None:
        bsz, seqlen, _ = x.shape
        ratio = self.ratio
        if ratio == 1:
            return self.norm(self.wkv(x))
        x = x.to(torch.float32)
        kv, score = self.wkv(x), self.wgate(x)
        lead = start_pos % ratio
        if lead:
            kv = torch.cat([self.kv_state[:bsz, :lead], kv], dim=1)
            score = torch.cat([self.score_state[:bsz, :lead], score], dim=1)
        total = lead + seqlen
        n_groups, rem = total // ratio, total % ratio
        if rem:
            self.kv_state[:bsz, :rem] = kv[:, total - rem :]
            self.score_state[:bsz, :rem] = score[:, total - rem :]
        if n_groups == 0:
            return None
        kv = kv[:, : n_groups * ratio].unflatten(1, (-1, ratio))
        score = score[:, : n_groups * ratio].unflatten(1, (-1, ratio))
        kv = (kv * score.softmax(dim=2)).sum(dim=2)
        return self.norm(self.num.bf(kv))


class Indexer:
    """Keeps the ``index_topk`` best compressed positions per query (reference ``Indexer``)."""

    def __init__(self, args: ModelArgs, sd, prefix: str, layer_id: int, shared: SharedAttentionRuntime, num: Numerics):
        self.num = num
        self.shared = shared
        self.owns_k = layer_id in args.kv_source_layers
        self.ratio = args.compress_ratios[layer_id]
        self.is_candidate_source = layer_id == args.candidate_source_layer
        self.uses_candidates = 0 <= args.candidate_source_layer < layer_id
        self.candidate_topk_blocks = args.candidate_topk_blocks
        self.candidate_block_size = args.candidate_block_size
        self.n_heads = args.index_n_heads
        self.index_head_dim = args.index_head_dim
        self.rope_head_dim = args.rope_head_dim
        self.index_topk = args.index_topk
        self.softmax_scale = self.index_head_dim**-0.5
        self.wq_b = Linear(sd, f"{prefix}.wq_b", "fp8", num)
        self.weights_proj = Linear(sd, f"{prefix}.weights_proj", "bf16", num)
        self.freqs_cis: torch.Tensor | None = None
        if self.owns_k:
            self.wk = Linear(sd, f"{prefix}.wk", "bf16", num)
            self.k_norm = RMSNorm(sd, f"{prefix}.k_norm", args.norm_eps, num)
            self.k_cache = torch.zeros(args.max_batch_size, args.max_seq_len // self.ratio, args.index_head_dim)

    def reset(self):
        if self.owns_k:
            self.k_cache.zero_()

    def publish_k(self, latent: torch.Tensor, start_pos: int) -> None:
        """Index keys from this layer's pre-RoPE latent: group ``j`` takes position ``j * ratio``."""
        assert self.owns_k and self.freqs_cis is not None
        bsz, n_groups, _ = latent.shape
        g0 = start_pos // self.ratio
        freqs = self.freqs_cis[(g0 + torch.arange(n_groups)) * self.ratio]
        k = self.k_norm(self.wk(latent))
        k = rope_tail(k, self.rope_head_dim, freqs, self.num)
        k = fp4_act_quant_roundtrip(k, FP4_BLOCK_SIZE, "e8m0")
        self.k_cache[:bsz, g0 : g0 + n_groups] = k
        self.shared.index_k = self.k_cache

    def __call__(self, x, qr, latent, start_pos: int, offset: int) -> torch.Tensor:
        assert self.freqs_cis is not None
        bsz, seqlen, _ = x.shape
        ratio, rd, end_pos = self.ratio, self.rope_head_dim, start_pos + seqlen
        if self.owns_k:
            if latent is not None:
                self.publish_k(latent, start_pos)
            # DEVIATION from the reference (treated as a reference bug): model.py only rebinds
            # ``shared_attn.index_k`` when a latent is published, so at a decode step where a
            # ratio-2 owner's group is still filling it scores against the LAST publisher's keys
            # (the ratio-1 kv source's cache from the previous step). An owner always scores
            # against its own keys here, which is also what an owner-mapped port does.
            self.shared.index_k = self.k_cache

        q = self.wq_b(qr).unflatten(-1, (self.n_heads, self.index_head_dim))
        q = rope_tail(q, rd, self.freqs_cis[start_pos:end_pos], self.num)
        q = fp4_act_quant_roundtrip(q, FP4_BLOCK_SIZE, "e8m0")

        index_k = self.shared.index_k[:bsz, : end_pos // ratio]
        weights = self.num.bf(self.weights_proj(x) * (self.softmax_scale * self.n_heads**-0.5))
        score = self.num.bf(torch.einsum("bshd,btd->bsht", q, index_k))
        score = self.num.bf(self.num.bf(score.relu() * weights.unsqueeze(-1)).sum(dim=2))

        # visibility: compressed block j is visible to query t once t >= (j + 1) * ratio - 1
        compress_lens = (torch.arange(start_pos + 1, end_pos + 1) // ratio).unsqueeze(-1)  # [s, 1]
        score = score.masked_fill(torch.arange(end_pos // ratio) >= compress_lens, -math.inf)

        if self.is_candidate_source:
            self.shared.candidates = select_candidate_blocks(
                score, compress_lens, self.candidate_topk_blocks, self.candidate_block_size
            )
        elif self.uses_candidates:
            score = score.masked_fill(~self.shared.candidates, -math.inf)

        topk = min(self.index_topk, end_pos // ratio)
        idxs = stable_topk_indices(score, topk).sort(dim=-1).values
        return torch.where(idxs < compress_lens, idxs + offset, -1).int()


class Attention:
    """Latent attention over the sliding window plus (ratio > 0) ``index_topk`` compressed
    positions, in one ``sparse_attn`` call (reference ``Attention``)."""

    def __init__(self, args: ModelArgs, sd, prefix: str, layer_id: int, shared: SharedAttentionRuntime, num: Numerics):
        self.args = args
        self.num = num
        self.shared = shared
        self.layer_id = layer_id
        self.n_heads = args.n_heads
        self.head_dim = args.head_dim
        self.rope_head_dim = args.rope_head_dim
        self.n_groups = args.o_groups
        self.o_lora_rank = args.o_lora_rank
        self.window_size = args.window_size
        self.ratio = args.compress_ratios[layer_id]
        self.softmax_scale = self.head_dim**-0.5
        self.attn_sink = sd[f"{prefix}.attn_sink"].to(torch.float32)
        self.wq_a = Linear(sd, f"{prefix}.wq_a", "fp8", num)
        self.q_norm = RMSNorm(sd, f"{prefix}.q_norm", args.norm_eps, num)
        self.wq_b = Linear(sd, f"{prefix}.wq_b", "fp8", num)
        self.wkv = Linear(sd, f"{prefix}.wkv", "fp8", num)
        self.kv_norm = RMSNorm(sd, f"{prefix}.kv_norm", args.norm_eps, num)
        # convert.py dequantizes wo_a to bf16 (exact: fp8 x 2^k); used through a grouped einsum
        self.wo_a = bf16_round(dequant_fp8_block(sd[f"{prefix}.wo_a.weight"], sd[f"{prefix}.wo_a.scale"], FP8_BLOCK_SIZE))
        self.wo_b = Linear(sd, f"{prefix}.wo_b", "fp8", num)

        is_backbone = layer_id < args.n_layers
        self.is_kv_source = is_backbone and layer_id in args.kv_source_layers
        self.is_index_source = is_backbone and layer_id in args.index_source_layers
        self.compressor = Compressor(args, sd, f"{prefix}.compressor", self.ratio, num) if self.is_kv_source else None
        self.indexer = Indexer(args, sd, f"{prefix}.indexer", layer_id, shared, num) if self.is_index_source else None

        self.window_kv_cache = torch.zeros(args.max_batch_size, args.window_size, self.head_dim)
        if self.is_kv_source:
            self.compress_kv_cache = torch.zeros(args.max_batch_size, args.max_seq_len // self.ratio, self.head_dim)
        if self.ratio:
            original_seq_len, rope_theta = args.original_seq_len, args.compress_rope_theta
        else:
            original_seq_len, rope_theta = 0, args.rope_theta
        self.freqs_cis = precompute_freqs_cis(
            self.rope_head_dim, args.max_seq_len, original_seq_len, rope_theta, args.rope_factor, args.beta_fast, args.beta_slow
        )
        if self.indexer is not None:
            self.indexer.freqs_cis = self.freqs_cis

    def reset(self):
        self.window_kv_cache.zero_()
        if self.is_kv_source:
            self.compress_kv_cache.zero_()
            self.compressor.reset()
        if self.indexer is not None:
            self.indexer.reset()

    # -- window -------------------------------------------------------------------------------

    def _new_kv(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        kv = self.kv_norm(self.wkv(x))
        kv = rope_tail(kv, self.rope_head_dim, freqs_cis, self.num)
        return act_quant_roundtrip(kv, FP8_BLOCK_SIZE)

    def _ring_write(self, kv: torch.Tensor, start_pos: int) -> None:
        bsz, seqlen, _ = kv.shape
        win = self.window_size
        if seqlen > win:
            kv, start_pos = kv[:, -win:], start_pos + seqlen - win
            seqlen = win
        slots = (start_pos + torch.arange(seqlen)) % win
        self.window_kv_cache[:bsz, slots] = kv

    def _window_kv(self, x, freqs_cis, start_pos: int, window_mode: str):
        bsz, seqlen, _ = x.shape
        win = self.window_size
        kv = self._new_kv(x, freqs_cis)
        if window_mode == "replay":
            # CED decoder replay: query t sees only replay rows [s, t] (s = start_pos, seqlen <= win)
            assert seqlen <= win
            window_kv, idxs = kv, get_window_topk_idxs(win, bsz, seqlen, 0)
        elif start_pos == 0:
            window_kv, idxs = kv, get_window_topk_idxs(win, bsz, seqlen, 0)
        elif seqlen == 1:
            self._ring_write(kv, start_pos)
            return self.window_kv_cache[:bsz].clone(), get_window_topk_idxs(win, bsz, 1, start_pos)
        else:  # oracle extension: chunked prefill
            window_kv = torch.cat([self.window_kv_cache[:bsz].clone(), kv], dim=1)
            idxs = chunk_window_topk_idxs(win, bsz, seqlen, start_pos)
        self._ring_write(kv, start_pos)
        return window_kv, idxs

    # -- compressed ---------------------------------------------------------------------------

    def _compress_topk_idxs(self, x, qr, latent, start_pos, offset, compress_len):
        if not self.is_index_source:
            return self.shared.topk_idxs
        bsz, seqlen, _ = x.shape
        if compress_len == 0:
            idxs = torch.empty(bsz, seqlen, 0, dtype=torch.int32)
        else:
            idxs = self.indexer(x, qr, latent, start_pos, offset)
        self.shared.topk_idxs = idxs
        return idxs

    def _write_compress_kv(self, latent: torch.Tensor, start_pos: int) -> None:
        """RoPE the latent at its group's first position, E2M1/16 with E4M3 scales, into the cache."""
        bsz, n_groups, _ = latent.shape
        g0 = start_pos // self.ratio
        freqs = self.freqs_cis[(g0 + torch.arange(n_groups)) * self.ratio]
        latent = rope_tail(latent, self.rope_head_dim, freqs, self.num)
        latent = fp4_act_quant_roundtrip(latent, CMP_FP4_GROUP, "e4m3")
        self.compress_kv_cache[:bsz, g0 : g0 + n_groups] = latent

    def produce_shared_kv(self, x: torch.Tensor, start_pos: int) -> None:
        """Hoisted production for :func:`ced_prefill`: this kv source's compressor + index keys +
        compressed KV for ``x`` (its attention input) without running attention."""
        assert self.is_kv_source
        latent = self.compressor(x, start_pos)
        self.shared.compress_kv = self.compress_kv_cache
        if latent is not None:
            self.indexer.publish_k(latent, start_pos)
            self._write_compress_kv(latent, start_pos)

    def _compress_kv(self, x, qr, start_pos: int, offset: int, produce_kv: bool):
        bsz, seqlen, _ = x.shape
        compress_len = (start_pos + seqlen) // self.ratio
        latent = None
        if self.is_kv_source and produce_kv:
            latent = self.compressor(x, start_pos)
            self.shared.compress_kv = self.compress_kv_cache
        idxs = self._compress_topk_idxs(x, qr, latent, start_pos, offset, compress_len)
        if latent is not None:
            self._write_compress_kv(latent, start_pos)
        return self.shared.compress_kv[:bsz, :compress_len], idxs

    # -- forward ------------------------------------------------------------------------------

    def __call__(self, x: torch.Tensor, start_pos: int, *, window_mode: str = "auto", produce_kv: bool = True) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
        rd = self.rope_head_dim
        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr).unflatten(-1, (self.n_heads, self.head_dim))
        q = rope_tail(q, rd, freqs_cis, self.num)

        kv, topk_idxs = self._window_kv(x, freqs_cis, start_pos, window_mode)
        if self.ratio:
            compress_kv, compress_idxs = self._compress_kv(x, qr, start_pos, kv.size(1), produce_kv)
            kv = torch.cat([kv, compress_kv], dim=1)
            topk_idxs = torch.cat([topk_idxs, compress_idxs], dim=-1)

        o = sparse_attn(q, kv, self.attn_sink, topk_idxs, self.softmax_scale, self.num)
        o = rope_tail(o, rd, freqs_cis, self.num, inverse=True)
        return self._output(o, bsz, seqlen)

    def _output(self, o: torch.Tensor, bsz: int, seqlen: int) -> torch.Tensor:
        o = o.reshape(bsz, seqlen, self.n_groups, -1)
        wo_a = self.wo_a.view(self.n_groups, self.o_lora_rank, -1)
        o = self.num.bf(torch.einsum("bsgd,grd->bsgr", o, wo_a))
        return self.wo_b(o.flatten(2))


class Gate:
    def __init__(self, args: ModelArgs, sd, prefix: str, layer_id: int):
        _, self.topk = args.get_moe_config(layer_id)
        self.score_func = args.score_func
        self.gate_temp = args.gate_temp
        self.norm_topk_prob = args.norm_topk_prob
        self.route_scale = args.route_scale
        self.weight = sd[f"{prefix}.weight"].to(torch.float32)
        self.bias = sd[f"{prefix}.bias"].to(torch.float32)

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = F.linear(x.to(torch.float32), self.weight) / self.gate_temp
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1)
        elif self.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:
            scores = F.softplus(scores).sqrt()
        indices = stable_topk_indices(scores + self.bias, self.topk)
        self.last_indices = indices  # the harness teacher-forces the port's routing with these
        weights = scores.gather(1, indices)
        if self.norm_topk_prob and self.topk > 1:
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        weights = weights * self.route_scale
        return weights, indices


class Expert:
    """SwiGLU FFN with the training clamps; ``kind`` fp4 (routed) or fp8 (shared)."""

    def __init__(self, sd, prefix: str, kind: str, swiglu_limit: float, num: Numerics):
        self.w1 = Linear(sd, f"{prefix}.w1", kind, num)
        self.w2 = Linear(sd, f"{prefix}.w2", kind, num)
        self.w3 = Linear(sd, f"{prefix}.w3", kind, num)
        self.swiglu_limit = swiglu_limit
        self.num = num

    def __call__(self, x: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
        gate = self.w1(x).to(torch.float32)
        up = self.w3(x).to(torch.float32)
        if self.swiglu_limit > 0:
            up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
            gate = torch.clamp(gate, max=self.swiglu_limit)
        x = F.silu(gate) * up
        if weights is not None:
            x = weights * x
        return self.w2(self.num.bf(x))


class MoE:
    def __init__(self, args: ModelArgs, sd, prefix: str, layer_id: int, num: Numerics):
        self.num = num
        self.dim = args.dim
        self.n_routed_experts, _ = args.get_moe_config(layer_id)
        self.gate = Gate(args, sd, f"{prefix}.gate", layer_id)
        self.experts = [
            Expert(sd, f"{prefix}.experts.{i}", "fp4", args.swiglu_limit, num) for i in range(self.n_routed_experts)
        ]
        self.shared_experts = Expert(sd, f"{prefix}.shared_experts", "fp8", args.swiglu_limit, num)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = x.reshape(-1, self.dim)
        weights, indices = self.gate(x)
        y = torch.zeros_like(x, dtype=torch.float32)
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
        for i in range(self.n_routed_experts):
            if counts[i] == 0:
                continue
            idx, top = torch.where(indices == i)
            y[idx] += self.experts[i](x[idx], weights[idx, top, None])
        y = y + self.shared_experts(x)
        return self.num.bf(y).view(shape)


class Block:
    """Hyper-connected block (reference ``Block``): ``forward(x, start_pos, pre_mix, **attn) ->
    (x, ffn_pre)``; ``pre_mix`` comes from the previous sublayer, attention's ``pre`` feeds the
    FFN, and the FFN's ``pre`` is handed to the next block."""

    attention_cls = Attention

    def __init__(self, args: ModelArgs, sd, prefix: str, layer_id: int, shared: SharedAttentionRuntime, num: Numerics):
        self.args = args
        self.num = num
        self.layer_id = layer_id
        self.norm_eps = args.norm_eps
        self.hc_mult = args.hc_mult
        self.hc_sinkhorn_iters = args.hc_sinkhorn_iters
        self.hc_eps = args.hc_eps
        self.attn = self.attention_cls(args, sd, f"{prefix}.attn", layer_id, shared, num)
        self.ffn = MoE(args, sd, f"{prefix}.ffn", layer_id, num)
        self.attn_norm = RMSNorm(sd, f"{prefix}.attn_norm", args.norm_eps, num)
        self.ffn_norm = RMSNorm(sd, f"{prefix}.ffn_norm", args.norm_eps, num)
        self.hc_attn_fn = sd[f"{prefix}.hc_attn_fn"].to(torch.float32)
        self.hc_ffn_fn = sd[f"{prefix}.hc_ffn_fn"].to(torch.float32)
        self.hc_attn_base = sd[f"{prefix}.hc_attn_base"].to(torch.float32)
        self.hc_ffn_base = sd[f"{prefix}.hc_ffn_base"].to(torch.float32)
        self.hc_attn_scale = sd[f"{prefix}.hc_attn_scale"].to(torch.float32)
        self.hc_ffn_scale = sd[f"{prefix}.hc_ffn_scale"].to(torch.float32)

    def hc_mixes(self, x, hc_fn, hc_scale, hc_base):
        x = x.flatten(2).to(torch.float32)
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x, hc_fn) * rsqrt
        return hc_split_sinkhorn(mixes, hc_scale, hc_base, self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps)

    def hc_pre(self, x: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
        """``[b,s,hc,d] x [b,s,hc] -> [b,s,d]``"""
        return self.num.bf(torch.sum(pre_mix.unsqueeze(-1) * x.to(torch.float32), dim=2))

    def hc_post(self, x, residual, post, comb):
        """output stream j = post[j] * x + sum_i comb[i, j] * residual[i]"""
        y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
        return self.num.bf(y)

    def __call__(self, x: torch.Tensor, start_pos: int, pre_mix: torch.Tensor, **attn_kwargs):
        residual = x
        attn_pre, attn_post, attn_comb = self.hc_mixes(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        x = self.hc_pre(x, pre_mix)
        x = self.attn_norm(x)
        x = self.attn(x, start_pos, **attn_kwargs)
        x = self.hc_post(x, residual, attn_post, attn_comb)

        residual = x
        ffn_pre, ffn_post, ffn_comb = self.hc_mixes(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        x = self.hc_pre(x, attn_pre)
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = self.hc_post(x, residual, ffn_post, ffn_comb)
        return x, ffn_pre


def make_identity_pre_mix(bsz: int, seqlen: int, hc_mult: int) -> torch.Tensor:
    pre_mix = torch.zeros(bsz, seqlen, hc_mult, dtype=torch.float32)
    pre_mix[:, :, 0] = 1.0
    return pre_mix


# --------------------------------------------------------------------------------------------
# DSpark
# --------------------------------------------------------------------------------------------


class DSparkAttention(Attention):
    """Window-only stage attention: the main window ring is fed by ``main_x`` (the target-layer
    hidden state projection); the block rows attend the whole filled ring plus each other."""

    def seed(self, main_x: torch.Tensor, start_pos: int) -> None:
        """Write ``main_x``'s window KV for positions ``start_pos ..`` into the ring."""
        bsz, seqlen, _ = main_x.shape
        main_kv = self._new_kv(main_x, self.freqs_cis[start_pos : start_pos + seqlen])
        self._ring_write(main_kv, start_pos)

    def __call__(self, x: torch.Tensor, start_pos: int, main_x: torch.Tensor, **_):
        assert self.ratio == 0
        bsz, seqlen, _ = main_x.shape
        win, rd = self.window_size, self.rope_head_dim
        _, block_size, _ = x.shape
        first = start_pos + seqlen
        freqs_cis = self.freqs_cis[first : first + block_size]

        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr).unflatten(-1, (self.n_heads, self.head_dim))
        q = rope_tail(q, rd, freqs_cis, self.num)
        kv = self._new_kv(x, freqs_cis)

        topk_idxs = get_dspark_topk_idxs(win, bsz, block_size, min(win, first))
        self.seed(main_x, start_pos)
        kv = torch.cat([self.window_kv_cache[:bsz].clone(), kv], dim=1)
        o = sparse_attn(q, kv, self.attn_sink, topk_idxs, self.softmax_scale, self.num)
        o = rope_tail(o, rd, freqs_cis, self.num, inverse=True)
        return self._output(o, bsz, block_size)


class DSparkBlock(Block):
    attention_cls = DSparkAttention

    def __init__(self, args, sd, prefix, layer_id, shared, num, embed: torch.Tensor, head: torch.Tensor):
        super().__init__(args, sd, prefix, layer_id, shared, num)
        stage_id = layer_id - args.n_layers
        self.block_size = args.dspark_block_size
        self.noise_token_id = args.dspark_noise_token_id
        self.embed_w = embed  # fp32 copy of the bf16 backbone embedding (tied)
        self.head_w = head  # fp32 copy of the bf16 backbone head (tied)
        if stage_id == 0:
            self.main_proj = Linear(sd, f"{prefix}.main_proj", "fp8", num)
            self.main_norm = RMSNorm(sd, f"{prefix}.main_norm", args.norm_eps, num)
        if stage_id == args.n_mtp_layers - 1:
            self.norm = RMSNorm(sd, f"{prefix}.norm", args.norm_eps, num)
            self.markov_embed = sd[f"{prefix}.markov_head.embed.weight"].to(torch.float32)
            self.markov_head = sd[f"{prefix}.markov_head.head.weight"].to(torch.float32)
            self.confidence_proj = sd[f"{prefix}.confidence_head.proj.weight"].to(torch.float32)

    def forward_embed(self, main_hidden: torch.Tensor, input_ids: torch.Tensor):
        main_x = self.main_norm(self.main_proj(main_hidden))
        draft_ids = input_ids.new_full((input_ids.size(0), self.block_size), self.noise_token_id)
        draft_ids[:, 0] = input_ids
        x = self.num.bf(self.embed_w[draft_ids])
        x = x.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
        return x, main_x

    def forward_head(self, x, pre_mix, input_ids, sampler: Callable[[torch.Tensor], torch.Tensor]):
        x = self.hc_pre(x, pre_mix)
        logits = F.linear(self.norm(x).to(torch.float32), self.head_w)  # [B, block, V] fp32
        output_ids = input_ids.new_empty(input_ids.size(0), self.block_size + 1)
        output_ids[:, 0] = input_ids
        markov_embeds = []
        for i in range(self.block_size):
            embed = self.num.bf(self.markov_embed[output_ids[:, i]])
            logits_bias = F.linear(embed.to(torch.float32), self.markov_head)
            logits[:, i] = logits[:, i] + logits_bias
            markov_embeds.append(embed)
            output_ids[:, i + 1] = sampler(logits[:, i])
        markov_embed = torch.stack(markov_embeds, dim=1)
        hidden = torch.cat([x, markov_embed], dim=-1).to(torch.float32)
        confidence = F.linear(hidden, self.confidence_proj).squeeze(-1)
        return output_ids, logits, confidence


# --------------------------------------------------------------------------------------------
# Transformer
# --------------------------------------------------------------------------------------------


@dataclass
class ForwardResult:
    logits: torch.Tensor  # [B, V] fp32, last position
    main_hidden: torch.Tensor | None  # [B, S, n_targets * dim] (bf16-valued) or None
    all_logits: torch.Tensor | None = None  # [B, S, V] when full_logits=True
    replay_start: int = 0  # ced_prefill: absolute position of main_hidden[:, 0]


@dataclass
class SpecResult:
    output_ids: torch.Tensor  # [B, block_size + 1]: anchor then the drafted tokens
    logits: torch.Tensor  # [B, block_size, V] fp32 (markov bias included)
    confidence: torch.Tensor  # [B, block_size] fp32 (pre-sigmoid)


def argmax_sampler(logits: torch.Tensor) -> torch.Tensor:
    return logits.argmax(dim=-1)


class ReferenceModel:
    """``Transformer`` port: embed -> hc_mult copies -> (Engram) blocks -> ``hc_pre`` collapse ->
    norm -> head. Holds every cache the reference holds (window rings, compressed KV / index K,
    compressor partial-group state) so a prefill followed by decode steps behaves like the
    reference generation loop."""

    def __init__(self, args: ModelArgs, state_dict: dict[str, torch.Tensor], numerics: Numerics | None = None):
        self.args = args
        self.num = numerics or Numerics()
        sd = state_dict
        self.shared = SharedAttentionRuntime()
        self.embed = sd["embed.weight"].to(torch.float32)
        self.head = sd["head.weight"].to(torch.float32)
        self.norm = RMSNorm(sd, "norm", args.norm_eps, self.num)
        self.hc_mult = args.hc_mult
        self.layers = [Block(args, sd, f"layers.{i}", i, self.shared, self.num) for i in range(args.n_layers)]
        self.engrams: dict[int, Engram] = {
            lid: Engram(args, sd, f"layers.{lid}.engram", self.num) for lid in args.engram_layer_ids
        }
        self.target_layer_ids = args.dspark_target_layer_ids
        self.mtp: list[DSparkBlock] = []
        if args.dspark_block_size:
            for st in range(args.n_mtp_layers):
                self.mtp.append(
                    DSparkBlock(args, sd, f"mtp.{st}", args.n_layers + st, self.shared, self.num, self.embed, self.head)
                )

    def reset_state(self) -> None:
        """Zero every cache / carried state (a fresh sequence)."""
        for layer in self.layers:
            layer.attn.reset()
        for stage in self.mtp:
            stage.attn.reset()
        self.shared.__init__()

    # -- helpers ------------------------------------------------------------------------------

    def _apply_engram(self, lid: int, h: torch.Tensor, hash_ids: torch.Tensor | None) -> torch.Tensor:
        if lid not in self.engrams:
            return h
        assert hash_ids is not None, f"layer {lid} has an Engram: pass hash_ids [B, L, n_engram_layers, n_hash_cols]"
        return self.engrams[lid](h, hash_ids[:, :, self.args.engram_layer_ids.index(lid), :])

    def logits_from(self, h: torch.Tensor, pre_mix: torch.Tensor, full_logits: bool):
        h = self.layers[-1].hc_pre(h, pre_mix)
        hn = self.norm(h)
        all_logits = F.linear(hn.to(torch.float32), self.head) if full_logits else None
        logits = all_logits[:, -1] if full_logits else F.linear(hn[:, -1].to(torch.float32), self.head)
        return logits, all_logits

    # -- reference forward --------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        start_pos: int = 0,
        hash_ids: torch.Tensor | None = None,
        full_logits: bool = False,
    ) -> ForwardResult:
        """``input_ids [B, S]`` at absolute positions ``start_pos ..``. ``start_pos == 0`` and
        single-token decode are the reference paths; ``start_pos > 0`` with ``S > 1`` is the
        oracle's chunked-prefill extension. ``hash_ids [B, S, n_engram_layers, n_hash_cols]``."""
        bsz, seqlen = input_ids.shape
        h = self.num.bf(self.embed[input_ids])
        h = h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
        main_hiddens = []
        pre_mix = make_identity_pre_mix(bsz, seqlen, self.hc_mult)
        for i, layer in enumerate(self.layers):
            h = self._apply_engram(i, h, hash_ids)
            if i in self.target_layer_ids:
                main_hiddens.append(self.num.bf(h.mean(dim=2)))
            h, pre_mix = layer(h, start_pos, pre_mix)
        logits, all_logits = self.logits_from(h, pre_mix, full_logits)
        main_hidden = torch.cat(main_hiddens, dim=-1) if main_hiddens else None
        return ForwardResult(logits, main_hidden, all_logits, start_pos)

    def forward_spec(
        self,
        input_ids: torch.Tensor,
        main_hidden: torch.Tensor,
        start_pos: int = 0,
        draft: bool | None = None,
        sampler: Callable[[torch.Tensor], torch.Tensor] = argmax_sampler,
    ) -> SpecResult | None:
        """DSpark. ``input_ids [B]``: the token that follows the last ``main_hidden`` row (the
        anchor); ``main_hidden [B, S, n_targets * dim]`` rows for positions ``start_pos ..``.
        Seeds every stage's main window ring with those rows; when ``draft`` (default: ``start_pos
        > 0``, the reference rule) the block ``[anchor, noise x (block_size - 1)]`` at positions
        ``start_pos + S ..`` is drafted and ``SpecResult`` returned, else ``None``."""
        assert self.mtp, "no DSpark stages"
        if draft is None:
            draft = start_pos > 0
        h, main_x = self.mtp[0].forward_embed(main_hidden, input_ids)
        if not draft:
            for stage in self.mtp:
                stage.attn.seed(main_x, start_pos)
            return None
        pre_mix = make_identity_pre_mix(h.size(0), h.size(1), self.hc_mult)
        for stage in self.mtp:
            h, pre_mix = stage(h, start_pos, pre_mix, main_x=main_x)
        output_ids, logits, confidence = self.mtp[-1].forward_head(h, pre_mix, input_ids, sampler)
        return SpecResult(output_ids, logits, confidence)


# --------------------------------------------------------------------------------------------
# Causal Encoder-Decoder prefill (approximation of ``forward``)
# --------------------------------------------------------------------------------------------


def ced_prefill(model: ReferenceModel, input_ids: torch.Tensor, hash_ids: torch.Tensor | None = None) -> ForwardResult:
    """Causal Encoder-Decoder prefill simulation. THIS IS AN APPROXIMATION OF ``forward``:

    1. encoder layers ``[0, D)`` (``D = args.decoder_start()``) run over all ``L`` tokens exactly
       like ``forward(input_ids, 0)`` (Engram, DSpark target stash included);
    2. layer ``D``'s compressor + indexer K run over ALL tokens from the encoder output (the
       attention input of layer ``D``), producing the shared compressed KV / index K for every
       position -- the same values ``forward`` would produce inside layer ``D``;
    3. the decoder layers ``[D, n_layers)`` run over only the LAST ``m = min(window_size, L)``
       tokens (positions ``s = L - m ..``) with the sliding window truncated to the replay
       segment (query ``t`` attends window keys in ``[max(s, t - window + 1), t]``) and compressed
       attention over all positions through the shared KV. Decoder hidden states for positions
       ``< s`` are never computed, so the window KV of those positions (which ``forward`` would
       have attended from queries near ``s``) is missing: that is the approximation.

    Leaves every cache in the state decode expects (window rings hold positions ``[L - window,
    L)``, ring slots of older positions are stale and never read by decode; compressed KV / index
    K hold all ``L`` positions; ratio-2 encoder compressor states as after a normal prefill).
    Returns the last-position logits and ``main_hidden`` for the replay rows (``replay_start =
    s``; seed DSpark with ``model.forward_spec(ids, main_hidden, start_pos=s, draft=False)``).
    With ``window_size >= L`` the truncation is a no-op and the result equals ``forward``."""
    args = model.args
    d = args.decoder_start()
    bsz, seqlen = input_ids.shape
    win = args.window_size
    h = model.num.bf(model.embed[input_ids])
    h = h.unsqueeze(2).repeat(1, 1, model.hc_mult, 1)
    pre_mix = make_identity_pre_mix(bsz, seqlen, model.hc_mult)
    main_hiddens = []
    for i in range(d):
        layer = model.layers[i]
        h = model._apply_engram(i, h, hash_ids)
        if i in model.target_layer_ids:
            main_hiddens.append(model.num.bf(h.mean(dim=2)))
        h, pre_mix = layer(h, 0, pre_mix)

    # hoisted shared KV production from layer D's attention input over all tokens
    dec = model.layers[d]
    h_d = model._apply_engram(d, h, hash_ids)
    x_d = dec.attn_norm(dec.hc_pre(h_d, pre_mix))
    dec.attn.produce_shared_kv(x_d, 0)

    # decoder SWA bounded replay over the last m tokens
    s = max(0, seqlen - win)
    h, pre_mix = h[:, s:], pre_mix[:, s:]
    hash_ids = hash_ids[:, s:] if hash_ids is not None else None
    main_hiddens = [mh[:, s:] for mh in main_hiddens]
    for i in range(d, args.n_layers):
        layer = model.layers[i]
        h = model._apply_engram(i, h, hash_ids)
        if i in model.target_layer_ids:
            main_hiddens.append(model.num.bf(h.mean(dim=2)))
        h, pre_mix = layer(h, s, pre_mix, window_mode="replay", produce_kv=(i != d))
    logits, _ = model.logits_from(h, pre_mix, False)
    main_hidden = torch.cat(main_hiddens, dim=-1) if main_hiddens else None
    return ForwardResult(logits, main_hidden, None, s)


__all__ = [
    "ATTN_SCORE_LOWER_BOUND",
    "CMP_FP4_GROUP",
    "E2M1_VALUES",
    "FP4_BLOCK_SIZE",
    "FP4_MAX",
    "FP4_TABLE",
    "FP8_BLOCK_SIZE",
    "FP8_MAX",
    "Attention",
    "Block",
    "Compressor",
    "DSparkAttention",
    "DSparkBlock",
    "Engram",
    "Expert",
    "ForwardResult",
    "Gate",
    "Indexer",
    "Linear",
    "MoE",
    "ModelArgs",
    "Numerics",
    "RMSNorm",
    "ReferenceModel",
    "SharedAttentionRuntime",
    "SpecResult",
    "act_quant",
    "act_quant_roundtrip",
    "argmax_sampler",
    "bf16_round",
    "ced_prefill",
    "chunk_window_topk_idxs",
    "dequant_fp8_block",
    "e4m3_round",
    "e8m0_scale",
    "fp4_act_quant",
    "fp4_act_quant_roundtrip",
    "fp4_gemm",
    "fp8_gemm",
    "get_dspark_topk_idxs",
    "get_window_topk_idxs",
    "hc_split_sinkhorn",
    "make_identity_pre_mix",
    "make_random_checkpoint",
    "pack_e2m1",
    "pow2_ceil_log2",
    "precompute_freqs_cis",
    "rope_tail",
    "rotate",
    "round_e2m1",
    "select_candidate_blocks",
    "sparse_attn",
    "stable_topk_indices",
    "unpack_e2m1",
]
