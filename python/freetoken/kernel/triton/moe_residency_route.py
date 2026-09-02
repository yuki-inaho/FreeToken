# SPDX-License-Identifier: Apache-2.0
"""Residency-aware routing: prefer a near-tied RESIDENT expert over one that costs a disk read.

Only meaningful under the NVMe expert tier (``FREETOKEN_TIER3_FTW``), where a layer's host pool
holds ``host_slots < num_experts`` experts and routing to a non-resident one stalls the decode
step on a 2.64 MiB O_DIRECT read. Measured on the 48 GiB profile (352 slots): host misses are
1.49 % of routed slots, carry 0.76x the mean gate weight, and 99.3 % of them have a resident
expert available outside the token's top-k.

The substitution is gated on how much routing weight it actually moves, so the perturbation is
bounded per decision rather than unbounded. ``tau`` is that bound; at 0.05 the offline replay
avoids 88.7 % of host misses while displacing 1.5e-4 of all routed expert weight.

This CHANGES MODEL OUTPUTS. It is opt-in (``--moe-residency-routing``), refused when the NVMe
tier is off, and its quality gates -- not the arithmetic above -- are what decide a tau.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# tl.constexpr(...) and NOT an annotation: `x: tl.constexpr = v` is rejected inside
# @jit'ed code (same trap as the NVFP4 store kernel).
NEG_INF = tl.constexpr(-3.4028234663852886e38)
BIG = tl.constexpr(3.4028234663852886e38)


@triton.jit
def _residency_route_kernel(
    logits_ptr,        # [T, E] router logits
    ids_ptr,           # [T, K] int32 -- MUTATED
    w_ptr,             # [T, K] fp32 -- MUTATED
    host_map_ptr,      # raw address of the [L, E] int32 residency map (-1 == not in RAM)
    stat_ptr,          # [2] int64: (substitutions, host-missing routes seen)
    stride_logits,
    layer_base,        # layer_id * num_experts
    tau,
    num_tokens,
    E: tl.constexpr,
    K: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_K: tl.constexpr,
    COLLECT_STATS: tl.constexpr,
):
    t = tl.program_id(0)
    if t >= num_tokens:
        return

    off_e = tl.arange(0, BLOCK_E)
    e_mask = off_e < E
    off_k = tl.arange(0, BLOCK_K)
    k_mask = off_k < K

    host_base = host_map_ptr.to(tl.int64).to(tl.pointer_type(tl.int32))
    ids = tl.load(ids_ptr + t * K + off_k, mask=k_mask, other=0)
    safe_ids = tl.maximum(ids, 0)
    lg_sel = tl.load(logits_ptr + t * stride_logits + safe_ids, mask=k_mask,
                     other=NEG_INF).to(tl.float32)
    hslot_sel = tl.load(host_base + layer_base + safe_ids, mask=k_mask, other=0)
    missing = k_mask & (ids >= 0) & (hslot_sel < 0)
    n_missing = tl.sum(missing.to(tl.int32))
    if COLLECT_STATS:
        tl.atomic_add(stat_ptr + 1, n_missing.to(tl.int64))
    if n_missing == 0:
        return

    # Best resident expert this token has NOT already selected. Its logit is <= every selected
    # logit by construction, so substituting always moves weight DOWN and the displacement is
    # non-negative -- which is what makes a single scalar tau a meaningful bound.
    lg_all = tl.load(logits_ptr + t * stride_logits + off_e, mask=e_mask,
                     other=NEG_INF).to(tl.float32)
    hslot_all = tl.load(host_base + layer_base + off_e, mask=e_mask, other=-1)
    selected = tl.zeros((BLOCK_E,), dtype=tl.int1)
    for j in tl.static_range(K):
        selected = selected | (off_e == tl.sum(tl.where(off_k == j, safe_ids, 0)))
    avail = e_mask & (hslot_all >= 0) & (~selected)
    lg_avail = tl.where(avail, lg_all, NEG_INF)
    cand_lg = tl.max(lg_avail, axis=0)
    cand_id = tl.argmax(lg_avail, axis=0).to(tl.int32)
    if cand_lg <= NEG_INF * 0.5:  # no resident expert outside the top-k: route as-is
        return

    # Weights are softmax over ALL experts, renormalised across the top-k, so the full
    # partition function cancels and the displacement is exact from the top-k logits alone.
    m = tl.maximum(tl.max(tl.where(k_mask, lg_sel, NEG_INF), axis=0), cand_lg)
    p = tl.where(k_mask, tl.exp(lg_sel - m), 0.0)
    s = tl.sum(p, axis=0)
    p_c = tl.exp(cand_lg - m)
    # Displacing slot j: the substitute lands with p_c / (s - p_j + p_c); the weight that
    # leaves expert j is the cost. Pick the cheapest missing slot.
    denom = s - p + p_c
    disp = tl.where(missing, p / s - p_c / denom, BIG)
    best = tl.min(disp, axis=0)
    if best > tau:
        return
    slot = tl.argmin(disp, axis=0)

    new_s = tl.sum(tl.where(off_k == slot, 0.0, p), axis=0) + p_c
    new_w = tl.where(off_k == slot, p_c, p) / new_s
    tl.store(ids_ptr + t * K + off_k, tl.where(off_k == slot, cand_id, ids), mask=k_mask)
    tl.store(w_ptr + t * K + off_k, new_w, mask=k_mask)
    if COLLECT_STATS:
        tl.atomic_add(stat_ptr + 0, 1)


def residency_route(
    router_logits: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    host_map_addr: int,
    layer_id: int,
    num_experts: int,
    tau: float,
    stats: torch.Tensor | None = None,
) -> None:
    """Rewrite ``topk_ids``/``topk_weights`` in place, substituting host-resident experts.

    One program per token; fixed shape, device-side throughout, so it captures into the decode
    graphs like every other decode kernel. ``host_map_addr`` is the raw address of the residency
    map (``offload_kernels._host_map_addr``), which may be a device mirror or the pinned map
    read over UVA.
    """
    T, K = topk_ids.shape
    assert topk_weights.shape == (T, K)
    assert router_logits.shape[0] == T and router_logits.shape[1] == num_experts
    assert topk_ids.dtype == torch.int32 and topk_weights.dtype == torch.float32
    assert router_logits.stride(1) == 1
    if stats is None:
        stats = torch.zeros(2, dtype=torch.int64, device=topk_ids.device)
    _residency_route_kernel[(T,)](
        router_logits, topk_ids, topk_weights, host_map_addr, stats,
        router_logits.stride(0), layer_id * num_experts, float(tau), T,
        E=num_experts, K=K,
        BLOCK_E=triton.next_power_of_2(num_experts),
        BLOCK_K=triton.next_power_of_2(K),
        COLLECT_STATS=True,
        num_warps=4,
    )
