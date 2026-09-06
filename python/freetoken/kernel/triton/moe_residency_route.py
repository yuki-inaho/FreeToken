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

v1 substituted at most ONE expert per (row, layer), which is why live NVMe reads only halved
where the offline model predicted 88.7 %: a read is charged per DISTINCT expert per (layer,
step), a speculative step verifies several rows, and a row with two misses kept one of them.
``max_subs`` raises that cap; every swap is still gated on ``tau`` individually and the running
total is bounded by ``tau_total``, so the per-token perturbation budget is the one v1 shipped
with. ``scripts/tier3-residency-replay.py`` measures the trade end to end (reads/step against
the tau=0 arm, on the same LRU streams).

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
    stat_ptr,          # [4] int64: (substitutions, host-missing routes, rows substituted,
                       #             rows that hit the substitution cap)
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
        tl.atomic_add(stat_ptr + 2, 1)
        # rows still holding a miss after their one allowed swap -- exactly what raising
        # max_subs can reach, so the counter says whether it is worth anything here
        left = tl.sum((missing & (off_k != slot)).to(tl.int32))
        tl.atomic_add(stat_ptr + 3, (left > 0).to(tl.int64))


@triton.jit
def _residency_route_multi_kernel(
    logits_ptr,        # [T, E] router logits
    ids_ptr,           # [T, K] int32 -- MUTATED
    w_ptr,             # [T, K] fp32 -- MUTATED
    host_map_ptr,      # raw address of the [L, E] int32 residency map (-1 == not in RAM)
    stat_ptr,          # [4] int64, as above
    stride_logits,
    layer_base,        # layer_id * num_experts
    tau,               # per-swap displacement bound
    tau_total,         # bound on the SUM of a row's displacements
    num_tokens,
    E: tl.constexpr,
    K: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_K: tl.constexpr,
    MAX_SUBS: tl.constexpr,
    COLLECT_STATS: tl.constexpr,
):
    """Up to ``MAX_SUBS`` substitutions per row, greedily cheapest first.

    Fully unrolled and branch-free inside the loop: every iteration runs, and a scalar
    ``active`` flag decides whether its swap is applied. That keeps the program fixed-shape, so
    it captures into the decode graphs exactly as v1 does, and makes ``MAX_SUBS == 1``
    bit-identical to ``_residency_route_kernel`` (same reductions, same expressions).
    """
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

    # ``m`` is fixed for the whole loop: displacement is a ratio and invariant to it, and every
    # later candidate has a logit <= this one, so nothing overflows.
    m = tl.maximum(tl.max(tl.where(k_mask, lg_sel, NEG_INF), axis=0), cand_lg)
    p = tl.where(k_mask, tl.exp(lg_sel - m), 0.0)
    s = tl.sum(p, axis=0)
    cum = 0.0
    n_subs = 0
    active = cand_lg > NEG_INF * 0.5
    for _i in tl.static_range(MAX_SUBS):
        p_c = tl.exp(cand_lg - m)
        denom = s - p + p_c
        disp = tl.where(missing, p / s - p_c / denom, BIG)
        best = tl.min(disp, axis=0)
        slot = tl.argmin(disp, axis=0)
        do_swap = active & (best <= tau) & (cum + best <= tau_total)
        hit = (off_k == slot) & do_swap
        # v1's exact expression for the new partition sum, so MAX_SUBS == 1 matches it bit for bit
        new_s = tl.sum(tl.where(off_k == slot, 0.0, p), axis=0) + p_c
        s = tl.where(do_swap, new_s, s)
        p = tl.where(hit, p_c, p)
        ids = tl.where(hit, cand_id, ids)
        missing = missing & (~hit)
        cum = tl.where(do_swap, cum + best, cum)
        n_subs += do_swap.to(tl.int32)
        # consume this candidate; the next one is the best resident expert still unselected
        avail = avail & (off_e != cand_id)
        lg_avail = tl.where(avail, lg_all, NEG_INF)
        cand_lg = tl.max(lg_avail, axis=0)
        cand_id = tl.argmax(lg_avail, axis=0).to(tl.int32)
        active = do_swap & (tl.sum(missing.to(tl.int32)) > 0) & (cand_lg > NEG_INF * 0.5)

    if n_subs == 0:
        return
    tl.store(ids_ptr + t * K + off_k, ids, mask=k_mask)
    tl.store(w_ptr + t * K + off_k, p / s, mask=k_mask)
    if COLLECT_STATS:
        tl.atomic_add(stat_ptr + 0, n_subs.to(tl.int64))
        tl.atomic_add(stat_ptr + 2, 1)
        tl.atomic_add(stat_ptr + 3, (tl.sum(missing.to(tl.int32)) > 0).to(tl.int64))



@triton.jit
def _residency_route_drop_kernel(
    logits_ptr,        # [T, E] router logits
    ids_ptr,           # [T, K] int32 -- MUTATED
    w_ptr,             # [T, K] fp32 -- MUTATED
    host_map_ptr,      # raw address of the [L, E] int32 residency map (-1 == not in RAM)
    norm_ptr,          # [L, E] fp32 measured mean output norm of each expert
    stat_ptr,          # [8] int64
    stride_logits,
    layer_base,        # layer_id * num_experts
    y_cal,             # this layer's calibration of the ||y|| estimator
    eps,               # per-action bound on ||dy|| / ||y||
    eps_total,         # bound on the row's actions in quadrature
    num_tokens,
    E: tl.constexpr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    MAX_ACTS: tl.constexpr,
    DROP_RENORM: tl.constexpr,
    COLLECT_STATS: tl.constexpr,
):
    """Drop a routed expert that is not in host RAM, priced in what it actually perturbs.

    Measuring every expert against every other (``ft moe subst``) showed that no expert stands in
    for another: the best of 511 candidates lands at cosine +0.02..+0.19, and only 6 pairs in
    46,080 are close enough that substituting beats simply removing the route. So this kernel
    does not substitute at all. It drops, and it prices the drop in the quantity that bounds the
    damage rather than in displaced routing weight:

        ||dy|| / ||y||  =  w_j * n_j / ||y||,     ||y|| ~ y_cal * sqrt(sum_k w_k^2 n_k^2)

    Zeroing the weight -- rather than renormalising the survivors -- is L2-optimal: ``y - w_j E_j``
    is already the best approximation of ``y`` in the span of the remaining experts, so no
    rescaling can improve it, and the block output feeds an additive residual with no norm after
    it. ``DROP_RENORM`` keeps the other form for the A/B.

    Fixed shape and branch-free inside the loop, so it captures into the decode graphs. Unlike
    the logit-gated kernels above it never touches a BLOCK_E-wide vector: everything it needs is
    K-wide, which makes it cheaper than what it replaces.
    """
    t = tl.program_id(0)
    if t >= num_tokens:
        return

    off_k = tl.arange(0, BLOCK_K)
    k_mask = off_k < K

    host_base = host_map_ptr.to(tl.int64).to(tl.pointer_type(tl.int32))
    ids = tl.load(ids_ptr + t * K + off_k, mask=k_mask, other=0)
    safe_ids = tl.maximum(ids, 0)
    lg_sel = tl.load(logits_ptr + t * stride_logits + safe_ids, mask=k_mask,
                     other=NEG_INF).to(tl.float32)
    hslot = tl.load(host_base + layer_base + safe_ids, mask=k_mask, other=0)
    missing = k_mask & (ids >= 0) & (hslot < 0)
    n_missing = tl.sum(missing.to(tl.int32))
    if COLLECT_STATS:
        tl.atomic_add(stat_ptr + 1, n_missing.to(tl.int64))
    if n_missing == 0:
        return

    m = tl.max(tl.where(k_mask, lg_sel, NEG_INF), axis=0)
    p = tl.where(k_mask, tl.exp(lg_sel - m), 0.0)
    s0 = tl.sum(p, axis=0)
    n_sel = tl.load(norm_ptr + layer_base + safe_ids, mask=k_mask, other=0.0).to(tl.float32)

    cum2 = 0.0
    n_drop = 0
    spent = 0.0
    active = n_missing > 0
    for _i in tl.static_range(MAX_ACTS):
        s = tl.sum(p, axis=0) if DROP_RENORM else s0
        w = p / tl.maximum(s, 1e-20)
        # ||y||^2 = sum_k w_k^2 n_k^2 holds when the experts are mutually orthogonal, which is
        # what the substitutability measurement established (mean pairwise cosine ~0); y_cal
        # carries the residual bias, and `ft moe subst` reports how well it holds per layer.
        y2 = tl.sum(w * w * n_sel * n_sel, axis=0)
        yn = tl.maximum(y_cal * tl.sqrt(y2), 1e-12)
        cost = tl.where(missing, w * n_sel / yn, BIG)
        best = tl.min(cost, axis=0)
        slot = tl.argmin(cost, axis=0)
        do = active & (best <= eps) & (cum2 + best * best <= eps_total * eps_total)
        hit = (off_k == slot) & do
        ids = tl.where(hit, -1, ids)
        p = tl.where(hit, 0.0, p)
        missing = missing & (~hit)
        cum2 += tl.where(do, best * best, 0.0)
        spent += tl.where(do, best, 0.0)
        n_drop += do.to(tl.int32)
        active = do & (tl.sum(missing.to(tl.int32)) > 0)

    if n_drop == 0:
        return
    s = tl.sum(p, axis=0) if DROP_RENORM else s0
    tl.store(ids_ptr + t * K + off_k, ids, mask=k_mask)
    tl.store(w_ptr + t * K + off_k, p / tl.maximum(s, 1e-20), mask=k_mask)
    if COLLECT_STATS:
        tl.atomic_add(stat_ptr + 2, 1)
        tl.atomic_add(stat_ptr + 3, (tl.sum(missing.to(tl.int32)) > 0).to(tl.int64))
        tl.atomic_add(stat_ptr + 4, n_drop.to(tl.int64))
        # Q20 fixed point, not a float atomic: one buffer, and deterministic across replays,
        # which matters because these get diffed between arms.
        tl.atomic_add(stat_ptr + 5, (spent * 1048576.0).to(tl.int64))


def residency_route(
    router_logits: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    host_map_addr: int,
    layer_id: int,
    num_experts: int,
    tau: float,
    stats: torch.Tensor | None = None,
    max_subs: int = 1,
    tau_total: float = 0.0,
    norm: torch.Tensor | None = None,
    y_cal: float = 1.0,
    eps: float = 0.0,
    eps_total: float = 0.0,
    drop_renorm: bool = False,
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
        stats = torch.zeros(8, dtype=torch.int64, device=topk_ids.device)
    if norm is not None and eps > 0.0:
        # measured policy: drop, priced in the fraction of the MoE block output it moves
        _residency_route_drop_kernel[(T,)](
            router_logits, topk_ids, topk_weights, host_map_addr, norm, stats,
            router_logits.stride(0), layer_id * num_experts, float(y_cal), float(eps),
            float(eps_total) if eps_total else float(eps), T,
            E=num_experts, K=K, BLOCK_K=triton.next_power_of_2(K),
            # never more than K-1: a row that dropped every route would emit nothing at all,
            # and no budget should be able to buy that
            MAX_ACTS=max(1, min(int(max_subs), K - 1)),
            DROP_RENORM=bool(drop_renorm), COLLECT_STATS=True,
            num_warps=4,
        )
        return
    if max_subs > 1:
        _residency_route_multi_kernel[(T,)](
            router_logits, topk_ids, topk_weights, host_map_addr, stats,
            router_logits.stride(0), layer_id * num_experts, float(tau),
            float(tau_total) if tau_total else float(tau), T,
            E=num_experts, K=K,
            BLOCK_E=triton.next_power_of_2(num_experts),
            BLOCK_K=triton.next_power_of_2(K),
            MAX_SUBS=int(max_subs),
            COLLECT_STATS=True,
            num_warps=4,
        )
        return
    _residency_route_kernel[(T,)](
        router_logits, topk_ids, topk_weights, host_map_addr, stats,
        router_logits.stride(0), layer_id * num_experts, float(tau), T,
        E=num_experts, K=K,
        BLOCK_E=triton.next_power_of_2(num_experts),
        BLOCK_K=triton.next_power_of_2(K),
        COLLECT_STATS=True,
        num_warps=4,
    )
