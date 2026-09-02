from __future__ import annotations

import os

import torch
import triton
import triton.language as tl
from flashlib.kernels.slot_cache import lru_ensure

# Hybrid backend: which of a step's missing experts to fetch (when capped below the miss
# count). "recency" (default) fetches the experts most-recently active before this step
# (LRU on the expert -> prioritizes recurring misses, lowering the steady miss rate);
# "lowest_id" fetches the smallest expert ids (the original, routing-blind heuristic).
_HYBRID_FETCH_BY_RECENCY = (
    os.getenv("FREETOKEN_HYBRID_FETCH", "recency").strip().lower() != "lowest_id"
)


def ensure_experts(cache, layer_id: int, expert_ids: torch.Tensor) -> None:
    """Make this layer's routed experts resident; rewrite ``expert_ids`` to slot ids.

    Delegates to flashlib's slot cache. ``id_base`` maps this layer's expert ids into the
    flat ``layer * num_experts + expert`` space the cache indexes by, and maps
    ``src_indices`` back, so ``copy_missing`` still resolves against this layer's own host
    tensor. ``out_indices`` aliases the input, preserving the in-place rewrite every
    downstream GEMM depends on.
    """
    lru_ensure(
        expert_ids,
        cache.slot_for_id.view(-1),
        cache.id_of_slot,
        cache.usage,
        cache.step,
        expert_ids,
        cache.src_indices,
        cache.evict_slots,
        cache.num_indices,
        stats=cache.lru_stats[layer_id] if cache.collect_stats else None,
        id_base=layer_id * cache.num_experts,
    )


# Dedup GEMV tables hold up to this many routes per distinct expert (one per token: a
# speculative verify is k+1 <= 4 rows). Larger decode batches keep the per-route GEMV.
DEDUP_MAX_TOKENS = 4


def ensure_experts_hybrid(
    cache, layer_id: int, expert_ids: torch.Tensor, max_fetch: int, fetch_fraction: float = 0.0,
    cpu_ids: torch.Tensor | None = None,
    hit_ids: torch.Tensor | None = None,
    fetched_ids: torch.Tensor | None = None,
    dedup: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    weights: torch.Tensor | None = None,
    cpu_w: torch.Tensor | None = None,
) -> None:
    """Capped-fetch variant of ``ensure_experts`` (hybrid backend).

    Identical LRU bookkeeping, but only the first ``max_fetch`` of this step's missing
    experts are given a slot and scheduled for copy; the overflow misses stay
    non-resident and their ``expert_ids`` positions are rewritten to ``-1`` (compute on
    the CPU). ``fetch_fraction`` > 0 replaces the fixed cap with the bandwidth-matched
    split (fraction = pcie_bw / cpu_bw): fetch ~fraction of the step's misses, rounded to
    the integer that makes the PCIe fetch and the CPU overflow compute finish closest to
    together. ``num_indices`` = capped fetch count (copy_missing); ``num_missing_full`` =
    pre-cap miss count (stats).

    ``cpu_ids`` (optional, same shape as ``expert_ids``) receives the CPU side of the
    split in the same launch: the raw expert id where ``expert_ids`` became ``-1``, else
    ``-1`` -- the hybrid layer's routing split used to be five elementwise kernels per
    layer per step. With ``cache.collect_stats`` the launch also accumulates the hybrid
    miss statistics in-kernel (``record_decode_stats_hybrid`` was eleven more).

    ``hit_ids`` / ``fetched_ids`` (both or neither): the GPU routes split by residency --
    ``hit_ids`` = slot for a route whose expert was already resident, ``fetched_ids`` = slot
    for one assigned a slot THIS step (its bytes land with ``copy_missing``), -1 elsewhere --
    so the layer can run the hit GEMV while the PCIe gather is still in flight.

    ``dedup`` = ``(uniq_slots, uniq_hit, uniq_fetched, pairs)``: the GPU routes grouped by
    expert for the dedup GEMV -- ``uniq_*[u]`` the u-th distinct expert's slot (all / hit
    only / fetched only, -1 elsewhere) and ``pairs[u, m]`` the route indices
    (``token * top_k + k``) sharing it, -1 padded (at most one per token)."""
    assert (hit_ids is None) == (fetched_ids is None)
    # Q16 fixed point so the GPU kernel and the CPU reference cap identically (no float).
    frac_q16 = min(1 << 16, max(0, round(fetch_fraction * (1 << 16))))
    if not expert_ids.is_cuda:
        assert dedup is None, "the CPU mirror does not build the dedup tables"
        return _ensure_experts_hybrid_cpu(
            cache, layer_id, expert_ids, max_fetch, frac_q16, cpu_ids=cpu_ids,
            hit_ids=hit_ids, fetched_ids=fetched_ids, weights=weights, cpu_w=cpu_w,
        )
    _ensure_experts_hybrid_gpu(
        cache, layer_id, expert_ids, max_fetch, frac_q16, cpu_ids=cpu_ids,
        hit_ids=hit_ids, fetched_ids=fetched_ids, weights=weights, cpu_w=cpu_w, dedup=dedup,
    )


def prefill_hit_compact(cache, layer_id: int, buffer_id: int) -> None:
    """Compact this layer's cache-resident experts into gather indices, device-side.

    hit = slot_for_id[layer_id][e] >= 2 * num_experts (the double buffer owns the
    slots below, so those bytes are volatile within a prefill chunk and classify
    as miss). Writes fixed-shape ``_prefill_hit_dst``/``_prefill_hit_src`` (buffer
    row / cache slot) and the count into ``_prefill_hit_num``; one launch on the
    current stream, no host sync. Safe against the concurrent buffer invalidation
    on the copy stream: that only rewrites entries already below the threshold."""
    num_experts = cache.num_experts
    _prefill_hit_compact_kernel[(1,)](
        cache.slot_for_id[layer_id],
        cache._prefill_hit_dst,
        cache._prefill_hit_src,
        cache._prefill_hit_num,
        buffer_id * num_experts,
        2 * num_experts,
        num_experts,
        BLOCK=triton.next_power_of_2(num_experts),
    )


def prefetch_admit(cache, cand: torch.Tensor, budget: int, plan: int = 0) -> None:
    """Admit up to ``budget`` predicted experts into the cache and stage ring plan ``plan``.

    ``cand`` is a score-DESCENDING [C] int32 tensor of flat expert ids
    (``layer * num_experts + expert``, -1 padded), typically a predictor's top
    candidates. For each candidate, in order: already-resident (which covers experts an
    earlier candidate just admitted -- the map is the dedup) -> skip; else claim the
    global argmin(usage) victim slot at or above ``prefetch_reserve_lo`` (the prefill
    double buffer's slots are never victims), update both maps, stamp ``usage`` with the
    CURRENT step (as recent as the last demand call, so demand eviction cannot reach it
    for ~cache_size/active-per-call calls), and append (slot, flat id) to plan ``plan``'s
    slices of ``evict_slots2``/``src_indices2``/``num_indices2``. The step clock is
    READ, not advanced; no routing tensor is touched -- the authoritative router path is
    byte-identical with prefetch on. Single fixed-shape launch, device-side counts,
    static slices per plan: CUDA-graph capturable (the in-graph lookahead captures one
    call per hooked layer with plan = target_layer % ring).
    """
    assert cache.num_indices2 is not None, "init_prefetch_plan must run first"
    b = cache.max_prefetch
    evict2 = cache.evict_slots2[plan * b : (plan + 1) * b]
    src2 = cache.src_indices2[plan * b : (plan + 1) * b]
    num2 = cache.num_indices2[plan : plan + 1]
    stamp = cache.pf_stamp[plan : plan + 1]
    if not cand.is_cuda:
        return _prefetch_admit_cpu(cache, cand, budget, evict2, src2, num2, stamp)
    block_c = triton.next_power_of_2(cache.cache_size)
    _prefetch_admit_kernel[(1,)](
        cand,
        cache.slot_for_id,
        cache.id_of_slot,
        cache.usage,
        cache.step,
        evict2,
        src2,
        num2,
        stamp,
        cache.pf_issued,
        cand.numel(),
        int(budget),
        int(cache.prefetch_reserve_lo),
        cache.cache_size,
        BLOCK_C=block_c,
        num_warps=8 if block_c >= 2048 else 4,
    )


def _prefetch_admit_cpu(cache, cand, budget, evict2, src2, num2, stamp) -> None:
    """CPU reference mirror of ``_prefetch_admit_kernel`` (decisions bit-identical)."""
    step = int(cache.step.item())
    stamp.fill_(step)
    reserve_lo = int(cache.prefetch_reserve_lo)
    usage = cache.usage.tolist()
    admitted = 0
    flat_map = cache.slot_for_id.view(-1)
    for flat in cand.view(-1).tolist():
        if admitted >= budget or flat < 0:
            continue
        if int(flat_map[flat].item()) >= 0:
            continue
        victim = min(
            range(reserve_lo, cache.cache_size), key=lambda s: (usage[s], s)
        )
        old_id = int(cache.id_of_slot[victim].item())
        if old_id >= 0:
            flat_map[old_id] = -1
        cache.id_of_slot[victim] = flat
        flat_map[flat] = victim
        cache.usage[victim] = step
        usage[victim] = 1 << 62  # claimed: out of the victim scan
        evict2[admitted] = victim
        src2[admitted] = flat
        admitted += 1
    num2.fill_(admitted)
    cache.pf_issued += admitted


@triton.jit(do_not_specialize=["num_cand", "budget", "reserve_lo"])
def _prefetch_admit_kernel(
    cand_ptr,          # [C] int32 flat expert ids, score-descending, -1 padded. READ-ONLY.
    slot_for_id_ptr,   # [L*E] int32 flat view (shared LRU maps)
    id_of_slot_ptr,    # [S] int32
    usage_ptr,         # [S] int64
    step_ptr,          # [] int64 -- READ only; admission does not advance the clock
    evict2_ptr,        # [budget] int32 out: victim slots (plan-2 dst)
    src2_ptr,          # [budget] int32 out: FLAT expert ids (plan-2 src)
    num2_ptr,          # [1] int64 out: admitted count
    pf_stamp_ptr,      # [1] int64 out: usage-clock snapshot (used-check baseline)
    pf_issued_ptr,     # [1] int64: cumulative admitted counter
    num_cand,
    budget,
    reserve_lo,        # 2 * num_experts under prefill_overlap else 0
    cache_size: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    step = tl.load(step_ptr)
    tl.store(pf_stamp_ptr, step)

    off_c = tl.arange(0, BLOCK_C)
    c_mask = (off_c < cache_size) & (off_c >= reserve_lo)
    # Usage stamps are step counts (< 2^31), so the victim scan runs on int32: half the
    # reduction traffic of the int64 argmin. Reserved / out-of-range slots pin to INF.
    INF32: tl.constexpr = 2147483647
    u64 = tl.load(usage_ptr + off_c, mask=c_mask, other=9223372036854775807).to(tl.int64)
    u = tl.minimum(tl.maximum(u64, 0), INF32 - 1).to(tl.int32)
    u = tl.where(~c_mask, INF32, u)

    admitted = tl.zeros((), dtype=tl.int64)
    for i in tl.range(num_cand):
        flat = tl.load(cand_ptr + i)
        slot = tl.load(slot_for_id_ptr + tl.maximum(flat, 0))
        do = (flat >= 0) & (slot < 0) & (admitted < budget)
        victim = tl.argmin(u, axis=0).to(tl.int32)
        # A slot claimed earlier in this launch sits at INF and is never re-selected, so the
        # victim's owner is still the pre-launch map value: one scalar load, no BLOCK_C-wide
        # gather-reduce per candidate.
        old_id = tl.load(id_of_slot_ptr + victim)
        tl.store(slot_for_id_ptr + tl.maximum(old_id, 0), -1, mask=do & (old_id >= 0))
        tl.store(id_of_slot_ptr + victim, flat, mask=do)
        tl.store(slot_for_id_ptr + tl.maximum(flat, 0), victim, mask=do)
        tl.store(usage_ptr + victim, step, mask=do)
        tl.store(evict2_ptr + admitted, victim, mask=do)
        tl.store(src2_ptr + admitted, flat, mask=do)
        # Claimed slots leave the victim scan; the map update above is the in-flight dedup.
        u = tl.where(do & (off_c == victim), INF32, u)
        admitted += do.to(tl.int64)
    tl.store(num2_ptr, admitted)
    tl.store(pf_issued_ptr, tl.load(pf_issued_ptr) + admitted)


def materialize_layer(cache, layer_id: int) -> None:
    _materialize_layer_gpu(cache, layer_id)


def reset_cache(cache) -> None:
    _reset_cache_gpu(cache)






def _ensure_experts_hybrid_gpu(
    cache, layer_id: int, expert_ids: torch.Tensor, max_fetch: int, frac_q16: int,
    cpu_ids: torch.Tensor | None = None,
    hit_ids: torch.Tensor | None = None,
    fetched_ids: torch.Tensor | None = None,
    dedup: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    weights: torch.Tensor | None = None,
    cpu_w: torch.Tensor | None = None,
) -> None:
    block_e = triton.next_power_of_2(cache.num_experts)
    block_c = triton.next_power_of_2(cache.cache_size)
    block_a = max(16, triton.next_power_of_2(expert_ids.numel()))
    num_warps = 8 if block_c >= 2048 else 4
    collect = bool(getattr(cache, "collect_stats", False))
    if dedup is None:
        uniq_slots = uniq_hit = uniq_fetched = pairs = expert_ids
    else:
        uniq_slots, uniq_hit, uniq_fetched, pairs = dedup
        assert pairs.numel() >= block_a * DEDUP_MAX_TOKENS and uniq_slots.numel() >= block_a
    _ensure_experts_hybrid_kernel[(1,)](
        expert_ids,
        expert_ids if cpu_ids is None else cpu_ids,
        expert_ids if hit_ids is None else hit_ids,
        expert_ids if fetched_ids is None else fetched_ids,
        expert_ids if weights is None else weights,
        expert_ids if cpu_w is None else cpu_w,
        uniq_slots,
        uniq_hit,
        uniq_fetched,
        pairs,
        cache.slot_for_id,
        cache.id_of_slot,
        cache.usage,
        cache.step,
        cache.active_mask,
        cache.evict_slots,
        cache.src_indices,
        cache.num_indices,
        cache.num_missing_full,
        cache.expert_recency,
        cache.stat_missing,
        cache.stat_fetched,
        cache.stat_active,
        cache.stat_calls,
        cache.stat_missing_layer,
        cache.stat_fetched_layer,
        cache.stat_active_layer,
        cache.stat_steps_layer,
        layer_id,
        expert_ids.numel(),
        int(max_fetch),
        int(frac_q16),
        cache.num_experts,
        cache.cache_size,
        BLOCK_E=block_e,
        BLOCK_C=block_c,
        BLOCK_A=block_a,
        BY_RECENCY=_HYBRID_FETCH_BY_RECENCY,
        EMIT_CPU_IDS=cpu_ids is not None,
        EMIT_CPU_W=cpu_w is not None and weights is not None,
        EMIT_SPLIT=hit_ids is not None,
        EMIT_DEDUP=dedup is not None,
        MAXB=DEDUP_MAX_TOKENS,
        COLLECT_STATS=collect,
        num_warps=num_warps,
    )


def _ensure_experts_hybrid_cpu(
    cache, layer_id: int, expert_ids: torch.Tensor, max_fetch: int, frac_q16: int,
    cpu_ids: torch.Tensor | None = None,
    hit_ids: torch.Tensor | None = None,
    fetched_ids: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
    cpu_w: torch.Tensor | None = None,
) -> None:
    """CPU reference mirror of the hybrid kernel (eviction/fetch decisions bit-identical to
    the GPU path; see tests/test_offload_lru_kernels.py). Fetches at most ``max_fetch`` (or
    the bandwidth-matched ``~frac_q16/2^16 * misses`` when ``frac_q16`` > 0) of the missing
    experts; overflow misses are rewritten to -1. With ``BY_RECENCY`` the fetch set is the
    most-recently-active misses (ties -> lower id); else the lowest ids."""
    seen = []
    for expert in expert_ids.view(-1).tolist():
        if expert not in seen:
            seen.append(expert)

    cache.active_mask.zero_()
    step = int(cache.step.item()) + 1
    cache.step.fill_(step)
    for expert in seen:
        cache.active_mask[expert] = 1

    for expert in seen:
        slot = int(cache.slot_for_id[layer_id, expert].item())
        if slot != -1:
            cache.usage[slot] = step

    missing = [e for e in seen if int(cache.slot_for_id[layer_id, e].item()) == -1]
    if _HYBRID_FETCH_BY_RECENCY:
        rec = cache.expert_recency[layer_id].tolist()
        missing.sort(key=lambda e: (-rec[e], e))
    else:
        missing.sort()
    if frac_q16 > 0:
        m, q = len(missing), 1 << 16
        lo = (m * frac_q16) >> 16
        cost = lambda f: max(f * (q - frac_q16), (m - f) * frac_q16)  # noqa: E731
        max_fetch = lo if cost(lo) <= cost(lo + 1) else lo + 1
    num_fetch = min(len(missing), int(max_fetch))
    cache.num_missing_full.fill_(len(missing))
    cache.num_indices.fill_(num_fetch)

    usage = cache.usage.tolist()
    fetched_set = set()
    for idx in range(num_fetch):
        expert = missing[idx]
        fetched_set.add(expert)
        victim = min(range(cache.cache_size), key=lambda s: (usage[s], s))
        old_id = int(cache.id_of_slot[victim].item())
        if old_id >= 0:
            cache.slot_for_id.view(-1)[old_id] = -1
        cache.id_of_slot[victim] = layer_id * cache.num_experts + expert
        cache.slot_for_id[layer_id, expert] = victim
        cache.usage[victim] = step
        usage[victim] = step
        cache.evict_slots[idx] = victim
        cache.src_indices[idx] = expert  # layer-local row

    if _HYBRID_FETCH_BY_RECENCY:
        for expert in seen:
            cache.expert_recency[layer_id, expert] = step

    if getattr(cache, "collect_stats", False):
        num_active = len(seen)
        cache.stat_missing += len(missing)
        cache.stat_fetched += num_fetch
        cache.stat_active += num_active
        cache.stat_calls += 1
        cache.stat_missing_layer[layer_id] += len(missing)
        cache.stat_fetched_layer[layer_id] += num_fetch
        cache.stat_active_layer[layer_id] += num_active
        cache.stat_steps_layer[layer_id] += 1

    # Overflow misses keep slot_for_id == -1, so the rewrite below yields -1 for them.
    flat = expert_ids.view(-1)
    cflat = None if cpu_ids is None else cpu_ids.view(-1)
    if cpu_w is not None and weights is not None:
        n_r = expert_ids.numel()
        cpu_w.view(-1)[:n_r].copy_(weights.reshape(-1)[:n_r].to(torch.float32).view(torch.int32))
    hflat = None if hit_ids is None else hit_ids.view(-1)
    fflat = None if fetched_ids is None else fetched_ids.view(-1)
    for i in range(flat.numel()):
        raw = int(flat[i].item())
        slot = int(cache.slot_for_id[layer_id, raw].item())
        flat[i] = slot
        if cflat is not None:
            cflat[i] = -1 if slot >= 0 else raw
        if hflat is not None:
            fetched = raw in fetched_set
            hflat[i] = -1 if (slot < 0 or fetched) else slot
            fflat[i] = slot if (slot >= 0 and fetched) else -1


def _materialize_layer_gpu(cache, layer_id: int) -> None:
    block = triton.next_power_of_2(max(cache.num_experts, cache.cache_size))
    _materialize_layer_kernel[(1,)](
        cache.slot_for_id,
        cache.id_of_slot,
        cache.usage,
        cache.step,
        cache.evict_slots,
        cache.src_indices,
        cache.num_indices,
        layer_id,
        cache.num_experts,
        cache.cache_size,
        BLOCK=block,
    )


def _reset_cache_gpu(cache) -> None:
    block = 256
    total_ids = cache.num_layers * cache.num_experts
    grid = (triton.cdiv(max(total_ids, cache.cache_size), block),)
    _reset_cache_kernel[grid](
        cache.slot_for_id,
        cache.id_of_slot,
        cache.usage,
        cache.step,
        cache.active_mask,
        cache.num_indices,
        total_ids,
        cache.num_experts,
        cache.cache_size,
        BLOCK=block,
    )


@triton.jit
def _reset_cache_kernel(
    slot_for_id_ptr,
    id_of_slot_ptr,
    usage_ptr,
    step_ptr,
    active_mask_ptr,
    num_indices_ptr,
    total_ids: tl.constexpr,
    num_experts: tl.constexpr,
    cache_size: tl.constexpr,
    BLOCK: tl.constexpr,
):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(slot_for_id_ptr + off, -1, mask=off < total_ids)
    tl.store(id_of_slot_ptr + off, -1, mask=off < cache_size)
    tl.store(usage_ptr + off, 0, mask=off < cache_size)
    tl.store(active_mask_ptr + off, 0, mask=off < num_experts)
    if tl.program_id(0) == 0:
        tl.store(step_ptr, 0)
        tl.store(num_indices_ptr, 0)


@triton.jit
def _materialize_layer_kernel(
    slot_for_id_ptr,
    id_of_slot_ptr,
    usage_ptr,
    step_ptr,
    evict_slots_ptr,
    src_indices_ptr,
    num_indices_ptr,
    layer_id: tl.constexpr,
    num_experts: tl.constexpr,
    cache_size: tl.constexpr,
    BLOCK: tl.constexpr,
):
    off = tl.arange(0, BLOCK)
    expert_mask = off < num_experts
    slot_mask = off < cache_size
    slot = off

    base = layer_id * num_experts
    old_id = tl.load(id_of_slot_ptr + slot, mask=slot_mask, other=-1)
    # Flat ids make "belongs to this layer" a range check instead of a field compare.
    same_layer = slot_mask & (old_id >= base) & (old_id < base + num_experts)
    tl.store(id_of_slot_ptr + slot, -1, mask=same_layer)
    tl.store(usage_ptr + slot, 0, mask=same_layer)

    old_valid = expert_mask & (old_id >= 0) & (~same_layer)
    tl.store(slot_for_id_ptr + old_id, -1, mask=old_valid)

    step = tl.load(step_ptr) + 1
    tl.store(step_ptr, step)
    tl.store(id_of_slot_ptr + slot, base + off, mask=expert_mask)
    tl.store(slot_for_id_ptr + base + off, slot, mask=expert_mask)
    tl.store(usage_ptr + slot, step, mask=expert_mask)
    tl.store(evict_slots_ptr + off, slot, mask=expert_mask)
    tl.store(src_indices_ptr + off, off, mask=expert_mask)  # layer-local row
    tl.store(num_indices_ptr, num_experts)




@triton.jit(do_not_specialize=["layer_id", "num_active", "max_fetch", "fetch_frac_q16"])
def _ensure_experts_hybrid_kernel(
    expert_ids_ptr,
    cpu_ids_ptr,
    hit_ids_ptr,
    fetched_ids_ptr,
    w_ptr,            # [A] fp32 router weights (EMIT_CPU_W)
    cpu_w_ptr,        # [A] int32 out: the weights' bits next to cpu_ids (one D2H for both)
    uniq_slots_ptr,
    uniq_hit_ptr,
    uniq_fetched_ptr,
    pairs_ptr,
    slot_for_id_ptr,
    id_of_slot_ptr,
    usage_ptr,
    step_ptr,
    active_mask_ptr,
    evict_slots_ptr,
    src_indices_ptr,
    num_indices_ptr,
    num_missing_full_ptr,
    expert_recency_ptr,
    stat_missing_ptr,
    stat_fetched_ptr,
    stat_active_ptr,
    stat_calls_ptr,
    stat_missing_layer_ptr,
    stat_fetched_layer_ptr,
    stat_active_layer_ptr,
    stat_steps_layer_ptr,
    layer_id,
    num_active,
    max_fetch,
    fetch_frac_q16,
    num_experts: tl.constexpr,
    cache_size: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_A: tl.constexpr,
    BY_RECENCY: tl.constexpr,
    EMIT_CPU_IDS: tl.constexpr,
    EMIT_CPU_W: tl.constexpr,
    EMIT_SPLIT: tl.constexpr,
    EMIT_DEDUP: tl.constexpr,
    MAXB: tl.constexpr,
    COLLECT_STATS: tl.constexpr,
):
    """Capped-fetch timestamp-LRU (hybrid backend).

    Same as ``_ensure_experts_lru_v2_kernel`` but only ``min(num_missing, max_fetch)``
    missing experts are evicted-into / scheduled for copy; the overflow misses stay
    non-resident, so Phase 3 rewrites their positions to -1 (the layer computes those on
    the CPU). ``fetch_frac_q16`` > 0 (Q16 fixed point) replaces the fixed cap with the
    bandwidth-matched split ``~frac * num_missing`` (see the Phase-1 comment), computed
    in-kernel because ``num_missing`` only exists device-side (CUDA graph). ``num_indices``
    = the capped fetch count (copy_missing), ``num_missing_full`` = the pre-cap miss count
    (stats).

    Which misses to fetch is the cap policy. ``BY_RECENCY`` (default) fetches the experts
    most-recently active before this step (LRU on the expert, via ``expert_recency``),
    breaking ties toward the lower expert id -- this prioritizes *recurring* misses for
    caching, lowering the steady miss rate. Otherwise the lowest expert ids are fetched
    (``missing_rank``), the original routing-blind heuristic.

    ``EMIT_CPU_IDS``: Phase 3 also writes ``cpu_ids`` = raw expert id for the overflow
    (-1) positions and -1 elsewhere, i.e. the CPU executor's route list, so the layer
    needs no elementwise split kernels. ``EMIT_SPLIT``: also ``hit_ids`` (slot for routes
    already resident before this step, else -1) and ``fetched_ids`` (slot for routes given
    a slot this step, else -1), the residency split behind the demand-copy overlap.
    ``COLLECT_STATS``: the hybrid miss/fetch/active counters (scalar + per-layer) are
    accumulated in this launch."""
    step = tl.load(step_ptr) + 1
    tl.store(step_ptr, step)
    base = layer_id * num_experts

    # ---- Phase 1: active + missing over experts ----
    off_e = tl.arange(0, BLOCK_E)
    e_mask = off_e < num_experts
    # All routes as one vector (BLOCK_A lanes, -1 past num_active): the active mask is a
    # [BLOCK_E, BLOCK_A] compare reduced over routes -- one vector op instead of
    # num_active dependent scalar loads.
    off_a = tl.arange(0, BLOCK_A)
    a_mask = off_a < num_active
    ids_vec = tl.load(expert_ids_ptr + off_a, mask=a_mask, other=-1)
    is_active = tl.sum((off_e[:, None] == ids_vec[None, :]).to(tl.int32), axis=1) > 0
    tl.store(active_mask_ptr + off_e, is_active.to(tl.int32), mask=e_mask)
    slot = tl.load(slot_for_id_ptr + base + off_e, mask=e_mask, other=-1)
    is_missing = is_active & (slot == -1) & e_mask
    num_missing = tl.sum(is_missing.to(tl.int32))
    # Cap the fetches; the overflow misses are computed on the CPU (left non-resident).
    if fetch_frac_q16 > 0:
        # Bandwidth-matched split (fetch_frac = pcie_bw / cpu_bw): fetch time scales with
        # F * (1 - frac), CPU time with (M - F) * frac; they balance at F = frac * M. Pick
        # the integer neighbor that minimizes the slower (max) side of the overlap.
        lo = (num_missing * fetch_frac_q16) >> 16
        cost_lo = tl.maximum(lo * ((1 << 16) - fetch_frac_q16), (num_missing - lo) * fetch_frac_q16)
        cost_hi = tl.maximum(
            (lo + 1) * ((1 << 16) - fetch_frac_q16), (num_missing - lo - 1) * fetch_frac_q16
        )
        max_fetch = tl.where(cost_lo <= cost_hi, lo, lo + 1)
    num_fetch = tl.minimum(num_missing, max_fetch)
    tl.store(num_missing_full_ptr, num_missing.to(tl.int64))
    tl.store(num_indices_ptr, num_fetch.to(tl.int64))
    is_hit = is_active & (slot >= 0)
    tl.store(usage_ptr + slot, step, mask=is_hit)

    # Fetch-selection priority: encode (recency desc, id asc) into one strictly-ordered
    # score so argmax has no ties (rec deltas are multiples of num_experts; the id term
    # spans only [0, num_experts), so it can only break exact-recency ties).
    if BY_RECENCY:
        rec = tl.load(expert_recency_ptr + base + off_e, mask=e_mask, other=-1).to(tl.int64)
        score = tl.where(
            is_missing, rec * num_experts + (num_experts - 1 - off_e), -1152921504606846976
        ).to(tl.int64)
    else:
        missing_rank = tl.cumsum(is_missing.to(tl.int32)) - 1

    # ---- Phase 2: evict victims by argmin(usage), only for the capped fetches ----
    fetched_e = tl.zeros((BLOCK_E,), dtype=tl.int1)
    if num_fetch > 0:
        off_c = tl.arange(0, BLOCK_C)
        c_mask = off_c < cache_size
        oid = tl.load(id_of_slot_ptr + off_c, mask=c_mask, other=-1)
        # Usage stamps are step counts (< 2^31): int32 victim scan (see _prefetch_admit_kernel).
        INF32: tl.constexpr = 2147483647
        u64 = tl.load(usage_ptr + off_c, mask=c_mask, other=9223372036854775807).to(tl.int64)
        u = tl.minimum(tl.maximum(u64, 0), INF32 - 1).to(tl.int32)
        # A slot whose owner is one of THIS layer's active experts can never be a victim:
        # gather the active mask stored above (same program; barrier orders store -> load).
        tl.debug_barrier()
        in_layer = c_mask & (oid >= base) & (oid < base + num_experts)
        owner_active = tl.load(
            active_mask_ptr + (oid - base), mask=in_layer, other=0
        ) > 0
        u = tl.where(owner_active | (~c_mask), INF32, u)
        for i in tl.range(num_fetch):
            victim = tl.argmin(u, axis=0).to(tl.int32)
            # Claimed slots sit at INF and are never re-selected, so the victim's owner is
            # still the pre-launch map value: a scalar load replaces the BLOCK_C-wide gather.
            old_id = tl.load(id_of_slot_ptr + victim)
            if old_id >= 0:
                tl.store(slot_for_id_ptr + old_id, -1)
            if BY_RECENCY:
                e = tl.argmax(score, axis=0).to(tl.int32)
                score = tl.where(off_e == e, -1152921504606846976, score)
            else:
                e = tl.sum(tl.where((missing_rank == i) & is_missing, off_e, 0))
            tl.store(id_of_slot_ptr + victim, base + e)
            tl.store(slot_for_id_ptr + base + e, victim)
            tl.store(usage_ptr + victim, step)
            tl.store(evict_slots_ptr + i, victim)
            tl.store(src_indices_ptr + i, e)  # layer-local row
            u = tl.where(off_c == victim, INF32, u)
            fetched_e = fetched_e | (off_e == e)

    # ---- Phase 3: rewrite expert_ids -> slot id (hit/fetched) or -1 (overflow -> CPU) ----
    # Phase 2's slot_for_id stores are ordered before this gather by the barrier.
    tl.debug_barrier()
    s_vec = tl.load(slot_for_id_ptr + base + ids_vec, mask=a_mask, other=-1)
    tl.store(expert_ids_ptr + off_a, s_vec, mask=a_mask)
    if EMIT_CPU_IDS:
        tl.store(cpu_ids_ptr + off_a, tl.where(s_vec >= 0, -1, ids_vec), mask=a_mask)
    if EMIT_CPU_W:
        wv = tl.load(w_ptr + off_a, mask=a_mask, other=0.0).to(tl.float32)
        tl.store(cpu_w_ptr + off_a, wv.to(tl.int32, bitcast=True), mask=a_mask)
    if EMIT_SPLIT or EMIT_DEDUP:
        # fetched-this-step flag per route: [BLOCK_E, BLOCK_A] compare against fetched_e
        f_vec = tl.sum(
            tl.where(off_e[:, None] == ids_vec[None, :], fetched_e.to(tl.int32)[:, None], 0),
            axis=0,
        ) > 0
    if EMIT_SPLIT:
        tl.store(hit_ids_ptr + off_a, tl.where((s_vec >= 0) & (~f_vec), s_vec, -1), mask=a_mask)
        tl.store(fetched_ids_ptr + off_a, tl.where((s_vec >= 0) & f_vec, s_vec, -1), mask=a_mask)
    if EMIT_DEDUP:
        # Group the GPU routes by expert: leader = lowest route lane with the same expert;
        # leaders get compact indices, every route its leader's index and its rank among
        # the routes sharing the expert (< MAXB: at most one route per token).
        gpu_route = a_mask & (s_vec >= 0)
        same = (ids_vec[:, None] == ids_vec[None, :]) & gpu_route[:, None] & gpu_route[None, :]
        leader = tl.min(tl.where(same, off_a[None, :], BLOCK_A), axis=1)
        is_leader = gpu_route & (leader == off_a)
        uniq_idx = tl.cumsum(is_leader.to(tl.int32), axis=0) - 1
        u_of_route = tl.sum(tl.where(same & is_leader[None, :], uniq_idx[None, :], 0), axis=1)
        rank_m = tl.sum((same & (off_a[None, :] < off_a[:, None])).to(tl.int32), axis=1)
        neg_a = tl.zeros((BLOCK_A,), dtype=tl.int32) - 1
        tl.store(uniq_slots_ptr + off_a, neg_a)
        tl.store(uniq_hit_ptr + off_a, neg_a)
        tl.store(uniq_fetched_ptr + off_a, neg_a)
        offs_p = tl.arange(0, BLOCK_A * MAXB)
        tl.store(pairs_ptr + offs_p, tl.zeros((BLOCK_A * MAXB,), dtype=tl.int32) - 1)
        tl.debug_barrier()
        tl.store(uniq_slots_ptr + uniq_idx, s_vec, mask=is_leader)
        tl.store(uniq_hit_ptr + uniq_idx, tl.where(f_vec, -1, s_vec), mask=is_leader)
        tl.store(uniq_fetched_ptr + uniq_idx, tl.where(f_vec, s_vec, -1), mask=is_leader)
        tl.store(pairs_ptr + u_of_route * MAXB + rank_m, off_a.to(tl.int32), mask=gpu_route)

    if COLLECT_STATS:
        num_act = tl.sum((is_active & e_mask).to(tl.int64))
        nm = num_missing.to(tl.int64)
        nf = num_fetch.to(tl.int64)
        tl.store(stat_missing_ptr, tl.load(stat_missing_ptr) + nm)
        tl.store(stat_fetched_ptr, tl.load(stat_fetched_ptr) + nf)
        tl.store(stat_active_ptr, tl.load(stat_active_ptr) + num_act)
        tl.store(stat_calls_ptr, tl.load(stat_calls_ptr) + 1)
        tl.store(stat_missing_layer_ptr + layer_id, tl.load(stat_missing_layer_ptr + layer_id) + nm)
        tl.store(stat_fetched_layer_ptr + layer_id, tl.load(stat_fetched_layer_ptr + layer_id) + nf)
        tl.store(stat_active_layer_ptr + layer_id, tl.load(stat_active_layer_ptr + layer_id) + num_act)
        tl.store(stat_steps_layer_ptr + layer_id, tl.load(stat_steps_layer_ptr + layer_id) + 1)

    # Bump every active expert's recency to this step (LRU on the expert): an overflow miss
    # computed on the CPU now ranks high if it recurs, so it gets fetched next time.
    if BY_RECENCY:
        step_vec = tl.zeros((BLOCK_E,), dtype=tl.int64) + step
        tl.store(expert_recency_ptr + base + off_e, step_vec, mask=is_active & e_mask)


@triton.jit(do_not_specialize=["buffer_base"])
def _prefill_hit_compact_kernel(
    slot_ptr,     # [num_experts] int32: this layer's slot_for_id row
    dst_ptr,      # [num_experts] int32 out: buffer rows, compacted
    src_ptr,      # [num_experts] int32 out: cache slots, compacted
    num_ptr,      # [1] int64 out: hit count
    buffer_base,  # buffer_id * num_experts
    threshold,    # 2 * num_experts
    num_experts,
    BLOCK: tl.constexpr,
):
    offs = tl.arange(0, BLOCK)
    lane = offs < num_experts
    slots = tl.load(slot_ptr + offs, mask=lane, other=-1)
    is_hit = lane & (slots >= threshold)
    pos = tl.cumsum(is_hit.to(tl.int32)) - 1
    tl.store(dst_ptr + pos, (buffer_base + offs).to(tl.int32), mask=is_hit)
    tl.store(src_ptr + pos, slots, mask=is_hit)
    tl.store(num_ptr, tl.sum(is_hit.to(tl.int64)))
