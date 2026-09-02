"""Turn on NVMe expert streaming on a live ``CpuMoeExecutor``.

Bridges the Python-side geometry (FTW index, host pool shape) to the C++ resolver that runs on
the MoE coordinator thread. Keep the returned handle alive for the process: it owns the pinned
residency map and the per-layer base-pointer tables the C++ dereferences by raw address.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import torch

from freetoken.utils import init_logger

from freetoken.tier3.budget import BankSplit
from freetoken.tier3.nvme_store import INDEX_NAME

logger = init_logger(__name__)


@dataclass
class Tier3Handle:
    """Owns everything the C++ side holds raw pointers into.

    Dropping this frees the pinned residency map and the base-address tables while the
    coordinator thread is still writing through them -- heap corruption, not a clean error. To
    make that unreachable, :func:`enable_cpu_tier3` also stores the handle on the executor, so
    it lives exactly as long as the C++ object that points into it.
    """

    host_slot_for_id: torch.Tensor          # [L, E] int32, pinned; also read by the GPU over UVA
    tables: list[torch.Tensor] = field(default_factory=list)   # per-bank [L] int64 base tables
    paged_banks: tuple[str, ...] = ()
    host_slots: int = 0
    # Layers actually present in the FTW corpus, after the clamp in enable_cpu_tier3. Callers
    # must use THIS for anything that indexes the corpus (prewarm), not the bank-width guess:
    # the two disagree exactly when host_slots == num_experts.
    num_layers: int = 0


def ftw_bank_geometry(ftw_dir: str, banks: tuple[str, ...]) -> tuple[dict, list]:
    """-> ({bank: {layer: (global_off, feat_bytes)}}, [(global_off, nbytes, path), ...])."""
    with open(os.path.join(ftw_dir, INDEX_NAME), encoding="utf-8") as fh:
        index = json.load(fh)
    entries: dict[str, dict[int, tuple[int, int]]] = {b: {} for b in banks}
    for t in index["tensors"]:
        name = t["name"]
        if "#L" not in name:
            continue
        bank, _, suffix = name.partition("#L")
        if bank in entries:
            num_experts = int(t["shape"][0])
            entries[bank][int(suffix)] = (int(t["global_off"]), int(t["nbytes"]) // num_experts)
    shards = [(int(s["global_off"]), int(s["nbytes"]), os.path.join(ftw_dir, s["file"]))
              for s in index["shards"]]
    shards.sort()
    return entries, shards


def enable_cpu_tier3(
    executor,
    bank_sources: dict[str, list[torch.Tensor]],
    ftw_dir: str,
    split: BankSplit,
    *,
    num_layers: int,
    num_experts: int,
    host_slots: int,
    map_layers: int | None = None,
    readers: int = 16,
    min_step_experts: int = 0,
    host_slot_for_id: torch.Tensor | None = None,
) -> Tier3Handle:
    """Install the residency map and start NVMe streaming for ``split.fetched``.

    ``bank_sources`` are the (already allocated) host banks: ``host_slots`` rows for every
    paged bank, full ``num_experts`` rows for the resident ones.

    ``min_step_experts`` is the largest number of DISTINCT experts one decode step can route to
    in a layer (``max_running_req * top_k``). The pool must exceed it: if a step cannot place
    every routed expert the resolver drops the ones it could not, which is wrong numbers rather
    than a crash. Checked here so a bad configuration fails at startup.
    """
    if min_step_experts and host_slots <= min_step_experts:
        raise ValueError(
            f"host pool of {host_slots} slots/layer cannot hold one step's working set of "
            f"{min_step_experts} experts; raise the host RAM budget or lower "
            f"max-running-requests (see tier3_stats()['pool_exhausted'])"
        )
    from freetoken.kernel.pinned import alloc_pinned_tensor

    # ``num_layers`` counts the PAGED layers (those present in the FTW corpus). The residency
    # map may be taller: a model can append resident bank layers past the main tower (qwen4_exp
    # appends the MTP draft experts, built from the raw checkpoint and full width). Those rows
    # are the identity, so every route reads its own expert directly and the resolver -- which
    # only ever looks at layers < num_layers -- never touches them.
    map_layers = map_layers or num_layers

    _raise_fd_limit(len(split.fetched) * 64 + 512)
    entries, shards = ftw_bank_geometry(ftw_dir, split.fetched)
    # The caller identifies the paged tower by bank width (rows == host_slots), which cannot
    # separate it from a resident full-width layer when host_slots == num_experts -- full
    # residency then counted the MTP draft bank as paged and indexed a layer the corpus does
    # not have (KeyError). The corpus is the authority on how many layers were converted.
    corpus_layers = min(len(entries[bank]) for bank in split.fetched)
    if num_layers > corpus_layers:
        logger.info_rank0(
            f"tier 3: {num_layers} bank layers look paged but the FTW corpus holds "
            f"{corpus_layers}; treating the remainder as resident (host_slots={host_slots} "
            f"== num_experts makes the width test ambiguous)"
        )
        num_layers = corpus_layers

    if host_slot_for_id is None:
        raw = alloc_pinned_tensor(map_layers * num_experts * 4, dtype=torch.uint8)
        host_slot_for_id = raw.view(torch.int32).view(map_layers, num_experts)
        host_slot_for_id[:num_layers].fill_(-1)
        if map_layers > num_layers:
            ident = torch.arange(num_experts, dtype=torch.int32)
            host_slot_for_id[num_layers:] = ident
    tables, feat_bytes, entry_off = [], [], []
    # split.fetched follows bank_sources' key order, which must equal the cache's bank_schema
    # order: the copy plan's feat_bytes/dst pointers are built in schema order and the tier-3
    # gathers index them positionally alongside these tables.
    for bank in split.fetched:
        per_layer = bank_sources[bank]
        assert len(per_layer) >= num_layers, (bank, len(per_layer), num_layers)
        assert per_layer[0].size(0) == host_slots, (bank, per_layer[0].shape, host_slots)
        # the same per-layer base-address table shape the executor builds for its own banks
        tbl = torch.tensor([t.data_ptr() for t in per_layer[:num_layers]], dtype=torch.int64)
        tables.append(tbl)
        feat = entries[bank][0][1]
        feat_bytes.append(feat)
        entry_off.extend(entries[bank][l][0] for l in range(num_layers))

    executor._ext.set_host_slot_table(host_slot_for_id.data_ptr())
    executor._ext.tier3_configure(
        num_layers, host_slots, readers, feat_bytes, [t.data_ptr() for t in tables], entry_off,
        [p for _, _, p in shards], [o for o, _, _ in shards], [n for _, n, _ in shards],
    )
    handle = Tier3Handle(host_slot_for_id=host_slot_for_id, tables=tables,
                         paged_banks=split.fetched, host_slots=host_slots,
                         num_layers=num_layers)
    # Tie the lifetime to the executor: the C++ resolver holds raw pointers into `handle`, so
    # letting the caller drop it would corrupt the heap (see Tier3Handle).
    executor._tier3_handle = handle
    return handle


def _raise_fd_limit(want: int) -> None:
    """Nudge the soft RLIMIT_NOFILE up if it is tight.

    The readers now share one descriptor per shard, so the real need is tens -- but the process
    also JIT-builds kernels (which forks and opens pipes) mid-serve, and a 1024 limit shared
    with everything else is close enough to the edge to be worth widening. Mirrors the
    RLIMIT_MEMLOCK nudge in moe.host_banks.
    """
    import resource

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft != resource.RLIM_INFINITY and soft < want:
        new = want if hard == resource.RLIM_INFINITY else min(want, hard)
        if new > soft:
            try:
                resource.setrlimit(resource.RLIMIT_NOFILE, (new, hard))
                logger.info_rank0(f"tier 3: raised RLIMIT_NOFILE {soft} -> {new}")
            except (OSError, ValueError):
                logger.warning(
                    f"tier 3: could not raise RLIMIT_NOFILE from {soft}; if the server dies "
                    f"with 'Too many open files', raise `ulimit -n` before launching")


def tier3_stats(executor) -> dict:
    hits, fetches, nbytes, stall_ns, calls, exhausted = executor._ext.tier3_stats()
    return {
        "host_hits": hits, "nvme_fetches": fetches, "nvme_bytes": nbytes,
        "stall_ms": stall_ns / 1e6, "calls": calls,
        "host_hit_rate": hits / max(1, hits + fetches),
        # MUST stay 0: a non-zero count means the pool ran out of evictable slots mid-step and
        # an expert's contribution was dropped -- wrong numbers, not just slow. Size the pool
        # above the per-step distinct-expert count (max_running_req * top_k).
        "pool_exhausted": exhausted,
    }


def prewarm_pool(executor, bank_sources: dict, store, handle: Tier3Handle, *,
                 num_layers: int, host_slots: int) -> int:
    """Fill every layer's pool with experts ``[0, host_slots)`` and publish the mapping.

    Without this the pool starts empty, so the residency map is all -1 and the ensure kernel
    can fetch nothing: every routed expert falls to the CPU branch and waits on a disk read.
    That is the cold-start cliff (measured 53 tok/s cold vs 78 warm on the same geometry).
    Which experts we choose barely matters -- any ``host_slots`` of them gives the same
    residency fraction -- so the leading block wins because it is one sequential read per
    (bank, layer) instead of thousands of scattered ones.

    Returns bytes read. The C++ side is told afterwards (``tier3_prewarm``) so its slot
    bookkeeping matches; otherwise it would hand out slots that already hold live rows.
    """
    total = 0
    for layer in range(num_layers):
        for bank in handle.paged_banks:
            t = bank_sources[bank][layer]
            mv = memoryview(t.view(torch.uint8).reshape(-1).numpy())
            total += store.read_block(layer, bank, 0, host_slots, mv)
        executor._ext.tier3_prewarm(layer, host_slots)
    return total
