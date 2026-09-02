"""Tier-2: the pinned host expert pool that sits between the GPU slot cache and NVMe.

One **per-layer** pool of ``host_slots`` expert rows per paged bank. Per-layer (rather than one
global pool) is load-bearing: the H2D gather addresses a row as
``base[layer, bank] + row * feat_bytes``, so a per-layer pool keeps that arithmetic correct
with the row index simply reinterpreted as a *host slot* -- the copy kernel does not change at
all. A global pool would force a ``[slot] -> (base, offset)`` indirection inside the kernel.

Policy is plain LRU. That is a measured choice, not a default: over the recorded routing
(see FINDINGS.md) LRU lands within ~1.5 points of the frequency oracle at the operating point
and beats a static hot set at small pool sizes, so the hot-set/LFU machinery is not worth its
calibration pass and staleness failure mode.

Two invariants the rest of the design leans on:

**I1 -- publish ordering.** A fetch is two-phase. :meth:`begin_fetch` clears the victim's
mapping *before* any bytes move; :meth:`commit_fetch` installs the new one *after* they have
landed. A concurrent reader (the GPU gathering over UVA, or the CPU executor) therefore sees
either "not resident" -- which is safe, the route falls through to the CPU branch -- or the
finished mapping. It never sees a slot whose bytes are mid-flight.

**I2 -- eviction exclusion.** The hybrid decode path runs the CPU pass for a layer
*concurrently* with that same layer's PCIe gather, which reads host slots over UVA. Slots named
in ``exclude`` (the routing pack's host-slot row) are therefore never chosen as victims.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Collection, Iterable, Sequence

import torch

NOT_RESIDENT = -1


@dataclass(frozen=True)
class Fetch:
    """One expert to read from NVMe into ``slot`` of ``layer``'s pool."""

    layer: int
    expert: int
    slot: int
    evicted: int  # the expert whose mapping was cleared to make room, or NOT_RESIDENT


class HostExpertPool:
    """Per-layer LRU over expert ids, with a two-phase fetch (see I1/I2 above).

    ``host_slot_for_id`` is the only table a reader outside this class touches; it is allocated
    pinned so the GPU can read it over UVA without a copy (identity-mapped on Linux). Everything
    else is host-private bookkeeping.
    """

    def __init__(self, num_layers: int, num_experts: int, host_slots: int, *,
                 pin: bool = False) -> None:
        assert 0 < host_slots <= num_experts, (host_slots, num_experts)
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.host_slots = host_slots
        self.host_slot_for_id = _alloc((num_layers, num_experts), torch.int32, NOT_RESIDENT, pin)
        self.id_of_host_slot = _alloc((num_layers, host_slots), torch.int32, NOT_RESIDENT, pin)
        # LRU clock. int64 so a long-running server cannot wrap it.
        self.usage = torch.zeros((num_layers, host_slots), dtype=torch.int64)
        self.clock = torch.zeros((num_layers,), dtype=torch.int64)
        self._free = [list(reversed(range(host_slots))) for _ in range(num_layers)]
        self._inflight: list[set[int]] = [set() for _ in range(num_layers)]
        self.hits = 0
        self.misses = 0

    # -- residency ------------------------------------------------------------------

    def slot_of(self, layer: int, expert: int) -> int:
        return int(self.host_slot_for_id[layer, expert])

    def is_resident(self, layer: int, expert: int) -> bool:
        return self.slot_of(layer, expert) != NOT_RESIDENT

    # -- the two-phase fetch --------------------------------------------------------

    def begin_fetch(self, layer: int, experts: Sequence[int], *,
                    exclude: Collection[int] = ()) -> list[Fetch]:
        """Touch one layer's active experts; return the ones that must be read from NVMe.

        Hits have their recency bumped. Misses claim a victim slot whose mapping is cleared
        immediately (I1) and which is marked in-flight so a second miss in the same step cannot
        claim it again. Call :meth:`commit_fetch` once the bytes have landed.

        Raises ``RuntimeError`` if the layer has no evictable slot -- that means ``exclude``
        plus in-flight fetches have pinned the whole pool, which is a sizing bug (the planner's
        ``min_host_slots`` floor exists to prevent it), not a runtime condition to paper over.
        """
        self.clock[layer] += 1
        now = int(self.clock[layer])
        excl = set(exclude)
        out: list[Fetch] = []
        for e in experts:
            slot = self.slot_of(layer, e)
            if slot != NOT_RESIDENT:
                self.hits += 1
                self.usage[layer, slot] = now
                continue
            self.misses += 1
            victim, evicted = self._claim(layer, excl, now)
            # I1: clear BEFORE the read. A reader now sees "not resident" for both experts,
            # which is conservative and safe; it never sees a half-written slot.
            if evicted != NOT_RESIDENT:
                self.host_slot_for_id[layer, evicted] = NOT_RESIDENT
            self.id_of_host_slot[layer, victim] = NOT_RESIDENT
            out.append(Fetch(layer=layer, expert=int(e), slot=victim, evicted=evicted))
        return out

    def commit_fetch(self, fetches: Iterable[Fetch]) -> None:
        """Publish fetched experts (I1). Call only after the bytes are in the slot."""
        for f in fetches:
            self.id_of_host_slot[f.layer, f.slot] = f.expert
            self.host_slot_for_id[f.layer, f.expert] = f.slot
            self._inflight[f.layer].discard(f.slot)

    def abort_fetch(self, fetches: Iterable[Fetch]) -> None:
        """Release slots for reads that never completed; the slots stay unmapped and free."""
        for f in fetches:
            self._inflight[f.layer].discard(f.slot)
            self._free[f.layer].append(f.slot)

    def _claim(self, layer: int, exclude: Collection[int], now: int) -> tuple[int, int]:
        """-> (slot, evicted_expert). Free slots first, then the least-recently-used slot that
        is neither excluded (I2) nor already in flight."""
        free = self._free[layer]
        while free:
            slot = free.pop()
            if slot not in exclude and slot not in self._inflight[layer]:
                self._inflight[layer].add(slot)
                self.usage[layer, slot] = now
                return slot, NOT_RESIDENT
        usage = self.usage[layer]
        blocked = set(exclude) | self._inflight[layer]
        best, best_ts = -1, None
        for slot in range(self.host_slots):
            if slot in blocked:
                continue
            ts = int(usage[slot])
            # strict < keeps the tie-break deterministic (lowest slot index wins), which is
            # what the C++ mirror must reproduce bit-for-bit
            if best_ts is None or ts < best_ts:
                best, best_ts = slot, ts
        if best < 0:
            raise RuntimeError(
                f"layer {layer}: all {self.host_slots} host slots are excluded or in flight "
                f"({len(exclude)} excluded, {len(self._inflight[layer])} in flight) -- the pool "
                f"is undersized for this batch"
            )
        self._inflight[layer].add(best)
        self.usage[layer, best] = now
        return best, int(self.id_of_host_slot[layer, best])


def _alloc(shape: tuple[int, ...], dtype: torch.dtype, fill: int, pin: bool) -> torch.Tensor:
    if pin:
        from freetoken.kernel.pinned import alloc_pinned_tensor

        t = alloc_pinned_tensor(int(torch.tensor(shape).prod()) * torch.empty((), dtype=dtype).element_size(),
                                dtype=torch.uint8).view(dtype).view(*shape)
    else:
        t = torch.empty(shape, dtype=dtype)
    t.fill_(fill)
    return t
