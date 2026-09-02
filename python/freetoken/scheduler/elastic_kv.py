"""Elastic KV pool policy (pure arithmetic, CPU-testable).

An offloaded-MoE server trades KV pages against expert slots one-for-one in VRAM. Most
requests need a small context, so the server boots with a small KV reservation (many expert
slots -> low miss rate -> high decode tok/s) and, when a request needs more context than the
pool holds, grows the pool through the idle-time runtime cache rebuild (paying expert slots),
then shrinks back to the base once a small request arrives after an idle period.
"""
from __future__ import annotations

import math

# generation room assumed on top of the prompt when sizing the pool for a request
GEN_SLACK_MIN = 1024
GEN_SLACK_MAX = 16384
# expert slots left unclaimed on top of the exact byte exchange (allocator granularity)
SLOT_MARGIN = 4


def needed_tokens(input_len: int, max_tokens: int, cap_tokens: int) -> int:
    """Tokens a request wants resident: prompt + a bounded generation allowance, capped."""
    gen = min(max(int(max_tokens), GEN_SLACK_MIN), GEN_SLACK_MAX)
    return min(input_len + gen, cap_tokens)


def plan_target_tokens(needed: int, base_tokens: int, cap_tokens: int) -> int:
    """Smallest power-of-two multiple of ``base_tokens`` >= ``needed`` (clamped to the cap):
    coarse steps keep the number of rebuilds small and make consecutive long requests hit the
    same geometry."""
    if needed <= base_tokens:
        return base_tokens
    t = base_tokens
    while t < needed and t < cap_tokens:
        t *= 2
    return min(t, cap_tokens)


def plan_moe_for_pages(
    cur_pages: int, cur_moe: int, target_pages: int, kv_bytes_per_page: int, moe_bytes_per_slot: int,
    margin_slots: int = SLOT_MARGIN,
) -> int:
    """Expert-slot count that keeps the VRAM budget balanced after moving the KV pool from
    ``cur_pages`` to ``target_pages`` (growth pays slots, shrink reclaims them)."""
    if moe_bytes_per_slot <= 0:
        return cur_moe
    delta_bytes = (target_pages - cur_pages) * kv_bytes_per_page
    if delta_bytes >= 0:
        return cur_moe - math.ceil(delta_bytes / moe_bytes_per_slot) - margin_slots
    return cur_moe + (-delta_bytes) // moe_bytes_per_slot - margin_slots


__all__ = ["GEN_SLACK_MAX", "GEN_SLACK_MIN", "SLOT_MARGIN", "needed_tokens", "plan_moe_for_pages", "plan_target_tokens"]
