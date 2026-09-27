"""Host-RAM preflight for pinned expert banks.

The offload MoE backends pin the whole routed-expert corpus in host RAM (DeepSeek-V4.1:
276 GiB with the DSpark stages) and the parallel bank reader holds a few whole-shard
buffers on top while booting. Today an over-committed box dies late and unhelpfully
(``cudaHostRegister`` failure, or the OOM killer after minutes of loading) -- or, worse,
boots and then starves the page cache the Engram tables live in, so every decode step hits
NVMe cold. :func:`plan_host_ram` is the pure arithmetic (unit-tested against hand-computed
budgets); the engine calls it right before ``load_expert_banks`` with the real numbers and
refuses, or warns, from the plan. ``FREETOKEN_HOST_RAM_PREFLIGHT=0`` disables the refusal.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# Non-bank headroom every boot needs after the banks: the CUDA context, the process, the
# tokenizer and the CPU executor's routing / activation buffers.
DEFAULT_RESERVE_BYTES = 4 << 30
# Below this much page cache the Engram row reads run effectively cold (plan: ~55-65 GiB
# on the 377 GB serving box is the healthy regime; 24 GiB is where it stops being useful).
DEFAULT_MIN_PAGE_CACHE_BYTES = 24 << 30


@dataclass(frozen=True)
class HostRamPlan:
    ok: bool
    refuse_reason: str | None
    warnings: tuple[str, ...]
    # Bytes left for the page cache once the pinned banks and the reserve are resident
    # (the boot transient is freed after loading, so it does not count here). Negative
    # when the banks themselves do not fit.
    page_cache_headroom: int


def _gib(n: int) -> str:
    return f"{n / 2**30:.1f} GiB"


def plan_host_ram(
    bank_bytes: int,
    extra_pinned_bytes: int,
    transient_bytes: int,
    mem_available: int,
    mem_total: int,
    is_wsl: bool,
    allow_wsl: bool = False,
    min_page_cache_bytes: int = DEFAULT_MIN_PAGE_CACHE_BYTES,
    reserve_bytes: int = DEFAULT_RESERVE_BYTES,
) -> HostRamPlan:
    """Decide whether the host can take ``bank_bytes + extra_pinned_bytes`` of pinned banks
    plus a ``transient_bytes`` boot peak out of ``mem_available``.

    Refuses when the peak (banks + extra + transient + reserve) exceeds MemAvailable, and
    under WSL unless ``allow_wsl`` (WSL2 sees a fraction of the host RAM and caps pinning
    near half of it; the split-residency plan is a second-class fallback the caller opts
    into). Warns when the post-boot page-cache headroom is below ``min_page_cache_bytes``
    (Engram tables are served from NVMe through the page cache) and when the pinned set
    alone is most of the machine.
    """
    pinned = int(bank_bytes) + int(extra_pinned_bytes)
    peak = pinned + int(transient_bytes) + int(reserve_bytes)
    headroom = int(mem_available) - pinned - int(reserve_bytes)
    warnings: list[str] = []

    if is_wsl and not allow_wsl:
        return HostRamPlan(
            ok=False,
            refuse_reason=(
                f"running under WSL2: the VM sees {_gib(mem_total)} of host RAM and caps CUDA "
                f"pinning near half of it, so {_gib(pinned)} of pinned expert banks cannot be "
                f"served as planned; boot the native Linux install, or set "
                f"FREETOKEN_ALLOW_WSL=1 to proceed with the split-residency fallback"
            ),
            warnings=(),
            page_cache_headroom=headroom,
        )
    if peak > int(mem_available):
        why = (
            f"host RAM preflight: need {_gib(peak)} available "
            f"(expert banks {_gib(bank_bytes)} + extra pinned {_gib(extra_pinned_bytes)} + "
            f"boot transient {_gib(transient_bytes)} + reserve {_gib(reserve_bytes)}) but "
            f"MemAvailable is {_gib(mem_available)} of {_gib(mem_total)} total"
        )
        if is_wsl:
            why += " (WSL2 VM; raise .wslconfig memory= or use the native Linux boot)"
        why += "; free host memory, or set FREETOKEN_HOST_RAM_PREFLIGHT=0 to try anyway"
        return HostRamPlan(ok=False, refuse_reason=why, warnings=(), page_cache_headroom=headroom)

    if headroom < int(min_page_cache_bytes):
        warnings.append(
            f"host RAM preflight: only {_gib(max(headroom, 0))} will be left for the page cache "
            f"after pinning {_gib(pinned)} (below the {_gib(min_page_cache_bytes)} the "
            f"NVMe-served tables want); expect cold Engram reads and lower decode throughput"
        )
    if mem_total > 0 and pinned > 0.9 * int(mem_total):
        warnings.append(
            f"host RAM preflight: pinned banks {_gib(pinned)} exceed 90% of the machine's "
            f"{_gib(mem_total)}; any other memory user (a second model, a browser) will "
            f"push the box into swap"
        )
    return HostRamPlan(
        ok=True, refuse_reason=None, warnings=tuple(warnings), page_cache_headroom=headroom
    )


def read_meminfo() -> dict[str, int] | None:
    """``{"MemTotal": bytes, "MemAvailable": bytes}`` from /proc/meminfo; ``None`` where
    there is no /proc (Windows, macOS) or the fields are missing -- callers stay inert."""
    out: dict[str, int] = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                key, _, rest = line.partition(":")
                if key in ("MemTotal", "MemAvailable"):
                    out[key] = int(rest.split()[0]) * 1024
    except OSError:
        return None
    if "MemTotal" not in out or "MemAvailable" not in out:
        return None
    return out


def is_wsl() -> bool:
    """WSL2 kernels carry the ``microsoft`` tag in their release string."""
    if not hasattr(os, "uname"):
        return False
    return "microsoft" in os.uname().release.lower()


__all__ = ["HostRamPlan", "plan_host_ram", "read_meminfo", "is_wsl"]
