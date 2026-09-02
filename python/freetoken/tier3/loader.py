"""Allocate the tier-2 host pool instead of loading the whole expert corpus.

The normal FTW path (``checkpoint.ftw.load_ftw_banks``) allocates one
``[num_experts, *row]`` pinned bank per layer and reads every byte off disk. Tier 3 allocates
``[host_slots, *row]`` instead and reads **nothing**: the pool starts empty and the C++
resolver pages experts in on demand. So boot is near-instant and host RAM is
``host_slots / num_experts`` of the corpus.

Every bank is paged, at the same host_slots width -- see :class:`freetoken.tier3.budget.BankSplit`
for why the fused H2D gather forces that.
"""

from __future__ import annotations

import json
import os

import torch

from freetoken.moe.expert_banks import ExpertBanks
from freetoken.tier3.nvme_store import INDEX_NAME
from freetoken.utils import init_logger

logger = init_logger(__name__)

_DTYPE = {
    "uint8": torch.uint8, "int8": torch.int8, "float16": torch.float16,
    "bfloat16": torch.bfloat16, "float32": torch.float32,
    "float8_e4m3fn": torch.float8_e4m3fn, "float8_e5m2": torch.float8_e5m2,
}


def ftw_expert_bank_shapes(ftw_dir: str) -> tuple[dict[str, dict[int, tuple]], str | None]:
    """-> ({bank: {layer: (shape, dtype_str)}}, quant_format)."""
    with open(os.path.join(ftw_dir, INDEX_NAME), encoding="utf-8") as fh:
        index = json.load(fh)
    out: dict[str, dict[int, tuple]] = {}
    for t in index["tensors"]:
        if t.get("kind") != "experts_bank" or "#L" not in t["name"]:
            continue
        bank, _, suffix = t["name"].partition("#L")
        out.setdefault(bank, {})[int(suffix)] = (tuple(t["shape"]), t["dtype"])
    if not out:
        raise ValueError(
            f"{ftw_dir} has no per-layer expert-bank entries; tier 3 needs a per-layer FTW "
            f"conversion (ft checkpoint --moe-backend offload)"
        )
    return out, index.get("quant_format")


def load_tier3_banks(ftw_dir: str, host_slots: int, *, pin: bool = True) -> ExpertBanks:
    """Empty pinned host pools sized for ``host_slots`` experts per layer.

    Nothing is read from disk here -- that is the point. The banks are zeroed, and every row
    becomes valid only once the resolver has fetched an expert into it and published the
    mapping, so no consumer may address a row the residency map does not name.
    """
    from freetoken.kernel.pinned import alloc_pinned_tensor

    shapes, quant_format = ftw_expert_bank_shapes(ftw_dir)
    num_layers = 1 + max(max(v) for v in shapes.values())
    num_experts = next(iter(shapes.values()))[0][0][0]
    if host_slots > num_experts:
        raise ValueError(f"host_slots {host_slots} exceeds num_experts {num_experts}")

    sources: dict[str, list[torch.Tensor]] = {}
    total = 0
    for bank, per_layer in shapes.items():
        assert len(per_layer) == num_layers, (bank, len(per_layer), num_layers)
        rows = []
        for layer in range(num_layers):
            shape, dt = per_layer[layer]
            dtype = _DTYPE[dt]
            row_shape = tuple(shape[1:])
            n = host_slots * int(torch.Size(row_shape).numel())
            if pin:
                # exact-size cudaHostAlloc: torch's caching allocator would round these up
                nbytes = n * torch.empty((), dtype=dtype).element_size()
                t = alloc_pinned_tensor(nbytes, dtype=torch.uint8).view(dtype).view(
                    host_slots, *row_shape)
            else:
                t = torch.empty((host_slots, *row_shape), dtype=dtype)
            t.zero_()
            rows.append(t)
            total += t.numel() * t.element_size()
        sources[bank] = rows

    logger.info_rank0(
        f"tier 3: host pool {host_slots}/{num_experts} experts per layer "
        f"({100 * host_slots / num_experts:.1f}% residency), {total / 2**30:.2f} GiB pinned, "
        f"{num_layers} layers, format {quant_format}"
    )
    return ExpertBanks(quant_format=quant_format or "nvfp4", sources=sources)
