"""Engram (DeepSeek-V4.1) host side: hash-table layout, n-gram hashing, compressed token map and
on-NVMe row stores. Torch-free except for the optional pinned staging buffers (lazy import);
the device dequant and the model module live under ``kernel/`` and ``models/deepseek_v41/``.

Exports resolve lazily so ``python -m freetoken.engram.token_map`` does not import the module
twice (the package must not pre-import the submodule ``-m`` is about to execute).
"""

from __future__ import annotations

import importlib

_EXPORTS = {
    "ArrayRowStore": "store",
    "EngramFetch": "staging",
    "EngramHistory": "hashing",
    "EngramLayout": "layout",
    "EngramRowSource": "store",
    "EngramRuntime": "runtime",
    "EngramStagingRing": "staging",
    "EngramTableLoc": "tables",
    "FetchDriver": "staging",
    "NGramHasher": "hashing",
    "PrefillItem": "runtime",
    "RowStoreStats": "store",
    "SafetensorsRowStore": "store",
    "build_compressed_token_map": "token_map",
    "compute_hash_multipliers": "layout",
    "find_next_prime": "layout",
    "is_prime": "layout",
    "load_or_build_token_map": "token_map",
    "read_safetensors_header": "tables",
    "resolve_engram_tables": "tables",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f"{__name__}.{module}"), name)


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))
