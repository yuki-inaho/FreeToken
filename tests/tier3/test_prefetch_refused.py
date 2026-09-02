"""Tier 3 must refuse --moe-prefetch at config time, not crash mid-decode.

The lookahead prefetcher addresses host bank rows by EXPERT ID. Under tier 3 a layer's pool
holds host_slots < num_experts rows and the demand path translates through row_of() /
host_slot_tbl; the prefetch admission path has no such indirection, so it reads past the pool
and the decode dies on a CUDA illegal memory access several layers later -- a crash with no
connection to the flag that caused it.
"""

from __future__ import annotations

import pytest


def _cfg(tmp_path, prefetch: str):
    from types import SimpleNamespace

    return SimpleNamespace(
        moe_prefetch=prefetch, moe_backend="hybrid", spec_enabled=True,
        moe_prefetch_budget=-1, moe_prefetch_topk=4,
    )


@pytest.mark.parametrize("prefetch", ["layer", "mtp"])
def test_tier3_refuses_moe_prefetch(monkeypatch, tmp_path, prefetch):
    from freetoken.engine.engine import _tier3_request

    monkeypatch.setenv("FREETOKEN_TIER3_FTW", str(tmp_path))
    monkeypatch.setenv("FREETOKEN_TIER3_HOST_SLOTS", "64")
    cfg = _cfg(tmp_path, prefetch)
    # The helper itself must report tier 3 as requested for the guard to fire.
    assert _tier3_request(cfg) is not None

    import freetoken.engine.engine as eng

    src = __import__("inspect").getsource(eng._adjust_config)
    assert "_tier3_request(config) is not None" in src, (
        "the --moe-prefetch guard is gone from _adjust_config"
    )
    assert "is not supported with the NVMe expert" in src
    # The refusal must be reachable by default: the opt-in is measurement-only.
    assert "_tier3_prefetch_optin()" in src


def test_prefetch_optin_is_off_unless_explicitly_set(monkeypatch):
    """FREETOKEN_TIER3_PREFETCH gates a numerically unsafe path, so anything other than an
    explicit 1/true must leave the refusal in place."""
    from freetoken.engine.engine import _tier3_prefetch_optin

    monkeypatch.delenv("FREETOKEN_TIER3_PREFETCH", raising=False)
    assert not _tier3_prefetch_optin()
    for off in ("", "0", "no", "off", "yes"):
        monkeypatch.setenv("FREETOKEN_TIER3_PREFETCH", off)
        assert not _tier3_prefetch_optin(), off
    for on in ("1", "true", "True"):
        monkeypatch.setenv("FREETOKEN_TIER3_PREFETCH", on)
        assert _tier3_prefetch_optin(), on


def test_tier3_allows_prefetch_off(monkeypatch, tmp_path):
    """The supported combination must stay supported."""
    from freetoken.engine.engine import _tier3_request

    monkeypatch.setenv("FREETOKEN_TIER3_FTW", str(tmp_path))
    monkeypatch.setenv("FREETOKEN_TIER3_HOST_SLOTS", "64")
    assert _tier3_request(_cfg(tmp_path, "off")) is not None
