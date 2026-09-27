"""Host-RAM preflight arithmetic (engine/host_preflight.py). CPU only.

Budgets are computed by hand from the serving box's numbers (377 GB host, 269 GiB of
tower banks, 6.7 GiB of DSpark banks) so a wrong sign or a dropped term fails here."""

from __future__ import annotations

import pytest

from freetoken.engine.host_preflight import (
    DEFAULT_RESERVE_BYTES,
    plan_host_ram,
    read_meminfo,
)

GiB = 2**30
TOWER = 269 * GiB
DSPARK = int(6.72 * GiB)
TRANSIENT = 3 * 11 * GiB  # (prefetch 2 + 1) x an 11 GiB expert shard
TOTAL = 351 * GiB


def test_serving_box_passes_with_page_cache_warning_only():
    plan = plan_host_ram(TOWER, DSPARK, TRANSIENT, mem_available=340 * GiB, mem_total=TOTAL, is_wsl=False)
    assert plan.ok and plan.refuse_reason is None
    # headroom = available - pinned - reserve (the transient is freed after boot)
    assert plan.page_cache_headroom == 340 * GiB - TOWER - DSPARK - DEFAULT_RESERVE_BYTES
    assert plan.page_cache_headroom > 24 * GiB
    assert plan.warnings == ()


def test_refuses_when_the_boot_peak_exceeds_available():
    need = TOWER + DSPARK + TRANSIENT + DEFAULT_RESERVE_BYTES
    ok = plan_host_ram(TOWER, DSPARK, TRANSIENT, need, TOTAL, is_wsl=False)
    assert ok.ok
    short = plan_host_ram(TOWER, DSPARK, TRANSIENT, need - 1, TOTAL, is_wsl=False)
    assert not short.ok
    # The refusal names every term and the override.
    for token in ("269.0 GiB", "6.7 GiB", "33.0 GiB", "4.0 GiB", "MemAvailable", "FREETOKEN_HOST_RAM_PREFLIGHT=0"):
        assert token in short.refuse_reason, token


def test_page_cache_headroom_warning_threshold():
    avail = TOWER + DSPARK + DEFAULT_RESERVE_BYTES + 10 * GiB  # transient 0: serial path
    plan = plan_host_ram(TOWER, DSPARK, 0, avail, TOTAL, is_wsl=False)
    assert plan.ok
    assert plan.page_cache_headroom == 10 * GiB
    assert len(plan.warnings) == 1 and "10.0 GiB" in plan.warnings[0] and "24.0 GiB" in plan.warnings[0]
    # A custom threshold below the headroom silences it.
    assert plan_host_ram(TOWER, DSPARK, 0, avail, TOTAL, False, min_page_cache_bytes=8 * GiB).warnings == ()


def test_wsl_refused_unless_allowed():
    avail = 340 * GiB
    plan = plan_host_ram(TOWER, DSPARK, TRANSIENT, avail, TOTAL, is_wsl=True)
    assert not plan.ok and "WSL2" in plan.refuse_reason and "FREETOKEN_ALLOW_WSL=1" in plan.refuse_reason
    allowed = plan_host_ram(TOWER, DSPARK, TRANSIENT, avail, TOTAL, is_wsl=True, allow_wsl=True)
    assert allowed.ok
    # Under WSL a short budget also says so (the VM sees a fraction of the host RAM).
    short = plan_host_ram(TOWER, DSPARK, TRANSIENT, 188 * GiB, 188 * GiB, is_wsl=True, allow_wsl=True)
    assert not short.ok and ".wslconfig" in short.refuse_reason


def test_pinned_over_ninety_percent_warns():
    total = 300 * GiB
    plan = plan_host_ram(TOWER, DSPARK, 0, 299 * GiB, total, is_wsl=False)
    assert plan.ok
    assert any("90%" in w for w in plan.warnings)


def test_small_model_no_warnings():
    plan = plan_host_ram(20 * GiB, 0, 6 * GiB, 120 * GiB, 128 * GiB, is_wsl=False)
    assert plan.ok and plan.warnings == ()
    assert plan.page_cache_headroom == 120 * GiB - 20 * GiB - DEFAULT_RESERVE_BYTES


def test_read_meminfo_shape():
    mem = read_meminfo()
    if mem is None:
        pytest.skip("no /proc/meminfo on this platform")
    assert set(mem) == {"MemTotal", "MemAvailable"}
    assert 0 < mem["MemAvailable"] <= mem["MemTotal"]


def test_engine_preflight_is_inert_without_meminfo(monkeypatch):
    """The engine hook must do nothing where /proc/meminfo is unavailable."""
    from types import SimpleNamespace

    from freetoken.engine import host_preflight
    from freetoken.engine.engine import Engine

    monkeypatch.setattr(host_preflight, "read_meminfo", lambda: None)
    cfg = SimpleNamespace(model_path="/nowhere", model_config=SimpleNamespace(), use_dummy_weight=True)
    Engine._host_ram_preflight(SimpleNamespace(), cfg, None)  # no raise, nothing touched


def test_engine_preflight_refuses_and_env_override(monkeypatch):
    from types import SimpleNamespace

    from freetoken.engine import host_preflight
    from freetoken.engine.engine import Engine
    from freetoken.moe import expert_banks

    monkeypatch.setattr(host_preflight, "read_meminfo", lambda: {"MemTotal": 64 * GiB, "MemAvailable": 50 * GiB})
    monkeypatch.setattr(host_preflight, "is_wsl", lambda: False)
    monkeypatch.setattr(expert_banks, "ftw_bank_bytes", lambda p: None)
    monkeypatch.setattr(expert_banks, "bank_bytes_estimate", lambda mc: 100 * GiB)
    monkeypatch.setattr(expert_banks, "parallel_reader_transient_bytes", lambda p: 0)
    cfg = SimpleNamespace(model_path="/nowhere", model_config=SimpleNamespace(dsv4_args=None), use_dummy_weight=False)
    monkeypatch.delenv("FREETOKEN_HOST_RAM_PREFLIGHT", raising=False)
    with pytest.raises(RuntimeError, match="host RAM preflight"):
        Engine._host_ram_preflight(SimpleNamespace(), cfg, None)
    monkeypatch.setenv("FREETOKEN_HOST_RAM_PREFLIGHT", "0")
    Engine._host_ram_preflight(SimpleNamespace(), cfg, None)  # warns instead
