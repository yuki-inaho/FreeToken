"""Adaptive draft count: the acceptance EMA and the next-k policy (pure host logic)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import freetoken.engine.spec as spec_mod
from freetoken.engine.spec import SpecExecutor, update_accept_ema


def _executor(k=2, k_max=3):
    ex = SimpleNamespace(k=k, k_max=k_max)
    ex._next_k = SpecExecutor._next_k.__get__(ex)
    return ex


def _req(staged=0, ema=None):
    return SimpleNamespace(spec_k_staged=staged, spec_ema=ema)


def test_ema_tracks_acceptance_fraction(monkeypatch):
    monkeypatch.setattr(spec_mod, "_ADAPT_ALPHA", 0.5)
    r = _req()
    update_accept_ema(r, 2, 2)          # both drafts accepted -> 1.0
    assert r.spec_ema == 1.0
    update_accept_ema(r, 2, 0)          # none -> halfway down
    assert r.spec_ema == pytest.approx(0.5)
    update_accept_ema(r, 3, 3)
    assert r.spec_ema == pytest.approx(0.75)
    update_accept_ema(r, 0, 0)          # a plain step (no drafts) leaves it alone
    assert r.spec_ema == pytest.approx(0.75)


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_next_k_fixed_when_adaptation_off(k):
    ex = _executor(k=k, k_max=k)
    assert ex._next_k(_req(staged=k, ema=1.0)) == k
    assert ex._next_k(_req(staged=0, ema=0.0)) == k


def test_next_k_steps_with_hysteresis(monkeypatch):
    monkeypatch.setattr(spec_mod, "_ADAPT_UP", 0.70)
    monkeypatch.setattr(spec_mod, "_ADAPT_DOWN", 0.55)
    ex = _executor(k=2, k_max=3)
    assert ex._next_k(_req(staged=0, ema=None)) == 2      # no history: base
    assert ex._next_k(_req(staged=2, ema=0.69)) == 2      # below UP: stay
    assert ex._next_k(_req(staged=2, ema=0.70)) == 3      # at UP: step up
    assert ex._next_k(_req(staged=3, ema=0.60)) == 3      # between: stay (hysteresis)
    assert ex._next_k(_req(staged=3, ema=0.54)) == 2      # below DOWN: step down
    assert ex._next_k(_req(staged=3, ema=0.99)) == 3      # capped at k_max
    assert ex._next_k(_req(staged=2, ema=0.10)) == 2      # floored at the base k


@pytest.mark.parametrize("ema", [None, 0.1, 0.6, 0.99])
def test_next_k_recovers_after_temporary_page_boundary_clip(ema):
    ex = _executor(k=2, k_max=3)
    assert ex._next_k(_req(staged=1, ema=ema)) >= 2
