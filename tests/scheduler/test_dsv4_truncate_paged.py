"""Speculative over-extension on the DSV4 (swa_paged) manager: ``truncate_paged`` must return
BOTH the full page and its 1:1-bound window page when a verify that crossed a 128-page
boundary is rolled back, and must leave a non-crossing rollback alone. CPU, no model."""

from __future__ import annotations

import torch

from .test_dsv4_generic_manager import P, _lifecycle, _req, _stack


def _running_req(cm, cached_len: int, n_decode: int = 600):
    # A cold-prefilled request decoded up to ``cached_len`` (the next decode step pending at
    # that position: device_len == cached_len + 1), not finished.
    prompt = torch.arange(1, 200, dtype=torch.int32)
    req = _lifecycle(cm, _req(0, prompt, n_decode=n_decode), total_len=cached_len, finished=False)
    assert req.cached_len == cached_len and req.device_len == cached_len + 1
    return req


def _finish(cm, req, steps: int = 3) -> None:
    for i in range(steps):
        req.append_host(torch.tensor([7000 + i], dtype=torch.int32))
        req.decode_batch_idx = 100 + i
        cm.maybe_free_swa_out_of_window([req], forward_iter=100 + i)
        cm.allocate_paged([req])
        req.complete_one()
    cm.cache_req(req, finished=True)


def test_page_crossing_rollback_returns_full_and_window_pages():
    cm, pool, _pt = _stack()
    req = _running_req(cm, cached_len=P * 2 - 1)  # position 255: the verify crosses into page 2
    free_full = len(cm.free_slots)
    free_swa = pool.swa_available_size()

    req.begin_verify(5)  # device_len 256 -> 261
    cm.allocate_paged([req])
    assert len(cm.free_slots) == free_full - 1
    assert pool.swa_available_size() == free_swa - P  # the window page bound to it
    old_device_len = req.device_len

    req.commit_verify(1)  # only the anchor emitted: cached_len 256, device_len 257
    cm.truncate_paged(req, old_device_len)
    assert len(cm.free_slots) == free_full, "full page not returned"
    assert pool.swa_available_size() == free_swa, "window page leaked by the rollback"
    # The mapping for the freed page is gone; the next step re-allocates it cleanly.
    assert int(pool.translate_full_to_window(torch.tensor([old_device_len - 1]))[0]) == -1
    cm.allocate_paged([req])
    assert len(cm.free_slots) == free_full - 1 and pool.swa_available_size() == free_swa - P
    req.complete_one()
    _finish(cm, req)
    cm.check_integrity()


def test_page_crossing_accepted_run_keeps_the_page():
    cm, pool, _pt = _stack()
    req = _running_req(cm, cached_len=P * 2 - 1)
    free_full = len(cm.free_slots)
    free_swa = pool.swa_available_size()
    req.begin_verify(5)
    cm.allocate_paged([req])
    old_device_len = req.device_len
    req.commit_verify(4)  # cached_len 259: the crossed page stays live
    cm.truncate_paged(req, old_device_len)
    assert len(cm.free_slots) == free_full - 1
    assert pool.swa_available_size() == free_swa - P
    req.complete_one()
    _finish(cm, req)
    cm.check_integrity()


def test_in_page_rollback_frees_nothing():
    cm, pool, _pt = _stack()
    req = _running_req(cm, cached_len=P * 2 + 44)  # position 300, inside page 2
    free_full = len(cm.free_slots)
    free_swa = pool.swa_available_size()
    req.begin_verify(5)  # 301 -> 306, still inside page 2
    cm.allocate_paged([req])
    assert len(cm.free_slots) == free_full and pool.swa_available_size() == free_swa
    old_device_len = req.device_len
    req.commit_verify(2)
    cm.truncate_paged(req, old_device_len)
    assert len(cm.free_slots) == free_full and pool.swa_available_size() == free_swa
    req.complete_one()
    _finish(cm, req)
    cm.check_integrity()
