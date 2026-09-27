"""The CED re-encode extension the DSV4 backend applies to V4.1 prefill segments (CPU).

Decoder SWA Bounded Replay needs the encoder output of the last ``window`` prompt tokens.
``_extend_ced_segments`` therefore extends a continuation shorter than the window backwards
to a page boundary at least ``window`` tokens before its end, rebuilds the batch's token /
position streams from the host prompt, and refuses when the window pages before the new
start are no longer retained. Expectations are hand-derived from that rule with P = win = 128.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.attention.dsv4_sparse import DSV4SparseAttnBackend

P = 128


class _Backend(DSV4SparseAttnBackend):
    """The extension logic with the pool/device seams stubbed (no CUDA, no engine ctx)."""

    def __init__(self, args, retained_from: int):
        self.config = SimpleNamespace(dsv4_args=args)
        self.device = torch.device("cpu")
        self._pool = SimpleNamespace(P=P)
        self._retained_from = retained_from  # positions >= this still translate to a slot

    @property
    def pool(self):
        return self._pool

    def window_slots_of(self, ti, lo, hi):
        pos = torch.arange(lo, hi)
        return torch.where(pos >= self._retained_from, pos, torch.full_like(pos, -1))


def _args():
    from freetoken.models.deepseek_v41.args import DeepseekV41Args

    return DeepseekV41Args(window_size=P, max_seq_len=1024)


def _batch(reqs):
    b = SimpleNamespace(reqs=reqs)
    ids = torch.cat([r.input_ids[r.cached_len : r.cached_len + r.extend_len] for r in reqs])
    b.input_ids = ids.to(torch.int32)
    b.positions = torch.cat(
        [torch.arange(r.cached_len, r.cached_len + r.extend_len) for r in reqs]
    ).to(torch.int32)
    return b


def _req(uid, total, cached, table_idx=0):
    ids = torch.arange(1000 + uid * 10000, 1000 + uid * 10000 + total, dtype=torch.int32)
    return SimpleNamespace(
        uid=uid, input_ids=ids, cached_len=cached, extend_len=total - cached, table_idx=table_idx
    )


def _segments(reqs):
    segs, off, last = [], 0, []
    for r in reqs:
        segs.append((off, r.extend_len, r.table_idx, r.cached_len))
        off += r.extend_len
        last.append(off - 1)
    return segs, torch.tensor(last, dtype=torch.int32)


def test_long_suffix_is_left_alone():
    be = _Backend(_args(), retained_from=0)
    r = _req(1, total=300, cached=128)
    b = _batch([r])
    segs, last = _segments([r])
    segs2, extras, last2 = be._extend_ced_segments(b, segs, last)
    assert segs2 == segs and extras == [(172, True)] and torch.equal(last2, last)
    assert b.input_ids.numel() == 172


def test_short_suffix_extends_back_to_a_page_boundary_at_least_a_window_before_the_end():
    be = _Backend(_args(), retained_from=0)
    r = _req(1, total=296, cached=256)  # 40 new tokens; end - win = 168 -> page 128
    b = _batch([r])
    segs, last = _segments([r])
    segs2, extras, last2 = be._extend_ced_segments(b, segs, last)
    assert segs2 == [(0, 168, 0, 128)]
    assert extras == [(40, True)]
    assert last2.tolist() == [167]
    assert b.input_ids.tolist() == r.input_ids[128:296].tolist()
    assert b.positions.tolist() == list(range(128, 296))


def test_extension_never_goes_before_the_cached_prefix_start_or_below_zero():
    be = _Backend(_args(), retained_from=0)
    r = _req(2, total=100, cached=60)  # end - win < 0 -> restart from 0
    b = _batch([r])
    segs, last = _segments([r])
    segs2, extras, _ = be._extend_ced_segments(b, segs, last)
    assert segs2 == [(0, 100, 0, 0)] and extras == [(40, True)]


def test_ragged_batch_rebuilds_every_stream_and_last_indices():
    be = _Backend(_args(), retained_from=0)
    a = _req(1, total=300, cached=128)   # long suffix: unchanged
    c = _req(2, total=296, cached=256, table_idx=1)  # short suffix: extended to 128
    b = _batch([a, c])
    segs, last = _segments([a, c])
    segs2, extras, last2 = be._extend_ced_segments(b, segs, last)
    assert segs2 == [(0, 172, 0, 128), (172, 168, 1, 128)]
    assert extras == [(172, True), (40, True)]
    assert last2.tolist() == [171, 339]
    assert b.input_ids.numel() == 340 and b.positions.numel() == 340
    assert b.positions[172:].tolist() == list(range(128, 296))


def test_missing_retention_before_the_extension_start_is_refused():
    be = _Backend(_args(), retained_from=64)  # positions < 64 slid out
    r = _req(3, total=296, cached=256)  # extends to 128, needs [1, 128) retained
    b = _batch([r])
    segs, last = _segments([r])
    with pytest.raises(RuntimeError, match="retained"):
        be._extend_ced_segments(b, segs, last)


def test_0731_args_take_the_full_path():
    from freetoken.models.deepseek_v4.args import DeepseekV4Args

    be = _Backend.__new__(_Backend)
    be.config = SimpleNamespace(dsv4_args=DeepseekV4Args())
    assert be._ced_prefill_mode() == "full"


def test_env_selects_the_mode(monkeypatch):
    be = _Backend(_args(), retained_from=0)
    monkeypatch.delenv("FREETOKEN_DSV41_PREFILL_MODE", raising=False)
    assert be._ced_prefill_mode() == "ced"
    monkeypatch.setenv("FREETOKEN_DSV41_PREFILL_MODE", "full")
    assert be._ced_prefill_mode() == "full"
