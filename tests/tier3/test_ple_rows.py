from __future__ import annotations

import os

import pytest
import torch
from safetensors.torch import save_file

from freetoken.tier3.ple_rows import PleRowCache, read_ple_layout

SHARDS, ROWS, HEAD_DIM = 4, 64, 160
PREFIX = "model.language_model.layers.2.ple.ple_embedding.ngram_embedding"


def _make_ple(tmp_path) -> tuple[str, torch.Tensor]:
    """A miniature PLE checkpoint: `SHARDS` row-blocks plus the scalar weight_scale."""
    d = str(tmp_path / "ple")
    os.makedirs(d, exist_ok=True)
    table = torch.arange(SHARDS * ROWS * HEAD_DIM, dtype=torch.int64) % 251
    table = table.to(torch.uint8).view(SHARDS * ROWS, HEAD_DIM)
    for s in range(SHARDS):
        block = table[s * ROWS:(s + 1) * ROWS].view(torch.float8_e4m3fn)
        tensors = {f"{PREFIX}.shard_{s}.weight": block}
        if s == 0:
            tensors[f"{PREFIX}.weight_scale"] = torch.tensor(0.25)
        save_file(tensors, os.path.join(d, f"model-plefp8-{s:05d}.safetensors"))
    return d, table


@pytest.fixture
def ple(tmp_path):
    d, table = _make_ple(tmp_path)
    yield read_ple_layout(d), table


def test_layout_reconstructs_the_logical_table(ple):
    lay, _ = ple
    assert len(lay.paths) == SHARDS
    assert lay.rows_per_shard == ROWS
    assert lay.head_dim == HEAD_DIM
    assert lay.num_rows == SHARDS * ROWS
    assert lay.scale == 0.25


def test_locate_maps_a_global_row_to_the_right_shard(ple):
    lay, _ = ple
    p0, _ = lay.locate(0)
    p_last, _ = lay.locate(SHARDS * ROWS - 1)
    assert p0 == lay.paths[0] and p_last == lay.paths[-1]
    # rows within a shard are contiguous and head_dim apart
    _, a = lay.locate(5)
    _, b = lay.locate(6)
    assert b - a == HEAD_DIM


def test_every_row_round_trips_byte_exactly(ple):
    """The rows are not block-aligned, so this exercises the aligned-window + copy path for
    every offset phase within a block."""
    lay, table = ple
    cache = PleRowCache(lay, cache_rows=lay.num_rows, pin=False, readers=4)
    try:
        ids = torch.arange(lay.num_rows, dtype=torch.int64)
        slots = cache.resolve(ids)
        for rid, slot in zip(ids.tolist(), slots.tolist()):
            got = cache.weight[slot].view(torch.uint8)
            assert torch.equal(got, table[rid]), rid
    finally:
        cache.reader.close()


def test_repeat_lookups_hit_and_keep_the_same_slot(ple):
    lay, _ = ple
    cache = PleRowCache(lay, cache_rows=32, pin=False)
    try:
        ids = torch.tensor([1, 7, 9], dtype=torch.int64)
        first = cache.resolve(ids)
        reads = cache.reader.reads
        again = cache.resolve(ids)
        assert torch.equal(first, again)
        assert cache.reader.reads == reads, "a hit went to disk"
        assert cache.hits == 3
    finally:
        cache.reader.close()


def test_duplicate_ids_in_one_call_share_a_slot_and_one_read(ple):
    lay, _ = ple
    cache = PleRowCache(lay, cache_rows=32, pin=False)
    try:
        slots = cache.resolve(torch.tensor([5] * 6 + [6] * 6, dtype=torch.int64))
        assert len(set(slots.tolist())) == 2
        assert cache.reader.reads == 2
    finally:
        cache.reader.close()


def test_all_rows_of_one_call_stay_simultaneously_resident(ple):
    """The invariant a direct-mapped cache cannot give: a step's rows must not evict each
    other, or the gather returns another n-gram's embedding."""
    lay, table = ple
    n = 16
    cache = PleRowCache(lay, cache_rows=n, pin=False)   # exactly as many slots as the step needs
    try:
        ids = torch.arange(n, dtype=torch.int64)
        slots = cache.resolve(ids)
        assert len(set(slots.tolist())) == n
        for rid, slot in zip(ids.tolist(), slots.tolist()):
            assert torch.equal(cache.weight[slot].view(torch.uint8), table[rid]), rid
    finally:
        cache.reader.close()


def test_a_step_wider_than_the_cache_is_refused_not_corrupted(ple):
    lay, _ = ple
    cache = PleRowCache(lay, cache_rows=4, pin=False)
    try:
        with pytest.raises(RuntimeError, match="raise the PLE row cache size"):
            cache.resolve(torch.arange(8, dtype=torch.int64))
    finally:
        cache.reader.close()


def test_shape_is_preserved(ple):
    lay, _ = ple
    cache = PleRowCache(lay, cache_rows=64, pin=False)
    try:
        ids = torch.arange(12, dtype=torch.int64).view(3, 4)
        assert cache.resolve(ids).shape == (3, 4)
    finally:
        cache.reader.close()


def test_eviction_reuses_slots_and_keeps_the_tag_consistent(ple):
    lay, table = ple
    cache = PleRowCache(lay, cache_rows=4, pin=False)
    try:
        cache.resolve(torch.tensor([0, 1, 2, 3], dtype=torch.int64))
        cache.resolve(torch.tensor([4, 5, 6, 7], dtype=torch.int64))
        live = [int(t) for t in cache.tag if int(t) >= 0]
        assert sorted(live) == [4, 5, 6, 7]
        for rid in live:
            slot = cache._pos[rid]
            assert int(cache.tag[slot]) == rid
            assert torch.equal(cache.weight[slot].view(torch.uint8), table[rid])
    finally:
        cache.reader.close()


def test_ple_descriptors_do_not_scale_with_reader_count(ple):
    """Same regression as the expert store: 64 readers must not mean 64 fds per shard."""
    import os as _os

    lay, _ = ple
    base = len(_os.listdir("/proc/self/fd"))
    cache = PleRowCache(lay, cache_rows=lay.num_rows, pin=False, readers=32)
    try:
        cache.resolve(torch.arange(lay.num_rows, dtype=torch.int64))
        grew = len(_os.listdir("/proc/self/fd")) - base
        assert grew <= len(lay.paths) + 8, (
            f"opened {grew} descriptors for {len(lay.paths)} shards with 32 readers")
    finally:
        cache.reader.close()
