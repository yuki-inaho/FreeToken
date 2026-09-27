"""expert_banks._host_ram_fits_parallel: only the shards the bank loaders read count.

A DeepSeek-V4.1 checkpoint keeps 203 GB of Engram tables in two shards no bank loader
opens; summing every *.safetensors (the old rule) called the 377 GB serving box too small
and silently fell back to the serial reader. Fake index + sparse shard files; no torch."""

from __future__ import annotations

import json

import pytest

from freetoken.moe.expert_banks import (
    _PARALLEL_PREFETCH,
    _host_ram_fits_parallel,
    expert_shard_sizes,
    parallel_reader_transient_bytes,
)

GiB = 2**30


def _sparse(path, nbytes: int) -> None:
    with open(path, "wb") as f:
        f.truncate(nbytes)


@pytest.fixture
def checkpoint(tmp_path):
    """3 expert shards (5, 10, 8 GiB), one dense shard (1 GiB), one 100 GiB table shard."""
    weight_map = {
        "layers.0.ffn.experts.0.w1.weight": "model-00001.safetensors",
        "layers.0.attn.wq_a.weight": "model-00001.safetensors",
        "layers.1.ffn.experts.3.w2.scale": "model-00002.safetensors",
        "mtp.0.ffn.experts.7.w3.weight": "model-00003.safetensors",
        "embed.weight": "model-00004.safetensors",
        "layers.1.engram.embed.weight": "model-00005.safetensors",
    }
    sizes = {"model-00001.safetensors": 5 * GiB, "model-00002.safetensors": 10 * GiB,
             "model-00003.safetensors": 8 * GiB, "model-00004.safetensors": 1 * GiB,
             "model-00005.safetensors": 100 * GiB}
    for name, n in sizes.items():
        _sparse(tmp_path / name, n)
    with open(tmp_path / "model.safetensors.index.json", "w") as f:
        json.dump({"weight_map": weight_map}, f)
    return str(tmp_path)


def _meminfo(monkeypatch, avail_bytes: int) -> None:
    import builtins

    real_open = builtins.open

    class _F:
        def __init__(self):
            self.lines = [f"MemTotal:       {2 * avail_bytes // 1024} kB\n",
                          f"MemAvailable:   {avail_bytes // 1024} kB\n"]

        def __enter__(self):
            return iter(self.lines)

        def __exit__(self, *a):
            return False

    def fake_open(path, *a, **k):
        if path == "/proc/meminfo":
            return _F()
        return real_open(path, *a, **k)

    monkeypatch.setattr(builtins, "open", fake_open)


def test_expert_shard_sizes_ignores_non_expert_shards(checkpoint):
    sizes = expert_shard_sizes(checkpoint)
    assert sorted(sizes) == [5 * GiB, 8 * GiB, 10 * GiB]
    # expert_only=False restores the old every-shard sum.
    assert sorted(expert_shard_sizes(checkpoint, expert_only=False)) == [1 * GiB, 5 * GiB, 8 * GiB, 10 * GiB, 100 * GiB]
    # A custom predicate (a model's own) narrows further.
    only_tower = expert_shard_sizes(checkpoint, lambda n: n.startswith("layers.") and ".experts." in n)
    assert sorted(only_tower) == [5 * GiB, 10 * GiB]


def test_transient_is_prefetch_plus_one_largest_expert_shard(checkpoint):
    assert parallel_reader_transient_bytes(checkpoint) == (_PARALLEL_PREFETCH + 1) * 10 * GiB
    assert _PARALLEL_PREFETCH == 2  # mirrors iter_expert_tensors_parallel's default


def test_fits_counts_expert_shards_only(checkpoint, monkeypatch):
    need = (5 + 10 + 8) * GiB + (_PARALLEL_PREFETCH + 1) * 10 * GiB  # 53 GiB
    _meminfo(monkeypatch, need + 1024)
    assert _host_ram_fits_parallel(checkpoint)
    _meminfo(monkeypatch, need)
    assert not _host_ram_fits_parallel(checkpoint)  # strict >
    # The old all-shards rule would need > 124 + 300 GiB; at 60 GiB it says no, we say yes.
    _meminfo(monkeypatch, 60 * GiB)
    assert _host_ram_fits_parallel(checkpoint)
    assert not _host_ram_fits_parallel(checkpoint, expert_only=False)


def test_no_index_falls_back_to_every_shard(tmp_path, monkeypatch):
    _sparse(tmp_path / "model.safetensors", 4 * GiB)
    assert expert_shard_sizes(str(tmp_path)) == [4 * GiB]
    _meminfo(monkeypatch, 4 * GiB + (_PARALLEL_PREFETCH + 1) * 4 * GiB + 1024)
    assert _host_ram_fits_parallel(str(tmp_path))
    _meminfo(monkeypatch, 4 * GiB)
    assert not _host_ram_fits_parallel(str(tmp_path))


def test_unknown_paths_keep_the_fast_path(tmp_path, monkeypatch):
    _meminfo(monkeypatch, 1)
    assert expert_shard_sizes(str(tmp_path)) is None  # no shards at all
    assert _host_ram_fits_parallel(str(tmp_path))
    assert parallel_reader_transient_bytes(str(tmp_path)) == 0
