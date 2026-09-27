"""Engram row stores over a synthetic safetensors shard.

References: the shard is written by hand (u64 header length + JSON + raw bytes, a decoy tensor
first so the table offsets are non-zero), so ``resolve_engram_tables`` is checked against the
offsets we wrote — and, when safetensors+torch are importable, against ``safe_open``'s view of
the same tensors. ``SafetensorsRowStore`` (mmap, and pread where the OS has it) must return
exactly what the in-RAM ``ArrayRowStore`` returns for random ids with duplicates.
"""

from __future__ import annotations

import json
import os
import shutil
import struct

import numpy as np
import pytest

from freetoken.engram.staging import EngramFetch, EngramStagingRing, FetchDriver
from freetoken.engram.store import ArrayRowStore, SafetensorsRowStore
from freetoken.engram.tables import EngramTableLoc, read_safetensors_header, resolve_engram_tables

ROWS, DIM, SCALE_COLS = 4096, 256, 8
LAYER = 1
SHARD = "model-00047-of-00048.safetensors"


def random_table(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    vals = rng.integers(0, 256, size=(ROWS, DIM), dtype=np.uint8)
    vals[(vals == 0x7F) | (vals == 0xFF)] = 0x40  # avoid e4m3 NaN codes
    scales = rng.integers(100, 140, size=(ROWS, SCALE_COLS), dtype=np.uint8)
    return vals, scales


def write_shard(path: str, vals: np.ndarray, scales: np.ndarray, *, weight_dtype="F8_E4M3", scale_dtype="F8_E8M0") -> dict:
    """Hand-written safetensors: decoy bf16 tensor first, then the table + scale. Returns the
    absolute byte offsets we wrote."""
    decoy = np.arange(100 * 8, dtype=np.uint16).tobytes()
    w = vals.tobytes()
    s = scales.tobytes()
    header = {
        "__metadata__": {"format": "pt"},
        "decoy": {"dtype": "BF16", "shape": [100, 8], "data_offsets": [0, len(decoy)]},
        f"layers.{LAYER}.engram.embed.weight": {
            "dtype": weight_dtype,
            "shape": list(vals.shape),
            "data_offsets": [len(decoy), len(decoy) + len(w)],
        },
        f"layers.{LAYER}.engram.embed.scale": {
            "dtype": scale_dtype,
            "shape": list(scales.shape),
            "data_offsets": [len(decoy) + len(w), len(decoy) + len(w) + len(s)],
        },
    }
    hj = json.dumps(header).encode()
    hj += b" " * (-len(hj) % 8)  # safetensors pads the header to 8 bytes
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hj)))
        f.write(hj)
        f.write(decoy)
        f.write(w)
        f.write(s)
    base = 8 + len(hj)
    return {"weight": base + len(decoy), "scale": base + len(decoy) + len(w)}


def write_index(model_dir: str) -> None:
    os.makedirs(model_dir, exist_ok=True)
    idx = {"weight_map": {f"layers.{LAYER}.engram.embed.weight": SHARD, f"layers.{LAYER}.engram.embed.scale": SHARD, "decoy": SHARD}}
    with open(os.path.join(model_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(idx, f)


@pytest.fixture
def model(tmp_path):
    vals, scales = random_table(0)
    model_dir = str(tmp_path / "model")
    write_index(model_dir)
    offsets = write_shard(os.path.join(model_dir, SHARD), vals, scales)
    return model_dir, vals, scales, offsets


def test_header_and_offsets(model, monkeypatch):
    model_dir, vals, scales, offsets = model
    monkeypatch.delenv("FREETOKEN_ENGRAM_DIR", raising=False)
    header_len, header = read_safetensors_header(os.path.join(model_dir, SHARD))
    assert header_len % 8 == 0 and set(header) >= {"decoy", f"layers.{LAYER}.engram.embed.weight"}
    locs = resolve_engram_tables(model_dir, [LAYER], expected_rows={LAYER: ROWS}, expected_dim=DIM)
    loc = locs[LAYER]
    assert isinstance(loc, EngramTableLoc)
    assert (loc.rows, loc.dim, loc.scale_cols, loc.block) == (ROWS, DIM, SCALE_COLS, 32)
    assert loc.weight_offset == offsets["weight"] and loc.scale_offset == offsets["scale"]
    assert loc.path == loc.scale_path == os.path.join(model_dir, SHARD) and loc.layer_id == LAYER
    # the offsets really address the rows we wrote
    with open(loc.path, "rb") as f:
        for rid in (0, 1, 777, ROWS - 1):
            f.seek(loc.weight_offset + rid * DIM)
            assert f.read(DIM) == vals[rid].tobytes()
            f.seek(loc.scale_offset + rid * SCALE_COLS)
            assert f.read(SCALE_COLS) == scales[rid].tobytes()


def test_offsets_agree_with_safetensors_library(model):
    model_dir, vals, scales, _ = model
    st = pytest.importorskip("safetensors")
    torch = pytest.importorskip("torch")
    loc = resolve_engram_tables(model_dir, [LAYER], engram_dir="")[LAYER]
    with st.safe_open(loc.path, framework="pt", device="cpu") as f:
        w = f.get_tensor(f"layers.{LAYER}.engram.embed.weight")
        assert w.dtype == torch.float8_e4m3fn and tuple(w.shape) == (ROWS, DIM)
        np.testing.assert_array_equal(w.view(torch.uint8).numpy(), vals)
        s = f.get_tensor(f"layers.{LAYER}.engram.embed.scale")
        np.testing.assert_array_equal(s.view(torch.uint8).numpy(), scales)


def test_resolve_validation(model, tmp_path, monkeypatch):
    model_dir, vals, scales, _ = model
    monkeypatch.delenv("FREETOKEN_ENGRAM_DIR", raising=False)
    with pytest.raises(ValueError, match="rows"):
        resolve_engram_tables(model_dir, [LAYER], expected_rows={LAYER: ROWS + 1})
    with pytest.raises(ValueError, match="dim"):
        resolve_engram_tables(model_dir, [LAYER], expected_dim=DIM * 2)
    with pytest.raises(KeyError):
        resolve_engram_tables(model_dir, [LAYER, 14])
    bad = str(tmp_path / "bad")
    write_index(bad)
    write_shard(os.path.join(bad, SHARD), vals, scales, weight_dtype="BF16")
    with pytest.raises(ValueError, match="F8_E4M3"):
        resolve_engram_tables(bad, [LAYER])
    # truncated shard
    trunc = str(tmp_path / "trunc")
    write_index(trunc)
    write_shard(os.path.join(trunc, SHARD), vals, scales)
    with open(os.path.join(trunc, SHARD), "r+b") as f:
        f.truncate(os.path.getsize(os.path.join(trunc, SHARD)) - 100)
    with pytest.raises(ValueError, match="truncated"):
        resolve_engram_tables(trunc, [LAYER])
    # missing shard
    empty = str(tmp_path / "empty")
    write_index(empty)
    with pytest.raises(FileNotFoundError):
        resolve_engram_tables(empty, [LAYER])


def test_engram_dir_takes_precedence(model, tmp_path, monkeypatch):
    model_dir, vals, scales, _ = model
    alt = str(tmp_path / "engram-ext4")
    os.makedirs(alt)
    other_vals, other_scales = random_table(1)
    write_shard(os.path.join(alt, SHARD), other_vals, other_scales)
    loc = resolve_engram_tables(model_dir, [LAYER], engram_dir=alt)[LAYER]
    assert loc.path == os.path.join(alt, SHARD)
    monkeypatch.setenv("FREETOKEN_ENGRAM_DIR", alt)
    assert resolve_engram_tables(model_dir, [LAYER])[LAYER].path == os.path.join(alt, SHARD)
    # a dir without the shard falls back to the checkpoint
    monkeypatch.setenv("FREETOKEN_ENGRAM_DIR", str(tmp_path / "nowhere"))
    assert resolve_engram_tables(model_dir, [LAYER])[LAYER].path == os.path.join(model_dir, SHARD)
    with SafetensorsRowStore(loc, io="mmap", readers=2) as store:
        v = np.empty((3, DIM), np.uint8)
        s = np.empty((3, SCALE_COLS), np.uint8)
        store.read_rows(np.array([5, 5, 9]), v, s)
        np.testing.assert_array_equal(v, other_vals[[5, 5, 9]])


def _ids(seed: int, n: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, ROWS, size=n, dtype=np.int64)
    if n:
        ids[::7] = ids[0]  # plenty of duplicates
    return ids


IO_MODES = ["mmap"] + (["pread"] if hasattr(os, "pread") else [])


@pytest.mark.parametrize("io", IO_MODES)
@pytest.mark.parametrize("resident_scales", [True, False])
def test_store_matches_array_oracle(model, io, resident_scales):
    model_dir, vals, scales, _ = model
    loc = resolve_engram_tables(model_dir, [LAYER], engram_dir="")[LAYER]
    oracle = ArrayRowStore(vals, scales)
    store = SafetensorsRowStore(loc, io=io, readers=4, resident_scales=resident_scales, chunk_rows=97)
    try:
        for seed, n in ((1, 1), (2, 50), (3, 3000), (4, 0)):
            ids = _ids(seed, n)
            v1, s1 = np.empty((n, DIM), np.uint8), np.empty((n, SCALE_COLS), np.uint8)
            v2, s2 = np.empty((n, DIM), np.uint8), np.empty((n, SCALE_COLS), np.uint8)
            oracle.read_rows(ids, v1, s1)
            store.read_rows(ids, v2, s2)
            np.testing.assert_array_equal(v1, v2)
            np.testing.assert_array_equal(s1, s2)
        st = store.stats.snapshot()
        assert st["rows_requested"] == 1 + 50 + 3000
        assert st["rows_unique"] == sum(int(np.unique(_ids(s, n)).shape[0]) for s, n in ((1, 1), (2, 50), (3, 3000)))
        row_bytes = DIM + (0 if resident_scales else SCALE_COLS)
        expected_bytes = st["rows_unique"] * row_bytes + (ROWS * SCALE_COLS if resident_scales else 0)
        assert st["bytes_read"] == expected_bytes
        assert st["calls"] == 3 + (1 if resident_scales else 0)
        with pytest.raises(IndexError):
            store.read_rows(np.array([ROWS]), np.empty((1, DIM), np.uint8), np.empty((1, SCALE_COLS), np.uint8))
        with pytest.raises(ValueError):
            store.read_rows(np.array([1]), np.empty((1, DIM + 1), np.uint8), np.empty((1, SCALE_COLS), np.uint8))
        store.warm(_ids(9, 200))
    finally:
        store.close()
    with pytest.raises(RuntimeError):
        store.read_rows(np.array([1]), np.empty((1, DIM), np.uint8), np.empty((1, SCALE_COLS), np.uint8))


@pytest.mark.parametrize("io", IO_MODES)
@pytest.mark.parametrize("resident_scales", [True, False])
def test_distinct_unsorted_ids_come_back_in_request_order(model, io, resident_scales):
    """All-distinct requests are the common case (a token's 24 hash columns, a chunk's rows
    rarely repeat an n-gram): the gather runs over the SORTED unique ids and must be put
    back in request order. The aliasing shortcut for ``U == R`` skipped that and returned
    the rows sorted by id -- every Engram lookup on the real checkpoint was permuted unless
    the chunk happened to contain a duplicate (found on the 5090, 2026-09-10)."""
    model_dir, vals, scales, _ = model
    loc = resolve_engram_tables(model_dir, [LAYER], engram_dir="")[LAYER]
    oracle = ArrayRowStore(vals, scales)
    rng = np.random.default_rng(11)
    with SafetensorsRowStore(loc, io=io, readers=3, resident_scales=resident_scales, chunk_rows=5) as store:
        for ids in (
            rng.permutation(ROWS)[:48].astype(np.int64),          # unsorted, all distinct
            np.sort(rng.permutation(ROWS)[:20]).astype(np.int64),  # sorted, all distinct (alias path)
            np.array([7, 3, 3, 9, 1], dtype=np.int64),              # duplicates, unsorted
        ):
            n = ids.shape[0]
            v1, s1 = np.empty((n, DIM), np.uint8), np.empty((n, SCALE_COLS), np.uint8)
            v2, s2 = np.empty((n, DIM), np.uint8), np.empty((n, SCALE_COLS), np.uint8)
            oracle.read_rows(ids, v1, s1)
            store.read_rows(ids, v2, s2)
            np.testing.assert_array_equal(v1, v2)
            np.testing.assert_array_equal(s1, s2)


def test_staging_ring_reuse_and_fetch(model):
    model_dir, vals, scales, _ = model
    loc = resolve_engram_tables(model_dir, [LAYER], engram_dir="")[LAYER]
    store = SafetensorsRowStore(loc, io="mmap", readers=2)
    ring = EngramStagingRing(2, cap_rows=64, dim=DIM, scale_cols=SCALE_COLS, pin=False)
    driver = FetchDriver(2)
    try:
        f1 = driver.submit(ring, store, _ids(11, 40))
        f2 = driver.submit(ring, store, _ids(12, 64))
        assert {f1.buffer_index, f2.buffer_index} == {0, 1}
        v, s = f1.wait()
        np.testing.assert_array_equal(v, vals[f1.row_ids])
        np.testing.assert_array_equal(s, scales[f1.row_ids])
        assert v.shape == (40, DIM) and f1.done() and f1.stats.wall_s is not None
        assert f1.stats.n_rows == 40 and f1.stats.n_unique == int(np.unique(f1.row_ids).shape[0])
        # both slots are owned by unreleased fetches: a third acquire must refuse, not overwrite
        with pytest.raises(RuntimeError, match="unreleased"):
            driver.submit(ring, store, _ids(13, 8))
        f1.release()
        f3 = driver.submit(ring, store, _ids(13, 8))
        assert f3.buffer_index == f1.buffer_index
        v3, _ = f3.wait()
        np.testing.assert_array_equal(v3, vals[f3.row_ids])
        with pytest.raises(ValueError):
            driver.submit(ring, store, _ids(14, 65))
        f2.wait()
        f2.release()
        f3.release()
        # multi-store fetch lays the parts out consecutively
        a, b = _ids(15, 10), _ids(16, 20)
        fm = driver.submit_multi(ring, [(store, a), (ArrayRowStore(vals, scales), b)])
        vm, sm = fm.wait()
        np.testing.assert_array_equal(vm, vals[np.concatenate([a, b])])
        np.testing.assert_array_equal(sm, scales[np.concatenate([a, b])])
        assert isinstance(fm, EngramFetch) and ring.acquires == 4  # the two refused acquires do not count
        # an empty fetch is legal and immediately done
        fe = driver.submit(ring, store, np.zeros(0, dtype=np.int64))
        assert fe.wait()[0].shape == (0, DIM) and fe.done()
    finally:
        driver.close()
        store.close()


def test_store_survives_shard_copy_on_other_dir(model, tmp_path):
    """The store reads through the resolved path, so a shard moved to FREETOKEN_ENGRAM_DIR (ext4
    copy) serves the same bytes as the checkpoint's."""
    model_dir, vals, scales, _ = model
    alt = str(tmp_path / "alt")
    os.makedirs(alt)
    shutil.copy(os.path.join(model_dir, SHARD), os.path.join(alt, SHARD))
    loc = resolve_engram_tables(model_dir, [LAYER], engram_dir=alt)[LAYER]
    with SafetensorsRowStore(loc, io="mmap", readers=1, resident_scales=False) as store:
        ids = _ids(20, 100)
        v, s = np.empty((100, DIM), np.uint8), np.empty((100, SCALE_COLS), np.uint8)
        store.read_rows(ids, v, s)
        np.testing.assert_array_equal(v, vals[ids])
        np.testing.assert_array_equal(s, scales[ids])
