from __future__ import annotations

import ctypes
import mmap
import os

import pytest
import torch

from freetoken.checkpoint.ftw import FTWWriter, layer_bank_entry_name
from freetoken.tier3.nvme_store import ALIGN, NvmeExpertStore, UnalignedRead

# Row bytes chosen to be whole blocks, like the real packed NVFP4 banks.
LAYERS, EXPERTS = 3, 8
FEAT = {"gate_up_packed": 4 * ALIGN, "down_packed": 2 * ALIGN}


def _make_ftw(tmp_path, shard_limit: int) -> tuple[str, dict]:
    """A miniature FTW whose expert rows carry a known, position-dependent pattern."""
    out = str(tmp_path / "ftw")
    os.makedirs(out, exist_ok=True)
    w = FTWWriter(out, shard_limit=shard_limit)
    truth: dict[tuple[str, int, int], torch.Tensor] = {}
    # a dense weight first, so the bank entries do not start at global offset 0
    w.add_tensor("model.norm.weight", torch.arange(64, dtype=torch.uint8), kind="weight")
    for layer in range(LAYERS):
        for bank, feat in FEAT.items():
            rows = []
            for e in range(EXPERTS):
                seed = (hash((bank, layer, e)) & 0xFF) or 1
                row = torch.full((feat,), seed, dtype=torch.uint8)
                row[0] = layer
                row[1] = e
                row[2] = len(bank)
                rows.append(row)
                truth[(bank, layer, e)] = row
            w.add_tensor(layer_bank_entry_name(bank, layer),
                         torch.stack(rows), kind="experts_bank")
    w.finalize({"quant_format": "test"})
    return out, truth


def _slot(nbytes: int):
    """A page-aligned destination, as a real pinned host slot would be."""
    m = mmap.mmap(-1, nbytes)
    mv = memoryview(m)
    assert ctypes.addressof(ctypes.c_char.from_buffer(mv)) % ALIGN == 0
    return m, mv


@pytest.fixture
def store_and_truth(tmp_path):
    # 64 KiB shards force the bank entries to straddle shard boundaries
    out, truth = _make_ftw(tmp_path, shard_limit=64 * 1024)
    st = NvmeExpertStore(out, list(FEAT))
    yield st, truth
    st.close()


# ---------------------------------------------------------------- addressing

def test_geometry_is_read_back_from_the_index(store_and_truth):
    st, _ = store_and_truth
    assert st.num_layers == LAYERS
    assert st.num_experts == EXPERTS
    assert st.feat_bytes == FEAT
    assert st.bytes_per_expert == sum(FEAT.values())


def test_every_piece_is_block_aligned_even_across_shard_boundaries(store_and_truth):
    st, _ = store_and_truth
    straddled = 0
    for layer in range(LAYERS):
        for e in range(EXPERTS):
            for bank in FEAT:
                pieces = st.pieces(layer, e, bank)
                straddled += len(pieces) > 1
                assert sum(p.nbytes for p in pieces) == FEAT[bank]
                for p in pieces:
                    assert p.file_offset % ALIGN == 0, p
                    assert p.nbytes % ALIGN == 0, p
    assert straddled > 0, "fixture never straddled a shard; it is not exercising the split"


# ---------------------------------------------------------------- round trip

def test_every_expert_row_round_trips_byte_exactly(store_and_truth):
    st, truth = store_and_truth
    for layer in range(LAYERS):
        for e in range(EXPERTS):
            for bank, feat in FEAT.items():
                m, mv = _slot(feat)
                try:
                    assert st.read_bank(layer, e, bank, mv) == feat
                    got = torch.frombuffer(bytearray(mv), dtype=torch.uint8)
                    assert torch.equal(got, truth[(bank, layer, e)]), (bank, layer, e)
                finally:
                    mv.release()
                    m.close()
    assert st.bounce_reads == 0, "a read fell back to a bounce buffer"


def test_read_expert_fills_every_bank(store_and_truth):
    st, truth = store_and_truth
    bufs = {b: _slot(f) for b, f in FEAT.items()}
    try:
        n = st.read_expert(2, 5, {b: mv for b, (_, mv) in bufs.items()})
        assert n == st.bytes_per_expert
        for b, (_, mv) in bufs.items():
            got = torch.frombuffer(bytearray(mv), dtype=torch.uint8)
            assert torch.equal(got, truth[(b, 2, 5)])
    finally:
        for m, mv in bufs.values():
            mv.release()
            m.close()


def test_parallel_batch_matches_serial(store_and_truth):
    st, truth = store_and_truth
    reqs, keep = [], []
    for layer in range(LAYERS):
        for e in range(EXPERTS):
            bufs = {b: _slot(f) for b, f in FEAT.items()}
            keep.append((layer, e, bufs))
            reqs.append((layer, e, {b: mv for b, (_, mv) in bufs.items()}))
    try:
        assert st.read_experts(reqs) == len(reqs) * st.bytes_per_expert
        for layer, e, bufs in keep:
            for b, (_, mv) in bufs.items():
                got = torch.frombuffer(bytearray(mv), dtype=torch.uint8)
                assert torch.equal(got, truth[(b, layer, e)]), (b, layer, e)
    finally:
        for _, _, bufs in keep:
            for m, mv in bufs.values():
                mv.release()
                m.close()


# ---------------------------------------------------------------- guardrails

def test_unaligned_destination_is_rejected_not_bounced(store_and_truth):
    st, _ = store_and_truth
    feat = FEAT["down_packed"]
    m, mv = _slot(feat + ALIGN)
    try:
        with pytest.raises(UnalignedRead, match="page-aligned"):
            st.read_bank(0, 0, "down_packed", mv[1:feat + 1])
        assert st.bounce_reads == 1
    finally:
        mv.release()
        m.close()


def test_wrong_sized_destination_is_rejected(store_and_truth):
    st, _ = store_and_truth
    m, mv = _slot(ALIGN)
    try:
        with pytest.raises(ValueError, match="destination is"):
            st.read_bank(0, 0, "gate_up_packed", mv)
    finally:
        mv.release()
        m.close()


def test_an_unaligned_bank_is_refused_at_construction(tmp_path):
    """budget.host_bank_split is supposed to exclude these; if the two disagree, say so loudly
    rather than silently paying a memcpy per expert on the decode path."""
    out = str(tmp_path / "bad")
    os.makedirs(out)
    w = FTWWriter(out, shard_limit=1 << 20)
    w.add_tensor(layer_bank_entry_name("odd", 0),
                 torch.zeros((4, 100), dtype=torch.uint8), kind="experts_bank")
    w.finalize({"quant_format": "test"})
    with pytest.raises(UnalignedRead, match="not a"):
        NvmeExpertStore(out, ["odd"])


def test_a_checkpoint_without_per_layer_entries_is_refused(tmp_path):
    out = str(tmp_path / "flat")
    os.makedirs(out)
    w = FTWWriter(out, shard_limit=1 << 20)
    w.add_tensor("gate_up_packed", torch.zeros((4, ALIGN), dtype=torch.uint8), kind="experts_bank")
    w.finalize({"quant_format": "test"})
    with pytest.raises(ValueError, match="per-layer"):
        NvmeExpertStore(out, ["gate_up_packed"])


def _open_fds() -> int:
    return len(os.listdir("/proc/self/fd"))


def test_descriptors_do_not_scale_with_reader_count(tmp_path):
    """Regression: the readers used to cache an fd per (thread, shard).

    With 64 PLE readers plus the expert pools that reached ~960 descriptors and killed a live
    server with `OSError: [Errno 24] Too many open files` -- during an unrelated JIT build, so
    the traceback pointed nowhere near here. pread carries its own offset, so one shared fd per
    shard is correct and the count must stay flat in the pool size.
    """
    out, _ = _make_ftw(tmp_path, shard_limit=64 * 1024)
    base = _open_fds()
    st = NvmeExpertStore(out, list(FEAT), readers=32)
    try:
        keep, reqs = [], []
        for layer in range(LAYERS):
            for e in range(EXPERTS):
                bufs = {b: _slot(f) for b, f in FEAT.items()}
                keep.append(bufs)
                reqs.append((layer, e, {b: mv for b, (_, mv) in bufs.items()}))
        st.read_experts(reqs)
        grew = _open_fds() - base
        n_shards = len(st._shards)
        assert grew <= n_shards + 8, (
            f"opened {grew} descriptors for {n_shards} shards with 32 readers")
    finally:
        for bufs in keep:
            for m, mv in bufs.values():
                mv.release()
                m.close()
        st.close()
    # close() must give them all back
    assert _open_fds() <= base + 2
