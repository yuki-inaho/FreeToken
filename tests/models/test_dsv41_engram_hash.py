"""DeepSeek-V4.1 Engram layout + hashing vs a verbatim copy of the reference ``inference/engram.py``.

Oracle: the reference ``find_next_prime`` / ``compute_hash_multipliers`` / ``EngramLayout`` /
``NgramHashState`` (torch, CPU) copied below; ``sympy.isprime`` replaced by trial division and
the token map injected instead of built from a tokenizer. Our numpy hasher must agree row for
row on whole sequences, chunked prefill, per-token decode and speculative verify rows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from freetoken.engram.hashing import EngramHistory, NGramHasher  # noqa: E402
from freetoken.engram.layout import EngramLayout, compute_hash_multipliers, find_next_prime, is_prime  # noqa: E402
from freetoken.engram.runtime import EngramRuntime, PrefillItem  # noqa: E402
from freetoken.engram.store import ArrayRowStore  # noqa: E402

# --------------------------------------------------------------------------- reference (verbatim)


def _isprime(n: int) -> bool:  # stands in for sympy.isprime
    if n < 2:
        return False
    i = 2
    while i * i <= n:
        if n % i == 0:
            return False
        i += 1
    return True


def ref_find_next_prime(start: int, seen_primes: set[int]) -> int:
    candidate = start + 1
    while not _isprime(candidate) or candidate in seen_primes:
        candidate += 1
    return candidate


def ref_compute_hash_multipliers(layer_ids, max_ngram_size: int, tokenizer_vocab_size: int) -> torch.Tensor:
    max_long = np.iinfo(np.int64).max
    multiplier_bound = max(1, (max_long // tokenizer_vocab_size) // 2)
    rows = []
    for layer_id in layer_ids:
        generator = np.random.default_rng(10007 * layer_id)
        values = generator.integers(
            low=0,
            high=multiplier_bound,
            size=(max_ngram_size,),
            dtype=np.int64,
        )
        rows.append(torch.tensor(values * 2 + 1))
    return torch.stack(rows)


@dataclass(frozen=True)
class RefEngramLayout:
    max_ngram_size: int
    layer_ids: tuple[int, ...]
    num_embeddings: tuple[int, ...]
    primes: tuple[tuple[tuple[int, ...], ...], ...]
    n_heads: int
    head_dim: int

    @classmethod
    def from_args(cls, args):
        layer_ids = tuple(args.engram_layer_ids)
        if not layer_ids:
            return None
        max_ngram_size, n_heads = args.engram_max_ngram_size, args.engram_n_heads
        primes, seen = [], set()
        for _ in layer_ids:
            per_ngram = []
            for _ in range(max_ngram_size - 1):
                sizes, current = [], args.engram_vocab_size - 1
                for _ in range(n_heads):
                    current = ref_find_next_prime(current, seen)
                    seen.add(current)
                    sizes.append(current)
                per_ngram.append(tuple(sizes))
            primes.append(tuple(per_ngram))
        return cls(
            max_ngram_size=max_ngram_size,
            layer_ids=layer_ids,
            num_embeddings=tuple(args.engram_num_embeddings),
            primes=tuple(primes),
            n_heads=n_heads,
            head_dim=args.engram_head_dim,
        )


def build_compressed_token_map(tokenizer):  # injected: the "tokenizer" is (token_map, size)
    return tokenizer


class RefNgramHashState(nn.Module):
    DEAD = -1

    def __init__(self, args, layout, tokenizer):
        super().__init__()
        self.layout = layout
        token_map, vocab_size = build_compressed_token_map(tokenizer)
        assert vocab_size == args.engram_compressed_vocab_size, (vocab_size, args.engram_compressed_vocab_size)
        self.pad_id = token_map[args.engram_pad_id]
        flat = [[p for per_ngram in layer for p in per_ngram] for layer in layout.primes]
        offsets = [np.cumsum([0, *sizes[:-1]]) for sizes in flat]
        multipliers = ref_compute_hash_multipliers(layout.layer_ids, layout.max_ngram_size, vocab_size)
        self.register_buffer("primes", torch.tensor(layout.primes), persistent=False)
        self.register_buffer("offsets", torch.tensor(np.array(offsets)), persistent=False)
        self.register_buffer("multipliers", multipliers, persistent=False)
        self.register_buffer("token_map", torch.tensor(token_map), persistent=False)
        self.register_buffer(
            "cache", torch.empty(args.max_batch_size, args.max_seq_len, dtype=torch.int64), persistent=False
        )

    @torch.inference_mode()
    def forward(self, input_ids: torch.Tensor, start_pos: int, token_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, seqlen = input_ids.shape
        compressed = self.token_map[input_ids]
        if token_mask is not None:
            compressed = torch.where(token_mask, compressed, self.DEAD)
        self.cache[:batch, start_pos : start_pos + seqlen] = compressed

        positions = torch.arange(start_pos, start_pos + seqlen, device=input_ids.device).expand(batch, seqlen)
        tokens, blocked = [], torch.zeros_like(positions, dtype=torch.bool)
        for shift in range(self.layout.max_ngram_size):
            source = self.cache[:batch].gather(1, (positions - shift).clamp_min(0))
            blocked = blocked | (positions < shift) | (source == self.DEAD)
            tokens.append(torch.where(blocked, self.pad_id, source))
        tokens = torch.stack(tokens, dim=-1)  # [B, L, max_ngram_size]

        products = tokens.unsqueeze(2) * self.multipliers  # [B, L, n_engram_layers, max_ngram_size]
        rolling, hashes = products[..., 0], []
        for i in range(1, self.layout.max_ngram_size):
            rolling = torch.bitwise_xor(rolling, products[..., i])
            hashes.append(rolling.unsqueeze(-1) % self.primes[:, i - 1])
        return torch.cat(hashes, dim=-1) + self.offsets


# --------------------------------------------------------------------------- fixtures


@dataclass
class TinyArgs:
    engram_layer_ids: tuple = (1, 3)
    engram_vocab_size: int = 101
    engram_max_ngram_size: int = 4
    engram_n_heads: int = 2
    engram_head_dim: int = 32
    engram_pad_id: int = 2
    engram_compressed_vocab_size: int = 37
    engram_num_embeddings: tuple = ()
    max_batch_size: int = 4
    max_seq_len: int = 512


@dataclass
class RealArgs:
    engram_layer_ids: tuple = (1, 14)
    engram_vocab_size: int = 16_000_000
    engram_num_embeddings: tuple = (384_006_168, 384_016_682)
    engram_max_ngram_size: int = 4
    engram_pad_id: int = 2
    engram_compressed_vocab_size: int = 99_092
    engram_n_heads: int = 8
    engram_head_dim: int = 256


VOCAB = 64  # raw token ids of the tiny model


def tiny_token_map(compressed: int, seed: int = 0) -> list[int]:
    """A fake compressed map covering every compressed id at least once (collapsing tokens)."""
    rng = np.random.default_rng(seed)
    m = rng.integers(0, compressed, size=VOCAB)
    m[:compressed] = np.arange(compressed)  # every compressed id is reachable
    return [int(x) for x in m]


def make_tiny():
    args = TinyArgs()
    tm = tiny_token_map(args.engram_compressed_vocab_size)
    layout = EngramLayout.from_args(args)
    ref = RefNgramHashState(args, RefEngramLayout.from_args(args), (tm, args.engram_compressed_vocab_size))
    mult = compute_hash_multipliers(layout.layer_ids, layout.max_ngram_size, args.engram_compressed_vocab_size)
    hasher = NGramHasher(layout, mult, tm[args.engram_pad_id])
    return args, layout, np.asarray(tm, dtype=np.int64), ref, hasher


def ref_rows(ref, token_ids: np.ndarray) -> np.ndarray:
    """Whole-sequence reference hashing: ``[T, n_layers, n_cols]``."""
    return ref(torch.tensor(token_ids, dtype=torch.int64)[None], 0)[0].numpy()


# --------------------------------------------------------------------------- layout


def test_is_prime_matches_trial_division():
    for n in range(0, 5000):
        assert is_prime(n) == _isprime(n), n
    for n in (15_999_989, 16_000_057, 16_000_129, 2**31 - 1, 2**61 - 1, (2**61 - 1) * 3):
        assert is_prime(n) == _isprime(n) if n < 10**12 else True


def test_find_next_prime_tiny_matches_reference():
    seen = set()
    ours, theirs = [], []
    for _ in range(20):
        p = find_next_prime(100, seen)
        seen.add(p)
        ours.append(p)
    seen = set()
    for _ in range(20):
        p = ref_find_next_prime(100, seen)
        seen.add(p)
        theirs.append(p)
    assert ours == theirs


def test_tiny_layout_primes_offsets_match_reference():
    args = TinyArgs()
    ours = EngramLayout.from_args(args)
    ref = RefEngramLayout.from_args(args)
    assert ours.primes == ref.primes
    assert ours.layer_ids == ref.layer_ids and ours.n_heads == ref.n_heads and ours.head_dim == ref.head_dim
    flat = [[p for per_ngram in layer for p in per_ngram] for layer in ref.primes]
    offsets = [list(np.cumsum([0, *sizes[:-1]])) for sizes in flat]
    assert [list(f) for f in ours.primes_flat] == flat
    assert [list(o) for o in ours.offsets] == offsets
    assert ours.n_hash_cols == (args.engram_max_ngram_size - 1) * args.engram_n_heads
    # the primes are disjoint across layers and n-gram sizes
    all_primes = [p for f in flat for p in f]
    assert len(set(all_primes)) == len(all_primes)


def test_real_layout_sum_identity_and_multipliers():
    """The primes of each layer must sum to the checkpoint's table rows, or every lookup is wrong."""
    args = RealArgs()
    layout = EngramLayout.from_args(args)  # raises on a mismatch
    for li, rows in enumerate(args.engram_num_embeddings):
        assert sum(layout.primes_flat[li]) == rows
    assert layout.n_hash_cols == 24 and layout.n_layers == 2
    # every prime really is prime and above the vocab, drawn in increasing order per (layer, n-gram)
    for per_layer in layout.primes:
        for per_ngram in per_layer:
            assert all(_isprime(p) for p in per_ngram)
            assert all(p >= args.engram_vocab_size for p in per_ngram)
            assert list(per_ngram) == sorted(per_ngram)
    assert layout.offsets[0][0] == 0 and layout.offsets[0][-1] + layout.primes_flat[0][-1] == args.engram_num_embeddings[0]
    # first prime of layer 1 is the first prime above 16M; layer 14 continues after layer 1's 24 primes
    assert layout.primes[0][0][0] == ref_find_next_prime(args.engram_vocab_size - 1, set())
    assert layout.primes[1][0][0] > layout.primes[0][-1][-1]

    ours = compute_hash_multipliers(args.engram_layer_ids, args.engram_max_ngram_size, args.engram_compressed_vocab_size)
    ref = ref_compute_hash_multipliers(args.engram_layer_ids, args.engram_max_ngram_size, args.engram_compressed_vocab_size)
    assert ours.dtype == np.int64
    np.testing.assert_array_equal(ours, ref.numpy())
    assert (ours % 2 == 1).all()
    # no int64 wrap for any compressed id: (vocab-1) * max multiplier < 2^63
    assert int(ours.max()) * (args.engram_compressed_vocab_size - 1) < 2**63


def test_real_layout_wrong_num_embeddings_is_loud():
    args = RealArgs(engram_num_embeddings=(384_006_168, 384_016_683))
    with pytest.raises(ValueError, match="layer 14"):
        EngramLayout.from_args(args)


def test_from_args_without_engram_layers_is_none():
    assert EngramLayout.from_args(TinyArgs(engram_layer_ids=())) is None


# --------------------------------------------------------------------------- hashing vs reference


def test_row_ids_whole_sequence_matches_reference():
    args, layout, tm, ref, hasher = make_tiny()
    rng = np.random.default_rng(1)
    for T in (1, 2, 3, 4, 5, 17, 200):
        ids = rng.integers(0, VOCAB, size=T)
        expect = ref_rows(ref, ids)
        got = hasher.row_ids(hasher.empty_history(), tm[ids])
        assert got.shape == (T, 2, 6) and got.dtype == np.int64
        np.testing.assert_array_equal(got, expect)
        # rows land inside their (layer, column) bucket range
        for li in range(layout.n_layers):
            lo = np.asarray(layout.offsets[li])
            hi = lo + np.asarray(layout.primes_flat[li])
            assert (got[:, li] >= lo).all() and (got[:, li] < hi).all()


def test_pad_semantics_at_sequence_start():
    """Positions 0..2 see the pad id for the missing lookbacks; the reference does that via
    ``positions < shift``, we do it via a pad-filled history — and both agree with hashing the
    same tokens after an explicit run of pad tokens (the pad id is a real compressed id)."""
    args, layout, tm, ref, hasher = make_tiny()
    pad_tok = args.engram_pad_id
    ids = np.array([5, 9, 13, 21, 34])
    cold = hasher.row_ids(hasher.empty_history(), tm[ids])
    np.testing.assert_array_equal(cold, ref_rows(ref, ids))
    # position 0 hashed with an explicit pad history == hashed after real pad tokens
    padded = hasher.row_ids(hasher.empty_history(), tm[np.array([pad_tok, pad_tok, pad_tok, *ids])])
    np.testing.assert_array_equal(padded[3:], cold)
    # a non-pad history changes the rows of the first 3 positions only
    hist = np.array([1, 4, 7], dtype=np.int64)
    warm = hasher.row_ids(hist, tm[ids])
    assert (warm[:3] != cold[:3]).any()
    np.testing.assert_array_equal(warm[3:], cold[3:])


def test_chunked_prefill_matches_whole_sequence():
    args, layout, tm, ref, hasher = make_tiny()
    rng = np.random.default_rng(2)
    ids = rng.integers(0, VOCAB, size=97)
    expect = ref_rows(ref, ids)
    comp = tm[ids]
    for bounds in ([0, 1, 2, 3, 97], [0, 40, 41, 97], [0, 33, 66, 97], [0, 97]):
        parts = []
        for s, e in zip(bounds[:-1], bounds[1:]):
            hist = EngramHistory.from_prompt(hasher.n_history, hasher.pad_id, comp, s)
            parts.append(hasher.row_ids(hist.view(), comp[s:e]))
        np.testing.assert_array_equal(np.concatenate(parts), expect)


def test_per_token_incremental_matches_whole_sequence():
    args, layout, tm, ref, hasher = make_tiny()
    rng = np.random.default_rng(3)
    ids = rng.integers(0, VOCAB, size=50)
    expect = ref_rows(ref, ids)
    hist = EngramHistory(hasher.n_history, hasher.pad_id)
    rows = []
    for t in range(ids.shape[0]):
        c = tm[ids[t : t + 1]]
        rows.append(hasher.row_ids(hist.view(), c))
        hist.advance(c)
    np.testing.assert_array_equal(np.concatenate(rows), expect)
    # the reference's own incremental path (start_pos advancing) agrees too
    inc = [ref(torch.tensor(ids[t : t + 1])[None], t)[0].numpy() for t in range(ids.shape[0])]
    np.testing.assert_array_equal(np.concatenate(inc), expect)


def test_verify_rows_snapshot_then_commit_accepted():
    """Speculative verify: hash anchor+drafts from the current history without advancing, commit
    only the accepted prefix, and the resulting stream must hash like the accepted sequence."""
    args, layout, tm, ref, hasher = make_tiny()
    rng = np.random.default_rng(4)
    prompt = rng.integers(0, VOCAB, size=12)
    hist = EngramHistory.from_prompt(hasher.n_history, hasher.pad_id, tm[prompt])
    accepted_stream = list(prompt)
    all_rows = [hasher.row_ids(hasher.empty_history(), tm[prompt])]
    for _ in range(6):
        anchor = int(rng.integers(0, VOCAB))
        drafts = rng.integers(0, VOCAB, size=4)
        toks = np.array([anchor, *drafts])
        snap = hist.snapshot()
        rows = hasher.row_ids(hist.view(), tm[toks])  # no advance
        np.testing.assert_array_equal(hist.view(), snap)
        n_acc = int(rng.integers(0, 5))  # accepted drafts
        kept = toks[: 1 + n_acc]
        # rows of the kept positions equal hashing them in-stream
        np.testing.assert_array_equal(rows[: 1 + n_acc], ref_rows(ref, np.array(accepted_stream + list(kept)))[len(accepted_stream) :])
        hist.advance(tm[kept])
        accepted_stream += list(kept)
        all_rows.append(rows[: 1 + n_acc])
        # restore is the explicit rollback path and reproduces the snapshot
        h2 = EngramHistory(hasher.n_history, hasher.pad_id, hist.snapshot())
        h2.advance(tm[np.array([1, 2, 3, 4])])
        h2.restore(hist.snapshot())
        np.testing.assert_array_equal(h2.view(), hist.view())
    np.testing.assert_array_equal(np.concatenate(all_rows), ref_rows(ref, np.array(accepted_stream)))


def test_row_ids_rejects_ids_that_would_wrap():
    args, layout, tm, ref, hasher = make_tiny()
    huge = np.array([2**62], dtype=np.int64)
    with pytest.raises(ValueError, match="wrap"):
        hasher.row_ids(hasher.empty_history(), huge)
    with pytest.raises(ValueError, match="history"):
        hasher.row_ids(np.zeros(2, dtype=np.int64), tm[np.array([1])])


# --------------------------------------------------------------------------- runtime over the hasher


def _runtime(args, layout, tm, seed=0):
    rng = np.random.default_rng(seed)
    tables = {}
    stores = {}
    for li, lid in enumerate(layout.layer_ids):
        rows = sum(layout.primes_flat[li])
        vals = rng.integers(0, 256, size=(rows, layout.head_dim), dtype=np.uint8)
        scales = rng.integers(100, 140, size=(rows, layout.head_dim // 32), dtype=np.uint8)
        tables[lid] = (vals, scales)
        stores[lid] = ArrayRowStore(vals, scales)
    rt = EngramRuntime.from_components(layout, tm, args.engram_compressed_vocab_size, args.engram_pad_id, stores, prefill_cap_tokens=64)
    return rt, tables


def _check_fetch(rt, tables, fetch, rows):
    """Fetched bytes == table[rows] for every layer, layer-major in the staging slot."""
    T = rows.shape[0]
    for lid in rt.layout.layer_ids:
        vals, scales = rt.layer_view(fetch, lid, T)
        li = rt.layout.layer_index(lid)
        tv, ts = tables[lid]
        np.testing.assert_array_equal(vals, tv[rows[:, li]])
        np.testing.assert_array_equal(scales, ts[rows[:, li]])


def test_runtime_prefill_decode_verify_match_reference():
    args, layout, tm, ref, hasher = make_tiny()
    rt, tables = _runtime(args, layout, tm)
    rng = np.random.default_rng(5)
    uid = "req-1"
    prompt = rng.integers(0, VOCAB, size=45)
    # chunked prefill, with the second chunk prefetched before the first is joined
    chunks = [(0, 20), (20, 41), (41, 45)]
    f_next = None
    got = []
    for i, (s, e) in enumerate(chunks):
        item = PrefillItem(uid, prompt, s, e)
        fetch = rt.prepare_prefill([item])
        if f_next is not None:
            assert fetch is f_next
        if i + 1 < len(chunks):
            f_next = rt.prefetch_prefill_next([PrefillItem(uid, prompt, *chunks[i + 1])])
        vals, scales = fetch.wait()
        rows = ref_rows(ref, prompt)[s:e]
        _check_fetch(rt, tables, fetch, rows)
        got.append(rows)
        fetch.release()
    assert rt.stats()["prefetch_hits"] == 2 and rt.stats()["blocked_joins"] == 1

    stream = list(prompt)
    # decode steps (history continues from the prompt)
    for _ in range(5):
        tok = int(rng.integers(0, VOCAB))
        fetch = rt.prepare_decode(np.array([tok]), [uid], bs=1)
        stream.append(tok)
        rows = ref_rows(ref, np.array(stream))[-1:]
        _check_fetch(rt, tables, fetch, rows)
    # verify rows, then commit a prefix
    for _ in range(4):
        anchor = int(rng.integers(0, VOCAB))
        drafts = rng.integers(0, VOCAB, size=3)
        fetch = rt.prepare_verify(uid, anchor, drafts)
        toks = [anchor, *drafts]
        rows = ref_rows(ref, np.array(stream + toks))[len(stream) :]
        _check_fetch(rt, tables, fetch, rows)
        fetch.release()
        n_acc = int(rng.integers(0, 4))
        rt.commit_verify(uid, toks[: 1 + n_acc])
        stream += toks[: 1 + n_acc]
    # a decode step after the verify commits still matches the accepted stream
    tok = int(rng.integers(0, VOCAB))
    fetch = rt.prepare_decode(np.array([tok]), [uid], bs=1)
    stream.append(tok)
    _check_fetch(rt, tables, fetch, ref_rows(ref, np.array(stream))[-1:])
    rt.drop(uid)
    assert rt.stats()["requests"] == 0
    rt.close()


def test_runtime_batched_decode_keeps_histories_apart():
    args, layout, tm, ref, hasher = make_tiny()
    rt, tables = _runtime(args, layout, tm)
    rng = np.random.default_rng(6)
    prompts = {u: rng.integers(0, VOCAB, size=n) for u, n in (("a", 7), ("b", 30))}
    for u, p in prompts.items():
        rt.prepare_prefill([PrefillItem(u, p, 0, len(p))]).wait()
    streams = {u: list(p) for u, p in prompts.items()}
    for _ in range(3):
        toks = rng.integers(0, VOCAB, size=2)
        fetch = rt.prepare_decode(toks, ["a", "b"], bs=3)  # bs > live requests: padding row
        for i, u in enumerate(("a", "b")):
            streams[u].append(int(toks[i]))
        rows = np.stack([ref_rows(ref, np.array(streams[u]))[-1] for u in ("a", "b")])
        for lid in layout.layer_ids:
            vals, _ = rt.layer_view(fetch, lid, 3)
            li = layout.layer_index(lid)
            np.testing.assert_array_equal(vals[:2], tables[lid][0][rows[:, li]])
    rt.close()
