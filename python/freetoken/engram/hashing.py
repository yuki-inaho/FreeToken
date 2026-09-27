"""N-gram hashing of compressed token ids into Engram table rows (numpy only, torch-free).

Mirrors the reference ``NgramHashState.forward`` exactly, but without its ``[B, max_seq_len]``
cache: the only state a position's hash needs is the ``max_ngram_size - 1`` compressed ids
before it, so :class:`EngramHistory` carries just those per request. That makes the hash of a
chunk a pure function of (history, chunk), which is what lets prefill chunks, decode steps and
speculative verify rows all hash identically to one whole-sequence pass — the tests pin that.

Semantics (per position ``t`` of the chunk, lookback ``k``): the id at ``t - k`` (reaching into
the history for ``t < k``), or the pad id where the sequence does not extend that far; products
``id * multiplier[layer][k]`` are XOR-folded one lookback at a time, and the fold after ``n - 1``
lookbacks is the ``n``-gram hash, bucketed as ``hash % prime + offset`` per head. Text-only:
the reference's DEAD (image span) tokens are not modelled.
"""

from __future__ import annotations

import numpy as np

from .layout import EngramLayout

_INT64_MAX = int(np.iinfo(np.int64).max)


class NGramHasher:
    """Hashes compressed ids into per-layer table rows for the given layout and multipliers."""

    def __init__(self, layout: EngramLayout, multipliers: np.ndarray, pad_compressed_id: int):
        self.layout = layout
        mult = np.asarray(multipliers, dtype=np.int64)
        if mult.shape != (layout.n_layers, layout.max_ngram_size):
            raise ValueError(f"multipliers {mult.shape} != ({layout.n_layers}, {layout.max_ngram_size})")
        self.multipliers = mult
        self.pad_id = int(pad_compressed_id)
        self.n_history = layout.max_ngram_size - 1
        # [n_layers, max_ngram-1, n_heads] moduli and [n_layers, n_hash_cols] offsets, hash-column order
        self._primes = layout.primes_array()
        self._offsets = layout.offsets_array()
        # Largest compressed id whose products stay inside int64. The reference bounds the
        # multipliers by the compressed vocab so this never trips for a valid token map; a
        # wrapped product would hash silently wrong, hence the check in row_ids.
        self._max_id = _INT64_MAX // int(mult.max())

    @property
    def n_hash_cols(self) -> int:
        return self.layout.n_hash_cols

    def empty_history(self) -> np.ndarray:
        return np.full(self.n_history, self.pad_id, dtype=np.int64)

    def row_ids(self, history: np.ndarray, tokens_compressed: np.ndarray) -> np.ndarray:
        """Table rows for a run of compressed ids, ``int64[T, n_layers, n_hash_cols]``.

        ``history``: the ``max_ngram_size - 1`` compressed ids before ``tokens_compressed[0]``,
        oldest first (``history[-1]`` immediately precedes the chunk), pad where the sequence is
        shorter. Rows are absolute (offsets applied), ready for the table lookup.
        """
        tokens = np.asarray(tokens_compressed, dtype=np.int64).reshape(-1)
        hist = np.asarray(history, dtype=np.int64).reshape(-1)
        if hist.shape[0] != self.n_history:
            raise ValueError(f"history has {hist.shape[0]} ids, expected {self.n_history}")
        n = self.layout.max_ngram_size
        T = tokens.shape[0]
        out = np.empty((T, self.layout.n_layers, self.n_hash_cols), dtype=np.int64)
        if T == 0:
            return out
        if tokens.min() < 0 or hist.min() < 0:
            raise ValueError("negative compressed id")
        if max(int(tokens.max()), int(hist.max())) > self._max_id:
            raise ValueError("compressed id too large: id * multiplier would wrap int64")
        ext = np.concatenate([hist, tokens])
        # lookback k reads ext[H + t - k]; H = n - 1 so every k <= n - 1 stays in range
        H = self.n_history
        looked = np.stack([ext[H - k : H - k + T] for k in range(n)], axis=-1)  # [T, n]
        products = looked[:, None, :] * self.multipliers[None, :, :]  # [T, L, n]
        rolling = products[..., 0]
        heads = self.layout.n_heads
        for i in range(1, n):
            rolling = np.bitwise_xor(rolling, products[..., i])
            out[:, :, (i - 1) * heads : i * heads] = rolling[..., None] % self._primes[:, i - 1, :][None]
        out += self._offsets[None]
        return out


class EngramHistory:
    """The last ``max_ngram_size - 1`` compressed ids of one request (pad where absent).

    ``snapshot``/``restore`` (or hashing without ``advance``) cover speculative verify rows:
    hash the anchor + drafts from the current history, then advance only by what was accepted,
    so no rollback is ever needed.
    """

    __slots__ = ("ids", "pad_id")

    def __init__(self, n_history: int, pad_id: int, ids=None):
        self.pad_id = int(pad_id)
        if ids is None:
            self.ids = np.full(n_history, self.pad_id, dtype=np.int64)
        else:
            self.ids = np.asarray(ids, dtype=np.int64).copy()
            if self.ids.shape != (n_history,):
                raise ValueError(f"history ids {self.ids.shape} != ({n_history},)")

    @classmethod
    def from_prompt(cls, n_history: int, pad_id: int, compressed_ids, upto: int | None = None) -> "EngramHistory":
        """History as it stands just before position ``upto`` of ``compressed_ids`` (default: after all)."""
        ids = np.asarray(compressed_ids, dtype=np.int64).reshape(-1)
        if upto is None:
            upto = ids.shape[0]
        if upto < 0 or upto > ids.shape[0]:
            raise ValueError(f"upto={upto} outside [0, {ids.shape[0]}]")
        h = cls(n_history, pad_id)
        h.advance(ids[:upto])
        return h

    def advance(self, compressed_ids) -> None:
        ids = np.asarray(compressed_ids, dtype=np.int64).reshape(-1)
        n = self.ids.shape[0]
        if ids.shape[0] >= n:
            self.ids[:] = ids[ids.shape[0] - n :] if n else ids[:0]
        elif ids.shape[0]:
            k = ids.shape[0]
            self.ids[:-k] = self.ids[k:]
            self.ids[-k:] = ids

    def snapshot(self) -> np.ndarray:
        return self.ids.copy()

    def restore(self, snap: np.ndarray) -> None:
        self.ids[:] = snap

    def view(self) -> np.ndarray:
        return self.ids


__all__ = ["EngramHistory", "NGramHasher"]
