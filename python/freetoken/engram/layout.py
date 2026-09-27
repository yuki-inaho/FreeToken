"""Engram hash-table layout: primes, column offsets and hash multipliers.

The layout must be bit-identical to the reference ``inference/engram.py`` shipped with
DeepSeek-V4.1-Flash: every (layer, n-gram size, head) pair owns a prime-sized bucket range of
the layer's embedding table, the primes are drawn in a fixed order and never reused, and the
per-layer hash multipliers come from a seeded numpy generator. A single wrong prime silently
reads the wrong rows for a whole head, so this module has no tunables and validates the one
identity the checkpoint pins down: the primes of a layer sum to its table's row count.

The reference uses ``sympy.isprime``; we use a deterministic Miller-Rabin instead (exact for
every 64-bit integer, no sympy dependency) and prove agreement in the tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Deterministic Miller-Rabin witnesses: this set is exact for every n < 3.3e24 (Sorenson &
# Webster 2015), which comfortably covers int64.
_MR_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41)


def is_prime(n: int) -> bool:
    """Exact primality for 0 <= n < 3.3e24 (deterministic Miller-Rabin)."""
    if n < 2:
        return False
    for p in _MR_BASES:
        if n == p:
            return True
        if n % p == 0:
            return False
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in _MR_BASES:
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def find_next_prime(start: int, seen_primes: set[int]) -> int:
    """The smallest prime above ``start`` that has not been handed out yet (reference semantics)."""
    candidate = start + 1
    while not is_prime(candidate) or candidate in seen_primes:
        candidate += 1
    return candidate


def compute_hash_multipliers(
    layer_ids: tuple[int, ...] | list[int], max_ngram_size: int, compressed_vocab_size: int
) -> np.ndarray:
    """One odd multiplier per (layer, lookback), ``int64[n_layers, max_ngram_size]``.

    Bit-exact with the reference: ``default_rng(10007 * layer_id).integers(0, bound, ...)*2+1``
    with ``bound = max(1, ((2**63-1) // vocab) // 2)``, so ``compressed_id * multiplier`` never
    exceeds int64 (the hashes rely on there being no wrap).
    """
    max_long = int(np.iinfo(np.int64).max)
    bound = max(1, (max_long // int(compressed_vocab_size)) // 2)
    rows = []
    for layer_id in layer_ids:
        generator = np.random.default_rng(10007 * int(layer_id))
        values = generator.integers(low=0, high=bound, size=(max_ngram_size,), dtype=np.int64)
        rows.append(values * 2 + 1)
    return np.stack(rows).astype(np.int64)


@dataclass(frozen=True)
class EngramLayout:
    """Bucket layout of the n-gram hash tables (see the reference ``EngramLayout``).

    ``primes[layer][ngram-1][head]`` is the bucket modulus of the (n-gram size, head) pair;
    ``primes_flat[layer]`` / ``offsets[layer]`` are the same in the hash-column order the tables
    are indexed with (n-gram-major, head-minor: column ``(n-2)*n_heads + head``), so
    ``row = hash % primes_flat[layer][col] + offsets[layer][col]``.
    """

    max_ngram_size: int
    layer_ids: tuple[int, ...]
    num_embeddings: tuple[int, ...]
    primes: tuple[tuple[tuple[int, ...], ...], ...]
    n_heads: int
    head_dim: int
    primes_flat: tuple[tuple[int, ...], ...] = field(init=False, repr=False)
    offsets: tuple[tuple[int, ...], ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        flat, offsets = [], []
        for per_layer in self.primes:
            sizes = tuple(int(p) for per_ngram in per_layer for p in per_ngram)
            flat.append(sizes)
            offsets.append(tuple(int(o) for o in np.cumsum([0, *sizes[:-1]])))
        object.__setattr__(self, "primes_flat", tuple(flat))
        object.__setattr__(self, "offsets", tuple(offsets))
        if len(self.primes) != len(self.layer_ids):
            raise ValueError(f"{len(self.primes)} prime sets for {len(self.layer_ids)} engram layers")
        for li, per_layer in enumerate(self.primes):
            if len(per_layer) != self.max_ngram_size - 1 or any(len(h) != self.n_heads for h in per_layer):
                raise ValueError(
                    f"layer {self.layer_ids[li]}: primes are not [{self.max_ngram_size - 1}][{self.n_heads}]"
                )
        # The table row count is the checkpoint's own statement of the layout. A mismatch means
        # the primes (or the vocab they start from) are wrong, and every row lookup would be too.
        if self.num_embeddings:
            if len(self.num_embeddings) != len(self.layer_ids):
                raise ValueError(
                    f"{len(self.num_embeddings)} num_embeddings for {len(self.layer_ids)} engram layers"
                )
            for li, (layer_id, rows) in enumerate(zip(self.layer_ids, self.num_embeddings)):
                total = sum(self.primes_flat[li])
                if total != int(rows):
                    raise ValueError(
                        f"engram layer {layer_id}: primes sum to {total} rows but the table has {rows}; "
                        "the layout does not match this checkpoint"
                    )

    @property
    def n_layers(self) -> int:
        return len(self.layer_ids)

    @property
    def n_hash_cols(self) -> int:
        """Hash columns per layer per position: (max_ngram_size - 1) n-gram sizes x n_heads."""
        return (self.max_ngram_size - 1) * self.n_heads

    def layer_index(self, layer_id: int) -> int:
        return self.layer_ids.index(layer_id)

    def primes_array(self) -> np.ndarray:
        """``int64[n_layers, max_ngram_size-1, n_heads]``."""
        return np.asarray(self.primes, dtype=np.int64)

    def primes_flat_array(self) -> np.ndarray:
        """``int64[n_layers, n_hash_cols]`` in hash-column order."""
        return np.asarray(self.primes_flat, dtype=np.int64)

    def offsets_array(self) -> np.ndarray:
        """``int64[n_layers, n_hash_cols]`` in hash-column order."""
        return np.asarray(self.offsets, dtype=np.int64)

    @classmethod
    def build(
        cls,
        *,
        layer_ids,
        vocab_size: int,
        max_ngram_size: int,
        n_heads: int,
        head_dim: int,
        num_embeddings=(),
    ) -> "EngramLayout":
        """Draw the primes exactly like the reference: per layer, per n-gram size, per head,
        restarting the search at ``vocab_size - 1`` each time and skipping primes already used."""
        layer_ids = tuple(int(x) for x in layer_ids)
        primes, seen = [], set()
        for _ in layer_ids:
            per_ngram = []
            for _ in range(max_ngram_size - 1):
                sizes, current = [], int(vocab_size) - 1
                for _ in range(n_heads):
                    current = find_next_prime(current, seen)
                    seen.add(current)
                    sizes.append(current)
                per_ngram.append(tuple(sizes))
            primes.append(tuple(per_ngram))
        return cls(
            max_ngram_size=int(max_ngram_size),
            layer_ids=layer_ids,
            num_embeddings=tuple(int(x) for x in num_embeddings),
            primes=tuple(primes),
            n_heads=int(n_heads),
            head_dim=int(head_dim),
        )

    @classmethod
    def from_args(cls, args) -> "EngramLayout | None":
        """From any object with the ``engram_*`` attributes of the V4.1 args (None if no engram layers)."""
        layer_ids = tuple(getattr(args, "engram_layer_ids", ()) or ())
        if not layer_ids:
            return None
        return cls.build(
            layer_ids=layer_ids,
            vocab_size=args.engram_vocab_size,
            max_ngram_size=args.engram_max_ngram_size,
            n_heads=args.engram_n_heads,
            head_dim=args.engram_head_dim,
            num_embeddings=tuple(getattr(args, "engram_num_embeddings", ()) or ()),
        )


__all__ = ["EngramLayout", "compute_hash_multipliers", "find_next_prime", "is_prime"]
