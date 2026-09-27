"""Host-side Engram runtime: token ids -> table rows -> staged FP8 bytes, per request.

Contract for the model/engine wiring (which lives elsewhere and calls only this API):

- Rows are a pure function of the token stream. ``rows_for_tokens(uid, token_ids)`` maps raw
  vocab ids through the compressed token map, hashes them against the request's
  :class:`EngramHistory` (the last ``max_ngram_size - 1`` compressed ids) and returns
  ``int64[T, n_layers, n_hash_cols]`` absolute table rows; by default it advances the history.
  Prefill never depends on stored state: ``prepare_prefill`` rebuilds each request's history
  from ``input_ids[:start]``, so chunk continuations and radix hits hash exactly like one cold
  pass over the whole prompt, and leaves the history positioned after ``input_ids[:end]``.
- Decode: ``prepare_decode(ids_cpu, uids, bs)`` hashes one input token per request (advancing)
  and fetches ``bs * n_layers * n_hash_cols`` rows into the fixed staging slot for that batch
  size — same host address every step, so a captured graph can copy from it. The previous
  decode fetch of that slot is released implicitly: decode steps are sequential.
- Speculative verify: ``prepare_verify(uid, anchor, drafts)`` hashes ``[anchor, *drafts]`` from
  a snapshot of the history without advancing; ``commit_verify(uid, accepted)`` advances by the
  tokens whose positions were kept (the anchor plus the accepted drafts, in order). Rejected
  drafts therefore never need a rollback.
- ``drop(uid)`` forgets a request. ``stats()`` merges the store and fetch counters.

Every fetch returns an :class:`EngramFetch`; ``wait()`` yields host uint8 views laid out as
``[n_layers][T][n_hash_cols]`` rows (layer-major, so each layer's rows are one contiguous block),
which the device side dequantizes (``value * 2^(scale-127)`` per 32-block). Fetches are issued
on a driver thread the moment they are requested, so the read overlaps the caller's GPU work.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass

import numpy as np

from .hashing import EngramHistory, NGramHasher
from .layout import EngramLayout, compute_hash_multipliers
from .staging import EngramFetch, EngramStagingRing, FetchDriver
from .store import ArrayRowStore, EngramRowSource, SafetensorsRowStore
from .tables import resolve_engram_tables
from .token_map import load_or_build_token_map

READERS_ENV = "FREETOKEN_ENGRAM_READERS"
IO_ENV = "FREETOKEN_ENGRAM_IO"


@dataclass(frozen=True)
class PrefillItem:
    """One request's share of a prefill chunk: positions ``[start, end)`` of ``input_ids``."""

    uid: object
    input_ids: np.ndarray
    start: int
    end: int


class EngramRuntime:
    """See the module docstring for the contract. Build with :meth:`build` (real checkpoint) or
    :meth:`from_components` (tests, dummy weights)."""

    def __init__(
        self,
        layout: EngramLayout,
        token_map: np.ndarray,
        compressed_vocab_size: int,
        pad_token_id: int,
        stores: dict[int, EngramRowSource],
        *,
        prefill_cap_rows: int,
        prefill_buffers: int = 2,
        max_decode_bs: int = 1,
        max_verify_rows: int = 8,
        pin: bool = True,
        driver_workers: int = 2,
    ):
        self.layout = layout
        self.token_map = np.ascontiguousarray(token_map, dtype=np.int64)
        self.compressed_vocab_size = int(compressed_vocab_size)
        if int(self.token_map.max()) >= self.compressed_vocab_size:
            raise ValueError("token map ids exceed the compressed vocab size")
        if set(stores) != set(layout.layer_ids):
            raise ValueError(f"stores for layers {sorted(stores)} != engram layers {layout.layer_ids}")
        self.stores = {int(k): v for k, v in stores.items()}
        self.multipliers = compute_hash_multipliers(layout.layer_ids, layout.max_ngram_size, self.compressed_vocab_size)
        self.pad_id = int(self.token_map[int(pad_token_id)])
        self.hasher = NGramHasher(layout, self.multipliers, self.pad_id)
        dim, scale_cols = self._table_geometry()
        self.dim, self.scale_cols = dim, scale_cols
        self.rows_per_token = layout.n_layers * layout.n_hash_cols
        self._driver = FetchDriver(driver_workers)
        self._prefill_ring = EngramStagingRing(prefill_buffers, max(1, prefill_cap_rows), dim, scale_cols, pin)
        self._verify_ring = EngramStagingRing(2, max(1, max_verify_rows) * self.rows_per_token, dim, scale_cols, pin)
        self._decode_rings: dict[int, EngramStagingRing] = {}
        self._decode_fetch: dict[int, EngramFetch] = {}
        self.max_decode_bs = int(max_decode_bs)
        self.pin = pin
        self._histories: dict[object, EngramHistory] = {}
        self._pending_prefill: dict[tuple, EngramFetch] = {}
        self._lock = threading.Lock()
        self.fetches = 0
        self.blocked_joins = 0
        self.prefetch_hits = 0

    # ------------------------------------------------------------------ construction
    @classmethod
    def build(
        cls,
        model_path: str,
        args,
        *,
        engram_dir: str | None = None,
        io: str | None = None,
        readers: int | None = None,
        resident_scales: bool = True,
        prefill_cap_tokens: int = 2 * 8192,
        **kw,
    ) -> "EngramRuntime | None":
        """Real checkpoint: token map (cached), layout from the args, one row store per table."""
        layout = EngramLayout.from_args(args)
        if layout is None:
            return None
        token_map, size = load_or_build_token_map(model_path, int(args.engram_compressed_vocab_size))
        io = io or os.environ.get(IO_ENV) or "mmap"
        readers = int(readers if readers is not None else os.environ.get(READERS_ENV, 64))
        locs = resolve_engram_tables(
            model_path,
            layout.layer_ids,
            engram_dir,
            expected_rows=dict(zip(layout.layer_ids, layout.num_embeddings)),
            expected_dim=layout.head_dim,
        )
        stores = {lid: SafetensorsRowStore(loc, io=io, readers=readers, resident_scales=resident_scales) for lid, loc in locs.items()}
        cap_rows = prefill_cap_tokens * layout.n_layers * layout.n_hash_cols
        return cls(layout, token_map, size, int(args.engram_pad_id), stores, prefill_cap_rows=cap_rows, **kw)

    @classmethod
    def from_components(
        cls,
        layout: EngramLayout,
        token_map: np.ndarray,
        compressed_vocab_size: int,
        pad_token_id: int,
        stores: dict[int, EngramRowSource],
        *,
        prefill_cap_tokens: int = 256,
        **kw,
    ) -> "EngramRuntime":
        cap_rows = prefill_cap_tokens * layout.n_layers * layout.n_hash_cols
        kw.setdefault("pin", False)
        return cls(layout, token_map, compressed_vocab_size, pad_token_id, stores, prefill_cap_rows=cap_rows, **kw)

    @classmethod
    def zeros(cls, layout: EngramLayout, vocab_size: int, pad_token_id: int = 0, rows: int = 1024, **kw) -> "EngramRuntime":
        """Dummy-weight runtime: identity token map, tiny zero tables (rows are taken modulo ``rows``
        by :class:`_ModRowStore`), for booting without the checkpoint."""
        stores = {lid: _ModRowStore(rows, layout.head_dim, layout.head_dim // 32 or 1) for lid in layout.layer_ids}
        return cls.from_components(layout, np.arange(vocab_size), vocab_size, pad_token_id, stores, **kw)

    def _table_geometry(self) -> tuple[int, int]:
        dims = {(s.dim, s.scale_cols) for s in self.stores.values()}
        if len(dims) != 1:
            raise ValueError(f"engram tables disagree on geometry: {dims}")
        (dim, scale_cols) = dims.pop()
        if dim != self.layout.head_dim:
            raise ValueError(f"table dim {dim} != engram_head_dim {self.layout.head_dim}")
        for lid, s in self.stores.items():
            need = sum(self.layout.primes_flat[self.layout.layer_index(lid)])
            if s.rows < need:
                raise ValueError(f"layer {lid}: store has {s.rows} rows, layout addresses {need}")
        return dim, scale_cols

    # ------------------------------------------------------------------ pure core
    def compress(self, token_ids) -> np.ndarray:
        ids = np.asarray(token_ids, dtype=np.int64).reshape(-1)
        if ids.shape[0] and (ids.min() < 0 or ids.max() >= self.token_map.shape[0]):
            raise IndexError(f"token id outside the token map [0, {self.token_map.shape[0]})")
        return self.token_map[ids]

    def history(self, uid) -> EngramHistory:
        h = self._histories.get(uid)
        if h is None:
            h = self._histories[uid] = EngramHistory(self.hasher.n_history, self.pad_id)
        return h

    def reset_history(self, uid, prefix_token_ids, upto: int | None = None) -> EngramHistory:
        """Position ``uid``'s history just before ``prefix_token_ids[upto]`` (default: after all)."""
        comp = self.compress(prefix_token_ids)
        h = EngramHistory.from_prompt(self.hasher.n_history, self.pad_id, comp, upto)
        self._histories[uid] = h
        return h

    def rows_for_tokens(self, uid, token_ids, *, advance: bool = True) -> np.ndarray:
        """``int64[T, n_layers, n_hash_cols]`` table rows for ``token_ids`` given ``uid``'s history."""
        comp = self.compress(token_ids)
        h = self.history(uid)
        rows = self.hasher.row_ids(h.view(), comp)
        if advance:
            h.advance(comp)
        return rows

    @staticmethod
    def flatten_layer_major(rows: np.ndarray) -> np.ndarray:
        """``[T, L, C] -> [L*T*C]`` so each layer's rows are one contiguous block of the fetch."""
        return np.ascontiguousarray(np.transpose(rows, (1, 0, 2))).reshape(-1)

    # ------------------------------------------------------------------ fetch plumbing
    def _submit(self, ring: EngramStagingRing, rows: np.ndarray) -> EngramFetch:
        """``rows``: ``[T, L, C]`` -> one fetch, layer-major, each layer read from its own table."""
        T = rows.shape[0]
        parts = [(self.stores[lid], np.ascontiguousarray(rows[:, li, :]).reshape(-1)) for li, lid in enumerate(self.layout.layer_ids)]
        fetch = self._driver.submit_multi(ring, parts)
        fetch.n_tokens = T  # type: ignore[attr-defined]
        self.fetches += 1
        return fetch

    def layer_view(self, fetch: EngramFetch, layer_id: int, n_tokens: int | None = None):
        """``(vals[T, C, dim], scales[T, C, scale_cols])`` host views of one layer's rows."""
        vals, scales = fetch.wait()
        li = self.layout.layer_index(layer_id)
        C = self.layout.n_hash_cols
        T = n_tokens if n_tokens is not None else fetch.n_rows // (self.layout.n_layers * C)
        s = li * T * C
        return vals[s : s + T * C].reshape(T, C, self.dim), scales[s : s + T * C].reshape(T, C, self.scale_cols)

    # ------------------------------------------------------------------ prefill
    def _prefill_rows(self, items) -> np.ndarray:
        parts = []
        for it in items:
            ids = np.asarray(it.input_ids, dtype=np.int64).reshape(-1)
            if not (0 <= it.start <= it.end <= ids.shape[0]):
                raise ValueError(f"uid {it.uid}: [{it.start}, {it.end}) outside input_ids[{ids.shape[0]}]")
            self.reset_history(it.uid, ids, it.start)
            parts.append(self.rows_for_tokens(it.uid, ids[it.start : it.end], advance=True))
        if not parts:
            return np.zeros((0, self.layout.n_layers, self.layout.n_hash_cols), dtype=np.int64)
        return np.concatenate(parts, axis=0)

    @staticmethod
    def _prefill_key(items) -> tuple:
        return tuple((it.uid, int(it.start), int(it.end)) for it in items)

    def prefetch_prefill_next(self, items) -> EngramFetch:
        """Hash and submit a coming chunk now; ``prepare_prefill`` with the same items returns it."""
        key = self._prefill_key(items)
        with self._lock:
            if key in self._pending_prefill:
                return self._pending_prefill[key]
            fetch = self._submit(self._prefill_ring, self._prefill_rows(items))
            self._pending_prefill[key] = fetch
            return fetch

    def prepare_prefill(self, items) -> EngramFetch:
        """The fetch for this chunk (a prefetched one if it exists, else issued now)."""
        key = self._prefill_key(items)
        with self._lock:
            fetch = self._pending_prefill.pop(key, None)
            if fetch is not None:
                self.prefetch_hits += 1
                return fetch
            self.blocked_joins += 1
            return self._submit(self._prefill_ring, self._prefill_rows(items))

    # ------------------------------------------------------------------ decode
    def decode_ring(self, bs: int) -> EngramStagingRing:
        ring = self._decode_rings.get(bs)
        if ring is None:
            ring = self._decode_rings[bs] = EngramStagingRing(1, bs * self.rows_per_token, self.dim, self.scale_cols, self.pin)
        return ring

    def prepare_decode(self, ids_cpu, uids, bs: int) -> EngramFetch:
        """Rows for one decode step: ``ids_cpu[i]`` is request ``uids[i]``'s input token
        (``i < len(uids) <= bs``; padding rows of the slot beyond ``len(uids)`` are left as-is)."""
        ids = np.asarray(ids_cpu, dtype=np.int64).reshape(-1)[: len(uids)]
        rows = np.empty((bs, self.layout.n_layers, self.layout.n_hash_cols), dtype=np.int64)
        for i, uid in enumerate(uids):
            rows[i] = self.rows_for_tokens(uid, ids[i : i + 1], advance=True)[0]
        for i in range(len(uids), bs):
            rows[i] = 0  # padded batch rows read row 0: harmless, never consumed
        prev = self._decode_fetch.get(bs)
        if prev is not None:
            prev.wait()
            prev.release()
        fetch = self._submit(self.decode_ring(bs), rows)
        self._decode_fetch[bs] = fetch
        return fetch

    # ------------------------------------------------------------------ speculative verify
    def prepare_verify(self, uid, anchor: int, drafts) -> EngramFetch:
        """Rows for ``[anchor, *drafts]`` without touching the history."""
        toks = np.concatenate([[int(anchor)], np.asarray(drafts, dtype=np.int64).reshape(-1)])
        rows = self.rows_for_tokens(uid, toks, advance=False)
        return self._submit(self._verify_ring, rows)

    def commit_verify(self, uid, accepted) -> None:
        """Advance by the tokens whose positions were kept (anchor + accepted drafts, in order)."""
        self.history(uid).advance(self.compress(accepted))

    # ------------------------------------------------------------------ lifecycle
    def drop(self, uid) -> None:
        self._histories.pop(uid, None)
        with self._lock:
            for key in [k for k in self._pending_prefill if any(e[0] == uid for e in k)]:
                f = self._pending_prefill.pop(key)
                f.wait()
                f.release()

    def stats(self) -> dict:
        out = {
            "fetches": self.fetches,
            "blocked_joins": self.blocked_joins,
            "prefetch_hits": self.prefetch_hits,
            "requests": len(self._histories),
            "staging_bytes": self._prefill_ring.bytes + self._verify_ring.bytes + sum(r.bytes for r in self._decode_rings.values()),
            "pinned": self._prefill_ring.pinned,
            "stores": {},
        }
        for lid, s in self.stores.items():
            st = getattr(s, "stats", None)
            out["stores"][lid] = st.snapshot() if st is not None else {}
        return out

    def close(self) -> None:
        for f in list(self._pending_prefill.values()):
            f.wait()
        self._pending_prefill.clear()
        for f in self._decode_fetch.values():
            f.wait()
        self._driver.close()
        for s in self.stores.values():
            s.close()


class _ModRowStore(ArrayRowStore):
    """Zero table serving any row id (modulo its size) — dummy weights."""

    def __init__(self, rows: int, dim: int, scale_cols: int):
        super().__init__(np.zeros((rows, dim), np.uint8), np.full((rows, scale_cols), 127, np.uint8))
        self._mod = rows

    def read_rows(self, row_ids, vals, scales) -> None:
        super().read_rows(np.asarray(row_ids, dtype=np.int64) % self._mod, vals, scales)


__all__ = ["EngramRuntime", "PrefillItem"]
