"""A ``PLETableBackend`` that keeps only a slice of the PLE n-gram table in RAM.

Qwen3.8-Flash-Next's table is 47.7 GiB pinned -- on a 64 GiB box it crowds out the expert pool
so hard that the pool drops to ~14% residency and decode becomes NVMe-bandwidth-bound (measured
16 tok/s). Tiering the table is therefore not a nicety: it is what buys the expert pool enough
RAM to stop thrashing.

Traffic is tiny -- 16 rows of 160 B per token -- so this is a latency problem, not a bandwidth
one, and a few GiB of cache plus deep-queue random reads covers it.

**How the slots reach a captured graph.** The gather kernel only ever computes
``base + idx * head_dim``, so feeding it *cache slots* instead of *row ids* needs no kernel
change. But the row ids are produced on the GPU inside the decode graph, and resolving them
means a host round trip. So the resolve runs **outside** the graph, in a pre-replay hook, and
writes the slots into a static device buffer the captured gather reads. Eager forwards (prefill,
warmup) resolve inline instead, and under capture the buffer's contents do not matter -- only
its address is recorded.
"""

from __future__ import annotations

import torch

from freetoken.tier3.ple_rows import PleRowCache, read_ple_layout
from freetoken.utils import init_logger

logger = init_logger(__name__)

# The PLE reads are 160 B rows scattered over 320 M of them: IOPS-bound, not bandwidth-bound,
# so the queue wants to be far deeper than the expert tier's (which moves 2.6 MiB per request).
DEFAULT_PLE_READERS = 64


class RowCacheTable:
    """Pinned row cache over the on-disk PLE table; a drop-in for ``PinnedUVATable``."""

    def __init__(self, model_path: str, cache_bytes: int, device: torch.device,
                 max_rows: int, readers: int = DEFAULT_PLE_READERS) -> None:
        from freetoken.kernel.pinned import device_ptr

        layout = read_ple_layout(model_path)
        rows = max(1, int(cache_bytes // layout.head_dim))
        self.cache = PleRowCache(layout, rows, device=device, pin=True, readers=readers)
        self.weight = self.cache.weight
        self.num_rows = rows
        self.head_dim = layout.head_dim
        self.scale = float(layout.scale)
        self.dtype = torch.bfloat16
        self._is_fp8 = self.weight.dtype == torch.float8_e4m3fn
        self._device = device
        self._table_ptr = device_ptr(self.weight)
        # Static slot buffer the captured gather indexes; the pre-replay hook fills it.
        self._slots = torch.zeros(max_rows, dtype=torch.int64, device=device)
        self._staging: torch.Tensor | None = None
        self._graph_staging: dict[int, torch.Tensor] = {}
        logger.info_rank0(
            f"tier 3 PLE: {rows:,} row cache ({rows * layout.head_dim / 2**30:.2f} GiB pinned) "
            f"over {layout.num_rows / 1e6:.0f}M rows on disk "
            f"({layout.num_rows * layout.head_dim / 2**30:.1f} GiB), {readers} readers")

    # -- the pre-replay hook -------------------------------------------------------------

    def resolve_for(self, row_ids: torch.Tensor) -> int:
        """Make every id resident and publish its slot. Must run OUTSIDE a captured graph."""
        flat = row_ids.reshape(-1)
        n = int(flat.numel())
        if n > self._slots.numel():
            raise ValueError(f"PLE slot buffer holds {self._slots.numel()} rows, need {n}")
        slots = self.cache.resolve(flat.to("cpu"))
        self._slots[:n].copy_(slots, non_blocking=True)
        return n

    # -- PLETableBackend -----------------------------------------------------------------

    def _stage(self, rows: int) -> torch.Tensor:
        if torch.cuda.is_current_stream_capturing():
            buf = self._graph_staging.get(rows)
            if buf is None:
                buf = torch.empty((rows, self.head_dim), dtype=self.dtype, device=self._device)
                self._graph_staging[rows] = buf
            return buf
        buf = self._staging
        if buf is None or buf.shape[0] < rows:
            buf = torch.empty((rows, self.head_dim), dtype=self.dtype, device=self._device)
            self._staging = buf
        return buf[:rows]

    def lookup(self, row_ids: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        from freetoken.kernel.triton.ple import ple_gather_rows

        n = int(row_ids.numel())
        if not torch.cuda.is_current_stream_capturing():
            # Eager (prefill, warmup): nothing pre-resolved this, so do it now -- a host sync
            # here is legal and cheap relative to a prefill chunk.
            self.resolve_for(row_ids)
        dst = self._stage(n)
        rows = ple_gather_rows(self._table_ptr, self.num_rows, self.head_dim,
                               self._slots[:n], dst, self.scale, self._is_fp8)
        rows = rows.view(*row_ids.shape[:-1], -1)
        if out is not None:
            out.copy_(rows)
            return out
        return rows

    def prefetch(self, row_ids: torch.Tensor) -> None:
        # No side-stream prefetch: the gather is 2.5 KB/token and the resolve already happened
        # (pre-replay hook) or will happen inline in lookup.
        return None

    @property
    def stats(self) -> dict:
        c = self.cache
        tot = c.hits + c.misses
        return {"ple_rows": self.num_rows, "ple_hits": c.hits, "ple_misses": c.misses,
                "ple_hit_rate": c.hits / max(1, tot), "ple_disk_reads": c.reader.reads}
