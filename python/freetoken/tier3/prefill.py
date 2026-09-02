"""Whole-layer staging for prefill under tier 3.

Prefill is the awkward half of the design. Decode touches ~21 experts per layer and the demand
resolver handles it; prefill materialises **every** expert of a layer into a GPU buffer where
position == expert id, which the host pool (``host_slots`` rows, arbitrary occupancy) cannot
provide. So a prefill layer is assembled in a staging bank of full ``num_experts`` width and the
existing whole-layer H2D copies from there, unchanged.

The staging bank is one layer's worth (~1.35 GiB for Qwen3.8-Flash-Next), allocated once and
reused for every layer and every chunk. Rows come from two places:

* experts already in the host pool -- one vectorised gather per bank, no disk;
* the rest -- parallel O_DIRECT reads, exactly the tier-3 decode path.

Deliberately NOT admitted into the pool: a prefill chunk scans the whole corpus, so admitting
would evict the entire decode-hot set on every chunk.
"""

from __future__ import annotations

import torch

from freetoken.tier3.nvme_store import ALIGN, NvmeExpertStore
from freetoken.utils import init_logger

logger = init_logger(__name__)


class PrefillStager:
    """Plans one prefill layer's fill and stages the NVMe half of it.

    Emits index tensors the caller feeds to ``fast_index_copy_multi_jit``:

    * ``res_dst``/``res_src`` -- expert id and its host pool row, for the experts already
      resident. These move pool -> prefill buffer directly over PCIe; no host compute.
    * ``miss_dst``/``miss_src`` -- expert id and its COMPACT staging row, for the rest, which
      are read from disk into the staging bank first.

    The staging bank is full ``num_experts`` width because a cold pool means every expert is a
    miss; steady state uses only the leading rows.

    Deliberately does NOT admit into the pool: a prefill chunk scans the whole corpus, so
    admitting would evict the decode-hot set on every chunk.
    """

    SLOTS = 2  # double-buffered, matching the prefill overlap buffers

    def __init__(self, store, pool_sources, host_slot_for_id, num_experts, device,
                 stage_rows: int | None = None):
        self.store = store
        self.pool = pool_sources
        self.host_slot_for_id = host_slot_for_id
        self.num_experts = num_experts
        # Staging only ever holds a layer's NON-resident experts. With a prewarmed pool that is
        # exactly num_experts - host_slots and never grows (the pool stays full; LRU replaces),
        # so sizing it for a cold pool wasted GiB that the pool itself wants. Callers that
        # disable prewarming must pass num_experts.
        self.stage_rows = int(stage_rows or num_experts)
        self.device = device
        self.bank_names = tuple(pool_sources)
        self.stage = [{} for _ in range(self.SLOTS)]
        self._u8 = [{} for _ in range(self.SLOTS)]
        self.stage_ptrs_dev = [None] * self.SLOTS
        self.bytes_from_disk = 0
        self._missing = [[] for _ in range(self.SLOTS)]
        self._inflight = [None] * self.SLOTS   # pending read future per staging slot
        self._plan = [(0, 0)] * self.SLOTS
        # Recorded after each staging slot's gather. The prefill buffers' ready events order
        # GPU streams, not the host -- without this the host would start refilling a staging
        # slot while the gather reading it is still in flight, which silently corrupts the
        # layer (it showed up as an empty first response).
        self._gather_done = [torch.cuda.Event() for _ in range(self.SLOTS)]
        self._gather_pending = [False] * self.SLOTS
        # Per staging slot, not shared: the gather for layer L is enqueued only after layer
        # L+1 has already been planned (that is the whole point of the read-ahead), so one set
        # of index tensors would have L reading L+1's plan.
        z = lambda: torch.zeros(num_experts, dtype=torch.int32, device=device)  # noqa: E731
        self.res_dst_dev = [z() for _ in range(self.SLOTS)]
        self.res_src_dev = [z() for _ in range(self.SLOTS)]
        self.miss_dst_dev = [z() for _ in range(self.SLOTS)]
        self.miss_src_dev = [z() for _ in range(self.SLOTS)]
        self.res_num = [torch.zeros(1, dtype=torch.int64, device=device) for _ in range(self.SLOTS)]
        self.miss_num = [torch.zeros(1, dtype=torch.int64, device=device) for _ in range(self.SLOTS)]

    def _ensure(self):
        if self.stage[0]:
            return
        from freetoken.kernel.pinned import alloc_pinned_tensor, device_ptr

        total = 0
        for b in range(self.SLOTS):
            ptrs = []
            for name, per_layer in self.pool.items():
                row = per_layer[0][0]
                nbytes = self.stage_rows * row.numel() * row.element_size()
                t = alloc_pinned_tensor(nbytes, dtype=torch.uint8).view(row.dtype).view(
                    self.stage_rows, *row.shape)
                self.stage[b][name] = t
                self._u8[b][name] = t.view(torch.uint8).reshape(self.stage_rows, -1)
                # the gather kernel dereferences host memory from the GPU: it needs the alias
                ptrs.append(device_ptr(t))
                total += nbytes
            self.stage_ptrs_dev[b] = torch.tensor(ptrs, dtype=torch.int64, device=self.device)
        logger.info_rank0(
            f"tier 3 prefill staging: {total / 2**30:.2f} GiB pinned "
            f"({self.SLOTS} x {self.stage_rows} experts x {len(self.stage[0])} banks)")

    def begin(self, layer_id: int) -> tuple[int, int]:
        """Plan ``layer_id`` and start its NVMe reads WITHOUT waiting.

        Prefill already runs one layer ahead (``prefetch_prefill_layer(L+1)``), so issuing the
        reads there and joining them only when the buffer is consumed hides them behind the
        previous layer's GEMMs. That is where the disk time actually is: decode is a pool hit
        in steady state, but a prefill chunk re-reads every non-resident expert of every layer.
        """
        b = layer_id % self.SLOTS
        if self._gather_pending[b]:
            self._gather_done[b].synchronize()   # see _gather_done
            self._gather_pending[b] = False
        n_res, n_miss = self.plan(layer_id)
        self._plan[b] = (n_res, n_miss)
        if n_miss:
            jobs = [(layer_id, e, name, memoryview(self._u8[b][name][i].numpy()))
                    for name in self.bank_names
                    for i, e in enumerate(self._missing[b])]
            self._inflight[b] = self.store.submit_rows(jobs)
        return n_res, n_miss

    def join(self, layer_id: int) -> tuple[int, int]:
        """Wait for a :meth:`begin` batch; returns its (n_resident, n_missing)."""
        b = layer_id % self.SLOTS
        fut, self._inflight[b] = self._inflight[b], None
        if fut is not None:
            self.bytes_from_disk += self.store.wait_rows(fut)
        return self._plan[b]

    def discard_inflight(self) -> None:
        """Abandon pending reads and pending gather guards (the buffers they target are gone).

        Called when the cache is rebuilt. The reads themselves are harmless -- they write into
        staging, which survives -- so they are simply waited out rather than cancelled, and the
        plans are cleared so nothing gathers from them.
        """
        for b in range(self.SLOTS):
            fut, self._inflight[b] = self._inflight[b], None
            if fut is not None:
                self.store.wait_rows(fut)
            if self._gather_pending[b]:
                self._gather_done[b].synchronize()
                self._gather_pending[b] = False
            self._plan[b] = (0, 0)
            self._missing[b] = []

    def mark_gathered(self, layer_id: int, stream=None) -> None:
        """Record that this layer's staging slot is now only owed to the GPU."""
        b = layer_id % self.SLOTS
        self._gather_done[b].record(stream or torch.cuda.current_stream())
        self._gather_pending[b] = True

    def plan(self, layer_id: int) -> tuple[int, int]:
        """-> (n_resident, n_missing) for ``layer_id``, with the index tensors filled."""
        self._ensure()
        slots = self.host_slot_for_id[layer_id]
        res = torch.nonzero(slots >= 0, as_tuple=True)[0].to(torch.int32)
        miss = torch.nonzero(slots < 0, as_tuple=True)[0].to(torch.int32)
        n_res, n_miss = int(res.numel()), int(miss.numel())
        if n_miss > self.stage_rows:
            raise RuntimeError(
                f"layer {layer_id} has {n_miss} non-resident experts but staging holds "
                f"{self.stage_rows}; the pool is emptier than the staging sizing assumed "
                f"(prewarm disabled?)")
        b = layer_id % self.SLOTS
        if n_res:
            self.res_dst_dev[b][:n_res].copy_(res, non_blocking=True)
            self.res_src_dev[b][:n_res].copy_(slots[res.long()].to(torch.int32), non_blocking=True)
            self.res_num[b].fill_(n_res)
        if n_miss:
            self.miss_dst_dev[b][:n_miss].copy_(miss, non_blocking=True)
            # compact staging rows 0..n_miss-1, in the same order the reads fill them
            self.miss_src_dev[b][:n_miss].copy_(
                torch.arange(n_miss, dtype=torch.int32), non_blocking=True)
            self.miss_num[b].fill_(n_miss)
        self._missing[layer_id % self.SLOTS] = miss.tolist()
        return n_res, n_miss

    def read_missing(self, layer_id: int) -> None:
        """Blocking form of :meth:`begin` + :meth:`join`, for the non-overlap prefill path."""
        b = layer_id % self.SLOTS
        jobs = [(layer_id, e, name, memoryview(self._u8[b][name][i].numpy()))
                for name in self.bank_names for i, e in enumerate(self._missing[b])]
        if jobs:
            self.bytes_from_disk += self.store.read_rows(jobs)

    def stage_ptrs(self, layer_id: int) -> torch.Tensor:
        return self.stage_ptrs_dev[layer_id % self.SLOTS]

    def idx(self, layer_id: int) -> dict:
        """The index tensors belonging to this layer's staging slot."""
        b = layer_id % self.SLOTS
        return {"res_dst": self.res_dst_dev[b], "res_src": self.res_src_dev[b],
                "res_num": self.res_num[b], "miss_dst": self.miss_dst_dev[b],
                "miss_src": self.miss_src_dev[b], "miss_num": self.miss_num[b]}
