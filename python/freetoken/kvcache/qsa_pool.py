"""QSA compressed-block sparse KV pool: paged GQA K/V + compressed index keys + pending ring.

Qwen3.8-Flash-Next scores whole ``index_ratio``-token groups instead of single tokens, so
its indexer slab holds ONE compressed key row per group, addressed by ``slot //
index_ratio``. Because ``page_size % index_ratio == 0``, a group's tokens always live in one
page at consecutive slots, which makes that division well-defined: the compressed rows are a
1/ratio shadow of the K/V pages and follow page sharing and eviction for free -- no
allocator, no free, no clear (SGLang qsa_kv_pool / vLLM compressed-region precedent).

Two tiers ride alongside the shadow slab and are NOT per-token:
- ``pending_ring``: the last ``ring_capacity`` pre-RoPE index keys of each running request (sized by ``ring_capacity_for``), indexed by ``Req.table_idx``. A group that straddles two forwards (chunked prefill, and
  every decode step) reads its already-consumed members from here. Never cleared: a new
  tenant of a table_idx starts at a group boundary (cached_len is 0 or a page multiple), so
  its first closing group takes every member from its own forward.
- scratch rows at ``cmp_scratch_base``: one row per request slot, the write target for rows
  whose group does not close in this forward, so the compress kernel scatters unconditionally
  with no negative index and no cross-row conflict (DSV4 precedent).

The slab is amortized into the per-token KV price (``unit_bytes``); the ring and scratch are
fixed and priced through ``kv_cost``'s ``fixed_cache_size``.

Storage dtypes. ``dtype`` is the COMPUTE dtype (16-bit): the index slab, the ring and every
backend scratch buffer ride it, and ``.dtype`` keeps reporting it. ``kv_dtype`` is the K/V
slab's storage dtype: the compute dtype (default, byte-identical to the historical pool) or
``torch.float8_e4m3fn`` (``--kv-cache-dtype fp8``), in which case a third tier appears --
``_kv_scale`` ``[2, layers, slots, kv_heads]`` fp32, one scale per (token slot, kv head) for
K and for V, addressed by the same token slot as the row so it follows page sharing and
eviction exactly like the shadow slab. The store quantizes (``qsa_store_kv_fp8``), the attend
kernel dequantizes in registers; zero codes x zero scales read as a finite 0, so a
never-written slot (the dummy page, padded rows) is harmless.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch

from .mha_pool import MHAKVCache

# The index tiers are always 2-byte (compute dtype); spec_kv_bytes_per_token budgets the same.
_INDEX_DTYPE_BYTES = 2
# fp8 K/V rows carry one fp32 scale per (token slot, kv head); spec_kv_bytes_per_token prices it.
FP8_KV_SCALE_BYTES = 4
_FP8_KV_DTYPE = torch.float8_e4m3fn


class QSAKVCache(MHAKVCache):
    """MHA paged pool + the compressed index-key slab + the per-request pending ring.

    ``cmp_k_cache(slot)`` is row-flat ``[num_pages * page_size // index_ratio + num_req_slots,
    index_head_dim]``: row ``r < cmp_scratch_base`` holds the compressed key of the token group
    whose K/V slots are ``[r * index_ratio, (r + 1) * index_ratio)``, and the rows from
    ``cmp_scratch_base`` on are the per-request-slot scratch sinks. ``slot`` is the sparse
    layer's order in the attention backend, same convention as BSAKVCache/DSAKVCache.
    """

    @classmethod
    def ring_capacity_for(cls, index_ratio: int, num_speculative_tokens: int = 0) -> int:
        """Ring depth: one row per pending position, keyed ``position % capacity``; spec decode widens by the draft depth (vLLM sizing)."""
        return index_ratio * math.ceil((index_ratio + num_speculative_tokens) / index_ratio)

    def __init__(
        self,
        num_kv_heads: int,
        num_layers: int,
        head_dim: int,
        num_pages: int,
        page_size: int,
        dtype: torch.dtype,
        device: torch.device,
        index_head_dim: int,
        num_index_layers: int,
        index_ratio: int,
        num_req_slots: int,
        ring_capacity: int | None = None,
        layer_ids: Sequence[int] | None = None,
        kv_dtype: torch.dtype | None = None,
    ) -> None:
        if index_ratio < 1 or page_size % index_ratio != 0:
            # slot // index_ratio only names one group when a group never straddles a page.
            raise ValueError(
                f"QSA needs page_size ({page_size}) divisible by index_ratio ({index_ratio})"
            )
        if ring_capacity is None:
            ring_capacity = self.ring_capacity_for(index_ratio)
        if ring_capacity < index_ratio:
            # A closing group reads up to index_ratio - 1 past members plus this forward's.
            raise ValueError(
                f"QSA needs ring_capacity ({ring_capacity}) >= index_ratio ({index_ratio})"
            )
        # Index keys ride the compute dtype (the model's index_k is engine-dtype). The KV cost
        # model budgets 2 bytes per token per index layer for the slab
        # (base.spec_kv_bytes_per_token); keep the two in lockstep.
        assert dtype.itemsize == _INDEX_DTYPE_BYTES, (
            f"QSA serves 16-bit compute only (the index slab budgets 2 bytes/token); got {dtype}"
        )
        kv_dtype = dtype if kv_dtype is None else kv_dtype
        if kv_dtype not in (dtype, _FP8_KV_DTYPE):
            raise ValueError(
                f"QSA K/V storage dtype must be the compute dtype ({dtype}) or "
                f"{_FP8_KV_DTYPE}; got {kv_dtype}"
            )
        self._index_head_dim = index_head_dim
        self._num_index_layers = num_index_layers
        self._index_ratio = index_ratio
        self._num_req_slots = num_req_slots
        self._ring_capacity = ring_capacity
        self._index_dtype = dtype
        self._compute_dtype = dtype
        self._kv_dtype = kv_dtype
        self._page_size = page_size
        self._kv_scale: torch.Tensor | None = None
        super().__init__(
            num_kv_heads=num_kv_heads,
            num_layers=num_layers,
            head_dim=head_dim,
            num_pages=num_pages,
            page_size=page_size,
            dtype=kv_dtype,
            device=device,
            layer_ids=layer_ids,
        )
        self._alloc_scale_slab()
        self._zero_kv_slabs()
        self._alloc_index_tiers(num_pages)

    # ----- storage dtypes ---------------------------------------------------------------------
    @property
    def dtype(self) -> torch.dtype:
        """The COMPUTE dtype (index slab, ring, backend scratch) -- not the K/V storage dtype."""
        return self._compute_dtype

    @property
    def kv_dtype(self) -> torch.dtype:
        """The K/V slab's storage dtype (``dtype`` or ``torch.float8_e4m3fn``)."""
        return self._kv_dtype

    @property
    def kv_quantized(self) -> bool:
        return self._kv_dtype is not self._compute_dtype and self._kv_dtype == _FP8_KV_DTYPE

    def _alloc_scale_slab(self) -> None:
        if not self.kv_quantized:
            self._kv_scale = None
            return
        _, layers, pages, page_size, heads, _ = self._kv_buffer.shape
        # zeros, not empty: a never-written slot must dequantize to a finite 0.
        self._kv_scale = torch.zeros(
            2, layers, pages * page_size, heads, dtype=torch.float32, device=self._device
        )

    def _zero_kv_slabs(self) -> None:
        # Defense-in-depth: the attend kernels pos-mask every K/V load (the real fix for
        # torch.empty's recycled NaN/Inf bit patterns), but a zeroed slab keeps any future
        # unmasked read finite instead of model-poisoning. One memset per (re)allocation.
        self._kv_buffer.zero_()
        if self._kv_scale is not None:
            self._kv_scale.zero_()

    def _alloc_index_tiers(self, num_pages: int) -> None:
        # ZERO-initialized: the score kernel reads whole rows of blocks unmasked and relies on
        # never-written tail rows dotting to a finite 0. Written rows are never cleared again,
        # so the kernel must clamp visible blocks to kvlen // index_ratio.
        self._cmp_scratch_base = num_pages * self._page_size // self._index_ratio
        self._cmp_k_buffer = torch.zeros(
            self._num_index_layers,
            self._cmp_scratch_base + self._num_req_slots,
            self._index_head_dim,
            dtype=self._index_dtype,
            device=self._device,
        )
        self._pending_ring = torch.zeros(
            self._num_req_slots,
            self._num_index_layers,
            self._ring_capacity,
            self._index_head_dim,
            dtype=self._index_dtype,
            device=self._device,
        )

    def rebuild(self, num_pages: int) -> None:
        # Free the index tiers (and the fp8 scale slab) BEFORE the K/V realloc (super().rebuild
        # frees + syncs + empty_cache), then re-derive them at the new page count. If the
        # index alloc itself fails (OOM), null the K/V slab too and re-raise: a pool with a
        # grown K/V slab and no index slab would mis-serve silently. Rebuild is idle-only, so
        # zeroing the ring here cannot drop a live request's pending members.
        self._cmp_k_buffer = None
        self._pending_ring = None
        self._kv_scale = None
        super().rebuild(num_pages)
        try:
            self._alloc_scale_slab()
            self._zero_kv_slabs()
            self._alloc_index_tiers(num_pages)
        except Exception:
            self._kv_buffer = None
            self._k_buffer = None
            self._v_buffer = None
            self._kv_scale = None
            raise

    # ----- K/V store / scale views ------------------------------------------------------------
    def store_kv(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        out_loc: torch.Tensor,
        layer_id: int,
    ) -> None:
        if not self.kv_quantized:
            return super().store_kv(k, v, out_loc, layer_id)
        from freetoken.kernel.triton.qsa import qsa_store_kv_fp8

        dense = self._dense(layer_id)
        qsa_store_kv_fp8(
            k, v, out_loc,
            self._k_buffer[dense].view(self._storage_shape),
            self._v_buffer[dense].view(self._storage_shape),
            self._kv_scale[0, dense],
            self._kv_scale[1, dense],
        )

    def k_scale(self, layer_id: int) -> torch.Tensor | None:
        """fp8 pools: the K scales of one layer, ``[num_pages * page_size, kv_heads]`` fp32
        (row = token slot); None when the K/V slab is 16-bit."""
        if self._kv_scale is None:
            return None
        return self._kv_scale[0, self._dense(layer_id)]

    def v_scale(self, layer_id: int) -> torch.Tensor | None:
        if self._kv_scale is None:
            return None
        return self._kv_scale[1, self._dense(layer_id)]

    # ----- sizing -------------------------------------------------------------------------------
    @classmethod
    def kv_cost(cls, config) -> tuple[int, int, int, int]:
        from .base import spec_kv_bytes_per_token
        from freetoken.attention import AttnType

        num_req_slots = config.max_running_req + 1
        per_token = 0
        fixed = 0
        for spec in config.model_config.kv_cache_group_specs():
            if spec.is_swa:
                continue
            per_token += spec_kv_bytes_per_token(spec, config)
            if spec.attn_type is AttnType.QSA:
                # One index-key row = all index layers at one position. The ring widens with
                # the speculative draft depth -- keep this in lockstep with the pool factory.
                row = spec.index_head_dim * spec.num_index_layers * _INDEX_DTYPE_BYTES
                ring = cls.ring_capacity_for(
                    spec.index_ratio,
                    getattr(config.model_config, "num_speculative_tokens", 0),
                )
                fixed += num_req_slots * row * (ring + 1)
        return per_token * config.page_size, fixed, config.page_size, 0

    def unit_bytes(self) -> tuple[int, int]:
        # Only the shadow slab (and the fp8 scale slab) scale with pages, and only the slab's
        # non-scratch rows; the ring and the scratch rows are the fixed term kv_cost reports
        # separately.
        kv, swa = super().unit_bytes()
        tokens = int(self._kv_buffer.shape[2]) * int(self._kv_buffer.shape[3])
        slab = (
            self._num_index_layers
            * self._cmp_scratch_base
            * self._index_head_dim
            * self._index_dtype.itemsize
        )
        scales = 0
        if self._kv_scale is not None:
            scales = self._kv_scale.numel() * self._kv_scale.element_size()
        return kv + (slab + scales) // tokens, swa

    def cmp_k_cache(self, slot: int) -> torch.Tensor:
        """Compressed index keys of one sparse layer: ``[rows, index_head_dim]``."""
        return self._cmp_k_buffer[slot]

    def pending_ring(self, slot: int) -> torch.Tensor:
        """One sparse layer's pending ring: ``[num_req_slots, ring_capacity, index_head_dim]``."""
        return self._pending_ring[:, slot]

    @property
    def cmp_scratch_base(self) -> int:
        """First scratch row of ``cmp_k_cache``; row ``cmp_scratch_base + table_idx`` sinks a
        forward whose group does not close."""
        return self._cmp_scratch_base

    @property
    def index_ratio(self) -> int:
        return self._index_ratio

    @property
    def index_head_dim(self) -> int:
        return self._index_head_dim

    @property
    def ring_capacity(self) -> int:
        return self._ring_capacity

    @property
    def num_req_slots(self) -> int:
        return self._num_req_slots


__all__ = ["FP8_KV_SCALE_BYTES", "QSAKVCache"]
