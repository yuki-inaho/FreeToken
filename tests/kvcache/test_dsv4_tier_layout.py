"""DSV4 tier layout (owner map) and the cost model / pool built on it. Sections:
  * the layout itself: 0731's degenerate map, a V4.1 CSA2 toy map, invalid sources;
  * cost-model byte totals: 0731 pinned to the pre-layout formulas (hand-derived constants for
    one geometry, a verbatim copy of the old per-ratio code across several), V4.1 counted at
    owners only and matched by what the pool really allocates;
  * the pool's owner accessors (readers see the owner's tensors);
  * the pure V4.1 indexer selection helpers against a verbatim copy of the reference
    ``inference/model.py`` functions on random scores.

Args are ``SimpleNamespace``s: the layout is duck-typed off ``n_layers`` / ``compress_ratios`` /
``kv_source_layers``, and the V4.1 args package is not a dependency of this layer.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from freetoken.attention.dsv4_indexer import (
    candidate_mask_from_blocks,
    indexer_select_prefill,
    select_candidate_block_ids,
    select_candidate_blocks,
)
from freetoken.kvcache.dsv4_cost_model import (
    dsv4_cache_per_page,
    dsv4_kv_unit_bytes,
    dsv4_pool_bytes,
    dsv4_pool_sizes,
    dsv4_tier_layout,
    dsv4_window_unit_bytes,
    ring_size_for_ratio,
)
from freetoken.kvcache.dsv4_paged_pool import DSV4PagedKVCache

P = 128
RATIOS_0731 = (0, 0, 4, 128, 4, 128, 4, 0)
DEVICE = torch.device("cpu")


def _args_0731(**over):
    base = dict(
        n_layers=8, compress_ratios=RATIOS_0731, head_dim=512, index_head_dim=128,
        window_size=P, n_extra_window_layers=0,
    )
    base.update(over)
    return SimpleNamespace(**base)


def _args_v41(**over):
    # The plan's toy V4.1 geometry: two ratio-2 encoder layers owned by 2, four ratio-1
    # decoder layers owned by 4; index queries at 2/4/6, candidate source 4.
    base = dict(
        n_layers=8, compress_ratios=[0, 0, 2, 2, 1, 1, 1, 1], kv_source_layers=[2, 4],
        index_source_layers=[2, 4, 6], candidate_source_layer=4, head_dim=512,
        index_head_dim=128, window_size=P, n_extra_window_layers=0,
    )
    base.update(over)
    return SimpleNamespace(**base)


# --------------------------------------------------------------------------- #
# the layout
# --------------------------------------------------------------------------- #
def test_ring_size_for_ratio_table():
    assert [ring_size_for_ratio(r) for r in (1, 2, 4, 128)] == [0, 2, 8, 128]
    with pytest.raises(ValueError):
        ring_size_for_ratio(3)
    with pytest.raises(ValueError):
        ring_size_for_ratio(0)


def test_0731_layout_is_the_degenerate_owner_map():
    # Pre-layout semantics: every compressed layer owns its cmp pool and ring (8 at ratio 4,
    # 128 at ratio 128); the indexer pool, its ring and the CSA overlap exist at ratio 4 only.
    lay = dsv4_tier_layout(_args_0731())
    assert lay.n_layers == 8 and lay.ratios == RATIOS_0731
    assert lay.cmp_owner == (None, None, 2, 3, 4, 5, 6, None)
    assert lay.idx_owner == lay.cmp_owner
    assert lay.owns_cmp == (False, False, True, True, True, True, True, False)
    assert lay.owns_idx == (False, False, True, False, True, False, True, False)
    assert lay.ring_size == (0, 0, 8, 128, 8, 128, 8, 0)
    assert lay.overlap == lay.owns_idx  # overlap <=> ratio 4
    assert lay.idx_ring == lay.owns_idx
    assert lay.idx_ring_size == 8
    assert lay.cmp_layers() == [2, 3, 4, 5, 6]


def test_0731_layout_truncates_the_trailing_ratio():
    # The checkpoint ships 44 ratios for 43 layers; only the first n_layers count.
    lay = dsv4_tier_layout(_args_0731(n_layers=7))
    assert lay.ratios == RATIOS_0731[:7] and len(lay.cmp_owner) == 7


def test_v41_toy_layout_shares_owners_downstream():
    lay = dsv4_tier_layout(_args_v41())
    assert lay.cmp_owner == (None, None, 2, 2, 4, 4, 4, 4)
    assert lay.idx_owner == lay.cmp_owner
    assert lay.owns_cmp == (False, False, True, False, True, False, False, False)
    assert lay.owns_idx == lay.owns_cmp
    # a ring only at owners with rolling state: ratio 2 at layer 2; ratio 1 carries nothing
    assert lay.ring_size == (0, 0, 2, 0, 0, 0, 0, 0)
    assert not any(lay.overlap) and not any(lay.idx_ring)


def test_v41_real_geometry_owner_runs():
    # The shipped V4.1-Flash: 40 layers, [0,0]+[2]*18+[1]*20, kv sources 2/8/14/20.
    args = SimpleNamespace(
        n_layers=40, compress_ratios=[0, 0] + [2] * 18 + [1] * 20 + [0, 0, 0],
        kv_source_layers=[2, 8, 14, 20],
    )
    lay = dsv4_tier_layout(args)
    expect = [None, None] + [2] * 6 + [8] * 6 + [14] * 6 + [20] * 20
    assert list(lay.cmp_owner) == expect
    assert [L for L in range(40) if lay.owns_cmp[L]] == [2, 8, 14, 20]
    assert [L for L in range(40) if lay.ring_size[L]] == [2, 8, 14]
    assert all(lay.ring_size[L] == 2 for L in (2, 8, 14))


def test_v41_layout_rejects_bad_sources():
    # A reader whose most recent source compresses at a different ratio.
    with pytest.raises(ValueError, match="must match its owner's ratio"):
        dsv4_tier_layout(_args_v41(kv_source_layers=[2]))
    # A compressed layer with no source at or before it.
    with pytest.raises(ValueError, match="no kv_source_layers entry precedes"):
        dsv4_tier_layout(_args_v41(kv_source_layers=[4]))
    # A source that is a window-only layer.
    with pytest.raises(ValueError, match="not a compressed main layer"):
        dsv4_tier_layout(_args_v41(kv_source_layers=[1, 2, 4]))


# --------------------------------------------------------------------------- #
# cost model: 0731 unchanged
# --------------------------------------------------------------------------- #
# Hand derivation for _args_0731() at P=128, num_pages=8, swa_ratio=0.5, n_scratch=1:
#   kv_b = 512*2 = 1024; idx_b = 128*2 = 256
#   state bytes: ratio 4 (overlap) 2*2*512*4 = 8192; ratio 128 2*512*4 = 4096; idx ring 2*2*128*4 = 2048
# cache_per_page: window 8 layers x round(0.5*128)=64 x 1024 = 524288
#   ratio-4 layer: cmp 32*1024 + idx 32*256 + idx ring round(0.5*8)=4 x 2048 + attn ring 4*8192
#                = 32768 + 8192 + 8192 + 32768 = 81920; x3 = 245760
#   ratio-128 layer: cmp 1*1024 + ring round(0.5*128)=64 x 4096 = 263168; x2 = 526336
#   total 1296384
# kv_unit_bytes: per page 128*8 (map) + 3*(32*1024 + 32*256) + 2*(1*1024) = 1024+122880+2048
#                = 125952 -> ceil(/128) = 984
# window_unit_bytes: per page 8*128*1024 + 3*(8*8192 + 8*2048) + 2*(128*4096)
#                = 1048576 + 245760 + 1048576 = 2342912 -> /128 = 18304
# pool_sizes: full_token 1024, n_win_pages ceil(512/128)=4, n_win_slots 512;
#   ratio 4: cmp 256, idx 256, state 4*8=32, idx_state 32; ratio 128: cmp 8, state 4*128=512
# pool_bytes: window 8*512*1024 = 4194304; map (1024+1)*8 = 8200
#   ratio-4: (256+1)*1024 + (32+1)*8192 + (256+1)*256 + (32+1)*2048
#          = 263168 + 270336 + 65792 + 67584 = 666880; x3 = 2000640
#   ratio-128: (8+1)*1024 + (512+1)*4096 = 9216 + 2101248 = 2110464; x2 = 4220928
#   total 10424072
_0731_CACHE_PER_PAGE = 1_296_384
_0731_KV_UNIT = 984
_0731_WINDOW_UNIT = 18_304
_0731_POOL_BYTES = 10_424_072


def test_0731_cost_model_matches_hand_derivation():
    args = _args_0731()
    assert dsv4_cache_per_page(args, 0.5, P) == _0731_CACHE_PER_PAGE
    assert dsv4_kv_unit_bytes(args, P) == _0731_KV_UNIT
    assert dsv4_window_unit_bytes(args, P) == _0731_WINDOW_UNIT
    sizes = dsv4_pool_sizes(num_pages=8, args=args, swa_ratio=0.5, P=P)
    assert (sizes.full_token, sizes.n_win_pages, sizes.n_win_slots) == (1024, 4, 512)
    assert sizes.cmp_blocks == [None, None, 256, 8, 256, 8, 256, None]
    assert sizes.idx_blocks == [None, None, 256, None, 256, None, 256, None]
    assert sizes.state_slots == [None, None, 32, 512, 32, 512, 32, None]
    assert sizes.ring_sizes == [None, None, 8, 128, 8, 128, 8, None]
    assert sizes.idx_state_slots == [None, None, 32, None, 32, None, 32, None]
    assert dsv4_pool_bytes(sizes, args, n_scratch=1) == _0731_POOL_BYTES


# Verbatim copies of the pre-layout (per-ratio) formulas, kept as the independent reference
# the layout-driven code must reproduce for 0731 args across geometries.
def _old_ring(ratio):
    return {4: 8, 128: 128}[ratio]


def _old_state_bytes(args, ratio):
    return 2 * (1 + (1 if ratio == 4 else 0)) * args.head_dim * 4


def _old_idx_state_bytes(args):
    return 2 * 2 * args.index_head_dim * 4


def _old_cache_per_page(args, swa_ratio, P):
    kv_b, idx_b = args.head_dim * 2, args.index_head_dim * 2
    total = 0
    for ratio in tuple(args.compress_ratios)[: args.n_layers]:
        total += round(swa_ratio * P) * kv_b
        if ratio == 0:
            continue
        total += (P // ratio) * kv_b
        if ratio == 4:
            total += (P // 4) * idx_b
            total += round(swa_ratio * _old_ring(4)) * _old_idx_state_bytes(args)
        total += round(swa_ratio * _old_ring(ratio)) * _old_state_bytes(args, ratio)
    total += args.n_extra_window_layers * round(swa_ratio * P) * kv_b
    return int(total)


def _old_kv_unit_bytes(args, P):
    kv_b, idx_b = args.head_dim * 2, args.index_head_dim * 2
    per_page = P * 8
    for ratio in tuple(args.compress_ratios)[: args.n_layers]:
        if ratio == 0:
            continue
        per_page += (P // ratio) * kv_b
        if ratio == 4:
            per_page += (P // 4) * idx_b
    return -(-per_page // P)


def _old_window_unit_bytes(args, P):
    kv_b = args.head_dim * 2
    ratios = tuple(args.compress_ratios)[: args.n_layers]
    per_page = (len(ratios) + args.n_extra_window_layers) * P * kv_b
    for ratio in ratios:
        if ratio == 0:
            continue
        per_page += _old_ring(ratio) * _old_state_bytes(args, ratio)
        if ratio == 4:
            per_page += _old_ring(4) * _old_idx_state_bytes(args)
    return -(-per_page // P)


def _old_pool_bytes(sizes, args, n_scratch):
    kv_b, idx_b = args.head_dim * 2, args.index_head_dim * 2
    ratios = tuple(args.compress_ratios)[: args.n_layers]
    total = (len(ratios) + args.n_extra_window_layers) * sizes.n_win_slots * kv_b
    total += (sizes.full_token + 1) * 8
    for L, ratio in enumerate(ratios):
        if ratio == 0:
            continue
        total += (sizes.full_token // ratio + n_scratch) * kv_b
        total += (sizes.n_win_pages * _old_ring(ratio) + 1) * _old_state_bytes(args, ratio)
        if ratio == 4:
            total += (sizes.full_token // 4 + n_scratch) * idx_b
            total += (sizes.n_win_pages * _old_ring(4) + 1) * _old_idx_state_bytes(args)
    return int(total)


@pytest.mark.parametrize("num_pages,swa_ratio,n_scratch,extra", [
    (8, 0.5, 1, 0), (33, 0.25, 5, 3), (100, 1.0, 2, 0), (7, 0.1, 1, 1),
])
def test_0731_cost_model_matches_old_formulas(num_pages, swa_ratio, n_scratch, extra):
    args = _args_0731(n_extra_window_layers=extra)
    assert dsv4_cache_per_page(args, swa_ratio, P) == _old_cache_per_page(args, swa_ratio, P)
    assert dsv4_kv_unit_bytes(args, P) == _old_kv_unit_bytes(args, P)
    assert dsv4_window_unit_bytes(args, P) == _old_window_unit_bytes(args, P)
    sizes = dsv4_pool_sizes(num_pages=num_pages, args=args, swa_ratio=swa_ratio, P=P)
    assert dsv4_pool_bytes(sizes, args, n_scratch) == _old_pool_bytes(sizes, args, n_scratch)


# --------------------------------------------------------------------------- #
# cost model: V4.1 counts tiers at owners only
# --------------------------------------------------------------------------- #
def test_v41_pool_sizes_only_at_owners():
    sizes = dsv4_pool_sizes(num_pages=8, args=_args_v41(), swa_ratio=0.5, P=P)
    ft = sizes.full_token
    assert sizes.cmp_blocks == [None, None, ft // 2, None, ft // 1, None, None, None]
    assert sizes.idx_blocks == sizes.cmp_blocks
    assert sizes.state_slots == [None, None, sizes.n_win_pages * 2, None, None, None, None, None]
    assert sizes.ring_sizes == [None, None, 2, None, None, None, None, None]
    assert sizes.idx_state_slots == [None] * 8


def test_v41_pool_bytes_hand_formula():
    # Owners: layer 2 (ratio 2: cmp + idx rows ft/2, ring 2 no overlap) and layer 4 (ratio 1:
    # cmp + idx rows ft, no ring); readers 3, 5, 6, 7 add nothing; window on all 8 (+extra).
    args = _args_v41(n_extra_window_layers=3)
    n_scratch = 3
    sizes = dsv4_pool_sizes(num_pages=10, args=args, swa_ratio=0.5, P=P)
    ft, win = sizes.full_token, sizes.n_win_slots
    kv_b, idx_b = 1024, 256
    ring2_bytes = 2 * 1 * 512 * 4  # fp32 kv|score, no overlap
    expect = (8 + 3) * win * kv_b + (ft + 1) * 8
    expect += (ft // 2 + n_scratch) * (kv_b + idx_b) + (sizes.n_win_pages * 2 + 1) * ring2_bytes
    expect += (ft + n_scratch) * (kv_b + idx_b)
    assert dsv4_pool_bytes(sizes, args, n_scratch) == expect
    # per-page and per-unit views agree with the same owner-only accounting
    assert dsv4_kv_unit_bytes(args, P) == -(-(P * 8 + (P // 2 + P) * (kv_b + idx_b)) // P)
    assert dsv4_window_unit_bytes(args, P) == ((8 + 3) * P * kv_b + 2 * ring2_bytes) // P
    per_page = 11 * round(0.5 * P) * kv_b + (P // 2 + P) * (kv_b + idx_b)
    per_page += round(0.5 * 2) * ring2_bytes
    assert dsv4_cache_per_page(args, 0.5, P) == per_page


# --------------------------------------------------------------------------- #
# the pool: allocation and owner accessors
# --------------------------------------------------------------------------- #
def _pool(args, num_pages=8, n_scratch=2):
    sizes = dsv4_pool_sizes(num_pages=num_pages, args=args, swa_ratio=0.5, P=P)
    return DSV4PagedKVCache(sizes=sizes, args=args, device=DEVICE, P=P, n_scratch=n_scratch), sizes


def test_v41_pool_allocates_at_owners_and_routes_readers():
    args = _args_v41(n_extra_window_layers=3)
    pool, sizes = _pool(args, n_scratch=2)
    assert len(pool.window_pool) == 11
    for L in range(8):
        assert (pool.cmp_pool[L] is not None) == (L in (2, 4))
        assert (pool.idx_pool[L] is not None) == (L in (2, 4))
        assert (pool.state_ring[L] is not None) == (L == 2)
        assert pool.indexer_state_ring[L] is None
    assert pool.cmp_pool[2].shape == (sizes.full_token // 2 + 2, 512)
    assert pool.idx_pool[4].shape == (sizes.full_token + 2, 128)
    assert pool.state_ring[2].ring_size == 2 and pool.state_ring[2].item_size == 512
    # readers resolve to the SAME tensors as their owner (aliases, not copies)
    for L in (3,):
        assert pool.cmp_owner_of(L) == 2 and pool.cmp_pool_of(L) is pool.cmp_pool[2]
        assert pool.idx_pool_of(L) is pool.idx_pool[2]
        assert pool.state_ring_of(L) is pool.state_ring[2]
        assert pool.ring_size_of(L) == 2
        assert pool.cmp_scratch_base_of(L) == sizes.full_token // 2
    for L in (5, 6, 7):
        assert pool.cmp_owner_of(L) == 4 and pool.cmp_pool_of(L) is pool.cmp_pool[4]
        assert pool.state_ring_of(L) is None and pool.ring_size_of(L) == 0
        assert pool.idx_scratch_base_of(L) == sizes.full_token
    for L in (0, 1):
        assert pool.cmp_owner_of(L) is None and pool.cmp_pool_of(L) is None
    # a store through a reader's id lands in the owner's pool
    k = torch.randn(1, 512, dtype=torch.bfloat16)
    pool.store_compressed(k, 7, torch.tensor([5]))
    assert torch.equal(pool.cmp_pool[4][5:6], k)
    # what was allocated (mapping included) is what the cost model said
    assert pool.total_bytes() == dsv4_pool_bytes(sizes, args, n_scratch=2)


def test_0731_pool_owner_accessors_are_identity():
    pool, sizes = _pool(_args_0731())
    for L, ratio in enumerate(RATIOS_0731):
        if ratio == 0:
            assert pool.cmp_owner_of(L) is None and pool.cmp_pool_of(L) is None
            continue
        assert pool.cmp_owner_of(L) == L
        assert pool.cmp_pool_of(L) is pool.cmp_pool[L]
        assert pool.state_ring_of(L) is pool.state_ring[L]
        assert pool.ring_size_of(L) == pool.ring_size(L) == {4: 8, 128: 128}[ratio]
        assert pool.cmp_scratch_base_of(L) == pool.cmp_scratch_base[L] == sizes.cmp_blocks[L]
        if ratio == 4:
            assert pool.idx_pool_of(L) is pool.idx_pool[L]
            assert pool.indexer_state_ring_of(L) is pool.indexer_state_ring[L]
        else:
            assert pool.idx_pool_of(L) is None and pool.idx_scratch_base_of(L) is None
    assert pool.total_bytes() == dsv4_pool_bytes(sizes, _args_0731(), n_scratch=2)


def test_pool_rejects_ratio_not_dividing_page():
    # An unsupported ratio is refused by the layout (no ring geometry for it) before any
    # pool math; a supported one that does not divide the window page dies at the pool.
    with pytest.raises(ValueError, match="no ring for ratio 3"):
        _pool(_args_v41(compress_ratios=[0, 0, 3, 3, 1, 1, 1, 1]))
    args = _args_0731(window_size=64)
    sizes = dsv4_pool_sizes(num_pages=8, args=args, swa_ratio=0.5, P=64)
    with pytest.raises(AssertionError, match="must be divisible by ratio 128"):
        DSV4PagedKVCache(sizes=sizes, args=args, device=DEVICE, P=64, n_scratch=1)


# --------------------------------------------------------------------------- #
# V4.1 indexer selection helpers vs the reference model.py (verbatim copies)
# --------------------------------------------------------------------------- #
def _ref_select_candidate_blocks(logits, compress_lens, topk_blocks, block_size):
    width = logits.size(-1)
    scores = F.pad(logits, (0, -width % block_size), value=-torch.inf)
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    num_blocks = scores.size(-1)
    last = (compress_lens - 1) // block_size
    scores = scores.masked_fill(torch.arange(num_blocks, device=logits.device) == last, torch.inf)
    top = scores.topk(min(topk_blocks, num_blocks), dim=-1)
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, top.indices, top.values > -torch.inf)
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


def _ref_indexer_tail(index_score, compress_lens, end_blocks, index_topk, offset, candidates):
    # Indexer.forward from the causal mask down: prefill (compress_lens [S,1], scores masked
    # by reach) or decode (compress_lens int == the slab width).
    if candidates is not None:
        index_score = index_score.masked_fill(~candidates, -torch.inf)
    topk = min(index_topk, end_blocks)
    idxs = index_score.topk(topk, dim=-1, sorted=False).indices.sort(dim=-1).values
    return torch.where(idxs < compress_lens, idxs + offset, -1).int()


def _prefill_scores(seqlen, ratio, seed):
    g = torch.Generator().manual_seed(seed)
    n_blocks = seqlen // ratio
    scores = torch.randn(1, seqlen, n_blocks, generator=g)
    compress_lens = (torch.arange(1, seqlen + 1) // ratio).unsqueeze(-1)
    scores.masked_fill_(torch.arange(n_blocks) >= compress_lens, -torch.inf)
    return scores, compress_lens


@pytest.mark.parametrize("seqlen,ratio,topk_blocks,block_size", [
    (37, 1, 4, 8), (300, 2, 4, 8), (64, 1, 100, 8), (9, 1, 2, 3),
])
def test_select_candidate_blocks_matches_reference_prefill(seqlen, ratio, topk_blocks, block_size):
    scores, compress_lens = _prefill_scores(seqlen, ratio, seed=seqlen)
    ref = _ref_select_candidate_blocks(scores, compress_lens, topk_blocks, block_size)
    got = select_candidate_blocks(scores, compress_lens, topk_blocks, block_size)
    assert torch.equal(got, ref)
    # the block-id form is the same selection: expanding it reproduces the mask
    ids = select_candidate_block_ids(scores, compress_lens, topk_blocks, block_size)
    assert ids.shape[-1] == topk_blocks
    assert torch.equal(candidate_mask_from_blocks(ids, scores.size(-1), block_size), ref)
    # a query that can reach nothing (position < ratio) keeps nothing
    assert not got[0, 0].any() if compress_lens[0].item() == 0 else True


@pytest.mark.parametrize("n_positions,topk_blocks,block_size", [(5, 4, 8), (77, 4, 8), (128, 64, 8)])
def test_select_candidate_blocks_matches_reference_decode(n_positions, topk_blocks, block_size):
    g = torch.Generator().manual_seed(n_positions)
    scores = torch.randn(3, 1, n_positions, generator=g)
    ref = _ref_select_candidate_blocks(scores, n_positions, topk_blocks, block_size)
    got = select_candidate_blocks(scores, n_positions, topk_blocks, block_size)
    assert torch.equal(got, ref)
    # the newest (partly filled) block is always pinned, whatever its scores
    newest = (n_positions - 1) // block_size
    assert got[..., newest * block_size: n_positions].all()
    ids = select_candidate_block_ids(scores, n_positions, topk_blocks, block_size)
    assert torch.equal(candidate_mask_from_blocks(ids, n_positions, block_size), ref)
    assert (ids == newest).any(dim=-1).all()


def test_candidate_mask_from_blocks_ignores_minus_one():
    ids = torch.tensor([[-1, -1, 2], [1, -1, -1]])
    mask = candidate_mask_from_blocks(ids, n_positions=20, block_size=8)
    assert mask.tolist() == [
        [False] * 16 + [True] * 4,
        [False] * 8 + [True] * 8 + [False] * 4,
    ]


@pytest.mark.parametrize("seqlen,ratio,index_topk", [(41, 1, 8), (256, 2, 16), (10, 1, 64)])
@pytest.mark.parametrize("with_candidates", [False, True])
def test_indexer_select_prefill_matches_reference(seqlen, ratio, index_topk, with_candidates):
    scores, compress_lens = _prefill_scores(seqlen, ratio, seed=seqlen * 7)
    n_blocks = seqlen // ratio
    cands = select_candidate_blocks(scores, compress_lens, 2, 8) if with_candidates else None
    ref = _ref_indexer_tail(scores, compress_lens, n_blocks, index_topk, 128, cands)
    got = indexer_select_prefill(scores, compress_lens, index_topk, cands, offset=128)
    assert torch.equal(got.int(), ref)
    # the helper masks reach itself: an unmasked slab gives the same pick
    raw = torch.where(torch.isinf(scores), torch.zeros_like(scores), scores)
    assert torch.equal(
        indexer_select_prefill(raw, compress_lens, index_topk, cands, offset=128).int(), ref
    )
    # -1 exactly where the pick lies past the query's reach (without candidates every reachable
    # position outscores -inf, so the live count is min(reach, k)); the rest ascend by position
    live = got >= 0
    if not with_candidates:
        k = min(index_topk, n_blocks)
        assert torch.equal(live.sum(-1)[0], compress_lens.squeeze(-1).clamp_max(k))
    for s in range(seqlen):
        row = got[0, s][live[0, s]]
        assert torch.equal(row, row.sort().values)


def test_indexer_select_decode_matches_reference():
    g = torch.Generator().manual_seed(3)
    n_positions, index_topk = 50, 16
    scores = torch.randn(4, 1, n_positions, generator=g)
    cands = select_candidate_blocks(scores, n_positions, 3, 8)
    ref = _ref_indexer_tail(scores, n_positions, n_positions, index_topk, 0, cands)
    got = indexer_select_prefill(scores, n_positions, index_topk, cands)
    assert torch.equal(got.int(), ref)
    assert (got >= 0).all()  # decode: every position is reachable
