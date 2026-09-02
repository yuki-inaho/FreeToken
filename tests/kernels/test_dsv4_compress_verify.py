"""DSV4 compressor for a speculative verify: ``Compressor.verify_rows`` folds k+1 consecutive
tokens of one request through one carry read and must reproduce, bit for bit, k+1 sequential
``decode_step`` calls -- the compressed rows it scatters, the ring block it leaves, and the
per-row carry intermediates; ``commit_carry`` must re-point the ring to any accepted prefix
(including the old page's boundary carry when the rows cross a 128-page boundary). GPU
(the reduction is a Triton kernel), random weights, both ratios and both tiers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.attention.dsv4_sparse import DSV4AttnMetadata, DSV4SparseAttnBackend
from freetoken.core import Batch, Context, Req, SamplingParams, get_global_ctx, set_global_ctx
from freetoken.kvcache.dsv4_cost_model import dsv4_pool_sizes
from freetoken.kvcache.dsv4_paged_pool import DSV4PagedKVCache
from freetoken.models.deepseek_v4.args import DeepseekV4Args
from freetoken.models.deepseek_v4.compress import Compressor
from freetoken.models.deepseek_v4.ops import precompute_freqs_cis

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

P, TI, N_SCRATCH = 128, 2, 8
RATIOS = (0, 4, 128, 4)
DEVICE = torch.device("cuda")


def _stack(max_seq_len=2048, num_pages=16):
    args = DeepseekV4Args(
        max_batch_size=2, dim=256, n_layers=len(RATIOS), n_heads=4, q_lora_rank=64,
        o_lora_rank=64, o_groups=2, moe_inter_dim=64, n_routed_experts=4,
        n_activated_experts=2, vocab_size=64, index_n_heads=2, index_topk=8,
        compress_ratios=RATIOS, max_seq_len=max_seq_len, head_dim=512, index_head_dim=128,
        window_size=P,
    )
    sizes = dsv4_pool_sizes(num_pages=num_pages + 1, args=args, swa_ratio=1.0, P=P)
    pool = DSV4PagedKVCache(sizes=sizes, args=args, device=DEVICE, P=P, n_scratch=N_SCRATCH)
    pool._init_paged_state(1, True)
    pt = torch.zeros(3, max_seq_len, dtype=torch.int32, device=DEVICE)
    pt[TI, :512] = torch.arange(512, dtype=torch.int32, device=DEVICE)
    for page in range(4):  # bind row TI's window pages (positions 0..511)
        pool.bind_window_pages(page * P, page * P)
    pool.full_loc_map = pt
    try:
        ctx = get_global_ctx()
    except AssertionError:
        ctx = Context(page_size=P)
        set_global_ctx(ctx)
    ctx.kv_cache = pool
    backend = DSV4SparseAttnBackend(SimpleNamespace(dsv4_args=args))
    ctx.attn_backend = backend
    return args, pool, pt, backend


def _batch(n_rows: int, pt: torch.Tensor, backend) -> Batch:
    # A decode-shaped batch whose snapshot has n_rows copies of row TI (a verify batch).
    reqs = [
        Req(input_ids=torch.zeros(1, dtype=torch.int32), table_idx=TI, cached_len=0,
            output_len=1, uid=i, sampling_params=SamplingParams(), cache_handle=None)
        for i in range(n_rows)
    ]
    batch = Batch(reqs=reqs, phase="decode")
    batch.padded_reqs = reqs
    batch.attn_metadata = DSV4AttnMetadata(
        last_indices=torch.arange(n_rows, dtype=torch.int32, device=DEVICE),
        full_snap=pt[TI].to(torch.int64).unsqueeze(0).expand(n_rows, -1).contiguous(),
        window_ar=backend._window_ar,
    )
    return batch


def _compressor(args, layer_id: int, tier: str, pool, seed: int) -> Compressor:
    ratio = RATIOS[layer_id]
    head_dim = args.head_dim if tier == "attn" else args.index_head_dim
    torch.manual_seed(seed)
    c = Compressor(args, ratio, head_dim, rotate=(tier == "idx")).to(DEVICE)
    with torch.no_grad():
        c.ape.normal_(0, 0.5)
        c.wkv.weight.normal_(0, 0.05)
        c.wgate.weight.normal_(0, 0.05)
        c.norm.weight.uniform_(0.5, 1.5)
    freqs = precompute_freqs_cis(
        args.rope_head_dim, args.max_seq_len, args.original_seq_len, args.compress_rope_theta,
        args.rope_factor, args.beta_fast, args.beta_slow,
    ).to(DEVICE)
    c.bind_paged(pool, layer_id, freqs, DEVICE, tier=tier)
    return c


def _ring(pool, layer_id, tier):
    return pool.state_ring[layer_id] if tier == "attn" else pool.indexer_state_ring[layer_id]


def _cmp_pool(pool, layer_id, tier):
    return pool.cmp_pool[layer_id] if tier == "attn" else pool.idx_pool[layer_id]


def _boundary_row(start: int, k1: int) -> int | None:
    # The first verify row that starts a new window page (positions start..start+k1-1).
    for j in range(1, k1):
        if (start + j) % P == 0:
            return j
    return None


def _close(a: torch.Tensor, b: torch.Tensor) -> bool:
    # -inf score slots compare equal; finite entries within fp32 GEMM reduction-order noise.
    return torch.allclose(a, b, rtol=1e-5, atol=1e-5, equal_nan=False)


def _seed_ring(pool, layer_id, tier, seed):
    # A random but structurally valid carry in every ring block (finite scores).
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    ring = _ring(pool, layer_id, tier)
    ring.buffer.normal_(0, 1, generator=g)
    ring._clear_scratch()


@pytest.mark.parametrize("layer_id,tier", [(1, "attn"), (2, "attn"), (3, "idx")])
@pytest.mark.parametrize("start", [250, 254, 379])  # inside a page / crossing 256 / crossing 384
def test_verify_rows_matches_sequential_decode_steps(layer_id, tier, start):
    args, pool, pt, backend = _stack()
    ratio = RATIOS[layer_id]
    k1 = 6
    comp = _compressor(args, layer_id, tier, pool, seed=layer_id * 10 + 1)
    translate = pool.translate_full_to_window
    positions = torch.arange(start, start + k1, device=DEVICE)
    x_rows = torch.randn(k1, args.dim, device=DEVICE, dtype=torch.bfloat16)
    boundary_row = _boundary_row(start, k1)

    # ---- reference: k1 sequential single-row decode steps ----
    _seed_ring(pool, layer_id, tier, seed=7)
    ring = _ring(pool, layer_id, tier)
    cmp = _cmp_pool(pool, layer_id, tier)
    cmp.zero_()
    ring0 = ring.buffer.clone()
    ref_blocks = []  # the ring block of row i's page after step i
    for i in range(k1):
        batch = _batch(1, pt, backend)
        with get_global_ctx().forward_batch(batch):
            pos = positions[i : i + 1]
            comp.decode_step(
                x_rows[i : i + 1].view(1, 1, -1), pos,
                translate(pt[TI, pos - 1]), translate(pt[TI, pos]),
                torch.zeros(1, dtype=torch.int64, device=DEVICE),
            )
        base = backend.ring_page_base(translate(pt[TI, positions[i : i + 1]]), comp.ring_size)
        ref_blocks.append(ring.get_blocks(base)[0].clone())
    ref_ring = ring.buffer.clone()
    ref_cmp = cmp.clone()

    # ---- chained: one verify_rows call over the same k1 rows ----
    ring.buffer.copy_(ring0)
    cmp.zero_()
    intermediates = torch.zeros(k1, comp.ring_size, 2 * comp.item_size, device=DEVICE)
    batch = _batch(k1, pt, backend)
    with get_global_ctx().forward_batch(batch):
        comp.verify_rows(
            x_rows, positions, translate(pt[TI, positions[:1] - 1]),
            translate(pt[TI, positions]), torch.arange(k1, device=DEVICE), intermediates,
            boundary_row=boundary_row,
        )
    n_scratch_rows = cmp.shape[0] - backend.compress_scratch_base(layer_id, tier)
    live = backend.compress_scratch_base(layer_id, tier)
    # Compressed rows: identical on every non-scratch row (scratch rows are discarded writes).
    assert torch.equal(cmp[:live], ref_cmp[:live])
    completed = ((positions + 1) % ratio == 0).sum().item()
    assert (cmp[:live] != 0).any(dim=1).sum().item() == completed
    assert n_scratch_rows == N_SCRATCH
    # Every row's intermediate is the sequential run's post-step block. The carries are fp32
    # and their kv/score projections come from a k1-row vs a 1-row GEMM (different reduction
    # order -> last-ulp differences, ~5e-7), the same M-dependent numerics the batched decode
    # path already has across batch sizes; the fp8/fp4-quantized compressed rows above absorb
    # it and stay bit-identical.
    for i in range(k1):
        assert _close(intermediates[i], ref_blocks[i]), f"row {i}"
    # The ring as a whole matches the accept-all sequential run (last page's block written).
    assert _close(ring.buffer, ref_ring)


@pytest.mark.parametrize("layer_id,tier", [(1, "attn"), (2, "attn"), (3, "idx")])
@pytest.mark.parametrize("start", [250, 254])
def test_commit_carry_restores_any_accepted_prefix(layer_id, tier, start):
    args, pool, pt, backend = _stack()
    k1 = 6
    comp = _compressor(args, layer_id, tier, pool, seed=layer_id * 10 + 2)
    translate = pool.translate_full_to_window
    positions = torch.arange(start, start + k1, device=DEVICE)
    x_rows = torch.randn(k1, args.dim, device=DEVICE, dtype=torch.bfloat16)
    boundary_row = _boundary_row(start, k1)
    ring = _ring(pool, layer_id, tier)

    # Reference ring after exactly n sequential steps, for every n.
    _seed_ring(pool, layer_id, tier, seed=11)
    ring0 = ring.buffer.clone()
    ref_after = []
    for n in range(1, k1 + 1):
        ring.buffer.copy_(ring0)
        for i in range(n):
            batch = _batch(1, pt, backend)
            with get_global_ctx().forward_batch(batch):
                pos = positions[i : i + 1]
                comp.decode_step(
                    x_rows[i : i + 1].view(1, 1, -1), pos,
                    translate(pt[TI, pos - 1]), translate(pt[TI, pos]),
                    torch.zeros(1, dtype=torch.int64, device=DEVICE),
                )
        ref_after.append(ring.buffer.clone())

    # Chained verify, then commit each possible acceptance count.
    intermediates = torch.zeros(k1, comp.ring_size, 2 * comp.item_size, device=DEVICE)
    window_slots = translate(pt[TI, positions])
    for n in range(1, k1 + 1):
        ring.buffer.copy_(ring0)
        batch = _batch(k1, pt, backend)
        with get_global_ctx().forward_batch(batch):
            comp.verify_rows(
                x_rows, positions, translate(pt[TI, positions[:1] - 1]), window_slots,
                torch.arange(k1, device=DEVICE), intermediates, boundary_row=boundary_row,
            )
            comp.commit_carry(
                intermediates, torch.tensor([n], device=DEVICE), window_slots, boundary_row
            )
        # Compare only the pages the verify touched (the rest of the ring is untouched by both).
        pages = sorted({int(s) // P for s in window_slots.tolist()})
        for page in pages:
            rows = torch.arange(page * comp.ring_size, (page + 1) * comp.ring_size, device=DEVICE)
            got, want = ring.buffer[rows], ref_after[n - 1][rows]
            if boundary_row is not None and n <= boundary_row and page == pages[-1]:
                # Acceptance stopped before the boundary: the NEW page's block is garbage
                # from rejected rows in the chained run and untouched in the reference --
                # never read (the next step reads the old page; a re-issue rewrites it).
                continue
            assert _close(got, want), f"n={n} page={page}"
