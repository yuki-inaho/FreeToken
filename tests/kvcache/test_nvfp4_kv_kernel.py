"""NVFP4 K/V cache: the real store kernel, the real attend dequant, and the real pool.

tests/kvcache/test_nvfp4_kv.py pins the *numerics* of the format through a torch fake-quant
probe. This file pins the shipped path -- ``qsa_store_kv_nvfp4``, the ``KV_NVFP4`` branch of the
QSA attend kernel, and ``QSAKVCache``'s packed slab + block-scale tier -- against that probe.
The split matters: the probe certified that NVFP4 KV is good enough to build, and these tests
certify that what got built computes the probe's answer.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

HEADS, HEAD_DIM = 4, 128


def _tp():
    from freetoken.distributed import set_tp_info, try_get_tp_info

    if try_get_tp_info() is None:
        set_tp_info(0, 1)


def _store(k, v, out_loc, slots):
    """Run the shipped store kernel into freshly allocated tiers; return them."""
    from freetoken.kernel.triton.qsa.store_nvfp4 import KV_BLOCK, qsa_store_kv_nvfp4

    dev = k.device
    kc = torch.zeros(slots, HEADS, HEAD_DIM // 2, dtype=torch.uint8, device=dev)
    vc = torch.zeros_like(kc)
    kb = torch.zeros(slots, HEADS, HEAD_DIM // KV_BLOCK, dtype=torch.float8_e4m3fn, device=dev)
    vb = torch.zeros_like(kb)
    kg = torch.zeros(slots, HEADS, dtype=torch.float32, device=dev)
    vg = torch.zeros_like(kg)
    qsa_store_kv_nvfp4(k, v, out_loc, kc, vc, kb, vb, kg, vg)
    return kc, vc, kb, vb, kg, vg


def test_store_kernel_matches_torch_reference():
    """The Triton store must produce exactly what nvfp4_fake_quant would -- same E2M1 grid,
    same two-level scale split. Any drift here invalidates the probe's quality gates."""
    from freetoken.kvcache.nvfp4_kv import dequant_packed, nvfp4_fake_quant

    torch.manual_seed(0)
    T, slots = 7, 16
    k = (torch.randn(T, HEADS * HEAD_DIM, device="cuda") * 2).to(torch.bfloat16)
    v = torch.randn(T, HEADS * HEAD_DIM, device="cuda").to(torch.bfloat16)
    loc = torch.arange(T, dtype=torch.int32, device="cuda")
    kc, _, kb, _, kg, _ = _store(k, v, loc, slots)

    got = dequant_packed(kc.cpu(), kb.cpu(), kg.cpu(), HEAD_DIM)[:T]
    want = nvfp4_fake_quant(k.float().cpu().view(T, HEADS, HEAD_DIM), global_dtype=torch.float32)
    assert torch.allclose(got, want, rtol=0, atol=1e-5), (got - want).abs().max()


def test_store_skips_negative_out_loc():
    """out_loc < 0 marks a padded row; the kernel must leave that slot untouched. A written
    slot would corrupt a live token's K/V, since padding rows carry garbage."""
    torch.manual_seed(1)
    T, slots = 5, 8
    k = torch.randn(T, HEADS * HEAD_DIM, device="cuda").to(torch.bfloat16)
    loc = torch.tensor([0, -1, 2, -1, 4], dtype=torch.int32, device="cuda")
    kc, _, kb, _, kg, _ = _store(k, k.clone(), loc, slots)
    for skipped in (1, 3, 5, 6, 7):
        assert not kc[skipped].any() and not kg[skipped].any()
    for written in (0, 2, 4):
        assert kg[written].any()


def test_zero_row_dequantizes_to_finite_zero():
    """A never-written slot (the dummy page, padded rows) must read as 0, not NaN: the block
    scale's denominator is the row global, which underflows to 0 for an all-zero row."""
    from freetoken.kvcache.nvfp4_kv import dequant_packed

    slots = 8
    k = torch.zeros(4, HEADS * HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    loc = torch.arange(4, dtype=torch.int32, device="cuda")
    kc, _, kb, _, kg, _ = _store(k, k.clone(), loc, slots)
    out = dequant_packed(kc.cpu(), kb.cpu(), kg.cpu(), HEAD_DIM)
    assert torch.isfinite(out).all() and not out.any()


def test_attend_matches_exact_attention_over_the_same_values():
    """The attend kernel's in-register dequant must reconstruct the stored values.

    Measured against fp32 torch attention over exactly what the slab holds, with the 16-bit
    cache path as the yardstick: the NVFP4 branch has to be at least as accurate as running
    the same values through a bf16 cache. That is the load-bearing check, because NVFP4's
    block scale varies ALONG the reduction dim -- unlike fp8's, it cannot be folded onto the
    scores after tl.dot, and an implementation that folded it anyway would still look
    plausible while being wrong per block. (Only the ROW global folds out, which is what the
    kernel does, and why it can end up slightly *more* accurate than the bf16 reference: that
    multiply lands in fp32 after the dot instead of inside a bf16 operand.)
    """
    from freetoken.kernel.triton.qsa.attend import qsa_sparse_paged_attention
    from freetoken.kvcache.nvfp4_kv import dequant_packed

    torch.manual_seed(7)
    pages, page_size, qh, rows, topk = 8, 16, 16, 6, 32
    slots = pages * page_size
    q = torch.randn(rows, qh, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    kt = (torch.randn(slots, HEADS * HEAD_DIM, device="cuda") * 1.7).to(torch.bfloat16)
    vt = torch.randn(slots, HEADS * HEAD_DIM, device="cuda").to(torch.bfloat16)
    loc = torch.arange(slots, dtype=torch.int32, device="cuda")
    kc, vc, kb, vb, kg, vg = _store(kt, vt, loc, slots)

    kd = dequant_packed(kc.cpu(), kb.cpu(), kg.cpu(), HEAD_DIM).to("cuda")
    vd = dequant_packed(vc.cpu(), vb.cpu(), vg.cpu(), HEAD_DIM).to("cuda")
    shape = (pages, page_size, HEADS, HEAD_DIM)
    k16 = kd.to(torch.bfloat16).view(shape).contiguous()
    v16 = vd.to(torch.bfloat16).view(shape).contiguous()

    # Every query attends to the same 4 slots, so the exact answer is computable in torch.
    sel = [0, 1, 2, 3]
    idx = torch.tensor([sel] * rows, dtype=torch.int32, device="cuda")
    table = torch.arange(pages, dtype=torch.int32, device="cuda").repeat(2, 1)
    t2r = torch.zeros(rows, dtype=torch.int32, device="cuda")

    group = qh // HEADS
    exact = torch.empty(rows, qh, HEAD_DIM, device="cuda", dtype=torch.float32)
    for h in range(qh):
        kv_h = h // group
        scores = (q.float()[:, h] @ kd[sel, kv_h].T) * HEAD_DIM**-0.5
        exact[:, h] = torch.softmax(scores, dim=-1) @ vd[sel, kv_h]

    def rel(x):
        return ((x.float() - exact).norm() / exact.norm()).item()

    bf16_path = rel(qsa_sparse_paged_attention(q, k16, v16, idx, table, t2r))
    nvfp4_path = rel(qsa_sparse_paged_attention(
        q, kc.view(shape[:3] + (HEAD_DIM // 2,)), vc.view(shape[:3] + (HEAD_DIM // 2,)),
        idx, table, t2r, k_scale=kg, v_scale=vg, k_blk=kb, v_blk=vb,
    ))
    assert nvfp4_path <= bf16_path * 1.05, (nvfp4_path, bf16_path)
    assert nvfp4_path < 0.01, nvfp4_path


def test_attend_applies_the_block_scale_before_the_dot():
    """A per-block scale folded onto the scores instead of the keys is the plausible-but-wrong
    implementation. Pin it with keys whose blocks have wildly different magnitudes, where the
    two spellings cannot agree."""
    from freetoken.kernel.triton.qsa.attend import qsa_sparse_paged_attention
    from freetoken.kvcache.nvfp4_kv import dequant_packed

    torch.manual_seed(11)
    pages, page_size, qh, rows, topk = 4, 16, 4, 2, 8
    slots = pages * page_size
    kt = torch.randn(slots, HEADS * HEAD_DIM, device="cuda")
    # Give each 16-element block its own decade, so one block scale cannot stand in for another.
    decades = torch.tensor([1e-2, 1e-1, 1.0, 10.0] * (HEAD_DIM // 16 // 4), device="cuda")
    kt = (kt.view(slots, HEADS, HEAD_DIM // 16, 16) * decades[None, None, :, None])
    kt = kt.reshape(slots, HEADS * HEAD_DIM).to(torch.bfloat16)
    vt = torch.randn(slots, HEADS * HEAD_DIM, device="cuda").to(torch.bfloat16)
    loc = torch.arange(slots, dtype=torch.int32, device="cuda")
    kc, vc, kb, vb, kg, vg = _store(kt, vt, loc, slots)

    kd = dequant_packed(kc.cpu(), kb.cpu(), kg.cpu(), HEAD_DIM).to("cuda")
    vd = dequant_packed(vc.cpu(), vb.cpu(), vg.cpu(), HEAD_DIM).to("cuda")
    q = torch.randn(rows, qh, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    sel = list(range(topk))
    idx = torch.tensor([sel] * rows, dtype=torch.int32, device="cuda")
    table = torch.arange(pages, dtype=torch.int32, device="cuda").repeat(2, 1)
    t2r = torch.zeros(rows, dtype=torch.int32, device="cuda")
    shape = (pages, page_size, HEADS, HEAD_DIM)

    got = qsa_sparse_paged_attention(
        q, kc.view(shape[:3] + (HEAD_DIM // 2,)), vc.view(shape[:3] + (HEAD_DIM // 2,)),
        idx, table, t2r, k_scale=kg, v_scale=vg, k_blk=kb, v_blk=vb,
    )
    group = qh // HEADS
    for h in range(qh):
        kv_h = h // group
        scores = (q.float()[:, h] @ kd[sel, kv_h].T) * HEAD_DIM**-0.5
        exact = torch.softmax(scores, dim=-1) @ vd[sel, kv_h]
        r = ((got.float()[:, h] - exact).norm() / exact.norm()).item()
        assert r < 0.02, (h, r)


def test_pool_allocates_packed_slab_and_block_scales():
    """The slab is uint8 at half the head_dim (two codes per byte); the block tier is e4m3 at
    head_dim // 16. QSAKVCache.kv_dtype stays the NVFP4 sentinel so callers can tell the
    formats apart -- the storage dtype alone cannot, since fp8 and packed NVFP4 are both 1 B."""
    from freetoken.kvcache.qsa_pool import NVFP4_KV_BLOCK, NVFP4_KV_DTYPE, QSAKVCache

    _tp()
    pool = QSAKVCache(
        num_kv_heads=HEADS, num_layers=2, head_dim=HEAD_DIM, num_pages=8, page_size=16,
        device=torch.device("cuda"), dtype=torch.bfloat16, index_head_dim=64,
        num_index_layers=2, index_ratio=4, num_req_slots=2, kv_dtype=NVFP4_KV_DTYPE,
    )
    assert pool._k_buffer.dtype == torch.uint8
    assert pool._k_buffer.shape[-1] == HEAD_DIM // 2
    assert pool._kv_blk.shape == (2, 2, 8 * 16, HEADS, HEAD_DIM // NVFP4_KV_BLOCK)
    assert pool._kv_blk.dtype == torch.float8_e4m3fn
    assert pool.kv_quantized and pool.kv_nvfp4 and pool.kv_dtype == NVFP4_KV_DTYPE
    assert pool.dtype == torch.bfloat16  # compute dtype is unchanged


def test_pool_store_and_rebuild_keep_both_scale_tiers():
    """rebuild() must re-derive the block tier at the new page count and zero it. The earlier
    elastic-KV crash came from exactly this class of bug: a tier that survives a realloc stale."""
    from freetoken.kvcache.nvfp4_kv import dequant_packed
    from freetoken.kvcache.qsa_pool import NVFP4_KV_DTYPE, QSAKVCache

    _tp()
    torch.manual_seed(3)
    pool = QSAKVCache(
        num_kv_heads=HEADS, num_layers=2, head_dim=HEAD_DIM, num_pages=8, page_size=16,
        device=torch.device("cuda"), dtype=torch.bfloat16, index_head_dim=64,
        num_index_layers=2, index_ratio=4, num_req_slots=2, kv_dtype=NVFP4_KV_DTYPE,
    )
    k = (torch.randn(20, HEADS * HEAD_DIM, device="cuda") * 1.4).to(torch.bfloat16)
    v = torch.randn(20, HEADS * HEAD_DIM, device="cuda").to(torch.bfloat16)
    pool.store_kv(k, v, torch.arange(20, dtype=torch.int32, device="cuda"), 0)
    got = dequant_packed(
        pool.k_cache(0).reshape(-1, HEADS, HEAD_DIM // 2).cpu(),
        pool.k_blk(0).cpu(), pool.k_scale(0).cpu(), HEAD_DIM,
    )[:20]
    ref = k.float().cpu().view(20, HEADS, HEAD_DIM)
    assert ((ref - got).norm() / ref.norm()).item() < 0.15

    pool.rebuild(12)
    assert pool._kv_blk.shape[2] == 12 * 16
    assert not pool._kv_blk.float().any()
    assert not pool._kv_scale.any()


def test_row_bytes_prices_every_tier():
    """spec_kv_bytes_per_token budgets the pool through kv_row_bytes; if it under-counts, the
    startup solve hands out pages the pool cannot allocate."""
    from freetoken.kvcache.base import kv_row_bytes
    from freetoken.kvcache.qsa_pool import NVFP4_KV_DTYPE

    assert kv_row_bytes(128, torch.bfloat16) == 256
    assert kv_row_bytes(128, torch.float8_e4m3fn) == 128 + 4
    # 64 packed code bytes + 8 e4m3 block scales + one fp32 row global
    assert kv_row_bytes(128, NVFP4_KV_DTYPE) == 64 + 8 + 4


def test_pool_unit_bytes_matches_row_arithmetic():
    """The measured per-token cost must equal 2 slabs x layers x heads x kv_row_bytes, so the
    budget solver and the allocator agree."""
    from freetoken.kvcache.base import kv_row_bytes
    from freetoken.kvcache.qsa_pool import NVFP4_KV_DTYPE, QSAKVCache

    _tp()
    pool = QSAKVCache(
        num_kv_heads=HEADS, num_layers=2, head_dim=HEAD_DIM, num_pages=8, page_size=16,
        device=torch.device("cuda"), dtype=torch.bfloat16, index_head_dim=64,
        num_index_layers=2, index_ratio=4, num_req_slots=2, kv_dtype=NVFP4_KV_DTYPE,
    )
    kv_term = 2 * 2 * HEADS * kv_row_bytes(HEAD_DIM, NVFP4_KV_DTYPE)
    index_term = 2 * 64 * 2 // 4  # layers x index_head_dim x 2 B / ratio
    assert pool.unit_bytes()[0] == kv_term + index_term


def test_head_dim_must_be_block_aligned():
    """A head_dim that is not a multiple of 32 would leave a partial packed byte or a partial
    block scale; reject it at construction rather than silently truncating."""
    from freetoken.kvcache.qsa_pool import NVFP4_KV_DTYPE, QSAKVCache

    _tp()
    with pytest.raises(ValueError, match="divisible by 32"):
        QSAKVCache(
            num_kv_heads=HEADS, num_layers=1, head_dim=48, num_pages=4, page_size=16,
            device=torch.device("cuda"), dtype=torch.bfloat16, index_head_dim=48,
            num_index_layers=1, index_ratio=4, num_req_slots=2, kv_dtype=NVFP4_KV_DTYPE,
        )


def test_nvfp4_pool_skips_the_fused_decode_store():
    """The fused decode kernel writes the fp8 slab. NVFP4 must take the separate store_kv path
    or it would write e4m3 codes into a packed uint8 slab -- silent corruption, not a crash."""
    import inspect

    from freetoken.attention import qsa_sparse

    # Pin the guard inside the `fused = (...)` predicate itself, not merely somewhere in the
    # module: the whole point is that NVFP4 must be excluded from that one condition.
    src = inspect.getsource(qsa_sparse)
    gate = src.split("fused = (", 1)[1].split("if not fused", 1)[0]
    assert "not self.kvcache.kv_nvfp4" in gate, gate
