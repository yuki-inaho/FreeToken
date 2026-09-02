"""The FP8 QSA K/V pool (--kv-cache-dtype fp8): the quantizing store, the attend kernel's
in-register dequantization, and the real Qwen4ExpAttention layer on top of it.

Two kinds of claims, kept apart:
  * KERNEL PARITY -- the fp8 kernel over quantized codes + scales must equal attention over
    the very values those codes decode to (an fp32 torch reference), to bf16-dot precision;
    and the 16-bit kernel over the bf16-rounded decode must sit at the same distance. This
    is what catches a broken scale fold, a scaled denominator, or an unmasked scale gather.
  * QUANTIZATION ERROR -- the fp8 layer vs the fp32 dense oracle over the ORIGINAL bf16 K/V,
    at a loosened tolerance (e4m3 keeps ~2.6% RMS per element).
GPU only (Triton); the shipping QSA geometry through the test fixture.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from .common import Fixture, parsed_config, requires_cuda, selection_spy

FP8 = torch.float8_e4m3fn
QSA_LAYER = 3
pytestmark = requires_cuda


def _dequant(codes: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """``codes`` [..., slots, H, D] e4m3, ``scale`` [..., slots, H] fp32 -> fp32 values."""
    return codes.float() * scale.unsqueeze(-1)


# --------------------------------------------------------------------------- store
def test_store_round_trip_is_within_e4m3_precision():
    config = parsed_config()
    fixture = Fixture(config, num_pages=8, kv_dtype=FP8)
    pool = fixture.pool
    assert pool.kv_quantized and pool.dtype == torch.bfloat16 and pool.kv_dtype == FP8
    heads, head_dim = pool.k_cache(QSA_LAYER).shape[2], pool.k_cache(QSA_LAYER).shape[3]
    T = 37
    gen = torch.Generator(device="cuda").manual_seed(3)
    k = torch.randn(T, heads * head_dim, device="cuda", dtype=torch.bfloat16, generator=gen) * 3
    v = torch.randn(T, heads * head_dim, device="cuda", dtype=torch.bfloat16, generator=gen)
    k[5] = 0  # an all-zero row must quantize to zeros with a finite scale
    v[6, :head_dim] = 0
    out_loc = torch.arange(T, device="cuda", dtype=torch.int32) * 3 + 2  # scattered slots
    pool.store_kv(k, v, out_loc, QSA_LAYER)

    slots = pool.k_cache(QSA_LAYER).shape[0] * pool.k_cache(QSA_LAYER).shape[1]
    k_codes = pool.k_cache(QSA_LAYER).reshape(slots, heads, head_dim)
    v_codes = pool.v_cache(QSA_LAYER).reshape(slots, heads, head_dim)
    k_deq = _dequant(k_codes, pool.k_scale(QSA_LAYER))[out_loc.long()]
    v_deq = _dequant(v_codes, pool.v_scale(QSA_LAYER))[out_loc.long()]
    for got, want in ((k_deq, k.float().view(T, heads, head_dim)),
                      (v_deq, v.float().view(T, heads, head_dim))):
        assert torch.isfinite(got).all()
        amax = want.abs().amax(dim=-1, keepdim=True)
        # e4m3 under a per-row amax/448 scale: |err| <= |x|/16 (3 mantissa bits) plus the
        # subnormal floor of the row (scale * 2^-9 -> amax/448 * 2^-9 * 448/448...), which
        # a generous amax * 2^-9 covers.
        assert (got - want).abs().le(want.abs() / 16 + amax * 2 ** -9 + 1e-6).all()
    assert torch.equal(k_deq[5], torch.zeros_like(k_deq[5]))
    # Slots that were never written stay zero code / zero scale -> finite zero.
    untouched = torch.tensor([0, 1], device="cuda")
    assert torch.equal(_dequant(k_codes, pool.k_scale(QSA_LAYER))[untouched], torch.zeros(2, heads, head_dim, device="cuda"))
    assert pool.k_scale(QSA_LAYER)[untouched].eq(0).all()


# --------------------------------------------------------------------------- kernel parity
def _reference_attention(q, k_vals, v_vals, indices, block_table, token_to_req, page_size):
    """fp32 sparse GQA over dequantized K/V: ``k_vals``/``v_vals`` [slots, H, D] fp32."""
    rows, hq, d = q.shape
    heads = k_vals.shape[1]
    group = hq // heads
    out = torch.zeros(rows, hq, d, device=q.device, dtype=torch.float32)
    qf = q.float()
    for r in range(rows):
        req = int(token_to_req[r])
        toks = indices[r][indices[r] >= 0].long()
        pages = block_table[req, toks // page_size].long()
        slots = pages * page_size + toks % page_size
        for h in range(hq):
            kvh = h // group
            kk = k_vals[slots, kvh]  # [n, D]
            vv = v_vals[slots, kvh]
            s = (kk @ qf[r, h]) * d ** -0.5
            p = torch.softmax(s, dim=0)
            out[r, h] = p @ vv
    return out


@pytest.mark.parametrize("rows", [1, 40], ids=["decode-profile", "prefill-profile"])
def test_attend_kernel_matches_reference_over_dequantized_kv(rows: int):
    from freetoken.kernel.triton.qsa import qsa_sparse_paged_attention, qsa_store_kv_fp8

    torch.manual_seed(7)
    pages, page_size, heads, d, hq = 16, 64, 2, 256, 24
    topk = 2051
    device = torch.device("cuda")
    slots = pages * page_size
    k = torch.randn(slots, heads * d, device=device, dtype=torch.bfloat16)
    v = torch.randn(slots, heads * d, device=device, dtype=torch.bfloat16)
    k_cache = torch.zeros(pages, page_size, heads, d, device=device, dtype=FP8)
    v_cache = torch.zeros_like(k_cache)
    k_scale = torch.zeros(slots, heads, device=device, dtype=torch.float32)
    v_scale = torch.zeros_like(k_scale)
    # Leave page 15 never-written (zero codes, zero scales) -- the "dummy page" case.
    written = torch.arange(15 * page_size, device=device, dtype=torch.int32)
    qsa_store_kv_fp8(
        k[: written.numel()], v[: written.numel()], written,
        k_cache.view(slots, heads, d), v_cache.view(slots, heads, d), k_scale, v_scale,
    )
    k_vals = _dequant(k_cache.view(slots, heads, d), k_scale)
    v_vals = _dequant(v_cache.view(slots, heads, d), v_scale)

    # One request whose logical pages map to physical pages in a shuffled order, plus the
    # never-written page at the end so some valid indices gather zero code / zero scale.
    block_table = torch.randperm(15, device=device)[None, :].to(torch.int32)
    block_table = torch.cat([block_table, torch.tensor([[15]], device=device, dtype=torch.int32)], dim=1)
    n_tokens = 16 * page_size  # 1,024 tokens; the 2,051-wide selection is mostly -1 padding
    q = torch.randn(rows, hq, d, device=device, dtype=torch.bfloat16)
    indices = torch.full((rows, topk), -1, device=device, dtype=torch.int32)
    for r in range(rows):
        # row 0 selects every token (incl. the never-written page); the others a random subset
        n_sel = torch.randint(1, n_tokens, (1,)).item() if r else n_tokens
        indices[r, :n_sel] = torch.randperm(n_tokens, device=device)[:n_sel].to(torch.int32)
    token_to_req = torch.zeros(rows, device=device, dtype=torch.int32)

    got = qsa_sparse_paged_attention(
        q, k_cache, v_cache, indices, block_table, token_to_req, k_scale=k_scale, v_scale=v_scale
    )
    assert torch.isfinite(got).all()
    reference = _reference_attention(q, k_vals, v_vals, indices, block_table, token_to_req, page_size)
    torch.testing.assert_close(got.float(), reference, rtol=2e-2, atol=2e-2)

    # The 16-bit kernel over the bf16-rounded decode sits at the same distance: the fp8
    # path's extra error is the fold, not the storage.
    bf16_kernel = qsa_sparse_paged_attention(
        q, k_vals.to(torch.bfloat16).view(pages, page_size, heads, d),
        v_vals.to(torch.bfloat16).view(pages, page_size, heads, d), indices, block_table, token_to_req,
    )
    torch.testing.assert_close(bf16_kernel.float(), reference, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(got.float(), bf16_kernel.float(), rtol=2e-2, atol=2e-2)


def test_attend_kernel_rejects_fp8_without_scales():
    from freetoken.kernel.triton.qsa import qsa_sparse_paged_attention

    device = torch.device("cuda")
    k_cache = torch.zeros(2, 64, 2, 256, device=device, dtype=FP8)
    q = torch.randn(1, 24, 256, device=device, dtype=torch.bfloat16)
    indices = torch.zeros(1, 16, device=device, dtype=torch.int32)
    block_table = torch.zeros(1, 2, device=device, dtype=torch.int32)
    token_to_req = torch.zeros(1, device=device, dtype=torch.int32)
    with pytest.raises(AssertionError, match="per-token scales"):
        qsa_sparse_paged_attention(q, k_cache, k_cache, indices, block_table, token_to_req)


# --------------------------------------------------------------------------- the layer
def _dense_oracle(fixture: Fixture):
    from freetoken.models.qwen4_exp.attention import TorchDenseQSAReference

    return TorchDenseQSAReference(
        fixture.config, num_slots=fixture.num_req_slots, max_len=4096,
        device=fixture.device, dtype=fixture.dtype,
    )


def _inputs(fixture, lengths, extra=0, seed=11):
    generator = torch.Generator(device=fixture.device).manual_seed(seed)
    return [
        torch.randn(n + extra, fixture.config.hidden_size, device=fixture.device,
                    dtype=fixture.dtype, generator=generator) * 0.5
        for n in lengths
    ]


def test_layer_prefill_and_decode_track_the_dense_oracle(monkeypatch):
    """The real layer on the fp8 pool vs the fp32 oracle over the original bf16 K/V:
    quantization error only (selection is dense below the budget, as in the bf16 suite)."""
    config = parsed_config()
    fixture = Fixture(config, num_pages=128, kv_dtype=FP8)
    attn = fixture.layer(QSA_LAYER)
    lengths, steps = [300, 411, 64], 4
    inputs = _inputs(fixture, lengths, extra=steps)
    oracle = _dense_oracle(fixture)
    reqs = [fixture.req(i, 0, n) for i, n in enumerate(lengths)]
    seen = selection_spy(monkeypatch, fixture.backend)
    steps_x = [torch.cat([row[:n] for row, n in zip(inputs, lengths)])]
    steps_x += [torch.stack([row[n + s] for row, n in zip(inputs, lengths)]) for s in range(steps)]
    for step, x in enumerate(steps_x):
        if step:
            for req in reqs:
                fixture.step(req)
        batch = fixture.batch(reqs, "prefill" if step == 0 else "decode")
        fixture.ctx.attn_backend = fixture.backend
        got = attn.forward(x, batch)
        assert torch.isfinite(got).all()
        for row, position in enumerate(batch.positions.tolist()):
            selected = seen["indices"][row][seen["indices"][row] >= 0]
            assert torch.equal(selected.sort().values,
                               torch.arange(position + 1, dtype=selected.dtype, device=selected.device))
        fixture.ctx.attn_backend = oracle
        reference = attn.forward(x, batch)
        torch.testing.assert_close(got.float(), reference.float(), rtol=6e-2, atol=6e-2)


@pytest.mark.parametrize("cut", [1001, 4096, 4097], ids=["unaligned", "page-boundary", "boundary+1"])
def test_chunked_prefill_matches_one_shot(cut: int):
    """The quantizing store is a pure per-row function, so cut points cannot change codes."""
    config = parsed_config()
    fixture = Fixture(config, num_pages=512, kv_dtype=FP8)
    attn = fixture.layer(QSA_LAYER)
    length = 5000
    x = _inputs(fixture, [length])[0]
    one_shot = attn.forward(x, fixture.batch([fixture.req(0, 0, length)], "prefill"))
    head = fixture.req(1, 0, cut)
    attn.forward(x[:cut], fixture.batch([head], "prefill"))
    tail = fixture.req(1, cut, length)
    got = attn.forward(x[cut:], fixture.batch([tail], "prefill"))
    assert torch.equal(got, one_shot[cut:])


def test_decode_graph_replay_matches_eager():
    config = parsed_config()
    fixture = Fixture(config, num_pages=256, kv_dtype=FP8)
    attn = fixture.layer(QSA_LAYER)
    lengths, steps = [300, 411], 4
    bs = len(lengths)
    inputs = _inputs(fixture, lengths, extra=steps)
    reqs = [fixture.req(i, 0, n) for i, n in enumerate(lengths)]
    attn.forward(torch.cat([row[:n] for row, n in zip(inputs, lengths)]), fixture.batch(reqs, "prefill"))

    fixture.backend.init_capture_graph(max_seq_len=fixture.page_table.shape[1], bs_list=[bs])
    dummy = SimpleNamespace(table_idx=fixture.num_req_slots - 1, cached_len=1, device_len=2, extend_len=1)
    static = {
        "x": torch.zeros(bs, config.hidden_size, device=fixture.device, dtype=fixture.dtype),
        "positions": torch.zeros(bs, dtype=torch.int32, device=fixture.device),
        "out_loc": torch.zeros(bs, dtype=torch.int32, device=fixture.device),
    }
    capture_batch = SimpleNamespace(
        padded_reqs=[dummy] * bs, reqs=[dummy] * bs, phase="decode", size=bs, padded_size=bs,
        is_prefill=False, is_decode=True, positions=static["positions"],
        out_loc=static["out_loc"], attn_metadata=None, active_table_idx=None,
    )
    fixture.backend.prepare_for_capture(capture_batch)
    attn.forward(static["x"], capture_batch)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_out = attn.forward(static["x"], capture_batch)
    torch.cuda.synchronize()
    for step in range(steps):
        for req in reqs:
            fixture.step(req)
        x = torch.stack([row[n + step] for row, n in zip(inputs, lengths)])
        batch = fixture.batch(reqs, "decode")
        static["x"].copy_(x)
        static["positions"].copy_(batch.positions)
        static["out_loc"].copy_(batch.out_loc)
        fixture.backend.prepare_for_replay(batch)
        graph.replay()
        replayed = captured_out.clone()
        eager = attn.forward(x, fixture.batch(reqs, "decode"))
        assert torch.isfinite(replayed).all()
        assert torch.equal(replayed, eager), f"graph replay diverged at decode step {step}"


def test_pool_reports_half_the_kv_bytes():
    from freetoken.kvcache.base import spec_kv_bytes_per_token

    config = parsed_config()
    bf16 = Fixture(config, num_pages=8)
    fp8 = Fixture(config, num_pages=8, kv_dtype=FP8)
    kv16, _ = bf16.pool.unit_bytes()
    kv8, _ = fp8.pool.unit_bytes()
    spec = config.kv_cache_group_specs()[0]
    cfg16 = SimpleNamespace(dtype=torch.bfloat16, tp_info=SimpleNamespace(size=1))
    cfg8 = SimpleNamespace(dtype=torch.bfloat16, kv_dtype=FP8, tp_info=SimpleNamespace(size=1))
    assert kv16 == spec_kv_bytes_per_token(spec, cfg16)
    assert kv8 == spec_kv_bytes_per_token(spec, cfg8)
    heads, d, layers = spec.num_kv_heads, spec.head_dim, spec.num_layers
    assert kv8 - (kv16 - 2 * heads * d * 2 * layers) == 2 * heads * layers * (d + 4)
