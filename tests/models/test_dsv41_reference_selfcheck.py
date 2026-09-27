"""Self-checks for the DeepSeek-V4.1-Flash pure-torch oracle (``dsv41_reference.py``). CPU only.

Toy geometry from the port plan. The oracle is the ground truth for the FreeToken
``deepseek_v41`` port, so these tests pin its internal consistency (prefill vs decode, chunked vs
single prefill, CED vs full), the DSpark path, and the quantization helpers against hand-computed
values."""

from __future__ import annotations

import math

import pytest
import torch

from .dsv41_reference import (
    ModelArgs,
    Numerics,
    ReferenceModel,
    act_quant,
    act_quant_roundtrip,
    ced_prefill,
    dequant_fp8_block,
    e4m3_round,
    e8m0_scale,
    fp4_act_quant,
    fp4_act_quant_roundtrip,
    fp4_gemm,
    fp8_gemm,
    hc_split_sinkhorn,
    make_random_checkpoint,
    pack_e2m1,
    round_e2m1,
    sparse_attn,
    unpack_e2m1,
)

torch.set_num_threads(max(1, min(8, torch.get_num_threads())))

TOY = dict(
    max_batch_size=2,
    max_seq_len=512,
    vocab_size=64,
    dim=256,
    moe_inter_dim=128,
    n_layers=8,
    n_mtp_layers=3,
    n_heads=2,
    n_routed_experts=8,
    n_activated_experts=2,
    score_func="sqrtsoftplus",
    route_scale=1.5,
    swiglu_limit=10.0,
    q_lora_rank=128,
    head_dim=128,
    rope_head_dim=32,
    norm_eps=1e-20,
    o_groups=2,
    o_lora_rank=64,
    window_size=128,
    compress_ratios=(0, 0, 2, 2, 1, 1, 1, 1, 0, 0, 0),
    kv_source_layers=(2, 4),
    index_source_layers=(2, 4, 6),
    compress_rope_theta=160000.0,
    original_seq_len=256,
    rope_theta=10000.0,
    rope_factor=16,
    beta_fast=32,
    beta_slow=1,
    index_n_heads=2,
    index_head_dim=64,
    index_topk=16,
    candidate_source_layer=4,
    candidate_topk_blocks=4,
    candidate_block_size=8,
    hc_mult=4,
    hc_sinkhorn_iters=20,
    hc_eps=1e-6,
    engram_layer_ids=(1,),
    engram_num_embeddings=(1000,),
    engram_max_ngram_size=4,
    engram_vocab_size=1000,
    engram_n_heads=8,
    engram_head_dim=32,
    engram_compressed_vocab_size=64,
    dspark_block_size=5,
    dspark_noise_token_id=1,
    dspark_target_layer_ids=(5, 6, 7),
    dspark_markov_rank=32,
    dspark_n_routed_experts=4,
    dspark_n_activated_experts=1,
)

SEED = 1234
B = 2


def toy_args(**overrides) -> ModelArgs:
    return ModelArgs(**{**TOY, **overrides})


@pytest.fixture(scope="module")
def args() -> ModelArgs:
    return toy_args()


@pytest.fixture(scope="module")
def checkpoint(args):
    return make_random_checkpoint(args, seed=SEED)


def make_inputs(args: ModelArgs, length: int, seed: int = 7):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, args.vocab_size, (B, length), generator=g)
    n_eng = len(args.engram_layer_ids)
    hash_ids = torch.randint(0, args.engram_num_embeddings[0], (B, length, n_eng, args.engram_n_hash_cols), generator=g)
    return ids, hash_ids


def rel_max_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-6)).item()


# ------------------------------------------------------------------------------------------------
# (1) shapes / finiteness
# ------------------------------------------------------------------------------------------------


def test_prefill_shapes_and_finite(args, checkpoint):
    model = ReferenceModel(args, checkpoint)
    ids, hash_ids = make_inputs(args, 300)
    out = model.forward(ids, 0, hash_ids, full_logits=True)
    assert out.logits.shape == (B, args.vocab_size)
    assert out.logits.dtype == torch.float32
    assert torch.isfinite(out.logits).all()
    assert out.all_logits.shape == (B, 300, args.vocab_size)
    assert torch.isfinite(out.all_logits).all()
    assert torch.equal(out.all_logits[:, -1], out.logits)
    assert out.main_hidden.shape == (B, 300, len(args.dspark_target_layer_ids) * args.dim)
    assert torch.isfinite(out.main_hidden).all()
    # logits are not degenerate (the random head actually separates tokens)
    assert out.logits.std(dim=-1).min() > 1e-3
    # every mode of the CSA2 topology is exercised by the toy layout
    assert [args.mode(i) for i in range(args.n_layers)] == [
        "window", "window", "full", "reuse", "full", "reuse", "reindex", "reuse"
    ]


def test_random_checkpoint_is_deterministic_and_in_stored_formats(args):
    a = make_random_checkpoint(args, seed=3)
    b = make_random_checkpoint(args, seed=3)
    c = make_random_checkpoint(args, seed=4)
    assert a.keys() == b.keys() == c.keys()
    for k in a:
        assert a[k].dtype == b[k].dtype and torch.equal(a[k].view(torch.uint8), b[k].view(torch.uint8)), k
    assert not torch.equal(a["embed.weight"], c["embed.weight"])
    # dtypes as in the V4.1 safetensors headers
    assert a["embed.weight"].dtype == torch.bfloat16 and a["head.weight"].dtype == torch.bfloat16
    assert a["layers.0.attn.wq_a.weight"].dtype == torch.float8_e4m3fn
    assert a["layers.0.attn.wq_a.scale"].dtype == torch.float8_e8m0fnu
    assert a["layers.0.attn.wq_a.scale"].shape == (args.q_lora_rank // 32, args.dim // 32)
    assert a["layers.0.attn.wo_a.weight"].dtype == torch.float8_e4m3fn  # dequantized to bf16 at load
    assert a["layers.0.ffn.experts.0.w1.weight"].dtype == torch.int8
    assert a["layers.0.ffn.experts.0.w1.weight"].shape == (args.moe_inter_dim, args.dim // 2)
    assert a["layers.0.ffn.experts.0.w1.scale"].shape == (args.moe_inter_dim, args.dim // 32)
    assert a["layers.0.ffn.experts.0.w2.weight"].shape == (args.dim, args.moe_inter_dim // 2)
    assert a["layers.0.ffn.gate.weight"].dtype == torch.bfloat16 and a["layers.0.ffn.gate.bias"].dtype == torch.float32
    assert a["layers.0.hc_attn_fn"].dtype == torch.float32 and a["layers.0.attn.attn_sink"].dtype == torch.float32
    assert a["layers.2.attn.compressor.wgate.weight"].dtype == torch.bfloat16
    assert "layers.4.attn.compressor.wgate.weight" not in a  # ratio 1: no gate
    assert a["layers.4.attn.indexer.wk.weight"].dtype == torch.bfloat16
    assert "layers.6.attn.indexer.wk.weight" not in a and "layers.6.attn.indexer.wq_b.weight" in a  # reindex
    assert "layers.5.attn.indexer.wq_b.weight" not in a  # reuse
    assert a["layers.1.engram.embed.weight"].shape == (1000, 32) and a["layers.1.engram.embed.scale"].shape == (1000, 1)
    assert a["layers.1.engram.wkv.weight"].shape == (args.dim * 5, 24 * 32)
    assert a["mtp.0.main_proj.weight"].shape == (args.dim, 3 * args.dim)
    assert a["mtp.2.confidence_head.proj.weight"].shape == (1, args.dim + args.dspark_markov_rank)
    assert a["mtp.2.confidence_head.proj.weight"].dtype == torch.bfloat16
    assert a["mtp.1.ffn.gate.weight"].shape == (args.dspark_n_routed_experts, args.dim)
    assert "mtp.0.ffn.experts.4.w1.weight" not in a and "mtp.0.ffn.experts.3.w1.weight" in a
    assert not any("bias_vl" in k or k.startswith("vision") for k in a)


# ------------------------------------------------------------------------------------------------
# (2) prefill + K decode steps == one longer prefill
# ------------------------------------------------------------------------------------------------


# Tolerances: in fp32 mode the only differences are fp32 reassociation (GEMV vs GEMM), observed
# ~4e-7. In the bf16-faithful mode those ulp differences flip bf16 roundings, which drift to ~4e-3
# after a few layers and then flip a near-tied top-k slot in the (bf16-scored) reindex layer --
# a discrete change that dominates (observed up to ~8e-2, thread-count dependent). The reference
# has the same property between its own prefill and decode, so the bf16 bound is deliberately
# loose: it guards against gross breakage, the fp32 case is the sharp one.
CONSISTENCY_CASES = [
    ("fp32", Numerics(bf16_activations=False, attn_probs_bf16=False), 2e-4, True),
    ("bf16-faithful", Numerics(), 2.5e-1, False),
]
CONSISTENCY_IDS = [c[0] for c in CONSISTENCY_CASES]


@pytest.mark.parametrize("ident, numerics, tol, check_argmax", CONSISTENCY_CASES, ids=CONSISTENCY_IDS)
def test_prefill_then_decode_matches_long_prefill(args, checkpoint, ident, numerics, tol, check_argmax):
    n, k = 300, 6
    ids, hash_ids = make_inputs(args, n + k)
    ref = ReferenceModel(args, checkpoint, numerics).forward(ids, 0, hash_ids, full_logits=True)

    model = ReferenceModel(args, checkpoint, numerics)
    out = model.forward(ids[:, :n], 0, hash_ids[:, :n], full_logits=True)
    err = rel_max_err(out.all_logits, ref.all_logits[:, :n])
    assert err <= tol, f"prefill prefix differs: {err}"
    worst = 0.0
    for j in range(k):
        pos = n + j
        step = model.forward(ids[:, pos : pos + 1], pos, hash_ids[:, pos : pos + 1])
        assert step.main_hidden.shape == (B, 1, 3 * args.dim)
        e = rel_max_err(step.logits, ref.all_logits[:, pos])
        worst = max(worst, e)
        assert e <= tol, f"decode step {j} (pos {pos}) differs: {e}"
        if check_argmax:
            assert torch.equal(step.logits.argmax(-1), ref.all_logits[:, pos].argmax(-1))
    print(f"prefill-vs-decode worst rel err ({ident}): {worst:.3e}")


# ------------------------------------------------------------------------------------------------
# (3) chunked prefill == single prefill
# ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("ident, numerics, tol, check_argmax", CONSISTENCY_CASES, ids=CONSISTENCY_IDS)
def test_chunked_prefill_matches_single(args, checkpoint, ident, numerics, tol, check_argmax):
    length, chunk = 300, 128
    ids, hash_ids = make_inputs(args, length)
    ref = ReferenceModel(args, checkpoint, numerics).forward(ids, 0, hash_ids, full_logits=True)

    model = ReferenceModel(args, checkpoint, numerics)
    pieces = []
    for start in range(0, length, chunk):
        end = min(length, start + chunk)
        out = model.forward(ids[:, start:end], start, hash_ids[:, start:end], full_logits=True)
        pieces.append(out.all_logits)
    chunked = torch.cat(pieces, dim=1)
    err = rel_max_err(chunked, ref.all_logits)
    assert err <= tol, f"chunked prefill differs: {err}"
    if check_argmax:
        assert torch.equal(chunked[:, -1].argmax(-1), ref.logits.argmax(-1))
    # the caches are left in the same state: one decode step agrees too
    step_a = model.forward(ids[:, :1], length, hash_ids[:, :1])
    model_b = ReferenceModel(args, checkpoint, numerics)
    model_b.forward(ids, 0, hash_ids)
    step_b = model_b.forward(ids[:, :1], length, hash_ids[:, :1])
    err_step = rel_max_err(step_a.logits, step_b.logits)
    assert err_step <= tol
    print(f"chunked-vs-single rel err ({ident}): prefill {err:.3e}, decode step {err_step:.3e}")


# ------------------------------------------------------------------------------------------------
# (4) DSpark forward_spec
# ------------------------------------------------------------------------------------------------


def test_forward_spec_drafts_block(args, checkpoint):
    model = ReferenceModel(args, checkpoint)
    n = 300
    ids, hash_ids = make_inputs(args, n + 1)
    out = model.forward(ids[:, :n], 0, hash_ids[:, :n])
    anchor = out.logits.argmax(-1)
    # prefill seeds the stage windows and drafts nothing (reference rule: start_pos == 0)
    assert model.forward_spec(anchor, out.main_hidden, 0) is None
    step = model.forward(ids[:, n : n + 1], n, hash_ids[:, n : n + 1])
    anchor = step.logits.argmax(-1)
    spec = model.forward_spec(anchor, step.main_hidden, n)
    bs = args.dspark_block_size
    assert spec.output_ids.shape == (B, bs + 1)
    assert spec.logits.shape == (B, bs, args.vocab_size) and spec.logits.dtype == torch.float32
    assert spec.confidence.shape == (B, bs) and spec.confidence.dtype == torch.float32
    assert torch.isfinite(spec.logits).all() and torch.isfinite(spec.confidence).all()
    assert torch.equal(spec.output_ids[:, 0], anchor)
    assert torch.equal(spec.output_ids[:, 1:], spec.logits.argmax(-1))
    # deterministic
    model2 = ReferenceModel(args, checkpoint)
    model2.forward(ids[:, :n], 0, hash_ids[:, :n])
    model2.forward_spec(anchor, out.main_hidden, 0)
    step2 = model2.forward(ids[:, n : n + 1], n, hash_ids[:, n : n + 1])
    spec2 = model2.forward_spec(step2.logits.argmax(-1), step2.main_hidden, n)
    assert torch.equal(spec.output_ids, spec2.output_ids)
    assert torch.equal(spec.logits, spec2.logits)
    # a custom sampler is honoured
    spec3 = model2.forward_spec(anchor, step2.main_hidden, n, sampler=lambda lg: lg.argmin(-1))
    assert torch.equal(spec3.output_ids[:, 1:], spec3.logits.argmin(-1))


# ------------------------------------------------------------------------------------------------
# (5) CED prefill
# ------------------------------------------------------------------------------------------------


def test_ced_prefill_approximates_forward(args, checkpoint):
    length = 300
    ids, hash_ids = make_inputs(args, length)
    full = ReferenceModel(args, checkpoint).forward(ids, 0, hash_ids)
    model = ReferenceModel(args, checkpoint)
    ced = ced_prefill(model, ids, hash_ids)
    assert ced.logits.shape == (B, args.vocab_size) and torch.isfinite(ced.logits).all()
    assert ced.replay_start == length - args.window_size
    assert ced.main_hidden.shape == (B, args.window_size, 3 * args.dim)
    err = rel_max_err(ced.logits, full.logits)
    print(f"CED-vs-forward rel err (L=300, window=128): {err:.3e}")
    assert err < 0.5, f"CED drifted implausibly far from forward: {err}"
    assert err > 1e-6, "CED with a truncated window should not be identical to forward"
    # the caches are usable for decode: DSpark seeding + a decode step run and stay finite
    assert model.forward_spec(ced.logits.argmax(-1), ced.main_hidden, ced.replay_start, draft=False) is None
    step = model.forward(ids[:, :1], length, hash_ids[:, :1])
    assert torch.isfinite(step.logits).all()
    spec = model.forward_spec(step.logits.argmax(-1), step.main_hidden, length)
    assert spec is not None and torch.isfinite(spec.logits).all()


def test_ced_prefill_equals_forward_when_window_covers_prompt():
    big = toy_args(window_size=512)
    sd = make_random_checkpoint(big, seed=SEED)
    length = 300
    ids, hash_ids = make_inputs(big, length + 1)
    full_model = ReferenceModel(big, sd)
    full = full_model.forward(ids[:, :length], 0, hash_ids[:, :length])
    ced_model = ReferenceModel(big, sd)
    ced = ced_prefill(ced_model, ids[:, :length], hash_ids[:, :length])
    assert ced.replay_start == 0
    assert torch.allclose(ced.logits, full.logits, rtol=1e-5, atol=1e-5), rel_max_err(ced.logits, full.logits)
    assert torch.allclose(ced.main_hidden, full.main_hidden, rtol=1e-5, atol=1e-5)
    # and decode continues identically from either prefill
    a = full_model.forward(ids[:, length : length + 1], length, hash_ids[:, length : length + 1])
    b = ced_model.forward(ids[:, length : length + 1], length, hash_ids[:, length : length + 1])
    assert torch.allclose(a.logits, b.logits, rtol=1e-5, atol=1e-5), rel_max_err(a.logits, b.logits)


# ------------------------------------------------------------------------------------------------
# (6) quantization helpers vs hand-computed values
# ------------------------------------------------------------------------------------------------


def test_e8m0_scales_hand_values():
    amax = torch.tensor([448.0, 449.0, 100.0, 1.0, 1e-4, 896.0])
    expect = torch.tensor([1.0, 2.0, 0.25, 2.0**-8, 2.0**-22, 2.0])
    assert torch.equal(e8m0_scale(amax, 448.0), expect)
    # fp4 e8m0 mode: floor at 6 * 2^-126
    q, s = fp4_act_quant(torch.zeros(1, 32), 32, "e8m0")
    assert s.item() == 2.0**-126 and (q == 0).all()
    q, s = fp4_act_quant(torch.full((1, 32), 7.0), 32, "e8m0")
    assert s.item() == 2.0  # 2^ceil(log2(7/6))
    assert (q == 4.0).all()  # 7/2 = 3.5 ties 3|4 -> even code -> 4.0
    q, s = fp4_act_quant(torch.full((1, 32), 6.0), 32, "e8m0")
    assert s.item() == 1.0 and (q == 6.0).all()


def test_e4m3_scales_hand_values():
    # amax floored at 6 * 2^-9 so an all-zero group keeps a nonzero (minimum) scale
    q, s = fp4_act_quant(torch.zeros(1, 16), 16, "e4m3")
    assert s.item() == 2.0**-9 and (q == 0).all()
    q, s = fp4_act_quant(torch.full((1, 16), 6.0), 16, "e4m3")
    assert s.item() == 1.0 and (q == 6.0).all()
    # s = e4m3(amax / 6) with round-to-nearest-even on the 3-bit mantissa
    _, s = fp4_act_quant(torch.full((1, 16), 6 * 1.0625), 16, "e4m3")
    assert s.item() == 1.0  # 1.0625 is a tie between 1.0 and 1.125 -> even -> 1.0
    _, s = fp4_act_quant(torch.full((1, 16), 6 * 1.1875), 16, "e4m3")
    assert s.item() == 1.25  # tie between 1.125 and 1.25 -> even -> 1.25
    _, s = fp4_act_quant(torch.full((1, 16), 6 * 1.1), 16, "e4m3")
    assert s.item() == 1.125
    assert e4m3_round(torch.tensor([500.0, -500.0, 0.3])).tolist() == [448.0, -448.0, 0.3125]


def test_e2m1_rounding_ties():
    x = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 0.24, 0.26, 7.0, -0.75, -5.0, 6.0, -6.0, 0.0])
    expect = torch.tensor([0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 0.0, 0.5, 6.0, -1.0, -4.0, 6.0, -6.0, 0.0])
    assert torch.equal(round_e2m1(x), expect)


def test_pack_unpack_e2m1_low_nibble_is_even_k():
    vals = torch.tensor([[0.5, -6.0, 0.0, 1.5]])
    packed = pack_e2m1(vals)
    assert packed.dtype == torch.int8 and packed.shape == (1, 2)
    assert packed.view(torch.uint8).tolist() == [[0x01 | (0x0F << 4), 0x00 | (0x03 << 4)]]
    assert torch.equal(unpack_e2m1(packed), vals)
    # every byte round-trips; the -0 nibble (8) decodes to 0 and repacks as +0 (0)
    codes = torch.arange(256, dtype=torch.uint8)
    dec = unpack_e2m1(codes.view(torch.int8).view(128, 2))
    assert dec.abs().max() == 6.0
    repacked = pack_e2m1(dec).view(torch.uint8).flatten()
    lo, hi = codes & 0x0F, codes >> 4
    lo = torch.where(lo == 8, torch.zeros_like(lo), lo)
    hi = torch.where(hi == 8, torch.zeros_like(hi), hi)
    assert torch.equal(repacked, lo | (hi << 4))


def test_act_quant_roundtrip_hand_values():
    x = torch.arange(1, 33, dtype=torch.float32).unsqueeze(0)  # amax 32 -> scale 2^ceil(log2(32/448)) = 1/8
    q, s = act_quant(x, 32)
    assert s.item() == 0.125
    y = act_quant_roundtrip(x, 32)
    # x/s = 8..256: integers up to 16 exact in e4m3, then spacing 2 (16..32), 4, 8, 16 (128..256)
    assert torch.equal(y[0, :16], x[0, :16])
    assert y[0, 16].item() == 16.0  # 17 -> 136/8: 136 ties 128|144 -> even mantissa -> 128 -> 16.0
    assert y[0, 17].item() == 18.0  # 18 -> 144 exact
    assert y[0, 18].item() == 20.0  # 19 -> 152 ties 144|160 -> 160 (even) -> 20.0
    assert (q.abs() <= 448).all()
    # bf16-valued inputs come back bf16-valued and inside the block's dynamic range
    g = torch.Generator().manual_seed(0)
    z = (torch.randn(4, 128, generator=g) * 3).to(torch.bfloat16).float()
    r = act_quant_roundtrip(z, 32)
    assert torch.equal(r, r.to(torch.bfloat16).float())
    assert ((r - z).abs() <= z.abs() * 2**-3 + 1e-6).all()


def test_block_gemms_match_dequantized_reference():
    g = torch.Generator().manual_seed(1)
    sd = make_random_checkpoint(toy_args(), seed=SEED)
    x = torch.randn(5, 256, generator=g).to(torch.bfloat16).float()
    # fp8: wq_a [128, 256] with [4, 8] scales
    w, ws = sd["layers.0.attn.wq_a.weight"], sd["layers.0.attn.wq_a.scale"]
    q, s = act_quant(x, 32)
    y = fp8_gemm(q, s, w.float(), ws.float(), 32)
    xd = (q.unflatten(-1, (-1, 32)) * s.unsqueeze(-1)).flatten(-2)
    ref = xd @ dequant_fp8_block(w, ws, 32).t()
    assert torch.allclose(y, ref, rtol=1e-5, atol=1e-5)
    # fp4: expert w1 [128, 256] packed with [128, 8] scales
    p, ps = sd["layers.0.ffn.experts.0.w1.weight"], sd["layers.0.ffn.experts.0.w1.scale"]
    wv = unpack_e2m1(p)
    y4 = fp4_gemm(q, s, wv, ps.float(), 32)
    ref4 = xd @ (wv * ps.float().repeat_interleave(32, dim=1)).t()
    assert torch.allclose(y4, ref4, rtol=1e-5, atol=1e-5)


def test_sparse_attn_hand_values():
    # one head, two keys, no masking: softmax with the sink in the denominator only
    q = torch.tensor([[[[1.0, 0.0]]]])  # [b=1, m=1, h=1, d=2]
    kv = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [5.0, 5.0]]])  # third row must be ignored (-1)
    sink = torch.tensor([0.5])
    idx = torch.tensor([[[0, 1, -1]]], dtype=torch.int32)
    num = Numerics(bf16_activations=False, attn_probs_bf16=False)
    o = sparse_attn(q, kv, sink, idx, 1.0, num)
    den = math.exp(1.0) + math.exp(0.0) + math.exp(0.5)
    expect = torch.tensor([[[[math.exp(1.0) / den, math.exp(0.0) / den]]]])
    assert torch.allclose(o, expect, atol=1e-6)
    # a row with nothing to attend to yields zeros (finite lower bound, not NaN)
    o0 = sparse_attn(q, kv, sink, torch.tensor([[[-1, -1]]], dtype=torch.int32), 1.0, num)
    assert torch.equal(o0, torch.zeros_like(o0))
    # the 64-blocking only changes bf16 rounding of the probabilities, never the fp32 math
    g = torch.Generator().manual_seed(2)
    qq = torch.randn(1, 3, 2, 8, generator=g)
    kk = torch.randn(1, 200, 8, generator=g)
    ii = torch.randint(-1, 200, (1, 3, 150), generator=g).int()
    a = sparse_attn(qq, kk, torch.zeros(2), ii, 0.3, num)
    b = sparse_attn(qq, kk, torch.zeros(2), ii, 0.3, Numerics(bf16_activations=False, attn_probs_bf16=False, attn_block=7))
    assert torch.allclose(a, b, rtol=1e-5, atol=1e-6)


def test_hc_split_sinkhorn_is_doubly_stochastic():
    g = torch.Generator().manual_seed(3)
    hc = 4
    mixes = torch.randn(2, 3, (2 + hc) * hc, generator=g)
    scale = torch.tensor([1.0, 1.0, 1.0])
    base = torch.zeros((2 + hc) * hc)
    pre, post, comb = hc_split_sinkhorn(mixes, scale, base, hc, 20, 1e-6)
    assert pre.shape == (2, 3, hc) and post.shape == (2, 3, hc) and comb.shape == (2, 3, hc, hc)
    assert torch.allclose(pre, torch.sigmoid(mixes[..., :hc]) + 1e-6)
    assert torch.allclose(post, 2 * torch.sigmoid(mixes[..., hc : 2 * hc]))
    assert torch.allclose(comb.sum(-2), torch.ones(2, 3, hc), atol=1e-4)
    assert torch.allclose(comb.sum(-1), torch.ones(2, 3, hc), atol=1e-4)
    # iteration order: after the softmax + eps the FIRST normalization is over columns
    pre1, post1, comb1 = hc_split_sinkhorn(mixes, scale, base, hc, 1, 1e-6)
    c = mixes[..., 2 * hc :].unflatten(-1, (hc, hc)).softmax(-1) + 1e-6
    c = c / (c.sum(-2, keepdim=True) + 1e-6)
    assert torch.allclose(comb1, c)
