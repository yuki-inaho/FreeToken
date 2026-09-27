"""DeepSeek-V4.1-Flash weight plan: the consumed-set invariant over the checkpoint's tensor
names, block-32 fp8 dequant against a torch reference, and the 128-row DSpark stage banks.
CPU only; no checkpoint needed -- the name manifest is generated from the released
geometry and the layer roles verified against the safetensors index."""

from __future__ import annotations

import re

import pytest
import torch

from freetoken.models.deepseek_v41.args import parse_raw_args
from freetoken.models.deepseek_v41.weight import (
    DEQUANT_WO_A,
    EXPERT,
    PASSTHROUGH,
    _dequant_fp8_block,
    _dsfp4_bank_specs,
    _place_dsfp4_parsed,
    dspark_bank_bytes_estimate,
    resident_tensor_plan,
)

from .test_dsv41_args import V41_CONFIG

# --------------------------------------------------------------------------- #
# Manifest generator: the V4.1 index names for a given (toy or real) topology.
# Layer roles as verified from model.safetensors.index.json:
#   every L: attn.{wq_a,wq_b,wkv,wo_a,wo_b}.{weight,scale}, attn.{q_norm,kv_norm}.weight,
#            attn.attn_sink, ffn.gate.{weight,bias,bias_vl}, ffn.shared_experts.w*.{weight,scale},
#            ffn.experts.E.w*.{weight,scale}, attn_norm/ffn_norm, hc_{attn,ffn}_{fn,base,scale}
#   kv sources: attn.compressor.{norm.weight, wkv.weight} (+ wgate.weight at ratio 2),
#               attn.indexer.{k_norm.weight, wk.weight}
#   index sources: attn.indexer.{wq_b.weight, wq_b.scale, weights_proj.weight}
#   engram layers: engram.embed.{weight,scale}, engram.wkv.{weight,scale}, engram.{q,k}_weight
#   mtp.s: the block layout with the stage expert count; mtp.0.main_proj.{weight,scale},
#          mtp.0.main_norm.weight; last: norm.weight, markov_head.{embed,head}.weight,
#          confidence_head.proj.weight
# --------------------------------------------------------------------------- #
def _block_names(prefix: str, n_experts: int) -> list[str]:
    names = []
    for p in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b"):
        names += [f"{prefix}.attn.{p}.weight", f"{prefix}.attn.{p}.scale"]
    names += [f"{prefix}.attn.q_norm.weight", f"{prefix}.attn.kv_norm.weight", f"{prefix}.attn.attn_sink"]
    names += [f"{prefix}.attn_norm.weight", f"{prefix}.ffn_norm.weight"]
    names += [f"{prefix}.ffn.gate.weight", f"{prefix}.ffn.gate.bias", f"{prefix}.ffn.gate.bias_vl"]
    for p in ("w1", "w2", "w3"):
        names += [f"{prefix}.ffn.shared_experts.{p}.weight", f"{prefix}.ffn.shared_experts.{p}.scale"]
    for e in range(n_experts):
        for p in ("w1", "w2", "w3"):
            names += [f"{prefix}.ffn.experts.{e}.{p}.weight", f"{prefix}.ffn.experts.{e}.{p}.scale"]
    for s in ("attn", "ffn"):
        for k in ("fn", "base", "scale"):
            names.append(f"{prefix}.hc_{s}_{k}")
    return names


def v41_index_names(args) -> list[str]:
    names = ["embed.weight", "head.weight", "norm.weight", "image_start", "image_end", "image_newline"]
    names += ["vision.embeddings.patch_embedding.weight", "vision.encoder.layers.0.attn.qkv.weight",
              "aligner.layers.0.weight", "aligner.layers.0.bias"]
    for L in range(args.n_layers):
        p = f"layers.{L}"
        names += _block_names(p, args.n_routed_experts)
        if args.is_kv_source(L):
            names += [f"{p}.attn.compressor.norm.weight", f"{p}.attn.compressor.wkv.weight"]
            if args.compress_ratios[L] == 2:
                names.append(f"{p}.attn.compressor.wgate.weight")
            names += [f"{p}.attn.indexer.k_norm.weight", f"{p}.attn.indexer.wk.weight"]
        if args.is_index_source(L):
            names += [f"{p}.attn.indexer.wq_b.weight", f"{p}.attn.indexer.wq_b.scale",
                      f"{p}.attn.indexer.weights_proj.weight"]
        if L in args.engram_layer_ids:
            names += [f"{p}.engram.embed.weight", f"{p}.engram.embed.scale",
                      f"{p}.engram.wkv.weight", f"{p}.engram.wkv.scale",
                      f"{p}.engram.q_weight", f"{p}.engram.k_weight"]
    n_stage_experts = args.get_moe_config(args.n_layers)[0]
    for s in range(args.n_mtp_layers):
        p = f"mtp.{s}"
        names += _block_names(p, n_stage_experts)
        if s == 0:
            names += [f"{p}.main_proj.weight", f"{p}.main_proj.scale", f"{p}.main_norm.weight"]
        if s == args.n_mtp_layers - 1:
            names += [f"{p}.norm.weight", f"{p}.markov_head.embed.weight",
                      f"{p}.markov_head.head.weight", f"{p}.confidence_head.proj.weight"]
    return names


# Toy topology with the same roles (2 = full at ratio 2, 4 = full at ratio 1 / decoder
# start + candidate source, 6 = reindex, engram at 1, 3 stages of 4 experts).
TOY = dict(
    V41_CONFIG,
    n_layers=8,
    n_routed_experts=8,
    n_activated_experts=2,
    dspark_n_routed_experts=4,
    dspark_n_activated_experts=1,
    compress_ratios=[0, 0, 2, 2, 1, 1, 1, 1] + [0, 0, 0],
    kv_source_layers=[2, 4],
    index_source_layers=[2, 4, 6],
    candidate_source_layer=4,
    candidate_topk_blocks=4,
    engram_layer_ids=[1],
    engram_num_embeddings=[1009],
    dspark_target_layer_ids=[5, 6, 7],
)


def _pattern_set(names: list[str]) -> set[str]:
    """Normalize layer / stage / expert indices so the manifest reads as ~120 patterns."""
    out = set()
    for n in names:
        n = re.sub(r"^layers\.\d+\.", "layers.L.", n)
        n = re.sub(r"^mtp\.\d+\.", "mtp.S.", n)
        n = re.sub(r"\.experts\.\d+\.", ".experts.E.", n)
        out.add(n)
    return out


def test_manifest_covers_the_documented_patterns():
    """The generated fixture must enumerate every tensor kind the index actually has."""
    patterns = _pattern_set(v41_index_names(parse_raw_args(V41_CONFIG)))
    for p in (
        "layers.L.attn.wo_a.scale", "layers.L.attn.compressor.wgate.weight",
        "layers.L.attn.indexer.wk.weight", "layers.L.attn.indexer.wq_b.scale",
        "layers.L.attn.indexer.weights_proj.weight", "layers.L.engram.embed.scale",
        "layers.L.engram.q_weight", "layers.L.ffn.gate.bias_vl",
        "layers.L.ffn.experts.E.w2.scale", "layers.L.hc_ffn_scale",
        "mtp.S.main_proj.scale", "mtp.S.main_norm.weight", "mtp.S.norm.weight",
        "mtp.S.markov_head.embed.weight", "mtp.S.markov_head.head.weight",
        "mtp.S.confidence_head.proj.weight", "mtp.S.ffn.experts.E.w1.weight",
        "head.weight", "image_newline",
    ):
        assert p in patterns, p
    assert 100 <= len(patterns) <= 140, len(patterns)


@pytest.mark.parametrize("raw", [TOY, V41_CONFIG], ids=["toy", "released"])
def test_plan_consumes_everything_with_dspark(raw):
    args = parse_raw_args(raw)
    names = v41_index_names(args)
    plan = resident_tensor_plan(names, args, dspark=True)
    assert plan.leftover == []
    # Skip classes and nothing else: vision / aligner / image tokens, gate.bias_vl, tables.
    skipped = set(plan.skipped)
    expect_skip = {n for n in names if n.startswith(("vision.", "aligner.", "image_"))}
    expect_skip |= {n for n in names if n.endswith(".ffn.gate.bias_vl")}
    expect_skip |= {n for n in names if ".engram.embed." in n}
    assert skipped == expect_skip
    # Every index name is accounted for exactly once (sources of yielded entries + skips).
    consumed = [s for _, sources, _ in plan.yielded for s in sources] + plan.skipped
    assert sorted(consumed) == sorted(names)
    by_kind = {}
    for param, sources, kind in plan.yielded:
        by_kind.setdefault(kind, []).append((param, sources))
    # Experts: the tower's E per layer plus the stage E per stage, x3 projections x2.
    E, E_s = args.n_routed_experts, args.get_moe_config(args.n_layers)[0]
    assert len(by_kind[EXPERT]) == 6 * (args.n_layers * E + args.n_mtp_layers * E_s)
    # wo_a: one bf16 param per block, fed by the (weight, scale) pair, named without a suffix.
    wo = dict(by_kind[DEQUANT_WO_A])
    assert len(wo) == args.n_layers + args.n_mtp_layers
    assert wo["layers.0.attn.wo_a"] == ("layers.0.attn.wo_a.weight", "layers.0.attn.wo_a.scale")
    assert f"mtp.{args.n_mtp_layers - 1}.attn.wo_a" in wo
    # Renames: head.weight -> head; everything else keeps the checkpoint name.
    passthrough = dict(by_kind[PASSTHROUGH])
    assert passthrough["head"] == ("head.weight",)
    for param, sources in passthrough.items():
        if param != "head":
            assert sources == (param,), param
    for L in args.engram_layer_ids:
        assert f"layers.{L}.engram.wkv.scale" in passthrough
        assert f"layers.{L}.engram.k_weight" in passthrough
    for L in args.kv_source_layers:
        assert f"layers.{L}.attn.indexer.wk.weight" in passthrough
    assert "mtp.0.main_proj.scale" in passthrough
    assert f"mtp.{args.n_mtp_layers - 1}.confidence_head.proj.weight" in passthrough
    assert "mtp.0.confidence_head.proj.weight" not in passthrough


def test_plan_skips_the_whole_drafter_when_not_built():
    args = parse_raw_args(TOY)
    names = v41_index_names(args)
    plan = resident_tensor_plan(names, args, dspark=False)
    assert plan.leftover == []
    assert all(not p.startswith("mtp.") for p, _, _ in plan.yielded)
    assert {n for n in names if n.startswith("mtp.")} <= set(plan.skipped)


@pytest.mark.parametrize(
    "stray",
    [
        "hc_head_fn",  # 0731 head mixes: gone in V4.1
        "layers.3.attn.compressor.wkv.weight",  # compressor at a non-kv-source
        "layers.5.attn.indexer.wq_b.weight",  # query side at a reuse layer
        "layers.2.engram.q_weight",  # engram at a non-engram layer
        "layers.1.engram.embed.weight.extra",
        "layers.0.ffn.experts.8.w1.weight",  # expert id beyond n_routed_experts
        "mtp.0.ffn.experts.4.w1.weight",  # beyond the stage width
        "mtp.1.main_proj.weight",  # main_proj is stage 0 only
        "mtp.0.norm.weight",  # last stage only
        "layers.8.attn.wq_a.weight",  # layer beyond n_layers
        "mtp.3.attn.wq_a.weight",  # stage beyond n_mtp_layers
        "layers.0.ffn.gate.tid2eid",  # 0731 hash routing: gone
        "layers.0.attn.compressor.ape",  # 0731 absolute position embedding: gone
        "some.renamed.tensor",
    ],
)
def test_plan_reports_unaccounted_names(stray):
    args = parse_raw_args(TOY)
    names = v41_index_names(args) + [stray]
    plan = resident_tensor_plan(names, args, dspark=True)
    assert plan.leftover == [stray]


def test_wo_a_weight_without_scale_is_leftover():
    args = parse_raw_args(TOY)
    names = [n for n in v41_index_names(args) if n != "layers.0.attn.wo_a.scale"]
    plan = resident_tensor_plan(names, args, dspark=True)
    assert plan.leftover == ["layers.0.attn.wo_a.weight"]


def test_iter_weights_raises_on_leftover(tmp_path, monkeypatch):
    """The disk-driving loader must refuse, not drop, an unaccounted tensor."""
    import json
    import os

    from freetoken.models.deepseek_v41 import weight as W

    args = parse_raw_args(TOY)
    os.makedirs(tmp_path / "inference")
    with open(tmp_path / "inference" / "config.json", "w") as f:
        json.dump(TOY, f)
    names = v41_index_names(args) + ["layers.0.attn.brand_new.weight"]
    with open(tmp_path / "model.safetensors.index.json", "w") as f:
        json.dump({"weight_map": {n: "model-00001-of-00001.safetensors" for n in names}}, f)
    with pytest.raises(RuntimeError, match="brand_new"):
        list(W.iter_weights(str(tmp_path), "cpu"))


# --------------------------------------------------------------------------- #
# fp8 block dequant with inferred block size
# --------------------------------------------------------------------------- #
def _reference_dequant(weight_fp8: torch.Tensor, codes: torch.Tensor, block: int) -> torch.Tensor:
    """Independent: expand each e8m0 code to its block by explicit indexing."""
    n, k = weight_fp8.shape
    w = weight_fp8.to(torch.float32)
    out = torch.empty_like(w)
    for i in range(n):
        for j in range(k):
            out[i, j] = w[i, j] * 2.0 ** (int(codes[i // block, j // block]) - 127)
    return out.to(torch.bfloat16)


@pytest.mark.parametrize("block, shape", [(32, (64, 96)), (128, (256, 128)), (32, (40, 70))])
def test_dequant_fp8_block_infers_the_block(block, shape):
    torch.manual_seed(0)
    n, k = shape
    weight = (torch.randn(n, k) * 4).to(torch.float8_e4m3fn)
    sn, sk = -(-n // block), -(-k // block)
    codes = torch.randint(120, 134, (sn, sk), dtype=torch.uint8)
    scale = codes.view(torch.float8_e8m0fnu)
    got = _dequant_fp8_block(weight, scale)  # block=None -> inferred
    ref = _reference_dequant(weight, codes, block)
    assert got.dtype == torch.bfloat16
    assert torch.equal(got, ref)
    assert torch.equal(_dequant_fp8_block(weight, scale, block=block), ref)


def test_dequant_fp8_block_refuses_ambiguous_shapes():
    weight = torch.zeros(32, 32, dtype=torch.float8_e4m3fn)
    scale = torch.zeros(1, 1, dtype=torch.uint8).view(torch.float8_e8m0fnu)  # fits 32/64/128
    with pytest.raises(ValueError, match="cannot infer"):
        _dequant_fp8_block(weight, scale)


# --------------------------------------------------------------------------- #
# 128-row DSpark stage banks
# --------------------------------------------------------------------------- #
def test_stage_bank_specs_and_placement_at_128_rows():
    E, H, I = 128, 5120, 2304
    specs = _dsfp4_bank_specs(E, H, I)
    assert specs["gate_up_packed"] == ((E, 2 * I, H // 2), torch.uint8)
    assert specs["gate_up_scale"][0] == (E, 2 * I, H // 32)
    assert specs["down_packed"] == ((E, H, I // 2), torch.uint8)
    assert specs["down_scale"][0] == (E, H, I // 32)
    # Bytes per expert row == the offload cache's ds_fp4 price (what the preflight uses).
    from freetoken.moe.offload_cache import _BANK_BYTES_PER_EXPERT

    per_expert = sum(
        torch.Size(shape[1:]).numel() * torch.empty(0, dtype=dt).element_size()
        for shape, dt in specs.values()
    )
    assert per_expert == _BANK_BYTES_PER_EXPERT["ds_fp4"](H, I) == 18_800_640

    # Placement into 128-row banks with a small I so the test stays cheap: every expert
    # row index up to 127 lands where its id says, w1/w3 split the gate_up row.
    Es, Hs, Is = 128, 64, 32
    small = _dsfp4_bank_specs(Es, Hs, Is)
    banks = {name: [torch.zeros(shape, dtype=dt) for _ in range(3)] for name, (shape, dt) in small.items()}
    w1 = torch.full((Is, Hs // 2), 1, dtype=torch.float8_e4m3fn)  # compared via the uint8 view
    w3 = torch.full((Is, Hs // 2), 2, dtype=torch.float8_e4m3fn)
    w2 = torch.full((Hs, Is // 2), 3, dtype=torch.float8_e4m3fn)
    assert _place_dsfp4_parsed(banks, 2, 127, "w1", "weight", w1, Is) == 2
    _place_dsfp4_parsed(banks, 2, 127, "w3", "weight", w3, Is)
    _place_dsfp4_parsed(banks, 2, 127, "w2", "weight", w2, Is)
    gu = banks["gate_up_packed"][2]
    assert torch.equal(gu[127, :Is], w1.view(torch.uint8))
    assert torch.equal(gu[127, Is:], w3.view(torch.uint8))
    assert torch.equal(banks["down_packed"][2][127], w2.view(torch.uint8))
    assert int(gu[126].sum()) == 0 and int(banks["gate_up_packed"][1].sum()) == 0
    s = torch.full((Is, Hs // 32), 130, dtype=torch.uint8).view(torch.float8_e8m0fnu)
    _place_dsfp4_parsed(banks, 0, 0, "w3", "scale", s, Is)
    assert torch.equal(banks["gate_up_scale"][0][0, Is:].view(torch.uint8), s.view(torch.uint8))
    with pytest.raises(IndexError):
        _place_dsfp4_parsed(banks, 0, 128, "w1", "weight", w1, Is)  # beyond the stage width


def test_dspark_bank_bytes_estimate():
    args = parse_raw_args(V41_CONFIG)
    assert dspark_bank_bytes_estimate(args, built=False) == 0
    got = dspark_bank_bytes_estimate(args, built=True)
    assert got == 3 * 128 * 18_800_640  # 6.7 GiB of pinned stage banks
    assert abs(got / 2**30 - 6.72) < 0.01
    # 0731-style args without dspark_n_routed_experts fall back to the tower width.
    from types import SimpleNamespace

    v4 = SimpleNamespace(n_mtp_layers=3, n_routed_experts=256, dim=4096, moe_inter_dim=2048)
    from freetoken.moe.offload_cache import _BANK_BYTES_PER_EXPERT

    assert dspark_bank_bytes_estimate(v4, built=True) == 3 * 256 * _BANK_BYTES_PER_EXPERT["ds_fp4"](4096, 2048)
