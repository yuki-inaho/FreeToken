"""DSpark drafter (DeepSeek-V4-Flash-0731): args, config extension, window-tier / scratch
sizing, the stage modules' parameter contract, and the expert-bank builders. CPU, toy geometry
(128-divisible dims so the block-FP8 scale shapes are well-formed; modules built on ``meta``)."""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest
import torch

from freetoken.kvcache.dsv4_cost_model import (
    dsv4_cache_per_page,
    dsv4_n_scratch,
    dsv4_pool_bytes,
    dsv4_pool_sizes,
    dsv4_window_unit_bytes,
)
from freetoken.kvcache.dsv4_paged_pool import DSV4PagedKVCache
from freetoken.models.deepseek_v4.args import DeepseekV4Args, load_args
from freetoken.models.deepseek_v4.config import extend_config_for_speculative, parse_config

P = 128
RATIOS = (0, 0, 4, 128, 4, 128, 4, 0)
TOY = dict(
    n_layers=8,
    n_hash_layers=1,
    compress_ratios=RATIOS,
    max_seq_len=1024,
    dim=128,
    vocab_size=64,
    n_heads=2,
    head_dim=128,
    rope_head_dim=32,
    q_lora_rank=128,
    o_groups=2,
    o_lora_rank=64,
    moe_inter_dim=64,
    n_routed_experts=8,
    n_activated_experts=2,
    index_n_heads=2,
    index_head_dim=128,
    window_size=P,
    n_mtp_layers=3,
    dspark_block_size=5,
    dspark_noise_token_id=7,
    dspark_target_layer_ids=(5, 6, 7),
    dspark_markov_rank=16,
)


def _args(**over) -> DeepseekV4Args:
    base = dict(TOY)
    base.update(over)
    return DeepseekV4Args(**base)


def _write_checkpoint_args(tmp_path, **over) -> str:
    raw = dict(TOY)
    raw.update(over)
    raw["compress_ratios"] = list(raw["compress_ratios"])
    raw["dspark_target_layer_ids"] = list(raw["dspark_target_layer_ids"])
    os.makedirs(tmp_path / "inference", exist_ok=True)
    with open(tmp_path / "inference" / "config.json", "w") as f:
        json.dump(raw, f)
    return str(tmp_path)


# --------------------------------------------------------------------------- #
# args / config
# --------------------------------------------------------------------------- #
def test_args_parse_dspark_fields(tmp_path):
    path = _write_checkpoint_args(tmp_path)
    args = load_args(path, max_batch_size=1)
    assert args.dspark_enabled
    assert args.dspark_block_size == 5
    assert args.dspark_noise_token_id == 7
    assert args.dspark_target_layer_ids == (5, 6, 7)  # JSON list -> tuple
    assert args.dspark_markov_rank == 16
    assert args.n_mtp_layers == 3
    assert args.n_extra_window_layers == 0 and args.n_window_layers == 8
    # A checkpoint without the drafter (the preview's plain MTP layer) is not DSpark.
    assert not DeepseekV4Args().dspark_enabled
    assert not _args(dspark_block_size=0).dspark_enabled


def test_extend_config_registers_window_layers_and_block(tmp_path):
    path = _write_checkpoint_args(tmp_path)
    hf = SimpleNamespace(_name_or_path=path, max_position_embeddings=2048)
    cfg = parse_config(hf)
    assert cfg.num_extra_kv_layers == 0 and cfg.num_speculative_tokens == 0
    ext = extend_config_for_speculative(cfg, 5)
    assert ext.num_extra_kv_layers == 3
    assert ext.num_speculative_tokens == 5  # the block size, not a free knob
    assert ext.kv_layer_id_bound == 11
    assert ext.num_layers == 8  # the main tower's count drives construction / MoE layers
    args = ext.dsv4_args
    assert args.n_extra_window_layers == 3 and args.n_window_layers == 11
    # The stages' ratios are appended (window-only) when the args list stops at the tower.
    assert len(args.compress_ratios) == 11 and args.compress_ratios[8:] == (0, 0, 0)
    # Verifying fewer than the block is allowed; more is not.
    assert extend_config_for_speculative(cfg, 2).num_speculative_tokens == 5
    with pytest.raises(ValueError, match=r"\[1, 5\]"):
        extend_config_for_speculative(cfg, 6)


def test_extend_config_rejects_non_dspark_checkpoint(tmp_path):
    path = _write_checkpoint_args(tmp_path, dspark_block_size=0)
    cfg = parse_config(SimpleNamespace(_name_or_path=path, max_position_embeddings=2048))
    with pytest.raises(ValueError, match="no DSpark drafter"):
        extend_config_for_speculative(cfg, 1)


def test_extend_config_rejects_compressed_stage_ratio(tmp_path):
    path = _write_checkpoint_args(tmp_path, compress_ratios=RATIOS + (4, 0, 0))
    cfg = parse_config(SimpleNamespace(_name_or_path=path, max_position_embeddings=2048))
    with pytest.raises(ValueError, match="window-only"):
        extend_config_for_speculative(cfg, 5)


# --------------------------------------------------------------------------- #
# pool / cost model
# --------------------------------------------------------------------------- #
def test_pool_grows_only_the_window_tier():
    base, spec = _args(), _args(n_extra_window_layers=3)
    sizes = dsv4_pool_sizes(num_pages=8, args=spec, swa_ratio=0.5, P=P)
    pool = DSV4PagedKVCache(
        sizes=sizes, args=spec, device=torch.device("cpu"), dtype=torch.bfloat16, P=P,
        n_scratch=1,
    )
    assert len(pool.window_pool) == 11
    assert len(pool.cmp_pool) == 8 and len(pool.state_ring) == 8
    assert pool.num_layers == 8
    # The extra layers are addressable by the generic window write.
    pool.store_window(torch.ones(2, 128, dtype=torch.bfloat16), 10, torch.tensor([0, 1]))
    assert pool.window_pool[10][1, 0].item() == 1.0
    # The exact byte model tracks the allocation, and only the window term moved.
    assert pool.total_bytes() == dsv4_pool_bytes(sizes, spec, n_scratch=1)
    base_sizes = dsv4_pool_sizes(num_pages=8, args=base, swa_ratio=0.5, P=P)
    assert dsv4_pool_bytes(sizes, spec, 1) - dsv4_pool_bytes(base_sizes, base, 1) == (
        3 * sizes.n_win_slots * 128 * 2
    )
    assert dsv4_window_unit_bytes(spec, P) - dsv4_window_unit_bytes(base, P) == 3 * 128 * 2
    assert dsv4_cache_per_page(spec, 0.5, P) - dsv4_cache_per_page(base, 0.5, P) == (
        3 * round(0.5 * P) * 128 * 2
    )


def test_n_scratch_covers_the_verify_rows():
    def cfg(mrr, spec):
        return SimpleNamespace(
            max_running_req=mrr, model_config=SimpleNamespace(num_speculative_tokens=spec)
        )

    assert dsv4_n_scratch(cfg(1, 0)) == 2  # the non-speculative default (max_running_req + 1)
    assert dsv4_n_scratch(cfg(1, 5)) == 6  # a k+1-row verify needs a scratch row per row
    assert dsv4_n_scratch(cfg(8, 5)) == 9


# --------------------------------------------------------------------------- #
# stage modules
# --------------------------------------------------------------------------- #
_BLOCK_PARAMS = {
    "attn.attn_sink",
    "attn.wq_a.weight", "attn.wq_a.scale",
    "attn.q_norm.weight",
    "attn.wq_b.weight", "attn.wq_b.scale",
    "attn.wkv.weight", "attn.wkv.scale",
    "attn.kv_norm.weight",
    "attn.wo_a",  # dequantized to bf16 at load (reference bf16 einsum)
    "attn.wo_b.weight", "attn.wo_b.scale",
    "attn_norm.weight", "ffn_norm.weight",
    "ffn.gate.weight", "ffn.gate.bias",
    "ffn.shared_experts.w1.weight", "ffn.shared_experts.w1.scale",
    "ffn.shared_experts.w2.weight", "ffn.shared_experts.w2.scale",
    "ffn.shared_experts.w3.weight", "ffn.shared_experts.w3.scale",
    "hc_attn_fn", "hc_ffn_fn", "hc_attn_base", "hc_ffn_base", "hc_attn_scale", "hc_ffn_scale",
}
_STAGE0_EXTRA = {"main_proj.weight", "main_proj.scale", "main_norm.weight"}
_LAST_EXTRA = {
    "norm.weight", "hc_head_fn", "hc_head_base", "hc_head_scale",
    "markov_head.markov_w1.weight", "markov_head.markov_w2.weight",
    "confidence_head.proj.weight",
}


def _stage_args() -> DeepseekV4Args:
    return _args(compress_ratios=RATIOS + (0, 0, 0), n_extra_window_layers=3)


def test_stage_parameter_contract_matches_checkpoint_layout():
    from freetoken.models.deepseek_v4.dspark import DSparkBlock

    args = _stage_args()
    with torch.device("meta"):
        stages = [DSparkBlock(args.n_layers + s, args, s) for s in range(3)]
    for s, stage in enumerate(stages):
        names = {n for n, _ in stage.named_parameters()}
        expected = set(_BLOCK_PARAMS)
        if s == 0:
            expected |= _STAGE0_EXTRA
        if s == 2:
            expected |= _LAST_EXTRA
        assert names == expected, (s, names ^ expected)
        assert stage.layer_id == 8 + s and stage.attn.compress_ratio == 0
        assert stage.ffn.experts.layer_id == 8 + s  # the offload-cache bank layer id
        assert not stage.ffn.gate.hash  # bias-routed
    assert stages[0].main_proj.weight.shape == (128, 3 * 128)
    assert stages[2].markov_head.markov_w1.weight.shape == (64, 16)
    assert stages[2].markov_head.markov_w2.weight.shape == (64, 16)
    assert stages[2].confidence_head.proj.weight.shape == (1, 128 + 16)
    assert stages[2].confidence_head.proj.weight.dtype == torch.float32
    assert stages[2].hc_head_fn.shape == (4, 4 * 128)


def test_transformer_builds_stages_only_when_enabled(monkeypatch):
    from freetoken.models.deepseek_v4.model import Transformer

    args = _stage_args()
    monkeypatch.delenv("FREETOKEN_LOAD_MTP", raising=False)
    with torch.device("meta"):
        plain = Transformer(args)
    assert len(plain.mtp) == 0 and not plain._keep_main_hidden

    monkeypatch.setenv("FREETOKEN_LOAD_MTP", "1")
    with torch.device("meta"):
        spec = Transformer(args)
    assert len(spec.mtp) == 3 and spec._keep_main_hidden
    assert spec.target_layer_ids == (5, 6, 7)
    assert [st.layer_id for st in spec.mtp] == [8, 9, 10]
    names = {n for n, _ in spec.named_parameters()}
    assert "mtp.0.main_proj.weight" in names and "mtp.2.markov_head.markov_w1.weight" in names
    assert "mtp.1.main_proj.weight" not in names and "mtp.1.norm.weight" not in names


def test_causal_lm_yields_stage_experts_after_the_tower(monkeypatch):
    from freetoken.models.deepseek_v4.model import DeepseekV4ForCausalLM

    args = _stage_args()
    monkeypatch.setenv("FREETOKEN_LOAD_MTP", "1")
    cfg = SimpleNamespace(dsv4_args=args)
    with torch.device("meta"):
        model = DeepseekV4ForCausalLM(cfg)
    layers = list(model._iter_offload_moe_layers())
    assert [m.layer_id for m in layers] == list(range(11))


def test_draft_block_ids():
    from freetoken.models.deepseek_v4.dspark import draft_block_ids

    ids = draft_block_ids(torch.tensor([3, 9], dtype=torch.int32), 5, 7)
    assert ids.tolist() == [[3, 7, 7, 7, 7], [9, 7, 7, 7, 7]]
    assert ids.dtype == torch.int32


# --------------------------------------------------------------------------- #
# expert banks
# --------------------------------------------------------------------------- #
def test_place_parsed_routes_projections():
    from freetoken.models.deepseek_v4.weight import _dsfp4_bank_specs, _place_dsfp4_parsed

    E, H, I = 2, 64, 32
    specs = _dsfp4_bank_specs(E, H, I)
    banks = {name: [torch.zeros(*shape, dtype=dtype) for _ in range(3)] for name, (shape, dtype) in specs.items()}
    w1 = torch.ones(I, H // 2, dtype=torch.uint8)
    w3 = torch.full((I, H // 2), 3, dtype=torch.uint8)
    w2 = torch.full((H, I // 2), 2, dtype=torch.uint8)
    assert _place_dsfp4_parsed(banks, 1, 1, "w1", "weight", w1, I) == 1
    _place_dsfp4_parsed(banks, 1, 1, "w3", "weight", w3, I)
    _place_dsfp4_parsed(banks, 1, 1, "w2", "weight", w2, I)
    gu = banks["gate_up_packed"][1]
    assert (gu[1, :I] == 1).all() and (gu[1, I:] == 3).all() and (gu[0] == 0).all()
    assert (banks["down_packed"][1][1] == 2).all()
    assert (banks["gate_up_packed"][0] == 0).all() and (banks["gate_up_packed"][2] == 0).all()
    s = torch.ones(I, H // 32, dtype=torch.float8_e8m0fnu)
    _place_dsfp4_parsed(banks, 2, 0, "w1", "scale", s, I)
    assert banks["gate_up_scale"][2][0, :I].view(torch.uint8).eq(s.view(torch.uint8)).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="pinning host banks needs CUDA")
def test_dummy_stage_banks_shape_and_pinning():
    from freetoken.models.deepseek_v4.weight import load_dspark_expert_bank_layers

    args = _stage_args()
    banks = load_dspark_expert_bank_layers(None, args, dummy=True)
    assert set(banks) == {"gate_up_packed", "gate_up_scale", "down_packed", "down_scale"}
    for name, per_stage in banks.items():
        assert len(per_stage) == 3
    E, H, I = args.n_routed_experts, args.dim, args.moe_inter_dim
    assert banks["gate_up_packed"][0].shape == (E, 2 * I, H // 2)
    assert banks["gate_up_scale"][0].shape == (E, 2 * I, H // 32)
    assert banks["down_packed"][0].shape == (E, H, I // 2)
    assert banks["down_scale"][0].shape == (E, H, I // 32)
    assert banks["gate_up_packed"][0].dtype == torch.uint8
    assert banks["gate_up_scale"][0].dtype == torch.float8_e8m0fnu
    # (Pinning goes through cudaHostRegister on the HostBank, which torch's is_pinned() does
    # not track -- the offload cache's copy plan is what validates it at boot.)
