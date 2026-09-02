"""Qwen4ExpMTPHead: state-dict contract, wiring variants, and a toy-geometry forward."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from .common import fill_weights, fresh_ctx, hf_config, requires_cuda, toy_hf_config

@pytest.fixture(autouse=True)
def _bf16_mtp_experts(monkeypatch):
    # These tests assert the bf16 resident-expert contract; the offload/fp8 modes need
    # a real offload cache / 128-divisible dims the toy geometries lack, and have their
    # own tests.
    monkeypatch.setenv("FREETOKEN_MTP_EXPERTS", "bf16")


MTP_TEXT_OVERRIDES = dict(
    mtp={
        "hybrid": True,
        "layer_types": ["full_attention"],
        "mtp_use_hidden_state_from_layer": None,
        "num_hidden_layers": 1,
        "rope_theta": 10000000,
    },
    mtp_use_dedicated_embeddings=False,
)


def _config(toy: bool = False, **overrides):
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.models.qwen4_exp.config import parse_config

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    make = toy_hf_config if toy else hf_config
    return parse_config(make(**{**MTP_TEXT_OVERRIDES, **overrides}))


def _head(config, wiring: str = "per_stream"):
    from freetoken.models.qwen4_exp.mtp import Qwen4ExpMTPHead

    return Qwen4ExpMTPHead(config, wiring=wiring)


def test_config_parses_mtp_section():
    args = _config().qwen4_args
    assert args.mtp_num_layers == 1
    assert args.mtp_layer_types == ("full_attention",)
    assert args.mtp_use_dedicated_embeddings is False


def test_config_without_mtp_section_defaults_to_no_head():
    args = _config(mtp=None).qwen4_args
    assert args.mtp_num_layers == 0
    with pytest.raises(ValueError, match="exactly one full_attention MTP layer"):
        _head(_config(mtp=None))


def test_extend_config_for_speculative():
    from freetoken.models.qwen4_exp.config import extend_config_for_speculative

    config = _config()
    extended = extend_config_for_speculative(config, num_draft_tokens=1)
    full = next(g for g in extended.attention_groups if g.name == "full")
    assert full.layer_ids[-1] == config.num_layers
    assert full.num_index_layers == len(full.layer_ids)
    assert extended.num_layers == config.num_layers  # main tower untouched
    assert extended.num_moe_layers == config.num_moe_layers
    assert extended.kv_layer_id_bound == config.num_layers + 1
    assert extended.num_speculative_tokens == 1
    assert not extended.is_linear_layer(config.num_layers)
    # the per-token KV cost grows by exactly one full-attention layer's bytes
    base_spec = next(s for s in config.kv_cache_group_specs() if s.name == "full")
    ext_spec = next(s for s in extended.kv_cache_group_specs() if s.name == "full")
    assert ext_spec.num_layers == base_spec.num_layers + 1

    with pytest.raises(ValueError, match="no MTP head"):
        extend_config_for_speculative(_config(mtp=None), num_draft_tokens=1)
    with pytest.raises(ValueError, match=">= 1"):
        extend_config_for_speculative(config, num_draft_tokens=0)


def test_mtp_decoder_config_idempotent_on_extended_config():
    from freetoken.models.qwen4_exp.config import extend_config_for_speculative
    from freetoken.models.qwen4_exp.mtp import mtp_decoder_config

    extended = extend_config_for_speculative(_config(), num_draft_tokens=1)
    layer_config = mtp_decoder_config(extended)
    full = next(g for g in layer_config.attention_groups if g.name == "full")
    assert full.layer_ids.count(extended.num_layers) == 1


def test_spec_extended_pool_accepts_mtp_layer_and_widens_ring():
    from freetoken.kvcache import create_kvcache_pool
    from freetoken.kvcache.qsa_pool import QSAKVCache
    from freetoken.models.qwen4_exp.config import extend_config_for_speculative

    config = extend_config_for_speculative(_config(toy=True), num_draft_tokens=1)
    pool = create_kvcache_pool(
        model_config=config,
        num_pages=4,
        page_size=64,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        num_req_slots=2,
    )
    assert isinstance(pool, QSAKVCache)
    assert pool._ring_capacity == QSAKVCache.ring_capacity_for(4, 1) == 8
    # the MTP layer id resolves to the 13th (toy: 2nd) storage slab without a bounds error
    assert pool.k_cache(config.num_layers).shape == pool.k_cache(3).shape
    # the cost model budgets the wider ring: fixed term grows vs the unextended config
    base = _config(toy=True)
    tp = SimpleNamespace(size=1)
    engine_cfg = SimpleNamespace(
        model_config=config, max_running_req=1, page_size=64, tp_info=tp, dtype=torch.bfloat16
    )
    base_cfg = SimpleNamespace(
        model_config=base, max_running_req=1, page_size=64, tp_info=tp, dtype=torch.bfloat16
    )
    per_ext, fixed_ext, _, _ = QSAKVCache.kv_cost(engine_cfg)
    per_base, fixed_base, _, _ = QSAKVCache.kv_cost(base_cfg)
    assert fixed_ext > fixed_base
    assert per_ext > per_base


def test_state_dict_contract():
    config = _config()
    head = _head(config)
    args = config.qwen4_args
    h, hc = args.hidden_size, args.hc_count
    i, e = config.moe_intermediate_size, config.num_experts
    sd = head.state_dict(prefix="mtp")

    layer = "mtp.layers.0"
    expected = {
        "mtp.pre_fc_norm_hidden.weight": (hc * h,),
        "mtp.pre_fc_norm_embedding.weight": (h,),
        "mtp.fc_hidden.weight": (h, h),
        "mtp.fc_embedding.weight": (h, h),
        "mtp.hyper_connection_mixer.hc_norm.weight": (hc * h,),
        "mtp.hyper_connection_mixer.input_mix_weight_down.weight": (args.hc_lowrank, hc * h),
        "mtp.hyper_connection_mixer.input_mix_weight_up.weight": (hc * h, args.hc_lowrank),
        f"{layer}.mlp.experts.gate_up_proj": (e, 2 * i, h),
        f"{layer}.mlp.experts.down_proj": (e, h, i),
        f"{layer}.mlp.gate.weight": (e, h),
        f"{layer}.mlp.shared_expert.gate_up_proj.weight": (
            2 * config.shared_expert_intermediate_size,
            h,
        ),
        f"{layer}.mlp.shared_expert.down_proj.weight": (
            h,
            config.shared_expert_intermediate_size,
        ),
        f"{layer}.mlp.shared_expert_gate.weight": (1, h),
    }
    for name, shape in expected.items():
        assert name in sd, name
        assert tuple(sd[name].shape) == shape, (name, tuple(sd[name].shape), shape)

    merged_rows = args.hc_lowrank + hc + (-(args.hc_lowrank + hc)) % 16
    for hc_block in ("attn_hyper_connection", "mlp_hyper_connection"):
        key = f"{layer}.{hc_block}.input_mix_weight_down_block_inject.weight"
        assert tuple(sd[key].shape) == (merged_rows, hc * h)

    qkv = sd[f"{layer}.self_attn.qkv_proj.weight"]
    assert qkv.shape == (
        2 * config.num_qo_heads * config.head_dim + 2 * config.num_kv_heads * config.head_dim,
        h,
    )
    # no PLE, no linear_attn, no duplicated embeddings/lm_head in the head's subtree
    assert not any(".ple." in k or ".linear_attn." in k for k in sd)
    assert not any("embed_tokens" in k or "lm_head" in k for k in sd)


def test_build_input_collapse_invariant():
    """With identical residual streams, per_stream and mean_collapse coincide; sum does not."""
    # distinct max_position: head construction is CPU here, and get_rope memoizes per
    # geometry -- sharing the CUDA forward test's key would cache a CPU cos/sin table
    config = _config(toy=True, max_position_embeddings=2048)
    args = config.qwen4_args
    heads = {w: _head(config, wiring=w) for w in ("per_stream", "mean_collapse", "sum_collapse")}
    # The grouped norm's weight is per feature, so identical streams diverge after (1+w)
    # scaling unless the weight repeats per stream; tie it so the invariant is exact.
    norm_base = torch.randn(args.hidden_size) * 0.05
    for head in heads.values():
        # same seed per head -> identical weights across the wiring variants
        fill_weights(head, seed=3, device=torch.device("cpu"))
        head.pre_fc_norm_hidden.weight = norm_base.repeat(args.hc_count)
    t = 5
    one_stream = torch.randn(t, args.hidden_size)
    premix = one_stream.repeat(1, args.hc_count)
    embeds = torch.randn(t, args.hidden_size)
    outs = {w: h.build_input(premix, embeds) for w, h in heads.items()}
    torch.testing.assert_close(outs["per_stream"], outs["mean_collapse"], rtol=2e-2, atol=2e-2)
    assert not torch.allclose(outs["per_stream"], outs["sum_collapse"], rtol=2e-2, atol=2e-2)


@requires_cuda
@pytest.mark.parametrize("wiring", ["per_stream", "mean_collapse", "sum_collapse"])
def test_forward_toy_geometry(wiring):
    from freetoken.models.qwen4_exp.attention import TorchDenseQSAReference
    from freetoken.moe.fused import FusedMoe
    from freetoken.utils.torch_utils import torch_dtype

    config = _config(toy=True)
    args = config.qwen4_args
    device = torch.device("cuda")
    t = 9

    with torch.device(device), torch_dtype(torch.bfloat16):
        head = _head(config, wiring=wiring)
    fill_weights(head, seed=11, device=device)

    fresh_ctx(
        attn_backend=TorchDenseQSAReference(
            config, num_slots=2, max_len=64, device=device, dtype=torch.bfloat16
        ),
        moe_backend=FusedMoe(),
    )
    batch = SimpleNamespace(
        positions=torch.arange(t, dtype=torch.int32, device=device),
        padded_reqs=[SimpleNamespace(extend_len=t, table_idx=0, cached_len=0)],
        is_prefill=True,
        is_decode=False,
    )
    premix = torch.randn(t, args.hc_count * args.hidden_size, device=device, dtype=torch.bfloat16)
    embeds = torch.randn(t, args.hidden_size, device=device, dtype=torch.bfloat16)
    out = head.forward(premix, embeds, batch)
    assert out.shape == (t, args.hidden_size)
    assert out.isfinite().all()
