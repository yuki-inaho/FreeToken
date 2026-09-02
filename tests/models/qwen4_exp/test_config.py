"""qwen4_exp.parse_config against a synthetic config shaped like the RadixArk NVFP4 checkpoint."""

from types import SimpleNamespace

import pytest

from freetoken.attention import AttnType
from freetoken.models.config import FullAttentionGroupConfig, LinearGatedDeltaGroupConfig
from freetoken.models.qwen4_exp.config import parse_config


def _text_config():
    return SimpleNamespace(
        num_hidden_layers=48,
        hidden_size=2560,
        vocab_size=248320,
        head_dim=256,
        num_attention_heads=24,
        num_key_value_heads=2,
        layer_types=[
            "full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(48)
        ],
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000000,
            "partial_rotary_factor": 0.25,
            "mrope_interleaved": True,
            "mrope_section": [11, 11, 10],
        },
        max_position_embeddings=262144,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        tie_word_embeddings=False,
        num_experts=512,
        num_experts_per_tok=10,
        moe_intermediate_size=640,
        shared_expert_intermediate_size=640,
        linear_num_key_heads=16,
        linear_num_value_heads=48,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        output_gate_type="sigmoid",
        indexer_n_heads=4,
        indexer_kv_heads=1,
        indexer_head_dim=128,
        indexer_budget=2048,
        indexer_compress_ratio=4,
        hc_count=4,
        hc_lowrank=320,
        ple_layer_ids=[2],
        ple_embed_dim=2560,
        ple_conv_kernel_size=4,
        ngram_size=3,
        heads_per_ngram=8,
        ngram_vocab_size_base=20000000,
        make_ngram_vocab_size_divisible_by=128,
        split_ngram_parts=128,
        bos_token_id=248044,
        eos_token_id=248044,
    )


def _hf_config():
    return SimpleNamespace(
        model_type="qwen4_exp",
        architectures=["Qwen4ExpForConditionalGeneration"],
        image_token_id=248056,
        text_config=_text_config(),
        quantization_config={
            "quant_algo": "NVFP4",
            "quant_method": "modelopt",
            "ignore": [
                "model.embed_tokens",
                "mtp.*",
                "model.mtp.*",
                "*.self_attn.*",
                "*.linear_attn.*",
                "*.mlp.gate*",
                "*.mlp.shared_expert.*",
                "*.mlp.shared_expert_gate*",
                "*hyper_connection*",
                "*.ple.*",
                "model.visual.*",
                "model.language_model.embed_tokens",
                "lm_head",
            ],
        },
    )


def test_groups_and_layer_split():
    cfg = parse_config(_hf_config())
    full = [g for g in cfg.attention_groups if isinstance(g, FullAttentionGroupConfig)]
    linear = [g for g in cfg.attention_groups if isinstance(g, LinearGatedDeltaGroupConfig)]
    assert len(full) == 1 and len(linear) == 1
    assert full[0].layer_ids == tuple(range(3, 48, 4))
    assert len(linear[0].layer_ids) == 36
    assert full[0].index_head_dim == 128
    assert full[0].num_index_layers == 12
    assert full[0].index_ratio == 4
    assert full[0].rotary_config.rotary_dim == 64
    assert full[0].rotary_config.scaling is None
    assert linear[0].output_gate == "sigmoid"
    assert linear[0].num_key_heads == 16 and linear[0].num_value_heads == 48


def test_kv_specs_resolve_qsa():
    cfg = parse_config(_hf_config())
    specs = {s.name: s for s in cfg.kv_cache_group_specs()}
    assert specs["full"].attn_type is AttnType.QSA
    assert specs["full"].index_ratio == 4
    assert cfg.attn_type_for_layer(3) is AttnType.QSA
    assert cfg.attn_type_for_layer(0) is AttnType.LINEAR
    assert cfg.has_linear_attention


def test_moe_and_quant_flags():
    cfg = parse_config(_hf_config())
    assert cfg.num_experts == 512
    assert cfg.num_experts_per_tok == 10
    assert cfg.norm_topk_prob is True
    assert cfg.moe_enabled
    assert cfg.expert_quant == "nvfp4"
    assert cfg.dense_quant == "none"
    assert cfg.attn_quant == "none"
    assert cfg.lm_head_quant == "none"


def test_unquantized_config_parses():
    hf = _hf_config()
    hf.quantization_config = None
    assert parse_config(hf).expert_quant == "none"


def test_qwen4_args_payload():
    args = parse_config(_hf_config()).qwen4_args
    assert args.ple_layer_ids == (1,)
    assert args.hc_count == 4 and args.hc_lowrank == 320
    assert args.index_topk_blocks == 512
    assert args.num_ngram_heads == 16
    assert args.ngram_head_dim == 160
    assert args.ple_conv_state_len == 9
    assert args.ple_state_width == 10240
    assert args.ngram_boundary_token_id == 248044


def test_ple_on_full_attention_layer_rejected():
    hf = _hf_config()
    hf.text_config.ple_layer_ids = [4]  # one-indexed 4 == zero-based 3, a full_attention layer
    with pytest.raises(ValueError, match="linear_attention"):
        parse_config(hf)


def test_output_gate_null_falls_back_to_hidden_act():
    hf = _hf_config()
    hf.text_config.output_gate_type = None
    linear = [
        g
        for g in parse_config(hf).attention_groups
        if isinstance(g, LinearGatedDeltaGroupConfig)
    ]
    assert linear[0].output_gate == "silu"


def test_eos_token_id_list_uses_the_first_entry():
    base = parse_config(_hf_config()).qwen4_args.ngram_boundary_token_id
    hf = _hf_config()
    hf.text_config.eos_token_id = [base, base + 1]
    assert parse_config(hf).qwen4_args.ngram_boundary_token_id == base


def test_dense_quant_override_upgrades_only_bf16_parts(monkeypatch):
    from freetoken.models.qwen4_exp.config import dense_fp8_part_of

    monkeypatch.setenv("FREETOKEN_DENSE_QUANT", "fp8")
    monkeypatch.delenv("FREETOKEN_DENSE_FP8_PARTS", raising=False)
    cfg = parse_config(_hf_config())
    assert cfg.expert_quant == "nvfp4"  # the routed experts' native quant is untouched
    assert cfg.attn_quant == "fp8_pertensor"
    assert cfg.dense_quant == "fp8_pertensor"
    assert cfg.lm_head_quant == "fp8_pertensor"
    args = cfg.qwen4_args
    assert set(args.dense_fp8_parts) == {"gdn", "attn", "hc", "shared", "lm_head", "mtp"}
    # part gating: selected parts take the flag, unselected ones stay bf16
    assert args.part_quant("gdn", cfg.attn_quant) == "fp8_pertensor"
    assert args.part_quant("index", cfg.attn_quant) == "none"
    assert args.part_quant("ple", cfg.dense_quant) == "none"
    # loader predicate mirrors the constructors
    parts = set(args.dense_fp8_parts)
    assert dense_fp8_part_of("model.layers.0.linear_attn.in_proj_qkvz.weight", parts) == "gdn"
    assert dense_fp8_part_of("model.layers.0.linear_attn.in_proj_ba.weight", parts) is None
    assert dense_fp8_part_of("model.layers.3.self_attn.qkv_proj.weight", parts) == "attn"
    assert dense_fp8_part_of("model.layers.3.self_attn.indexer.index_qk_proj.weight", parts) is None
    assert dense_fp8_part_of("model.hyper_connection_mixer.input_mix_weight_down.weight", parts) == "hc"
    assert dense_fp8_part_of("model.layers.0.mlp.shared_expert.down_proj.weight", parts) == "shared"
    assert dense_fp8_part_of("model.layers.0.mlp.gate.weight", parts) is None
    assert dense_fp8_part_of("lm_head.weight", parts) == "lm_head"
    # "mtp" is in the default set: the MTP layer's attention takes the attn part; without it, None
    assert dense_fp8_part_of("mtp.layers.0.self_attn.o_proj.weight", parts) == "attn"
    assert dense_fp8_part_of("mtp.layers.0.self_attn.o_proj.weight", parts - {"mtp"}) is None
    assert dense_fp8_part_of("mtp.layers.0.self_attn.o_proj.weight", parts | {"mtp"}) == "attn"
    assert dense_fp8_part_of("mtp.fc_hidden.weight", parts | {"mtp"}) is None


def test_dense_quant_parts_subset(monkeypatch):
    monkeypatch.setenv("FREETOKEN_DENSE_QUANT", "fp8")
    monkeypatch.setenv("FREETOKEN_DENSE_FP8_PARTS", "gdn,lm_head")
    cfg = parse_config(_hf_config())
    assert cfg.attn_quant == "fp8_pertensor"
    assert cfg.dense_quant == "none"
    assert cfg.lm_head_quant == "fp8_pertensor"
    assert cfg.qwen4_args.dense_fp8_parts == ("gdn", "lm_head")
    assert cfg.qwen4_args.part_quant("attn", cfg.attn_quant) == "none"
    monkeypatch.setenv("FREETOKEN_DENSE_FP8_PARTS", "gdn,bogus")
    with pytest.raises(ValueError):
        parse_config(_hf_config())


@pytest.mark.parametrize("value", ["auto", "bf16", None])
def test_dense_quant_off_is_byte_identical_config(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("FREETOKEN_DENSE_QUANT", raising=False)
    else:
        monkeypatch.setenv("FREETOKEN_DENSE_QUANT", value)
    cfg = parse_config(_hf_config())
    assert (cfg.attn_quant, cfg.dense_quant, cfg.lm_head_quant) == ("none", "none", "none")
    assert cfg.qwen4_args.dense_fp8_parts == ()
