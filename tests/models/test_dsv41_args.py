"""DeepSeek-V4.1-Flash args: the strict inference/config.json loader, the topology
validation, and the CSA2 / CED helpers on the released V4.1-Flash config. CPU only.

The expectations come from the checkpoint itself (the `inference/config.json` values below
are the released ones) and from the tech report's layer roles, not from the args code."""

from __future__ import annotations

import json
import os

import pytest

from freetoken.models.deepseek_v41.args import DeepseekV41Args, load_args, parse_raw_args

# The released DeepSeek-V4.1-Flash inference/config.json (text-relevant keys).
V41_CONFIG = {
    "vocab_size": 129280,
    "dim": 5120,
    "moe_inter_dim": 2304,
    "n_layers": 40,
    "n_mtp_layers": 3,
    "n_heads": 64,
    "n_routed_experts": 384,
    "n_shared_experts": 1,
    "n_activated_experts": 6,
    "score_func": "sqrtsoftplus",
    "route_scale": 1.5,
    "swiglu_limit": 10.0,
    "q_lora_rank": 1280,
    "head_dim": 512,
    "rope_head_dim": 64,
    "norm_eps": 1e-20,
    "o_groups": 8,
    "o_lora_rank": 1024,
    "window_size": 128,
    "compress_ratios": [0, 0] + [2] * 18 + [1] * 20 + [0, 0, 0],
    "kv_source_layers": [2, 8, 14, 20],
    "index_source_layers": [2, 8, 14, 20, 24, 28, 32, 36],
    "compress_rope_theta": 160000,
    "original_seq_len": 65536,
    "rope_theta": 10000,
    "rope_factor": 16,
    "beta_fast": 32,
    "beta_slow": 1,
    "index_n_heads": 32,
    "index_head_dim": 128,
    "index_topk": 512,
    "candidate_source_layer": 20,
    "candidate_topk_blocks": 2048,
    "candidate_block_size": 8,
    "hc_mult": 4,
    "hc_sinkhorn_iters": 20,
    "engram_layer_ids": [1, 14],
    "engram_num_embeddings": [384006168, 384016682],
    "engram_max_ngram_size": 4,
    "engram_vocab_size": 16000000,
    "engram_n_heads": 8,
    "engram_head_dim": 256,
    "engram_pad_id": 2,
    "engram_compressed_vocab_size": 99092,
    "dspark_block_size": 5,
    "dspark_noise_token_id": 128799,
    "dspark_target_layer_ids": [37, 38, 39],
    "dspark_markov_rank": 256,
    "dspark_n_routed_experts": 128,
    "dspark_n_activated_experts": 3,
    "dtype": "fp8",
    "expert_dtype": "fp4",
}


def _write(tmp_path, raw: dict) -> str:
    os.makedirs(tmp_path / "inference", exist_ok=True)
    with open(tmp_path / "inference" / "config.json", "w") as f:
        json.dump(raw, f)
    return str(tmp_path)


def test_load_args_round_trips_the_released_config(tmp_path):
    args = load_args(_write(tmp_path, V41_CONFIG), max_seq_len=8192)
    for key, value in V41_CONFIG.items():
        got = getattr(args, key)
        expect = tuple(value) if isinstance(value, list) else value
        assert got == expect, key
    assert args.max_seq_len == 8192  # override wins over the file
    assert isinstance(args.compress_ratios, tuple)  # JSON lists tuple-ized
    # Reference defaults absent from the JSON, and the format constants.
    assert args.gate_temp == 1.0 and args.norm_topk_prob is True
    assert DeepseekV41Args.fp8_block_size == 32
    assert DeepseekV41Args.fp4_block_size == 32
    assert DeepseekV41Args.cmp_fp4_group == 16


def test_unknown_key_raises_and_names_it(tmp_path):
    raw = dict(V41_CONFIG, brand_new_knob=3, another_one=[1])
    with pytest.raises(ValueError, match=r"another_one.*brand_new_knob"):
        load_args(_write(tmp_path, raw))


def test_strict_false_downgrades_unknown_key_to_warning(tmp_path, caplog):
    raw = dict(V41_CONFIG, brand_new_knob=3)
    args = load_args(_write(tmp_path, raw), strict=False)
    assert args.dim == 5120
    assert any("brand_new_knob" in r.getMessage() for r in caplog.records)


def test_ignore_set_warns_but_loads(caplog):
    raw = dict(
        V41_CONFIG, vision_n_layers=27, vision_dim=1152, image_token_id=129264, temperature=1.0
    )
    args = parse_raw_args(raw)
    assert args.n_layers == 40
    assert not hasattr(args, "vision_n_layers")
    msgs = " ".join(r.getMessage() for r in caplog.records)
    for key in ("vision_n_layers", "vision_dim", "image_token_id", "temperature"):
        assert key in msgs


@pytest.mark.parametrize(
    "override, pattern",
    [
        # a kv source must also be an index source
        ({"kv_source_layers": [2, 8, 14, 20, 30]}, "subset of index_source_layers"),
        # a compressed layer whose owner has a different ratio
        ({"compress_ratios": [0, 0] + [2] * 18 + [1, 2] + [1] * 18 + [0, 0, 0]}, "match its owner"),
        # a compressed layer before any kv source
        ({"compress_ratios": [2, 0] + [2] * 18 + [1] * 20 + [0, 0, 0]}, "no kv_source layer"),
        # a source layer with nothing to compress
        ({"compress_ratios": [0, 0, 0] + [2] * 17 + [1] * 20 + [0, 0, 0]}, "kv/index source"),
        # the candidate pool must be picked at an index source
        ({"candidate_source_layer": 21}, "candidate_source_layer 21"),
        ({"candidate_topk_blocks": 0}, "candidate_topk_blocks"),
        # ratio must divide the window/page
        ({"window_size": 129}, "multiple of"),
        ({"dspark_target_layer_ids": [37, 40]}, "dspark_target_layer_ids"),
        ({"engram_layer_ids": [1, 40]}, "engram_layer_ids"),
        ({"engram_num_embeddings": [1]}, "engram_num_embeddings"),
        ({"compress_ratios": [0] * 39}, "compress_ratios has 39"),
        ({"dspark_n_activated_experts": 129}, "dspark_n_activated_experts"),
        ({"kv_source_layers": [2, 8, 14, 40]}, "kv_source_layers entries"),
    ],
)
def test_validation_failures(override, pattern):
    with pytest.raises(ValueError, match=pattern):
        parse_raw_args(dict(V41_CONFIG, **override))


def test_layer_roles_on_the_released_topology():
    args = parse_raw_args(V41_CONFIG)
    # Modes per the tech report: window 0/1, full at the four kv sources, reindex at the
    # decoder's extra index sources, reuse everywhere else.
    modes = {L: args.mode(L) for L in range(40)}
    assert modes[0] == modes[1] == "window"
    assert all(modes[L] == "full" for L in (2, 8, 14, 20))
    assert all(modes[L] == "reindex" for L in (24, 28, 32, 36))
    reuse = [L for L in range(40) if L not in (0, 1, 2, 8, 14, 20, 24, 28, 32, 36)]
    assert all(modes[L] == "reuse" for L in reuse)
    # Compressed-KV ownership: the latest kv source at or before the layer.
    assert args.cmp_owner(0) is None and args.cmp_owner(1) is None
    assert [args.cmp_owner(L) for L in (2, 3, 7, 8, 9, 13, 14, 15, 19)] == [2, 2, 2, 8, 8, 8, 14, 14, 14]
    assert all(args.cmp_owner(L) == 20 for L in range(20, 40))
    # Indexer ownership: the latest index source at or before the layer.
    assert [args.idx_owner(L) for L in (20, 23, 24, 27, 28, 36, 39)] == [20, 20, 24, 24, 28, 36, 36]
    assert args.is_kv_source(8) and not args.is_kv_source(24)
    assert args.is_index_source(24) and not args.is_index_source(25)
    # The candidate pool: picked at 20, used by the reindex layers after it only.
    assert args.decoder_start() == 20
    assert [L for L in range(40) if args.uses_candidates(L)] == [24, 28, 32, 36]
    # MoE geometry: the tower's 384/6 vs the 128/3 DSpark stages (ids n_layers..).
    assert args.get_moe_config(0) == (384, 6) and args.get_moe_config(39) == (384, 6)
    assert args.get_moe_config(40) == (128, 3) and args.get_moe_config(42) == (128, 3)
    assert args.dspark_enabled and args.has_engram
    assert args.nope_head_dim == 448
    assert args.n_window_layers == 40  # stages join only via extend_config_for_speculative


def test_decoder_start_requires_a_single_decoder_ratio():
    raw = dict(V41_CONFIG)
    raw["compress_ratios"] = [0, 0] + [2] * 18 + [1] * 20 + [0, 0, 0]
    raw["kv_source_layers"] = [2, 8, 14, 20, 30]
    raw["index_source_layers"] = [2, 8, 14, 20, 24, 28, 30, 32, 36]
    args = parse_raw_args(raw)  # valid topology, but two decoder kv sources
    with pytest.raises(ValueError, match="candidate_source_layer 20 must be the decoder start 30"):
        args.decoder_start()


def test_config_path_fallback_and_missing(tmp_path):
    with open(tmp_path / "model_args.json", "w") as f:
        json.dump(V41_CONFIG, f)
    assert load_args(str(tmp_path)).dim == 5120
    with pytest.raises(FileNotFoundError):
        load_args(str(tmp_path / "nowhere"))
