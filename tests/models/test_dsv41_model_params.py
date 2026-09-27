"""DeepSeek-V4.1 model construction on the toy geometry (meta device, CPU) and the contract
between the model's parameter names and the weight loader's plan.

The loader's ``resident_tensor_plan`` decides, from the checkpoint INDEX alone, which tensor
lands under which parameter name; the model's ``named_parameters`` is what ``load_state_dict``
demands (strict both ways). This test synthesizes the index a V4.1 checkpoint would carry for
the toy geometry and pins that the two sets are identical, so a renamed module or a plan
omission fails here rather than at the first real boot.
"""

from __future__ import annotations

import os

import pytest
import torch

from freetoken.models.deepseek_v41.args import DeepseekV41Args
from freetoken.models.deepseek_v41.weight import resident_tensor_plan

TOY = dict(
    dim=256, q_lora_rank=128, n_heads=2, head_dim=128, rope_head_dim=32, o_groups=2,
    o_lora_rank=64, moe_inter_dim=128, n_routed_experts=8, n_activated_experts=2,
    dspark_n_routed_experts=4, dspark_n_activated_experts=1, index_n_heads=2,
    index_head_dim=64, index_topk=16, n_layers=8,
    compress_ratios=[0, 0, 2, 2, 1, 1, 1, 1, 0, 0, 0], kv_source_layers=[2, 4],
    index_source_layers=[2, 4, 6], candidate_source_layer=4, candidate_topk_blocks=4,
    candidate_block_size=8, window_size=128, engram_layer_ids=[1],
    engram_num_embeddings=[1009], engram_vocab_size=101, engram_n_heads=2,
    engram_max_ngram_size=4, engram_head_dim=64, engram_compressed_vocab_size=50,
    n_mtp_layers=3, dspark_block_size=5, dspark_target_layer_ids=[5, 6, 7], vocab_size=64,
    max_seq_len=512,
)


@pytest.fixture(autouse=True)
def _runtime(monkeypatch):
    import freetoken.core as core
    from freetoken.distributed import set_tp_info, try_get_tp_info

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    monkeypatch.setenv("FREETOKEN_LOAD_MTP", "1")
    core._GLOBAL_CTX = None
    yield
    core._GLOBAL_CTX = None


def _synthetic_index(args: DeepseekV41Args, names: set[str]) -> list[str]:
    """The checkpoint index a V4.1 export of this geometry carries: the model's resident
    tensors under their checkpoint names, the routed experts, and the documented skips."""
    idx = []
    for n in names:
        if n == "head":
            idx.append("head.weight")
        elif n.endswith(".attn.wo_a"):
            idx += [n + ".weight", n + ".scale"]
        else:
            idx.append(n)
    for L in range(args.n_layers):
        for e in range(args.n_routed_experts):
            for p in ("w1", "w2", "w3"):
                idx += [f"layers.{L}.ffn.experts.{e}.{p}.weight", f"layers.{L}.ffn.experts.{e}.{p}.scale"]
        idx.append(f"layers.{L}.ffn.gate.bias_vl")
    for s in range(args.n_mtp_layers):
        for e in range(args.dspark_n_routed_experts):
            for p in ("w1", "w2", "w3"):
                idx += [f"mtp.{s}.ffn.experts.{e}.{p}.weight", f"mtp.{s}.ffn.experts.{e}.{p}.scale"]
        idx.append(f"mtp.{s}.ffn.gate.bias_vl")
    for L in args.engram_layer_ids:
        idx += [f"layers.{L}.engram.embed.weight", f"layers.{L}.engram.embed.scale"]
    idx += ["vision.blocks.0.attn.wo.weight", "aligner.w1.weight", "image_start"]
    return idx


def _build(prefill_mode: str):
    from freetoken.models.deepseek_v41.model import Transformer

    args = DeepseekV41Args(**TOY)
    with torch.device("meta"):
        return args, Transformer(args, prefill_mode=prefill_mode)


@pytest.mark.parametrize("prefill_mode", ["ced", "full"])
def test_parameter_names_match_the_loader_plan(prefill_mode):
    args, t = _build(prefill_mode)
    names = {n for n, _ in t.named_parameters()}
    plan = resident_tensor_plan(_synthetic_index(args, names), args, dspark=True)
    yielded = {p for p, _src, kind in plan.yielded if kind != "expert"}
    assert plan.leftover == []
    assert names == yielded, (sorted(names - yielded), sorted(yielded - names))


def test_layer_roles_follow_the_args():
    args, t = _build("ced")
    modes = [t.layers[L].attn.mode for L in range(args.n_layers)]
    assert modes == ["window", "window", "full", "reuse", "full", "reuse", "reindex", "reuse"]
    assert t.decoder_start == 4
    assert t.layers[2].attn.compressor is not None and t.layers[2].attn.compressor.gated
    assert t.layers[4].attn.compressor is not None and not t.layers[4].attn.compressor.gated
    assert t.layers[3].attn.compressor is None and t.layers[3].attn.indexer is None
    assert t.layers[6].attn.indexer is not None and t.layers[6].attn.indexer.uses_candidates
    assert t.layers[4].attn.indexer.is_candidate_source
    assert getattr(t.layers[1], "engram", None) is not None
    assert getattr(t.layers[0], "engram", None) is None
    # DSpark stages: window-only, the stage MoE geometry, heads on the last stage only
    assert len(t.mtp) == 3
    assert t.mtp[0].ffn.experts.num_experts == 4 and t.mtp[0].ffn.experts.top_k == 1
    assert t.mtp[0].is_first and hasattr(t.mtp[0], "main_proj")
    assert t.mtp[2].is_last and hasattr(t.mtp[2], "markov_head") and not hasattr(t.mtp[1], "markov_head")
    assert not any(n.startswith("hc_head") for n, _ in t.named_parameters())


def test_ced_prefill_rejects_engram_in_the_decoder():
    from freetoken.models.deepseek_v41.model import Transformer

    args = DeepseekV41Args(**{**TOY, "engram_layer_ids": [5], "engram_num_embeddings": [1009]})
    with pytest.raises(AssertionError):
        with torch.device("meta"):
            Transformer(args, prefill_mode="ced")


def test_prefill_mode_env(monkeypatch):
    from freetoken.models.deepseek_v41.model import PREFILL_MODE_ENV, prefill_mode_from_env

    monkeypatch.delenv(PREFILL_MODE_ENV, raising=False)
    assert prefill_mode_from_env() == "ced"
    monkeypatch.setenv(PREFILL_MODE_ENV, "FULL")
    assert prefill_mode_from_env() == "full"
    monkeypatch.setenv(PREFILL_MODE_ENV, "nope")
    with pytest.raises(ValueError):
        prefill_mode_from_env()


def test_spec_scratch_carries_only_gated_kv_sources():
    from freetoken.models.deepseek_v41.spec_state import DSV41SpecScratch

    args = DeepseekV41Args(**TOY)
    s = DSV41SpecScratch(args, num_draft_tokens=3, vocab_size=64, device=torch.device("cpu"))
    assert set(s.carry) == {(2, "attn")}  # layer 4 is ratio 1: stateless
    assert s.carry[(2, "attn")].shape == (4, 2, 2 * args.head_dim)
    assert len(s.block_kv) == 3 and s.block_kv[0].shape == (5, args.head_dim)
    s.set_step(cached_len=126, num_input_tokens=4)
    assert s.boundary_row == 2  # position 128 is row 2 of 126..129
