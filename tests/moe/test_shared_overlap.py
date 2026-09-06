"""Shared-expert scheduling must preserve its input and leave unsupported paths eager."""
from types import SimpleNamespace

import pytest
import torch

import freetoken.layers.moe as moe_mod
from freetoken.kernel.triton.moe_finalize import MoeCombineEpilogue
from freetoken.layers.moe import OffloadMoELayer


def test_shared_materializes_once():
    shared = torch.ones(1, 2)
    calls = []

    def produce():
        calls.append("shared")
        return shared

    ep = MoeCombineEpilogue(None, None, shared, shared, 1, shared_fn=produce)
    assert ep.has_shared
    assert ep.materialize_shared() is shared
    assert ep.materialize_shared() is shared
    assert calls == ["shared"] and ep.shared_fn is None


@pytest.mark.parametrize("unsupported", [None, "disabled", "serial", "prefill", "tp", "cpu", "gpu", "format"])
def test_overlap_eligibility_uses_only_runtime_configuration(unsupported, monkeypatch):
    monkeypatch.setattr(moe_mod, "_HYBRID_SHARED_OVERLAP", unsupported != "disabled")
    monkeypatch.setattr(moe_mod, "_HYBRID_OVERLAP", unsupported != "serial")
    monkeypatch.setattr(moe_mod, "get_global_ctx", lambda: SimpleNamespace(
        batch=SimpleNamespace(is_prefill=unsupported == "prefill")))
    layer = OffloadMoELayer.__new__(OffloadMoELayer)
    layer.layer_id = 0
    layer.tp_size = 2 if unsupported == "tp" else 1
    layer.offload_cache = SimpleNamespace(
        decode_target="gpu" if unsupported == "gpu" else "hybrid",
        quant_format="fp8_block" if unsupported == "format" else "nvfp4",
        is_cpu_layer=lambda _: unsupported == "cpu",
    )
    assert layer.can_overlap_shared() == (unsupported is None)


def test_shared_runs_after_submit_before_expert_input_is_overwritten(monkeypatch):
    monkeypatch.setattr(moe_mod, "_HYBRID_OVERLAP", True)
    events = []
    hidden = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    original = hidden.clone()
    weights = torch.ones(2, 2)
    ids = torch.zeros(2, 2, dtype=torch.int32)
    gate_weight = torch.ones(4)

    def submit(layer_id, x, w, cpu_ids, *, gate, routing_pack):
        events.append("submit")
        assert torch.equal(x, original)
        gate[1].copy_(torch.sigmoid(x.sum(-1)))
        return "pending"

    def sync(pending):
        assert pending == "pending"
        events.append("sync")
        return torch.zeros_like(hidden)

    def shared():
        events.append("shared")
        assert torch.equal(hidden, original)
        return hidden.square()

    cache = SimpleNamespace(
        cpu_executor=SimpleNamespace(can_fuse_gate=lambda: True, decode_submit=submit, decode_sync=sync),
        quant_format="nvfp4",
        ensure_experts_hybrid=lambda *args, **kwargs: ids,
        copy_missing=lambda: events.append("copy"),
        bank_views=lambda: (), alphas_for_slots=lambda _: None,
    )
    ep = MoeCombineEpilogue(None, None, hidden, weights, 1,
                            gate_weight=gate_weight, shared_fn=shared)
    layer = OffloadMoELayer.__new__(OffloadMoELayer)
    layer.layer_id = 0

    def experts(*args, finalize, epilogue, **kwargs):
        events.append("experts")
        hidden.fill_(-999)
        assert epilogue.shared_fn is None
        assert torch.equal(epilogue.shared, original.square())
        finalize()
        return epilogue.shared

    layer._expert_gemm = experts
    out = layer._decode_hybrid(cache, hidden, weights, ids, ep)
    assert events == ["submit", "shared", "copy", "experts", "sync"]
    assert torch.equal(out, original.square())
    torch.testing.assert_close(ep.gate, torch.sigmoid(original.sum(-1)))


@pytest.mark.parametrize("prefill", [False, True])
def test_unsupported_forward_materializes_before_mutating_fallback(prefill, monkeypatch):
    monkeypatch.setattr(moe_mod, "get_global_ctx", lambda: SimpleNamespace(
        batch=SimpleNamespace(is_prefill=prefill)))
    hidden = torch.arange(4, dtype=torch.float32).reshape(1, 4)
    expected = hidden.square()
    events = []
    ep = MoeCombineEpilogue(None, torch.ones(1), hidden, torch.ones(1, 1), 1,
                            shared_fn=lambda: (events.append("shared"), hidden.square())[1])
    layer = OffloadMoELayer.__new__(OffloadMoELayer)
    layer.tp_size = 1
    layer.can_overlap_shared = lambda: False
    layer._maybe_all_reduce = lambda x: x

    def fallback(x, *args):
        events.append("fallback")
        x.fill_(999)
        return x

    layer.prefill_forward = layer.decode_forward = fallback
    layer.forward(hidden, epilogue=ep)
    assert events == ["shared", "fallback"]
    assert ep.result is None and torch.equal(ep.shared, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("defer,fused", [(False, True), (True, True), (True, False)])
def test_qwen_shared_scheduling_preserves_fallback_input(defer, fused, monkeypatch):
    import freetoken.models.qwen4_exp.moe as qwen_moe
    from freetoken.kernel.triton.moe_shared_gate import shared_gate_sigmoid

    monkeypatch.setattr(qwen_moe, "_FUSED_FINALIZE", fused)
    hidden = torch.arange(16, device="cuda", dtype=torch.bfloat16).reshape(2, 8) / 16
    original = hidden.clone()
    residual = torch.zeros(2, 8, device="cuda", dtype=torch.bfloat16)
    inject = torch.ones(2, 1, device="cuda", dtype=torch.bfloat16)
    gate_weight = torch.ones(1, 8, device="cuda", dtype=torch.bfloat16)
    events = []

    def shared(x):
        events.append("shared")
        return x.square()

    def experts(*, hidden_states, router_logits, epilogue):
        events.append("submit")
        if epilogue is not None:
            epilogue.gate = shared_gate_sigmoid(hidden_states, gate_weight.view(-1))
            epilogue.materialize_shared()
        # An unsupported expert backend can overwrite its input and return routed
        # output without finishing the epilogue. The model must use the original shared.
        hidden_states.fill_(999)
        return torch.zeros_like(hidden_states)

    block = SimpleNamespace(
        gate=SimpleNamespace(forward=lambda x: x),
        shared_expert=SimpleNamespace(forward=shared),
        shared_expert_gate=SimpleNamespace(weight=gate_weight),
        experts=SimpleNamespace(can_overlap_shared=lambda: defer, forward=experts),
    )
    hc = SimpleNamespace(hc_count=1, combine=lambda r, y, inj: r + y)
    out = qwen_moe.Qwen4ExpMoE.forward_combine(block, hidden, residual, inject, hc)
    expected = (original.square().float() * torch.sigmoid(original.float().sum(-1))[:, None]).to(out.dtype)
    torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)
    assert events == (["submit", "shared"] if defer and fused else ["shared", "submit"])
