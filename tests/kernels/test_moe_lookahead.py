"""Fused lookahead router kernels vs the GatedResidual + gate torch reference."""

from types import SimpleNamespace

import pytest
import torch

from freetoken.distributed import set_tp_info, try_get_tp_info

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _setup():
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    from freetoken.models.qwen4_exp.hc import GatedResidual

    torch.manual_seed(0)
    dev, dt = torch.device("cuda"), torch.bfloat16
    cfg = SimpleNamespace(
        rms_norm_eps=1e-6,
        qwen4_args=SimpleNamespace(
            hc_count=4, hidden_size=2560, hc_lowrank=320, ple_state_width=10240
        ),
    )
    with torch.device(dev):
        gr = GatedResidual(cfg)
    gr.hc_norm.weight = (torch.randn(10240, device=dev) * 0.05).to(dt)
    gr.input_mix_weight_down_block_inject.weight = (
        torch.randn(336, 10240, device=dev) * 0.02
    ).to(dt)
    gr.input_mix_weight_up.weight = (torch.randn(10240, 320, device=dev) * 0.02).to(dt)
    gate_w = (torch.randn(512, 2560, device=dev) * 0.02).to(dt)
    x = torch.randn(3, 10240, device=dev).to(dt)
    with torch.no_grad():
        ref_logits = (gr.mix(x)[0] @ gate_w.t()).float()
    ref_top = torch.topk(ref_logits, 4, dim=-1).indices
    return gr, gate_w, x, ref_logits, ref_top


def _check(cand, logits, ref_logits, ref_top, base):
    rel = (logits - ref_logits).abs().max() / ref_logits.abs().max()
    assert rel < 0.02, f"logits diverge from the torch chain: rel={rel}"
    for r in range(3):
        got = set((cand.view(3, 4)[r] - base).tolist())
        assert got == set(ref_top[r].tolist()), (r, got, ref_top[r])


def test_mono_kernel_matches_torch_chain():
    from freetoken.kernel.triton.moe_lookahead import lookahead_route

    gr, gate_w, x, ref_logits, ref_top = _setup()
    logits = torch.zeros(3, 512, device="cuda", dtype=torch.float32)
    cand = lookahead_route(
        x, gr.hc_norm.weight, gr.input_mix_weight_down_block_inject.weight,
        gr.input_mix_weight_up.weight, gate_w,
        hc=4, eps=1e-6, topk=4, flat_base=9 * 512, logits_out=logits,
    )
    torch.cuda.synchronize()
    _check(cand, logits, ref_logits, ref_top, 9 * 512)


def test_split_pipeline_matches_torch_chain():
    from freetoken.kernel.triton.moe_lookahead import (
        LookaheadScratch,
        lookahead_route_split,
    )

    gr, gate_w, x, ref_logits, ref_top = _setup()
    scratch = LookaheadScratch(torch.device("cuda"), hc=4)
    cand = torch.zeros(12, dtype=torch.int32, device="cuda")
    lookahead_route_split(
        x, gr.hc_norm.weight, gr.input_mix_weight_down_block_inject.weight,
        gr.input_mix_weight_up.weight, gate_w, scratch, cand,
        hc=4, eps=1e-6, topk=4, flat_base=9 * 512,
    )
    torch.cuda.synchronize()
    _check(cand, scratch.logits[:3], ref_logits, ref_top, 9 * 512)
    # scratch is reused across hooks: a second call must not leak the first's sums
    lookahead_route_split(
        x, gr.hc_norm.weight, gr.input_mix_weight_down_block_inject.weight,
        gr.input_mix_weight_up.weight, gate_w, scratch, cand,
        hc=4, eps=1e-6, topk=4, flat_base=9 * 512,
    )
    torch.cuda.synchronize()
    _check(cand, scratch.logits[:3], ref_logits, ref_top, 9 * 512)
