"""--dense-quant fp8 (load-time per-channel FP8 of qwen4_exp's bf16 dense linears).

Loader contract (CPU): under FREETOKEN_DENSE_QUANT=fp8 ``iter_weights`` emits fp8 codes plus a
``.weight_scale`` sibling for exactly the selected parts, splits the GDN ``in_proj`` into the
fp8 ``in_proj_qkvz`` + bf16 ``in_proj_ba`` the layer builds, and leaves everything else
byte-identical to the default mode. Module parity (GPU): GDN / hyper-connection / shared
expert built in fp8 mode on their quantized state dicts agree with the bf16 modules.
"""

from __future__ import annotations

import pytest
import torch
from safetensors.torch import save_file

from freetoken.distributed import set_tp_info, try_get_tp_info
from freetoken.kernel.triton.fp8_pertensor_linear import FP8, quantize_fp8_per_row
from freetoken.models.qwen4_exp.weight import iter_weights

from .test_weight import (
    HC,
    LR,
    _expected_names,
    _ngram_table,
    _raw_checkpoint,
)


@pytest.fixture(scope="session", autouse=True)
def _tp_info():
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory) -> tuple[str, dict[str, torch.Tensor]]:
    torch.manual_seed(1)
    folder = tmp_path_factory.mktemp("qwen4_exp_ckpt_fp8")
    raw = _raw_checkpoint()
    table, _ = _ngram_table()
    names = sorted(raw)
    save_file({n: raw[n] for n in names[::2]}, str(folder / "model-bf16-00001.safetensors"))
    save_file({n: raw[n] for n in names[1::2]}, str(folder / "model-bf16-00002.safetensors"))
    shard_names = sorted(table)
    save_file({n: table[n] for n in shard_names[:2]}, str(folder / "model-plefp8-00000.safetensors"))
    save_file({n: table[n] for n in shard_names[2:]}, str(folder / "model-plefp8-00001.safetensors"))
    return str(folder), {**raw, **table}


def _load(folder: str) -> dict[str, torch.Tensor]:
    return {
        name: tensor.clone()
        for name, tensor in iter_weights(
            folder, torch.device("cpu"), include_moe_experts=True, include_non_moe=True
        )
    }


def _dequant(q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    return q.to(torch.float32) * s[:, None]


_FP8_TARGETS = {
    "lm_head.weight",
    "model.hyper_connection_mixer.input_mix_weight_down.weight",
    "model.hyper_connection_mixer.input_mix_weight_up.weight",
    "model.layers.0.linear_attn.in_proj_qkvz.weight",
    "model.layers.0.linear_attn.out_proj.weight",
    "model.layers.1.self_attn.qkv_proj.weight",
    "model.layers.1.self_attn.o_proj.weight",
} | {
    f"model.layers.{layer}.{hc}.{leaf}"
    for layer in (0, 1)
    for hc in ("attn_hyper_connection", "mlp_hyper_connection")
    for leaf in ("input_mix_weight_down_block_inject.weight", "input_mix_weight_up.weight")
} | {
    f"model.layers.{layer}.mlp.shared_expert.{leaf}"
    for layer in (0, 1)
    for leaf in ("gate_up_proj.weight", "down_proj.weight")
}


def test_default_mode_is_unchanged(checkpoint, monkeypatch):
    monkeypatch.delenv("FREETOKEN_DENSE_QUANT", raising=False)
    loaded = _load(checkpoint[0])
    assert set(loaded) == _expected_names()
    assert all(not n.endswith(".weight_scale") for n in loaded)


def test_fp8_mode_emits_codes_and_scales_for_the_selected_parts(checkpoint, monkeypatch):
    folder, raw = checkpoint
    monkeypatch.delenv("FREETOKEN_DENSE_QUANT", raising=False)
    base = _load(folder)
    monkeypatch.setenv("FREETOKEN_DENSE_QUANT", "fp8")
    monkeypatch.delenv("FREETOKEN_DENSE_FP8_PARTS", raising=False)
    fp8 = _load(folder)

    expected = (_expected_names() - {"model.layers.0.linear_attn.in_proj.weight"}) | {
        "model.layers.0.linear_attn.in_proj_qkvz.weight",
        "model.layers.0.linear_attn.in_proj_ba.weight",
    }
    expected |= {n[: -len(".weight")] + ".weight_scale" for n in _FP8_TARGETS}
    assert set(fp8) == expected

    for name in _FP8_TARGETS:
        q, s = fp8[name], fp8[name[: -len(".weight")] + ".weight_scale"]
        assert q.dtype == FP8 and s.dtype == torch.float32 and s.shape == (q.shape[0],), name
        ref = base.get(name)
        if ref is None:  # the split GDN projection: rebuild the bf16 reference from the parts
            lm = "model.language_model.layers.0.linear_attn"
            ref = torch.cat([raw[f"{lm}.in_proj_qkv.weight"], raw[f"{lm}.in_proj_z.weight"]], 0)
        assert q.shape == ref.shape, name
        err = (_dequant(q, s) - ref.float()).abs()
        amax = ref.float().abs().amax(dim=1, keepdim=True)
        assert (err <= ref.float().abs() / 16 + amax * 2 ** -9).all(), name

    # the bf16 remainder of the GDN projection is the raw b|a concat, bitwise
    lm = "model.language_model.layers.0.linear_attn"
    ba = torch.cat([raw[f"{lm}.in_proj_b.weight"], raw[f"{lm}.in_proj_a.weight"]], 0)
    assert torch.equal(fp8["model.layers.0.linear_attn.in_proj_ba.weight"], ba)
    # HC merged GEMM: the zero pad rows quantize to zero codes with a finite scale
    q = fp8["model.layers.0.attn_hyper_connection.input_mix_weight_down_block_inject.weight"]
    s = fp8["model.layers.0.attn_hyper_connection.input_mix_weight_down_block_inject.weight_scale"]
    pad = q.shape[0] - (LR + HC)
    assert pad == 12
    assert torch.equal(q[-pad:].to(torch.float32), torch.zeros(pad, q.shape[1]))
    assert torch.isfinite(s).all()
    # everything not selected is byte-identical to the default mode
    for name, t in base.items():
        if name in _FP8_TARGETS or name == "model.layers.0.linear_attn.in_proj.weight":
            continue
        assert torch.equal(fp8[name], t), name
    for name in ("model.layers.1.self_attn.indexer.index_qk_proj.weight",
                 "model.layers.0.mlp.gate.weight", "model.layers.0.ple.key_proj.weight"):
        assert fp8[name].dtype == torch.bfloat16


def test_fp8_parts_subset_only_touches_its_parts(checkpoint, monkeypatch):
    folder, _ = checkpoint
    monkeypatch.setenv("FREETOKEN_DENSE_QUANT", "fp8")
    monkeypatch.setenv("FREETOKEN_DENSE_FP8_PARTS", "lm_head,shared")
    fp8 = _load(folder)
    scaled = {n for n in fp8 if n.endswith(".weight_scale")}
    assert scaled == {"lm_head.weight_scale"} | {
        f"model.layers.{layer}.mlp.shared_expert.{leaf}.weight_scale"
        for layer in (0, 1) for leaf in ("gate_up_proj", "down_proj")
    }
    assert "model.layers.0.linear_attn.in_proj.weight" in fp8  # GDN untouched -> fused bf16


# ---------------------------------------------------------------------------------------
# module parity (GPU)
# ---------------------------------------------------------------------------------------
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
DEV = torch.device("cuda")


def _q(sd: dict[str, torch.Tensor], names: set[str]) -> dict[str, torch.Tensor]:
    out = {}
    for k, v in sd.items():
        if k in names:
            q, s = quantize_fp8_per_row(v)
            out[k] = q
            out[k[: -len(".weight")] + ".weight_scale"] = s
        else:
            out[k] = v
    return out


def _dq(sd: dict[str, torch.Tensor], names: set[str]) -> dict[str, torch.Tensor]:
    """The bf16 oracle's weights: fp8-quantize then dequantize the selected tensors, so the
    only differences left against the fp8 module are kernel numerics + activation quant."""
    out = {}
    for k, v in sd.items():
        if k in names:
            q, s = quantize_fp8_per_row(v)
            out[k] = _dequant(q, s).to(v.dtype)
        else:
            out[k] = v
    return out


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-6)).item()


@cuda
@pytest.mark.parametrize("tokens", [1, 3, 64])
def test_hc_and_shared_expert_fp8_match_bf16(tokens, monkeypatch):
    from dataclasses import replace

    from freetoken.models.qwen3_5_moe.moe import _SharedExpert
    from freetoken.models.qwen4_exp.config import parse_config
    from freetoken.models.qwen4_exp.hc import GatedResidual
    from freetoken.utils import torch_dtype

    from .common import toy_hf_config

    monkeypatch.delenv("FREETOKEN_DENSE_QUANT", raising=False)
    cfg = parse_config(toy_hf_config(4))
    monkeypatch.setenv("FREETOKEN_DENSE_QUANT", "fp8")
    cfg8 = parse_config(toy_hf_config(4))
    assert cfg8.dense_quant == "fp8_pertensor"
    gen = torch.Generator(device="cpu").manual_seed(0)

    for use_combine in (True, False):
        with torch.device("meta"), torch_dtype(torch.bfloat16):
            ref = GatedResidual(cfg, use_combine=use_combine)
            fp8 = GatedResidual(cfg8, use_combine=use_combine)
        sd = {}
        for k, v in ref.state_dict().items():
            t = torch.randn(v.shape, generator=gen) * 0.05
            sd[k] = t.to(DEV, torch.bfloat16)
        ref.load_state_dict({k: v.clone() for k, v in sd.items()})
        fp8_names = {k for k in sd if "input_mix_weight" in k}
        fp8.load_state_dict(_q(sd, fp8_names))
        R = (torch.randn(tokens, cfg.qwen4_args.ple_state_width, generator=gen) * 0.5).to(DEV, torch.bfloat16)
        x_ref, s_ref = ref.mix(R)
        x_fp8, s_fp8 = fp8.mix(R)
        assert _rel(x_fp8, x_ref) < 2e-2
        if s_ref is not None:
            # the 4-wide inject logits: fp8 rounding shows more in a tiny output; they only
            # feed 2*sigmoid(s/hc) downstream
            assert _rel(s_fp8, s_ref) < 5e-2

    with torch.device("meta"), torch_dtype(torch.bfloat16):
        se_ref = _SharedExpert(replace(cfg, expert_quant="none"), cfg.hidden_size, cfg.shared_expert_intermediate_size)
        se_dq = _SharedExpert(replace(cfg, expert_quant="none"), cfg.hidden_size, cfg.shared_expert_intermediate_size)
        se_fp8 = _SharedExpert(replace(cfg, expert_quant="none"), cfg.hidden_size, cfg.shared_expert_intermediate_size, fp8_pertensor=True)
    sd = {k: (torch.randn(v.shape, generator=gen) * 0.05).to(DEV, torch.bfloat16) for k, v in se_ref.state_dict().items()}
    se_ref.load_state_dict({k: v.clone() for k, v in sd.items()})
    se_dq.load_state_dict(_dq(sd, set(sd)))
    se_fp8.load_state_dict(_q(sd, set(sd)))
    x = (torch.randn(tokens, cfg.hidden_size, generator=gen)).to(DEV, torch.bfloat16)
    y8, ydq, yref = se_fp8.forward(x), se_dq.forward(x), se_ref.forward(x)
    # Against the dequantized oracle only kernel + activation error remains. M=1 runs the
    # exact-activation W8A16 GEMV (~1e-3); M>1 runs per-token dynamic W8A8 (the production
    # scheme, also the 27B deployment's): e4m3 activations cost ~2.5% per GEMM, ~4% through
    # gate_up -> silu*up -> down. The kernel-level test holds the W8A8 path to its own
    # per-token-quantized reference at 1e-2.
    # (M <= 4 now runs the exact-activation rows GEMV as well; the 6e-2 budget only applies
    # once a batch falls to the per-token W8A8 tile GEMM)
    assert _rel(y8, ydq) < (1.5e-2 if tokens <= 4 else 6e-2)
    # weight rounding vs the raw bf16 module: e4m3 (3 mantissa bits) on random N(0, .05)
    # weights compounds over gate_up -> silu*up -> down; the model-level gates judge quality
    assert _rel(y8, yref) < 0.12


@cuda
@pytest.mark.parametrize("tokens", [1, 3])
def test_gdn_fp8_projections_match_bf16(tokens):
    """The fp8 GDN layout (fp8 in_proj_qkvz + bf16 in_proj_ba + fp8 out_proj) on the same
    weights as the bf16 layer: a decode step from a fresh state agrees within fp8 tolerance."""
    from freetoken.models.qwen4_exp.gdn import Qwen4ExpGatedDeltaNet
    from freetoken.utils import torch_dtype

    from .test_gdn import CONV_K, EPS, HEAD_DIM, HEADS, HIDDEN, _ctx, _prefill, _state_dict, _make_layer

    op_ref, ref = _make_layer(3, seed=3)
    sd = _state_dict(ref)
    # oracle: the bf16 op on the DEQUANTIZED projections (kernel/activation error only)
    sd_dq = _dq(sd, {"in_proj.weight", "out_proj.weight"})
    num_k, num_v = HEADS[3]
    # conv_dim covers q|k|v (the value width is convolved too): qkvz = conv_dim + value_dim rows
    value_dim = num_v * HEAD_DIM
    conv_dim = 2 * num_k * HEAD_DIM + value_dim
    in_proj = sd.pop("in_proj.weight")
    qkvz, ba = in_proj[: conv_dim + value_dim], in_proj[conv_dim + value_dim :]
    q1, s1 = quantize_fp8_per_row(qkvz)
    q2, s2 = quantize_fp8_per_row(sd.pop("out_proj.weight"))
    sd8 = {**sd, "in_proj_qkvz.weight": q1, "in_proj_qkvz.weight_scale": s1,
           "in_proj_ba.weight": ba, "out_proj.weight": q2, "out_proj.weight_scale": s2}
    # the oracle keeps b|a exact, like the fp8 layout does
    in_dq = sd_dq["in_proj.weight"].clone()
    in_dq[conv_dim + value_dim :] = ba
    sd_dq["in_proj.weight"] = in_dq
    with torch.device("meta"), torch_dtype(torch.bfloat16):
        op_dq = Qwen4ExpGatedDeltaNet(
            hidden_size=HIDDEN, num_k_heads=num_k, num_v_heads=num_v, head_k_dim=HEAD_DIM,
            head_v_dim=HEAD_DIM, conv_kernel_size=CONV_K, rms_norm_eps=EPS, layer_id=0,
            output_gate="sigmoid",
        )
    op_dq.load_state_dict(sd_dq)
    with torch.device("meta"), torch_dtype(torch.bfloat16):
        op8 = Qwen4ExpGatedDeltaNet(
            hidden_size=HIDDEN, num_k_heads=num_k, num_v_heads=num_v, head_k_dim=HEAD_DIM,
            head_v_dim=HEAD_DIM, conv_kernel_size=CONV_K, rms_norm_eps=EPS, layer_id=0,
            output_gate="sigmoid", attn_quant="fp8_pertensor",
        )
    op8.load_state_dict(sd8)
    # one short prefill through both ops (fresh slots), compare the packed outputs
    ctx = _ctx(3)
    _, _, out_ref = _prefill(op_ref, ctx, [tokens, 5], seed=5)
    ctx = _ctx(3)
    _, _, out_dq = _prefill(op_dq, ctx, [tokens, 5], seed=5)
    ctx = _ctx(3)
    _, _, out_8 = _prefill(op8, ctx, [tokens, 5], seed=5)
    # prefill rows (M >= 6) run per-token W8A8 on qkv|z and out_proj: activation quant
    # (~2.5%/GEMM) through the conv + recurrence + gated norm chain
    assert _rel(out_8, out_dq) < 8e-2
    assert _rel(out_8, out_ref) < 0.15    # weight rounding on top (sanity)


@cuda
@pytest.mark.parametrize("tokens", [1, 3, 4])
@pytest.mark.parametrize("fp8", [False, True])
def test_hc_fused_epilogue_matches_unfused(tokens, fp8, monkeypatch):
    """GatedResidual.mix with the rows-GEMV fused epilogues (silu / gate-mix inside the
    reduce passes) vs the separate-kernel path, for bf16 and load-time-FP8 mixes."""
    import freetoken.models.qwen4_exp.hc as hc_mod
    from freetoken.models.qwen4_exp.config import parse_config
    from freetoken.models.qwen4_exp.hc import GatedResidual
    from freetoken.utils import torch_dtype

    from .common import toy_hf_config

    if fp8:
        monkeypatch.setenv("FREETOKEN_DENSE_QUANT", "fp8")
    else:
        monkeypatch.delenv("FREETOKEN_DENSE_QUANT", raising=False)
    cfg = parse_config(toy_hf_config(4))
    gen = torch.Generator(device="cpu").manual_seed(3)
    for use_combine in (True, False):
        with torch.device("meta"), torch_dtype(torch.bfloat16):
            mod = GatedResidual(cfg, use_combine=use_combine)
        sd = {}
        for k, v in mod.state_dict().items():
            if k.endswith(".weight_scale"):
                continue
            w = (torch.randn(v.shape, generator=gen) * 0.05).to(DEV, torch.bfloat16)
            if v.dtype == torch.float8_e4m3fn:
                # the module was built fp8 (dense_quant=fp8): feed it load-time quantized weights
                q, sc = quantize_fp8_per_row(w)
                sd[k] = q
                sd[k[: -len(".weight")] + ".weight_scale"] = sc
            else:
                sd[k] = w
        mod.load_state_dict(sd)
        R = (torch.randn(tokens, cfg.qwen4_args.ple_state_width, generator=gen) * 0.5).to(DEV, torch.bfloat16)
        monkeypatch.setattr(hc_mod, "_HC_FUSED_EPILOGUE", False)
        x_ref, s_ref = mod.mix(R)
        monkeypatch.setattr(hc_mod, "_HC_FUSED_EPILOGUE", True)
        x_f, s_f = mod.mix(R)
        torch.cuda.synchronize()
        assert _rel(x_f, x_ref) < 1e-2, (use_combine, _rel(x_f, x_ref))
        if s_ref is not None:
            torch.testing.assert_close(s_f.float(), s_ref.float(), rtol=2e-2, atol=2e-2)
        else:
            assert s_f is None
