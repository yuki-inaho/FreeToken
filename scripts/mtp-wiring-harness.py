#!/usr/bin/env python
"""Settle the Qwen3.8-Flash-Next MTP head's stream wiring against the real checkpoint.

Inputs: the checkpoint and a pre-mixer residual dump captured by serving one natural-text
prompt with ``FREETOKEN_MTP_DEBUG_DUMP=<dir>``. The prompt must stay
under ~2,000 tokens so QSA equals dense attention and the whole prefill lands in one chunk.

For every candidate wiring (x position offset) the head runs teacher-forced over the prompt:
row ``i`` gets the main tower's pre-mixer residual ``R_i`` plus the embedding of the actual
next token ``t_{i+1}``, and predicts ``t_{i+2}``. A correct wiring agrees with the target
model's own next-token argmax 55-75% of the time; a wrong one is near chance (<1%).

Usage:
    python scripts/mtp-wiring-harness.py --dump-dir /tmp/premix [--model-path ...] [--skip 32]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from types import SimpleNamespace

os.environ.setdefault("FREETOKEN_LOAD_MTP", "1")

import torch


def load_dump(dump_dir: str) -> dict:
    """The largest dump file is the prompt prefill (graph warmup produces tiny ones)."""
    files = sorted(glob.glob(os.path.join(dump_dir, "premix_*.pt")))
    if not files:
        raise SystemExit(f"no premix_*.pt files in {dump_dir}")
    best, best_rows = None, -1
    for path in files:
        data = torch.load(path, map_location="cpu", weights_only=True)
        if data["premix"].shape[0] > best_rows:
            best, best_rows = data, data["premix"].shape[0]
    positions = best["positions"].tolist()
    if positions != list(range(len(positions))):
        raise SystemExit("the largest dump is not a fresh full prefill (positions not 0..T-1)")
    print(f"dump: {best_rows} prompt tokens from {len(files)} file(s)")
    return best


def targeted_state_dict(model_path: str, keys: set[str], device: torch.device) -> dict:
    """Load exactly ``keys`` (raw checkpoint names) through the production rename+fusion path."""
    from freetoken.models.qwen4_exp.weight import _rename, _try_fuse

    with open(os.path.join(model_path, "model.safetensors.index.json"), encoding="utf-8") as fh:
        weight_map = json.load(fh)["weight_map"]
    by_shard: dict[str, list[str]] = {}
    for key in keys:
        by_shard.setdefault(weight_map[key], []).append(key)

    import safetensors

    out: dict[str, torch.Tensor] = {}
    fuse_buf: dict[str, dict[int, torch.Tensor]] = {}
    for shard, names in sorted(by_shard.items()):
        with safetensors.safe_open(
            os.path.join(model_path, shard), framework="pt", device=str(device)
        ) as f:
            for raw_name in sorted(names):
                name = _rename(raw_name)
                assert name is not None, raw_name
                tensor = f.get_tensor(raw_name)
                fused = _try_fuse(name, tensor, fuse_buf)
                if fused is None:
                    out[name] = tensor
                elif fused != ():
                    out[fused[0]] = fused[1]
    assert not fuse_buf, f"incomplete fusions: {sorted(fuse_buf)}"
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/media/nuko/New Volume/Qwen3.8-Flash-Next-NVFP4")
    parser.add_argument("--dump-dir", required=True)
    parser.add_argument("--skip", type=int, default=32, help="rows to skip at the front")
    parser.add_argument("--wirings", nargs="*", default=None)
    parser.add_argument("--pos-offsets", nargs="*", type=int, default=[0, 1])
    parser.add_argument(
        "--chain",
        action="store_true",
        help="also measure the CHAINED second draft (k=2 feasibility): feed the MTP "
        "layer's own 4-stream output back as the next step's premix, teacher-forcing "
        "the true next token, and score the next-next-token agreement",
    )
    args = parser.parse_args()

    # The head must allocate bf16 resident experts to accept the checkpoint's raw stacked
    # tensors (the serving default quantizes them into the offload cache instead).
    os.environ.setdefault("FREETOKEN_MTP_EXPERTS", "bf16")

    from transformers import AutoConfig

    from freetoken.core import Context, set_global_ctx
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.models.qwen4_exp.attention import TorchDenseQSAReference
    from freetoken.models.qwen4_exp.config import parse_config
    from freetoken.models.qwen4_exp.hc import GatedResidual
    from freetoken.models.qwen4_exp.mtp import MTP_WIRINGS, Qwen4ExpMTPHead
    from freetoken.moe.fused import FusedMoe
    from freetoken.utils.torch_utils import torch_dtype

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    wirings = args.wirings or list(MTP_WIRINGS)

    config = parse_config(AutoConfig.from_pretrained(args.model_path, local_files_only=True))

    dump = load_dump(args.dump_dir)
    premix = dump["premix"].to(device, dtype)  # [T, hc*hidden]
    input_ids = dump["input_ids"].to(device)
    if premix.shape[0] > 2000:
        # QSA is exactly dense only below ~2051 visible tokens; truncate so the dense
        # replay of the MTP layer stays an exact oracle.
        print(f"truncating {premix.shape[0]} dumped rows to 2000 for exact-dense QSA")
        premix, input_ids = premix[:2000], input_ids[:2000]
    t = premix.shape[0]

    lm = "model.language_model"
    shared = targeted_state_dict(
        args.model_path,
        {
            f"{lm}.embed_tokens.weight",
            "lm_head.weight",
            f"{lm}.hyper_connection_mixer.hc_norm.weight",
            f"{lm}.hyper_connection_mixer.input_mix_weight_down.weight",
            f"{lm}.hyper_connection_mixer.input_mix_weight_up.weight",
        },
        device,
    )
    embed_weight = shared["model.embed_tokens.weight"]
    lm_head_weight = shared["lm_head.weight"]

    with torch.device(device), torch_dtype(dtype):
        main_mixer = GatedResidual(config, use_combine=False)
    main_mixer.load_state_dict(
        {k.split("hyper_connection_mixer.", 1)[1]: v for k, v in shared.items() if "mixer" in k}
    )

    with open(
        os.path.join(args.model_path, "model.safetensors.index.json"), encoding="utf-8"
    ) as fh:
        mtp_raw_keys = {k for k in json.load(fh)["weight_map"] if k.startswith("mtp.")}
    print(f"loading {len(mtp_raw_keys)} mtp tensors...")
    mtp_sd = targeted_state_dict(args.model_path, mtp_raw_keys, device)

    with torch.no_grad():
        target_hidden = main_mixer.mix(premix)[0]
        target_argmax = torch.argmax(target_hidden @ lm_head_weight.T, dim=-1)  # [T]
        text_agree = (target_argmax[args.skip : -1] == input_ids[args.skip + 1 :]).float().mean()
    print(f"dump sanity: target argmax vs actual text = {text_agree:.1%} (healthy: 40-90%)")

    # Teacher-forced rows: R_i + embed(t_{i+1}) -> predict t_{i+2}, for i in [0, T-2).
    rows = t - 1
    premix_rows = premix[:rows]
    next_embeds = embed_weight[input_ids[1 : rows + 1]].to(dtype)
    actual_next2 = input_ids[2 : rows + 1]  # length rows-1
    target_next2 = target_argmax[1:rows]  # target's own prediction for the same slot

    with torch.device(device), torch_dtype(dtype):
        head = Qwen4ExpMTPHead(config, wiring="per_stream")
    head.load_state_dict(dict(mtp_sd), prefix="mtp")

    def plain_grouped_rmsnorm(x: torch.Tensor, weight: torch.Tensor, groups: int) -> torch.Tensor:
        xf = x.float().unflatten(-1, (groups, -1))
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + config.rms_norm_eps)
        return (xf.flatten(-2) * weight.float()).to(x.dtype)

    def merged_variants() -> dict[str, torch.Tensor]:
        """Every shape-consistent stream wiring, from shared per-component tensors."""
        t = rows
        hc, h = head._hc_count, head._hidden
        rn = head.pre_fc_norm_hidden.forward(premix_rows)  # (1+w) grouped
        rn_plain = plain_grouped_rmsnorm(premix_rows, head.pre_fc_norm_hidden.weight, hc)
        en = head.pre_fc_norm_embedding.forward(next_embeds)  # (1+w)
        en_plain = plain_grouped_rmsnorm(next_embeds, head.pre_fc_norm_embedding.weight, 1)

        def fh_of(x):  # per-stream shared fc_hidden
            return head.fc_hidden.forward(x.reshape(t * hc, h)).reshape(t, hc, h)

        fh, fh_plain = fh_of(rn), fh_of(rn_plain)
        fe, fe_plain = head.fc_embedding.forward(en), head.fc_embedding.forward(en_plain)
        fe_swap = head.fc_hidden.forward(en)  # swapped fc roles

        def fh_swap_of(x):
            return head.fc_embedding.forward(x.reshape(t * hc, h)).reshape(t, hc, h)

        streams = premix_rows.reshape(t, hc, h)
        out = {
            "per_stream": fh + fe.unsqueeze(1),
            "per_stream_emb_plain": fh + fe_plain.unsqueeze(1),
            "per_stream_all_plain": fh_plain + fe_plain.unsqueeze(1),
            "per_stream_hid_plain": fh_plain + fe.unsqueeze(1),
            "per_stream_residual": streams + fh + fe.unsqueeze(1),
            "per_stream_residual_x2": streams + 2.0 * (fh + fe.unsqueeze(1)),
            "per_stream_residual_x4": streams + 4.0 * (fh + fe.unsqueeze(1)),
            "per_stream_half": 0.5 * (fh + fe.unsqueeze(1)),
            "swap_fc_roles": fh_swap_of(rn) + fe_swap.unsqueeze(1),
            "emb_stream_last": torch.cat([fh[:, : hc - 1], fe.unsqueeze(1)], dim=1),
            "emb_stream_first": torch.cat([fe.unsqueeze(1), fh[:, 1:]], dim=1),
            "mean_collapse": (fh.mean(dim=1) + fe).unsqueeze(1).expand(t, hc, h),
        }
        return {name: v.reshape(t, hc * h) for name, v in out.items()}

    def run_layer(merged: torch.Tensor, pos_offset: int = 0) -> torch.Tensor:
        """One fresh full-sequence pass of the MTP decoder layer; returns the 4-stream
        residual output (pre-mixer)."""
        import freetoken.core as core

        n = merged.shape[0]
        core._GLOBAL_CTX = None  # standalone script: each run builds a fresh ctx
        ctx = Context(page_size=64)
        ctx.attn_backend = TorchDenseQSAReference(
            config, num_slots=1, max_len=n + pos_offset + 8, device=device, dtype=dtype
        )
        ctx.moe_backend = FusedMoe()
        set_global_ctx(ctx)
        batch = SimpleNamespace(
            positions=torch.arange(pos_offset, n + pos_offset, dtype=torch.int32, device=device),
            padded_reqs=[SimpleNamespace(extend_len=n, table_idx=0, cached_len=0)],
            is_prefill=True,
            is_decode=False,
        )
        return head.layers.op_list[0].forward(merged, batch)

    def run_tail(merged: torch.Tensor) -> torch.Tensor:
        return head.hyper_connection_mixer.mix(run_layer(merged))[0]

    results = []
    with torch.no_grad():
        variants = merged_variants()
        wanted = args.wirings or list(variants)
        for name in wanted:
            pred = torch.argmax(run_tail(variants[name]) @ lm_head_weight.T, dim=-1)
            sl = slice(args.skip, rows - 1)
            vs_target = (pred[sl] == target_next2[sl]).float().mean().item()
            vs_text = (pred[sl] == actual_next2[sl]).float().mean().item()
            # echo diagnostics: a broken wiring often reproduces its INPUT token t_{i+1}
            # or the target's same-slot t_{i+1} prediction instead of predicting t_{i+2}
            echo_input = (pred[sl] == input_ids[args.skip + 1 : rows]).float().mean().item()
            echo_slot = (pred[sl] == target_argmax[args.skip : rows - 1]).float().mean().item()
            results.append((name, vs_target, vs_text))
            print(
                f"wiring={name:<22} vs target argmax: {vs_target:6.1%}   "
                f"vs actual text: {vs_text:6.1%}   echo(input): {echo_input:5.1%}   "
                f"echo(slot): {echo_slot:5.1%}"
            )

    best = max(results, key=lambda r: r[1])
    print(
        f"\nbest: wiring={best[0]} ({best[1]:.1%} vs target, {best[2]:.1%} vs text). "
        + ("LOOKS CORRECT" if best[1] > 0.35 else "ALL CANDIDATES LOOK WRONG -- do not proceed")
    )

    if args.chain:
        # k=2 feasibility: the engine would draft a SECOND token by re-running the head with
        # its own 4-stream layer output as the premix (positions shifted one right) and the
        # first draft as the teacher token. Teacher-force both inputs here (true premix
        # chain, true t_{i+2}) so the score isolates the chain wiring itself. Chained row i
        # then predicts t_{i+3}; attention runs dense-causal over the chained rows, a mild
        # approximation of the engine's first-pass-KV context.
        with torch.no_grad():
            r1 = run_layer(variants["per_stream"])  # [rows, hc*h] pre-mixer output
            pred1 = torch.argmax(
                (head.hyper_connection_mixer.mix(r1)[0] @ lm_head_weight.T), dim=-1
            )
            rows2 = rows - 1  # chained row i needs true t_{i+2} = input_ids[i+2], i+2 <= rows
            emb2 = embed_weight[input_ids[2 : rows2 + 2]].to(dtype)
            m2 = head.build_input(r1[:rows2], emb2)
            pred2 = torch.argmax(
                (head.hyper_connection_mixer.mix(run_layer(m2, pos_offset=1))[0]
                 @ lm_head_weight.T),
                dim=-1,
            )
            sl = slice(args.skip, rows2 - 2)  # i+3 must stay within the dump
            vs_target = (pred2[sl] == target_argmax[2 : rows2 + 2][sl]).float().mean().item()
            vs_text = (pred2[sl] == input_ids[3 : rows2 + 3][sl]).float().mean().item()
            echo_teacher = (pred2[sl] == input_ids[2 : rows2 + 2][sl]).float().mean().item()
            echo_first = (pred2[sl] == pred1[:rows2][sl]).float().mean().item()
        print(
            f"\nchained draft (2nd token): vs target argmax: {vs_target:6.1%}   "
            f"vs actual text: {vs_text:6.1%}   echo(teacher tok): {echo_teacher:5.1%}   "
            f"echo(1st pred): {echo_first:5.1%}"
        )
        print(
            "chain verdict: "
            + ("GO for k=2" if vs_target > 0.30 else "chained wiring looks WRONG -- k=2 not viable as wired")
        )

    sys.exit(0 if best[1] > 0.35 else 1)


if __name__ == "__main__":
    main()
