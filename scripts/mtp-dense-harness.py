#!/usr/bin/env python
"""Offline wiring harness for the dense Qwen3.5/3.8 MTP head (pure torch, no engine).

Feed it the rows dumped by a live prefill with ``FREETOKEN_MTP_DEBUG_DUMP=<dir>`` (the
qwen3_5_moe model seam writes ``rows_*.pt`` with input_ids / positions / pre-norm residual /
post-norm hidden per forward). The head is re-implemented here in plain torch straight from
the checkpoint's raw ``mtp.*`` tensors, so it depends on none of FreeToken's kernels, loader
fusions or executor plumbing: a variant that scores well proves the wiring; a zoo that all
scores at chance points at the tensors / conventions themselves.

    python scripts/mtp-dense-harness.py --model-path <ckpt> --dump-dir <dir>
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import torch
from safetensors import safe_open


def load_dump(dump_dir: str) -> dict:
    files = sorted(glob.glob(os.path.join(dump_dir, "rows_*.pt")))
    if not files:
        raise SystemExit(f"no rows_*.pt in {dump_dir}")
    best, best_rows = None, -1
    for f in files:
        d = torch.load(f)
        if d["input_ids"].shape[0] > best_rows:
            best, best_rows = d, d["input_ids"].shape[0]
    print(f"dump: {best_rows} rows from {len(files)} file(s)")
    return best


def load_raw(model_path: str, names: set[str], device) -> dict[str, torch.Tensor]:
    with open(os.path.join(model_path, "model.safetensors.index.json"), encoding="utf-8") as fh:
        wm = json.load(fh)["weight_map"]
    out: dict[str, torch.Tensor] = {}
    by_file: dict[str, list[str]] = {}
    for n in names:
        by_file.setdefault(wm[n], []).append(n)
    for f, ns in by_file.items():
        with safe_open(os.path.join(model_path, f), "pt", device=str(device)) as sf:
            for n in ns:
                out[n] = sf.get_tensor(n)
    return out


def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float, plus1: bool) -> torch.Tensor:
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    scale = (1.0 + w.float()) if plus1 else w.float()
    return (xf * scale).to(x.dtype)


def apply_rope(x: torch.Tensor, positions: torch.Tensor, rotary_dim: int, base: float):
    """Neox-style rotate-half over the first ``rotary_dim`` dims of each head. x [T, H, D]."""
    t, h, d = x.shape
    inv = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, device=x.device).float() / rotary_dim))
    freqs = positions.float()[:, None] * inv[None, :]  # [T, rd/2]
    cos, sin = freqs.cos()[:, None, :], freqs.sin()[:, None, :]
    xr, xp = x[..., :rotary_dim].float(), x[..., rotary_dim:]
    x1, x2 = xr[..., : rotary_dim // 2], xr[..., rotary_dim // 2 :]
    rot = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
    return torch.cat([rot.to(x.dtype), xp], dim=-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--dump-dir", required=True)
    ap.add_argument("--skip", type=int, default=16)
    ap.add_argument("--max-rows", type=int, default=1024)
    args = ap.parse_args()

    from transformers import AutoConfig

    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.models.qwen3_5_moe.config import parse_config

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    cfg = parse_config(AutoConfig.from_pretrained(args.model_path, local_files_only=True))
    device = torch.device("cuda")
    dtype = torch.bfloat16
    H, NQ, NKV, HD = cfg.hidden_size, cfg.num_qo_heads, cfg.num_kv_heads, cfg.head_dim
    eps = cfg.rms_norm_eps
    rotary_dim, base = cfg.rotary_config.rotary_dim, cfg.rotary_config.base
    print(f"config: hidden={H} heads={NQ}/{NKV} head_dim={HD} rotary_dim={rotary_dim} base={base}")

    d = load_dump(args.dump_dir)
    ids = d["input_ids"].to(device)[: args.max_rows]
    pos = d["positions"].to(device)[: args.max_rows]
    pre = d["pre_norm"].to(device, dtype)[: args.max_rows]
    post = d["post_norm"].to(device, dtype)[: args.max_rows]
    T = ids.shape[0]

    mtp = "mtp.layers.0"
    names = {
        "model.language_model.embed_tokens.weight", "lm_head.weight",
        "mtp.fc.weight", "mtp.norm.weight", "mtp.pre_fc_norm_hidden.weight",
        "mtp.pre_fc_norm_embedding.weight",
        f"{mtp}.input_layernorm.weight", f"{mtp}.post_attention_layernorm.weight",
        f"{mtp}.self_attn.q_proj.weight", f"{mtp}.self_attn.k_proj.weight",
        f"{mtp}.self_attn.v_proj.weight", f"{mtp}.self_attn.o_proj.weight",
        f"{mtp}.self_attn.q_norm.weight", f"{mtp}.self_attn.k_norm.weight",
        f"{mtp}.mlp.gate_proj.weight", f"{mtp}.mlp.up_proj.weight", f"{mtp}.mlp.down_proj.weight",
    }
    W = load_raw(args.model_path, names, device)
    embed = W["model.language_model.embed_tokens.weight"]
    lm_head = W["lm_head.weight"]

    with torch.no_grad():
        target_logits = post.float() @ lm_head.float().T if False else (post @ lm_head.T).float()
        target_argmax = target_logits.argmax(-1)
        text_agree = (target_argmax[args.skip:-1] == ids[args.skip + 1:]).float().mean().item()
        print(f"dump sanity: target argmax vs actual text = {text_agree:.1%} (healthy: 40-90%)")

    def head(rows_h: torch.Tensor, next_ids: torch.Tensor, positions: torch.Tensor,
             *, embed_first: bool, pre_plus1: bool) -> torch.Tensor:
        hn = rmsnorm(rows_h, W["mtp.pre_fc_norm_hidden.weight"], eps, pre_plus1)
        en = rmsnorm(embed[next_ids], W["mtp.pre_fc_norm_embedding.weight"], eps, pre_plus1)
        cat = torch.cat([en, hn] if embed_first else [hn, en], dim=-1)
        x = cat @ W["mtp.fc.weight"].T
        residual = x
        h = rmsnorm(x, W[f"{mtp}.input_layernorm.weight"], eps, True)
        n = h.shape[0]
        qg = (h @ W[f"{mtp}.self_attn.q_proj.weight"].T).view(n, NQ, 2 * HD)
        q, gate = qg[..., :HD], qg[..., HD:].reshape(n, NQ * HD)
        k = (h @ W[f"{mtp}.self_attn.k_proj.weight"].T).view(n, NKV, HD)
        v = (h @ W[f"{mtp}.self_attn.v_proj.weight"].T).view(n, NKV, HD)
        q = rmsnorm(q, W[f"{mtp}.self_attn.q_norm.weight"], eps, True)
        k = rmsnorm(k, W[f"{mtp}.self_attn.k_norm.weight"], eps, True)
        q = apply_rope(q, positions, rotary_dim, base)
        k = apply_rope(k, positions, rotary_dim, base)
        rep = NQ // NKV
        kk = k.repeat_interleave(rep, dim=1).transpose(0, 1).float()  # [NQ, T, HD]
        vv = v.repeat_interleave(rep, dim=1).transpose(0, 1).float()
        qq = q.transpose(0, 1).float()
        scores = qq @ kk.transpose(1, 2) / (HD ** 0.5)
        mask = torch.triu(torch.ones(n, n, device=device, dtype=torch.bool), 1)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = (scores.softmax(-1) @ vv).transpose(0, 1).reshape(n, NQ * HD)
        o = (attn.to(dtype) * torch.sigmoid(gate)) @ W[f"{mtp}.self_attn.o_proj.weight"].T
        residual = residual + o
        h = rmsnorm(residual, W[f"{mtp}.post_attention_layernorm.weight"], eps, True)
        g = h @ W[f"{mtp}.mlp.gate_proj.weight"].T
        u = h @ W[f"{mtp}.mlp.up_proj.weight"].T
        m = (torch.nn.functional.silu(g) * u) @ W[f"{mtp}.mlp.down_proj.weight"].T
        residual = residual + m
        out = rmsnorm(residual, W["mtp.norm.weight"], eps, True)
        return (out @ lm_head.T).float()

    rows = T - 1  # row i pairs h_i with t_{i+1}, predicts t_{i+2}
    actual_next2 = ids[2 : rows + 1]
    target_next2 = target_argmax[1:rows]
    sl = slice(args.skip, rows - 1)
    results = []
    with torch.no_grad():
        for hidden_name, rows_h in (("pre_norm", pre), ("post_norm", post)):
            for embed_first in (False, True):
                for pre_plus1 in (True, False):
                    logits = head(rows_h[:rows], ids[1 : rows + 1], pos[:rows],
                                  embed_first=embed_first, pre_plus1=pre_plus1)
                    pred = logits.argmax(-1)
                    vs_target = (pred[sl] == target_next2[sl]).float().mean().item()
                    vs_text = (pred[sl] == actual_next2[sl]).float().mean().item()
                    echo_in = (pred[sl] == ids[args.skip + 1 : rows]).float().mean().item()
                    name = f"{hidden_name:<9} {'embed|hidden' if embed_first else 'hidden|embed'} prefc{'+1' if pre_plus1 else ' plain'}"
                    results.append((name, vs_target, vs_text))
                    print(f"{name:<40} vs target: {vs_target:6.1%}  vs text: {vs_text:6.1%}  echo(input): {echo_in:5.1%}")
    best = max(results, key=lambda r: r[2])
    print(f"\nbest: {best[0]} ({best[2]:.1%} vs text) -> " + ("LOOKS CORRECT" if best[2] > 0.3 else "ALL WRONG"))


if __name__ == "__main__":
    main()
