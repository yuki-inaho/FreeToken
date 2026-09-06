#!/usr/bin/env python3
"""How wrong is the answer when one expert stands in for another?

`--moe-residency-routing` swaps a routed expert for a near-tied resident one and bounds the
swap by how much ROUTING WEIGHT it moves. That bound says nothing about whether the substitute
computes anything like the expert it replaced, and the quality gates cannot separate the two:
GSM8K and MMLU came out byte-identical with and without the feature, which rules out gross
breakage and nothing finer.

This measures the thing the bound leaves open, directly. For one MoE layer:

  * take the real expert inputs from a route dump (`block_input` is what the experts consume --
    there is no norm between it and them, see Qwen4ExpMoE.forward_combine),
  * dequantize all 512 NVFP4 experts of the layer,
  * run EVERY expert on every token, and
  * for each expert i, over the tokens that actually routed to i, measure how far each other
    expert j lands from it.

Two errors, because they answer different questions:

  ``expert``  ||E_j(x) - E_i(x)|| / ||E_i(x)||          -- how different the two functions are
  ``block``   w_i * ||E_j(x) - E_i(x)|| / ||sum_k w_k E_k(x)||
              -- how much the MoE block's output actually moves, which is what the residual
                 stream sees and the only one that bounds downstream damage

The diagonal is exactly zero by construction, which is the harness's own correctness check.

Output: an N x N matrix per layer (npz) plus the summary that matters -- the error of the best
possible substitute, and the error of the substitute the shipped policy actually picks (the
next-highest-logit expert outside the token's top-k).

  scripts/tier3-expert-substitutability.py --layers 0 23 47 --out $SCRATCH/subst
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time

import torch
from safetensors import safe_open

MODEL_DEFAULT = "/media/nuko/New Volume/Qwen3.8-Flash-Next-NVFP4"
DUMPS_DEFAULT = "/media/nuko/New Volume/route-dumps-m1"


def load_layer_experts(model: str, idx: dict, layer: int, num_experts: int, device):
    """All of one layer's experts, dequantized to bf16.

    Stacks the 512 per-expert NVFP4 tensors and dequantizes in three kernel calls instead of
    1,536, and reads shard by shard so the checkpoint is touched once per file.
    """
    from freetoken.kernel.triton.nvfp4_dequant import dequant_nvfp4

    prefix = f"model.language_model.layers.{layer}.mlp.experts"
    out = {}
    for proj in ("gate_proj", "up_proj", "down_proj"):
        # group the expert ids by which shard holds them
        by_shard: dict[str, list[int]] = {}
        for e in range(num_experts):
            by_shard.setdefault(idx[f"{prefix}.{e}.{proj}.weight"], []).append(e)
        packed = scale = None
        glob_scale = torch.empty(num_experts, dtype=torch.float32)
        for shard, ids in by_shard.items():
            with safe_open(os.path.join(model, shard), framework="pt") as f:
                for e in ids:
                    w = f.get_tensor(f"{prefix}.{e}.{proj}.weight")
                    s = f.get_tensor(f"{prefix}.{e}.{proj}.weight_scale")
                    g = f.get_tensor(f"{prefix}.{e}.{proj}.weight_scale_2")
                    if packed is None:
                        packed = torch.empty((num_experts, *w.shape), dtype=w.dtype)
                        scale = torch.empty((num_experts, *s.shape), dtype=s.dtype)
                    packed[e] = w
                    scale[e] = s
                    glob_scale[e] = g.reshape(1).float()
        O = packed.shape[1]
        # the dequant kernel takes a per-output-row global scale; the checkpoint's is per tensor
        gl = glob_scale.to(torch.float16)[:, None].expand(num_experts, O).contiguous()
        out[proj] = dequant_nvfp4(
            packed.to(device).contiguous(), scale.to(device).contiguous(), gl.to(device),
            torch.arange(num_experts, dtype=torch.int32, device=device),
            dtype=torch.bfloat16,
        )
        del packed, scale
        packed = scale = None
    return out["gate_proj"], out["up_proj"], out["down_proj"]


def experts_on(gate_w, up_w, down_w, X):
    """Every expert applied to every token. -> [E, H, C] (H-major keeps the reductions cheap).

    ``X`` is [C, H]. gate/up are [E, I, H]; down is [E, H, I].
    """
    xt = X.t().contiguous()                       # [H, C]
    g = torch.matmul(gate_w, xt)                  # [E, I, C]
    u = torch.matmul(up_w, xt)                    # [E, I, C]
    act = torch.nn.functional.silu(g.float()).to(g.dtype) * u
    del g, u
    return torch.matmul(down_w, act)              # [E, H, C]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--dumps", default=DUMPS_DEFAULT)
    ap.add_argument("--layers", nargs="*", type=int, default=[0, 23, 47])
    ap.add_argument("--num-experts", type=int, default=512)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--tokens", type=int, default=0, help="cap on tokens per layer (0 = all)")
    ap.add_argument("--chunk", type=int, default=192, help="tokens per pass over the experts")
    ap.add_argument("--tau", type=float, default=0.05,
                    help="also report the error restricted to the (token, slot) pairs where the "
                         "tau gate would actually permit the swap")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    dev = torch.device("cuda")
    idx = json.load(open(f"{args.model}/model.safetensors.index.json"))["weight_map"]
    files = sorted(glob.glob(f"{args.dumps}/route_*.pt"))
    if not files:
        raise SystemExit(f"no route_*.pt under {args.dumps}")
    E, K = args.num_experts, args.top_k

    for layer in args.layers:
        t0 = time.perf_counter()
        xs, lgs = [], []
        for path in files:
            blob = torch.load(path, map_location="cpu", weights_only=True)
            ent = blob["layers"][layer]
            xs.append(ent["block_input"])
            lgs.append(ent["logits"])
            del blob
        X_all = torch.cat(xs).to(torch.bfloat16)
        LG_all = torch.cat(lgs).float()
        del xs, lgs
        if args.tokens:
            X_all, LG_all = X_all[: args.tokens], LG_all[: args.tokens]
        T = X_all.shape[0]
        # the model's own routing: softmax over all experts, top-k, renormalised (norm_topk_prob)
        top = torch.topk(LG_all, K, dim=-1)
        W_all = torch.softmax(top.values, dim=-1)
        R_all = top.indices
        # the substitute the shipped policy would reach for: the best expert OUTSIDE the top-k
        nxt_all = torch.topk(LG_all, K + 1, dim=-1).indices[:, K]
        print(f"\n=== layer {layer}: {T} tokens, {E} experts, top-{K} "
              f"({time.perf_counter() - t0:.0f}s to load activations)")

        t0 = time.perf_counter()
        gate_w, up_w, down_w = load_layer_experts(args.model, idx, layer, E, dev)
        print(f"    experts dequantized in {time.perf_counter() - t0:.0f}s "
              f"({sum(t.numel() * 2 for t in (gate_w, up_w, down_w)) / 2**30:.1f} GiB bf16)")

        # what the shipped policy would really do: substitute slot k with `nxt` only when the
        # displacement it costs is <= tau. Restricting to those pairs separates "how bad is
        # substitution" from "how bad is the substitution we actually perform".
        LGd = LG_all.to(dev)
        pk = torch.softmax(torch.topk(LGd, K, dim=-1).values, dim=-1)     # [T, K]
        lg_c = LGd.gather(1, nxt_all.to(dev)[:, None])                    # [T, 1]
        lg_sel = LGd.gather(1, R_all.to(dev))                             # [T, K]
        m_ = torch.maximum(lg_sel.max(1, keepdim=True).values, lg_c)
        pe_ = torch.exp(lg_sel - m_)
        s_ = pe_.sum(1, keepdim=True)
        pc_ = torch.exp(lg_c - m_)
        disp_all = pe_ / s_ - pc_ / (s_ - pe_ + pc_)                      # [T, K]
        gate_all = (disp_all <= args.tau).cpu()
        del LGd, pk, lg_c, lg_sel, m_, pe_, s_, pc_, disp_all
        g_err = torch.zeros(E, dtype=torch.float64, device=dev)
        g_blk = torch.zeros(E, dtype=torch.float64, device=dev)
        g_cos = torch.zeros(E, dtype=torch.float64, device=dev)
        g_cnt = torch.zeros(E, dtype=torch.float64, device=dev)
        err_expert = torch.zeros(E, E, dtype=torch.float64, device=dev)
        err_block = torch.zeros(E, E, dtype=torch.float64, device=dev)
        err_resid = torch.zeros(E, E, dtype=torch.float64, device=dev)
        cos_sum = torch.zeros(E, E, dtype=torch.float64, device=dev)
        cnt = torch.zeros(E, dtype=torch.float64, device=dev)
        t0 = time.perf_counter()
        for lo in range(0, T, args.chunk):
            hi = min(T, lo + args.chunk)
            C = hi - lo
            X = X_all[lo:hi].to(dev)
            R = R_all[lo:hi].to(dev)          # [C, K]
            Wt = W_all[lo:hi].to(dev)         # [C, K]
            Y = experts_on(gate_w, up_w, down_w, X)          # [E, H, C] bf16
            Yf = Y.float()
            del Y
            nrm = Yf.pow(2).sum(1).clamp_min(1e-12).sqrt()   # [E, C]
            cidx = torch.arange(C, device=dev)
            Ref = Yf[R, :, cidx[:, None]]                    # [C, K, H]
            y_moe = torch.einsum("ck,ckh->ch", Wt, Ref)      # [C, H]
            block_n = y_moe.pow(2).sum(-1).clamp_min(1e-12).sqrt()   # [C]
            resid_n = X.float().pow(2).sum(-1).clamp_min(1e-12).sqrt()  # [C]
            # The Gram identity ||a-b||^2 = |a|^2 - 2<a,b> + |b|^2 cancels catastrophically in
            # fp32 exactly where the answer is small -- it put 3e-3 on a diagonal that is
            # identically zero. Ten explicit differences (one per routed slot) cost one extra
            # pass over Y and are exact, which the diagonal check then confirms.
            for k in range(K):
                refk = Ref[:, k, :].t().contiguous()         # [H, C]
                diff = Yf - refk[None, :, :]                 # [E, H, C]
                d = diff.pow(2).sum(1).clamp_min(0).sqrt()   # [E, C]
                del diff
                dotk = (Yf * refk[None, :, :]).sum(1)        # [E, C]
                rn = nrm[R[:, k], cidx]                      # [C]
                i = R[:, k]
                err_expert.index_add_(0, i, (d / rn[None, :]).t().double())
                err_block.index_add_(0, i, (d * Wt[:, k][None, :] / block_n[None, :]).t().double())
                err_resid.index_add_(0, i, (d * Wt[:, k][None, :] / resid_n[None, :]).t().double())
                cos_sum.index_add_(0, i, (dotk / (nrm * rn[None, :])).t().double())
                cnt.index_add_(0, i, torch.ones(C, dtype=torch.float64, device=dev))
                # the same numbers, restricted to slots the tau gate would let through, and
                # to the one substitute the policy would pick
                gm = gate_all[lo:hi, k].to(dev)
                if gm.any():
                    jj = nxt_all[lo:hi].to(dev)
                    dj = d[jj, torch.arange(C, device=dev)]
                    rel_j = dj / rn
                    blk_j = dj * Wt[:, k] / block_n
                    cos_j = dotk[jj, torch.arange(C, device=dev)] / (nrm[jj, torch.arange(C, device=dev)] * rn)
                    g_err.index_add_(0, i[gm], rel_j[gm].double())
                    g_blk.index_add_(0, i[gm], blk_j[gm].double())
                    g_cos.index_add_(0, i[gm], cos_j[gm].double())
                    g_cnt.index_add_(0, i[gm], torch.ones(int(gm.sum()), dtype=torch.float64, device=dev))
                del d, dotk
            del Yf, Ref, y_moe
        torch.cuda.empty_cache()
        print(f"    {T * E:,} expert-token evaluations in {time.perf_counter() - t0:.0f}s")

        seen = cnt.clamp_min(1)[:, None]
        Mexp = (err_expert / seen).float().cpu()
        Mblk = (err_block / seen).float().cpu()
        Mres = (err_resid / seen).float().cpu()
        Mcos = (cos_sum / seen).float().cpu()
        gn = g_cnt.clamp_min(1)
        gated = ((g_err / gn).cpu(), (g_blk / gn).cpu(), (g_cos / gn).cpu(), g_cnt.cpu())
        counts = cnt.cpu()
        report(layer, Mexp, Mblk, Mres, Mcos, counts, R_all, nxt_all, LG_all, K, args, gated)


def report(layer, Mexp, Mblk, Mres, Mcos, counts, R_all, nxt_all, LG_all, K, args, gated=None) -> None:
    E = Mexp.shape[0]
    diag = Mexp.diagonal()
    live = counts > 0
    eye = torch.eye(E, dtype=torch.bool)
    off = Mexp.masked_fill(eye, float("inf"))
    best_val, best_j = off.min(dim=1)
    offb = Mblk.masked_fill(eye, float("inf"))
    best_blk = offb.min(dim=1).values
    offr = Mres.masked_fill(eye, float("inf"))
    best_res = offr.min(dim=1).values
    cos_off = Mcos.masked_fill(eye, float("-inf"))
    best_cos = cos_off.max(dim=1).values

    q = lambda t, f: t.sort().values[min(len(t) - 1, int(f * len(t)))].item()  # noqa: E731
    print(f"    tokens per expert: min {int(counts[live].min())}, "
          f"median {int(counts[live].median())}, max {int(counts.max())}")
    print(f"    diagonal (must be 0): max {diag.abs().max():.2e}")

    bl, bb = best_val[live], best_blk[live]
    br, bc = best_res[live], best_cos[live]
    print(f"\n    BEST substitute for an expert, over the tokens routed to it:")
    print(f"      expert-output error   p10 {q(bl,0.1):.3f}  p50 {q(bl,0.5):.3f}  p90 {q(bl,0.9):.3f}")
    print(f"      cosine w/ the original       p50 {q(bc,0.5):+.3f}  p90 {q(bc,0.9):+.3f}")
    print(f"      MoE block output moves       p50 {q(bb,0.5):.4f}  p90 {q(bb,0.9):.4f}")
    print(f"      residual stream moves        p50 {q(br,0.5):.4f}  p90 {q(br,0.9):.4f}")

    # a random other expert, for scale
    rnd = off.masked_fill(off == float("inf"), float("nan"))
    rmean = torch.nanmean(rnd, dim=1)[live]
    rndb = offb.masked_fill(offb == float("inf"), float("nan"))
    rmeanb = torch.nanmean(rndb, dim=1)[live]
    rndc = Mcos.masked_fill(eye, float("nan"))
    rmeanc = torch.nanmean(rndc, dim=1)[live]
    rndr = Mres.masked_fill(eye, float("nan"))
    rmeanr = torch.nanmean(rndr, dim=1)[live]
    print(f"    an AVERAGE other expert, same tokens:")
    print(f"      expert-output error   p50 {q(rmean,0.5):.3f}   (sqrt(2)=1.414 means orthogonal)")
    print(f"      cosine w/ the original       p50 {q(rmeanc,0.5):+.3f}")
    print(f"      MoE block output moves       p50 {q(rmeanb,0.5):.4f}")
    print(f"      residual stream moves        p50 {q(rmeanr,0.5):.4f}")

    # what the shipped policy actually picks: the best expert outside the token's top-k.
    # Averaged per (reference expert) over its own tokens, so it is comparable to the above.
    pol_e, pol_b, pol_r, pol_c, pol_n = (torch.zeros(E) for _ in range(5))
    for k in range(K):
        i, j = R_all[:, k], nxt_all
        pol_e.index_add_(0, i, Mexp[i, j])
        pol_b.index_add_(0, i, Mblk[i, j])
        pol_r.index_add_(0, i, Mres[i, j])
        pol_c.index_add_(0, i, Mcos[i, j])
        pol_n.index_add_(0, i, torch.ones(i.shape[0]))
    n = pol_n.clamp_min(1)
    pe, pb, pr, pc = ((v / n)[live] for v in (pol_e, pol_b, pol_r, pol_c))
    print(f"    the substitute the SHIPPED POLICY picks (best expert outside the top-k):")
    print(f"      expert-output error   p10 {q(pe,0.1):.3f}  p50 {q(pe,0.5):.3f}  p90 {q(pe,0.9):.3f}")
    print(f"      cosine w/ the original       p50 {q(pc,0.5):+.3f}")
    print(f"      MoE block output moves       p50 {q(pb,0.5):.4f}  p90 {q(pb,0.9):.4f}")
    print(f"      residual stream moves        p50 {q(pr,0.5):.4f}  p90 {q(pr,0.9):.4f}")

    if gated is not None:
        ge, gb, gc, gcnt = gated
        m = gcnt >= 20
        if m.any():
            print(f"    ... and restricted to the pairs tau={args.tau} would actually permit "
                  f"({int(gcnt.sum())} of {int(counts.sum())} routed slots):")
            print(f"      expert-output error   p10 {q(ge[m],0.1):.3f}  p50 {q(ge[m],0.5):.3f}  "
                  f"p90 {q(ge[m],0.9):.3f}")
            print(f"      cosine w/ the original       p50 {q(gc[m],0.5):+.3f}")
            print(f"      MoE block output moves       p50 {q(gb[m],0.5):.4f}  p90 {q(gb[m],0.9):.4f}")

    # does the router's own score predict substitutability? if not, the whole gating premise
    # of --moe-residency-routing (near-tied in logit space => interchangeable) is wrong.
    gate_norm = LG_all.std(dim=0)
    del gate_norm
    ranks = torch.argsort(torch.argsort(LG_all, dim=1, descending=True), dim=1)
    sel = torch.randint(0, LG_all.shape[0], (min(2000, LG_all.shape[0]),))
    xs, ys = [], []
    for t in sel.tolist():
        i = int(R_all[t, 0])
        for j in range(0, E, 17):           # a stride keeps it cheap and unbiased
            if j == i:
                continue
            xs.append(float(ranks[t, j]))
            ys.append(float(Mexp[i, j]))
    xt, yt = torch.tensor(xs), torch.tensor(ys)
    if xt.numel() > 10:
        c = torch.corrcoef(torch.stack([xt, yt]))[0, 1]
        print(f"    correlation between a substitute's ROUTER RANK and its expert-output "
              f"error: {c:+.3f}")
        for lo, hi in ((0, 16), (16, 64), (64, 256), (256, E)):
            m = (xt >= lo) & (xt < hi)
            if m.any():
                print(f"      rank {lo:>3}-{hi:<3}: mean error {yt[m].mean():.3f} "
                      f"({int(m.sum())} pairs)")

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        p = os.path.join(args.out, f"subst-layer{layer}.npz")
        import numpy as np
        np.savez_compressed(p, expert_error=Mexp.numpy(), block_error=Mblk.numpy(),
                            resid_error=Mres.numpy(), cosine=Mcos.numpy(),
                            counts=counts.numpy(), best_j=best_j.numpy())
        print(f"    matrix -> {p}")


if __name__ == "__main__":
    main()
