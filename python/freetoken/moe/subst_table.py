# SPDX-License-Identifier: Apache-2.0
"""Expert substitution table: what it costs to replace one expert with another, measured.

``--moe-residency-routing`` swaps a routed expert that is not in host RAM for the highest-logit
resident one, and bounds the swap by the ROUTING WEIGHT it displaces. Both halves are measured
to be wrong: experts are near-orthogonal (the best of 511 candidates reaches cosine only
+0.02..+0.19), and the router's ranking does not predict substitutability at all (rank-vs-error
correlation -0.16..+0.20, flat across rank bands). So the shipped policy picks what is
statistically a random expert, and prices it with a proxy that does not track the damage.

This builds the table that fixes both: for every (layer, expert i) it records how far each other
expert lands over the tokens that actually route to i, plus expert i's own mean output norm.
That is enough for a policy to price the two things it can actually do, in the units that matter
(fraction of the MoE block output moved):

    substitute i -> c, transplanting w_i:   ||dy|| = w_i * r_ic * n_i
    drop i, zeroing its weight:             ||dy|| = w_i * n_i

both divided by ||y||, estimated as ``sqrt(sum_k w_k^2 n_k^2)`` under the near-orthogonality the
measurement itself establishes, and calibrated per layer by ``y_cal``.

Dropping is L2-optimal among "remove this expert" actions -- ``y - w_i E_i`` is already the best
approximation of ``y`` in the span of the remaining experts, so renormalising cannot improve it --
and it beats substituting unless the substitute's cosine exceeds ~0.44. The shortlist is stored
unpruned so the offline replay can measure how often that happens rather than assuming.

Build:  ``ft moe subst --model <ckpt> --dumps <route dumps> --out <path>``
"""

from __future__ import annotations

import glob
import json
import os
import time

import torch

from freetoken.utils import init_logger

logger = init_logger(__name__)

FORMAT_VERSION = "1"
FILENAME = "expert_subst.safetensors"
DEFAULT_M = 32
# An expert the corpus barely exercised has no trustworthy statistic. Its measured norm would be
# near zero (or exactly zero if it was never routed), which would make DROPPING it look free --
# a silent invitation to drop precisely the experts nothing is known about. Below this floor the
# shortlist is emptied and the norm is replaced by a conservative one, so every predicted cost
# comes out high and the gate refuses to act.
MIN_COUNT = 64
# p75 of a Gaussian is mu + 0.6745 sigma. The gate wants to bound the token in front of it, not
# the average one, but keeping a per-pair distribution would cost 512*512*T values per layer;
# two accumulators and this constant cost nothing. `validate_estimator` is what checks it.
_P75_Z = 0.6745


# ---------------------------------------------------------------------------------------------
# expert weights
# ---------------------------------------------------------------------------------------------

def load_layer_experts(model: str, idx: dict, layer: int, num_experts: int, device):
    """All of one layer's experts, dequantized to bf16 -> (gate [E,I,H], up [E,I,H], down [E,H,I]).

    Stacks the per-expert NVFP4 tensors and dequantizes in three kernel calls instead of
    3*num_experts, and reads shard by shard so each checkpoint file is opened once.
    """
    from safetensors import safe_open

    from freetoken.kernel.triton.nvfp4_dequant import dequant_nvfp4

    prefix = f"model.language_model.layers.{layer}.mlp.experts"
    out = {}
    for proj in ("gate_proj", "up_proj", "down_proj"):
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
                    packed[e], scale[e] = w, s
                    glob_scale[e] = g.reshape(1).float()
        O = packed.shape[1]
        gl = glob_scale.to(torch.float16)[:, None].expand(num_experts, O).contiguous()
        out[proj] = dequant_nvfp4(
            packed.to(device).contiguous(), scale.to(device).contiguous(), gl.to(device),
            torch.arange(num_experts, dtype=torch.int32, device=device), dtype=torch.bfloat16,
        )
        del packed, scale
    return out["gate_proj"], out["up_proj"], out["down_proj"]


def experts_on(gate_w, up_w, down_w, X):
    """Every expert applied to every token -> [E, H, C]. ``X`` is [C, H].

    ``down(silu(gate(x)) * up(x))``, matching ``_reference_fused_experts_decode``.
    """
    xt = X.t().contiguous()
    g = torch.matmul(gate_w, xt)
    u = torch.matmul(up_w, xt)
    act = torch.nn.functional.silu(g.float()).to(g.dtype) * u
    del g, u
    return torch.matmul(down_w, act)


# ---------------------------------------------------------------------------------------------
# the corpus
# ---------------------------------------------------------------------------------------------

def dump_files(dumps: str) -> list[str]:
    return sorted(glob.glob(os.path.join(dumps, "route_*.pt")))


def iter_layer(files: list[str], layer: int, min_tokens: int = 256):
    """(block_input [T,H] bf16, logits [T,E] fp32) per dump, for one layer.

    ``mmap=True`` is what makes a layer-major build affordable: a dump is gigabytes across all
    layers, and this touches only the pages of the layer asked for.
    """
    for p in files:
        blob = torch.load(p, map_location="cpu", weights_only=True, mmap=True)
        ent = blob["layers"].get(layer)
        if ent is None or "block_input" not in ent:
            continue
        x = ent["block_input"]
        if x.shape[0] < min_tokens:  # decode-step dumps carry a token or four: no statistic
            continue
        yield x, ent["logits"].float()


# ---------------------------------------------------------------------------------------------
# the build
# ---------------------------------------------------------------------------------------------

def build_layer(model, idx, layer, files, *, num_experts, top_k, m, chunk, device,
                min_tokens=256):
    """One layer's row of the table. Returns a dict of CPU tensors."""
    gate_w, up_w, down_w = load_layer_experts(model, idx, layer, num_experts, device)
    E = num_experts
    err_sum = torch.zeros(E, E, dtype=torch.float64, device=device)
    err_sq = torch.zeros(E, E, dtype=torch.float64, device=device)
    cos_sum = torch.zeros(E, E, dtype=torch.float64, device=device)
    nrm_sum = torch.zeros(E, dtype=torch.float64, device=device)
    cnt = torch.zeros(E, dtype=torch.float64, device=device)
    cal_ratio: list[torch.Tensor] = []

    for X_cpu, LG_cpu in iter_layer(files, layer, min_tokens):
        T = X_cpu.shape[0]
        top = torch.topk(LG_cpu, top_k, dim=-1)
        W_all = torch.softmax(top.values, dim=-1)
        R_all = top.indices
        for lo in range(0, T, chunk):
            hi = min(T, lo + chunk)
            C = hi - lo
            X = X_cpu[lo:hi].to(device, torch.bfloat16)
            R = R_all[lo:hi].to(device)
            Wt = W_all[lo:hi].to(device)
            Y = experts_on(gate_w, up_w, down_w, X).float()      # [E, H, C]
            nrm = Y.pow(2).sum(1).clamp_min(1e-12).sqrt()        # [E, C]
            cidx = torch.arange(C, device=device)
            Ref = Y[R, :, cidx[:, None]]                          # [C, K, H]
            # the true ||y||, for the estimator's calibration
            y_true = torch.einsum("ck,ckh->ch", Wt, Ref).pow(2).sum(-1).clamp_min(1e-12).sqrt()
            for k in range(top_k):
                refk = Ref[:, k, :].t().contiguous()               # [H, C]
                # explicit difference, not the Gram identity: ||a-b||^2 = |a|^2-2<a,b>+|b|^2
                # cancels catastrophically in fp32 exactly where the answer is small, and put
                # 3e-3 on a diagonal that is identically zero.
                d = (Y - refk[None, :, :]).pow(2).sum(1).clamp_min(0).sqrt()   # [E, C]
                dotk = (Y * refk[None, :, :]).sum(1)
                i = R[:, k]
                rn = nrm[i, cidx]
                rel = (d / rn[None, :]).t()                        # [C, E]
                err_sum.index_add_(0, i, rel.double())
                err_sq.index_add_(0, i, rel.double().pow(2))
                cos_sum.index_add_(0, i, (dotk / (nrm * rn[None, :])).t().double())
                nrm_sum.index_add_(0, i, rn.double())
                cnt.index_add_(0, i, torch.ones(C, dtype=torch.float64, device=device))
                del d, dotk, rel
            # Keep a subsample for y_cal. The per-expert norms are only final after the whole
            # pass, so only the true ||y|| can be banked here; the ratio is finished below.
            if len(cal_ratio) < 8:
                cal_ratio.append((y_true.cpu(), R.cpu(), Wt.cpu()))
            del Y, Ref, nrm, y_true
    del gate_w, up_w, down_w
    torch.cuda.empty_cache()

    seen = cnt.clamp_min(1)
    mean = (err_sum / seen[:, None])
    var = (err_sq / seen[:, None] - mean.pow(2)).clamp_min(0)
    p75 = mean + _P75_Z * var.sqrt()
    cos = (cos_sum / seen[:, None])
    norm = (nrm_sum / seen)
    thin = cnt < MIN_COUNT
    if thin.any():
        well = norm[~thin]
        conservative = (well.quantile(0.9) if well.numel() else norm.max().clamp_min(1.0))
        norm = torch.where(thin, conservative, norm)

    # rank the shortlist by the gate statistic itself, so "first resident entry" is also "best
    # under the metric the gate uses". Unpruned: whether substitution ever beats dropping is a
    # measurement (see the offline replay), not an assumption baked into the artifact.
    eye = torch.eye(E, dtype=torch.bool, device=device)
    p75_off = p75.masked_fill(eye, float("inf"))
    best = torch.topk(p75_off, m, dim=1, largest=False)
    sub_ids = best.indices.to(torch.int32)
    sub_err = best.values.float()
    # under-observed experts: no shortlist, and a cost the gate will refuse
    sub_ids = torch.where(thin[:, None], torch.full_like(sub_ids, -1), sub_ids)
    sub_err = torch.where(thin[:, None], torch.full_like(sub_err, 9.0), sub_err)
    gather = lambda t: t.gather(1, best.indices).float()  # noqa: E731
    out = {
        "sub_ids": sub_ids.cpu(),
        "sub_err": sub_err.cpu().to(torch.float16),
        "sub_err_mean": gather(mean).cpu().to(torch.float16),
        "sub_cos": gather(cos).cpu().to(torch.float16),
        "norm": norm.float().cpu(),
        "count": cnt.to(torch.int32).cpu(),
    }
    # y_cal: median(||y||_true / sqrt(sum_k w_k^2 n_k^2)) with the finished norms
    ratios = []
    norm_c = norm.float().cpu()
    for y_true, R, Wt in cal_ratio:
        n_sel = norm_c[R]                           # [C, K]
        yhat = (Wt.pow(2) * n_sel.pow(2)).sum(-1).clamp_min(1e-12).sqrt()
        ratios.append(y_true / yhat)
    out["y_cal"] = (torch.cat(ratios).median() if ratios else torch.tensor(1.0)).reshape(1).float()
    out["_diag"] = p75.diagonal().abs().max().reshape(1).float().cpu()
    return out


def build_table(model: str, dumps: str, out_path: str, *, layers=None, num_experts=512,
                top_k=10, m=DEFAULT_M, chunk=96, min_tokens=256, resume=True) -> str:
    """Build the whole table and write it atomically. Per-layer partials make it resumable."""
    from safetensors.torch import save_file

    dev = torch.device("cuda")
    idx = json.load(open(f"{model}/model.safetensors.index.json"))["weight_map"]
    files = dump_files(dumps)
    if not files:
        raise SystemExit(f"no route_*.pt under {dumps}")
    if layers is None:
        # model.language_model.layers.<N>.mlp.experts.0.gate_proj.weight -> field 3
        layers = sorted(
            int(k.split(".")[3]) for k in idx
            if k.startswith("model.language_model.layers.")
            and k.endswith(".mlp.experts.0.gate_proj.weight")
        )
    work = out_path + ".partial"
    os.makedirs(work, exist_ok=True)
    for layer in layers:
        part = os.path.join(work, f"layer{layer}.pt")
        if resume and os.path.exists(part):
            logger.info_rank0(f"layer {layer}: already built, skipping")
            continue
        t0 = time.perf_counter()
        row = build_layer(model, idx, layer, files, num_experts=num_experts, top_k=top_k, m=m,
                          chunk=chunk, device=dev, min_tokens=min_tokens)
        torch.save(row, part)
        logger.info_rank0(
            f"layer {layer}: {time.perf_counter() - t0:.0f}s, tokens/expert "
            f"min {int(row['count'].min())} median {int(row['count'].median())}, "
            f"under-observed {int((row['count'] < MIN_COUNT).sum())}/{num_experts}, "
            f"best-substitute err p50 {row['sub_err'][:, 0].float().median():.3f}, "
            f"diag {float(row['_diag']):.1e}"
        )

    rows = [torch.load(os.path.join(work, f"layer{l}.pt"), weights_only=False) for l in layers]
    tensors = {
        k: torch.stack([r[k] for r in rows])
        for k in ("sub_ids", "sub_err", "sub_err_mean", "sub_cos", "norm", "count")
    }
    tensors["y_cal"] = torch.cat([r["y_cal"] for r in rows])
    meta = {
        "freetoken_subst_version": FORMAT_VERSION,
        "fingerprint": _fingerprint(model),
        "source_model_path": model,
        "num_layers": str(len(layers)), "num_experts": str(num_experts),
        "top_k": str(top_k), "shortlist_m": str(m), "statistic": "p75-gaussian",
        "corpus_files": str(len(files)),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    tmp = out_path + ".tmp"
    save_file(tensors, tmp, metadata=meta)
    os.replace(tmp, out_path)
    logger.info_rank0(f"wrote {out_path} ({os.path.getsize(out_path) / 2**20:.1f} MiB)")
    return out_path


def _fingerprint(model: str) -> str:
    """Identity of the checkpoint this table was built from.

    Deliberately not ``checkpoint.convert._source_fingerprint``: that one also folds in the quant
    format and the CUDA compute capability, which it needs because it describes a *converted*
    artifact. This table depends on the source weights and nothing else, and must be loadable
    without a GPU.
    """
    import hashlib
    h = hashlib.sha256()
    for f in sorted(glob.glob(os.path.join(model, "*.safetensors"))):
        st = os.stat(f)
        h.update(f"{os.path.basename(f)}:{st.st_size}|".encode())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------------------------
# load / report
# ---------------------------------------------------------------------------------------------

def default_paths(model: str, ftw_dir: str | None) -> list[str]:
    out = []
    if ftw_dir:
        out.append(os.path.join(ftw_dir, FILENAME))
    cache = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    out.append(os.path.join(cache, "freetoken", "subst", f"{_fingerprint(model)}.safetensors"))
    return out


def load_table(path: str, *, model: str, num_layers: int, num_experts: int, top_k: int,
               device=None):
    """Load and validate. Warn-and-ignore on any mismatch, as the bench profile does: a
    half-trusted table would silently misprice the perturbation budget.

    ``num_layers`` is the number of layers the table must COVER -- the paged tower, not the
    cache's layer count, which may include appended resident bank layers the table has no rows
    for and never needs any.
    """
    from safetensors import safe_open

    try:
        with safe_open(path, framework="pt") as f:
            meta = f.metadata() or {}
            t = {k: f.get_tensor(k) for k in f.keys()}
    except Exception as e:  # noqa: BLE001
        logger.warning(f"expert substitution table {path}: unreadable ({e!r}); ignoring")
        return None
    if meta.get("freetoken_subst_version") != FORMAT_VERSION:
        logger.warning(f"{path}: version {meta.get('freetoken_subst_version')!r} is not "
                       f"{FORMAT_VERSION!r}; ignoring")
        return None
    fp = _fingerprint(model)
    if meta.get("fingerprint") != fp:
        logger.warning(f"{path}: built from a different checkpoint "
                       f"({meta.get('fingerprint')} != {fp}); ignoring")
        return None
    if (int(meta.get("num_experts", 0)) != num_experts or int(meta.get("top_k", 0)) != top_k):
        logger.warning(f"{path}: geometry mismatch (experts/top_k); ignoring")
        return None
    if t["sub_ids"].shape[0] < num_layers:
        logger.warning(f"{path}: covers {t['sub_ids'].shape[0]} layers, but {num_layers} paged "
                       "layers need one; ignoring")
        return None
    if device is not None:
        t = {k: v.to(device) for k, v in t.items()}
    t["_meta"] = meta
    return t


def report(path: str) -> None:
    from safetensors import safe_open
    with safe_open(path, framework="pt") as f:
        meta = f.metadata() or {}
        t = {k: f.get_tensor(k) for k in f.keys()}
    print(json.dumps(meta, indent=2))
    err, cos, cnt = t["sub_err"].float(), t["sub_cos"].float(), t["count"]
    L = err.shape[0]
    print(f"\n{'layer':>6} {'tok/expert min':>15} {'median':>8} {'best err p50':>13} "
          f"{'best cos p50':>13} {'cos>0.44':>9}")
    for l in range(L):
        live = cnt[l] >= 64
        if not live.any():
            continue
        b_err, b_cos = err[l][live, 0], cos[l][live, 0]
        print(f"{l:>6} {int(cnt[l].min()):>15} {int(cnt[l].median()):>8} "
              f"{b_err.median():>13.3f} {b_cos.median():>13.3f} "
              f"{int((cos[l][live] > 0.44).any(dim=1).sum()):>9}")
    print("\n(cos > 0.44 is the threshold above which substituting beats dropping; the count is "
          "experts with at least one such candidate anywhere in their shortlist)")


__all__ = ["build_table", "load_table", "report", "default_paths", "load_layer_experts",
           "experts_on", "dump_files", "iter_layer", "FILENAME"]
