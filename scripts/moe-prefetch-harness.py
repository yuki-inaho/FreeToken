#!/usr/bin/env python
"""Offline evaluation of speculative MoE expert-prefetch predictors (M1 of the prefetch plan).

Inputs are routing-truth dumps captured by serving natural prompts with
``FREETOKEN_ROUTE_DUMP=<dir>`` (per-layer router logits; add ``FREETOKEN_ROUTE_DUMP_R=1``
for the pre-mlp-mix residual R and the mixed block input) and, for the MTP-based
predictors, a premix dump from the same run (``FREETOKEN_MTP_DEBUG_DUMP=<dir>``).
Teacher-forced prefill routing is identical to decode routing over the same text, so the
dumps are exact decode ground truth.

The harness replays the token stream through an offline LRU cache simulator shaped like
the serving expert cache (unified slot pool, timestamp LRU) to obtain per-step MISS sets
-- the only experts a prefetcher can help with -- then scores candidate predictors by how
much of each step's miss set their top-``budget`` NON-RESIDENT candidates cover:

    marginal recall   = |top-B(pred) \\cap miss| / |miss|      (the number that becomes tok/s)
    precision-on-miss = |top-B(pred) \\cap miss| / B           (1 - waste fraction)
    useful MB/step    = hits * bytes_per_expert

Predictors (availability depends on what was dumped):
    prev        route(i) := route(i-1) per layer (floor; LRU already retains most of it)
    table       token-identity table fit on the first half of the steps, tested on the rest
    embed       gate_L(rmsnorm(embed(tok_i)))                  future-token, zero state
    r1          gate_L(uniform-mix(r1_{i-1}))                  future-token via the MTP head
    r1mix       gate_L(true mix_L(r1_{i-1}))                   upper bound for r1's uniform mix
    blend       per-layer better of embed/r1 chosen on the first half, tested on the rest
    xlayerN     gate_L(uniform-mix(R_{L-N}[i]))  N in {1,2,4,8}  in-graph lookahead (M3 gate)
    xmixN       gate_L(true mix_L(R_{L-N}[i]))                 upper bound for xlayerN
    oracle      the truth itself (must score marginal recall 1.0 -- self-check)

The evaluation is conservative: predictor hits do not mutate the simulator's cache state,
so reported marginal recall is against the unpolluted demand-only baseline.

Usage:
    python scripts/moe-prefetch-harness.py --dump-dir /tmp/route [--premix-dir /tmp/route]
        [--mtp] [--slots 5060] [--budgets 8 16 24 32 48 64 96 128] [--verify-rows 3]
        [--csv /tmp/prefetch-curves.csv]
"""

from __future__ import annotations

import argparse
import glob
import heapq
import json
import os
import sys
from collections import defaultdict

os.environ.setdefault("FREETOKEN_LOAD_MTP", "1")

import torch

TOP_K = 10  # experts per token per layer (num_experts_per_tok)


def targeted_state_dict(model_path: str, keys: set[str], device: torch.device) -> dict:
    """Load exactly ``keys`` (raw checkpoint names) through the production rename+fusion
    path (same helper as scripts/mtp-wiring-harness.py)."""
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


def load_route_chunks(dump_dir: str, min_tokens: int) -> list[dict]:
    """Every route dump with at least ``min_tokens`` rows (warmup/boot forwards are tiny),
    in capture order. Each chunk is one contiguous prefill forward."""
    files = sorted(glob.glob(os.path.join(dump_dir, "route_*.pt")))
    if not files:
        raise SystemExit(f"no route_*.pt files in {dump_dir} (serve with FREETOKEN_ROUTE_DUMP)")
    chunks = []
    for path in files:
        data = torch.load(path, map_location="cpu", weights_only=True)
        if data["input_ids"].shape[0] >= min_tokens:
            data["path"] = path
            chunks.append(data)
    if not chunks:
        raise SystemExit(f"no route dump with >= {min_tokens} tokens in {dump_dir}")
    total = sum(c["input_ids"].shape[0] for c in chunks)
    print(f"route dumps: {len(chunks)}/{len(files)} usable chunks, {total} tokens total")
    return chunks


def uniform_mix(r: torch.Tensor, hc: int, eps: float) -> torch.Tensor:
    """The cheap stand-in for GatedResidual.mix: per-stream fp32 RMS-norm, mean over the
    streams. Drops the (1+w) norm weights and the sigmoid stream gates -- a uniform
    positive rescale keeps the router's top-k SET invariant, so only the per-stream
    reweighting is approximated (quantified by the r1mix/xmix upper-bound predictors)."""
    xf = r.float().unflatten(-1, (hc, r.shape[-1] // hc))
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return xf.mean(-2)


class LRUSim:
    """Timestamp-LRU expert cache over the flat (layer, expert) id space, matching the
    serving cache's semantics (unified pool, global argmin eviction, per-call clock)."""

    def __init__(self, slots: int) -> None:
        self.slots = slots
        self.last: dict[int, int] = {}  # flat id -> last-used clock
        self.heap: list[tuple[int, int]] = []  # (clock, id), lazy deletion
        self.clock = 0
        self.hits = 0
        self.misses = 0

    def access(self, ids: list[int]) -> list[int]:
        """Touch one step-layer's active set; returns the ids that missed."""
        self.clock += 1
        missed = []
        for i in ids:
            if i in self.last:
                self.hits += 1
            else:
                self.misses += 1
                missed.append(i)
                while len(self.last) >= self.slots:
                    ts, victim = heapq.heappop(self.heap)
                    if self.last.get(victim) == ts:
                        del self.last[victim]
            self.last[i] = self.clock
            heapq.heappush(self.heap, (self.clock, i))
        return missed


def evaluate(
    name: str,
    scores_per_step: list[torch.Tensor | None],
    miss_sets: list[torch.Tensor],
    resident_masks: list[torch.Tensor],
    eligible: list[bool],
    budgets: list[int],
    bytes_per_expert: float,
    eval_range: range | None = None,
) -> dict[int, tuple[float, float, float]]:
    """Score one predictor: per step, take the top-B non-resident candidates from its
    [L, E] score map and intersect with that step's miss set. Returns
    {budget: (marginal_recall, precision_on_miss, useful_MB_per_step)}."""
    bmax = max(budgets)
    hit_at = {b: 0 for b in budgets}
    miss_total = 0
    steps = 0
    for i in eval_range if eval_range is not None else range(len(miss_sets)):
        s = scores_per_step[i]
        miss = miss_sets[i]
        if s is None or miss is None or not eligible[i]:
            continue
        steps += 1
        miss_total += int(miss.sum())
        masked = s.masked_fill(resident_masks[i], float("-inf"))
        top = torch.topk(masked.reshape(-1), bmax).indices
        in_miss = miss.reshape(-1)[top]
        cum = torch.cumsum(in_miss.int(), 0)
        for b in budgets:
            hit_at[b] += int(cum[b - 1])
    out = {}
    for b in budgets:
        h = hit_at[b]
        out[b] = (
            h / miss_total if miss_total else 0.0,
            h / (b * steps) if steps else 0.0,
            h / steps * bytes_per_expert / (1 << 20) if steps else 0.0,
        )
    return out


def main() -> None:  # noqa: PLR0915 -- linear analysis script
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/media/nuko/New Volume/Qwen3.8-Flash-Next-NVFP4")
    parser.add_argument("--dump-dir", required=True, help="FREETOKEN_ROUTE_DUMP directory")
    parser.add_argument("--premix-dir", default=None, help="FREETOKEN_MTP_DEBUG_DUMP directory (default: --dump-dir)")
    parser.add_argument("--mtp", action="store_true", help="evaluate the r1 (MTP-head) predictors -- loads the full head")
    parser.add_argument("--slots", type=int, default=5060, help="simulated expert-cache slots")
    parser.add_argument("--num-layers", type=int, default=48)
    parser.add_argument("--budgets", nargs="*", type=int, default=[8, 16, 24, 32, 48, 64, 96, 128])
    parser.add_argument("--verify-rows", type=int, default=3, help="tokens per simulated decode step (k=2 MTP verifies 3 rows)")
    parser.add_argument("--verify-stride", type=int, default=2, help="tokens the stream advances per step (~accepted/step)")
    parser.add_argument("--lookaheads", nargs="*", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--per-layer", action="store_true", help="per-layer miss density + xmix coverage table (in-graph design input)")
    parser.add_argument("--layer-budgets", nargs="*", type=int, default=[2, 4, 8], help="per-layer admission budgets for --per-layer")
    parser.add_argument("--min-tokens", type=int, default=256, help="ignore route dumps smaller than this (warmup)")
    parser.add_argument("--max-tokens", type=int, default=2000, help="per-chunk row cap (dense-QSA exactness bound for the MTP head)")
    parser.add_argument("--skip", type=int, default=32, help="steps to skip at each chunk's front")
    parser.add_argument("--bytes-per-expert", type=float, default=2772480.0, help="NVFP4 bank bytes per expert-layer")
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    L = args.num_layers

    from transformers import AutoConfig

    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.models.qwen4_exp.config import parse_config

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    config = parse_config(AutoConfig.from_pretrained(args.model_path, local_files_only=True))
    hc, eps = config.qwen4_args.hc_count, config.rms_norm_eps

    chunks = load_route_chunks(args.dump_dir, args.min_tokens)
    have_r = all("r" in c["layers"][0] for c in chunks)

    # ---- weights: 48 router gates (stacked), the embedding, per-layer mlp mixes -------------
    lm = "model.language_model"
    gate_keys = {f"{lm}.layers.{i}.mlp.gate.weight" for i in range(L)}
    mix_keys = {
        f"{lm}.layers.{i}.mlp_hyper_connection.{part}.weight"
        for i in range(L)
        for part in ("hc_norm", "input_mix_weight_down", "input_mix_weight_up", "block_inject_weight")
    }
    print("loading gates / embedding / mlp mixes...")
    sd = targeted_state_dict(
        args.model_path, gate_keys | mix_keys | {f"{lm}.embed_tokens.weight"}, device
    )
    gates = torch.stack([sd[f"model.layers.{i}.mlp.gate.weight"] for i in range(L)]).to(dtype)
    embed_weight = sd["model.embed_tokens.weight"].to(dtype)  # [V, H]

    from freetoken.models.qwen4_exp.hc import GatedResidual
    from freetoken.utils.torch_utils import torch_dtype

    mixes = []
    for i in range(L):
        with torch.device(device), torch_dtype(dtype):
            m = GatedResidual(config)
        m.load_state_dict(
            {
                k.split("mlp_hyper_connection.", 1)[1]: v
                for k, v in sd.items()
                if f"model.layers.{i}.mlp_hyper_connection." in k
            }
        )
        mixes.append(m)

    # ---- optional: teacher-forced MTP-head r1 rows over each chunk's premix ------------------
    r1_by_chunk: dict[int, torch.Tensor] = {}
    if args.mtp:
        os.environ.setdefault("FREETOKEN_MTP_EXPERTS", "bf16")
        from types import SimpleNamespace

        from freetoken.core import Context, set_global_ctx
        from freetoken.models.qwen4_exp.attention import TorchDenseQSAReference
        from freetoken.models.qwen4_exp.mtp import Qwen4ExpMTPHead
        from freetoken.moe.fused import FusedMoe

        premix_dir = args.premix_dir or args.dump_dir
        premix_files = sorted(glob.glob(os.path.join(premix_dir, "premix_*.pt")))
        if not premix_files:
            raise SystemExit(f"--mtp needs premix_*.pt in {premix_dir} (FREETOKEN_MTP_DEBUG_DUMP)")
        with open(os.path.join(args.model_path, "model.safetensors.index.json"), encoding="utf-8") as fh:
            mtp_raw = {k for k in json.load(fh)["weight_map"] if k.startswith("mtp.")}
        print(f"loading {len(mtp_raw)} mtp tensors (bf16 resident experts)...")
        mtp_sd = targeted_state_dict(args.model_path, mtp_raw, device)
        with torch.device(device), torch_dtype(dtype):
            head = Qwen4ExpMTPHead(config, wiring="per_stream")
        head.load_state_dict(dict(mtp_sd), prefix="mtp")
        del mtp_sd

        premix_by_key = {}
        for path in premix_files:
            d = torch.load(path, map_location="cpu", weights_only=True)
            if d["input_ids"].shape[0] >= args.min_tokens:
                key = (d["input_ids"].shape[0], int(d["positions"][0]))
                premix_by_key[key] = d["premix"]

        import freetoken.core as core

        def run_mtp_layer(merged: torch.Tensor) -> torch.Tensor:
            n = merged.shape[0]
            core._GLOBAL_CTX = None
            ctx = Context(page_size=64)
            ctx.attn_backend = TorchDenseQSAReference(
                config, num_slots=1, max_len=n + 8, device=device, dtype=dtype
            )
            ctx.moe_backend = FusedMoe()
            set_global_ctx(ctx)
            batch = SimpleNamespace(
                positions=torch.arange(n, dtype=torch.int32, device=device),
                padded_reqs=[SimpleNamespace(extend_len=n, table_idx=0, cached_len=0)],
                is_prefill=True,
                is_decode=False,
            )
            return head.layers.op_list[0].forward(merged, batch)

        for ci, chunk in enumerate(chunks):
            ids = chunk["input_ids"]
            key = (ids.shape[0], int(chunk["positions"][0]))
            premix = premix_by_key.get(key)
            if premix is None:
                print(f"  [mtp] no premix match for chunk {ci} {key}; r1 predictors skip it")
                continue
            t = min(ids.shape[0], args.max_tokens)
            rows = t - 1  # teacher rows need the TRUE next token, so the last row drops
            with torch.no_grad():
                merged = head.build_input(
                    premix[:rows].to(device, dtype),
                    embed_weight[ids[1 : rows + 1].long().to(device)],
                )
                # r1[i] is the head's 4-stream output given R_i and the true token t_{i+1}
                # -- exactly serving's r1_rows semantics (the premix proxy for row i+1).
                r1_by_chunk[ci] = run_mtp_layer(merged)
        print(f"  [mtp] r1 rows computed for {len(r1_by_chunk)}/{len(chunks)} chunks")

    # ---- per-chunk truth, LRU baseline, per-step score maps ----------------------------------
    rows_per_step, stride = args.verify_rows, args.verify_stride
    sim = LRUSim(args.slots)
    unreachable_misses = 0
    miss_sets: list[torch.Tensor] = []  # [L, E] bool per step (reachable at prediction time)
    resident_masks: list[torch.Tensor] = []
    eligible: list[bool] = []  # False for each chunk's first --skip steps (LRU warm-up)
    step_scores: dict[str, list[torch.Tensor | None]] = defaultdict(list)
    step_tokens: list[list[int]] = []  # tokens of each step's rows (for the table predictor)
    truth_by_step: list[torch.Tensor] = []  # [rows, L, TOP_K] expert ids
    E = None

    xlayer_names = [f"xlayer{n}" for n in args.lookaheads] + [f"xmix{n}" for n in args.lookaheads]

    for ci, chunk in enumerate(chunks):
        ids = chunk["input_ids"]
        t = min(ids.shape[0], args.max_tokens)
        logits = torch.stack(
            [chunk["layers"][i]["logits"][:t] for i in range(L)], dim=1
        )  # [t, L, E] bf16 cpu
        E = logits.shape[-1]
        truth = torch.topk(logits.float(), TOP_K, dim=-1).indices  # [t, L, TOP_K]

        # sanity: recomputed gate logits vs the dumped block input (when R was dumped)
        if ci == 0 and have_r:
            bi = chunk["layers"][0]["block_input"][:t].to(device, dtype)
            re_logits = (bi @ gates[0].t()).float().cpu()
            overlap = (
                torch.topk(re_logits, TOP_K, -1).indices.unsqueeze(-1)
                == truth[:, 0].unsqueeze(-2)
            ).any(-1).float().mean()
            print(f"sanity: gate(block_input) top-{TOP_K} overlap vs dumped logits = {overlap:.3f}")

        # ---- score maps for every step of this chunk ----
        with torch.no_grad():
            emb_proxy = uniform_mix(
                embed_weight[ids[:t].long().to(device)].repeat(1, hc), hc, eps
            ).to(dtype)  # [t, H] (streams identical -> plain rmsnorm)
            emb_scores = torch.einsum("th,leh->tle", emb_proxy, gates).float().softmax(-1).cpu()

            r_scores = {}
            if have_r:
                r_all = torch.stack(
                    [chunk["layers"][i]["r"][:t] for i in range(L)], dim=1
                )  # [t, L, hc*H] cpu bf16
                for n in args.lookaheads:
                    uni = torch.empty(t, L, E)
                    tru = torch.empty(t, L, E)
                    for dst in range(L):
                        src = max(0, dst - n)
                        r_src = r_all[:, src].to(device, dtype)
                        uni[:, dst] = (
                            (uniform_mix(r_src, hc, eps).to(dtype) @ gates[dst].t())
                            .float().softmax(-1).cpu()
                        )
                        tru[:, dst] = (
                            (mixes[dst].mix(r_src)[0] @ gates[dst].t())
                            .float().softmax(-1).cpu()
                        )
                    r_scores[f"xlayer{n}"] = uni
                    r_scores[f"xmix{n}"] = tru

            r1_scores = r1_mix_scores = None
            if ci in r1_by_chunk:
                r1 = r1_by_chunk[ci].to(device, dtype)  # [t-1, hc*H]
                u = uniform_mix(r1, hc, eps).to(dtype)
                r1_scores = torch.einsum("th,leh->tle", u, gates).float().softmax(-1).cpu()
                mix_rows = torch.stack([mixes[d].mix(r1)[0] for d in range(L)], dim=1)
                r1_mix_scores = (
                    torch.einsum("tlh,leh->tle", mix_rows.to(dtype), gates)
                    .float().softmax(-1).cpu()
                )

        # ---- steps: rows [p, p+rows) advancing by stride; predictors use info from row p-1 --
        p = 1  # row 0 has no predecessor for the r1/prev predictors
        chunk_step = 0
        while p + rows_per_step <= t:
            rows = list(range(p, p + rows_per_step))
            step_tokens.append([int(ids[i]) for i in rows])
            truth_by_step.append(truth[rows])
            eligible.append(chunk_step >= args.skip)
            chunk_step += 1

            resident = torch.zeros(L, E, dtype=torch.bool)
            for flat in sim.last:
                resident[flat // E, flat % E] = True
            resident_masks.append(resident)

            miss = torch.zeros(L, E, dtype=torch.bool)
            for layer in range(L):
                active = set()
                for r_i in rows:
                    active.update(truth[r_i, layer].tolist())
                for e in sim.access(sorted(layer * E + x for x in active)):
                    miss[layer, e % E] = True
            # An expert resident at prediction time but evicted by this step's own earlier
            # layers is unreachable for an admission-time prefetcher (the admission kernel
            # skips residents); evaluate against the reachable misses and report the rest.
            unreachable = miss & resident
            unreachable_misses += int(unreachable.sum())
            miss_sets.append(miss & ~resident)

            # row confidence mirrors serving: the step's later rows are deeper drafts
            conf = [1.0, 0.85, 0.6] + [0.5] * max(0, rows_per_step - 3)
            emb_s = torch.zeros(L, E)
            for j, r_i in enumerate(rows):
                emb_s = torch.maximum(emb_s, conf[j] * emb_scores[r_i])
            step_scores["embed"].append(emb_s)

            if r1_scores is not None and p - 1 < r1_scores.shape[0]:
                # serving has r1 for the step's FIRST row only (row p predicted from row
                # p-1's head output); later rows fall back to their embed proxy.
                step_scores["r1"].append(torch.maximum(conf[0] * r1_scores[p - 1], emb_s))
                step_scores["r1mix"].append(torch.maximum(conf[0] * r1_mix_scores[p - 1], emb_s))
            else:
                step_scores["r1"].append(None)
                step_scores["r1mix"].append(None)

            for name in xlayer_names:
                if have_r:
                    s = torch.zeros(L, E)
                    for j, r_i in enumerate(rows):
                        s = torch.maximum(s, conf[j] * r_scores[name][r_i])
                    step_scores[name].append(s)
                else:
                    step_scores[name].append(None)

            prev = torch.zeros(L, E)
            for layer in range(L):
                prev[layer, truth[p - 1, layer]] = 1.0
            step_scores["prev"].append(prev)

            oracle = torch.zeros(L, E)
            for j, r_i in enumerate(rows):
                for layer in range(L):
                    oracle[layer, truth[r_i, layer]] = 1.0
            step_scores["oracle"].append(oracle)
            p += stride

    n_steps = len(miss_sets)
    n_eligible = sum(eligible)
    miss_per_step = sum(
        int(m.sum()) for m, e in zip(miss_sets, eligible) if e
    ) / max(1, n_eligible)
    hit_rate = sim.hits / max(1, sim.hits + sim.misses)
    print(
        f"\nLRU baseline ({args.slots} slots, {rows_per_step}-row steps, stride {stride}): "
        f"hit_rate={hit_rate:.3f}  reachable miss/step={miss_per_step:.1f} "
        f"(~{miss_per_step * args.bytes_per_expert / (1 << 20):.0f} MB/step)"
        f"  unreachable (same-step evictions): {unreachable_misses / max(1, n_steps):.2f}/step"
        f"  [live decode_miss_stats should be within a few points]"
    )

    # ---- token-identity table: fit first half, test second half ------------------------------
    half = n_steps // 2
    counts: dict[tuple[int, int], torch.Tensor] = {}
    for i in range(half):
        for j, tok in enumerate(step_tokens[i]):
            for layer in range(L):
                key = (tok, layer)
                if key not in counts:
                    counts[key] = torch.zeros(E)
                counts[key][truth_by_step[i][j, layer]] += 1.0
    for i in range(n_steps):
        if i < half:
            step_scores["table"].append(None)
            continue
        s = torch.zeros(L, E)
        conf = [1.0, 0.85, 0.6] + [0.5] * max(0, rows_per_step - 3)
        for j, tok in enumerate(step_tokens[i]):
            for layer in range(L):
                c = counts.get((tok, layer))
                if c is not None:
                    s[layer] = torch.maximum(s[layer], conf[j] * c / c.sum())
        step_scores["table"].append(s)

    # per-layer blend: pick embed vs r1 per layer on the first half, evaluate on the rest
    if any(s is not None for s in step_scores["r1"]):
        per_layer_wins = torch.zeros(L)
        for i in range(half):
            if step_scores["r1"][i] is None or not eligible[i]:
                continue
            m = miss_sets[i]
            for nm, sign in (("r1", 1.0), ("embed", -1.0)):
                top = torch.topk(step_scores[nm][i], 4, dim=-1).indices  # per-layer top-4
                hits = torch.gather(m, 1, top).sum(1).float()
                per_layer_wins += sign * hits
        r1_layers = per_layer_wins > 0
        print(
            f"blend: r1 wins layers {r1_layers.nonzero().flatten().tolist()}"
            f" (embed wins the rest) -- crossover profile for ExpertPrefetcher.late_w"
        )
        for i in range(n_steps):
            r1_s, emb_s = step_scores["r1"][i], step_scores["embed"][i]
            if r1_s is None:
                step_scores["blend"].append(None)
            else:
                step_scores["blend"].append(torch.where(r1_layers[:, None], r1_s, emb_s))

    # ---- results -----------------------------------------------------------------------------
    order = ["oracle", "prev", "table", "embed", "r1", "r1mix", "blend"] + xlayer_names
    rows_out = []
    print(f"\n{'predictor':<10} " + " ".join(f"B={b:<4d}" for b in args.budgets) + "   (marginal recall | precision-on-miss | useful MB/step)")
    for name in order:
        scores = step_scores.get(name)
        if not scores or all(s is None for s in scores):
            continue
        eval_range = range(half, n_steps) if name in ("table", "blend") else None
        res = evaluate(
            name, scores, miss_sets, resident_masks, eligible, args.budgets,
            args.bytes_per_expert, eval_range,
        )
        for metric_i, label in ((0, "recall"), (1, "prec"), (2, "MB")):
            vals = " ".join(
                (f"{res[b][metric_i]:.3f}" if metric_i < 2 else f"{res[b][metric_i]:5.1f}").ljust(6)
                for b in args.budgets
            )
            print(f"{name if metric_i == 0 else '':<10} {vals}   {label}")
        for b in args.budgets:
            rows_out.append((name, b, *res[b]))
        print()

    if args.csv:
        with open(args.csv, "w", encoding="utf-8") as fh:
            fh.write("predictor,budget,marginal_recall,precision_on_miss,useful_mb_per_step\n")
            for r in rows_out:
                fh.write(",".join(str(x) for x in r) + "\n")
        print(f"curves written to {args.csv}")

    # ---- per-layer breakdown for the IN-GRAPH lookahead design (--per-layer) -----------------
    # For each layer: its share of the miss traffic, and how much of it the cross-layer
    # proxy covers at a PER-LAYER admission budget b (what the in-graph hook would issue).
    # This decides WHICH layers to hook (the predictor cost is per hooked layer) and N.
    if args.per_layer and have_r:
        keep = [i for i in range(n_steps) if eligible[i]]
        miss_stack = torch.stack([miss_sets[i] for i in keep])  # [S, L, E] bool
        res_stack = torch.stack([resident_masks[i] for i in keep])
        miss_per_layer = miss_stack.sum(dim=(0, 2)).float() / len(keep)  # [L]
        per_layer_cov = {}
        for name in [f"xmix{n}" for n in args.lookaheads]:
            scores = torch.stack([step_scores[name][i] for i in keep])  # [S, L, E]
            scores = scores.masked_fill(res_stack, float("-inf"))
            covs = []
            for b in args.layer_budgets:
                top = torch.topk(scores, b, dim=-1).indices  # [S, L, b]
                hits = torch.gather(miss_stack, 2, top).sum(dim=(0, 2)).float()  # [L]
                covs.append(hits / miss_stack.sum(dim=(0, 2)).clamp(min=1).float())
            per_layer_cov[name] = covs
        bcols = " ".join(
            " ".join(f"{name}@{b}" for b in args.layer_budgets)
            for name in per_layer_cov
        )
        print(f"\nper-layer: miss/step, then covered fraction of that layer's misses ({bcols})")
        order_by_miss = torch.argsort(miss_per_layer, descending=True)
        cum = 0.0
        total_miss = float(miss_per_layer.sum())
        for rank, layer in enumerate(order_by_miss.tolist()):
            cum += float(miss_per_layer[layer])
            cells = "  ".join(
                " ".join(f"{float(covs[bi][layer]):.2f}" for bi in range(len(args.layer_budgets)))
                for covs in per_layer_cov.values()
            )
            print(
                f"L{layer:>2}  miss/step={float(miss_per_layer[layer]):5.2f}"
                f"  cum={cum / total_miss:4.0%}  {cells}"
            )
        if args.csv:
            path = args.csv.replace(".csv", "-per-layer.csv")
            with open(path, "w", encoding="utf-8") as fh:
                names = list(per_layer_cov)
                fh.write("layer,miss_per_step," + ",".join(
                    f"{nm}_cov_at_{b}" for nm in names for b in args.layer_budgets) + "\n")
                for layer in range(L):
                    row = [str(layer), f"{float(miss_per_layer[layer]):.3f}"] + [
                        f"{float(per_layer_cov[nm][bi][layer]):.4f}"
                        for nm in names for bi in range(len(args.layer_budgets))
                    ]
                    fh.write(",".join(row) + "\n")
            print(f"per-layer table written to {path}")

    oracle_res = evaluate(
        "oracle", step_scores["oracle"], miss_sets, resident_masks, eligible,
        [max(args.budgets)], args.bytes_per_expert,
    )
    # The oracle can only recover min(B, |miss|) per step; scale the bar by the cap.
    bmax = max(args.budgets)
    bar = 0.95 * min(1.0, bmax / max(1.0, miss_per_step))
    ok = oracle_res[bmax][0] > bar
    if not ok:
        print(
            f"SELF-CHECK FAILED: oracle recall {oracle_res[bmax][0]:.3f} < {bar:.3f} "
            "(the miss sets and score maps disagree)"
        )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
