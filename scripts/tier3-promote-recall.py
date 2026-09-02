"""Can a cross-layer predictor name the experts that MISS the host pool?

That is the whole question for tier3->tier2 speculative pre-transfer. The known result is that
token-conditioned expert prediction runs ~4% precision because it concentrates on the head,
while the misses are the cold tail. The counter-argument is that a WIDE prediction (top-32)
filtered to non-resident turns a low-precision predictor into a low-volume high-value one.

Method, all offline on real route dumps:
  * predict layer J+N's router logits from layer J's residual, using layer J+N's own gate
    (the xnorm/uniform proxy: mean over the hyper-connection streams -> rmsnorm -> gate)
  * replay the real routes through an LRU host pool of `host_slots` to know, at every
    (step, layer), which experts are actually resident
  * for every host MISS at layer J+N, ask whether it was in the top-K predicted at layer J
"""
import glob, json, sys, torch
from safetensors import safe_open

MODEL = "/media/nuko/New Volume/Qwen3.8-Flash-Next-NVFP4"
DUMPS = "/media/nuko/New Volume/route-dumps-m1"
HOST_SLOTS, EXPERTS, TOPK_ROUTE, N_AHEAD = 352, 512, 10, 2
K_LIST = (4, 16, 32, 64)

idx = json.load(open(f"{MODEL}/model.safetensors.index.json"))["weight_map"]
def load(name):
    with safe_open(f"{MODEL}/{idx[name]}", framework="pt") as f:
        return f.get_tensor(name)

files = sorted(glob.glob(f"{DUMPS}/route_*.pt"))
d0 = torch.load(files[0], map_location="cpu", weights_only=True)
layer_ids = sorted(d0["layers"])
gates = {l: load(f"model.language_model.layers.{l}.mlp.gate.weight").float() for l in layer_ids}
print(f"{len(layer_ids)} MoE layers, gate {tuple(gates[layer_ids[0]].shape)}, "
      f"host_slots={HOST_SLOTS}, N={N_AHEAD}")

hit = {k: 0 for k in K_LIST}
# cost side: how many of the top-K predictions are NOT host-resident, i.e. how many extra
# NVMe reads a residency-filtered promoter would issue. Bandwidth is the binding constraint.
cand = {k: 0 for k in K_LIST}
miss_total = 0
n_tok = 0
resident = {l: set() for l in layer_ids}      # LRU pool per layer
recency = {l: {} for l in layer_ids}
clock = 0

for path in files:
    d = torch.load(path, map_location="cpu", weights_only=True)
    L = d["layers"]
    if not L:
        continue
    T = L[layer_ids[0]]["logits"].shape[0]
    actual = {l: torch.topk(L[l]["logits"].float(), TOPK_ROUTE, -1).indices for l in layer_ids}
    # predicted logits for layer l from layer (l - N)'s residual
    pred = {}
    for i, l in enumerate(layer_ids):
        j = layer_ids[i - N_AHEAD] if i >= N_AHEAD else None
        if j is None:
            continue
        r = L[j]["r"].float()                      # [T, streams * hidden]
        h = r.view(T, -1, gates[l].shape[1]).mean(1)
        h = h / (h.pow(2).mean(-1, keepdim=True) + 1e-6).sqrt()
        pred[l] = torch.topk(h @ gates[l].T, max(K_LIST), -1).indices
    for t in range(T):
        clock += 1
        n_tok += 1
        for i, l in enumerate(layer_ids):
            if l in pred:
                p = pred[l][t].tolist()
                res = resident[l]
                for k in K_LIST:
                    cand[k] += sum(1 for e in p[:k] if e not in res)
            routed = actual[l][t].tolist()
            for e in routed:
                if e not in resident[l]:
                    miss_total += 1
                    if l in pred:
                        p = pred[l][t]
                        for k in K_LIST:
                            if (p[:k] == e).any():
                                hit[k] += 1
                    if len(resident[l]) >= HOST_SLOTS:
                        victim = min(resident[l], key=lambda x: recency[l][x])
                        resident[l].discard(victim); recency[l].pop(victim, None)
                    resident[l].add(e)
                recency[l][e] = clock
    del d

print(f"\nhost misses seen: {miss_total}")
misses_per_tok = miss_total / max(1, n_tok)
print(f"per token: {misses_per_tok:.1f} host misses "
      f"({misses_per_tok * 2.64:.1f} MiB of demand reads)")
print(f"\n{'top-K':>6} {'recall':>8} {'promotions/token':>18} {'promo MiB/token':>16} "
      f"{'useful':>8}")
for k in K_LIST:
    c = cand[k] / max(1, n_tok)
    print(f"{k:>6} {hit[k]/max(1,miss_total):>8.1%} {c:>18.1f} {c * 2.64:>16.1f} "
          f"{hit[k]/max(1,cand[k]):>8.1%}")
print("\n(recall = of the experts that actually had to come off NVMe, the fraction a top-K\n"
      " prediction made 2 layers earlier would have named -- i.e. the fraction of the 7.4 ms\n"
      " stall that speculative pre-transfer could even attempt to hide)")
