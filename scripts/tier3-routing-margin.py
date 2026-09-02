"""How much does the router actually care about the experts we pay NVMe for?

If host misses are concentrated in the low-weight tail of the top-k, then a residency-aware
router -- prefer a near-tied RESIDENT expert over a non-resident one -- buys back disk reads for
a bounded perturbation. This measures the two quantities that decide it, on real route dumps:

  1. the softmax weight (within the top-k, as the MoE normalises it) of the experts that miss
  2. the weight gap to the best resident expert not already selected, i.e. what a substitution
     would actually cost
"""
import glob, torch

DUMPS = "/media/nuko/New Volume/route-dumps-m1"
HOST_SLOTS, TOPK = 352, 10

files = sorted(glob.glob(f"{DUMPS}/route_*.pt"))
layer_ids = sorted(torch.load(files[0], map_location="cpu", weights_only=True)["layers"])
resident = {l: set() for l in layer_ids}
recency = {l: {} for l in layer_ids}
clock = 0

miss_w, miss_rank, sub_cost, no_sub = [], [], [], 0
all_w = []

for path in files:
    L = torch.load(path, map_location="cpu", weights_only=True)["layers"]
    if not L:
        continue
    T = L[layer_ids[0]]["logits"].shape[0]
    for l in layer_ids:
        lg = L[l]["logits"].float()
        top = torch.topk(lg, TOPK, -1)
        w = torch.softmax(top.values, -1)          # the weights the MoE actually applies
        idx = top.indices
        for t in range(T):
            ids_t, w_t = idx[t].tolist(), w[t].tolist()
            all_w.extend(w_t)
            res = resident[l]
            for rank, (e, we) in enumerate(zip(ids_t, w_t)):
                if e not in res:
                    miss_w.append(we)
                    miss_rank.append(rank)
                    # cheapest resident stand-in: highest-logit resident expert not already
                    # in this token's top-k. Cost ~ the weight we would move onto it.
                    cand = [(lg[t, c].item(), c) for c in res if c not in ids_t]
                    if cand:
                        best = max(cand)[0]
                        # weight the substitute would carry if it replaced e in the softmax
                        sub_cost.append(we - torch.softmax(
                            torch.tensor([best] + [lg[t, c].item() for c in ids_t if c != e]),
                            -1)[0].item())
                    else:
                        no_sub += 1
                    if len(res) >= HOST_SLOTS:
                        v = min(res, key=lambda x: recency[l][x])
                        res.discard(v); recency[l].pop(v, None)
                    res.add(e)
                recency[l][e] = clock
                clock += 1

mw = torch.tensor(miss_w); aw = torch.tensor(all_w); mr = torch.tensor(miss_rank, dtype=torch.float)
sc = torch.tensor(sub_cost)
print(f"routed slots seen: {len(all_w)},  host misses: {len(miss_w)} "
      f"({100*len(miss_w)/len(all_w):.2f}% of routed slots)")
print(f"\nmean gate weight   all routed: {aw.mean():.4f}   MISSED: {mw.mean():.4f}"
      f"   ratio {mw.mean()/aw.mean():.2f}x")
print(f"mean rank within top-{TOPK}   all: {(TOPK-1)/2:.1f}   MISSED: {mr.mean():.2f}")
print(f"\nmissed-expert weight percentiles: "
      + "  ".join(f"p{p}={torch.quantile(mw, p/100):.4f}" for p in (10, 50, 90)))
print(f"share of MISSES whose gate weight is below the mean routed weight: "
      f"{(mw < aw.mean()).float().mean():.1%}")
print(f"\nsubstitution cost (weight displaced onto a resident stand-in):")
print(f"  mean {sc.mean():.4f}   "
      + "  ".join(f"p{p}={torch.quantile(sc, p/100):.4f}" for p in (50, 90, 99)))
print(f"  misses with a resident substitute available: {100*(1-no_sub/max(1,len(miss_w))):.1f}%")
print(f"\nmargin gate: substitute only when the displaced weight is <= tau")
print(f"{'tau':>7} {'misses avoided':>15} {'NVMe reads/token':>17} {'weight displaced':>18}")
n_tokens = len(all_w) / (TOPK * len(layer_ids))
base = len(miss_w) / n_tokens                 # host misses per TOKEN (all layers)
for tau in (0.005, 0.01, 0.02, 0.05, 0.10):
    cov = (sc <= tau).float().mean().item()
    left = base * (1 - cov)
    # aggregate: fraction of ALL routed weight that moves to a different expert
    pert = (sc[sc <= tau].sum().item() / max(1, len(all_w)))
    print(f"{tau:>7.3f} {cov:>14.1%} {left:>17.2f} {pert:>17.2e}")
