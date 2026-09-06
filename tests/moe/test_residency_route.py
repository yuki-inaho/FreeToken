"""Residency-aware routing: substitute only when it is cheap, and never silently.

The kernel edits the authoritative routing decision, so the properties that matter are (a) it
leaves resident-only tokens byte-identical, (b) it never substitutes past tau, (c) the weights
it writes are a correct renormalised softmax over the substituted set, and (d) the substitute
is genuinely resident and not already selected.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

E, K = 64, 8


def _setup(resident_ids, seed=0, T=16):
    """Build logits + a top-k routing + a residency map holding only ``resident_ids``."""
    from freetoken.kernel.triton.moe_router import fused_topk_softmax

    torch.manual_seed(seed)
    logits = torch.randn(T, E, device="cuda", dtype=torch.float32)
    w, ids = fused_topk_softmax(logits, K, renormalize=True)
    hmap = torch.full((E,), -1, dtype=torch.int32, device="cuda")
    for slot, e in enumerate(resident_ids):
        hmap[e] = slot
    return logits, w.contiguous(), ids.contiguous(), hmap


def _run(logits, w, ids, hmap, tau):
    from freetoken.kernel.triton.moe_residency_route import residency_route

    stats = torch.zeros(2, dtype=torch.int64, device="cuda")
    residency_route(logits, ids, w, hmap.data_ptr(), 0, E, tau, stats=stats)
    return stats


def test_no_substitution_when_everything_is_resident():
    """The whole feature must vanish when the host pool holds every expert -- that is the
    condition the engine refuses the flag under, and the kernel must agree."""
    logits, w, ids, hmap = _setup(range(E))
    w0, ids0 = w.clone(), ids.clone()
    stats = _run(logits, w, ids, hmap, tau=1.0)
    assert torch.equal(ids, ids0) and torch.equal(w, w0)
    assert stats[0].item() == 0


def test_tau_zero_substitutes_nothing():
    """tau is the contract: at 0 no weight may move, however many experts are missing."""
    logits, w, ids, hmap = _setup(range(0, E, 2))  # half resident
    w0, ids0 = w.clone(), ids.clone()
    stats = _run(logits, w, ids, hmap, tau=0.0)
    assert torch.equal(ids, ids0) and torch.equal(w, w0)
    assert stats[1].item() > 0, "the fixture must actually route to missing experts"


def test_substitute_is_resident_and_not_already_selected():
    logits, w, ids, hmap = _setup(range(0, E, 2), seed=3)
    ids0 = ids.clone()
    _run(logits, w, ids, hmap, tau=1.0)
    changed = (ids != ids0)
    assert changed.any(), "tau=1 must substitute something on this fixture"
    for t in range(ids.shape[0]):
        row = ids[t].tolist()
        assert len(set(row)) == K, f"row {t} has a duplicate expert: {row}"
        for j in range(K):
            if changed[t, j]:
                assert hmap[row[j]].item() >= 0, "substituted into a NON-resident expert"
                assert row[j] not in ids0[t].tolist(), "substituted an already-selected expert"


def test_weights_are_a_correct_renormalised_softmax_over_the_new_set():
    logits, w, ids, hmap = _setup(range(0, E, 2), seed=5)
    ids0, w0 = ids.clone(), w.clone()
    _run(logits, w, ids, hmap, tau=1.0)
    for t in range(ids.shape[0]):
        sel = logits[t, ids[t].long()]
        want = torch.softmax(sel.double(), 0)
        assert torch.allclose(w[t].double(), want, atol=1e-5), (t, w[t], want)
    # untouched rows must be bit-identical, not merely close
    same = (ids == ids0).all(1)
    assert torch.equal(w[same], w0[same])


@pytest.mark.parametrize("tau", [0.001, 0.01, 0.05, 0.2])
def test_displaced_weight_never_exceeds_tau(tau):
    """The bound is the whole safety argument: whatever weight leaves the missing expert must
    be <= tau, measured against the ORIGINAL weight of the slot that was replaced."""
    logits, w, ids, hmap = _setup(range(0, E, 3), seed=7, T=64)
    ids0, w0 = ids.clone(), w.clone()
    _run(logits, w, ids, hmap, tau=tau)
    changed = (ids != ids0)
    for t in range(ids.shape[0]):
        for j in range(K):
            if changed[t, j]:
                displaced = w0[t, j].item() - w[t, j].item()
                assert displaced <= tau + 1e-6, (t, j, displaced, tau)
                assert displaced >= -1e-6, "a substitute may not gain weight"


def test_more_tau_substitutes_at_least_as_much():
    """Monotonicity: raising the budget can only admit more swaps."""
    counts = []
    for tau in (0.0, 0.01, 0.05, 0.5):
        logits, w, ids, hmap = _setup(range(0, E, 3), seed=11, T=64)
        ids0 = ids.clone()
        _run(logits, w, ids, hmap, tau=tau)
        counts.append(int((ids != ids0).any(1).sum()))
    assert counts == sorted(counts), counts
    assert counts[-1] > counts[0]


# ---- multi-substitution (--moe-residency-routing-max-subs) -----------------------------------
#
# v1 leaves a row's second miss in place, and the disk charges one read per DISTINCT expert per
# (layer, step), so a kept miss is a paid read. These pin the increment: same per-swap tau gate,
# a new bound on their SUM, and byte-identical behaviour at max_subs=1.

import math


def _reference(logits, ids0, hmap, tau, tau_total, max_subs):
    """Greedy substitution in float64 -- what the kernel must reproduce.

    Mirrors the kernel exactly, including that the candidate advances on every iteration and
    that ``active`` latches off after the first refused swap.
    """
    lg = logits.double().cpu()
    host = hmap.cpu().tolist()
    ids = [int(x) for x in ids0.cpu().tolist()]
    K = len(ids)
    sel = set(ids)
    missing = [host[e] < 0 for e in ids]
    if not any(missing):
        return ids, [], 0

    def next_cand(seen):
        best_e, best_l = None, None
        for e in range(len(host)):
            if host[e] < 0 or e in sel or e in seen:
                continue
            if best_l is None or lg[e].item() > best_l:
                best_e, best_l = e, lg[e].item()
        return best_e, best_l

    consumed: set[int] = set()
    cand, cand_lg = next_cand(consumed)
    if cand is None:
        return ids, [], 0
    m = max(max(lg[e].item() for e in ids), cand_lg)
    p = [math.exp(lg[e].item() - m) for e in ids]
    s = sum(p)
    cum, disps, n, active = 0.0, [], 0, True
    for _ in range(max_subs):
        if cand is None:
            break
        p_c = math.exp(cand_lg - m)
        best_j, best_d = None, None
        for j in range(K):
            if not missing[j]:
                continue
            d = p[j] / s - p_c / (s - p[j] + p_c)
            if best_d is None or d < best_d:
                best_j, best_d = j, d
        do_swap = active and best_d is not None and best_d <= tau and cum + best_d <= tau_total
        if do_swap:
            s = sum(p[j] for j in range(K) if j != best_j) + p_c
            p[best_j] = p_c
            ids[best_j] = cand
            missing[best_j] = False
            sel.add(cand)
            cum += best_d
            disps.append(best_d)
            n += 1
        consumed.add(cand)
        cand, cand_lg = next_cand(consumed)
        active = do_swap and any(missing)
    return ids, disps, n


def _run_multi(logits, w, ids, hmap, tau, max_subs, tau_total=0.0):
    from freetoken.kernel.triton.moe_residency_route import residency_route

    stats = torch.zeros(4, dtype=torch.int64, device="cuda")
    residency_route(logits, ids, w, hmap.data_ptr(), 0, E, tau, stats=stats,
                    max_subs=max_subs, tau_total=tau_total)
    return stats


def test_max_subs_one_is_bit_identical_to_v1():
    """The fallback has to be exact, not close: it is what every earlier measurement and every
    quality gate on this feature was taken against."""
    a = _setup(range(0, E, 3), seed=17, T=64)
    b = _setup(range(0, E, 3), seed=17, T=64)
    _run(a[0], a[1], a[2], a[3], tau=0.05)
    _run_multi(b[0], b[1], b[2], b[3], tau=0.05, max_subs=1)
    assert torch.equal(a[2], b[2]), "ids diverged from v1"
    assert torch.equal(a[1], b[1]), "weights diverged from v1"


@pytest.mark.parametrize("max_subs", [2, 4, K])
@pytest.mark.parametrize("tau", [0.01, 0.05, 0.2])
def test_multi_matches_the_greedy_reference(max_subs, tau):
    logits, w, ids, hmap = _setup(range(0, E, 3), seed=23, T=32)
    ids0 = ids.clone()
    _run_multi(logits, w, ids, hmap, tau=tau, max_subs=max_subs)
    for t in range(ids.shape[0]):
        want_ids, _, _ = _reference(logits[t], ids0[t], hmap, tau, tau, max_subs)
        assert ids[t].tolist() == want_ids, (t, ids[t].tolist(), want_ids)
        sel = logits[t, ids[t].long()]
        assert torch.allclose(w[t].double(), torch.softmax(sel.double(), 0), atol=1e-5)


@pytest.mark.parametrize("tau", [0.01, 0.05])
def test_total_displacement_stays_inside_the_budget(tau):
    """The safety argument for raising max_subs: more swaps, same per-token budget."""
    logits, w, ids, hmap = _setup(range(0, E, 3), seed=29, T=64)
    ids0 = ids.clone()
    _run_multi(logits, w, ids, hmap, tau=tau, max_subs=K)
    for t in range(ids.shape[0]):
        _, disps, _ = _reference(logits[t], ids0[t], hmap, tau, tau, K)
        assert all(d <= tau + 1e-9 for d in disps), (t, disps, tau)
        assert sum(disps) <= tau + 1e-9, (t, sum(disps), tau)


def test_tau_total_bounds_the_sum_independently_of_tau():
    """A generous per-swap tau with a tight total must still stop early."""
    logits, w, ids, hmap = _setup(range(0, E, 3), seed=31, T=32)
    ids0 = ids.clone()
    _run_multi(logits, w, ids, hmap, tau=0.5, max_subs=K, tau_total=0.02)
    for t in range(ids.shape[0]):
        want_ids, disps, _ = _reference(logits[t], ids0[t], hmap, 0.5, 0.02, K)
        assert ids[t].tolist() == want_ids
        assert sum(disps) <= 0.02 + 1e-9


def test_multi_is_a_no_op_where_v1_is():
    for tau, resident in ((1.0, range(E)), (0.0, range(0, E, 2))):
        logits, w, ids, hmap = _setup(resident, seed=37)
        w0, ids0 = w.clone(), ids.clone()
        stats = _run_multi(logits, w, ids, hmap, tau=tau, max_subs=K)
        assert torch.equal(ids, ids0) and torch.equal(w, w0)
        assert stats[0].item() == 0


def test_raising_max_subs_adds_swaps_without_perturbing_more_rows():
    """The first swap decision is v1's, so the SET of rows touched must not grow -- only the
    number of substitutions inside them. That is what keeps the quality argument the same."""
    rows, subs = [], []
    for max_subs in (1, 2, K):
        logits, w, ids, hmap = _setup(range(0, E, 3), seed=41, T=64)
        ids0 = ids.clone()
        st = _run_multi(logits, w, ids, hmap, tau=0.05, max_subs=max_subs)
        rows.append(int((ids != ids0).any(1).sum()))
        subs.append(int(st[0].item()))
        assert int(st[2].item()) == rows[-1], "rows-substituted counter disagrees with the ids"
    assert len(set(rows)) == 1, f"rows perturbed changed with max_subs: {rows}"
    assert subs[0] < subs[-1], subs
    assert subs == sorted(subs), subs


def test_capped_counter_reports_what_a_higher_cap_could_still_reach():
    logits, w, ids, hmap = _setup(range(0, E, 3), seed=43, T=64)
    st1 = _run_multi(logits, w, ids, hmap, tau=1.0, max_subs=1)
    logits, w, ids, hmap = _setup(range(0, E, 3), seed=43, T=64)
    stk = _run_multi(logits, w, ids, hmap, tau=1.0, max_subs=K)
    assert st1[3].item() > 0, "the fixture must leave misses behind at max_subs=1"
    assert stk[3].item() <= st1[3].item(), "a higher cap cannot leave MORE rows capped"


# ---- the measured policy: drop, priced in what it moves ---------------------------------------
#
# Measuring every expert against every other showed no expert stands in for another (best cosine
# +0.02..+0.19, and only 6 candidate pairs in 46,080 close enough that substituting beats simply
# removing the route). So the table-driven policy drops, and gates on the fraction of the MoE
# block output the drop actually moves rather than on displaced routing weight.

def _norms(seed=0, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(1, E, generator=g) * scale + 0.5).float().cuda()


def _run_drop(logits, w, ids, hmap, norm, eps, max_acts=4, eps_total=0.0, renorm=False,
              y_cal=1.0):
    from freetoken.kernel.triton.moe_residency_route import residency_route

    stats = torch.zeros(8, dtype=torch.int64, device="cuda")
    residency_route(logits, ids, w, hmap.data_ptr(), 0, E, 0.0, stats=stats,
                    max_subs=max_acts, norm=norm, y_cal=y_cal, eps=eps,
                    eps_total=eps_total, drop_renorm=renorm)
    return stats


def _drop_reference(logits_t, ids_t, hmap, norm_t, eps, eps_total, max_acts, renorm, y_cal):
    """float64 mirror of the kernel, one row."""
    import math
    host = hmap.cpu().tolist()
    ids = [int(x) for x in ids_t.cpu().tolist()]
    lg = [float(logits_t[e]) for e in ids]
    n = [float(norm_t[0, e]) for e in ids]
    K_ = len(ids)
    missing = [host[e] < 0 for e in ids]
    if not any(missing):
        return ids, [1.0], []
    m = max(lg)
    p = [math.exp(v - m) for v in lg]
    s0 = sum(p)
    cum2, acts = 0.0, []
    for _ in range(max(1, min(max_acts, K_ - 1))):
        s = sum(p) if renorm else s0
        w = [v / max(s, 1e-20) for v in p]
        y2 = sum((w[j] * n[j]) ** 2 for j in range(K_))
        yn = max(y_cal * math.sqrt(y2), 1e-12)
        best, slot = None, None
        for j in range(K_):
            if not missing[j]:
                continue
            c = w[j] * n[j] / yn
            if best is None or c < best:
                best, slot = c, j
        if best is None or best > eps or cum2 + best * best > (eps_total or eps) ** 2:
            break
        p[slot] = 0.0
        ids[slot] = -1
        missing[slot] = False
        cum2 += best * best
        acts.append(best)
    s = sum(p) if renorm else s0
    return ids, [v / max(s, 1e-20) for v in p], acts


def test_measured_policy_is_off_without_a_table_or_without_eps():
    """Both switches must be able to turn it off, and off must be exactly the old behaviour."""
    for kwargs in ({"norm": None, "eps": 0.05}, {"norm": _norms(), "eps": 0.0}):
        a = _setup(range(0, E, 3), seed=17, T=32)
        b = _setup(range(0, E, 3), seed=17, T=32)
        _run(a[0], a[1], a[2], a[3], tau=0.05)
        from freetoken.kernel.triton.moe_residency_route import residency_route
        residency_route(b[0], b[2], b[1], b[3].data_ptr(), 0, E, 0.05,
                        stats=torch.zeros(8, dtype=torch.int64, device="cuda"), **kwargs)
        assert torch.equal(a[2], b[2]) and torch.equal(a[1], b[1])


def test_a_drop_writes_minus_one_and_zeroes_only_that_weight():
    logits, w, ids, hmap = _setup(range(0, E, 3), seed=23, T=32)
    ids0, w0 = ids.clone(), w.clone()
    _run_drop(logits, w, ids, hmap, _norms(1), eps=1.0, max_acts=1)
    changed = ids != ids0
    assert changed.any(), "the fixture must actually drop something"
    assert (ids[changed] == -1).all(), "a dropped route must be -1, not another expert"
    assert (w[changed] == 0).all(), "a dropped route must carry zero weight"
    # zero-drop leaves the survivors alone: it is the L2-optimal removal
    keep = ~changed
    assert torch.allclose(w[keep], w0[keep], atol=1e-6)


@pytest.mark.parametrize("eps", [0.02, 0.08, 0.3])
@pytest.mark.parametrize("renorm", [False, True])
def test_drop_matches_the_float64_reference(eps, renorm):
    logits, w, ids, hmap = _setup(range(0, E, 3), seed=29, T=48)
    norm = _norms(2)
    ids0 = ids.clone()
    _run_drop(logits, w, ids, hmap, norm, eps=eps, max_acts=4, renorm=renorm)
    for t in range(ids.shape[0]):
        want_ids, want_w, _ = _drop_reference(logits[t], ids0[t], hmap, norm, eps, 0.0, 4,
                                              renorm, 1.0)
        assert ids[t].tolist() == want_ids, (t, ids[t].tolist(), want_ids)
        assert torch.allclose(w[t].double().cpu(), torch.tensor(want_w, dtype=torch.float64),
                              atol=1e-5)


@pytest.mark.parametrize("eps", [0.02, 0.08])
def test_every_action_and_their_quadrature_sum_stay_inside_the_budget(eps):
    logits, w, ids, hmap = _setup(range(0, E, 3), seed=31, T=64)
    norm = _norms(3)
    ids0 = ids.clone()
    _run_drop(logits, w, ids, hmap, norm, eps=eps, max_acts=4)
    for t in range(ids.shape[0]):
        _, _, acts = _drop_reference(logits[t], ids0[t], hmap, norm, eps, 0.0, 4, False, 1.0)
        assert all(a <= eps + 1e-9 for a in acts), (t, acts, eps)
        assert sum(a * a for a in acts) <= eps * eps + 1e-9


def test_a_row_never_loses_every_route():
    """No budget may buy a row that emits nothing: MAX_ACTS is clamped to K-1 at the launch."""
    logits, w, ids, hmap = _setup([], seed=37, T=32)   # nothing resident: every route missing
    _run_drop(logits, w, ids, hmap, _norms(4), eps=1e9, max_acts=E, eps_total=1e9)
    live = (ids >= 0).sum(dim=1)
    assert int(live.min()) >= 1, "a row was left with no experts at all"
    assert float(w.sum(dim=1).min()) > 0


def test_nothing_missing_is_a_no_op():
    logits, w, ids, hmap = _setup(range(E), seed=41)
    w0, ids0 = w.clone(), ids.clone()
    st = _run_drop(logits, w, ids, hmap, _norms(5), eps=1.0, max_acts=4)
    assert torch.equal(ids, ids0) and torch.equal(w, w0)
    assert int(st[4]) == 0


def test_counters_report_drops_and_the_perturbation_spent():
    logits, w, ids, hmap = _setup(range(0, E, 3), seed=43, T=64)
    ids0 = ids.clone()
    st = _run_drop(logits, w, ids, hmap, _norms(6), eps=0.1, max_acts=4)
    n_drop = int((ids != ids0).sum())
    assert int(st[4]) == n_drop, "the drop counter disagrees with the ids"
    assert int(st[1]) > 0, "host-missing routes must be counted"
    spent = int(st[5]) / 1048576.0
    assert 0 < spent <= n_drop * 0.1 + 1e-6, "spent perturbation outside its own budget"
