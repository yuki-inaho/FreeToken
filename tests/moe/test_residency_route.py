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
