"""Chain speculative sampling math (engine/spec.py): exactness vs a sequential reference
and the core distribution-preservation invariant (the emitted token's marginal equals the
target distribution regardless of the draft distribution)."""

from __future__ import annotations

import pytest
import torch

from freetoken.engine.spec import chain_accept

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

DEV = torch.device("cuda")


def _random_dist(rows: int, vocab: int, gen: torch.Generator, sparsity: float = 0.0):
    logits = torch.randn(rows, vocab, generator=gen, device=DEV)
    if sparsity:
        mask = torch.rand(rows, vocab, generator=gen, device=DEV) < sparsity
        logits = logits.masked_fill(mask, float("-inf"))
    return torch.softmax(logits, dim=-1)


def _reference(p, q, d, u):
    """Sequential chain acceptance, one position at a time."""
    k = d.shape[0]
    for j in range(k):
        pj, qj = p[j, d[j]].item(), q[j, d[j]].item()
        if u[j].item() * qj <= pj:
            continue
        res = (p[j] - q[j]).clamp_min(0)
        return j, res / res.sum().clamp_min(1e-20)
    return k, p[k]


@pytest.mark.parametrize("k", [1, 2, 4])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_chain_accept_matches_sequential_reference(k, seed):
    gen = torch.Generator(device=DEV).manual_seed(seed)
    vocab = 733
    p = _random_dist(k + 1, vocab, gen)
    q = _random_dist(k, vocab, gen, sparsity=0.3)
    d = torch.multinomial(q, 1, generator=gen).squeeze(1).to(torch.int32)
    u = torch.rand(k, generator=gen, device=DEV)

    n_acc, fallback = chain_accept(p, q, d, u)
    ref_n, ref_row = _reference(p, q, d, u)
    assert int(n_acc.item()) == ref_n
    torch.testing.assert_close(fallback[ref_n], ref_row, rtol=1e-5, atol=1e-7)


def test_emitted_marginal_equals_target():
    """Simulate many chain steps at k=1: the emitted first token's empirical distribution
    must match the target regardless of the (different) draft distribution."""
    gen = torch.Generator(device=DEV).manual_seed(7)
    vocab, trials = 40, 400_000
    p = _random_dist(2, vocab, gen)
    q = _random_dist(1, vocab, gen, sparsity=0.4)

    d = torch.multinomial(q.expand(trials, vocab), 1, generator=gen).squeeze(1)
    u = torch.rand(trials, generator=gen, device=DEV)
    p_d = p[0].index_select(0, d)
    q_d = q[0].index_select(0, d)
    accept = u * q_d <= p_d
    res = (p[0] - q[0]).clamp_min(0)
    res = res / res.sum()
    corrections = torch.multinomial(res.expand(trials, vocab), 1, generator=gen).squeeze(1)
    emitted = torch.where(accept, d, corrections)

    empirical = torch.bincount(emitted, minlength=vocab).float() / trials
    tv = 0.5 * (empirical - p[0]).abs().sum().item()
    assert tv < 0.01, f"total variation {tv:.4f} vs target"
    # sanity: the draft distribution is genuinely different from the target
    assert 0.5 * (q[0] - p[0]).abs().sum().item() > 0.2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("n", [1, 2, 3])
def test_fused_gdn_commit_matches_torch(n):
    """commit_gdn_states (one launch) == index_select + index_copy_ of the accepted row."""
    from freetoken.kernel.triton.spec_commit import commit_gdn_states

    torch.manual_seed(n)
    L, K1, HV, KD, VD, SLOTS = 3, 3, 4, 16, 16, 5
    dev = torch.device("cuda")
    gdn_states = torch.randn(L, 1, K1, HV, KD, VD, device=dev, dtype=torch.float32)
    pool_ref = torch.randn(L, SLOTS, HV, KD, VD, device=dev, dtype=torch.float32)
    pool_fused = pool_ref.clone()
    num_inputs = torch.tensor([n], device=dev, dtype=torch.int32)
    slot = torch.tensor([2], device=dev, dtype=torch.int32)
    accepted = gdn_states[:, 0].index_select(1, (num_inputs - 1)).squeeze(1)
    pool_ref.index_copy_(1, slot.long(), accepted.unsqueeze(1))
    commit_gdn_states(gdn_states, pool_fused, num_inputs, slot)
    torch.cuda.synchronize()
    assert torch.equal(pool_fused, pool_ref)
