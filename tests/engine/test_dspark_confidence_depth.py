"""The DSpark confidence-scheduled verify depth (pure rule, CPU).

``confidence_verify_depth(surv, tau, k_max)`` counts the leading drafts whose prefix-survival
probability (``cumprod(sigmoid(conf))``) meets ``tau``; the result is clamped to ``[1, k_max]``
because a verify always carries the anchor plus at least one draft and never more than the
launcher's draft count. The expectations below are hand-derived from that definition.
"""

from __future__ import annotations

import torch

from freetoken.engine.spec_dspark import confidence_verify_depth


def _surv(*probs):
    return torch.cumprod(torch.tensor(probs, dtype=torch.float32), dim=0)


def test_counts_leading_drafts_above_threshold():
    # per-position 0.9, 0.8, 0.7, 0.6, 0.5 -> survival 0.9, 0.72, 0.504, 0.302, 0.151
    s = _surv(0.9, 0.8, 0.7, 0.6, 0.5)
    assert int(confidence_verify_depth(s, 0.5, 5)) == 3
    assert int(confidence_verify_depth(s, 0.3, 5)) == 4
    assert int(confidence_verify_depth(s, 0.75, 5)) == 1


def test_clamps_to_one_and_to_k_max():
    s = _surv(0.1, 0.1, 0.1, 0.1, 0.1)
    assert int(confidence_verify_depth(s, 0.5, 5)) == 1  # nothing survives: still one draft
    s = _surv(0.99, 0.99, 0.99, 0.99, 0.99)
    assert int(confidence_verify_depth(s, 0.5, 5)) == 5
    assert int(confidence_verify_depth(s, 0.5, 2)) == 2  # the launcher's draft count caps it


def test_survival_is_monotone_so_the_count_is_a_prefix():
    # cumprod is non-increasing, so the count equals the index of the first miss
    s = _surv(0.95, 0.4, 0.99, 0.99, 0.99)
    assert int(confidence_verify_depth(s, 0.5, 5)) == 1
