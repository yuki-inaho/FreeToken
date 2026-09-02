"""gdn_verify_fla (speculative target-verify) vs sequential gdn_decode_fla steps.

The verify path must (a) produce the same per-token outputs as stepping the fused decode
kernel one token at a time, (b) record the post-token recurrent state of every step in the
intermediate buffer, and (c) leave the live state slot untouched, so that committing the
accepted step's buffer row reproduces exactly the state sequential decode would have.
"""

from __future__ import annotations

import pytest
import torch

from freetoken.models.qwen3_5_moe.gdn_kernels import gdn_decode_fla, gdn_verify_fla

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

DEV = torch.device("cuda")
NUM_K, NUM_V, K = 16, 48, 128  # the Qwen3.8-Flash-Next GDN geometry (V == K == 128)
SCALE = K**-0.5


def _inputs(total: int, seed: int):
    gen = torch.Generator(device=DEV).manual_seed(seed)

    def rand(*shape, dtype=torch.bfloat16):
        return torch.randn(*shape, generator=gen, device=DEV, dtype=torch.float32).to(dtype)

    q = rand(1, total, NUM_K, K)
    k = rand(1, total, NUM_K, K)
    v = rand(1, total, NUM_V, K)
    a = rand(total, NUM_V, dtype=torch.float32)
    b = rand(total, NUM_V, dtype=torch.float32)
    A_log = torch.rand(NUM_V, generator=gen, device=DEV).mul(15.99).add(0.01).log()
    dt_bias = rand(NUM_V, dtype=torch.float32)
    return q, k, v, a, b, A_log, dt_bias


def _rand_state(num_slots: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device=DEV).manual_seed(seed)
    return torch.randn(num_slots, NUM_V, K, K, generator=gen, device=DEV, dtype=torch.float32)


@pytest.mark.parametrize("steps", [1, 2, 4])
def test_verify_matches_sequential_decode(steps):
    q, k, v, a, b, A_log, dt_bias = _inputs(steps, seed=1)
    slot = 1

    # Reference: one fused-decode call per token, live state updated in place.
    ref_states = []
    ref_pool = _rand_state(3, seed=2)
    ref_out = []
    for t in range(steps):
        out = gdn_decode_fla(
            q[:, t : t + 1], k[:, t : t + 1], v[:, t : t + 1], a[t : t + 1], b[t : t + 1],
            A_log=A_log, dt_bias=dt_bias, state_source=ref_pool,
            indices=torch.tensor([slot], device=DEV, dtype=torch.int32),
            cu_seqlens=torch.arange(2, device=DEV, dtype=torch.int32),
            scale=SCALE,
        )
        ref_out.append(out)
        ref_states.append(ref_pool[slot].clone())
    ref_out = torch.cat(ref_out, dim=0)

    # Verify: one call, k tokens, state write-back disabled, intermediates recorded.
    pool = _rand_state(3, seed=2)
    pool_before = pool.clone()
    inter = torch.full((1, steps, NUM_V, K, K), float("nan"), device=DEV, dtype=torch.float32)
    out = gdn_verify_fla(
        q, k, v, a, b,
        A_log=A_log, dt_bias=dt_bias, state_source=pool,
        indices=torch.tensor([slot], device=DEV, dtype=torch.int32),
        cu_seqlens=torch.tensor([0, steps], device=DEV, dtype=torch.int32),
        intermediate_states=inter,
        intermediate_indices=torch.tensor([0], device=DEV, dtype=torch.int32),
        scale=SCALE,
    )

    torch.testing.assert_close(out, ref_out, rtol=0, atol=0)  # same kernel math, same order
    assert torch.equal(pool, pool_before), "verify must not touch the live state"
    for t in range(steps):
        torch.testing.assert_close(inter[0, t], ref_states[t], rtol=0, atol=0)

    # Committing the accepted step's row reproduces sequential decode's state exactly.
    for accepted in range(1, steps + 1):
        pool[slot].copy_(inter[0, accepted - 1])
        assert torch.equal(pool[slot], ref_states[accepted - 1])


def test_verify_ragged_two_sequences():
    """Two packed sequences of different verify lengths in one call."""
    lens = (3, 2)
    total = sum(lens)
    q, k, v, a, b, A_log, dt_bias = _inputs(total, seed=5)
    pool = _rand_state(4, seed=6)
    slots = torch.tensor([2, 0], device=DEV, dtype=torch.int32)
    inter = torch.zeros(2, max(lens), NUM_V, K, K, device=DEV, dtype=torch.float32)

    out = gdn_verify_fla(
        q, k, v, a, b,
        A_log=A_log, dt_bias=dt_bias, state_source=pool,
        indices=slots,
        cu_seqlens=torch.tensor([0, lens[0], total], device=DEV, dtype=torch.int32),
        intermediate_states=inter,
        intermediate_indices=torch.arange(2, device=DEV, dtype=torch.int32),
        scale=SCALE,
    )

    # Per-sequence reference via single-sequence verify calls on the same pool contents.
    offset = 0
    for i, n in enumerate(lens):
        ref_inter = torch.zeros(1, n, NUM_V, K, K, device=DEV, dtype=torch.float32)
        ref_out = gdn_verify_fla(
            q[:, offset : offset + n], k[:, offset : offset + n], v[:, offset : offset + n],
            a[offset : offset + n], b[offset : offset + n],
            A_log=A_log, dt_bias=dt_bias, state_source=pool,
            indices=slots[i : i + 1],
            cu_seqlens=torch.tensor([0, n], device=DEV, dtype=torch.int32),
            intermediate_states=ref_inter,
            intermediate_indices=torch.zeros(1, device=DEV, dtype=torch.int32),
            scale=SCALE,
        )
        torch.testing.assert_close(out[offset : offset + n], ref_out, rtol=0, atol=0)
        torch.testing.assert_close(inter[i, :n], ref_inter[0], rtol=0, atol=0)
        offset += n
