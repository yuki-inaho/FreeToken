"""The C++ executor's tier-3 row indirection.

Under tier 3 a layer's host bank holds only ``host_slots`` expert rows, and which row holds
which expert is decided at runtime. The executor must address weights by that row, not by the
expert id. This is the one change in the C++ kernel, and getting it wrong produces *plausible
but wrong* activations -- another expert's weights -- so it is checked against an explicit
permutation rather than a smoke test.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

L, E, H, I, TOP_K = 4, 16, 512, 256, 4
LAYER = 2


def _cache_from(gate_up: torch.Tensor, down: torch.Tensor, num_experts: int):
    return SimpleNamespace(
        quant_format="bf16",
        bank_sources={"gate_up": list(gate_up.split(num_experts)),
                      "down": list(down.split(num_experts))},
        num_layers=L, num_experts=num_experts, decode_target="cpu", cpu_executor=None,
    )


def _banks(rows_per_layer: int, seed: int = 0):
    from freetoken.kernel.pinned import alloc_pinned_tensor

    torch.manual_seed(seed)
    gu = alloc_pinned_tensor(L * rows_per_layer, 2 * I, H, dtype=torch.bfloat16)
    dn = alloc_pinned_tensor(L * rows_per_layer, H, I, dtype=torch.bfloat16)
    gu.copy_(torch.randn(L * rows_per_layer, 2 * I, H) * 0.1)
    dn.copy_(torch.randn(L * rows_per_layer, H, I) * 0.1)
    return gu, dn


def _executor(cache, bs, dev):
    from freetoken.moe.cpu_executor import CpuMoeExecutor

    return CpuMoeExecutor(cache, top_k=TOP_K, activation="silu",
                          apply_router_weight_on_input=False, num_threads=0,
                          max_tokens=bs, device=dev)


@cuda
@pytest.mark.parametrize("bs", [1, 4])
def test_permuted_host_rows_give_the_same_result_as_full_residency(bs):
    """Same weights, shuffled into different host rows, plus a map that undoes the shuffle,
    must reproduce the identity-layout result bit for bit."""
    dev = torch.device("cuda")
    gu, dn = _banks(E, seed=bs)

    # reference: identity layout, no host map
    ref_cache = _cache_from(gu, dn, E)
    ref = _executor(ref_cache, bs, dev)

    # tier 3: row r of each layer holds expert perm[r]; the map sends expert -> its row
    g = torch.Generator().manual_seed(77 + bs)
    perm = torch.randperm(E, generator=g)
    assert not torch.equal(perm, torch.arange(E)), "permutation must actually permute"
    gu2 = torch.empty_like(gu)
    dn2 = torch.empty_like(dn)
    host_map = torch.empty((L, E), dtype=torch.int32)
    for layer in range(L):
        for row in range(E):
            e = int(perm[row])
            gu2[layer * E + row] = gu[layer * E + e]
            dn2[layer * E + row] = dn[layer * E + e]
            host_map[layer, e] = row
    tier_cache = _cache_from(gu2, dn2, E)
    tier = _executor(tier_cache, bs, dev)
    tier._cpu_moe.CpuMoeExecutor  # noqa: B018 - module handle exists
    tier._ext.set_host_slot_table(host_map.contiguous().data_ptr())

    hidden = torch.randn(bs, H, device=dev, dtype=torch.bfloat16)
    ids = torch.stack([torch.randperm(E, device=dev)[:TOP_K] for _ in range(bs)]).to(torch.int32)
    w = torch.rand(bs, TOP_K, device=dev, dtype=torch.float32)

    out_ref = ref.decode(LAYER, hidden, w, ids.clone()).float()
    out_tier = tier.decode(LAYER, hidden, w, ids.clone()).float()
    torch.cuda.synchronize()
    torch.testing.assert_close(out_tier, out_ref, rtol=0, atol=0)


@cuda
def test_identity_map_is_a_no_op():
    """host_slot_tbl set to the identity must be indistinguishable from not setting it -- the
    guard that the zero-overhead default really is the old behaviour."""
    dev = torch.device("cuda")
    gu, dn = _banks(E, seed=5)
    plain = _executor(_cache_from(gu, dn, E), 2, dev)
    mapped = _executor(_cache_from(gu, dn, E), 2, dev)
    ident = torch.arange(E, dtype=torch.int32).repeat(L, 1).contiguous()
    mapped._ext.set_host_slot_table(ident.data_ptr())

    hidden = torch.randn(2, H, device=dev, dtype=torch.bfloat16)
    ids = torch.stack([torch.randperm(E, device=dev)[:TOP_K] for _ in range(2)]).to(torch.int32)
    w = torch.rand(2, TOP_K, device=dev, dtype=torch.float32)
    a = plain.decode(LAYER, hidden, w, ids.clone()).float()
    b = mapped.decode(LAYER, hidden, w, ids.clone()).float()
    torch.cuda.synchronize()
    torch.testing.assert_close(b, a, rtol=0, atol=0)


@cuda
def test_a_wrong_map_actually_changes_the_output():
    """Negative control: if the executor ignored host_slot_tbl this suite would pass
    vacuously, so prove a non-identity map is observable."""
    dev = torch.device("cuda")
    gu, dn = _banks(E, seed=9)
    plain = _executor(_cache_from(gu, dn, E), 1, dev)
    shifted = _executor(_cache_from(gu, dn, E), 1, dev)
    rolled = ((torch.arange(E) + 1) % E).to(torch.int32).repeat(L, 1).contiguous()
    shifted._ext.set_host_slot_table(rolled.data_ptr())

    hidden = torch.randn(1, H, device=dev, dtype=torch.bfloat16)
    ids = torch.arange(TOP_K, device=dev, dtype=torch.int32).view(1, TOP_K)
    w = torch.ones(1, TOP_K, device=dev, dtype=torch.float32)
    a = plain.decode(LAYER, hidden, w, ids.clone()).float()
    b = shifted.decode(LAYER, hidden, w, ids.clone()).float()
    torch.cuda.synchronize()
    assert not torch.allclose(a, b), "host_slot_tbl had no effect -- the indirection is dead"
