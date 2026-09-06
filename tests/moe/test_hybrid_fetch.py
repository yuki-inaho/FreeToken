"""Hybrid decode's bandwidth-matched fetch split.

Covers the two halves of --moe-hybrid-max-fetch auto: the profile reader that turns
`ft bench bw` kernel bandwidths into a fetch fraction, and the ensure kernel's
per-step integer split (GPU kernel vs CPU reference mirror, and the balance rule).
"""

import json
import os

import pytest
import torch

from freetoken.moe.bench_profile import default_profile_path, load_backend_recommendation, load_hybrid_fetch_fraction
from freetoken.moe.offload_cache import OffloadMoeCache

Q = 1 << 16


def _balanced_fetch(num_missing: int, frac_q16: int) -> int:
    """Reference split: F ~ frac * misses, rounded to whichever integer neighbor
    minimizes the slower overlapped side (fetch ~ F*(1-frac), CPU ~ (M-F)*frac)."""
    lo = (num_missing * frac_q16) >> 16
    cost = lambda f: max(f * (Q - frac_q16), (num_missing - f) * frac_q16)  # noqa: E731
    return min(num_missing, lo if cost(lo) <= cost(lo + 1) else lo + 1)


def test_balanced_fetch_tracks_fraction():
    # The split follows fetched : cpu = pcie : (cpu - pcie) up to integer rounding, and
    # never over/under-shoots by more than one expert.
    for frac in (0.1, 0.415, 0.454, 0.7, 1.0):
        q = round(frac * Q)
        for m in range(0, 65):
            f = _balanced_fetch(m, q)
            assert 0 <= f <= m
            assert abs(f - frac * m) <= 1.0
    # ceil would over-fetch here (the regression this rule fixed): 41.5% of 3 misses is
    # 1.24 -> fetching 2 makes the PCIe side ~1.6x slower than balance; keep it at 1.
    assert _balanced_fetch(3, round(0.415 * Q)) == 1
    assert _balanced_fetch(4, round(0.415 * Q)) == 2


def test_load_hybrid_fetch_fraction(tmp_path):
    prof = {
        "gpu": {"name": "FAKE GPU"},
        "dtype_kernels": {
            "bf16": {"cpu_moe_gbs": 100.0, "pcie_gather_gbs": 40.0},
            # overlapped (contended) pair wins over the standalone numbers when present
            "nvfp4_x": {"cpu_moe_gbs": 100.0, "pcie_gather_gbs": 40.0,
                        "cpu_moe_overlap_gbs": 90.0, "pcie_gather_overlap_gbs": 30.0},
        },
        "workloads": {
            "m": {"kernels": {"ds_fp4": {"cpu_moe_gbs": 80.0, "pcie_gather_gbs": 50.0}}}
        },
    }
    path = tmp_path / "benchbw.json"
    path.write_text(json.dumps(prof))
    # standalone fallback: full-contention assumption -> pcie / cpu
    assert load_hybrid_fetch_fraction("bf16", path=str(path)) == pytest.approx(0.4)
    # overlapped pair preferred: pcie_ov / (pcie_ov + cpu_ov)
    assert load_hybrid_fetch_fraction("nvfp4_x", path=str(path)) == pytest.approx(0.25)
    # per-model fallback when there is no per-dtype entry for the format
    assert load_hybrid_fetch_fraction("ds_fp4", path=str(path)) == pytest.approx(0.625)
    assert load_hybrid_fetch_fraction("nvfp4", path=str(path)) is None
    # a profile from different hardware is ignored
    assert load_hybrid_fetch_fraction("bf16", gpu_name="OTHER", path=str(path)) is None


def test_profile_lookup_prefers_the_gpu_uuid_file(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.delenv("FREETOKEN_BENCHBW_PATH", raising=False)
    uuid = "GPU-2f3a9b1c-0000-1111-2222-333344445555"

    def write(path, name, verdict):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"gpu": {"name": name}, "dtypes": {"bf16": verdict}}, f)

    # legacy single file only: used when the name matches, ignored otherwise
    write(default_profile_path(), "FAKE GPU", "hybrid")
    assert load_backend_recommendation("bf16", gpu_name="FAKE GPU", gpu_uuid=uuid) == "hybrid"
    assert load_backend_recommendation("bf16", gpu_name="OTHER", gpu_uuid=uuid) is None
    # this card's own file wins over the legacy one
    write(default_profile_path(uuid), "FAKE GPU", "offload")
    assert load_backend_recommendation("bf16", gpu_name="FAKE GPU", gpu_uuid=uuid) == "offload"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_hybrid_fraction_gpu_matches_cpu_reference(monkeypatch):
    # Keep the original per-call balance contract covered independently of the
    # amortized admission policy, which intentionally lifts a rounded zero cap.
    monkeypatch.setattr("freetoken.moe.offload_kernels._HYBRID_ADMISSION_CREDIT", False)
    torch.manual_seed(0)
    num_experts, cache_size, top_k, frac = 32, 40, 8, 0.415

    def make():
        return OffloadMoeCache(
            num_layers=2, num_experts=num_experts, cache_size=cache_size,
            device=torch.device("cuda"), quant_format="bf16", decode_target="hybrid",
            hybrid_max_fetch=num_experts, hybrid_fetch_fraction=frac,
        )

    gpu, ref = make(), make()
    frac_q16 = round(frac * Q)
    gpu.collect_stats = ref.collect_stats = True
    for step in range(64):
        ids = torch.randperm(num_experts)[:top_k].to(torch.int32)
        g, c = ids.clone().cuda(), ids.clone()  # a CPU ids tensor drives the reference path
        g_cpu_ids = gpu.ensure_experts_hybrid(0, g)
        c_cpu_ids = ref.ensure_experts_hybrid(0, c)
        missing = int(gpu.num_missing_full.item())
        fetched = int(gpu.num_indices.item())
        assert missing == int(ref.num_missing_full.item())
        assert fetched == int(ref.num_indices.item()) == _balanced_fetch(missing, frac_q16)
        # slot rewrites (hit/fetched -> slot, overflow -> -1) and LRU state stay identical
        assert torch.equal(g.cpu(), c)
        assert torch.equal(gpu.slot_for_id.cpu(), ref.slot_for_id.cpu())
        assert torch.equal(gpu.id_of_slot.cpu(), ref.id_of_slot.cpu())
        assert (g >= 0).sum().item() == len(set(ids.tolist())) - (missing - fetched)
        # the CPU route list emitted by the same launch: raw id where the slot is -1
        expect_cpu = torch.where(c >= 0, torch.full_like(ids, -1), ids)
        assert torch.equal(g_cpu_ids.cpu(), expect_cpu)
        assert torch.equal(c_cpu_ids.cpu(), expect_cpu)
    # in-kernel stats match the CPU mirror's accumulation
    for name in ("stat_missing", "stat_fetched", "stat_active", "stat_calls",
                 "stat_missing_layer", "stat_fetched_layer", "stat_active_layer",
                 "stat_steps_layer"):
        assert torch.equal(getattr(gpu, name).cpu(), getattr(ref, name).cpu()), name
    assert int(gpu.stat_calls.item()) == 64


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_hybrid_fixed_cap_unchanged():
    # fraction 0 (no profile / explicit --moe-hybrid-max-fetch) keeps the fixed cap.
    cache = OffloadMoeCache(
        num_layers=1, num_experts=32, cache_size=40, device=torch.device("cuda"),
        quant_format="bf16", decode_target="hybrid", hybrid_max_fetch=1,
    )
    ids = torch.arange(8, dtype=torch.int32).cuda()
    cache.ensure_experts_hybrid(0, ids)
    assert int(cache.num_missing_full.item()) == 8
    assert int(cache.num_indices.item()) == 1


_ROOMY_FRACTION = 39.67 / (39.67 + 90.55)
_ROOMY_Q16 = round(_ROOMY_FRACTION * Q)
_DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs CUDA"))]


def _admission_cache(device, *, frac=_ROOMY_FRACTION, max_fetch=32):
    cache = OffloadMoeCache(
        num_layers=2, num_experts=32, cache_size=40,
        device=torch.device(device), quant_format="bf16", decode_target="hybrid",
        hybrid_max_fetch=max_fetch, hybrid_fetch_fraction=frac,
    )
    # One resident route lets hit-only calls exercise paused credit.
    cache.slot_for_id[0, 0] = 0
    cache.id_of_slot[0] = 0
    return cache


def _visit(cache, raw, layer=0):
    ids = torch.tensor(raw, dtype=torch.int32, device=cache.device)
    cpu = cache.ensure_experts_hybrid(layer, ids).clone()
    return int(cache.num_indices.item()), ids, cpu


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("rows", [1, 4])
@pytest.mark.parametrize("intermittent", [False, True])
def test_recurring_singleton_admits_with_bounded_credit(device, rows, intermittent):
    cache = _admission_cache(device)
    fetched = []
    for step in range(6):
        count, ids, cpu = _visit(cache, [[0, 30]] * rows)
        fetched.append(count)
        credit = int(cache.hybrid_admission_credit[0].item())
        assert 0 <= credit < Q
        if step < 4:
            assert (cpu[:, 1] == 30).all()
            # Repeated token rows do not multiply distinct-expert credit.
            assert credit == step * _ROOMY_Q16
        else:
            assert (cpu == -1).all()
            assert (ids >= 0).all()
        if intermittent:
            _visit(cache, [[0]])
            assert int(cache.hybrid_admission_credit[0].item()) == credit
    assert fetched == [0, 0, 0, 0, 1, 0]
    assert cache.hybrid_admission_credit[1].item() == 0


@pytest.mark.parametrize("device", _DEVICES)
def test_credit_cannot_admit_unseen_transient_misses(device):
    cache = _admission_cache(device)
    for _ in range(4):
        _visit(cache, [[30]])
    credit = int(cache.hybrid_admission_credit[0].item())
    for expert in range(1, 20):
        assert _visit(cache, [[expert]])[0] == 0
        assert cache.hybrid_admission_credit[0].item() == credit
    # A fresh lower-id candidate cannot consume the credit of the recurring one.
    count, ids, _ = _visit(cache, [[20, 30]])
    assert count == 1
    assert ids[0, 0] == -1 and ids[0, 1] >= 0


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("misses", [2, 3])
def test_small_recurring_miss_set_eventually_becomes_fully_resident(device, misses):
    cache = _admission_cache(device)
    raw = [list(range(32 - misses, 32))] * 4
    counts = [_visit(cache, raw)[0] for _ in range(7)]
    assert counts == [int(misses == 3), 0, 1, 0, 0, 1, 0]
    assert (cache.slot_for_id[0, 32 - misses:] >= 0).all()


@pytest.mark.parametrize("device", _DEVICES)
def test_positive_balanced_fetch_resets_credit(device):
    cache = _admission_cache(device)
    for _ in range(4):
        _visit(cache, [[30]])
    assert cache.hybrid_admission_credit[0].item() > 0
    # Three distinct misses already get one fetch under the original policy.
    assert _visit(cache, [[29, 30, 31]])[0] == 1
    assert cache.hybrid_admission_credit[0].item() == 0


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("mode", ["ablation", "fixed_zero", "fixed_one", "tier3"])
def test_admission_preserves_disabled_fixed_and_tier3_paths(device, mode, monkeypatch):
    if mode == "ablation":
        monkeypatch.setattr("freetoken.moe.offload_kernels._HYBRID_ADMISSION_CREDIT", False)
    fixed = mode.startswith("fixed")
    cache = _admission_cache(device, frac=0.0 if fixed else _ROOMY_FRACTION,
                             max_fetch=1 if mode == "fixed_one" else 0)
    if mode == "tier3":
        cache.host_slot_for_id = torch.arange(32, dtype=torch.int32).repeat(2, 1)
        if device == "cuda":
            cache.host_slot_dev = cache.host_slot_for_id.to(device)
    counts = [_visit(cache, [[30]])[0] for _ in range(12)]
    assert counts == ([1] + [0] * 11 if mode == "fixed_one" else [0] * 12)
    assert cache.hybrid_admission_credit.count_nonzero().item() == 0


@pytest.mark.parametrize("device", _DEVICES)
def test_admission_credit_reset_lifecycle(device, monkeypatch):
    cache = _admission_cache(device)
    for _ in range(3):
        _visit(cache, [[30]])
    before = cache.hybrid_admission_credit.clone()
    cache.reset_stats()
    assert torch.equal(cache.hybrid_admission_credit, before)
    if device == "cpu":
        # The generic slot-cache reset only has a CUDA kernel; isolate the
        # lifecycle of the new, device-independent credit in the CPU case.
        monkeypatch.setattr("freetoken.moe.offload_kernels.reset_cache", lambda c: None)
    cache.reset()
    assert cache.hybrid_admission_credit.count_nonzero().item() == 0
    assert (cache.expert_recency == -1).all()
    cache.hybrid_admission_credit.fill_(12345)
    # Tiny registered sources are sufficient to exercise a real cache rebuild.
    cache.set_bank_sources({
        name: [torch.zeros((32, 16), dtype=torch.bfloat16, device=cache.device) for _ in range(2)]
        for name in cache.bank_schema
    })
    cache.rebuild(48)
    assert cache.hybrid_admission_credit.count_nonzero().item() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("recency", [False, True])
def test_admission_cpu_gpu_parity_during_churn(recency, monkeypatch):
    monkeypatch.setattr("freetoken.moe.offload_kernels._HYBRID_FETCH_BY_RECENCY", recency)
    gpu, ref = _admission_cache("cuda"), _admission_cache("cpu")
    gpu.collect_stats = ref.collect_stats = True
    gen = torch.Generator().manual_seed(8)
    calls = [(0, [[0, 30]] * 4), (1, [[31]])] * 6
    calls += [(i % 2, [torch.randperm(32, generator=gen)[:(i % 7 + 1)].tolist()]) for i in range(80)]
    calls += [(0, [[29, 30]])] * 12
    for layer, raw in calls:
        gf, gi, gc = _visit(gpu, raw, layer)
        rf, ri, rc = _visit(ref, raw, layer)
        assert gf == rf
        assert torch.equal(gi.cpu(), ri)
        assert torch.equal(gc.cpu(), rc)
        for name in ("slot_for_id", "id_of_slot", "usage", "expert_recency", "hybrid_admission_credit",
                     "stat_missing", "stat_fetched", "stat_active", "stat_calls"):
            assert torch.equal(getattr(gpu, name).cpu(), getattr(ref, name)), name
        assert ((gpu.hybrid_admission_credit >= 0) & (gpu.hybrid_admission_credit < Q)).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_admission_credit_advances_under_captured_replay():
    cache = _admission_cache("cuda")
    raw = torch.tensor([[0, 30]] * 4, dtype=torch.int32, device="cuda")
    ids = torch.empty_like(raw)
    # Warm compilation before capture, then start the captured sequence cold.
    ids.copy_(raw)
    cache.ensure_experts_hybrid(0, ids)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ids.copy_(raw)
        cache.ensure_experts_hybrid(0, ids)
    cache.reset()
    cache.slot_for_id[0, 0] = 0
    cache.id_of_slot[0] = 0
    counts = []
    for _ in range(6):
        graph.replay()
        counts.append(int(cache.num_indices.item()))
    assert counts == [0, 0, 0, 0, 1, 0]
