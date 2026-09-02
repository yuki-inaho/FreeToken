"""FREETOKEN_GPU_MEM_CAP_MB must hard-cap the allocator, not merely steer the budget solver.

--gpu-reserve-mb shrinks what --moe-cache-auto gives the pools and does nothing about transient
allocations, so a configuration that overflows a 16 GB card still runs on a 32 GB one. The cap
is what makes "does this fit?" answerable; these tests pin its contract without needing a GPU.
"""

from __future__ import annotations

import pytest


def test_cap_is_applied_before_any_allocation():
    """It must run before _adjust_config and the first CUDA allocation, or the weights are
    already resident and the cap covers only what comes after."""
    import inspect

    from freetoken.engine import engine as eng

    src = inspect.getsource(eng.Engine.__init__)
    assert "_apply_gpu_mem_cap" in src
    assert src.index("_apply_gpu_mem_cap") < src.index("_adjust_config(config)")


def test_unset_or_zero_is_a_no_op(monkeypatch):
    """The default must not touch the allocator: capping is opt-in emulation, and a real card
    needs none."""
    from freetoken.engine.engine import _apply_gpu_mem_cap

    called = []
    monkeypatch.setattr(
        "torch.cuda.set_per_process_memory_fraction",
        lambda *a, **k: called.append(a),
    )
    for value in ("", "0"):
        monkeypatch.setenv("FREETOKEN_GPU_MEM_CAP_MB", value)
        _apply_gpu_mem_cap("cuda:0")
    monkeypatch.delenv("FREETOKEN_GPU_MEM_CAP_MB", raising=False)
    _apply_gpu_mem_cap("cuda:0")
    assert called == []


@pytest.mark.skipif(
    not __import__("torch").cuda.is_available(), reason="needs a device to read total memory"
)
def test_cap_sets_the_fraction_against_total_memory(monkeypatch):
    """torch caps against TOTAL device memory, not free memory -- a fraction computed against
    free would cap far too low on a busy card."""
    import torch

    from freetoken.engine.engine import _apply_gpu_mem_cap

    total_mib = torch.cuda.get_device_properties(0).total_memory >> 20
    want_mib = total_mib // 4
    seen = {}
    monkeypatch.setattr(
        "torch.cuda.set_per_process_memory_fraction",
        lambda frac, dev: seen.update(frac=frac),
    )
    monkeypatch.setenv("FREETOKEN_GPU_MEM_CAP_MB", str(want_mib))
    _apply_gpu_mem_cap(torch.device("cuda", 0))
    assert seen["frac"] == pytest.approx(want_mib / total_mib, rel=1e-3)


@pytest.mark.skipif(
    not __import__("torch").cuda.is_available(), reason="needs a device to read total memory"
)
def test_cap_larger_than_the_device_is_ignored(monkeypatch):
    """Asking for more than the card has is a no-op with a warning, not a fraction > 1."""
    import torch

    from freetoken.engine.engine import _apply_gpu_mem_cap

    total_mib = torch.cuda.get_device_properties(0).total_memory >> 20
    called = []
    monkeypatch.setattr(
        "torch.cuda.set_per_process_memory_fraction", lambda *a: called.append(a)
    )
    monkeypatch.setenv("FREETOKEN_GPU_MEM_CAP_MB", str(total_mib * 2))
    _apply_gpu_mem_cap(torch.device("cuda", 0))
    assert called == []
