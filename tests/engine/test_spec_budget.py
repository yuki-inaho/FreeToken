"""spec_fixed_bytes: the speculative executor's out-of-pool VRAM, priced into the cache budget."""
from __future__ import annotations

from types import SimpleNamespace

import torch

from freetoken.engine.spec import SPEC_GRAPH_SET_RESERVE, spec_fixed_bytes


def _cfg(k_base=2, k_max=3, spec=True):
    from freetoken.models.qwen4_exp.config import parse_config
    from tests.models.qwen4_exp.common import toy_hf_config

    mc = parse_config(toy_hf_config(4))
    return SimpleNamespace(
        model_config=mc, spec_enabled=spec, num_spec_tokens=(k_max if spec else 0),
        num_draft_tokens=k_base, dtype=torch.bfloat16,
    )


def test_spec_fixed_bytes_off_without_speculation():
    assert spec_fixed_bytes(_cfg(spec=False)) == 0


def test_spec_fixed_bytes_scales_with_rows_and_graph_sets(monkeypatch):
    monkeypatch.delenv("FREETOKEN_SPEC_GRAPH", raising=False)
    fixed_k2 = spec_fixed_bytes(_cfg(k_base=2, k_max=2))
    adaptive = spec_fixed_bytes(_cfg(k_base=2, k_max=3))
    mc = _cfg().model_config
    lin = mc.linear_attention_group()
    n_lin = len(lin.layer_ids)
    hv, kd, vd = lin.num_value_heads, lin.key_head_dim, lin.value_head_dim
    one_row = n_lin * hv * kd * vd * 4                 # one more fp32 GDN intermediate row per layer
    assert adaptive - fixed_k2 >= one_row + SPEC_GRAPH_SET_RESERVE  # + the extra graph set
    # no graph sets when captures are disabled
    monkeypatch.setenv("FREETOKEN_SPEC_GRAPH", "0")
    assert spec_fixed_bytes(_cfg(k_base=2, k_max=3)) == adaptive - 2 * SPEC_GRAPH_SET_RESERVE
