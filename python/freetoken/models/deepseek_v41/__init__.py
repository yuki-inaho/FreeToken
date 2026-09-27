"""DeepSeek-V4.1-Flash support for FreeToken.

Sibling of :mod:`freetoken.models.deepseek_v4` (the 0731 port) sharing its leaf pieces
(fp8/bf16 linears, rope/YaRN/sinkhorn ops, the ``ds_fp4`` expert banks and the offload MoE
layer) and rewriting the topology: the Causal Encoder-Decoder split, CSA2 compressed-KV
sharing with the hierarchical indexer candidate pool, single-pass hyper-connections, the
Engram n-gram tables (served from NVMe by :mod:`freetoken.engram`) and the 128-expert DSpark
stage MoE.

The model class is imported lazily: the args / config / weight-plan surface is torch-light
and used by tooling (converter refusal, host-RAM preflight, tests) that must not pull the
Triton-backed model code.
"""

from __future__ import annotations

from .args import DeepseekV41Args, load_args, parse_raw_args
from .config import dspark_build_enabled, extend_config_for_speculative, parse_config
from .weight import (
    dspark_bank_bytes_estimate,
    iter_weights,
    load_dsfp4_expert_sources,
    load_dsfp4_expert_sources_parallel,
    load_dspark_expert_bank_layers,
    resident_tensor_plan,
)

__all__ = [
    "DeepseekV41Args",
    "DeepseekV41ForCausalLM",
    "load_args",
    "parse_raw_args",
    "parse_config",
    "dspark_build_enabled",
    "extend_config_for_speculative",
    "iter_weights",
    "resident_tensor_plan",
    "load_dsfp4_expert_sources",
    "load_dsfp4_expert_sources_parallel",
    "load_dspark_expert_bank_layers",
    "dspark_bank_bytes_estimate",
]


def __getattr__(name: str):
    if name == "DeepseekV41ForCausalLM":
        from .model import DeepseekV41ForCausalLM

        return DeepseekV41ForCausalLM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
