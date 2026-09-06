from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, List

import torch
from freetoken.distributed import DistributedInfo
from freetoken.models.register import _load_attr, get_model_spec
from freetoken.utils import cached_load_hf_config

if TYPE_CHECKING:
    from freetoken.models import ModelConfig


@dataclass(frozen=True)
class EngineConfig:
    model_path: str
    tp_info: DistributedInfo
    dtype: torch.dtype
    max_running_req: int = 4
    attention_backend: str = "auto"
    moe_backend: str = "auto"
    # NVFP4 routed-expert GEMM backend (--nvfp4-backend): auto|marlin|flashinfer|triton.
    nvfp4_backend: str = "triton"
    # Expert-bank host load (--expert-load): auto|serial|parallel. "auto" reads scattered
    # experts in parallel but falls back to serial when free RAM can't cover the banks + the
    # parallel reader's extra (non-reclaimable) whole-shard buffer; "serial" forces the
    # low-memory reclaimable read; "parallel" forces the fast read.
    expert_load: str = "auto"
    moe_cache_size: int = 0
    moe_cache_rate: float | None = None
    moe_cache_auto: bool = False
    kv_reserve_tokens: int = 8192  # KV floor for --moe-cache-auto; small by design (MoE-priority)
    # Elastic KV pool (offloaded MoE): boot at kv_reserve_tokens (max expert slots) and grow the
    # pool up to kv_elastic_max_tokens through the idle-time runtime rebuild when a request needs
    # more context (paying expert slots), shrinking back to the base once a small request arrives
    # after kv_elastic_idle_s of inactivity. 0 disables.
    kv_elastic_max_tokens: int = 0
    kv_elastic_idle_s: float = 300.0
    # VRAM (MiB) kept free for OTHER processes on the GPU (desktop compositor, browsers, OBS...):
    # subtracted from the cache budget like a fixed cost. The (1 - memory_ratio) remainder only
    # covers this process's own prefill/graph transients.
    gpu_reserve_mb: int = 0
    moe_cache_policy: str = "lru"
    moe_prefill_overlap: bool = True
    # Prefill hit/miss split: serve cache-resident experts D2D during prefill
    # prefetch instead of re-streaming the full layer over PCIe. Needs CUDA >= 12.8
    # (cudaMemcpyBatchAsync); no-op unless moe_cache_size > 2 * num_experts.
    moe_prefill_hit_d2d: bool = False
    moe_collect_stats: bool = False  # capture decode miss-rate counters into the cuda graph
    # CPU MoE backend (--moe-backend cpu): number of CPU worker threads computing
    # the decode experts. 0 = auto (physical cores). Ignored by other backends.
    moe_cpu_threads: int = 0
    # Hybrid CPU/GPU decode (--moe-backend offload only): which MoE layers decode on
    # the CPU executor instead of the GPU offload/PCIe path. Spec is an explicit id
    # list ("3,7,11"), a count ("8" -> 8 layers evenly strided across depth), or a
    # fraction ("0.5"). None/"" = all layers on GPU (plain offload). --moe-backend cpu
    # already means all layers on CPU and ignores this.
    moe_cpu_layers: str | None = None
    # Hybrid MoE backend (--moe-backend hybrid): max experts fetched over PCIe per
    # (layer, decode step); the rest of that step's misses are computed on the CPU.
    # -1 (default) = auto: fetch the benched pcie_bw/cpu_bw fraction of each step's
    # misses so the PCIe fetch and the CPU compute finish together (perfect overlap);
    # falls back to a fixed cap of 1 without a usable `ft bench bw` profile.
    moe_hybrid_max_fetch: int = -1
    # Speculative expert prefetch (--moe-prefetch): off|mtp|layer. Predictions never
    # touch routing; requires the offload/hybrid MoE backend. "layer" = in-graph
    # cross-layer lookahead (layer J speculatively routes layer J+N with J+N's own
    # mix+gate and streams the predicted experts in while J..J+N-1 compute; captured
    # into the decode graphs). "mtp" = eager-window future-token predictor (measured
    # weak -- ~4% precision -- kept for experiments; needs --speculative-algorithm mtp,
    # downgraded to off with a warning otherwise). See moe/prefetch.py.
    moe_prefetch: str = "off"
    # Max experts admitted+copied per step by the prefetcher (the speculative bandwidth
    # cap). -1 = auto (35, sized to the ~3-4 ms idle-PCIe window of a spec step).
    moe_prefetch_budget: int = -1
    # Prediction candidates kept per (layer, verify row) before the global budget sort.
    moe_prefetch_topk: int = 4
    cuda_graph_bs: List[int] | None = None
    cuda_graph_max_bs: int | None = None
    page_size: int = 1
    memory_ratio: float = 0.9
    # Hybrid GDN models default to the HybridRadixCache (cross-request GDN-state prefix reuse);
    # `--cache-type naive` opts out. linear_state_cache_ratio sizes the GDN snapshot cache as
    # ceil(ratio * max_running_req) extra slots.
    linear_state_cache_ratio: float = 2.0
    # Explicit cross-request GDN snapshot-cache slot count (hybrid_radix); -1 = derive from the
    # ratio (floored at 4). Each slot is ~110 MiB on Qwen3.8 (= ~40 NVFP4 expert slots); 1 keeps
    # prefix reuse for the most recent conversation only.
    linear_state_cache: int = -1
    # Window/full ratio for the SWA radix cache (`--cache-type radix` on SWA models) and the DSV4
    # window tier: the DEFAULT window-pool size = max(working-set floor, ratio x full-pool tokens).
    # < 1.0 trades retained window-prefix capacity for memory savings; must be in (0, 1]. It is the
    # DSV4 window/full ratio directly. Used only when swa_num_pages_override is None (a runtime
    # rebuild can pin an absolute window instead).
    swa_full_tokens_ratio: float = 0.2
    # Absolute window-pool size in the pool's own pages (usable, dummy excluded); None -> use the
    # ratio default above. A runtime cache rebuild sets this (num_swa_pages) to pin the window
    # regardless of the full anchor; the ratio is the startup default and the fallback.
    swa_num_pages_override: int | None = None
    distributed_timeout: float = 60.0
    use_dummy_weight: bool = False
    use_pynccl: bool = True
    max_seq_len_override: int | None = None
    num_page_override: int | None = None  # if not None, will override the number of pages
    # KV capacity in tokens; resolved into num_page_override by _adjust_config once page_size
    # is final. Mutually exclusive with num_page_override.
    num_token_override: int | None = None
    # Speculative decoding (--speculative-algorithm): "off" | "mtp" | "dspark". "mtp" drafts
    # with the checkpoint's own MTP head, "dspark" with DeepSeek-V4-Flash-0731's block
    # drafter; both engage only while exactly one request is decoding. Requires a model
    # module exporting extend_config_for_speculative (and, for weight loading,
    # FREETOKEN_LOAD_MTP -- the server args set it).
    speculative_algorithm: str = "off"
    # Draft tokens per verify step (--speculative-num-draft-tokens); a verify forward runs
    # num_draft_tokens + 1 tokens per sequence.
    num_draft_tokens: int = 1
    # Adaptive draft count: when > num_draft_tokens, the executor captures graph sets for
    # every k in [num_draft_tokens, spec_adaptive_k_max] and picks the NEXT step's draft
    # count per request from its running draft-acceptance rate (code-like sessions with
    # high acceptance use more drafts; prose falls back to the base). 0 = fixed k.
    spec_adaptive_k_max: int = 0
    # KV-cache storage dtype (--kv-cache-dtype): "auto"|"bf16"|"fp8"|"nvfp4". "auto"/"bf16" store
    # the compute dtype; "fp8" stores the paged K/V slabs as e4m3 codes with a per-(token, kv
    # head) fp32 scale -- half the KV bytes; "nvfp4" packs two E2M1 codes per byte plus an
    # e4m3 scale per 16 elements and the same fp32 row global, 0.58x fp8. Both are implemented
    # for the QSA pool (Qwen3.8-Flash-Next) only. _adjust_config resolves the choice into
    # kv_dtype (a torch dtype; NVFP4's is a sentinel -- the slab is uint8) before pool sizing.
    # --moe-residency-routing TAU: under the NVMe expert tier only, substitute a routed
    # expert that is not in host RAM with the best near-tied one that is, when the swap
    # displaces <= TAU of the token's routed weight. 0 disables. Changes model outputs.
    moe_residency_routing: float = 0.0
    # How many of a row's missing experts one layer may substitute (1 = the shipped v1
    # behaviour). A read is charged per DISTINCT expert per (layer, step), so a row that keeps
    # one of its two misses still pays for it; raising this is what turns a high substitution
    # COVERAGE into fewer reads. Every swap is still gated on tau individually.
    moe_residency_routing_max_subs: int = 1
    # Bound on the SUM of one row's displacements (0 = same as tau), so raising max_subs does
    # not raise the per-token perturbation budget.
    moe_residency_routing_total: float = 0.0
    # The measured policy (``ft moe subst``): "auto" looks for the table beside the FTW corpus
    # and in the cache dir, "off" disables it, anything else is an explicit path. With a table
    # AND a non-zero eps the engine drops host-missing experts, gated on the fraction of the MoE
    # block output the drop moves -- the quantity that actually bounds the damage, rather than
    # the displaced routing weight the tau gate uses.
    moe_residency_routing_table: str = "auto"
    moe_residency_routing_eps: float = 0.0
    moe_residency_routing_eps_total: float = 0.0
    moe_residency_routing_drop_renorm: bool = False
    kv_cache_dtype: str = "auto"
    kv_dtype: torch.dtype | None = None
    # Resident dense-linear weight format (--dense-quant): "auto" | "bf16" keep the checkpoint's
    # dtypes; "fp8" quantizes Qwen3.8-Flash-Next's bf16 GDN/attention/hyper-connection/shared-
    # expert projections and lm_head to per-channel FP8 at load (FREETOKEN_DENSE_QUANT carries it
    # into parse_config and the engine workers; FREETOKEN_DENSE_FP8_PARTS narrows the set).
    dense_quant: str = "auto"

    @property
    def spec_enabled(self) -> bool:
        return self.speculative_algorithm != "off"

    @property
    def num_spec_tokens(self) -> int:
        """The MAXIMUM drafts per step (scratch / graph / budget sizing)."""
        if not self.spec_enabled:
            return 0
        return max(self.num_draft_tokens, self.spec_adaptive_k_max)

    @cached_property
    def hf_config(self):
        return cached_load_hf_config(self.model_path)

    @cached_property
    def model_config(self) -> ModelConfig:
        spec = get_model_spec(self.hf_config.architectures[0])
        parse_config = _load_attr(spec.module, spec.parse_config)
        config = parse_config(self.hf_config)
        if self.spec_enabled:
            try:
                extend = _load_attr(spec.module, "extend_config_for_speculative")
            except AttributeError as exc:
                raise ValueError(
                    f"--speculative-algorithm {self.speculative_algorithm} is not supported "
                    f"by {self.hf_config.architectures[0]}"
                ) from exc
            config = extend(config, self.num_draft_tokens)
        return config

    @property
    def max_seq_len(self) -> int:
        if self.max_seq_len_override is not None:
            return self.max_seq_len_override
        return self.model_config.rotary_config.max_position

    @property
    def max_forward_len(self) -> int:
        return self.max_seq_len

    @property
    def distributed_addr(self) -> str:
        return "tcp://127.0.0.1:2333"
