"""Speculative-decoding step state (MTP verify/draft cycle, batch size 1).

A verify step runs the target model over ``[last_accepted_token, draft_1..k]`` as one ragged
``phase="decode"`` batch with ``Batch.spec`` set. The recurrent GDN update runs with the live
state write-back disabled and every per-token post-state recorded (``gdn_verify_fla``); the
in-place conv/PLE/ngram advances are allowed to overshoot to k+1 tokens because ``commit``
rewrites them from stashes to the accepted prefix afterwards. All commit arithmetic is
device-side, driven by the accepted-input count as a device scalar, so it stays capture-safe
for the later CUDA-graph milestone.

Rollback invariants per state family:
- GDN recurrent: untouched during verify; commit copies the accepted step's intermediate.
- GDN conv:     state = last (kernel-1) RAW conv inputs. Stash the pre-verify window and this
                step's conv inputs; commit selects the window ending at the accepted token.
- PLE conv:     same shape of argument over the post-norm_conv activation stream ``x``.
- PLE ngram:    last (ngram_size-1) token ids; recomputed from the token pool at commit.
- QSA KV/slabs: no commit needed -- the ``device_len`` rewind makes rejected rows unreachable
                and re-issued positions overwrite them (ring widened via ring_capacity_for).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from freetoken.utils import init_logger

logger = init_logger(__name__)

_SPEC_DEBUG_STEPS = int(os.getenv("FREETOKEN_SPEC_DEBUG", "0"))
_SPEC_TIME = os.getenv("FREETOKEN_SPEC_TIME", "0") != "0"
# Host-side wall breakdown per step (perf_counter, no GPU syncs added): inter-step gap,
# split into the drain's copy_done wait, the drain's own host work, and the scheduler
# remainder (loop + prepare + staging). Printed as medians every 200 steps.
_SPEC_HOST_TIME = os.getenv("FREETOKEN_SPEC_HOST_TIME", "0") != "0"
# Section-level breakdown of the host half of verify_step (which sections of Python /
# eager-op dispatch the ~1.8 ms of `verify_host` is actually spent in). Costs ~100 ns per
# mark, i.e. <0.2% of a step, and only when enabled.
_SPEC_HOST_DETAIL = os.getenv("FREETOKEN_SPEC_HOST_DETAIL", "0") != "0"
# =2 drains the CUDA launch queue before the section clock starts. Without it, a host that
# runs ~12 ms ahead of the GPU blocks inside cudaLaunchKernel whenever the queue is full, and
# that GPU wait is charged to whichever host section happened to launch -- so the sections
# read as "expensive Python" when they are really "waiting for the GPU". With the queue
# drained, the section times are the true host cost (the step itself gets slower, since the
# host/GPU overlap is given up; use it only to attribute, never to quote step_wall).
_SPEC_HOST_DETAIL_SYNC = os.getenv("FREETOKEN_SPEC_HOST_DETAIL", "0") == "2"
# torch.profiler window over whole verify steps: FREETOKEN_SPEC_PROFILE="<skip>:<steps>"
# starts the profiler at the top of step `skip` and stops it at the top of step
# `skip+steps`, exporting a chrome trace (scripts/spec-trace-report.py consumes it) to
# FREETOKEN_SPEC_PROFILE_DIR (default: cwd). FREETOKEN_SPEC_NSYS with the same syntax
# brackets the window with cudaProfilerStart/Stop for `nsys --capture-range=cudaProfilerApi`.
def _parse_window(name: str) -> tuple[int, int] | None:
    raw = os.getenv(name)
    if not raw:
        return None
    skip, _, steps = raw.partition(":")
    return int(skip), int(steps or "20")


_SPEC_PROFILE = _parse_window("FREETOKEN_SPEC_PROFILE")
# Adaptive draft count thresholds on the per-request draft-acceptance EMA (fraction of the
# staged drafts accepted per step): step k up at/above UP, down below DOWN (hysteresis).
_ADAPT_UP = float(os.getenv("FREETOKEN_SPEC_ADAPT_UP", "0.70"))
_ADAPT_DOWN = float(os.getenv("FREETOKEN_SPEC_ADAPT_DOWN", "0.55"))
_ADAPT_ALPHA = float(os.getenv("FREETOKEN_SPEC_ADAPT_ALPHA", "0.25"))


def update_accept_ema(req, k_used: int, accepted: int) -> None:
    """Scheduler-side (drain) update of ``req.spec_ema`` after a verify with ``k_used``
    staged drafts accepted ``accepted`` of them."""
    if k_used <= 0:
        return
    rate = min(max(accepted / k_used, 0.0), 1.0)
    req.spec_ema = rate if req.spec_ema is None else (
        _ADAPT_ALPHA * rate + (1.0 - _ADAPT_ALPHA) * req.spec_ema
    )
# Fused single-pass GDN state commit (FREETOKEN_SPEC_FUSED_COMMIT=0 restores the two torch ops).
_FUSED_GDN_COMMIT = os.getenv("FREETOKEN_SPEC_FUSED_COMMIT", "1") != "0"
_SPEC_NSYS = _parse_window("FREETOKEN_SPEC_NSYS")

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class SpecStateScratch:
    """Per-step device buffers for one speculative request (bs=1)."""

    def __init__(
        self,
        model_config: ModelConfig,
        num_draft_tokens: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.k1 = num_draft_tokens + 1
        linear = model_config.linear_attention_group()
        self.num_linear = len(linear.layer_ids)
        conv_dim = (
            2 * linear.num_key_heads * linear.key_head_dim
            + linear.num_value_heads * linear.value_head_dim
        )
        km1 = linear.conv_kernel_dim - 1
        hv, k, v = linear.num_value_heads, linear.key_head_dim, linear.value_head_dim
        from freetoken.kvcache.linear_state_pool import ssm_state_dtype

        # Per-GDN-layer, per-token post-states in the fused kernel's slot layout (identical
        # to the pool's because head_k_dim == head_v_dim -- see gdn.py's transposition note).
        self.gdn_states = torch.zeros(
            self.num_linear, 1, self.k1, hv, k, v, device=device, dtype=ssm_state_dtype()
        )
        # Stochastic sessions: the renormalized draft distribution each staged draft was
        # sampled from (one row per draft), consumed by the next verify's chain acceptance.
        self.draft_probs = torch.zeros(
            max(num_draft_tokens, 1), model_config.vocab_size, device=device, dtype=torch.float32
        )
        self.gdn_state_indices = torch.zeros(1, device=device, dtype=torch.int32)
        self.conv_pre = torch.zeros(self.num_linear, conv_dim, km1, device=device, dtype=dtype)
        self.conv_in = torch.zeros(self.num_linear, conv_dim, self.k1, device=device, dtype=dtype)
        # PLE stashes, one per PLE layer id (empty dicts for models without PLE).
        self.ple_pre: dict[int, torch.Tensor] = {}
        self.ple_x: dict[int, torch.Tensor] = {}
        self._ple_device, self._ple_dtype = device, dtype
        # The main tower's pre-mixer residual rows for the draft pass (set by the model).
        self.premix_rows: torch.Tensor | None = None
        # The MTP layer's own 4-stream output rows from the draft pass (filled via
        # ``mtp.forward(..., r1_out=...)``): row i stands in for the next position's premix
        # when chaining a second draft (k >= 2). Allocated by the executor (width is the
        # model's premix width, unknown here).
        self.r1_rows: torch.Tensor | None = None

    # ----- stashes (called from inside the verify forward) --------------------------------

    def stash_gdn(
        self, local_idx: int, conv_states: torch.Tensor, slot: torch.Tensor, conv_in: torch.Tensor
    ) -> None:
        """Record layer ``local_idx``'s pre-verify conv window and this step's raw conv
        inputs. ``conv_states`` is the pool's [num_slots, conv_dim, K-1] slab for the layer,
        ``slot`` the [1] device slot index, ``conv_in`` this forward's [T, conv_dim] rows
        (T <= k+1: the last verify of a nearly-exhausted budget can be shorter)."""
        self.conv_pre[local_idx].copy_(conv_states.index_select(0, slot).squeeze(0))
        self.conv_in[local_idx, :, : conv_in.shape[0]].copy_(conv_in.transpose(0, 1))

    def stash_ple(self, layer_id: int, state: torch.Tensor, x: torch.Tensor) -> None:
        """Record a PLE layer's pre-verify conv window ([1, width, state_len], as read by
        _prefill_conv) and this step's post-norm_conv activations ``x`` [T, width]."""
        if layer_id not in self.ple_pre:
            width, state_len = state.shape[-2], state.shape[-1]
            self.ple_pre[layer_id] = torch.zeros(
                width, state_len, device=self._ple_device, dtype=self._ple_dtype
            )
            self.ple_x[layer_id] = torch.zeros(
                width, self.k1, device=self._ple_device, dtype=self._ple_dtype
            )
        self.ple_pre[layer_id].copy_(state.squeeze(0))
        self.ple_x[layer_id][:, : x.shape[0]].copy_(x.transpose(0, 1))

    # ----- commit (after acceptance) ------------------------------------------------------

    def commit(
        self,
        pool,
        gdn_slot: torch.Tensor,
        ngram_slot: torch.Tensor | None,
        num_inputs: torch.Tensor,
        token_pool_row: torch.Tensor | None,
        new_device_len: torch.Tensor | None,
        ple_layer_ids: tuple[int, ...] = (),
        ngram_state_name: str = "ple_ngram_ctx",
        ple_state_name: str = "ple_conv",
    ) -> None:
        """Rewrite the per-request state to the accepted prefix. Device-side only.

        ``num_inputs`` [1] int64/int32 device scalar: how many of the k+1 verify input rows
        are valid (>= 1). ``gdn_slot`` [1] is the linear-state slot. ``token_pool_row`` is
        ``token_pool[table_idx]`` and ``new_device_len`` the post-commit device_len as a
        device scalar; both may be None for models without the PLE ngram state.
        """
        n = num_inputs.reshape(1).long()
        last = (n - 1).to(torch.int32)
        # GDN recurrent: the accepted step's intermediate, all layers in one launch that
        # reads the accepted row and writes the slot once (was index_select + index_copy_,
        # two full passes over ~113 MB).
        if self.gdn_states.is_cuda and _FUSED_GDN_COMMIT:
            from freetoken.kernel.triton.spec_commit import commit_gdn_states

            commit_gdn_states(self.gdn_states, pool.recurrent_states, n, gdn_slot)
        else:
            accepted = self.gdn_states[:, 0].index_select(1, last).squeeze(1)
            pool.recurrent_states.index_copy_(
                1, gdn_slot.long(), accepted.unsqueeze(1).to(pool.recurrent_states.dtype)
            )
        # GDN conv: window of the last (K-1) raw inputs ending at the accepted token.
        km1 = self.conv_pre.shape[-1]
        cols = torch.arange(km1, device=n.device) + n
        window = torch.cat([self.conv_pre, self.conv_in], dim=-1).index_select(-1, cols)
        pool.conv_states.index_copy_(
            1, gdn_slot.long(), window.unsqueeze(1).to(pool.conv_states.dtype)
        )
        # PLE conv windows, same argument over the activation stream.
        for layer_id in ple_layer_ids:
            states = pool.slot_state(ple_state_name, layer_id)
            state_len = states.shape[-1]
            pcols = torch.arange(state_len, device=n.device) + n
            pwin = torch.cat([self.ple_pre[layer_id], self.ple_x[layer_id]], dim=-1)
            states.index_copy_(
                0,
                (ngram_slot if ngram_slot is not None else gdn_slot).long(),
                pwin.index_select(-1, pcols).unsqueeze(0).to(states.dtype),
            )
        # PLE ngram context: the ids immediately before the next forward's first token.
        if ngram_slot is not None and pool.has_slot_state(ngram_state_name):
            ctx_pool = pool.slot_state(ngram_state_name)
            ctx_len = ctx_pool.shape[-1]
            positions = (new_device_len.reshape(1).long() - 1 - ctx_len
                         + torch.arange(ctx_len, device=n.device)).clamp_min(0)
            ctx = token_pool_row.index_select(0, positions)
            ctx_pool.index_copy_(0, ngram_slot.long(), ctx.unsqueeze(0).to(ctx_pool.dtype))


# Per captured verify/draft graph set: activations of a k+1-row forward through every layer
# plus the [k+1, vocab] fp32 logits. Measured on Qwen3.8 (boot logs, free VRAM before/after
# capture): decode graphs + BOTH spec sets (k=2,3) cost ~110 MiB together, so 64 MiB per set
# is ~2x the observed per-set cost.
SPEC_GRAPH_SET_RESERVE = 64 << 20


def spec_fixed_bytes(config) -> int:
    """VRAM the speculative executor allocates OUTSIDE the pools (scratch intermediates for
    k_max+1 rows, the draft distributions, and one graph set per draft count), so the cache
    budget can price it instead of leaving it to the (1 - memory_ratio) headroom."""
    if not getattr(config, "spec_enabled", False) or config.num_spec_tokens <= 0:
        return 0
    mc = config.model_config
    linear = mc.linear_attention_group()
    k_max = config.num_spec_tokens
    k1 = k_max + 1
    total = 0
    if linear is not None:
        from freetoken.kvcache.linear_state_pool import ssm_state_dtype

        n_lin = len(linear.layer_ids)
        conv_dim = 2 * linear.num_key_heads * linear.key_head_dim + linear.num_value_heads * linear.value_head_dim
        hv, kd, vd = linear.num_value_heads, linear.key_head_dim, linear.value_head_dim
        total += n_lin * k1 * hv * kd * vd * ssm_state_dtype().itemsize          # gdn_states
        total += n_lin * conv_dim * (linear.conv_kernel_dim - 1 + k1) * config.dtype.itemsize  # conv stashes
    total += max(k_max, 1) * mc.vocab_size * 4                                    # draft_probs
    n_sets = k_max - config.num_draft_tokens + 1 if os.getenv("FREETOKEN_SPEC_GRAPH", "1") != "0" else 0
    total += n_sets * SPEC_GRAPH_SET_RESERVE
    return int(total)


@dataclass
class SpecBatchInfo:
    """Attached as ``Batch.spec`` on a verify batch."""

    scratch: SpecStateScratch
    num_input_tokens: int  # k+1
    # Schedule-time draft depth before the output budget clips the verify. The executor
    # overwrites req.spec_k_staged with the NEXT round's depth before this round drains.
    num_staged_drafts: int | None = None


class SpecGraphs:
    """CUDA graphs for the speculative cycle: graph A replays the k+1-row verify forward
    (logits into a static buffer), graph B the MTP draft pass + lm_head. The acceptance,
    state commit and token scatters stay eager between the two replays -- they are a
    handful of small ops, and keeping sampling outside graphs matches the engine's decode
    convention.

    Static inputs mirror ``GraphCaptureBuffer`` but (k+1)-token shaped: input_ids /
    positions / out_loc per token, the GDN slot and the ragged QSA metadata per request.
    Graph B reads graph A's captured pre-mixer intermediate directly (its address is
    stable across replays), so the two graphs form one pipeline. Capture shares the decode
    graphs' memory pool.
    """

    def __init__(self, executor: "SpecExecutor", k: int | None = None) -> None:
        engine = executor.engine
        self.engine = engine
        self.executor = executor
        self.k = executor.k if k is None else int(k)
        self.k1 = self.k + 1
        device = engine.device
        k1 = self.k1
        self.input_ids = torch.zeros(k1, dtype=torch.int32, device=device)
        self.positions = torch.zeros(k1, dtype=torch.int32, device=device)
        self.out_loc = torch.zeros(k1, dtype=engine.page_table.dtype, device=device)
        self.linear_slot = torch.zeros(1, dtype=torch.int32, device=device)
        self.cand = torch.zeros(k1, dtype=torch.int32, device=device)
        self._table_host = torch.zeros(
            1, dtype=torch.int64, pin_memory=torch.cuda.is_available()
        )
        self._pos_host = torch.zeros(
            k1, dtype=torch.int32, pin_memory=torch.cuda.is_available()
        )
        self._table_dev = torch.zeros(1, dtype=torch.int64, device=device)
        from freetoken.attention.linear import FLAMetadata

        self.fla = FLAMetadata(
            cu_seqlens=torch.tensor([0, k1], dtype=torch.int32, device=device),
            cache_indices=self.linear_slot,
            has_initial_state=torch.ones(1, dtype=torch.bool, device=device),
        )
        self.md = None
        self.graph_a = torch.cuda.CUDAGraph()
        self.graph_b = torch.cuda.CUDAGraph()
        self.logits: torch.Tensor | None = None
        self.dlogits: torch.Tensor | None = None
        # k >= 2: graph C replays the 1-row chained second draft (the eager pass is
        # launch-bound at ~3-4 ms; the replay is sub-millisecond).
        self.graph_c: torch.cuda.CUDAGraph | None = None
        self.dlogits2: torch.Tensor | None = None
        self.md_c = None
        if self.k >= 2:
            width = executor.scratch.r1_rows.shape[1]
            self.chain_r1 = torch.zeros(1, width, dtype=engine.dtype, device=device)
            # The chain row's own 4-stream layer output: the premix of the NEXT chain
            # row when k >= 3 (chain-of-chain).
            self.chain_r1_out = torch.zeros(1, width, dtype=engine.dtype, device=device)
            self.chain_tok = torch.zeros(1, dtype=torch.int32, device=device)
            self.chain_pos = torch.zeros(1, dtype=torch.int32, device=device)
            self.chain_out_loc = torch.zeros(1, dtype=engine.page_table.dtype, device=device)
        self._capture()

    def _capture(self) -> None:
        from freetoken.core import Batch, Req, SamplingParams, get_global_ctx

        engine = self.engine
        k1 = self.k1
        device = engine.device
        pool_slot = (
            engine.linear_state_pool.padding_slot
            if engine.linear_state_pool is not None
            else engine.dummy_req.table_idx
        )
        base = 128  # arbitrary capture depth; every position restages per replay
        capture_req = Req(
            input_ids=torch.zeros(base + 1, dtype=torch.int32),
            table_idx=engine.dummy_req.table_idx,
            cached_len=0,
            output_len=k1 + 1,
            uid=-2,
            sampling_params=SamplingParams(),
            cache_handle=None,
        )
        capture_req.linear_slot_idx = pool_slot
        capture_req.cached_len, capture_req.device_len = base, base + k1

        batch = Batch(reqs=[capture_req], phase="decode")
        batch.padded_reqs = batch.reqs
        batch.spec = SpecBatchInfo(scratch=self.executor.scratch, num_input_tokens=k1)
        batch.linear_table_idx = self.linear_slot
        batch.fla_metadata = self.fla
        batch.input_ids = self.input_ids
        batch.positions = self.positions
        batch.out_loc = self.out_loc
        self.positions.copy_(
            torch.arange(base, base + k1, dtype=torch.int32, device=device)
        )
        self.linear_slot.fill_(pool_slot)
        self.out_loc.copy_(engine.page_table[capture_req.table_idx, base : base + k1])
        # The ragged QSA metadata: built once here; its device tensors become the static
        # buffers, refreshed per replay by stage() (block_table/seq_lens/ring_slots) or by
        # the captured device-side plan recomputation (cmp/ring rows).
        engine.attn_backend.prepare_metadata(batch)
        self.md = batch.attn_metadata
        # Backends whose per-replay addressing is host-planned (FlashInfer) bind the spec
        # graphs to their own static-buffer wrappers; QSA restages device tensors in stage().
        if hasattr(engine.attn_backend, "prepare_spec_capture"):
            engine.attn_backend.prepare_spec_capture(batch, "verify")

        graphs = getattr(engine.graph_runner, "graph_map", {})
        shared_pool = next(iter(graphs.values())).pool() if graphs else None

        with get_global_ctx().forward_batch(batch):
            engine.model.forward()  # warmup: JIT/autotune outside the capture
            with torch.cuda.graph(self.graph_a, pool=shared_pool, stream=engine.stream):
                self.logits = engine.model.forward()
            # Graph B reads A's captured pre-mixer intermediate: take the reference AFTER
            # A's capture bound it (the model's premix seam ran inside the capture).
            premix = self.executor.scratch.premix_rows
            r1_out = self.executor.scratch.r1_rows
            engine.model.mtp.forward(
                premix, self.executor._embed(self.cand), batch, r1_out=r1_out
            )
            with torch.cuda.graph(
                self.graph_b,
                pool=shared_pool if shared_pool is not None else self.graph_a.pool(),
                stream=engine.stream,
            ):
                hidden = engine.model.mtp.forward(
                    premix, self.executor._embed(self.cand), batch, r1_out=r1_out
                )
                self.dlogits = self.executor._draft_lm_head().forward(hidden)

        if self.k >= 2:
            # Graph C: the chained second draft -- ONE row through the MTP layer at a
            # host-staged position. Shaped as a 1-row spec batch so the attention takes
            # the same ragged spec path graph A already captures and restages.
            chain_req = Req(
                input_ids=torch.zeros(base + k1 + 2, dtype=torch.int32),
                table_idx=engine.dummy_req.table_idx,
                cached_len=0,
                output_len=2,
                uid=-3,
                sampling_params=SamplingParams(),
                cache_handle=None,
            )
            chain_req.linear_slot_idx = pool_slot
            chain_req.cached_len, chain_req.device_len = base + k1, base + k1 + 1
            cbatch = Batch(reqs=[chain_req], phase="decode")
            cbatch.padded_reqs = cbatch.reqs
            cbatch.spec = SpecBatchInfo(scratch=self.executor.scratch, num_input_tokens=1)
            cbatch.linear_table_idx = self.linear_slot
            cbatch.fla_metadata = self.fla
            cbatch.input_ids = self.chain_tok
            cbatch.positions = self.chain_pos
            cbatch.out_loc = self.chain_out_loc
            self.chain_pos.fill_(base + k1)
            self.chain_out_loc.copy_(
                engine.page_table[chain_req.table_idx, base + k1 : base + k1 + 1]
            )
            engine.attn_backend.prepare_metadata(cbatch)
            self.md_c = cbatch.attn_metadata
            if hasattr(engine.attn_backend, "prepare_spec_capture"):
                engine.attn_backend.prepare_spec_capture(cbatch, "chain")
            with get_global_ctx().forward_batch(cbatch):
                engine.model.mtp.forward(
                    self.chain_r1,
                    self.executor._embed(self.chain_tok),
                    cbatch,
                    r1_out=self.chain_r1_out,
                )  # warmup outside the capture
                self.graph_c = torch.cuda.CUDAGraph()
                with torch.cuda.graph(
                    self.graph_c,
                    pool=shared_pool if shared_pool is not None else self.graph_a.pool(),
                    stream=engine.stream,
                ):
                    hidden2 = engine.model.mtp.forward(
                        self.chain_r1,
                        self.executor._embed(self.chain_tok),
                        cbatch,
                        r1_out=self.chain_r1_out,
                    )
                    self.dlogits2 = self.executor._draft_lm_head().forward(hidden2)
        engine.graph_runner._reset_moe_offload_cache()

    def stage(self, batch, req) -> None:
        """Stage this step's addressing into the static buffers (host-cheap, restage-per-
        replay like the decode graphs). Under the light-prepare path the scheduler skips
        its per-step metadata building entirely: a verify step's inputs are contiguous
        [cached_len, device_len) slices of the token pool and page table, derived here."""
        c, k1 = req.cached_len, self.k1
        self.input_ids.copy_(self.executor.token_pool[req.table_idx, c : c + k1])
        torch.arange(c, c + k1, dtype=torch.int32, out=self._pos_host)
        self.positions.copy_(self._pos_host, non_blocking=True)
        self.out_loc.copy_(self.engine.page_table[req.table_idx, c : c + k1])
        batch.input_ids = self.input_ids  # the acceptance reads draft ids from here
        batch.positions = self.positions
        batch.out_loc = self.out_loc
        self.linear_slot.fill_(self.executor._gdn_slot(req))
        md = self.md
        backend = self.engine.attn_backend
        if hasattr(backend, "stage_spec"):
            # Host-planned backend (FlashInfer): re-plan the verify graph's static buffers.
            backend.stage_spec(md, batch, "verify")
            batch.attn_metadata = md
            return
        self._table_host.fill_(req.table_idx)
        self._table_dev.copy_(self._table_host, non_blocking=True)
        md.block_table.copy_(
            (
                backend._block_base_view().index_select(0, self._table_dev)
                // backend.page_size
            ).to(md.block_table.dtype)
        )
        md.seq_lens.fill_(req.device_len)
        md.ring_slots.fill_(req.table_idx)


def renorm_probs(logits: torch.Tensor, temperature: float, top_k: int, top_p: float):
    """The request's sampling distribution, explicitly materialized: softmax at temperature,
    then top-k / top-p filtering RENORMALIZED. Chain speculative sampling needs the same
    post-filter distribution on both the draft and target sides (the joint sampling kernels
    filter internally without exposing it). Triton throughout (freetoken.kernel.triton.sampling)."""
    from freetoken.kernel.triton.sampling import (
        softmax,
        top_k_renorm_probs,
        top_p_renorm_probs,
    )

    probs = softmax(logits, max(float(temperature), 1e-6))
    if top_k and top_k > 0:
        probs = top_k_renorm_probs(probs, top_k)
    if top_p < 1.0:
        probs = top_p_renorm_probs(probs, top_p)
    return probs


def chain_accept(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    draft_ids: torch.Tensor,
    uniforms: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Branchless chain speculative acceptance (Leviathan et al.).

    ``target_probs`` [k+1, V] and ``draft_probs`` [k, V] are the RENORMALIZED distributions;
    ``draft_ids`` [k] the staged drafts; ``uniforms`` [k] the acceptance draws. Returns
    ``(n_acc, fallback)``: ``n_acc`` [] = accepted drafts (0..k); ``fallback`` [k+1, V] holds
    the renormalized residual max(p-q, 0) for each draft position and the bonus distribution
    (the last target row) at index k -- the correction token is one draw from row ``n_acc``,
    which preserves the target distribution exactly for any acceptance outcome."""
    k = draft_ids.shape[0]
    ids = draft_ids.view(-1, 1).to(torch.int64)
    p_d = target_probs[:k].gather(1, ids).squeeze(1)
    q_d = draft_probs.gather(1, ids).squeeze(1)
    # u <= p/q as u*q <= p: div-free, and a zero-q draft (cannot happen from a correct
    # draw, but fp renorm can round there) accepts iff the target gives it any mass.
    accept = (uniforms * q_d <= p_d).to(torch.int32)
    n_acc = accept.cumprod(0).sum()
    residual = (target_probs[:k] - draft_probs).clamp_min(0)
    residual = residual / residual.sum(-1, keepdim=True).clamp_min(1e-20)
    return n_acc, torch.cat([residual, target_probs[k:]], dim=0)


@dataclass
class SpecForwardOutput:
    """What a verify step hands the scheduler drain (duck-sibling of ForwardOutput).

    ``candidates`` holds all k+1 greedy target tokens (position ``cached_len + 1 + i`` for
    row i); the first ``num_emitted`` of them were emitted this step. All candidates are
    already scattered into the token pool, and the states are committed to the emitted
    prefix, so the drain only settles host bookkeeping.
    """

    candidates_gpu: torch.Tensor
    candidates_cpu: torch.Tensor
    num_emitted_cpu: torch.Tensor
    copy_done: torch.cuda.Event

    @property
    def next_tokens_gpu(self) -> torch.Tensor:  # ForwardOutput duck compat (unused for spec)
        return self.candidates_gpu


class SpecExecutor:
    """Drives the MTP verify/draft cycle for one speculative request (batch size 1).

    Owned by the engine; the scheduler wires ``token_pool`` after construction and gates
    which batches carry ``Batch.spec`` / ``spec_warmup``. Greedy-match acceptance only for
    now (the scheduler engages speculation only for greedy requests); distribution-preserving
    rejection sampling replaces `_greedy_accept` in the M3 milestone.
    """

    def __init__(self, engine) -> None:
        config = engine.config
        self.engine = engine
        self.model = engine.model
        self.device = engine.device
        self.dtype = engine.dtype
        # k: the base draft count (the launcher's --speculative-num-draft-tokens); k_max: the
        # largest count the adaptive policy may stage (== k when adaptation is off). Scratch,
        # graph inputs and the token budget are sized for k_max.
        self.k = config.num_draft_tokens
        self.k_max = config.num_spec_tokens
        self.scratch = SpecStateScratch(
            config.model_config, self.k_max, self.device, self.dtype
        )
        self.ple_layer_ids = tuple(
            getattr(getattr(config.model_config, "qwen4_args", None), "ple_layer_ids", ()) or ()
        )
        self.token_pool: torch.Tensor | None = None  # wired by the scheduler
        # R of each request's last prefilled row, carried between prefill chunks and consumed
        # by the warmup draft. Keyed by uid; dropped on finish/abort.
        self._premix_carry: dict[int, torch.Tensor] = {}
        k1 = self.k_max + 1
        pin = torch.cuda.is_available()
        self._cand_cpu = torch.zeros(k1, dtype=torch.int32, pin_memory=pin)
        self._n_cpu = torch.zeros((), dtype=torch.int32, pin_memory=pin)
        self._gen = torch.Generator(device=self.device)
        self._gen.manual_seed(0xF7EE)
        # Small device constants, created once: building a CUDA tensor from a Python int
        # mid-step is a pageable H2D copy, which stream-syncs the host (the host then
        # cannot enqueue the rest of the step while graph A runs).
        self._row_const: dict[int, torch.Tensor] = {}
        self._gdn_slot_host = torch.zeros(1, dtype=torch.int32, pin_memory=pin)
        self._gdn_slot_dev = torch.zeros(1, dtype=torch.int32, device=self.device)
        self._prof = None
        self._prof_step = 0
        mtp = getattr(self.model, "mtp", None)
        if mtp is not None:
            width = mtp.pre_fc_norm_hidden.weight.numel()
            self.scratch.r1_rows = torch.zeros(k1, width, device=self.device, dtype=self.dtype)
        # Speculative expert prefetch (--moe-prefetch mtp): predicts the NEXT step's
        # routed experts in the window-3 tail and copies them in on a side stream.
        self.prefetcher = None
        cache = engine.moe_offload_cache
        if (
            getattr(config, "moe_prefetch", "off") == "mtp"
            and cache is not None
            and cache.num_indices2 is not None
            and mtp is not None
        ):
            from freetoken.moe.prefetch import ExpertPrefetcher

            self.prefetcher = ExpertPrefetcher(
                cache,
                self.model,
                num_rows=k1,
                budget=cache.max_prefetch,
                topk=getattr(config, "moe_prefetch_topk", 4),
                device=self.device,
            )
            logger.info_rank0(
                f"Expert prefetch armed: budget {cache.max_prefetch} experts/step, "
                f"top-{self.prefetcher.topk} per (layer, row)"
            )
        # FREETOKEN_SPEC_HOST_TIME accumulators (medians printed every 200 steps).
        self._ht_sync: list[float] = []
        self._ht_drain: list[float] = []
        self._ht_sched: list[float] = []
        self._ht_verify: list[float] = []
        self._ht_wall: list[float] = []
        self._ht_last_exit: float | None = None
        self._ht_last_entry: float | None = None
        self._ht_detail: dict[str, list[float]] = {}
        self._hd_last: float = 0.0
        # CUDA graphs for the verify/draft cycle (the launch overhead of an eager 48-layer
        # verify step dwarfs its compute), one set per draft count in [k, k_max]. Falls back
        # to eager on capture failure.
        self.graph_sets: dict[int, SpecGraphs] = {}
        if os.getenv("FREETOKEN_SPEC_GRAPH", "1") != "0":
            self.capture_graphs()

    def _hd(self, name: str) -> None:
        """Charge the time since the previous mark to `name` (FREETOKEN_SPEC_HOST_DETAIL)."""
        now = time.perf_counter()
        self._ht_detail.setdefault(name, []).append(now - self._hd_last)
        self._hd_last = now

    def capture_graphs(self) -> None:
        """(Re)capture the verify/draft graph sets for every draft count in [k, k_max]."""
        self.graph_sets = {}
        try:
            for kk in range(self.k, self.k_max + 1):
                self.graph_sets[kk] = SpecGraphs(self, k=kk)
            logger.info_rank0(
                "Speculative verify/draft CUDA graphs captured"
                + (f" for k in {sorted(self.graph_sets)}" if len(self.graph_sets) > 1 else "")
            )
        except Exception:
            logger.warning(
                "Speculative CUDA-graph capture failed; running verify eagerly", exc_info=True
            )
            self.graph_sets = {}

    @property
    def graphs(self) -> "SpecGraphs | None":
        """The base-k graph set (None when running eagerly); assigning None drops every set."""
        return self.graph_sets.get(self.k)

    @graphs.setter
    def graphs(self, value) -> None:
        if value is None:
            self.graph_sets = {}
        else:
            self.graph_sets[value.k] = value

    def has_graph(self, k1: int) -> bool:
        return (k1 - 1) in self.graph_sets

    def _next_k(self, req) -> int:
        """Drafts to stage for ``req``'s next verify: the base k, or -- with adaptation on --
        one more/less than the current count once the request's running draft-acceptance
        rate (spec_ema, updated by the scheduler's drain) crosses the thresholds."""
        if self.k_max <= self.k:
            return self.k
        # Page-boundary clipping can temporarily stage fewer drafts than the base k.
        # Recover the configured depth as soon as the next allocated page permits it.
        cur = max(self.k, req.spec_k_staged or self.k)
        ema = req.spec_ema
        if ema is None:
            return cur
        if ema >= _ADAPT_UP and cur < self.k_max:
            return cur + 1
        if ema < _ADAPT_DOWN and cur > self.k:
            return cur - 1
        return cur

    def _draft_k_with_page_room(self, input_end: int, k: int) -> int:
        """Bound draft KV writes to pages allocated for this forward's input rows.

        Draft 1 comes from the ordinary MTP pass. Chaining drafts 2..k writes up
        to position input_end+k-2 on full acceptance, but allocate_paged owns only
        the pages covering [0, input_end). A shallower next verify crosses the
        boundary normally and allocates its own page; no speculative page lease or
        change to the accepted prefix is needed.
        """
        page_size = self.engine.config.page_size
        page_end = ((input_end + page_size - 1) // page_size) * page_size
        return min(k, page_end - input_end + 1)

    def drop(self, uid: int) -> None:
        self._premix_carry.pop(uid, None)

    # ----- shared helpers -----------------------------------------------------------------

    def _embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.model.model.embed_tokens.forward(token_ids.to(torch.int64))

    def _draft_lm_head(self):
        """The lm_head the DRAFT passes use: a model may carry a lighter (NVFP4) copy for
        them -- draft logits only steer proposals, verification keeps the real head."""
        return getattr(self.model, "draft_lm_head", None) or self.model.lm_head

    def _premix(self, batch=None) -> torch.Tensor:
        """This forward's pre-mixer residual rows. A graph-replayed decode batch never ran
        the eager stash, but its captured graph's static intermediate (keyed by padded batch
        size) holds this step's rows after every replay."""
        inner = self.model.model
        if batch is not None and self.engine.graph_runner.can_use_cuda_graph(batch):
            hidden = inner._premix_graph.get(batch.padded_size)
            assert hidden is not None, "graph batch without a captured premix stash"
            return hidden[: sum(r.extend_len for r in batch.reqs)]
        hidden = inner._premix_hidden
        assert hidden is not None, "the model must be built with the premix seam armed"
        return hidden

    def _gdn_slot_tensor(self, req, staged: bool) -> torch.Tensor:
        """The request's linear-state slot as a [1] device tensor without a host sync:
        the graph path already staged it (``SpecGraphs.stage`` fills ``linear_slot`` with a
        kernel); the eager path goes through a pinned buffer + non-blocking copy."""
        if staged and self.graphs is not None:
            return self.graphs.linear_slot
        self._gdn_slot_host.fill_(self._gdn_slot(req))
        self._gdn_slot_dev.copy_(self._gdn_slot_host, non_blocking=True)
        return self._gdn_slot_dev

    def _profile_tick(self) -> None:
        """FREETOKEN_SPEC_PROFILE / FREETOKEN_SPEC_NSYS window bookkeeping (top of a step)."""
        step = self._prof_step
        self._prof_step += 1
        if _SPEC_NSYS is not None:
            skip, steps = _SPEC_NSYS
            if step == skip:
                torch.cuda.cudart().cudaProfilerStart()
                logger.info_rank0(f"[spec-profile] nsys capture range started at step {step}")
            elif step == skip + steps:
                torch.cuda.synchronize()
                torch.cuda.cudart().cudaProfilerStop()
                logger.info_rank0(f"[spec-profile] nsys capture range stopped at step {step}")
        if _SPEC_PROFILE is not None:
            skip, steps = _SPEC_PROFILE
            if step == skip:
                from torch.profiler import ProfilerActivity, profile

                # FREETOKEN_SPEC_PROFILE_STACK=1 adds op shapes + Python stacks (heavier;
                # meant for an eager FREETOKEN_SPEC_GRAPH=0 boot, where every kernel is
                # attributable to its launching op -- graph replays record kernels only).
                stack = os.getenv("FREETOKEN_SPEC_PROFILE_STACK", "0") == "1"
                self._prof = profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=stack,
                    with_stack=stack,
                    profile_memory=False,
                )
                self._prof.start()
                logger.info_rank0(f"[spec-profile] torch.profiler started at step {step}")
            elif step == skip + steps and self._prof is not None:
                torch.cuda.synchronize()
                self._prof.stop()
                out_dir = os.getenv("FREETOKEN_SPEC_PROFILE_DIR", os.getcwd())
                os.makedirs(out_dir, exist_ok=True)
                path = os.path.join(out_dir, f"spec-trace-{os.getpid()}.json")
                self._prof.export_chrome_trace(path)
                self._prof = None
                logger.info_rank0(f"[spec-profile] chrome trace written: {path}")

    def _gdn_slot(self, req) -> int:
        return req.linear_slot_idx if req.linear_slot_idx is not None else req.table_idx

    def _shadow_batch(self, rows: list[tuple[int, int, int]]):
        """A minimal prefill-shaped batch for an MTP-only pass.

        ``rows`` is one ``(table_idx, start_pos, end_pos)`` triple per request; positions
        are the MTP rows to process (KV written at those positions, ragged causal attend).
        """
        from types import SimpleNamespace

        reqs = [
            SimpleNamespace(table_idx=t, cached_len=p0, device_len=p1, extend_len=p1 - p0)
            for t, p0, p1 in rows
        ]
        positions = torch.cat(
            [torch.arange(p0, p1, dtype=torch.int32, device=self.device) for _, p0, p1 in rows]
        )
        out_loc = torch.cat(
            [self.engine.page_table[t, p0:p1] for t, p0, p1 in rows]
        ).contiguous()
        batch = SimpleNamespace(
            reqs=reqs,
            padded_reqs=reqs,
            phase="prefill",
            is_prefill=True,
            is_decode=False,
            size=len(reqs),
            padded_size=len(reqs),
            positions=positions,
            out_loc=out_loc,
            attn_metadata=None,
            fla_metadata=None,
            linear_table_idx=None,
            active_table_idx=None,
            mm_embeds=None,
            spec=None,
        )
        self.engine.attn_backend.prepare_metadata(batch)
        return batch

    # ----- prefill: populate the MTP layer's KV over the prompt ---------------------------

    def prefill_mtp(self, batch) -> None:
        """Run the MTP layer over this prefill chunk's rows (one-row lag: row for position p
        needs the NEXT token, so each chunk's last row is carried and processed with the
        following chunk; the prompt's final row waits for the warmup draft)."""
        premix = self._premix()
        rows: list[tuple[int, int, int]] = []
        rows_r: list[torch.Tensor] = []
        rows_tok: list[torch.Tensor] = []
        offset = 0
        for req in batch.reqs:
            t = req.extend_len
            chunk_r = premix[offset : offset + t]
            chunk_ids = batch.input_ids[offset : offset + t]
            offset += t
            carry = self._premix_carry.get(req.uid)
            if carry is not None:
                r = torch.cat([carry, chunk_r[:-1]], dim=0)
                toks = chunk_ids
                p0 = req.cached_len - 1
            else:
                r = chunk_r[:-1]
                toks = chunk_ids[1:]
                p0 = req.cached_len
            self._premix_carry[req.uid] = chunk_r[-1:].clone()
            if r.shape[0] == 0:
                continue
            rows.append((req.table_idx, p0, p0 + r.shape[0]))
            rows_r.append(r)
            rows_tok.append(toks)
        if not rows:
            return
        shadow = self._shadow_batch(rows)
        self.model.mtp.forward(
            torch.cat(rows_r, dim=0), self._embed(torch.cat(rows_tok, dim=0)), shadow
        )

    # ----- warmup: the request's first decode step also stages the first draft ------------

    def warmup_draft(self, batch, next_tokens_gpu: torch.Tensor) -> None:
        """After a (normal) first decode step: run the MTP layer over the prompt's final row
        and the decode row, and stage the first draft token in the token pool."""
        req = batch.reqs[0]
        carry = self._premix_carry.pop(req.uid, None)
        assert carry is not None, "warmup requires the prefill premix carry"
        # complete_one already advanced device_len, while page allocation covered
        # the pre-complete input span. Never chain into an unallocated next page.
        k_warmup = self._draft_k_with_page_room(req.device_len - 1, self.k)
        req.spec_k_staged = k_warmup
        req.spec_ema = None
        premix_row = self._premix(batch)[:1]
        # positions: L-1 (prompt's last row, next token = this step's input) and L (this
        # decode row, next token = the just-sampled one). cached/device use PRE-complete
        # values: complete_one already ran, so device_len is input position + 2.
        pos_in = req.device_len - 2
        rows_r = torch.cat([carry, premix_row], dim=0)
        toks = torch.cat([batch.input_ids[:1], next_tokens_gpu[:1].to(torch.int32)])
        shadow = self._shadow_batch([(req.table_idx, pos_in - 1, pos_in + 1)])
        hidden = self.model.mtp.forward(
            rows_r, self._embed(toks), shadow, r1_out=self.scratch.r1_rows
        )
        dlogits = self._draft_lm_head().forward(hidden)
        draft = self._draw_draft(dlogits, torch.ones((), device=self.device, dtype=torch.int64),
                                 req.sampling_params)
        # next verify's draft slot: input position + 2 == current (post-complete) device_len
        self.token_pool[req.table_idx].index_copy_(
            0, torch.tensor([req.device_len], device=self.device), draft.view(1)
        )
        if k_warmup >= 2:
            # Chain warmup drafts 2..k the same way the steady-state tail does: each chain
            # row's premix is the previous pass's own 4-stream output (eager here -- warmup
            # runs once per request).
            d = draft
            src = self.scratch.r1_rows[1:2]
            for j in range(1, k_warmup):
                r1_tmp = torch.empty_like(src)
                shadow_j = self._shadow_batch(
                    [(req.table_idx, pos_in + j, pos_in + j + 1)]
                )
                hidden_j = self.model.mtp.forward(
                    src, self._embed(d), shadow_j, r1_out=r1_tmp
                )
                d = self._draw_draft(
                    self._draft_lm_head().forward(hidden_j), 0, req.sampling_params, slot=j
                )
                self.token_pool[req.table_idx].index_copy_(
                    0, torch.tensor([req.device_len + j], device=self.device), d.view(1)
                )
                src = r1_tmp

    def _draw_draft(self, dlogits: torch.Tensor, row, params, slot: int = 0) -> torch.Tensor:
        """The next draft token from the MTP logits' selected row: argmax for argmax
        sessions; otherwise a sample from the renormalized draft distribution, which is also
        stashed (``draft_probs[slot]``) for the next verify's chain acceptance."""
        from freetoken.kernel.triton.sampling import sampling_from_probs

        if not torch.is_tensor(row):
            key = int(row)
            row = self._row_const.get(key)
            if row is None:
                row = torch.full((1,), key, dtype=torch.int64, device=self.device)
                self._row_const[key] = row
        sel = row.reshape(1).to(torch.int64)
        if params.is_argmax:
            return torch.argmax(dlogits, dim=-1).index_select(0, sel).to(torch.int32)
        dprobs = renorm_probs(dlogits, params.temperature, params.top_k, params.top_p)
        self.scratch.draft_probs[slot].copy_(dprobs.index_select(0, sel).squeeze(0))
        return sampling_from_probs(dprobs, indices=sel).to(torch.int32)

    # ----- verify step --------------------------------------------------------------------

    def verify_step(self, batch, sample_args) -> SpecForwardOutput:
        """One speculative step: verify the staged draft(s), commit states to the accepted
        prefix, scatter the emitted tokens, and stage the next draft.

        Argmax sessions use greedy-match acceptance; stochastic sessions use chain
        speculative sampling over the renormalized target/draft distributions, which
        preserves the per-token sampling distribution exactly."""
        del sample_args  # the executor derives the distribution from req.sampling_params
        from freetoken.kernel.triton.sampling import sampling_from_probs

        req = batch.reqs[0]
        params = req.sampling_params
        k1 = batch.spec.num_input_tokens
        if _SPEC_PROFILE is not None or _SPEC_NSYS is not None:
            self._profile_tick()
        if _SPEC_HOST_TIME:
            if _SPEC_HOST_DETAIL_SYNC:
                torch.cuda.synchronize()
            self._ht_last_entry = time.perf_counter()
            self._hd_last = self._ht_last_entry
            if self._ht_last_exit is not None:
                gap = self._ht_last_entry - self._ht_last_exit
                if self._ht_verify:
                    self._ht_wall.append(gap + self._ht_verify[-1])
                tracked = (self._ht_sync[-1] if self._ht_sync else 0.0) + (
                    self._ht_drain[-1] if self._ht_drain else 0.0
                )
                self._ht_sched.append(gap - tracked)
        c = req.cached_len
        pool_row = self.token_pool[req.table_idx]

        g = self.graph_sets.get(k1 - 1)
        use_graph = g is not None
        # Drafts to stage for the NEXT verify (adaptive: from the acceptance EMA the
        # scheduler updated through the previous drain); recorded on the request so the
        # scheduler stages exactly that many rows.
        k_next = self._draft_k_with_page_room(req.device_len, self._next_k(req))
        req.spec_k_staged = k_next
        if _SPEC_TIME:
            evts = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
            evts[0].record()
        if use_graph:
            g.stage(batch, req)
            # Graph A is the verify pass over the MAIN tower, so it contains the PLE layer and
            # needs its host-resolved cache slots published first (same contract as
            # GraphRunner.replay). Graphs B and C chain the MTP head only, which has no PLE
            # layer -- ple_layer_ids indexes the main tower.
            prep = getattr(self.model, "prepare_host_tables_for_replay", None)
            if prep is not None:
                prep(batch)
            if _SPEC_HOST_DETAIL:
                self._hd("stage")
            g.graph_a.replay()
            logits = g.logits
        else:
            logits = self.model.forward()  # [k1, vocab] -- decode phase computes every row
        if _SPEC_HOST_DETAIL:
            self._hd("graphA_launch")
        draft_ids = batch.input_ids[1:k1]
        if params.is_argmax:
            cand = torch.argmax(logits, dim=-1).to(torch.int32)  # greedy target per row
            matches = (draft_ids == cand[: k1 - 1]).to(torch.int32)
            n = matches.cumprod(0).sum() + 1  # [] emitted count in [1, k1]
        else:
            target_probs = renorm_probs(logits, params.temperature, params.top_k, params.top_p)
            uniforms = torch.rand(k1 - 1, device=self.device, generator=self._gen)
            n_acc, fallback = chain_accept(
                target_probs, self.scratch.draft_probs[: k1 - 1], draft_ids, uniforms
            )
            n = n_acc + 1
            # One draw covers every outcome: the residual of the first rejected position,
            # or the bonus distribution when every draft was accepted.
            correction = sampling_from_probs(
                fallback, indices=n_acc.reshape(1).to(torch.int64)
            ).to(torch.int32)
            cand = torch.zeros(k1, dtype=torch.int32, device=self.device)
            cand[: k1 - 1] = draft_ids
            cand.scatter_(0, n_acc.reshape(1).to(torch.int64), correction)

        if _SPEC_HOST_DETAIL:
            self._hd("accept")
        # All candidates go to the pool: cand[i] is the token at position c+1+i. Beyond the
        # accepted run they are placeholders that the next steps' scatters overwrite before
        # any gather reads them (single speculative request).
        pool_row[c + 1 : c + 1 + k1] = cand
        gdn_slot = self._gdn_slot_tensor(req, staged=use_graph)
        new_device_len = (n.to(torch.int64) + (c + 1)).reshape(1)
        self.scratch.commit(
            self.engine.linear_state_pool,
            gdn_slot=gdn_slot,
            ngram_slot=gdn_slot if self.ple_layer_ids else None,
            num_inputs=n,
            token_pool_row=pool_row,
            new_device_len=new_device_len,
            ple_layer_ids=self.ple_layer_ids,
        )

        if _SPEC_HOST_DETAIL:
            self._hd("commit")
        # The emitted run is final here (nothing below touches cand or n), so enqueue the
        # D2H copies now: copy_done then fires before the draft pass and the scheduler's
        # drain (detokenize, bookkeeping, next-step staging) overlaps with graph B instead
        # of idling behind it.
        self._cand_cpu[:k1].copy_(cand, non_blocking=True)
        self._n_cpu.copy_(n, non_blocking=True)
        copy_done = torch.cuda.Event()
        copy_done.record(self.engine.stream)
        if self.prefetcher is not None:
            # Score the previous step's prefetch plan (device-side counters) after the
            # join (forward_batch) and before this step's run() rewrites plan-2.
            self.prefetcher.used_check()

        if _SPEC_HOST_DETAIL:
            self._hd("d2h")
        if _SPEC_TIME:
            evts[1].record()
        # Draft pass: chain the MTP layer over all k1 rows (row i: R at position c+i, next
        # token cand[i]); rows past the accepted run write KV the device_len rewind orphans.
        if use_graph:
            g.cand.copy_(cand)
            g.graph_b.replay()
            dlogits = g.dlogits
        else:
            hidden = self.model.mtp.forward(
                self.scratch.premix_rows,
                self._embed(cand),
                batch,
                r1_out=self.scratch.r1_rows,
            )
            dlogits = self._draft_lm_head().forward(hidden)
        if _SPEC_HOST_DETAIL:
            self._hd("graphB")
        if _SPEC_TIME:
            evts[2].record()
        if k_next >= 2:
            # Chain the second draft here, all device-side (no host accept count needed):
            # d1 from dlogits row n-1 (device gather, as in k=1), then the chained MTP row
            # replayed as graph C with device-computed staging -- the row sits at position
            # c+n (= new_device_len-1), its premix is the draft pass's own output row n-1
            # (harness-validated chain wiring), and its visible KV length is new_device_len.
            # Everything below is enqueue-only, so the whole tail overlaps the drain.
            sel = (n - 1).reshape(1).to(torch.int64)
            next_draft = self._draw_draft(dlogits, n - 1, params, slot=0)
            pool_row.index_copy_(0, new_device_len, next_draft)
            if g is None:
                # eager verify: any captured set's chain graph serves (the chain row is
                # a 1-row pass, identical across k)
                g = next(iter(self.graph_sets.values()), None)
            row_cap = pool_row.shape[0] - 1
            if g is not None and g.graph_c is not None:
                # Drafts 2..k chain through graph C: replay j's premix is replay j-1's own
                # output row (chain_r1_out), position/KV length advance by one row.
                backend = self.engine.attn_backend
                host_planned = hasattr(backend, "stage_spec")
                page_row = self.engine.page_table[req.table_idx]
                if host_planned:
                    # FlashInfer plans KV ranges host-side, where the accepted count is not
                    # known yet: plan rows [0, c] plus one self slot, then patch that slot's
                    # page index device-side to the chain row's true (device-computed)
                    # position. The row attends every verified row plus itself; the up-to-k
                    # accepted-run rows between are the one approximation.
                    from types import SimpleNamespace

                    cb = SimpleNamespace(
                        padded_reqs=[SimpleNamespace(
                            table_idx=req.table_idx, cached_len=c + 1, device_len=c + 2,
                            extend_len=1,
                        )],
                        is_decode=True,
                        is_prefill=False,
                        spec=batch.spec,
                        attn_metadata=None,
                    )
                    backend.stage_spec(g.md_c, cb, "chain")
                    idx_buf = backend.spec_indices_buffer("chain")
                else:
                    g._table_host.fill_(req.table_idx)
                    g._table_dev.copy_(g._table_host, non_blocking=True)
                    g.md_c.block_table.copy_(
                        (
                            backend._block_base_view().index_select(0, g._table_dev)
                            // backend.page_size
                        ).to(g.md_c.block_table.dtype)
                    )
                    g.md_c.ring_slots.fill_(req.table_idx)
                if _SPEC_HOST_DETAIL:
                    self._hd("chain_plan")
                d = next_draft
                for j in range(1, k_next):
                    g.chain_tok.copy_(d)
                    g.chain_r1.copy_(
                        self.scratch.r1_rows.index_select(0, sel)
                        if j == 1
                        else g.chain_r1_out
                    )
                    pos = new_device_len + (j - 2)  # chain row j sits at c+n+j-1
                    g.chain_pos.copy_(pos.to(torch.int32))
                    out_loc = page_row.index_select(0, torch.clamp(pos, max=row_cap))
                    g.chain_out_loc.copy_(out_loc)
                    if host_planned:
                        idx_buf[c + 1 : c + 2].copy_(out_loc.to(idx_buf.dtype))
                    else:
                        g.md_c.seq_lens.copy_(
                            (pos + 1).to(g.md_c.seq_lens.dtype).expand_as(g.md_c.seq_lens)
                        )
                    g.graph_c.replay()
                    d = self._draw_draft(g.dlogits2, 0, params, slot=j)
                    pool_row.index_copy_(
                        0, torch.clamp(new_device_len + j, max=row_cap), d
                    )
                if _SPEC_HOST_DETAIL:
                    self._hd("chain_loop")
            else:
                # No chain graph (eager/debug mode): same-pass drafts from consecutive
                # dlogits rows -- still sound (each token is drawn from exactly the
                # distribution that is stashed), just weaker past full acceptance.
                for j in range(1, k_next):
                    d = self._draw_draft(
                        dlogits, torch.clamp(n - 1 + j, max=k1 - 1), params, slot=j
                    )
                    pool_row.index_copy_(
                        0, torch.clamp(new_device_len + j, max=row_cap), d
                    )
        else:
            next_draft = self._draw_draft(dlogits, n - 1, params)
            pool_row.index_copy_(0, new_device_len, next_draft)
        if self.prefetcher is not None:
            # Window-3 tail: the NEXT step's verify tokens are all in the pool now
            # (bonus cand at new_device_len-1, d1..dk after it) and the MTP hidden
            # states for the first rows are live -- predict, admit, and start the
            # side-stream copy so it overlaps the drain (enqueue-only, no host sync).
            gr = g
            self.prefetcher.run(
                pool_row,
                new_device_len,
                (n - 1).reshape(1).to(torch.int64),
                self.scratch.r1_rows,
                gr.chain_r1_out
                if (k_next >= 2 and gr is not None and gr.graph_c is not None)
                else None,
            )
        if _SPEC_TIME:
            evts[3].record()
            evts[3].synchronize()
            self._times = getattr(self, "_times", [])
            self._times.append(tuple(evts[i].elapsed_time(evts[i + 1]) for i in range(3)))
            if len(self._times) % 200 == 0:
                import statistics
                cols = list(zip(*self._times[-200:]))
                print("[spec-time ms] " + "  ".join(
                    f"{name}={statistics.median(col):.2f}"
                    for name, col in zip(("A+accept+commit", "B", "draw+scatter"), cols)),
                    flush=True)

        if _SPEC_DEBUG_STEPS:
            self._debug_steps = getattr(self, "_debug_steps", 0) + 1
            if self._debug_steps <= _SPEC_DEBUG_STEPS:
                torch.cuda.synchronize()
                drafted = "deferred" if next_draft is None else int(next_draft.item())
                print(
                    f"[spec {self._debug_steps}] c={c} inputs={batch.input_ids.tolist()} "
                    f"cand={cand.tolist()} n={int(n.item())} next_draft={drafted} "
                    f"pool[c:c+5]={pool_row[c:c+5].tolist()}",
                    flush=True,
                )

        if _SPEC_HOST_DETAIL:
            self._hd("tail")
        if _SPEC_HOST_TIME:
            self._ht_last_exit = time.perf_counter()
            self._ht_verify.append(self._ht_last_exit - self._ht_last_entry)
            if len(self._ht_wall) % 200 == 0 and self._ht_wall:
                import statistics as _st

                rows = {
                    "step_wall": self._ht_wall,
                    "verify_host": self._ht_verify,
                    "sync_wait": self._ht_sync,
                    "drain_host": self._ht_drain,
                    "sched_rest": self._ht_sched,
                }
                print(
                    "[spec-host ms] "
                    + "  ".join(
                        f"{k}={_st.median(v[-200:]) * 1e3:.2f}" for k, v in rows.items() if v
                    ),
                    flush=True,
                )
                if self._ht_detail:
                    print(
                        "[spec-host detail us] "
                        + "  ".join(
                            f"{k}={_st.median(v[-200:]) * 1e6:.0f}"
                            for k, v in sorted(
                                self._ht_detail.items(),
                                key=lambda kv: -_st.median(kv[1][-200:]),
                            )
                            if v
                        ),
                        flush=True,
                    )

        return SpecForwardOutput(
            candidates_gpu=cand,
            candidates_cpu=self._cand_cpu[:k1],
            num_emitted_cpu=self._n_cpu,
            copy_done=copy_done,
        )

    def recommit(self, req, num_inputs: int) -> None:
        """Re-run the state commit at a smaller accepted count (drain-side truncation: EOS,
        stop string or length cap inside the emitted run). The scratch still holds this
        step's intermediates, so the states realign exactly with the kept prefix."""
        n = torch.tensor([num_inputs], device=self.device, dtype=torch.int32)
        gdn_slot = self._gdn_slot_tensor(req, staged=False)
        new_device_len = torch.tensor(
            [req.cached_len + num_inputs + 1], device=self.device, dtype=torch.int64
        )
        self.scratch.commit(
            self.engine.linear_state_pool,
            gdn_slot=gdn_slot,
            ngram_slot=gdn_slot if self.ple_layer_ids else None,
            num_inputs=n,
            token_pool_row=self.token_pool[req.table_idx],
            new_device_len=new_device_len,
            ple_layer_ids=self.ple_layer_ids,
        )


__all__ = ["SpecBatchInfo", "SpecExecutor", "SpecForwardOutput", "SpecStateScratch"]
