from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, List, NamedTuple, NoReturn, Set, Tuple, TypeAlias

import torch
from freetoken.attention.linear import build_fla_metadata
from freetoken.core import Batch, Req
from freetoken.env import ENV
from freetoken.gpu_select import gpu_identity
from freetoken.message import (
    AbortBackendMsg,
    BaseBackendMsg,
    BatchBackendMsg,
    CacheRebuildBackendMsg,
    CacheRebuildResultMsg,
    DetokenizeMsg,
    ErrorReplyMsg,
    ExitMsg,
    PromptAdmittedMsg,
    UserMsg,
)
from freetoken.utils import (
    init_logger,
    load_eos_token_ids,
    load_tokenizer,
    load_toolcall_anchor_id,
)

from .cache import CacheManager
from .config import SchedulerConfig
from .decode import DecodeManager
from .io import SchedulerIOMixin
from .prefill import ChunkedReq, PrefillManager
from .status import SchedulerStatusReporter
from .table import TableManager

if TYPE_CHECKING:
    from freetoken.engine import BatchSamplingArgs, ForwardOutput


logger = init_logger(__name__)

Indice2D: TypeAlias = Tuple[torch.Tensor, torch.Tensor]

# Decode MoE cache stats cadence (steps per [moe-stats] line, 0 = off). Read once at
# import like the other FREETOKEN_* diagnostics; also force-enables cache.collect_stats
# at engine init (see _init_offload_moe_cache).
from freetoken.moe.offload_cache import moe_stats_log_interval

_MOE_STATS_LOG = moe_stats_log_interval()

# Diagnostic counters use values already copied for the scheduler drain. No device reads,
# extra events, or per-step logging; disabled by default for ordinary serving.
_SPEC_POSITION_STATS = os.getenv("FREETOKEN_SPEC_POSITION_STATS", "0") == "1"

# Shrink the elastic KV pool to the smallest ladder step a request needs, rather than only all
# the way to the base for a request that fits it. 0 restores the shrink-to-base-only policy.
_ELASTIC_STEPPED = os.environ.get("FREETOKEN_KV_ELASTIC_STEPPED", "1").strip().lower() not in (
    "0", "false", "off",
)


def _gib(n_bytes: int) -> str:
    return f"{n_bytes / (1 << 30):.2f} GiB"


# For overlap scheduling, we also need to cache some other data to avoid IMA
class ForwardInput(NamedTuple):
    batch: Batch
    sample_args: BatchSamplingArgs
    input_tuple: Indice2D  # (token_mapping, positions)
    write_tuple: Indice2D  # (req_mapping, seq_lens or -1)


ForwardData: TypeAlias = "Tuple[ForwardInput, ForwardOutput]"


def _record_spec_position_stats(
    req: "Req", *, verified_depth: int, staged_depth: int, num_emitted: int, kept: int
) -> None:
    """Count a drained verify using host integers, after terminal-token truncation.

    Array index zero is draft position 1. ``proposed`` counts drafts actually submitted
    to verification, excluding staged drafts clipped by the output budget. ``accepted``
    is prefix acceptance before EOS/stop/length truncation; ``useful_accepted`` counts
    accepted draft tokens retained by the scheduler (including a retained terminal token).
    The correction/bonus token is at index ``num_emitted - 1``, so an early terminal token
    in the accepted prefix retains ``min(num_emitted - 1, kept)`` accepted drafts.
    """
    stats = getattr(req, "stat_spec_positions", None)
    if stats is None:
        stats = req.stat_spec_positions = {
            "rounds": 0,
            "proposed": [],
            "accepted": [],
            "useful_accepted": [],
            "verified_depth_histogram": {},
            "staged_depth_histogram": {},
            "budget_clipped_rounds": 0,
            "budget_clipped_drafts": 0,
            "truncated_rounds": 0,
            "truncated_tokens": 0,
            "truncated_accepted_tokens": 0,
        }
    accepted = num_emitted - 1
    useful = min(accepted, kept)
    stats["rounds"] += 1
    for key, limit in (
        ("proposed", verified_depth), ("accepted", accepted), ("useful_accepted", useful)
    ):
        counts = stats[key]
        counts.extend([0] * (staged_depth - len(counts)))
        for i in range(limit):
            counts[i] += 1
    for key, depth in (
        ("verified_depth_histogram", verified_depth), ("staged_depth_histogram", staged_depth)
    ):
        histogram = stats[key]
        depth_key = str(depth)  # stable keys through JSON/msgpack round trips
        histogram[depth_key] = histogram.get(depth_key, 0) + 1
    if verified_depth < staged_depth:
        stats["budget_clipped_rounds"] += 1
        stats["budget_clipped_drafts"] += staged_depth - verified_depth
    if kept < num_emitted:
        stats["truncated_rounds"] += 1
        stats["truncated_tokens"] += num_emitted - kept
        stats["truncated_accepted_tokens"] += accepted - useful


def _engine_stats(req: "Req", now: float) -> dict:
    """Per-request engine telemetry for the finishing reply (usage["freetoken"]).

    ``prefill_s`` is the in-engine prefill wall time (first chunk scheduled -> first token
    drained), excluding HTTP/tokenizer time; ``decode_s`` spans first -> last emitted token;
    a verify step counts as one decode step. getattr defaults keep stub requests in unit
    tests working. Costs a few ints and two clock stamps per request lifetime.
    """
    first = getattr(req, "stat_first_token_t", None)
    prefill_s = getattr(req, "stat_prefill_s", None)
    stats = {
        "prefill_s": None if prefill_s is None else round(prefill_s, 4),
        "decode_s": None if first is None else round(now - first, 4),
        "decode_steps": getattr(req, "stat_decode_steps", 0),
        "spec_steps": getattr(req, "stat_spec_steps", 0),
        "draft_tokens": getattr(req, "stat_draft_tokens", 0),
        "accepted_tokens": getattr(req, "stat_accepted_tokens", 0),
    }
    positions = getattr(req, "stat_spec_positions", None)
    if positions is not None:
        stats["spec_positions"] = positions
    return stats


class Scheduler(SchedulerIOMixin):
    def __init__(self, config: SchedulerConfig):
        from freetoken.engine import Engine

        self.engine = Engine(config)

        # use another stream to overlap metadata processing with computation
        self.device = self.engine.device
        self.stream = torch.cuda.Stream(device=self.device)
        self.engine_stream_ctx = torch.cuda.stream(self.engine.stream)
        torch.cuda.set_stream(self.stream)
        # sent on the readiness ack for /v1/stats gpus; a list so TP can add one entry per rank
        self.gpus = [gpu_identity(self.device.index)] if self.device.type == "cuda" else []

        # initialize other managers
        self.table_manager = TableManager(config.max_running_req, self.engine.page_table)
        # ONE cache manager for every model (ShadowRadix layering): the shared page table is the
        # virtual full-token coordinate; model-specific tiers ride the plug-ins -- DSV4's
        # window/cmp/idx shadows via swa_pool, Gemma's swa via swa_pool, GDN state via
        # linear_state_pool. No model supplies its own manager.
        self.cache_manager = CacheManager(
            self.engine.num_pages, config.page_size, self.engine.page_table, config.cache_type,
            linear_state_pool=self.engine.linear_state_pool,
            swa_pool=self.engine.kv_cache,
            sliding_window_size=next(
                (g.sliding_window for g in config.model_config.kv_cache_group_specs() if g.is_swa),
                None,
            ) or getattr(self.engine.kv_cache, "sliding_window_size", None),
        )
        self.decode_manager = DecodeManager(config.page_size)
        self.prefill_manager = PrefillManager(
            self.cache_manager, self.table_manager, self.decode_manager
        )

        # some alias for easy access
        self.finished_reqs: Set[Req] = set()
        # Abort acknowledgements are a terminal accounting barrier. Queue them while processing
        # inbound control messages, then flush only AFTER _process_last_data publishes any
        # sampled replies from the prior overlapped forward.
        self._pending_abort_acks: Set[int] = set()
        # With multiple tokenizer workers, an AbortBackendMsg and its earlier UserMsg can arrive
        # through different PUSH producers and be observed out of order. Preserve a bounded
        # tombstone so an abort-before-admission request can never be resurrected after its
        # terminal accounting acknowledgement has already been published.
        self._abort_tombstones: dict[int, None] = {}
        self._forward_iter = 0  # global forward counter; drives the SWA proactive-eviction cadence
        # The launched-but-not-yet-drained batch (overlap): set at the top of each overlap_loop
        # iteration so the abort handler can tell whether a request's forward is still in flight
        # (mark it, defer the free to _process_last_data) or not (free immediately). Stays None
        # in normal_loop, where a batch launches and drains within one iteration.
        self._last_data: ForwardData | None = None
        # A received-but-not-yet-executed runtime cache rebuild (CacheRebuildBackendMsg),
        # run at the next idle safe point in overlap_loop. None when no rebuild is pending.
        self._pending_rebuild: CacheRebuildBackendMsg | None = None
        # Elastic KV pool (see scheduler/elastic_kv.py): the startup geometry is the base;
        # requests that triggered a grow/shrink wait here until the rebuild has landed.
        self._elastic_base_pages: int = self.engine.num_pages
        self._elastic_last_activity: float = time.monotonic()
        self._elastic_seq: int = 0
        self._elastic_attempted: set[int] = set()
        self._elastic_backoff_until: float = 0.0
        self._parked_msgs: list = []
        self.tokenizer = load_tokenizer(config.model_path)
        self.eos_token_ids = load_eos_token_ids(config.model_path, self.tokenizer)
        self.toolcall_anchor_id = None
        if config.special_token_ckpt and (
            self.cache_manager.is_hybrid or self.cache_manager.is_swa
        ):
            from freetoken.server.function_call_parser import toolcall_opener_for

            self.toolcall_anchor_id = load_toolcall_anchor_id(
                self.tokenizer,
                toolcall_opener_for(getattr(config, "tool_call_parser", "")),
            )
        self.token_pool = self.table_manager.token_pool
        # Speculative decoding (MTP): the engine owns the executor; the scheduler owns the
        # session state machine (Req.spec_state), engages verify batches only while exactly
        # one greedy request is decoding, and wires the token pool the executor scatters into.
        self.spec_k = config.num_spec_tokens if self.engine.spec_executor is not None else 0
        # First-prefill-chunk schedule time per uid; consumed at first-token drain to report
        # the request's in-engine prefill wall time (usage["freetoken"].prefill_s).
        self._prefill_t0: dict[int, float] = {}
        if self.engine.spec_executor is not None:
            self.engine.spec_executor.token_pool = self.token_pool
        # Floor the prefill chunk by the cache manager's cap (DSV4: ~half the window pool) so a
        # sliding-window cache chunks long prompts and frees out-of-window pages between chunks
        # instead of OOMing _alloc_window on a prompt longer than the window pool.
        _chunk_cap = self.cache_manager.prefill_chunk_budget
        self.prefill_budget = (
            min(config.max_extend_tokens, _chunk_cap) if _chunk_cap else config.max_extend_tokens
        )
        self.config = config
        self.status_reporter = SchedulerStatusReporter(
            log=logger.info_rank0,
            decode_log_interval=config.decode_log_interval,
        )

        # Initialize the I/O mixin
        super().__init__(config, self.engine.tp_cpu_group)

    def run_when_idle(self) -> None:
        """Called when the scheduler is idle to perform background tasks."""
        logger.info_rank0("Scheduler is idle, waiting for new reqs...")
        self.cache_manager.check_integrity()

    @torch.inference_mode()
    def rebuild_cache(
        self,
        *,
        moe_cache_size: int | None = None,
        num_pages: int | None = None,
        num_mamba_slots: int | None = None,
        num_swa_pages: int | None = None,
    ) -> None:
        """Idle-only runtime cache rebuild: resize the MoE slot cache, KV pages, GDN (mamba) state
        pool, and/or the window pool (num_swa_pages), re-capture CUDA graphs, and re-thread the
        page managers (clearing the prefix cache on a KV/mamba/window resize). The caller MUST
        guarantee the scheduler is idle — no pending prefill, no running decode, no in-flight
        finished requests. All TP ranks must call this with identical arguments.
        """
        assert not self.prefill_manager.runnable, "rebuild requires no pending prefill"
        assert not self.decode_manager.runnable, "rebuild requires no running decode"
        torch.cuda.synchronize(self.device)
        if self.config.tp_info.size > 1:
            self.sync_all_ranks()
        self.engine.rebuild_runtime_cache(
            moe_cache_size=moe_cache_size, num_pages=num_pages, num_mamba_slots=num_mamba_slots,
            num_swa_pages=num_swa_pages,
        )
        if num_pages is not None or num_mamba_slots is not None or num_swa_pages is not None:
            # Any of these resizes invalidates the prefix cache: a KV resize leaves stale page
            # indices, a mamba resize leaves stale GDN-snapshot slot ids, and a window-pool resize
            # (num_swa_pages) reallocates the SWA/window token pool, leaving stale slot ids in the
            # radix tree. Rebuild the prefix cache + reclaim the resized free-lists.
            self.cache_manager.rebuild(self.engine.num_pages, self.engine.page_table)
            if num_pages is not None:
                # token_pool is sized to the page table; only a KV-page resize reallocates it.
                # A mamba-only rebuild leaves the page table untouched, so skip this (else it
                # needlessly reallocates + zeros the whole GPU token_pool every mamba resize).
                self.table_manager.rebuild(self.engine.page_table)
                self.token_pool = self.table_manager.token_pool
                if getattr(self.engine, "spec_executor", None) is not None:
                    self.engine.spec_executor.token_pool = self.token_pool
            self.cache_manager.check_integrity()
        # The prefill chunk cap tracks the CURRENT window-pool size (DSV4); a rebuild that
        # shrank the pool must shrink the cap too, or the next long prompt is chunked against
        # the stale budget and crashes _alloc_window.
        _chunk_cap = self.cache_manager.prefill_chunk_budget
        self.prefill_budget = (
            min(self.config.max_extend_tokens, _chunk_cap)
            if _chunk_cap else self.config.max_extend_tokens
        )
        if self.config.tp_info.size > 1:
            self.sync_all_ranks()

    def overlap_loop(self, last_data: ForwardData | None) -> ForwardData | None:
        """
        The main loop of overlapping scheduling and execution.

        It will overlap the execution of current batch and processing of last batch's results,
        which can effectively hide CPU latency and improve GPU utilization.
        """
        # Expose the un-drained batch to _process_one_msg (abort in-flight check). Assigning
        # before the message loop is what makes the check airtight: the batch launched later
        # this iteration can only be probed by messages of the NEXT iteration, which sees it here.
        self._last_data = last_data
        blocking = not (
            last_data is not None  # don't block if we have a batch to be processed
            or self.prefill_manager.runnable
            or self.decode_manager.runnable
            or self._pending_rebuild is not None  # a queued rebuild to drain toward + execute
        )
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        # Execute a queued cache rebuild once the scheduler is fully idle (the safe point):
        # no last batch to process, no pending prefill, no running decode. finished_reqs is
        # NOT a gate — those requests are already freed (no live GPU/page resources).
        if self._pending_rebuild is not None and last_data is None and not (
            self.prefill_manager.runnable or self.decode_manager.runnable
        ):
            self._execute_pending_rebuild()

        # Order this iteration's host->device token_pool copies (issued on ``self.stream``
        # during scheduling) after the previous batch's sampled-token writes (issued on the
        # engine stream in ``_forward``). Without this, a request that reuses a just-freed
        # table_idx can have its freshly copied prompt clobbered by the prior occupant's
        # still-pending output write -- corrupting tokens (e.g. dropping an image
        # placeholder, which the multimodal merge then rejects).
        self.stream.wait_stream(self.engine.stream)
        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            with self.engine_stream_ctx:  # run the batch in the engine's stream
                self.engine.stream.wait_stream(self.stream)
                # COW-restore GDN snapshots for prefix hits ON THE ENGINE STREAM, after the
                # cross-stream wait and before the forward reads the live slot (program order
                # vs the prior batch's snapshot writes). Doing this on self.stream would race.
                self._restore_linear_states(forward_input.batch)
                fwd_out = self._forward(forward_input)
                # None: a prefill hit CUDA OOM; its requests were aborted (client told), nothing to drain
                ongoing_data = None if fwd_out is None else (forward_input, fwd_out)

        # The drain issues GPU-visible writes to state the batch just launched still reads: the
        # page-table re-point and, for the paged-SWA pools, the full->swa (DSV4: full->window)
        # sentinel scatter. DSV4 stages the page table at replay time and translates
        # full_to_window INSIDE the captured graph, so an unordered drain can redirect an
        # in-flight forward. copy_done only covers batch N; order against N+1 explicitly.
        self.stream.wait_stream(self.engine.stream)
        self._process_last_data(last_data)
        if ongoing_data is not None and (
            ongoing_data[0].batch.spec is not None
            or getattr(ongoing_data[0].batch, "spec_warmup", None) is not None
        ):
            # Speculative steps drain immediately (non-overlap): a verify's next scheduling
            # needs the host-side accepted count, and a warmup's session-state flip must land
            # before the next decode step is staged (else it would warm up twice).
            self._process_last_data(ongoing_data)
            ongoing_data = None
        self._flush_abort_acks()
        return ongoing_data

    def normal_loop(self) -> None:
        blocking = not (
            self.prefill_manager.runnable
            or self.decode_manager.runnable
            or self._pending_rebuild is not None  # a queued rebuild to execute at idle
        )
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        # Non-overlap mode has no last_data to drain; execute a queued rebuild as soon as
        # the scheduler is idle (no pending prefill / running decode). Without this, a
        # rebuild in DISABLE_OVERLAP_SCHEDULING mode stays pending until the HTTP timeout.
        if self._pending_rebuild is not None and not (
            self.prefill_manager.runnable or self.decode_manager.runnable
        ):
            self._execute_pending_rebuild()

        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            # already inside engine_stream_ctx (run_forever); restore on the engine stream
            self._restore_linear_states(forward_input.batch)
            fwd_out = self._forward(forward_input)
            # None: a prefill hit CUDA OOM; its requests were aborted (client told), nothing to drain
            ongoing_data = None if fwd_out is None else (forward_input, fwd_out)

        self._process_last_data(ongoing_data)
        self._flush_abort_acks()

    @torch.inference_mode()
    def run_forever(self) -> NoReturn:
        # DSV4 (owned-KV) decode reads its per-token window/cmp/idx slot maps off the attention
        # backend's per-batch SNAPSHOT (staged in prepare_for_replay right before the replay, on
        # the same stream, like the generic out_loc copy_from), not the live slot maps -- so the
        # next batch's allocate_paged cannot corrupt the in-flight graph replay. DSV4 overlaps.
        if ENV.DISABLE_OVERLAP_SCHEDULING:
            with self.engine_stream_ctx:
                self.engine.stream.wait_stream(self.stream)
                while True:
                    self.normal_loop()
        else:
            assert torch.cuda.current_stream() == self.stream
            data = None
            while True:
                data = self.overlap_loop(data)

    def shutdown(self) -> None:
        torch.cuda.synchronize(self.device)
        self.sync_all_ranks()
        self.engine.shutdown()

    def _process_last_data(self, last_data: ForwardData | None) -> None:
        if last_data is None:
            return
        self._elastic_last_activity = time.monotonic()

        if _MOE_STATS_LOG and last_data[0].batch.is_decode:
            self._moe_stats_steps = getattr(self, "_moe_stats_steps", 0) + 1
            if self._moe_stats_steps >= _MOE_STATS_LOG:
                self._moe_stats_steps = 0
                self._log_moe_decode_stats()

        if getattr(last_data[0].batch, "spec", None) is not None:
            self._process_spec_last_data(last_data[0].batch, last_data[1])
            return

        batch, (_, next_tokens_cpu, copy_done) = last_data[0].batch, last_data[1]
        copy_done.synchronize()
        now = time.monotonic()
        reply: List[DetokenizeMsg] = []
        new_finished_reqs: Set[Req] = set()
        with self.cache_manager.lazy_free_region():
            for i, req in enumerate(batch.reqs):
                if isinstance(req, ChunkedReq):
                    # Don't cache intermediate chunks; the full prompt is cached once when the
                    # final chunk is processed. Caching here snapshots a handle the next chunk
                    # already copied (overlap), so cache_req double-frees the prior chunk.
                    if req.aborted:
                        # Aborted mid-chunked-prefill while this chunk was in flight: the abort
                        # popped the pending continuation (no next chunk launches), and this
                        # drain point frees the chunk's pages/slots exactly once.
                        self._free_req_resources(req)
                    continue
                if req.aborted:
                    # Aborted while this final-chunk prefill / decode step was in flight: free
                    # here (the forward is drained) and finish the request. No DetokenizeMsg --
                    # the abort ack flushed after this method stays the uid's terminal reply.
                    self.decode_manager.remove_req(req)
                    self._free_req_resources(req)
                    new_finished_reqs.add(req)
                    continue
                if req in self.finished_reqs:
                    # Overlap scheduling launched one more decode step for a request that
                    # already terminated (filter_reqs keeps it while output budget remains,
                    # and the next batch is scheduled before this drain runs). Its resources
                    # are freed below/already; shipping this token would append past the
                    # client's terminal reply.
                    continue
                next_token = next_tokens_cpu[i]
                req.append_host(next_token.unsqueeze(0))
                next_token = int(next_token.item())
                if batch.is_prefill:
                    _t0 = self.__dict__.setdefault("_prefill_t0", {}).pop(req.uid, None)
                    if _t0 is not None:
                        req.stat_prefill_s = now - _t0
                    req.stat_first_token_t = now
                else:
                    req.stat_decode_steps = getattr(req, "stat_decode_steps", 0) + 1
                # EOS / stop-string -> "stop", output budget exhausted -> "length";
                # EOS and stop strings win over length.
                # In overlap mode the NEXT forward may already have advanced device_len
                # to the limit. Only the drained host prefix determines whether this
                # reply is final, otherwise the penultimate token ends the request and
                # the already-computed last token is discarded as a post-finish reply.
                hit_length = req.input_ids.numel() >= req.max_device_len
                hit_eos = (
                    not req.sampling_params.ignore_eos and next_token in self.eos_token_ids
                )
                matched_stop = (
                    self._match_stop_str(req)
                    if not hit_eos and req.sampling_params.stop_strs
                    else None
                )
                finished = hit_length or hit_eos or matched_stop is not None
                finish_reason = (
                    ("stop" if (hit_eos or matched_stop is not None) else "length")
                    if finished
                    else None
                )
                if (
                    next_token == self.toolcall_anchor_id
                    and req.toolcall_anchor_len is None
                    and not finished
                ):
                    req.toolcall_anchor_len = req.input_ids.numel()
                if getattr(batch, "spec_warmup", None) is req and not finished:
                    # The warmup step ran and its draft is staged; decode continues as
                    # verify steps.
                    req.spec_state = "ready"
                reply.append(
                    DetokenizeMsg(
                        uid=req.uid,
                        next_token=next_token,
                        finished=finished,
                        finish_reason=finish_reason,
                        matched_stop=matched_stop,
                        stop_strs=req.sampling_params.stop_strs or None,
                        engine_stats=_engine_stats(req, now) if finished else None,
                    )
                )

                # NOTE: overlap scheduling may make the request freed twice, skip second free
                if finished and req not in self.finished_reqs:
                    self.decode_manager.remove_req(req)
                    self._free_req_resources(req)
                    new_finished_reqs.add(req)
                elif batch.is_prefill and req.table_idx != -1:
                    # for prefill, non-chunk req, cache the prefix.
                    # Polymorphic: the DSV4 naive manager keeps the request's slots (no-op);
                    # the generic manager inserts the prefix into its radix/naive cache.
                    # table_idx == -1 is defense-in-depth: aborts mark in-flight requests
                    # instead of freeing them (handled above), so a freed request should
                    # never reach this commit -- but if a future path frees one early, skip
                    # rather than re-read the freed page-table row (and on hybrid, deref the
                    # None'd GDN ping-pong slots).
                    self.cache_manager.cache_req(req, finished=False)

        self.finished_reqs = new_finished_reqs
        # Stamp each reply with the post-batch KV page occupancy so the frontend (shell
        # status bar) can show live KV usage without a separate query.
        used, total = self._kv_usage_pages()
        mamba_slots = self._mamba_slot_usage()
        swa_tokens = self._swa_token_usage()
        if reply:
            mem = self._gpu_mem_bytes()
            mamba_used, mamba_total = mamba_slots or (0, 0)
            swa_used, swa_total = swa_tokens or (0, 0)
            for m in reply:
                m.kv_used_pages = used
                m.kv_total_pages = total
                m.mamba_used_slots = mamba_used
                m.mamba_total_slots = mamba_total
                m.swa_used_tokens = swa_used
                m.swa_total_tokens = swa_total
                m.gpu_mem_bytes = mem
        self.status_reporter.report_batch(
            batch,
            running_reqs=len(self.decode_manager.running_reqs),
            queue_reqs=len(self.prefill_manager.pending_list),
            kv_used_pages=used,
            kv_total_pages=total,
            page_size=self.config.page_size,
            mamba_slots=mamba_slots,
            swa_tokens=swa_tokens,
        )
        self.send_result(reply)

    def _process_spec_last_data(self, batch: Batch, output) -> None:
        """Drain one speculative verify step: settle Req bookkeeping to the emitted run,
        truncating at the first terminal condition (EOS / stop string / length). The device
        side already committed states and scattered tokens for the full accepted count; a
        truncation re-commits at the kept count (the scratch still holds the step's
        intermediates), so the state, ``cached_len`` and ``input_ids`` stay aligned and the
        finish-time prefix donation remains valid."""
        executor = self.engine.spec_executor
        req = batch.reqs[0]
        from freetoken.engine.spec import _SPEC_HOST_TIME

        if _SPEC_HOST_TIME:
            import time as _time

            _t0 = _time.perf_counter()
            output.copy_done.synchronize()
            _t1 = _time.perf_counter()
            executor._ht_sync.append(_t1 - _t0)
        else:
            output.copy_done.synchronize()
        n = int(output.num_emitted_cpu.item())
        cands = output.candidates_cpu
        if batch.spec is not None:
            from freetoken.engine.spec import update_accept_ema

            update_accept_ema(req, batch.spec.num_input_tokens - 1, n - 1)
            req.stat_decode_steps = getattr(req, "stat_decode_steps", 0) + 1
            req.stat_spec_steps = getattr(req, "stat_spec_steps", 0) + 1
            req.stat_draft_tokens = getattr(req, "stat_draft_tokens", 0) + (
                batch.spec.num_input_tokens - 1
            )
            req.stat_accepted_tokens = getattr(req, "stat_accepted_tokens", 0) + max(n - 1, 0)

        if req in self.finished_reqs:
            # Overlap edge: the previous drain finished this request after this verify was
            # already launched. Everything the step wrote lands in freed slots that any
            # later owner re-initializes; nothing to settle.
            return
        if req.aborted:
            # Keep the invariant state == cached_len == len(input_ids) with the minimum
            # emitted token, then free through the normal abort path (no reply).
            executor.recommit(req, 1)
            req.append_host(cands[:1])
            old_device_len = req.device_len
            req.commit_verify(1)
            self.cache_manager.truncate_paged(req, old_device_len)
            req.spec_state = "off"
            executor.drop(req.uid)
            self.decode_manager.remove_req(req)
            with self.cache_manager.lazy_free_region():
                self._free_req_resources(req)
            self.finished_reqs = self.finished_reqs | {req}
            return

        kept = 0
        finished = False
        finish_reason = None
        matched_stop = None
        tokens: List[int] = []
        for j in range(n):
            token_t = cands[j : j + 1]
            req.append_host(token_t)
            kept += 1
            token = int(token_t.item())
            tokens.append(token)
            hit_length = req.input_ids.numel() >= req.max_device_len
            hit_eos = not req.sampling_params.ignore_eos and token in self.eos_token_ids
            matched_stop = (
                self._match_stop_str(req)
                if not hit_eos and req.sampling_params.stop_strs
                else None
            )
            finished = hit_length or hit_eos or matched_stop is not None
            finish_reason = (
                ("stop" if (hit_eos or matched_stop is not None) else "length")
                if finished
                else None
            )
            if finished:
                break
        if _SPEC_POSITION_STATS and batch.spec is not None:
            verified_depth = batch.spec.num_input_tokens - 1
            _record_spec_position_stats(
                req,
                verified_depth=verified_depth,
                staged_depth=batch.spec.num_staged_drafts or verified_depth,
                num_emitted=n,
                kept=kept,
            )
        # One message for the whole run: the detokenizer's batch decode requires at most one
        # DetokenizeMsg per uid per batch, and next_token must stay the run's LAST token so
        # the finished/eos handling keyed on it applies to the token that triggered it.
        reply: List[DetokenizeMsg] = [
            DetokenizeMsg(
                uid=req.uid,
                next_token=tokens[-1],
                extra_tokens=tokens[:-1] or None,
                finished=finished,
                finish_reason=finish_reason,
                matched_stop=matched_stop,
                stop_strs=req.sampling_params.stop_strs or None,
                engine_stats=_engine_stats(req, time.monotonic()) if finished else None,
            )
        ]

        if kept < n:
            executor.recommit(req, kept)  # before commit_verify: reads the pre-commit cached_len
        old_device_len = req.device_len
        req.commit_verify(kept)
        # Give back the whole pages the optimistic bump allocated past the settled length.
        with self.cache_manager.lazy_free_region():
            self.cache_manager.truncate_paged(req, old_device_len)

        if finished:
            req.spec_state = "off"
            executor.drop(req.uid)
            self.decode_manager.remove_req(req)
            with self.cache_manager.lazy_free_region():
                self._free_req_resources(req)
            self.finished_reqs = self.finished_reqs | {req}
        elif req not in self.decode_manager.running_reqs:
            # The launch-time filter saw the optimistic (over-bumped) device_len; the
            # settled books may still have budget.
            self.decode_manager.filter_reqs([req])

        used, total = self._kv_usage_pages()
        mamba_slots = self._mamba_slot_usage()
        swa_tokens = self._swa_token_usage()
        if reply:
            mem = self._gpu_mem_bytes()
            mamba_used, mamba_total = mamba_slots or (0, 0)
            swa_used, swa_total = swa_tokens or (0, 0)
            for m in reply:
                m.kv_used_pages = used
                m.kv_total_pages = total
                m.mamba_used_slots = mamba_used
                m.mamba_total_slots = mamba_total
                m.swa_used_tokens = swa_used
                m.swa_total_tokens = swa_total
                m.gpu_mem_bytes = mem
        self.status_reporter.report_batch(
            batch,
            running_reqs=len(self.decode_manager.running_reqs),
            queue_reqs=len(self.prefill_manager.pending_list),
            kv_used_pages=used,
            kv_total_pages=total,
            page_size=self.config.page_size,
            mamba_slots=mamba_slots,
            swa_tokens=swa_tokens,
            generated_tokens=kept,
        )
        self.send_result(reply)
        if _SPEC_HOST_TIME:
            executor._ht_drain.append(_time.perf_counter() - _t1)

    def _match_stop_str(self, req: Req) -> str | None:
        """First stop string present in this request's generated tail, else None. Decodes
        only a short suffix (bounded by the longest stop string's char length, so a stop of
        N chars spans at most N tokens) to keep the per-step cost small."""
        stop_strs = req.sampling_params.stop_strs
        prompt_len = req.max_device_len - req.output_len
        if len(req.input_ids) <= prompt_len:
            return None
        max_chars = max(len(s) for s in stop_strs)
        tail_start = max(prompt_len, len(req.input_ids) - (max_chars + 1))
        tail = self.tokenizer.decode(req.input_ids[tail_start:].tolist())
        for s in stop_strs:
            if s in tail:
                return s
        return None

    def _kv_usage_pages(self) -> Tuple[int, int]:
        """(used_pages, total_pages) of the KV page pool.

        ``used`` follows SGLang's logging semantics: allocated pages that are not
        evictable (active requests + protected prefix cache). Evictable prefix-cache
        pages are available to future requests, so they are excluded from usage.
        Always the manager's own primary pool (for DSV4 the FULL cmp/idx tier); the
        window (swa) tier is reported separately by ``_swa_token_usage``.
        """
        return self.cache_manager.page_usage()

    def _mamba_slot_usage(self) -> Tuple[int, int] | None:
        """(used_slots, total_slots) of the GDN-state (mamba) pool for hybrid models, else None.

        Mirrors SGLang's mamba-pool semantics: ``total`` excludes the reserved padding
        sink (slot 0); ``used`` excludes free slots and evictable tree snapshots.
        """
        if not self.cache_manager.is_hybrid:
            return None
        total = self.cache_manager.linear_state_pool.num_slots - 1
        return total - self.cache_manager.mamba_available_size, total

    def _swa_token_usage(self) -> Tuple[int, int] | None:
        """(used_tokens, total_tokens) of the window (swa) pool for SWA models, else None.

        Mirrors the mamba accounting: ``total`` excludes the pool's reserved sentinel
        unit; ``used`` excludes free slots and evictable (unlocked) tree tokens.
        """
        cm = self.cache_manager
        if not cm.swa_paged:
            return None
        total = cm.swa_pool.swa_num_tokens - 1
        return total - cm.swa_available_size, total

    def _gpu_mem_bytes(self) -> int:
        """Bytes this engine process holds on the GPU (torch's reserved caching-allocator
        pool: weights + KV + MoE cache + graphs). 0 on CPU. Cheap, no device sync."""
        if self.device.type != "cuda":
            return 0
        return torch.cuda.memory_reserved(self.device)

    def _process_one_msg(self, msg: BaseBackendMsg) -> None:
        if isinstance(msg, BatchBackendMsg):
            for msg in msg.data:
                self._process_one_msg(msg)
        elif isinstance(msg, ExitMsg):
            raise KeyboardInterrupt
        elif isinstance(msg, UserMsg):
            logger.debug_rank0("Received user msg: %s", msg)
            tombstones = getattr(self, "_abort_tombstones", None)
            if tombstones is not None and msg.uid in tombstones:
                tombstones.pop(msg.uid, None)
                logger.debug_rank0(
                    "Dropping request %d because its abort arrived before admission", msg.uid
                )
                return
            input_len, max_seq_len = len(msg.input_ids), self.engine.max_seq_len
            max_output_len = max_seq_len - input_len
            if self._elastic_kv_maybe_resize(msg, input_len):
                return  # parked: re-processed once the queued rebuild has landed
            self.__dict__.setdefault("_elastic_attempted", set()).discard(msg.uid)
            self._elastic_last_activity = time.monotonic()
            if max_output_len <= 0:
                logger.warning_rank0(
                    f"Input sequence length {input_len} exceeds {max_seq_len}, "
                    f"request {msg.uid} is dropped."
                )
                # Tell the client instead of dropping silently — otherwise its wait_for_ack
                # never sees a `finished` reply and hangs until the request times out.
                self.send_result(
                    [
                        ErrorReplyMsg(
                            uid=msg.uid,
                            # "prompt is too long: N tokens > M" is the phrasing Claude Code and
                            # OpenClaw match on; the Anthropic wire has no error code to read.
                            error=(
                                f"prompt is too long: {input_len} tokens > {max_seq_len} maximum "
                                f"(prompt + generation); shorten the prompt or increase the KV "
                                f"cache budget"
                            ),
                            # OpenAI's standard class for this, for clients that read a code.
                            code="context_length_exceeded",
                        )
                    ]
                )
                return
            if msg.sampling_params.max_tokens > max_output_len:
                msg.sampling_params.max_tokens = max_output_len
                logger.warning_rank0(
                    f"Adjust max_tokens to {max_output_len} for request {msg.uid}."
                )
            self.prefill_manager.add_one_req(msg)
        elif isinstance(msg, AbortBackendMsg):
            logger.debug_rank0("Aborting request %d", msg.uid)
            tombstones = getattr(self, "_abort_tombstones", None)
            if tombstones is None:
                tombstones = self._abort_tombstones = {}
            tombstones[msg.uid] = None
            # Unknown aborts normally consume their tombstone when the cross-worker UserMsg
            # catches up. Bound hostile/no-followup abort traffic without affecting realistic
            # in-flight concurrency.
            while len(tombstones) > 65_536:
                tombstones.pop(next(iter(tombstones)))
            req_to_free = self.prefill_manager.abort_req(msg.uid)
            req_to_free = req_to_free or self.decode_manager.abort_req(msg.uid)
            if req_to_free is not None:
                # SGLang-style abort: never free resources under an in-flight forward. If the
                # request is in the launched-but-not-drained batch (overlap), only mark it;
                # _process_last_data frees it this same iteration, after copy_done.synchronize()
                # -- so its KV pages / GDN slots are never recycled mid-write, and the
                # finished=False prefix-commit can't run on a freed request. A request with no
                # forward in flight (e.g. a decode req starved behind a long chunked prefill)
                # is freed immediately -- deferring would leak until its next batch, which
                # strict prefill-priority puts arbitrarily far away.
                inflight = (
                    self._last_data is not None
                    and req_to_free in self._last_data[0].batch.reqs
                )
                if inflight:
                    req_to_free.aborted = True
                else:
                    self._free_req_resources(req_to_free)
            # Always acknowledge the abort, even when the request already left the manager,
            # but NOT yet: overlap_loop still has to publish the prior forward's sampled reply.
            # _flush_abort_acks runs after _process_last_data, making this a true terminal
            # accounting barrier for FrontendManager/prepare-stop.
            self._pending_abort_acks.add(msg.uid)
        elif isinstance(msg, CacheRebuildBackendMsg):
            # v1 scope: only if_idle, single-rank, non-owned-KV. drain mode and TP rebuild
            # need the drain-gate / all-rank failure-agreement machinery (deferred), so we
            # reject them cleanly rather than ship hang-prone half-wired paths.
            if not self.cache_manager.supports_runtime_rebuild:
                self._reply_rebuild(
                    msg.request_id, "unsupported", "this model's cache does not support runtime rebuild"
                )
            elif msg.mode != "if_idle":
                self._reply_rebuild(
                    msg.request_id, "unsupported", f"mode {msg.mode!r} unsupported (use if_idle)"
                )
            elif self.config.tp_info.size > 1:
                self._reply_rebuild(
                    msg.request_id, "unsupported", "runtime rebuild unsupported under TP > 1"
                )
            elif self.prefill_manager.runnable or self.decode_manager.runnable:
                # if_idle: refuse rather than wait. (finished_reqs hold no resources — they
                # are already freed — so they do not block a rebuild.)
                self._reply_rebuild(msg.request_id, "busy")
            else:
                self._pending_rebuild = msg
        else:
            logger.error(f"Unknown message type: {type(msg)}")
            raise NotImplementedError

    def _restore_linear_states(self, batch) -> None:
        """COW-restore a hybrid prefix hit's GDN snapshot into its freshly-allocated live slot
        (first chunk only). MUST run on the ENGINE stream so it is program-ordered after the
        prior batch's snapshot writes and before this forward reads the live slot."""
        pool = self.engine.linear_state_pool
        if pool is None or not batch.is_prefill:
            return
        for req in batch.reqs:
            if req.mamba_restore_src is not None:
                pool.copy_from(req.mamba_restore_src, req.linear_slot_idx)
                req.mamba_restore_src = None  # consumed: restore exactly once

    def _free_req_resources(self, req: Req) -> None:
        self.__dict__.setdefault("_prefill_t0", {}).pop(req.uid, None)
        # Idempotent: an EOS-finished request can stay in running_reqs (output budget left), so an
        # abort in the same overlap iteration races _process_last_data and would free it twice --
        # double-freeing its table_idx and (hybrid) GDN slots onto the free-list, handing the same
        # slots to two later requests. table_idx == -1 marks an already-freed request.
        if req.table_idx == -1:
            return
        if getattr(self, "spec_k", 0):
            self.engine.spec_executor.drop(req.uid)
        # Polymorphic free: the DSV4 manager returns the request's window pages + cmp/idx blocks
        # to their tier free-lists; the generic manager frees its KV pages (it reads
        # page_table[req.table_idx], so free the table entry after).
        self.cache_manager.cache_req(req, finished=True)
        self.table_manager.free(req.table_idx)
        req.table_idx = -1

    def _reply_rebuild(self, request_id: str, status: str, error: str | None = None) -> None:
        if request_id.startswith("auto:") and status != "ok":
            # elastic KV: do not hammer a resize that just failed (e.g. transient OOM)
            self._elastic_backoff_until = time.monotonic() + 600.0
            logger.warning_rank0(f"[elastic-kv] automatic resize {status}: {error}; backing off 10 min")
        # Single source of truth with the rollback snapshot (_current_cache_geometry): mamba is
        # usable slots (padding sink excluded, matching the status-bar gauge), and num_swa_pages
        # reports 0 unless the model actually has a window pool.
        geo = self._current_cache_geometry()
        self.send_result(
            [
                CacheRebuildResultMsg(
                    request_id=request_id,
                    status=status,
                    moe_cache_size=geo["moe_cache_size"] or 0,
                    num_pages=geo["num_pages"],
                    mamba_slots=geo["num_mamba_slots"] or 0,
                    num_swa_pages=geo["num_swa_pages"] or 0,
                    error=error,
                )
            ]
        )

    def _elastic_kv_enabled(self) -> bool:
        cfg = getattr(self, "config", None)
        if cfg is None:
            return False  # stub schedulers in unit tests carry no config
        return (
            cfg.kv_elastic_max_tokens > 0
            and cfg.tp_info.size == 1
            and self.engine.moe_offload_cache is not None
            and self.engine.num_pages > 0
        )

    def _elastic_kv_maybe_resize(self, msg, input_len: int) -> bool:
        """Elastic KV policy at admission. Returns True when ``msg`` was parked behind a queued
        grow (the pool is too small for it) or shrink (it fits the base and the pool has been
        idle long enough); the message is re-processed after the rebuild. A request that
        already triggered one attempt is never parked twice (a rejected grow falls through to
        the normal too-long error)."""
        if not self._elastic_kv_enabled() or msg.uid in self._elastic_attempted:
            return False
        if time.monotonic() < self._elastic_backoff_until:
            return False  # a recent automatic resize was rejected/failed: leave the pool alone
        if self._pending_rebuild is not None:
            # a resize is already queued (maybe by an earlier request): ride along
            self._parked_msgs.append(msg)
            self._elastic_attempted.add(msg.uid)
            return True
        from freetoken.kvcache.cache_status import compute_cache_unit_bytes
        from freetoken.scheduler.elastic_kv import (
            needed_tokens,
            plan_moe_for_pages,
            plan_shrink_tokens,
            plan_target_tokens,
        )

        cfg = self.config
        page = cfg.page_size
        cap = cfg.kv_elastic_max_tokens
        base_pages = self._elastic_base_pages
        cur_pages = self.engine.num_pages
        base_tokens, cur_tokens = base_pages * page, cur_pages * page
        needed = needed_tokens(input_len, int(getattr(msg.sampling_params, "max_tokens", 1024) or 1024), cap)
        target_pages = None
        if needed > cur_tokens and input_len < cap:
            target_pages = (plan_target_tokens(needed, base_tokens, cap) + page - 1) // page
            if target_pages <= cur_pages:
                target_pages = None
            reason = "grow"
        elif (
            cur_pages > base_pages
            and time.monotonic() - self._elastic_last_activity >= cfg.kv_elastic_idle_s
        ):
            # Stepped: give back everything above the smallest ladder step this request needs,
            # not only the whole way down for a request that fits the base.
            # FREETOKEN_KV_ELASTIC_STEPPED=0 restores the shrink-to-base-only behaviour.
            shrink_to = (
                plan_shrink_tokens(needed, cur_tokens, base_tokens, cap)
                if _ELASTIC_STEPPED
                else (base_tokens if needed <= base_tokens else None)
            )
            if shrink_to is not None:
                target_pages = (shrink_to + page - 1) // page
                reason = "shrink"
        if target_pages is None:
            return False
        unit = compute_cache_unit_bytes(self.engine)
        kv_per_page = unit["kv_bytes_per_token"] * page
        slot_bytes = unit["moe_bytes_per_expert"]
        cur_moe = self.engine.moe_offload_cache.cache_size
        moe_target = plan_moe_for_pages(cur_pages, cur_moe, target_pages, kv_per_page, slot_bytes)
        if moe_target <= 0 or kv_per_page <= 0:
            return False
        self._elastic_seq += 1
        logger.info_rank0(
            f"[elastic-kv] {reason}: request {msg.uid} (prompt {input_len} tok, needs ~{needed}) "
            f"-> KV {cur_tokens} -> {target_pages * page} tokens, expert slots {cur_moe} -> {moe_target}"
        )
        self._pending_rebuild = CacheRebuildBackendMsg(
            request_id=f"auto:{self._elastic_seq}", moe_cache_size=moe_target, num_pages=target_pages,
        )
        self._parked_msgs.append(msg)
        self._elastic_attempted.add(msg.uid)
        return True

    def _execute_pending_rebuild(self) -> None:
        # Requests parked behind an elastic resize are re-processed against the new geometry
        # whatever the outcome (a rejected grow surfaces as the normal too-long error).
        parked, self._parked_msgs = self._parked_msgs, []
        try:
            self._execute_pending_rebuild_inner()
        finally:
            for m in parked:
                self._process_one_msg(m)

    def _execute_pending_rebuild_inner(self) -> None:
        from freetoken.engine.engine import CacheRebuildRejected

        msg = self._pending_rebuild
        assert msg is not None
        self._pending_rebuild = None
        requested = {
            "moe_cache_size": msg.moe_cache_size,
            "num_pages": msg.num_pages,
            "num_mamba_slots": msg.num_mamba_slots,
            "num_swa_pages": msg.num_swa_pages,
        }
        # Rollback target: the CURRENT (serving) sizes of ONLY the pools this request touches.
        # Passing the untouched pools too would trip rebuild_cache's KV/mamba/SWA gate and wipe
        # the prefix cache that a successful resize of just the requested pool preserves.
        snapshot = self._current_cache_geometry()
        prior = {k: snapshot[k] for k, v in requested.items() if v is not None}
        # Cleared here, set by engine.rebuild_runtime_cache at its point of no return — lets the
        # except below tell a pre-teardown failure (engine untouched) from a mid-teardown one.
        self.engine.rebuild_teardown_started = False
        try:
            self.rebuild_cache(**requested)
        except CacheRebuildRejected as e:
            # Rejected before any destructive free — old cache intact, keep serving.
            logger.warning(f"cache rebuild rejected: {e}")
            self._reply_rebuild(msg.request_id, "rejected", error=str(e))
            return
        except Exception as e:  # noqa: BLE001
            if not getattr(self.engine, "rebuild_teardown_started", True):
                # Failed before the destructive phase began: graphs and pools are untouched and
                # the engine is still serving. A destructive rollback would only add risk.
                logger.error(f"cache rebuild failed before teardown: {e!r} — old cache intact")
                self._reply_rebuild(msg.request_id, "rejected", error=repr(e))
                return
            if self.config.tp_info.size > 1:
                # A lone-rank failure cannot be rolled back symmetrically: rebuild_cache runs TP
                # barriers, and ranks that succeeded will not re-enter them — a solo rollback
                # would desync the group. Keep the latch-failed behavior for tp>1.
                logger.error(f"cache rebuild failed: {e!r} — tp>1, latching failed")
                self._reply_rebuild(msg.request_id, "failed", error=repr(e))
                return
            # The destructive phase failed — typically a CUDA OOM while reallocating a pool or
            # recapturing graphs. The graphs/pools are already torn down, so the engine cannot
            # serve as-is. Rather than latch "failed" (which forces a full process restart),
            # rebuild the touched pools back to the sizes that were serving a moment ago: they
            # fit before, so shrinking back frees the just-attempted allocation and restores
            # service. Only if the rollback ALSO fails is the engine genuinely wedged. (Post-OOM
            # CUDA state is not guaranteed sane — a rollback that succeeds here may still surface
            # a deferred fault on a later request; that residual risk is accepted over always
            # forcing a restart.)
            logger.error(f"cache rebuild failed: {e!r} — rolling back to the previous geometry")
            try:
                self.rebuild_cache(**prior)
            except Exception as e2:  # noqa: BLE001 — rollback failed too; genuinely unrecoverable
                logger.error(f"cache rebuild rollback failed: {e2!r} — server latched failed")
                self._reply_rebuild(
                    msg.request_id,
                    "failed",
                    error=f"{e!r}; rollback to the prior geometry also failed: {e2!r}",
                )
                return
            logger.warning("cache rebuild rolled back to the previous geometry — still serving")
            self._log_cache_geometry("Cache rolled back")
            self._reply_rebuild(
                msg.request_id, "rejected", error=f"rebuild failed and was rolled back: {e!r}"
            )
            return
        # Outside the try: an ack/send failure after a fully-applied rebuild must not be
        # mistaken for a rebuild failure and roll back the geometry the engine now serves.
        self._log_cache_geometry("Cache rebuilt")
        self._reply_rebuild(msg.request_id, "ok")

    def _log_moe_decode_stats(self) -> None:
        """Periodic decode MoE cache readout: one line per FREETOKEN_MOE_STATS_LOG drained
        decode steps, each line covering exactly that window (stats reset after logging).
        The .item() reads sync on the engine stream's already-enqueued accumulations --
        negligible at the default cadence, and this only runs when the env asks for it."""
        cache = self.engine.moe_offload_cache
        if cache is None:
            return
        self._log_cpu_moe_timing(cache)
        self._log_tier3_stats(cache)
        if not cache.collect_stats:
            return
        try:
            s = cache.decode_miss_stats()
            per_step = s["missing_per_layer"] * cache.num_layers
            try:
                from freetoken.kvcache.cache_status import compute_cache_unit_bytes

                fetch_mb = (
                    s["fetched_per_layer"] * cache.num_layers
                    * compute_cache_unit_bytes(self.engine)["moe_bytes_per_expert"] / (1 << 20)
                )
            except Exception:  # noqa: BLE001 -- byte figure is best-effort decoration
                fetch_mb = 0.0
            extras = "".join(
                f" {k}={v:.3f}" if isinstance(v, float) else f" {k}={v}"
                for k, v in s.items()
                if k.startswith("prefetch_") or k.startswith("pf_")
            )
            logger.info_rank0(
                f"[moe-stats] miss_rate={s['miss_rate']:.3f}"
                f" active/layer={s['active_per_layer']:.1f}"
                f" missing/layer={s['missing_per_layer']:.1f}"
                f" fetched/layer={s['fetched_per_layer']:.1f}"
                f" cpu/layer={s['cpu_per_layer']:.1f}"
                f" missing/step={per_step:.0f}"
                f" fetch_MB/step={fetch_mb:.0f}"
                f" layer_calls={s['layer_calls']}" + extras
            )
            cache.reset_stats()
        except Exception as e:  # noqa: BLE001 -- diagnostics must never take down the loop
            logger.warning(f"could not log MoE decode stats: {e!r}")

    def _log_tier3_stats(self, cache) -> None:
        """NVMe tier readout at the moe-stats cadence, as PER-STEP deltas over the window.

        The C++ counters are cumulative and include the prefill scan, which dwarfs decode by
        orders of magnitude -- a cumulative figure would read as "always paging" no matter what
        decode does. Differencing against the previous window is the only form of this number
        that answers the question the tier exists to answer: does a warm decode step touch the
        disk at all, and if so how much?"""
        executor = getattr(cache, "cpu_executor", None)
        if executor is None or getattr(executor, "_tier3_handle", None) is None:
            return
        try:
            from freetoken.tier3.wiring import tier3_stats

            s = tier3_stats(executor)
        except Exception as e:  # noqa: BLE001 -- diagnostics must never take down the loop
            logger.warning(f"could not read tier-3 stats: {e!r}")
            return
        prev = getattr(self, "_tier3_prev", None)
        self._tier3_prev = s
        steps = max(1, _MOE_STATS_LOG)
        if prev is None:
            logger.info_rank0(
                f"[tier3] cumulative host_hit_rate={s['host_hit_rate']:.4f}"
                f" nvme_fetches={s['nvme_fetches']} (first window includes prefill)"
            )
            return
        d_hits = s["host_hits"] - prev["host_hits"]
        d_fetch = s["nvme_fetches"] - prev["nvme_fetches"]
        d_bytes = s["nvme_bytes"] - prev["nvme_bytes"]
        d_stall = s["stall_ms"] - prev["stall_ms"]
        rate = d_hits / max(1, d_hits + d_fetch)
        extra = ""
        rr = getattr(cache, "residency_route_stats", None)
        if rr is not None:
            # Without this the feature is unobservable: "no quality change" and "the kernel
            # never fired" look identical from the outside.
            vals = [int(x) for x in rr.tolist()]
            vals += [0] * (8 - len(vals))  # tolerate the older [2]- and [4]-wide tensors
            subs, seen, rows, capped, drops, spent_q20 = vals[:6]
            prev_rr = getattr(self, "_rr_prev", (0,) * 6)
            self._rr_prev = (subs, seen, rows, capped, drops, spent_q20)
            d_sub, d_seen = subs - prev_rr[0], seen - prev_rr[1]
            d_rows, d_capped = rows - prev_rr[2], capped - prev_rr[3]
            d_drop, d_spent = drops - prev_rr[4], (spent_q20 - prev_rr[5]) / 1048576.0
            d_act = d_sub + d_drop
            extra = (f" rroute_subs/step={d_sub / steps:.2f}"
                     f" rroute_drops/step={d_drop / steps:.2f}"
                     f" rroute_missing/step={d_seen / steps:.2f}"
                     f" rroute_cover={d_act / max(1, d_seen):.3f}"
                     # rows that still hold a miss after acting: what raising max_subs (or the
                     # budget) can still reach
                     f" rroute_rows/step={d_rows / steps:.2f}"
                     f" rroute_capped/step={d_capped / steps:.2f}")
            if d_spent > 0:
                # the perturbation the policy actually spent, as a fraction of the MoE block
                # output -- the quantity the budget is denominated in, so the quality/throughput
                # trade is legible without a separate measurement
                extra += (f" rroute_pert/step={d_spent / steps:.4f}"
                          f" rroute_pert/act={d_spent / max(1, d_act):.4f}")
        logger.info_rank0(
            f"[tier3] host_hit_rate={rate:.4f}"
            f" nvme_reads/step={d_fetch / steps:.2f}"
            f" nvme_MB/step={d_bytes / steps / (1 << 20):.3f}"
            f" stall_us/step={d_stall * 1000 / steps:.1f}"
            f" pool_exhausted={s['pool_exhausted']}" + extra
        )

    def _log_cpu_moe_timing(self, cache) -> None:
        """FREETOKEN_CPU_MOE_TIMING=1: the hybrid/cpu backend's host-side per-layer
        latency breakdown (microseconds, means over the window) at the moe-stats cadence."""
        executor = getattr(cache, "cpu_executor", None)
        if executor is None or not getattr(executor, "timing_enabled", False):
            return
        try:
            report = executor.timing_report(reset=True)
        except Exception as e:  # noqa: BLE001 -- diagnostics must never take down the loop
            logger.warning(f"could not read CPU MoE timing: {e!r}")
            return
        for bs, r in report.items():
            logger.info_rank0(
                f"[cpu-moe-timing] bs={bs} n={int(r['n'])} total={r['total']:.1f}us"
                f" prep={r['prep']:.1f} wake={r['wake']:.1f} p1={r['p1']:.1f}"
                f" mid={r['mid']:.1f} p2={r['p2']:.1f} max={r['max']:.1f}"
            )

    def _current_cache_geometry(self) -> dict:
        """The pools' current (serving) sizes as rebuild_cache kwargs — the rollback snapshot and
        the single source for _reply_rebuild's readout. None for a pool this model lacks
        (rebuild_cache skips those; the reply maps them to the wire format's 0). num_swa_pages is
        the CONCRETE current window (usable pages) so a rollback restores it byte-for-byte,
        whether it was pinned or ratio-derived."""
        eng = self.engine
        config = self.config
        mc = config.model_config
        num_swa_pages = None
        if getattr(mc, "dsv4_args", None) is not None:
            sizes = getattr(eng.kv_cache, "sizes", None)
            if sizes is not None:  # usable window pages = physical n_win_pages minus the dummy page
                num_swa_pages = max(0, sizes.n_win_pages - 1)
        elif getattr(mc, "has_swa_attention", False) and (
            getattr(config, "cache_type", None) == "swa_radix"
        ):  # usable window tokens = pool tokens minus the slot-0 sentinel
            num_swa_pages = max(0, int(getattr(eng.kv_cache, "swa_num_tokens", 0) or 0) - 1)
        return dict(
            num_pages=eng.num_pages,
            moe_cache_size=eng.moe_offload_cache.cache_size if eng.moe_offload_cache is not None else None,
            num_mamba_slots=(eng.linear_state_pool.num_slots - 1) if eng.linear_state_pool is not None else None,
            num_swa_pages=num_swa_pages,
        )

    def _log_cache_geometry(self, event: str) -> None:
        """One-line readout of every pool's new size + VRAM after a rebuild changed them:
        full KV always; swa/mamba/MoE only for models with the pool. Byte figures are
        best-effort (0 when a unit cost cannot be measured) and must never block the reply."""
        from freetoken.kvcache.cache_status import compute_cache_pools, compute_cache_unit_bytes

        try:
            pools = compute_cache_pools(self.engine)
            unit = compute_cache_unit_bytes(self.engine)
            kv_tokens = pools["num_pages"] * pools["page_size"]
            parts = [
                f"KV {pools['num_pages']} pages"
                f" ({kv_tokens} tokens, {_gib(kv_tokens * unit['kv_bytes_per_token'])})"
            ]
            if pools["num_swa_pages"]:
                swa_tokens = pools["num_swa_pages"] * pools["swa_page_size"]
                parts.append(
                    f"swa {pools['num_swa_pages']} pages"
                    f" ({swa_tokens} tokens, {_gib(swa_tokens * unit['swa_bytes_per_token'])})"
                )
            if pools["num_mamba_slots"]:
                parts.append(
                    f"mamba {pools['num_mamba_slots']} slots"
                    f" ({_gib(pools['num_mamba_slots'] * unit['mamba_bytes_per_slot'])})"
                )
            moe = self.engine.moe_offload_cache
            if moe is not None:
                parts.append(
                    f"MoE cache {moe.cache_size}/{moe.num_layers * moe.num_experts}"
                    f" ({_gib(moe.cache_size * unit['moe_bytes_per_expert'])})"
                )
            logger.info_rank0(f"{event}: " + ", ".join(parts))
        except Exception as e:  # noqa: BLE001
            logger.warning(f"could not log cache geometry: {e!r}")

    def _prepare_batch(self, batch: Batch) -> ForwardInput:
        self.engine.graph_runner.pad_batch(batch)
        self._forward_iter += 1
        if batch.is_decode:
            # Free each decoding request's now-out-of-window SWA slots BEFORE the alloc below,
            # so they can back the new token -- this is what bounds the per-request swa
            # footprint during decode. (no-op unless the model is SWA / paged swa pool.)
            self.cache_manager.maybe_free_swa_out_of_window(
                batch.reqs, forward_iter=self._forward_iter)
            for req in batch.reqs:
                req.decode_batch_idx += 1
        else:
            # Prefill sibling of the decode driver: free out-of-window swa BEFORE allocating
            # this chunk, so a chunked prompt longer than the swa pool never accumulates its
            # whole swa footprint (which would exhaust alloc_swa). No-op unless SWA/paged.
            self.cache_manager.free_swa_out_of_window_extend(batch.reqs)
        # Polymorphic page allocation: DSV4 allocates window pages + cmp/idx blocks into its
        # slot maps; the generic manager allocates KV pages into the page table.
        self.cache_manager.allocate_paged(batch.reqs)
        if (
            batch.spec is not None
            and self.engine.spec_executor.has_graph(batch.spec.num_input_tokens)
        ):
            # Graph-mode verify step: the executor stages inputs/positions/out_loc as
            # contiguous token-pool/page-table slices and the graphs carry their own
            # metadata, so the per-step building below would be dead work.
            return ForwardInput(batch=batch, sample_args=None, input_tuple=None, write_tuple=None)
        if batch.is_prefill:
            self._gather_multimodal(batch)
        batch.positions = _make_positions(batch, self.device)
        input_mapping = _make_input_tuple(batch, self.device)
        write_mapping = _make_write_tuple(batch, self.device)
        batch.out_loc = self.engine.page_table[input_mapping]
        if self.engine.linear_state_pool is not None:
            if batch.is_decode:
                # GPU GDN-state slot (one per padded request) for the decode gather/scatter;
                # lands in the CUDA-graph input buffer via copy_from. Gate on the cache mode,
                # NOT on whether any padded req has a linear_slot_idx -- the persistent dummy
                # req always carries one (= padding_slot), so that test is True even for naive
                # and would collapse all real naive reqs onto the padding slot. Hybrid: build
                # per padded req from Req.linear_slot_idx (dummy -> padding_slot). Naive: keep
                # the old keying = input_mapping's table_idx column (already staged, no H2D).
                if self.cache_manager.is_hybrid:
                    pool = self.engine.linear_state_pool
                    slots = [r.linear_slot_idx if r.linear_slot_idx is not None
                             else pool.padding_slot for r in batch.padded_reqs]
                    batch.linear_table_idx = torch.tensor(
                        slots, dtype=torch.int32, device="cpu", pin_memory=True
                    ).to(self.device, non_blocking=True)
                else:
                    batch.linear_table_idx = input_mapping[0].to(torch.int32)
            # Per-forward GDN metadata (cu_seqlens / cache_indices / continuation flags),
            # built once here instead of rebuilt in each of the 30 GDN layers. For decode
            # under CUDA graph the persistent cu_seqlens buffer is supplied by set_batch.
            batch.fla_metadata = build_fla_metadata(batch, self.device)
        if batch.is_decode:
            # This batch's padded per-row page-table rows. Backends that snapshot the table for
            # a captured replay (DSV4) read them in prepare_metadata / prepare_for_replay.
            batch.active_table_idx = input_mapping[0].view(-1)
        self.engine.attn_backend.prepare_metadata(batch)
        return ForwardInput(
            batch=batch,
            sample_args=self.engine.sampler.prepare(batch),
            input_tuple=input_mapping,
            write_tuple=write_mapping,
        )

    def _gather_multimodal(self, batch: Batch) -> None:
        """Concatenate per-request vision soft tokens (in request order) for a prefill
        batch so the model can scatter them at image-token positions. ``req.mm_embeds``
        is kept (not cleared) so the cache manager can recognize multimodal requests and
        keep them out of the shared prefix cache (image placeholders share a token id but
        carry per-image content)."""
        parts = [req.mm_embeds for req in batch.reqs if req.mm_embeds is not None]
        if parts:
            batch.mm_embeds = torch.cat(parts, dim=0)

    def _schedule_next_batch(self) -> ForwardInput | None:
        # TODO: support other policies: e.g. DECODE first
        batch = (
            self.prefill_manager.schedule_next_batch(self.prefill_budget)
            or self.decode_manager.schedule_next_batch()
        )
        if batch is None:
            return None
        if batch.is_prefill:
            # Stamp each request's first prefill schedule (getattr: unit tests drive this
            # method on stub schedulers/batches that predate the telemetry fields).
            _now = time.monotonic()
            _t0map = self.__dict__.setdefault("_prefill_t0", {})
            for _req in getattr(batch, "reqs", ()):
                _t0map.setdefault(_req.uid, _now)
        if getattr(self, "spec_k", 0):
            if batch.is_decode:
                self._stage_spec(batch)
            else:
                self._stage_spec_prefill(batch)
        forward_input = self._prepare_batch(batch)
        self._report_prompt_admissions(batch)
        return forward_input

    def _stage_spec_prefill(self, batch: Batch) -> None:
        """Arm the speculative session when a prompt's FINAL prefill chunk is scheduled.

        Marked at schedule time (not at drain): under overlap scheduling the first decode
        step is staged before the prefill batch drains, and the warmup draft must ride that
        first step -- a later warmup would leave a gap in the MTP layer's KV."""
        for req in batch.reqs:
            if isinstance(req, ChunkedReq):
                continue  # intermediate chunk; the final chunk carries the original req
            req.spec_state = "warmup"

    def _stage_spec(self, batch: Batch) -> None:
        """Turn an eligible one-request decode batch into a verify (or warmup) step.

        Speculation engages only for a single greedy request. Any decode step outside that
        shape ends the request's session for good (session-terminating fallback): its MTP
        KV would go stale for normally-decoded tokens, so it only re-arms with a new prefill.
        """
        from freetoken.engine.spec import SpecBatchInfo

        if len(batch.reqs) != 1:
            for req in batch.reqs:
                if req.spec_state != "off":
                    req.spec_state = "off"
                    self.engine.spec_executor.drop(req.uid)
            return
        req = batch.reqs[0]
        if req.spec_state == "off":
            return
        # Verifying k drafts reads k+1 rows but can emit k+1 candidates, followed by
        # another complete draft chain. Reserve that entire worst-case tail BEFORE
        # begin_verify changes the request's lengths or any CUDA graph is launched.
        # The page/token tables may be padded beyond the model's actual context limit;
        # those padding columns are not additional positions the model can consume.
        # DSpark stages its whole block even when it verifies fewer draft tokens.
        draft_capacity = getattr(self.engine.spec_executor, "block", self.spec_k)
        context_limit = min(self.engine.max_seq_len, self.token_pool.shape[1])
        if req.spec_state == "warmup":
            # complete_one advances device_len before warmup_draft writes its drafts.
            # With only one output token left there cannot be another verify anyway.
            if (req.remain_len > 1
                    and req.device_len + 1 + draft_capacity <= context_limit):
                batch.spec_warmup = req
            else:
                req.spec_state = "off"
                self.engine.spec_executor.drop(req.uid)
            return
        # ready: stage a verify of up to spec_k drafts, clipped by the output budget. With
        # no room for even one draft this is the request's last token -- plain decode, and
        # the session ends (the plain step's position gets no MTP KV).
        # the executor staged req.spec_k_staged drafts for this step (adaptive k); the
        # budget may still clip the verify (a smaller captured set or eager serves it)
        k_staged = min(req.spec_k_staged or self.spec_k, self.spec_k)
        k_eff = min(k_staged, req.max_device_len - req.device_len)
        if (k_eff >= 1
                and req.device_len + k_eff + 1 + draft_capacity <= context_limit):
            req.begin_verify(k_eff)
            batch.spec = SpecBatchInfo(
                scratch=self.engine.spec_executor.scratch, num_input_tokens=k_eff + 1,
                num_staged_drafts=k_staged,
            )
        else:
            # Ordinary decode finishes the last few context positions. Ending the
            # speculative session keeps the committed GDN/PLE state and accepted token
            # prefix intact; clamping draft writes could overwrite that accepted prefix.
            req.spec_state = "off"
            self.engine.spec_executor.drop(req.uid)

    def _report_prompt_admissions(self, batch: Batch) -> None:
        """Publish first-prefill accounting only after batch preparation succeeded.

        ``send_result`` is rank-aware: TP rank 0 forwards the signal, other ranks are
        no-ops. The offline handler explicitly ignores this online-accounting message.
        """
        if not batch.is_prefill or not batch.prompt_admissions:
            return
        self.send_result(
            [
                PromptAdmittedMsg(uid=uid, prompt_tokens=prompt_tokens, cached_tokens=cached_tokens)
                for uid, prompt_tokens, cached_tokens in batch.prompt_admissions
            ]
        )

    def _flush_abort_acks(self) -> None:
        pending = getattr(self, "_pending_abort_acks", None)
        if not pending:
            return
        uids = sorted(pending)
        pending.clear()
        self.send_result([ErrorReplyMsg(uid=uid, error="request aborted") for uid in uids])

    def _abort_prefill_oom(self, batch: Batch, err: BaseException) -> None:
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        try:
            free_b, total_b = torch.cuda.mem_get_info(self.device)
            mem = f"{free_b / 2**30:.2f} GiB free of {total_b / 2**30:.2f} GiB after the failure"
        except Exception:  # noqa: BLE001
            mem = "memory state unavailable"
        logger.error(
            f"prefill batch of {len(batch.reqs)} request(s) hit CUDA OOM ({mem}); aborting "
            f"the request(s) and continuing: {str(err).splitlines()[0][:200]}"
        )
        replies = []
        for req in list(batch.reqs):
            freed = self.prefill_manager.abort_req(req.uid) or self.decode_manager.abort_req(req.uid)
            target = freed or req
            with self.cache_manager.lazy_free_region():
                self._free_req_resources(target)
            replies.append(
                ErrorReplyMsg(
                    uid=req.uid,
                    error=(
                        "GPU out of memory while prefilling this prompt (another process may be "
                        "holding VRAM, or the prompt is too long for the remaining headroom); the "
                        "request was aborted"
                    ),
                    code="server_error",
                )
            )
        if replies:
            self.send_result(replies)

    def _forward(self, forward_input: ForwardInput) -> ForwardOutput | None:
        batch, sample_args, input_mapping, output_mapping = forward_input
        if input_mapping is not None:
            batch.input_ids = self.token_pool[input_mapping]
        if self.toolcall_anchor_id is not None and not batch.is_prefill:
            self.cache_manager.snapshot_toolcall_anchor(batch.reqs)
        try:
            forward_output = self.engine.forward_batch(batch, sample_args)
        except torch.OutOfMemoryError as e:
            if not batch.is_prefill:
                raise
            # A prefill chunk's transient did not fit (typically another process grabbed VRAM
            # after boot): abort the batch's requests with an error instead of taking the
            # scheduler down. Nothing of this chunk was committed (cached_len is the last
            # finished chunk), so the abort path frees a consistent request.
            self._abort_prefill_oom(batch, e)
            return None
        if batch.spec is None:
            self.token_pool[output_mapping] = forward_output.next_tokens_gpu
        # else: the spec executor already scattered the emitted run and the next draft
        self.decode_manager.filter_reqs(forward_input.batch.reqs)
        return forward_output


def _make_positions(batch: Batch, device: torch.device) -> torch.Tensor:
    needed_size = sum(r.extend_len for r in batch.padded_reqs)
    indices_host = torch.empty(needed_size, dtype=torch.int32, pin_memory=True)
    offset = 0
    for req in batch.padded_reqs:
        length = req.extend_len
        torch.arange(
            req.cached_len,
            req.device_len,
            dtype=torch.int32,
            out=indices_host[offset : offset + length],
        )
        offset += length
    return indices_host.to(device, non_blocking=True)


def _make_input_tuple(batch: Batch, device: torch.device) -> Indice2D:
    mapping_host = torch.empty(len(batch.positions), dtype=torch.int64, pin_memory=True)
    offset = 0
    for req in batch.padded_reqs:
        length = req.extend_len
        mapping_host[offset : offset + length].fill_(req.table_idx)
        offset += length
    return mapping_host.to(device, non_blocking=True), batch.positions.to(torch.int64)


def _make_write_tuple(batch: Batch, device: torch.device) -> Indice2D:
    mapping_list = [req.table_idx for req in batch.reqs]
    mapping_host = torch.tensor(mapping_list, dtype=torch.int64, pin_memory=True)
    write_list = [(req.device_len if req.can_decode else -1) for req in batch.reqs]
    write_host = torch.tensor(write_list, dtype=torch.int64, pin_memory=True)
    return mapping_host.to(device, non_blocking=True), write_host.to(device, non_blocking=True)
