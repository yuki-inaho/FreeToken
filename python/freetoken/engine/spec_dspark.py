"""DSpark speculative decoding for DeepSeek-V4-Flash-0731 (batch size 1).

The verify half is the generic one (``engine/spec.py``): the target runs ``[anchor,
draft_1..k]`` as one ``phase="decode"`` batch with ``Batch.spec`` set, chain speculative
sampling (or greedy match) accepts a prefix, every candidate is scattered into the token pool
and the per-request state is committed to the accepted prefix device-side. What differs from
the MTP executor is the model family (DSV4: the compressors' carry is the only state that
needs a commit, ``models/deepseek_v4/spec_state.py``) and the drafter: DSpark emits a whole
block of ``dspark_block_size`` drafts from ONE pass over the target's layer-40/41/42 hidden
states (``models/deepseek_v4/dspark.py``), so there is no chained per-draft pass.

Step (``verify_step``), with ``c = cached_len`` and ``k1 = k + 1`` verify rows at positions
``c..c+k``:
  1. verify forward -> logits ``[k1, V]``; the model also stashes the rows' DSpark context
     (``scratch.main_hidden_rows``) and the carry intermediates.
  2. accept -> ``cand`` (k1 target tokens for positions ``c+1..``), ``n`` = emitted count.
  3. commit the compressor rings to the first ``n`` rows; D2H of ``cand``/``n``.
  4. draft: write the k1 rows' DSpark main-sequence KV (layers ``n_layers..``), then run the
     block ``[cand[n-1], noise x (block-1)]`` at positions ``c+n ..`` over the main window
     ``<= c+n-1``; the Markov head biases each row by the previous (sampled) token; the
     block's tokens go to ``token_pool[c+n+1 ..]`` and their distributions to
     ``scratch.draft_probs``.
Eager for now (the CUDA-graph family is the next milestone); the sampling stays eager per
the engine's convention either way.
"""

from __future__ import annotations

import os
import time

import torch

from freetoken.engine.spec import (
    SpecForwardOutput,
    _SPEC_DEBUG_STEPS,
    _SPEC_TIME,
    chain_accept,
    renorm_probs,
)
from freetoken.utils import init_logger

logger = init_logger(__name__)


class DsparkExecutor:
    """Drives the DSpark verify/draft cycle for one speculative request (batch size 1)."""

    def __init__(self, engine) -> None:
        from freetoken.models.deepseek_v4.spec_state import DSV4SpecScratch

        config = engine.config
        self.engine = engine
        self.model = engine.model
        self.transformer = engine.model._transformer
        self.device = engine.device
        self.dtype = engine.dtype
        self.k = config.num_spec_tokens
        args = config.model_config.dsv4_args
        assert len(self.transformer.mtp) > 0, (
            "--speculative-algorithm dspark needs the DSpark drafter (is FREETOKEN_LOAD_MTP "
            "set and does the checkpoint ship mtp.* stages?)"
        )
        self.args = args
        self.block = args.dspark_block_size
        self.noise_token_id = args.dspark_noise_token_id
        assert 1 <= self.k <= self.block
        self.scratch = DSV4SpecScratch(
            args, self.k, config.model_config.vocab_size, self.device, self.dtype
        )
        self.token_pool: torch.Tensor | None = None  # wired by the scheduler
        k1 = self.k + 1
        pin = torch.cuda.is_available()
        self._cand_cpu = torch.zeros(k1, dtype=torch.int32, pin_memory=pin)
        self._n_cpu = torch.zeros((), dtype=torch.int32, pin_memory=pin)
        self._conf_cpu = torch.zeros(self.block, dtype=torch.float32, pin_memory=pin)
        self._gen = torch.Generator(device=self.device)
        self._gen.manual_seed(0xD5BA)
        self._window_ar = torch.arange(args.window_size, device=self.device)
        self._block_ar = torch.arange(self.block, device=self.device)
        self.graphs = None  # eager (the scheduler's light-prepare path keys on this)
        self._times: list[tuple[float, ...]] = []

    # ----- engine hooks shared with the MTP executor's names --------------------------------

    def drop(self, uid: int) -> None:  # no per-uid carry: the prefill wrote the draft KV
        return None

    def prefill_mtp(self, batch) -> None:
        # The DSV4 prefill writes the stages' main-sequence KV itself (Transformer._dspark_prefill).
        return None

    # ----- helpers ------------------------------------------------------------------------

    def _stage0(self):
        return self.transformer.mtp[0]

    def _stage_last(self):
        return self.transformer.mtp[-1]

    def _freqs(self):
        return self._stage0().attn.freqs_cis

    def _window_topk(self, table_idx: int, last_pos: torch.Tensor, m: int) -> torch.Tensor:
        """GLOBAL window-pool slots of the main positions ``<= last_pos`` (a ``[1]`` device
        scalar) in the 128-ring, ``-1`` where no such position exists, expanded to the ``m``
        block rows: ``[1, m, win]``. Same rule as the backend's ``window_ctx`` (ring slot j
        holds the latest position ``p <= last_pos`` with ``p % win == j``), read off the
        request's live page-table row -- the spec step drains before anything mutates it."""
        pool = self.engine.ctx.kv_cache
        j = self._window_ar
        lp = last_pos.view(1).to(torch.int64)
        p = lp[:, None] - ((lp[:, None] - j[None, :]) % j.shape[0])  # [1, win]
        row = self.engine.page_table[table_idx]
        ws = pool.translate_full_to_window(row.index_select(0, p.clamp(min=0).view(-1)))
        ws = torch.where(p.view(-1) >= 0, ws, torch.full_like(ws, -1))
        return ws.view(1, 1, -1).expand(1, m, -1).to(torch.int32)

    def _store_main_kv(self, main_hidden: torch.Tensor, positions: torch.Tensor,
                       window_slots: torch.Tensor) -> None:
        """Write the DSpark stages' main-sequence KV for ``main_hidden`` ``[T, n_targets*dim]``
        rows at ``positions`` ``[T]`` (their window slots ``[T]``)."""
        main_x = self._stage0().main_x(main_hidden)
        freqs = self._freqs().index_select(0, positions.to(torch.int64))
        backend = self.engine.attn_backend
        for stage in self.transformer.mtp:
            kv = stage.attn.main_kv(main_x, freqs)
            backend.store_window(kv, stage.layer_id, window_slots)

    def _draw(self, logits_row: torch.Tensor, params, slot: int) -> torch.Tensor:
        """One draft token from a ``[1, V]`` logits row (Markov bias already applied): argmax
        for argmax sessions, else a sample from the renormalized distribution, which is stashed
        as ``draft_probs[slot]`` for the next verify's chain acceptance."""
        from freetoken.kernel.triton.sampling import sampling_from_probs

        if params.is_argmax:
            return torch.argmax(logits_row, dim=-1).to(torch.int32)
        probs = renorm_probs(logits_row, params.temperature, params.top_k, params.top_p)
        self.scratch.draft_probs[slot].copy_(probs[0])
        return sampling_from_probs(
            probs, indices=torch.zeros(1, dtype=torch.int64, device=self.device)
        ).to(torch.int32)

    def _draft_block(self, table_idx: int, anchor: torch.Tensor, anchor_pos: torch.Tensor,
                     params) -> tuple[torch.Tensor, torch.Tensor]:
        """Draft a block from ``anchor`` (``[1]`` int32, the token at position ``anchor_pos``,
        a ``[1]`` device scalar, whose main KV is NOT yet written -- the block rows sit at
        ``anchor_pos + i`` and attend the main window ``<= anchor_pos - 1``). Returns the
        ``[block]`` int32 draft tokens (for positions ``anchor_pos + 1 + i``) and the
        ``[block]`` fp32 confidence logits."""
        from freetoken.models.deepseek_v4.dspark import draft_block_forward, draft_block_ids

        m = self.block
        ids = draft_block_ids(anchor.view(1).to(torch.int32), m, self.noise_token_id)  # [1, m]
        pos = anchor_pos.view(1).to(torch.int64) + self._block_ar  # [m]
        freqs = self._freqs().index_select(0, pos)
        topk = self._window_topk(table_idx, anchor_pos.view(1) - 1, m)
        hidden, logits = draft_block_forward(
            self.transformer, ids, freqs, topk, self.scratch.block_kv
        )  # [m, dim], [m, V]
        last = self._stage_last()
        drafts = torch.zeros(m, dtype=torch.int32, device=self.device)
        conf = torch.zeros(m, dtype=torch.float32, device=self.device)
        prev = anchor.view(1)
        for i in range(m):
            bias, emb = last.markov_head(prev)  # [1, V], [1, rank]
            d = self._draw(logits[i : i + 1] + bias, params, slot=i)
            drafts[i : i + 1] = d
            conf[i : i + 1] = last.confidence_head(hidden[i : i + 1], emb)
            prev = d
        return drafts, conf

    # ----- warmup: the request's first decode step also stages the first block ------------

    def warmup_draft(self, batch, next_tokens_gpu: torch.Tensor) -> None:
        """After the (normal, graph-replayed) first decode step: write that step's DSpark
        main KV (its context rows come from the captured stash) and draft the first block
        from the just-sampled token."""
        req = batch.reqs[0]
        inner = self.transformer
        if self.engine.graph_runner.can_use_cuda_graph(batch):
            mh = inner._main_hidden_graph.get(batch.padded_size)
            assert mh is not None, "graph decode batch without a captured DSpark context stash"
        else:
            mh = inner._main_hidden
        assert mh is not None, "the model must be built with the DSpark context seam armed"
        # complete_one already ran: the decode input sat at device_len - 2, the sampled token
        # (the anchor) at device_len - 1 == cached_len.
        pos_in = req.device_len - 2
        pool = self.engine.ctx.kv_cache
        slot = pool.translate_full_to_window(
            self.engine.page_table[req.table_idx, pos_in : pos_in + 1]
        )
        self._store_main_kv(
            mh[:1], torch.tensor([pos_in], device=self.device), slot
        )
        anchor = next_tokens_gpu[:1].to(torch.int32)
        anchor_pos = torch.tensor([pos_in + 1], device=self.device)
        drafts, conf = self._draft_block(req.table_idx, anchor, anchor_pos, req.sampling_params)
        # next verify's draft slots: positions cached_len + 1 .. == device_len ..
        row = self.token_pool[req.table_idx]
        row[req.device_len : req.device_len + self.block] = drafts
        self._conf_cpu.copy_(torch.sigmoid(conf), non_blocking=True)

    # ----- verify step --------------------------------------------------------------------

    def verify_step(self, batch, sample_args) -> SpecForwardOutput:
        del sample_args
        from freetoken.kernel.triton.sampling import sampling_from_probs

        req = batch.reqs[0]
        params = req.sampling_params
        k1 = batch.spec.num_input_tokens
        c = req.cached_len
        pool_row = self.token_pool[req.table_idx]
        self.scratch.set_step(c, k1)
        if _SPEC_TIME:
            evts = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
            evts[0].record()

        logits = self.model.forward()  # [k1, vocab]: the spec branch computes every row
        draft_ids = batch.input_ids[1:k1]
        if params.is_argmax:
            cand = torch.argmax(logits, dim=-1).to(torch.int32)
            matches = (draft_ids == cand[: k1 - 1]).to(torch.int32)
            n = matches.cumprod(0).sum() + 1
        else:
            target_probs = renorm_probs(logits, params.temperature, params.top_k, params.top_p)
            uniforms = torch.rand(k1 - 1, device=self.device, generator=self._gen)
            n_acc, fallback = chain_accept(
                target_probs, self.scratch.draft_probs[: k1 - 1], draft_ids, uniforms
            )
            n = n_acc + 1
            correction = sampling_from_probs(
                fallback, indices=n_acc.reshape(1).to(torch.int64)
            ).to(torch.int32)
            cand = torch.zeros(k1, dtype=torch.int32, device=self.device)
            cand[: k1 - 1] = draft_ids
            cand.scatter_(0, n_acc.reshape(1).to(torch.int64), correction)

        pool_row[c + 1 : c + 1 + k1] = cand
        n_dev = n.reshape(1).to(torch.int32)
        self.scratch.commit(self.transformer, n_dev)
        new_device_len = (n.to(torch.int64) + (c + 1)).reshape(1)  # == new cached_len + 1

        self._cand_cpu[:k1].copy_(cand, non_blocking=True)
        self._n_cpu.copy_(n, non_blocking=True)
        copy_done = torch.cuda.Event()
        copy_done.record(self.engine.stream)
        if _SPEC_TIME:
            evts[1].record()

        # Draft: the verify rows' DSpark main KV (rows past the accepted run are never read:
        # the block's window is bounded by the new anchor's predecessor), then the block from
        # the new anchor cand[n-1] at position c+n.
        positions = torch.arange(c, c + k1, device=self.device)
        self._store_main_kv(
            self.scratch.main_hidden_rows[:k1], positions, self.scratch.window_slots
        )
        sel = (n - 1).reshape(1).to(torch.int64)
        anchor = cand.index_select(0, sel)
        anchor_pos = new_device_len - 1
        drafts, conf = self._draft_block(req.table_idx, anchor, anchor_pos, params)
        row_cap = pool_row.shape[0] - 1
        dst = torch.clamp(new_device_len + self._block_ar, max=row_cap)
        pool_row.index_copy_(0, dst, drafts)
        self._conf_cpu.copy_(torch.sigmoid(conf), non_blocking=True)
        if _SPEC_TIME:
            evts[2].record()
            evts[3].record()
            evts[3].synchronize()
            self._times.append(tuple(evts[i].elapsed_time(evts[i + 1]) for i in range(3)))
            if len(self._times) % 100 == 0:
                import statistics

                cols = list(zip(*self._times[-100:]))
                print(
                    "[dspark-time ms] " + "  ".join(
                        f"{name}={statistics.median(col):.2f}"
                        for name, col in zip(("verify+accept+commit", "draft", "-"), cols)
                    ),
                    flush=True,
                )
        if _SPEC_DEBUG_STEPS:
            self._debug_steps = getattr(self, "_debug_steps", 0) + 1
            if self._debug_steps <= _SPEC_DEBUG_STEPS:
                torch.cuda.synchronize()
                print(
                    f"[dspark {self._debug_steps}] c={c} inputs={batch.input_ids.tolist()} "
                    f"cand={cand.tolist()} n={int(n.item())} drafts={drafts.tolist()} "
                    f"conf={[round(v, 3) for v in torch.sigmoid(conf).tolist()]}",
                    flush=True,
                )
        return SpecForwardOutput(
            candidates_gpu=cand,
            candidates_cpu=self._cand_cpu[:k1],
            num_emitted_cpu=self._n_cpu,
            copy_done=copy_done,
        )

    def recommit(self, req, num_inputs: int) -> None:
        """Re-run the carry commit at a smaller accepted count (drain-side truncation)."""
        n = torch.tensor([num_inputs], device=self.device, dtype=torch.int32)
        self.scratch.commit(self.transformer, n)


__all__ = ["DsparkExecutor"]
