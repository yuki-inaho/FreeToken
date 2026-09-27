"""DeepSeek-V4.1-Flash model (engine-native FreeToken port of inference/model.py).

The block chain is the reference's (Single-Pass mHC, CSA2 attention, bias-routed MoE,
Engram at layers 1 and 14) on FreeToken's paged DSV4 engine seams: KV in the DSV4 paged
pools (window rings per layer, compressed / indexer tiers at the CSA2 owners), routed FP4
experts from the offload cache, the DSpark drafter as extra window-tier layers.

Prefill has two modes (``FREETOKEN_DSV41_PREFILL_MODE``):

* ``ced`` (default) -- the tech report's deployment: the 20-layer causal encoder runs over
  every chunk; the decoder's global KV (layer ``decoder_start``'s compressed KV + index keys)
  is produced once from the encoder output over ALL chunk tokens; then the decoder replays
  only the last ``window`` tokens of each request (Decoder SWA Bounded Replay: the sliding
  window truncated to the replay segment) to seed the decoder window KV and produce the
  logits. Approximate by design (post-training simulated it); halves prefill compute.
* ``full`` -- all 40 layers over every token, the reference-exact path (the oracle).

Decode and speculative verify run all 40 layers per token in both modes.
"""

from __future__ import annotations

import os

import torch
from torch import nn

from freetoken.core import get_global_ctx
from freetoken.kernel.triton.dsv4.bf16_linear import bf16_linear_fp32
from freetoken.models.blocks import BaseLLMModel
from freetoken.utils import init_logger

from .args import DeepseekV41Args
from .attention import Attention
from .block import Block
from .engram import Engram
from .layers import RMSNorm, make_identity_pre_mix
from .moe import MoE
from .runtime import SharedAttentionRuntime

logger = init_logger(__name__)

PREFILL_MODE_ENV = "FREETOKEN_DSV41_PREFILL_MODE"


def prefill_mode_from_env() -> str:
    mode = os.environ.get(PREFILL_MODE_ENV, "ced").strip().lower()
    if mode not in ("ced", "full"):
        raise ValueError(f"{PREFILL_MODE_ENV} must be 'ced' or 'full', got {mode!r}")
    return mode


class Transformer(nn.Module):
    def __init__(self, args: DeepseekV41Args, *, prefill_mode: str | None = None, quant_act: bool = False):
        super().__init__()
        self.args = args
        self.norm_eps = args.norm_eps
        self.hc_mult = hc_mult = args.hc_mult
        self.dim = args.dim
        self.prefill_mode = prefill_mode or prefill_mode_from_env()
        self.decoder_start = args.decoder_start() if self.prefill_mode == "ced" else args.n_layers
        if self.prefill_mode == "ced":
            late = [L for L in args.engram_layer_ids if L >= self.decoder_start]
            assert not late, f"CED prefill needs Engram layers in the encoder, got {late}"
        self.embed = nn.Embedding(args.vocab_size, args.dim)
        self.embed.weight.requires_grad_(False)
        self.layers = nn.ModuleList(
            [Block(i, args, attn=Attention(i, args), ffn=MoE(i, args, quant_act=quant_act))
             for i in range(args.n_layers)]
        )
        for L in args.engram_layer_ids:
            self.layers[L].engram = Engram(args, L)
        self.norm = RMSNorm(args.dim, self.norm_eps)
        self.head = nn.Parameter(torch.empty(args.vocab_size, args.dim, dtype=torch.bfloat16), requires_grad=False)

        # DSpark drafter stages (checkpoint ``mtp.*``), built only for speculative decoding.
        self.mtp = nn.ModuleList()
        self.target_layer_ids: tuple[int, ...] = tuple(args.dspark_target_layer_ids)
        self._keep_main_hidden = False
        self._main_hidden: torch.Tensor | None = None
        self._main_hidden_graph: dict[int, torch.Tensor] = {}
        self._main_hidden_last: torch.Tensor | None = None
        from .config import dspark_build_enabled

        if dspark_build_enabled(args):
            from .dspark import DSparkBlock

            self.mtp = nn.ModuleList(
                [DSparkBlock(args.n_layers + s, args, s) for s in range(args.n_mtp_layers)]
            )
            self._keep_main_hidden = True

    # ----- binding ----------------------------------------------------------------------
    def bind(self, pool, device: torch.device) -> None:
        for layer in self.layers:
            layer.attn.bind(pool, device)
        for stage in self.mtp:
            stage.attn.bind(pool, device)

    # ----- head -------------------------------------------------------------------------
    def logits(self, h: torch.Tensor, pre_mix: torch.Tensor, rows: torch.Tensor | None = None) -> torch.Tensor:
        """``h`` ``[1, T, hc, dim]`` (or ``[B, 1, hc, dim]``), ``pre_mix`` ``[M, hc]`` (the last
        block's ``ffn_pre``) -> fp32 logits for the selected rows (all rows when None)."""
        last = self.layers[-1]
        hidden = last.pre_apply(h, pre_mix)  # [.., dim]
        hidden = hidden.reshape(-1, self.dim)
        if rows is not None:
            hidden = hidden.index_select(0, rows)
        return bf16_linear_fp32(self.norm(hidden), self.head).float()

    # ----- DSpark context ---------------------------------------------------------------
    def _stash_main_hidden(self, mains: list[torch.Tensor], B: int) -> None:
        mh = torch.cat(mains, dim=-1).view(B, -1)
        if mh.is_cuda and torch.cuda.is_current_stream_capturing():
            self._main_hidden_graph[B] = mh
        else:
            self._main_hidden = mh

    def _dspark_prefill(self, main_hidden, segments, flat_positions, last_indices) -> None:
        """Write the stages' main-sequence KV for the given prefill rows and keep each
        request's final context row for the warmup draft. ``main_hidden`` ``[T, n_targets*dim]``."""
        stage0 = self.mtp[0]
        main_x = stage0.main_x(main_hidden)
        self._main_hidden_last = main_hidden.index_select(0, last_indices)
        attn0 = stage0.attn
        freqs = attn0.freqs_cis.index_select(0, flat_positions)
        slots = torch.cat(
            [attn0.attn.window_slots_of(ti, start, start + n) for (_off, n, ti, start) in segments]
        )
        for stage in self.mtp:
            kv = stage.attn.main_kv(main_x, freqs)
            stage.attn.attn.store_window(kv, stage.layer_id, slots)

    # ----- prefill ----------------------------------------------------------------------
    def _run_layers(
        self, h, pre_mix, layer_ids, segments, flat_positions, runtime, window_mode,
        engram_feats, mains,
    ):
        for L in layer_ids:
            layer = self.layers[L]
            eng = getattr(layer, "engram", None)
            if eng is not None:
                h = eng(h, engram_feats[L])
            if self._keep_main_hidden and L in self.target_layer_ids:
                mains.append(h.mean(dim=2))  # [1, T, dim]: the DSpark context (block INPUT)
            attn = layer.attn
            h, pre_mix = layer(
                h, pre_mix,
                attn_call=lambda y, attn=attn: attn.forward_ragged(
                    y, segments, flat_positions, runtime, window_mode
                ),
            )
        return h, pre_mix

    def prefill_batched(
        self, input_ids: torch.Tensor, segments, flat_positions: torch.Tensor,
        last_indices: torch.Tensor, engram_feats: dict[int, torch.Tensor] | None = None,
        segment_new: list[int] | None = None,
    ) -> torch.Tensor:
        """Ragged batched prefill. ``input_ids`` ``[1, T]`` (the requests' token streams
        concatenated; under CED a stream may have been extended backwards to re-encode the
        tail, ``segment_new[i]`` being the genuinely new count), ``segments``
        ``[(offset, n, table_idx, start_pos)]``, ``flat_positions`` ``[T]``, ``last_indices``
        ``[B]`` (each request's final row), ``engram_feats`` per Engram layer ``[T, feat]``."""
        device = input_ids.device
        T = input_ids.shape[1]
        h = self.embed(input_ids).unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
        pre_mix = make_identity_pre_mix(T, self.hc_mult, device)
        engram_feats = engram_feats or {}
        mains: list[torch.Tensor] = []
        runtime = SharedAttentionRuntime()

        if self.prefill_mode == "full":
            h, pre_mix = self._run_layers(
                h, pre_mix, range(self.args.n_layers), segments, flat_positions, runtime,
                "causal", engram_feats, mains,
            )
            if mains:
                self._dspark_prefill(torch.cat(mains, dim=-1)[0], segments, flat_positions, last_indices)
            logits = self.logits(h, pre_mix, last_indices)
            self._maybe_dspark_harness(input_ids, segments)
            return logits

        # ---- CED: encoder over every token ----
        dec = self.decoder_start
        h, pre_mix = self._run_layers(
            h, pre_mix, range(dec), segments, flat_positions, runtime, "causal", engram_feats, mains,
        )
        # ---- the decoder's global KV, from the encoder output, over ALL tokens ----
        dec_layer = self.layers[dec]
        x_dec = dec_layer.attn_input(h, pre_mix)  # [1, T, dim]
        for off, n, ti, start in segments:
            slots = dec_layer.attn.attn.window_slots_of(ti, start, start + n)
            dec_layer.attn.produce_shared_kv(x_dec[:, off:off + n], start, ti, slots)
        # ---- Decoder SWA Bounded Replay over each request's last `window` rows ----
        win = self.args.window_size
        rows, segs_r, off_r = [], [], 0
        for off, n, ti, start in segments:
            m = min(win, start + n)
            # The replay needs the encoder output of the last `win` prompt tokens; a
            # continuation shorter than that must have been extended backwards by the
            # backend (attention/dsv4_sparse.py `_extend_ced_segments`) so those rows are
            # in this stream. Refuse rather than replay a truncated window silently.
            assert n >= m, (
                f"CED replay: segment (start={start}, n={n}) carries fewer rows than the "
                f"replay window {m}; the prefill metadata must extend short continuations"
            )
            rows.append(torch.arange(off + n - m, off + n, device=device))
            segs_r.append((off_r, m, ti, start + n - m))
            off_r += m
        rows_r = torch.cat(rows)
        h_r = h.index_select(1, rows_r)
        pre_r = pre_mix.index_select(0, rows_r)
        pos_r = flat_positions.index_select(0, rows_r)
        last_r = torch.tensor([o + m - 1 for (o, m, _ti, _s) in segs_r], device=device, dtype=torch.long)
        runtime_r = SharedAttentionRuntime(produce_kv=False)
        mains_r: list[torch.Tensor] = []
        h_r, pre_r = self._run_layers(
            h_r, pre_r, range(dec, self.args.n_layers), segs_r, pos_r, runtime_r, "replay",
            {}, mains_r,
        )
        if mains_r:
            self._dspark_prefill(torch.cat(mains_r, dim=-1)[0], segs_r, pos_r, last_r)
        logits = self.logits(h_r, pre_r, last_r)
        self._maybe_dspark_harness(input_ids, segments)
        return logits

    def _maybe_dspark_harness(self, input_ids: torch.Tensor, segments) -> None:
        """``FREETOKEN_DSPARK_HARNESS=1``: after a cold single-request prompt, score the
        drafter teacher-forced against the prompt (``dspark.teacher_forced_report``; under
        CED restricted to the replayed tail, the only positions with stage KV)."""
        if (
            os.getenv("FREETOKEN_DSPARK_HARNESS")
            and len(self.mtp) > 0
            and len(segments) == 1
            and segments[0][3] == 0
        ):
            from .dspark import teacher_forced_report

            teacher_forced_report(self, input_ids.view(-1), segments[0][2])

    # ----- decode / verify --------------------------------------------------------------
    def decode(
        self, input_ids: torch.Tensor, pos: torch.Tensor, cmp_stage_cap: int,
        engram_feats: dict[int, torch.Tensor] | None = None, spec=None,
    ) -> torch.Tensor:
        """``input_ids`` ``[B, 1]``, ``pos`` ``[B]``; ``spec`` marks a speculative verify
        (the B rows are consecutive tokens of one request)."""
        B = input_ids.size(0)
        device = input_ids.device
        rows = torch.arange(B, device=device)
        h = self.embed(input_ids).unsqueeze(2).repeat(1, 1, self.hc_mult, 1)  # [B, 1, hc, dim]
        pre_mix = make_identity_pre_mix(B, self.hc_mult, device)
        wctx = get_global_ctx().batch.attn_metadata.window_ctx(pos, rows)
        if spec is not None:
            spec.scratch.window_slots = wctx[0]
        engram_feats = engram_feats or {}
        runtime = SharedAttentionRuntime()
        mains: list[torch.Tensor] = []
        for L, layer in enumerate(self.layers):
            eng = getattr(layer, "engram", None)
            if eng is not None:
                h = eng(h, engram_feats[L])
            if self._keep_main_hidden and L in self.target_layer_ids:
                mains.append(h.mean(dim=2))  # [B, 1, dim]
            attn = layer.attn
            h, pre_mix = layer(
                h, pre_mix,
                attn_call=lambda y, attn=attn: attn.decode_step(
                    y, pos, rows, cmp_stage_cap, wctx, runtime, spec
                ),
            )
        if mains:
            self._stash_main_hidden(mains, B)
            if spec is not None:
                spec.scratch.main_hidden_rows[:B].copy_(torch.cat(mains, dim=-1).view(B, -1))
        return self.logits(h, pre_mix)


class DeepseekV41ForCausalLM(BaseLLMModel):
    """Engine adapter (see the 0731 adapter for the binding/rebuild contract)."""

    def __init__(self, config):
        self._config = config
        self._args: DeepseekV41Args = config.dsv4_args
        self._transformer = Transformer(self._args)
        self._bound = False
        self._engram = None            # freetoken.engram.EngramRuntime (host side)
        self._engram_dec: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._engram_prefill_fetch = None

    # ----- binding ----------------------------------------------------------------------
    def _ensure_bound(self) -> None:
        if self._bound:
            return
        pool = get_global_ctx().kv_cache
        self._transformer.bind(pool, pool.device)
        self._bound = True

    def mark_for_rebind(self) -> None:
        self._bound = False

    # ----- offload cache hooks (same contract as 0731) ----------------------------------
    def _iter_offload_moe_layers(self):
        for block in self._transformer.layers:
            yield block.ffn.experts
        for stage in self._transformer.mtp:
            yield stage.ffn.experts

    def append_spec_expert_bank(self, banks, engine_config, device: torch.device) -> int:
        stages = len(self._transformer.mtp)
        if stages == 0:
            return 0
        if banks.quant_format != "ds_fp4":
            raise ValueError(
                "DSpark experts join the expert cache as extra ds_fp4 bank layers and require "
                f"the ds_fp4 format, got {banks.quant_format!r}"
            )
        from .weight import load_dspark_expert_bank_layers

        layer_banks = load_dspark_expert_bank_layers(
            engine_config.model_path, self._args, dummy=engine_config.use_dummy_weight
        )
        for name, tensors in layer_banks.items():
            banks.sources[name].extend(tensors)
        if banks.layer_residency is not None:
            from freetoken.moe.host_banks import HostResidency

            banks.layer_residency.extend([HostResidency.PINNED.value] * stages)
        return stages

    # ----- Engram host tables -----------------------------------------------------------
    def load_host_tables(self, engine_config) -> int:
        """Engine hook: build the Engram runtime (token map, hash layout, the two NVMe-backed
        row stores, pinned staging). Returns the pinned bytes it holds."""
        args = self._args
        if not args.has_engram:
            return 0
        from freetoken.engram.runtime import EngramRuntime

        block = args.dspark_block_size if args.dspark_enabled else 0
        max_bs = max(1, int(getattr(engine_config, "max_running_req", 1) or 1))
        if engine_config.use_dummy_weight or os.environ.get("FREETOKEN_ENGRAM", "").lower() == "zero":
            # FREETOKEN_ENGRAM=zero: the ablation A/B from the plan (tables replaced by zero
            # rows; the Engram gate then contributes nothing) -- "are the tables wired right".
            from freetoken.engram.layout import EngramLayout

            self._engram = EngramRuntime.zeros(
                EngramLayout.from_args(args), args.vocab_size, args.engram_pad_id,
                max_decode_bs=max_bs, max_verify_rows=block + 1,
            )
        else:
            self._engram = EngramRuntime.build(
                engine_config.model_path, args,
                engram_dir=os.environ.get("FREETOKEN_ENGRAM_DIR") or None,
                max_decode_bs=max_bs, max_verify_rows=block + 1,
            )
        st = self._engram.stats()
        logger.info_rank0(
            f"Engram: {len(args.engram_layer_ids)} tables served from disk, staging "
            f"{st['staging_bytes'] / 2**20:.0f} MiB (pinned={st['pinned']})"
        )
        return int(st["staging_bytes"]) if st.get("pinned") else 0

    def engram_stats(self) -> dict:
        return self._engram.stats() if self._engram is not None else {}

    def _feats_from_views(self, vals, scales, device, out=None) -> torch.Tensor:
        """Host uint8 rows ``[T, C, dim]`` / ``[T, C, 8]`` -> device bf16 ``[T, C*dim]``."""
        from freetoken.kernel.triton.dsv4.engram_dequant import engram_dequant

        T, C, D = vals.shape
        v = torch.from_numpy(vals).reshape(T * C, D).to(device, non_blocking=True)
        s = torch.from_numpy(scales).reshape(T * C, -1).to(device, non_blocking=True)
        feats = engram_dequant(v, s, head_dim=D, block=D // scales.shape[-1])
        return feats.view(T, C * D)

    def _engram_prefill_feats(self, batch, segments, T: int) -> dict[int, torch.Tensor]:
        if self._engram is None:
            return {}
        from freetoken.engram.runtime import PrefillItem

        device = get_global_ctx().kv_cache.device
        items = []
        for req, (_off, n, _ti, start) in zip(batch.reqs, segments):
            ids = req.input_ids.numpy() if hasattr(req.input_ids, "numpy") else req.input_ids
            items.append(PrefillItem(uid=req.uid, input_ids=ids, start=start, end=start + n))
        # The staging ring holds ``prefill_cap_tokens`` tokens per fetch (16K by default); the
        # scheduler's prefill chunk can be larger (DSV4: ~half the window pool -- 25.7K tokens
        # with a 262K KV floor, which raised "rows exceed the staging capacity" on the 5090).
        # Split such a chunk into sub-fetches in token order (the per-uid n-gram history
        # advances with each), staging each one through the device before the next.
        layout = self._engram.layout
        cap_tokens = max(1, self._engram._prefill_ring.cap_rows // (layout.n_layers * layout.n_hash_cols))
        if T <= cap_tokens:
            fetch = self._engram.prepare_prefill(items)
            feats = {}
            for L in self._args.engram_layer_ids:
                vals, scales = self._engram.layer_view(fetch, L, T)
                feats[L] = self._feats_from_views(vals, scales, device)
            self._engram_prefill_fetch = fetch
            return feats
        parts: dict[int, list[torch.Tensor]] = {L: [] for L in self._args.engram_layer_ids}
        for it in items:
            for s in range(it.start, it.end, cap_tokens):
                e = min(it.end, s + cap_tokens)
                fetch = self._engram.prepare_prefill([PrefillItem(uid=it.uid, input_ids=it.input_ids, start=s, end=e)])
                for L in self._args.engram_layer_ids:
                    vals, scales = self._engram.layer_view(fetch, L, e - s)
                    parts[L].append(self._feats_from_views(vals, scales, device))
                torch.cuda.current_stream().synchronize()  # the H2D copies read the slot
                fetch.release()
        self._engram_prefill_fetch = None
        return {L: torch.cat(chunks, dim=0) for L, chunks in parts.items()}

    def _engram_decode_buffers(self, bs: int, device):
        bufs = self._engram_dec.get(bs)
        if bufs is None:
            layout = self._engram.layout
            C, D = layout.n_hash_cols, layout.head_dim
            n = len(self._args.engram_layer_ids)
            vals = torch.zeros(n, bs * C, D, dtype=torch.uint8, device=device)
            scales = torch.zeros(n, bs * C, D // 32, dtype=torch.uint8, device=device)
            bufs = self._engram_dec[bs] = (vals, scales)
        return bufs

    def _engram_decode_feats(self, bs: int, device) -> dict[int, torch.Tensor]:
        """Dequantize the static decode staging buffers (graph-capturable: fixed addresses)."""
        if self._engram is None:
            return {}
        from freetoken.kernel.triton.dsv4.engram_dequant import engram_dequant

        vals, scales = self._engram_decode_buffers(bs, device)
        layout = self._engram.layout
        C, D = layout.n_hash_cols, layout.head_dim
        feats = {}
        for li, L in enumerate(self._args.engram_layer_ids):
            feats[L] = engram_dequant(vals[li], scales[li], head_dim=D, block=32).view(bs, C * D)
        return feats

    def prepare_host_tables_for_replay(self, batch) -> None:
        """Engine hook (graph replay / eager decode): hash this step's input tokens, fetch
        their Engram rows and stage them into the static decode buffers the forward reads."""
        if self._engram is None:
            return
        spec = getattr(batch, "spec", None)
        bs = spec.num_input_tokens if spec is not None else batch.padded_size
        device = get_global_ctx().kv_cache.device
        vals_d, scales_d = self._engram_decode_buffers(bs, device)
        ids = batch.input_ids[:bs].cpu()
        if spec is not None:
            # verify rows: the anchor + drafts of ONE request, hashed without advancing;
            # the executor commits the accepted tokens via ``engram_commit``.
            req = batch.reqs[0]
            fetch = self._engram.prepare_verify(req.uid, int(ids[0]), ids[1:].tolist())
            T = bs
        else:
            uids = [r.uid for r in batch.reqs][:bs]
            fetch = self._engram.prepare_decode(ids, uids, bs)
            T = bs
        for li, L in enumerate(self._args.engram_layer_ids):
            vals, scales = self._engram.layer_view(fetch, L, T)
            C, D = vals.shape[1], vals.shape[2]
            vals_d[li].copy_(torch.from_numpy(vals).reshape(T * C, D), non_blocking=True)
            scales_d[li].copy_(torch.from_numpy(scales).reshape(T * C, -1), non_blocking=True)
        if spec is not None:
            torch.cuda.current_stream().synchronize()
            fetch.release()

    def engram_commit(self, uid, accepted_tokens) -> None:
        """Speculative executor hook: advance the request's n-gram history by the accepted
        verify tokens (anchor + accepted drafts, in order)."""
        if self._engram is not None:
            self._engram.commit_verify(uid, accepted_tokens)

    def drop_request(self, uid) -> None:
        if self._engram is not None:
            self._engram.drop(uid)

    # ----- state dict -------------------------------------------------------------------
    def state_dict(self, *, prefix: str = "", result=None):
        result = {} if result is None else result
        for name, p in self._transformer.named_parameters():
            result[f"{prefix}.{name}" if prefix else name] = p
        return result

    def load_state_dict(self, state_dict, *, prefix: str = "", _internal: bool = False) -> None:
        casted = {}
        for name, p in self._transformer.named_parameters():
            key = f"{prefix}.{name}" if prefix else name
            if key not in state_dict:
                raise RuntimeError(f"Missing weight for DeepSeek-V4.1 parameter: {key}")
            casted[name] = state_dict.pop(key).to(p.dtype)
        if state_dict and not _internal:
            raise RuntimeError(f"Unexpected keys in state_dict: {list(state_dict.keys())}")
        self._transformer.load_state_dict(casted, assign=True, strict=False)

    # ----- forward ----------------------------------------------------------------------
    def forward(self) -> torch.Tensor:
        self._ensure_bound()
        batch = get_global_ctx().batch
        input_ids = batch.input_ids.long()
        md = batch.attn_metadata
        if batch.is_prefill:
            segments = md.segments
            T = input_ids.numel()
            feats = self._engram_prefill_feats(batch, segments, T)
            try:
                return self._transformer.prefill_batched(
                    input_ids.view(1, -1), segments, batch.positions.long(),
                    md.last_indices.long(), engram_feats=feats,
                )
            finally:
                if self._engram_prefill_fetch is not None:
                    torch.cuda.current_stream().synchronize()
                    self._engram_prefill_fetch.release()
                    self._engram_prefill_fetch = None
        spec = getattr(batch, "spec", None)
        B = spec.num_input_tokens if spec is not None else batch.padded_size
        pos = batch.positions.long().view(-1)[:B]
        device = input_ids.device
        capturing = torch.cuda.is_current_stream_capturing()
        if capturing:
            cmp_stage_cap = md.stage_width - 1
        else:
            cmp_stage_cap = int(pos.max().item())
            if self._engram is not None and md.full_snap is None:
                # eager decode / verify: the graph runner did not stage the rows for us
                self.prepare_host_tables_for_replay(batch)
        feats = self._engram_decode_feats(B, device)
        return self._transformer.decode(input_ids.view(B, 1), pos, cmp_stage_cap, feats, spec)


__all__ = ["Transformer", "DeepseekV41ForCausalLM", "prefill_mode_from_env"]
