"""DSpark for DeepSeek-V4.1: the checkpoint's ``mtp.*`` 3-stage block drafter.

Same shape as the 0731 drafter (see ``models/deepseek_v4/dspark.py``): ``n_mtp_layers``
V4.1 blocks -- window-only MLA attention over the MAIN sequence's draft KV (paged window
tier, layers ``n_layers + stage``) plus the block's own rows (bidirectional inside the
block), single-pass mHC, and a MoE with the stage geometry (128 routed FP4 experts, top-3,
served from the shared offload cache as extra bank layers) -- drafting ``dspark_block_size``
tokens in one forward from stage 0's projection of the target's hidden states (the mean over
the HC streams of the block INPUT at ``dspark_target_layer_ids``).

V4.1 differences: no ``hc_head_*`` (the last stage collapses its streams with its own
``ffn_pre``, exactly like the main head), the Markov head is ``markov_head.{embed, head}``,
the confidence projection ships bf16 and is held fp32, and the attention/kv numerics follow
the V4.1 layer (no per-head q norm, whole-vector FP8 window KV with 32-blocks).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from freetoken.kernel.triton.dsv4.bf16_linear import bf16_linear_fp32
from freetoken.kernel.triton.dsv4.fp8_linear import act_quant_fp8_inplace
from freetoken.models.deepseek_v4.ops import apply_rotary_emb

from .args import DeepseekV41Args
from .attention import Attention
from .block import Block
from .layers import Linear, RMSNorm, make_identity_pre_mix
from .moe import MoE


class DSparkAttention(Attention):
    """Window-only attention for a DSpark stage: keys are the main-sequence draft KV written
    from ``main_x`` (paged window tier, layer ``self.layer_id``) plus the block's own rows."""

    def __init__(self, layer_id: int, args: DeepseekV41Args):
        super().__init__(layer_id, args)
        assert self.mode == "window", "DSpark stages are window-only (compress_ratio 0)"

    def main_kv(self, main_x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        """Main-sequence KV rows for this stage: ``main_x`` ``[T, dim]`` at per-row rope
        ``freqs`` ``[T, rd//2]`` -> ``[T, head_dim]`` ready for ``store_window``."""
        rd = self.rope_head_dim
        kv = self.kv_norm(self.wkv(main_x.unsqueeze(0)))  # [1, T, head_dim]
        apply_rotary_emb(kv[..., -rd:], freqs)
        act_quant_fp8_inplace(kv, self.fp8_block)
        return kv[0]

    def draft(
        self, x: torch.Tensor, freqs: torch.Tensor, window_topk: torch.Tensor,
        block_kv: torch.Tensor,
    ) -> torch.Tensor:
        """``x`` ``[1, m, dim]`` (post-norm block rows), ``freqs`` ``[m, rd//2]``,
        ``window_topk`` ``[1, m, n_window]`` global window slots (-1 masked), ``block_kv``
        ``[m, head_dim]`` scratch this call fills with the rows' own KV (read as the
        "compressed" half so every row attends every block row)."""
        rd = self.rope_head_dim
        m = x.size(1)
        _, q, kv = self._qkv(x, freqs, decode=False)
        block_kv.copy_(kv[0])
        n_window = window_topk.shape[-1]
        block_cols = torch.arange(m, device=x.device, dtype=torch.int32).view(1, 1, m).expand(1, m, m)
        topk_idxs = torch.cat([window_topk.int(), block_cols], dim=-1)
        o = self.attn.attend(
            q, self.layer_id, topk_idxs, n_window, self.attn_sink, self.softmax_scale,
            has_compression=True, cmp_override=block_kv,
        )
        apply_rotary_emb(o[..., -rd:], freqs, True)
        return self._wo(o, 1, m)


class DSparkMarkovHead(nn.Module):
    """Low-rank token-transition bias: ``bias(prev) = head(embed[prev])`` over the vocab."""

    def __init__(self, vocab_size: int, rank: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, rank, dtype=torch.bfloat16)
        self.embed.weight.requires_grad_(False)
        self.head = Linear(rank, vocab_size, kind="bf16")

    def forward(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``token_ids`` ``[B]`` -> (``logits_bias`` ``[B, vocab]`` fp32, ``embed`` ``[B, rank]``)."""
        embed = self.embed.weight.index_select(0, token_ids.to(torch.int64))
        bias = bf16_linear_fp32(embed, self.head.weight)
        return bias.float(), embed


class DSparkConfidenceHead(nn.Module):
    """One fp32 projection over ``cat(hidden, markov_embed)`` -> a survival logit."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.proj = Linear(input_dim, 1, kind="fp32")

    def forward(self, hidden: torch.Tensor, markov_embed: torch.Tensor) -> torch.Tensor:
        x = torch.cat([hidden.float(), markov_embed.float()], dim=-1)
        return self.proj(x).squeeze(-1)


class DSparkBlock(Block):
    """One DSpark stage (checkpoint ``mtp.{stage_id}.*``)."""

    def __init__(self, layer_id: int, args: DeepseekV41Args, stage_id: int):
        super().__init__(layer_id, args, attn=DSparkAttention(layer_id, args), ffn=MoE(layer_id, args))
        self.stage_id = stage_id
        self.block_size = args.dspark_block_size
        self.noise_token_id = args.dspark_noise_token_id
        self.n_targets = len(args.dspark_target_layer_ids)
        self.is_first = stage_id == 0
        self.is_last = stage_id == args.n_mtp_layers - 1
        if self.is_first:
            self.main_proj = Linear(
                args.dim * self.n_targets, args.dim, kind="fp8", block=args.fp8_block_size
            )
            self.main_norm = RMSNorm(args.dim, args.norm_eps)
        if self.is_last:
            self.norm = RMSNorm(args.dim, args.norm_eps)
            self.markov_head = DSparkMarkovHead(args.vocab_size, args.dspark_markov_rank)
            self.confidence_head = DSparkConfidenceHead(args.dim + args.dspark_markov_rank)

    def main_x(self, main_hidden: torch.Tensor) -> torch.Tensor:
        """``main_hidden`` ``[T, n_targets * dim]`` -> ``main_x`` ``[T, dim]``."""
        assert self.is_first, "main_x lives on stage 0"
        return self.main_norm(self.main_proj(main_hidden))

    def draft_step(
        self, x: torch.Tensor, pre_mix: torch.Tensor, freqs: torch.Tensor,
        window_topk: torch.Tensor, block_kv: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``x`` ``[1, m, hc, dim]`` block rows -> (new rows, this stage's ``ffn_pre``)."""
        return self.forward(
            x, pre_mix, attn_call=lambda y: self.attn.draft(y, freqs, window_topk, block_kv)
        )

    def head_hidden(self, x: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
        """Collapse the streams for the head with the last stage's own ``ffn_pre``:
        ``[1, m, hc, dim]`` -> ``[m, dim]`` (pre-norm; the confidence head reads this)."""
        assert self.is_last
        return self.pre_apply(x, pre_mix)[0]

    def block_logits(self, hidden: torch.Tensor, head: torch.Tensor) -> torch.Tensor:
        assert self.is_last
        return bf16_linear_fp32(self.norm(hidden), head).float()


def draft_block_ids(anchor: torch.Tensor, block_size: int, noise_token_id: int) -> torch.Tensor:
    ids = anchor.new_full((anchor.numel(), block_size), noise_token_id)
    ids[:, 0] = anchor
    return ids


def draft_block_forward(
    transformer, ids: torch.Tensor, freqs: torch.Tensor, window_topk: torch.Tensor,
    block_kv: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the block ``ids`` ``[1, m]`` through every stage; returns the last stage's pre-norm
    head hidden ``[m, dim]`` and the base block logits ``[m, vocab]`` fp32."""
    stages = list(transformer.mtp)
    x = transformer.embed(ids.to(torch.int64)).unsqueeze(2).repeat(1, 1, transformer.hc_mult, 1)
    pre_mix = make_identity_pre_mix(ids.numel(), transformer.hc_mult, x.device)
    for stage, buf in zip(stages, block_kv):
        x, pre_mix = stage.draft_step(x, pre_mix, freqs, window_topk, buf)
    last = stages[-1]
    hidden = last.head_hidden(x, pre_mix)
    return hidden, last.block_logits(hidden, transformer.head)


@torch.no_grad()
def teacher_forced_report(
    transformer, input_ids: torch.Tensor, ti: int, *, stride: int = 16, min_ctx: int = 8,
    max_positions: int = 256,
) -> dict[str, list]:
    """Wiring check (``FREETOKEN_DSPARK_HARNESS=1``), run right after a cold single-request
    prefill wrote the stages' main-sequence KV: at every ``stride``-th prompt position ``p``
    draft a block from the TRUE anchor ``t[p+1]`` over the main window ``<= p`` and score the
    top-1 agreement of draft row ``i`` (with the Markov bias of the true previous token, and
    without) against the true continuation ``t[p+2+i]``. A miswired drafter scores ~0% at
    row 0; a correct one decays from well above 50%. Port of the 0731 harness.

    Under CED prefill the stages' main KV exists only for the replayed tail (the last
    ``window`` prompt positions), so both the scored positions and each draft's window picks
    are restricted to that tail: positions start at ``max(min_ctx, T - window)`` and a pick
    never reaches below ``T - window`` (in ``full`` mode the whole prompt is available).
    Returns the per-row agreement lists and the positions scored."""
    from freetoken.utils import init_logger

    logger = init_logger(__name__)
    stages = list(transformer.mtp)
    last = stages[-1]
    block, noise = last.block_size, last.noise_token_id
    attn0 = stages[0].attn
    backend = attn0.attn
    win = attn0.window_size
    ids_all = input_ids.view(-1).to(torch.int64)
    device = ids_all.device
    T = ids_all.numel()
    floor = max(0, T - win) if getattr(transformer, "prefill_mode", "full") == "ced" else 0
    positions = list(range(max(min_ctx, floor), T - block - 1, stride))[:max_positions]
    if not positions:
        logger.info_rank0(f"DSpark harness: prompt too short ({T} tokens), nothing to score")
        return {"with_bias": [], "without_bias": [], "positions": []}
    hits_bias = torch.zeros(block, device=device)
    hits_plain = torch.zeros(block, device=device)
    block_kv = [
        torch.empty(block, attn0.head_dim, dtype=torch.bfloat16, device=device) for _ in stages
    ]
    for p in positions:
        ids = draft_block_ids(ids_all[p + 1 : p + 2].to(torch.int32), block, noise)  # [1, block]
        freqs = attn0.freqs_cis[p + 1 : p + 1 + block]
        lo = max(floor, p - win + 1)
        slots = backend.window_slots_of(ti, lo, p + 1)
        topk = torch.full((1, block, win), -1, dtype=torch.int32, device=device)
        topk[..., : slots.numel()] = slots.to(torch.int32)
        _hidden, logits = draft_block_forward(transformer, ids, freqs, topk, block_kv)
        prev = ids_all[p + 1 : p + 1 + block]  # teacher-forced previous token per row
        bias, _ = last.markov_head(prev)
        target = ids_all[p + 2 : p + 2 + block]
        hits_bias += (logits + bias).argmax(-1).eq(target).float()
        hits_plain += logits.argmax(-1).eq(target).float()
    n = len(positions)
    with_bias = (hits_bias / n).tolist()
    without = (hits_plain / n).tolist()
    per_row = ", ".join(
        f"row {i + 1}: {b:.1%} / {w:.1%}" for i, (b, w) in enumerate(zip(with_bias, without))
    )
    tail = f" (CED: last {T - floor} prompt positions)" if floor else ""
    logger.info_rank0(
        f"DSpark harness: {n} positions of a {T}-token prompt{tail}, teacher-forced top-1 "
        f"agreement (with Markov bias / without): {per_row}"
    )
    return {"with_bias": with_bias, "without_bias": without, "positions": positions}


__all__ = [
    "DSparkAttention",
    "DSparkBlock",
    "DSparkConfidenceHead",
    "DSparkMarkovHead",
    "draft_block_forward",
    "draft_block_ids",
    "teacher_forced_report",
]
