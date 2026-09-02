"""DSpark: DeepSeek-V4-Flash-0731's block speculative drafter (the checkpoint's ``mtp.*``).

Reference: the checkpoint's ``inference/model.py`` (``DSparkBlock`` / ``DSparkAttention`` /
``DSparkMarkovHead`` / ``DSparkConfidenceHead``) and arXiv 2607.05147. ``n_mtp_layers``
full DSV4 blocks -- MLA attention with ``compress_ratio == 0`` (window-only), the 4-stream
hyper-connections, and a MoE with their OWN 256 routed FP4 experts + shared expert -- draft
a block of ``dspark_block_size`` tokens in ONE forward:

* **Context**: the target's hidden states after ``dspark_target_layer_ids`` (mean over the
  HC streams, concatenated) go through stage 0's ``main_proj`` + ``main_norm`` -> ``main_x``
  [T, dim]. Every stage turns ``main_x`` into a window-only KV over the MAIN sequence
  (``kv_norm(wkv(main_x))``, rope at the main token's position, fp8 act-quant) -- in
  FreeToken that KV lives in the paged window tier as extra layers ``n_layers + stage``
  (same window slots as the main layers, so SWA-radix retention covers it).
* **Block**: the anchor token (the step's newest sampled token) followed by
  ``dspark_block_size - 1`` ``dspark_noise_token_id`` rows, embedded with the shared
  ``embed`` and replicated to the HC streams. Each block row attends the main window
  (positions <= the anchor's predecessor) plus every block row (bidirectional).
* **Head** (last stage): ``hc_head -> norm -> shared lm head`` gives the block's base logits;
  the rank-``dspark_markov_rank`` Markov head adds a transition bias from the previous
  (sampled) token so the block is semi-autoregressive; the confidence head scores each
  position's chance of surviving verification.

The sampling loop (Markov bias -> renorm -> draw, per position) and the confidence
scheduling live in the engine's speculative executor; this module holds the weights and the
per-stage math.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from freetoken.kernel.triton.dsv4.fp8_linear import act_quant_fp8_inplace
from freetoken.kernel.triton.dsv4.norm import rms_norm

from .args import DeepseekV4Args
from .attention import Attention
from .layers import Linear, RMSNorm, hc_head_combine
from .model import Block
from .ops import apply_rotary_emb


class DSparkAttention(Attention):
    """Window-only MLA attention for a DSpark stage: keys are the main-sequence KV written
    from ``main_x`` (paged window tier, layer ``self.layer_id``) plus the block's own rows."""

    def __init__(self, layer_id: int, args: DeepseekV4Args):
        super().__init__(layer_id, args)
        assert self.compress_ratio == 0, "DSpark stages are window-only (compress_ratio 0)"

    def main_kv(self, main_x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        """Main-sequence KV rows for this stage: ``main_x`` ``[T, dim]`` at per-row rope
        ``freqs`` ``[T, rd//2]`` -> ``[T, head_dim]`` (fp8-act-quantized nope part, bf16 storage),
        ready for ``store_window``."""
        rd = self.rope_head_dim
        kv = self.kv_norm(self.wkv(main_x.unsqueeze(0)))  # [1, T, head_dim]
        apply_rotary_emb(kv[..., -rd:], freqs)
        act_quant_fp8_inplace(kv[..., :-rd], 64)
        return kv[0]

    def draft(
        self, x: torch.Tensor, freqs: torch.Tensor, window_topk: torch.Tensor,
        block_kv: torch.Tensor,
    ) -> torch.Tensor:
        """Attention for the draft block. ``x`` ``[1, m, dim]`` (post-norm block rows),
        ``freqs`` ``[m, rd//2]`` (the rows' positions), ``window_topk`` ``[1, m, n_window]``
        GLOBAL window-pool slots (-1 masked; positions <= the anchor's predecessor), and
        ``block_kv`` ``[m, head_dim]`` bf16 scratch this call fills with the rows' own KV --
        the kernel reads it as the "compressed" pool half, so every row attends every block
        row (bidirectional within the block, as in the reference)."""
        rd = self.rope_head_dim
        m = x.size(1)
        q = self.q_norm(self.wq_a(x))
        q = self.wq_b(q).unflatten(-1, (self.n_heads, self.head_dim))
        q = rms_norm(q, None, self.eps)
        apply_rotary_emb(q[..., -rd:], freqs)

        kv = self.kv_norm(self.wkv(x))
        apply_rotary_emb(kv[..., -rd:], freqs)
        act_quant_fp8_inplace(kv[..., :-rd], 64)
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
    """Low-rank token-transition bias: ``bias(prev) = W2 · W1[prev]`` over the vocab."""

    def __init__(self, vocab_size: int, rank: int):
        super().__init__()
        self.markov_w1 = nn.Embedding(vocab_size, rank, dtype=torch.bfloat16)
        self.markov_w1.weight.requires_grad_(False)
        self.markov_w2 = Linear(rank, vocab_size, kind="bf16")

    def forward(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``token_ids`` ``[B]`` -> (``logits_bias`` ``[B, vocab]`` fp32, ``embed`` ``[B, rank]``)."""
        embed = self.markov_w1.weight.index_select(0, token_ids.to(torch.int64))
        bias = F.linear(embed, self.markov_w2.weight)
        return bias.float(), embed


class DSparkConfidenceHead(nn.Module):
    """Scores a draft position's chance of surviving verification from its backbone hidden
    and the previous token's Markov embedding (a sigmoid over one fp32 projection)."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.proj = Linear(input_dim, 1, kind="fp32")

    def forward(self, hidden: torch.Tensor, markov_embed: torch.Tensor) -> torch.Tensor:
        """``hidden`` ``[.., dim]``, ``markov_embed`` ``[.., rank]`` -> confidence LOGITS ``[..]``
        (apply ``sigmoid`` for the survival probability)."""
        x = torch.cat([hidden.float(), markov_embed.float()], dim=-1)
        return self.proj(x).squeeze(-1)


class DSparkBlock(Block):
    """One DSpark stage (checkpoint ``mtp.{stage_id}.*``): a DSV4 block whose attention keys
    the main-sequence draft KV, plus stage 0's context projection and the last stage's head."""

    def __init__(self, layer_id: int, args: DeepseekV4Args, stage_id: int):
        super().__init__(layer_id, args, attention_cls=DSparkAttention)
        self.stage_id = stage_id
        self.block_size = args.dspark_block_size
        self.noise_token_id = args.dspark_noise_token_id
        self.n_targets = len(args.dspark_target_layer_ids)
        self.is_first = stage_id == 0
        self.is_last = stage_id == args.n_mtp_layers - 1
        if self.is_first:
            self.main_proj = Linear(args.dim * self.n_targets, args.dim, kind="fp8")
            self.main_norm = RMSNorm(args.dim, args.norm_eps)
        if self.is_last:
            self.norm = RMSNorm(args.dim, args.norm_eps)
            self.markov_head = DSparkMarkovHead(args.vocab_size, args.dspark_markov_rank)
            self.confidence_head = DSparkConfidenceHead(args.dim + args.dspark_markov_rank)
            hc_dim = self.hc_mult * args.dim
            self.hc_head_fn = nn.Parameter(
                torch.empty(self.hc_mult, hc_dim, dtype=torch.float32), requires_grad=False
            )
            self.hc_head_base = nn.Parameter(
                torch.empty(self.hc_mult, dtype=torch.float32), requires_grad=False
            )
            self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32), requires_grad=False)

    # ----- stage-0 context -----
    def main_x(self, main_hidden: torch.Tensor) -> torch.Tensor:
        """``main_hidden`` ``[T, n_targets * dim]`` (concatenated HC-stream means after the target
        layers) -> ``main_x`` ``[T, dim]``, the context every stage keys its draft KV from."""
        assert self.is_first, "main_x lives on stage 0"
        return self.main_norm(self.main_proj(main_hidden))

    # ----- per-stage block forward -----
    def draft_step(
        self, x: torch.Tensor, input_ids: torch.Tensor, freqs: torch.Tensor,
        window_topk: torch.Tensor, block_kv: torch.Tensor,
    ) -> torch.Tensor:
        """``x`` ``[1, m, hc_mult, dim]`` block rows (HC residual streams) -> same shape.
        ``input_ids`` ``[1, m]`` (the block's token ids; the router is bias-routed on these
        layers so they only feed the generic MoE signature)."""
        residual = x
        x, post, comb = self.hc_pre(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        x = self.attn_norm(x)
        x = self.attn.draft(x, freqs, window_topk, block_kv)
        x = self.hc_post(x, residual, post, comb)

        residual = x
        x, post, comb = self.hc_pre(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        x = self.ffn_norm(x)
        x = self.ffn(x, input_ids)
        x = self.hc_post(x, residual, post, comb)
        return x

    # ----- last-stage head -----
    def head_hidden(self, x: torch.Tensor) -> torch.Tensor:
        """Collapse the HC streams for the head: ``[1, m, hc_mult, dim]`` -> ``[1, m, dim]``
        (pre-norm; the confidence head reads this, the lm head reads ``self.norm`` of it)."""
        assert self.is_last, "the head lives on the last stage"
        return hc_head_combine(
            x, self.hc_head_fn, self.hc_head_scale, self.hc_head_base,
            hc_mult=self.hc_mult, dim=self.dim, norm_eps=self.norm_eps, hc_eps=self.hc_eps,
        )

    def block_logits(self, hidden: torch.Tensor, head: torch.Tensor) -> torch.Tensor:
        """``hidden`` ``[m, dim]`` (from :meth:`head_hidden`) x the shared lm ``head``
        ``[vocab, dim]`` -> base block logits ``[m, vocab]`` (before the Markov bias)."""
        assert self.is_last
        return F.linear(self.norm(hidden), head)


def draft_block_ids(anchor: torch.Tensor, block_size: int, noise_token_id: int) -> torch.Tensor:
    """The block's input ids ``[B, block_size]``: the anchor(s) ``[B]`` then noise tokens."""
    ids = anchor.new_full((anchor.numel(), block_size), noise_token_id)
    ids[:, 0] = anchor
    return ids


def draft_block_forward(
    transformer, ids: torch.Tensor, freqs: torch.Tensor, window_topk: torch.Tensor,
    block_kv: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the block ``ids`` ``[1, m]`` through every stage. ``freqs`` ``[m, rd//2]`` are the
    rows' rope positions, ``window_topk`` ``[1, m, n_window]`` the main-window slots each row
    may attend, ``block_kv`` one ``[m, head_dim]`` bf16 scratch per stage. Returns the last
    stage's pre-norm head hidden ``[m, dim]`` and the base block logits ``[m, vocab]`` fp32
    (before the Markov bias)."""
    stages = list(transformer.mtp)
    x = transformer.embed(ids.to(torch.int64)).unsqueeze(2).repeat(1, 1, transformer.hc_mult, 1)
    for stage, buf in zip(stages, block_kv):
        x = stage.draft_step(x, ids, freqs, window_topk, buf)
    last = stages[-1]
    hidden = last.head_hidden(x)[0]
    return hidden, last.block_logits(hidden, transformer.head).float()


@torch.no_grad()
def teacher_forced_report(
    transformer, input_ids: torch.Tensor, ti: int, *, stride: int = 16, min_ctx: int = 8,
    max_positions: int = 256,
) -> dict[str, list[float]]:
    """Wiring check (``FREETOKEN_DSPARK_HARNESS=1``), run right after a cold prefill wrote the
    stages' main-sequence KV: at every ``stride``-th prompt position ``p`` draft a block from
    the TRUE anchor ``t[p+1]`` over the main window ``<= p`` and score top-1 agreement of
    draft row ``i`` (with the Markov bias of the true previous token, and without) against
    the true continuation ``t[p+2+i]``. A miswired drafter scores ~0% at row 0; a correct
    one decays from well above 50%."""
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
    positions = list(range(min_ctx, T - block - 1, stride))[:max_positions]
    if not positions:
        logger.info_rank0(f"DSpark harness: prompt too short ({T} tokens), nothing to score")
        return {"with_bias": [], "without_bias": []}
    hits_bias = torch.zeros(block, device=device)
    hits_plain = torch.zeros(block, device=device)
    block_kv = [
        torch.empty(block, attn0.head_dim, dtype=torch.bfloat16, device=device) for _ in stages
    ]
    for p in positions:
        ids = draft_block_ids(ids_all[p + 1 : p + 2].to(torch.int32), block, noise)  # [1, block]
        freqs = attn0.freqs_cis[p + 1 : p + 1 + block]
        lo = max(0, p - win + 1)
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
    logger.info_rank0(
        f"DSpark harness: {n} positions of a {T}-token prompt, teacher-forced top-1 agreement "
        f"(with Markov bias / without): {per_row}"
    )
    return {"with_bias": with_bias, "without_bias": without}


__all__ = [
    "DSparkAttention",
    "DSparkBlock",
    "DSparkConfidenceHead",
    "DSparkMarkovHead",
    "draft_block_forward",
    "draft_block_ids",
    "teacher_forced_report",
]
