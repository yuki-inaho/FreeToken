from __future__ import annotations

import torch


def gdn_prefill_chunk_fla(
    q: torch.Tensor,        # [1, total, num_k_heads, head_k_dim] bf16 (NOT GQA-expanded)
    k: torch.Tensor,        # [1, total, num_k_heads, head_k_dim] bf16
    v: torch.Tensor,        # [1, total, num_v_heads, head_v_dim] bf16
    g: torch.Tensor,        # [1, total, num_v_heads] log-decay (<=0), fp32
    beta: torch.Tensor,     # [1, total, num_v_heads] fp32
    *,
    state_source: torch.Tensor,  # [num_slots, num_v_heads, head_k_dim, head_v_dim] fp32 (in place)
    indices: torch.Tensor,       # [num_seqs] slot id per sequence
    cu_seqlens: torch.Tensor,    # [num_seqs+1] int64
    scale: float,
    return_h: bool = False,
) -> torch.Tensor:
    """Chunked gated-delta-rule prefill via the vendored fla kernel. GQA is handled
    in-kernel (q/k at num_k_heads), q/k l2norm is done in-kernel, and the per-sequence
    recurrent state is read from and written back to ``state_source[indices]`` IN PLACE
    (no external l2norm, no Python stack of initial states, no copy_ writeback loop).
    Fresh sequences must have their ``state_source`` slot pre-zeroed by the caller.
    Returns ``o`` of shape ``[total, num_v_heads, head_v_dim]`` (bf16).

    When ``return_h=True`` also returns the per-chunk hidden-state buffer ``h`` of shape
    ``[1, NT_total, num_v_heads, head_v_dim, head_k_dim]`` (bf16). ``h[0, boh_i + c]`` is the
    recurrent state after ``c*64`` tokens of packed sequence ``i`` (chunk granularity 64), where
    ``boh_i = prepare_chunk_offsets(cu_seqlens, 64)[i]``. Note the last two dims are ``[V, K]`` --
    transposed vs ``state_source``'s ``[K, V]``. Used by the hybrid-radix track-checkpoint path."""
    from freetoken.kernel.fla import chunk_gated_delta_rule

    o, _, h = chunk_gated_delta_rule(
        q=q, k=k, v=v, g=g, beta=beta, scale=scale,
        initial_state=state_source, initial_state_indices=indices.to(torch.int32),
        cu_seqlens=cu_seqlens.to(torch.int64), head_first=False,
        use_qk_l2norm_in_kernel=True,
    )
    if return_h:
        return o[0], h  # h: [1, NT_total, num_v_heads, head_v_dim, head_k_dim]
    return o[0]  # [total, num_v_heads, head_v_dim]


def gdn_decode_fla(
    q: torch.Tensor,        # [1, B, num_k_heads, head_k_dim] bf16 (NOT GQA-expanded)
    k: torch.Tensor,        # [1, B, num_k_heads, head_k_dim] bf16
    v: torch.Tensor,        # [1, B, num_v_heads, head_v_dim] bf16
    a: torch.Tensor,        # [B, num_v_heads] raw
    b: torch.Tensor,        # [B, num_v_heads] raw
    *,
    A_log: torch.Tensor,        # [num_v_heads]
    dt_bias: torch.Tensor,      # [num_v_heads]
    state_source: torch.Tensor,  # [num_slots, num_v_heads, head_k_dim, head_v_dim] fp32 (in place)
    indices: torch.Tensor,      # [B] int32 slot id per request
    cu_seqlens: torch.Tensor,   # [B+1] query indptr (arange) from FLAMetadata
    scale: float,
) -> torch.Tensor:
    """Fused sigmoid-gating gated-delta-rule decode (vendored fla triton kernel): gating +
    in-kernel l2norm + recurrent update + state read/write-by-index in one kernel, with no
    external gating or gather/scatter/clone glue. Returns [B, num_v, V]."""
    from freetoken.kernel.fla import fused_sigmoid_gating_delta_rule_update

    o = fused_sigmoid_gating_delta_rule_update(
        A_log=A_log, a=a, dt_bias=dt_bias,  # already fp32 (stored fp32)
        softplus_beta=1.0, softplus_threshold=20.0,
        q=q, k=k, v=v, b=b,
        initial_state_source=state_source,
        initial_state_indices=indices,  # already int32 (built int32 in the scheduler)
        scale=scale, use_qk_l2norm_in_kernel=True, cu_seqlens=cu_seqlens,
    )
    # kernel returns o = [NK, *v.shape] then squeeze(NK) -> [1, B, num_v, V].
    # o[0] -> [B, num_v, V] (all B decode tokens; o[0,0] would drop B>1).
    return o[0]


def gdn_verify_fla(
    q: torch.Tensor,        # [1, total, num_k_heads, head_k_dim] bf16 (NOT GQA-expanded)
    k: torch.Tensor,        # [1, total, num_k_heads, head_k_dim] bf16
    v: torch.Tensor,        # [1, total, num_v_heads, head_v_dim] bf16
    a: torch.Tensor,        # [total, num_v_heads] raw
    b: torch.Tensor,        # [total, num_v_heads] raw
    *,
    A_log: torch.Tensor,        # [num_v_heads]
    dt_bias: torch.Tensor,      # [num_v_heads]
    state_source: torch.Tensor,  # [num_slots, num_v_heads, head_k_dim, head_v_dim] fp32
    indices: torch.Tensor,      # [num_seqs] int32 slot id per request
    cu_seqlens: torch.Tensor,   # [num_seqs+1] ragged indptr (multi-token per sequence)
    intermediate_states: torch.Tensor,  # [num_seqs, max_steps, num_v_heads, K, V] fp32
    intermediate_indices: torch.Tensor,  # [num_seqs] int32 row per sequence
    scale: float,
) -> torch.Tensor:
    """Speculative-verify variant of :func:`gdn_decode_fla`: run each sequence's k+1 tokens
    through the same fused recurrent kernel WITHOUT touching the live state slot, and record
    the post-token recurrent state of every step into ``intermediate_states``.

    After acceptance the caller commits ``intermediate_states[row, n_accepted-1]`` into the
    live pool slot (a plain copy: the buffer's trailing dims deliberately mirror the pool's
    ``[K, V]``-declared/``[V, K]``-addressed layout, which coincide because K == V -- see the
    transposition note in qwen4_exp/gdn.py). Returns [total, num_v, V].
    """
    from freetoken.kernel.fla import fused_sigmoid_gating_delta_rule_update

    assert intermediate_states.dtype == state_source.dtype, "buffer must match the pool dtype"
    # The fused kernel's pointer walk assumes unit stride on the trailing dim; a reshape of
    # the transposed varlen-conv output can legally be a strided VIEW (e.g. token stride 1,
    # feature stride T), which it would silently misread. Same guard the chunk path gets
    # from its own wrapper.
    if q.stride(-1) != 1:
        q = q.contiguous()
    if k.stride(-1) != 1:
        k = k.contiguous()
    if v.stride(-1) != 1:
        v = v.contiguous()
    if a.stride(-1) != 1:
        a = a.contiguous()
    if b.stride(-1) != 1:
        b = b.contiguous()
    o = fused_sigmoid_gating_delta_rule_update(
        A_log=A_log, a=a, dt_bias=dt_bias,
        softplus_beta=1.0, softplus_threshold=20.0,
        q=q, k=k, v=v, b=b,
        initial_state_source=state_source,
        initial_state_indices=indices,
        scale=scale, use_qk_l2norm_in_kernel=True, cu_seqlens=cu_seqlens,
        disable_state_update=True,
        intermediate_states_buffer=intermediate_states,
        intermediate_state_indices=intermediate_indices,
    )
    return o[0]


__all__ = ["gdn_prefill_chunk_fla", "gdn_decode_fla", "gdn_verify_fla"]
