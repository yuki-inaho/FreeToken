"""Parity harness: drive the FreeToken ``deepseek_v41`` port on the toy geometry against the
pure-torch oracle (``dsv41_reference.py``) without an engine.

What it provides (see ``test_dsv41_parity.py`` for the tests built on it):

* ``build_port`` -- the port ``Transformer`` on a device, loaded from the ORACLE's random
  checkpoint through the adapter's naming (``head.weight`` -> ``head``, ``attn.wo_a.{weight,
  scale}`` -> bf16 ``attn.wo_a`` via ``weight._dequant_fp8_block``, dtypes cast like
  ``DeepseekV41ForCausalLM.load_state_dict``), with every routed-expert layer's
  ``routed_forward`` REPLACED by a dense torch reference over the oracle's packed FP4 experts
  (the oracle's own ``Expert`` math: FP8 activation quant, ``fp4_gemm``, SwiGLU clamps, routed
  weights). The offload expert cache is covered by ``tests/moe``; here the parity of attention /
  CSA2 / mHC / Engram / DSpark must not depend on it.
* ``make_ctx`` -- a fake engine context with a REAL paged pool + sparse backend (the 0731 fixture
  pattern from ``tests/dsv4/test_dsv4_attn_backend.py``), one request per page-table row with
  identity full->window page binding.
* ``prefill`` / ``prefill_via_backend`` / ``decode_step`` -- build the ragged segments,
  positions and decode ``Batch``/``DSV4AttnMetadata`` the engine would, and return logits.
* ``engram_feats`` -- the port's Engram input (bf16 ``[T, n_hash_cols*head_dim]``) dequantized
  from the oracle's table rows for the same hash ids the oracle consumes.
* ``draft_block`` / ``window_topk`` -- the DSpark draft addressing of
  ``engine/spec_dspark.py`` (``_draft_block`` / ``_window_topk``) for a teacher-forced check.

Device knob: ``FREETOKEN_TEST_DEVICE`` (default ``cuda``). ``cpu`` runs the Triton kernels under
the interpreter and needs ``TRITON_INTERPRET=1 FREETOKEN_FORCE_E4M3_EMU=1`` set BEFORE the
process starts (the e4m3 flag is read once at import). Interpreter caveats handled here:

* ``tl.dot`` on bf16 operands multiplies the raw uint16 storage in the interpreter (triton 3.8);
  ``_patch_interpreter_bf16_dot`` upcasts them to fp32 first (same fix as the kernels agent's
  scratch runner). The fp8 GEMM, indexer and sparse-attention kernels all hit this path.
* the interpreter's fp32 -> bf16 cast TRUNCATES (no RNE): every kernel's bf16 store was biased
  by up to 2^-8; ``_patch_interpreter_bf16_cast`` routes it through torch's RNE. Without it the
  toy logits sat ~50% off the oracle (layer 0 alone 1e-2); with it layer 0 is bit-exact.
* the decode split-K FP8 GEMV is ~5k interpreter programs per step (60 s); on the interpreter
  it is replaced by an exact torch equivalent (``FREETOKEN_DSV41_PARITY_REAL_GEMV=1`` keeps
  the kernel) -- the kernel is covered by ``tests/kernels/test_dsv4_fp8_linear_block32``.
* Interpreter runs are slow (~0.5 s per 200-row GEMM, ~10 s per decode step): the whole
  parity file takes ~15 min on CPU. Triton 3.6's interpreter (the WSL venv) fails on these
  kernels with a scalar-conversion TypeError; the Windows venv's triton 3.8 works.
"""

from __future__ import annotations

import contextlib
import os
from types import SimpleNamespace

import torch

from .dsv41_reference import Expert, ModelArgs, Numerics, unpack_e2m1  # noqa: F401

DEV = os.environ.get("FREETOKEN_TEST_DEVICE", "cuda")
INTERPRET = os.environ.get("TRITON_INTERPRET", "0") == "1"
P = 128


def skip_reason() -> str | None:
    """None when the harness can run here, else the reason to skip (names the blocker)."""
    if DEV == "cuda":
        if not torch.cuda.is_available():
            return "needs CUDA (or FREETOKEN_TEST_DEVICE=cpu with TRITON_INTERPRET=1)"
        return None
    if DEV == "cpu":
        if not INTERPRET:
            return "FREETOKEN_TEST_DEVICE=cpu needs TRITON_INTERPRET=1 (Triton CPU interpreter)"
        if os.environ.get("FREETOKEN_FORCE_E4M3_EMU", "").lower() not in ("1", "true", "yes", "on"):
            return "the interpreter has no fp8 type: set FREETOKEN_FORCE_E4M3_EMU=1"
        return None
    return f"unknown FREETOKEN_TEST_DEVICE={DEV!r}"


def device() -> torch.device:
    return torch.device(DEV)


# --------------------------------------------------------------------------------------------
# Interpreter patches (CPU only)
# --------------------------------------------------------------------------------------------
_PATCHED = False


def _patch_interpreter_bf16_dot() -> None:
    import numpy as np
    import triton.language as tl
    from triton.runtime import interpreter as I

    if getattr(I.InterpreterBuilder.create_dot, "_dsv41_bf16_patch", False):
        return
    _orig = I.InterpreterBuilder.create_dot

    def create_dot(self, a, b, d, input_precision, max_num_imprecise_acc):
        def as_f32(h):
            if h.dtype == tl.bfloat16:
                return (h.data.astype(np.uint32) << 16).view(np.float32)
            return None

        a32, b32 = as_f32(a), as_f32(b)
        if a32 is None and b32 is None:
            return _orig(self, a, b, d, input_precision, max_num_imprecise_acc)
        a_data = a32 if a32 is not None else a.data
        b_data = b32 if b32 is not None else b.data
        return I.TensorHandle(np.matmul(a_data, b_data, dtype=d.data.dtype) + d.data, d.dtype.scalar)

    create_dot._dsv41_bf16_patch = True
    I.InterpreterBuilder.create_dot = create_dot


def _patch_interpreter_bf16_cast() -> None:
    """The interpreter's fp32 -> bf16 cast TRUNCATES the mantissa (``_convert_float`` with
    ``rounding_mode=None``; probe: ``1 + 3*2^-9 -> 1.0``, torch RNE gives ``1.0078125``), so
    every kernel's bf16 store carried a one-sided error of up to 2^-8 relative -- across 8
    layers that alone put the toy logits ~50% off the oracle. Route the two bf16 casts
    through torch's round-to-nearest-even (what the GPU does)."""
    import numpy as np
    import triton.language as tl
    from triton.runtime import interpreter as I

    if getattr(I.InterpreterBuilder.cast_impl, "_dsv41_bf16_patch", False):
        return
    _orig_cast = I.InterpreterBuilder.cast_impl
    _orig_fp_to_fp = I.InterpreterBuilder.create_fp_to_fp

    def _bf16_pair(src, dst_type):
        s, d = src.dtype.scalar, dst_type.scalar
        if s == tl.float32 and d == tl.bfloat16:
            data = np.ascontiguousarray(src.data, dtype=np.float32)
            out = torch.from_numpy(data.reshape(-1)).to(torch.bfloat16).view(torch.int16).numpy()
            return I.TensorHandle(out.view(np.uint16).reshape(data.shape), d)
        if s == tl.bfloat16 and d == tl.float32:
            return I.TensorHandle((src.data.astype(np.uint32) << 16).view(np.float32), d)
        return None

    def cast_impl(self, src, dst_type):
        r = _bf16_pair(src, dst_type)
        return r if r is not None else _orig_cast(self, src, dst_type)

    def create_fp_to_fp(self, src, dst_type, rounding_mode):
        r = _bf16_pair(src, dst_type)
        return r if r is not None else _orig_fp_to_fp(self, src, dst_type, rounding_mode)

    cast_impl._dsv41_bf16_patch = True
    I.InterpreterBuilder.cast_impl = cast_impl
    I.InterpreterBuilder.create_fp_to_fp = create_fp_to_fp


def _patch_interpreter_fp8_gemv() -> None:
    """Interpreter only: the decode split-K FP8 GEMV launches ~5k programs per decode step
    (12 ms each under the interpreter: 60+ s per step). Substitute the same block-scaled math
    in torch (exact power-of-two scaling, fp32 accumulation; differs from the kernel only by
    reassociation). The kernel itself is covered by ``tests/kernels/test_dsv4_fp8_linear_block32``
    (M == 1 cases) under the interpreter and on the GPU. ``FREETOKEN_DSV41_PARITY_REAL_GEMV=1``
    keeps the kernel."""
    if os.environ.get("FREETOKEN_DSV41_PARITY_REAL_GEMV", "0") == "1":
        return
    from freetoken.kernel.triton.dsv4 import fp8_linear as fl

    if getattr(fl._fp8_act_gemv, "_dsv41_patch", False):
        return

    def _fp8_act_gemv(a_fp8, sa, weight, sb, out_dtype, block):
        N, K = weight.shape
        a = a_fp8.view(torch.float8_e4m3fn).float() if a_fp8.dtype == torch.uint8 else a_fp8.float()
        w = weight.view(torch.float8_e4m3fn).float() if weight.dtype == torch.uint8 else weight.float()
        sa_f = torch.exp2(sa.view(torch.uint8).float() - 127.0)  # [K//block]
        sb_f = torch.exp2(sb.view(torch.uint8).float() - 127.0)  # [ceil(N/block), K//block]
        a_scaled = (a.view(-1, block) * sa_f[:, None]).reshape(K)
        sb_rows = sb_f.repeat_interleave(block, dim=0)[:N]  # [N, K//block]
        w_scaled = (w.view(N, K // block, block) * sb_rows[:, :, None]).reshape(N, K)
        return (w_scaled @ a_scaled).to(out_dtype)

    _fp8_act_gemv._dsv41_patch = True
    fl._fp8_act_gemv = _fp8_act_gemv


def apply_runtime_patches() -> None:
    """Idempotent: the interpreter fixes (bf16 dot / bf16 RNE cast / torch GEMV) and the CPU
    ``split_count`` stub."""
    global _PATCHED
    if _PATCHED:
        return
    if INTERPRET:
        _patch_interpreter_bf16_dot()
        _patch_interpreter_bf16_cast()
        _patch_interpreter_fp8_gemv()
    from freetoken.distributed import set_tp_info, try_get_tp_info

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    _PATCHED = True


# --------------------------------------------------------------------------------------------
# Toy geometry (the plan's, with index_n_heads raised to 16: with h index heads a fraction
# 2^-h of the indexer scores is exactly 0 (relu), and the oracle breaks those ties toward the
# lower index while the port's torch.topk order is unspecified -- at 16 heads ties are gone;
# 16 is also the smallest head count the GPU ``tl.dot`` tile of the indexer kernel accepts.)
# --------------------------------------------------------------------------------------------
TOY = dict(
    max_batch_size=1,
    max_seq_len=512,
    vocab_size=64,
    dim=256,
    moe_inter_dim=128,
    n_layers=8,
    n_mtp_layers=3,
    n_heads=2,
    n_routed_experts=8,
    n_activated_experts=2,
    score_func="sqrtsoftplus",
    route_scale=1.5,
    swiglu_limit=10.0,
    q_lora_rank=128,
    head_dim=128,
    rope_head_dim=32,
    norm_eps=1e-20,
    o_groups=2,
    o_lora_rank=64,
    window_size=P,
    compress_ratios=(0, 0, 2, 2, 1, 1, 1, 1, 0, 0, 0),
    kv_source_layers=(2, 4),
    index_source_layers=(2, 4, 6),
    compress_rope_theta=160000.0,
    original_seq_len=256,
    rope_theta=10000.0,
    rope_factor=16,
    beta_fast=32,
    beta_slow=1,
    index_n_heads=16,
    index_head_dim=64,
    index_topk=16,
    candidate_source_layer=4,
    candidate_topk_blocks=4,
    candidate_block_size=8,
    hc_mult=4,
    hc_sinkhorn_iters=20,
    hc_eps=1e-6,
    engram_layer_ids=(1,),
    engram_num_embeddings=(1000,),
    engram_max_ngram_size=4,
    engram_vocab_size=1000,
    engram_n_heads=8,
    engram_head_dim=32,
    engram_compressed_vocab_size=64,
    dspark_block_size=5,
    dspark_noise_token_id=1,
    dspark_target_layer_ids=(5, 6, 7),
    dspark_markov_rank=32,
    dspark_n_routed_experts=4,
    dspark_n_activated_experts=1,
)


def oracle_args(**over) -> ModelArgs:
    return ModelArgs(**{**TOY, **over})


def port_args(**over):
    from freetoken.models.deepseek_v41.args import DeepseekV41Args

    kw = {**TOY, **over}
    # The DSpark stages ride the pool as extra WINDOW-tier layers (what
    # ``extend_config_for_speculative`` sets); always sized so one pool serves either build.
    kw.setdefault("n_extra_window_layers", kw["n_mtp_layers"])
    return DeepseekV41Args(**kw)


def make_inputs(args: ModelArgs, length: int, seed: int = 7, bsz: int = 1):
    """Token ids ``[B, L]`` and Engram hash ids ``[B, L, n_engram_layers, n_hash_cols]``."""
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, args.vocab_size, (bsz, length), generator=g)
    n_eng = len(args.engram_layer_ids)
    rows = args.engram_num_embeddings[0] if n_eng else 1
    hash_ids = torch.randint(0, rows, (bsz, length, n_eng, args.engram_n_hash_cols), generator=g)
    return ids, hash_ids


def rel_max_err(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().float().cpu()
    b = b.detach().float().cpu()
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-6)).item()


def argmax_agreement(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().float().cpu().reshape(-1, a.shape[-1])
    b = b.detach().float().cpu().reshape(-1, b.shape[-1])
    return (a.argmax(-1) == b.argmax(-1)).float().mean().item()


# --------------------------------------------------------------------------------------------
# Engine context: real paged pool + sparse backend, identity page binding per request row
# --------------------------------------------------------------------------------------------
@contextlib.contextmanager
def _env(name: str, value: str | None):
    old = os.environ.get(name)
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = old


def make_ctx(args, device: torch.device, num_pages: int | None = None, max_running_req: int = 1,
             n_scratch: int | None = None):
    """``(pool, backend, ctx)`` -- resets the global context. Request row ``ti`` owns full pages
    ``[ti * pages_per_req, (ti + 1) * pages_per_req)`` (``pages_per_req = max_seq_len // P``);
    ``bind_request`` maps its positions; the last row is the engine's dummy page."""
    import freetoken.core as core
    from freetoken.attention.dsv4_sparse import DSV4SparseAttnBackend
    from freetoken.core import Context, set_global_ctx
    from freetoken.kvcache.dsv4_cost_model import dsv4_pool_sizes
    from freetoken.kvcache.dsv4_paged_pool import DSV4PagedKVCache

    apply_runtime_patches()
    assert args.max_seq_len % P == 0
    pages_per_req = args.max_seq_len // P
    if num_pages is None:
        num_pages = max_running_req * pages_per_req
    assert num_pages >= max_running_req * pages_per_req
    core._GLOBAL_CTX = None
    ctx = Context(page_size=P)
    set_global_ctx(ctx)
    sizes = dsv4_pool_sizes(num_pages=num_pages + 1, args=args, swa_ratio=1.0, P=P)
    if n_scratch is None:
        n_scratch = max_running_req + int(getattr(args, "dspark_block_size", 0) or 0) + 2
    pool = DSV4PagedKVCache(sizes=sizes, args=args, device=device, P=P, n_scratch=n_scratch)
    pool._init_paged_state(max_running_req, True)
    pt = torch.zeros(max_running_req + 1, args.max_seq_len, dtype=torch.int32, device=device)
    pt[max_running_req].fill_(num_pages * P)  # engine's dummy-row convention
    pool.full_loc_map = pt
    ctx.kv_cache = pool
    backend = DSV4SparseAttnBackend(SimpleNamespace(dsv4_args=args))
    ctx.attn_backend = backend
    ctx._dsv41_pages_per_req = pages_per_req
    return pool, backend, ctx


def bind_request(ctx, table_idx: int) -> None:
    """Map every position of request row ``table_idx`` (identity full->window pages)."""
    pool = ctx.kv_cache
    pages_per_req = ctx._dsv41_pages_per_req
    base = table_idx * pages_per_req * P
    n = pages_per_req * P
    pool.full_loc_map[table_idx, :n] = torch.arange(base, base + n, dtype=torch.int32, device=pool.device)
    for k in range(pages_per_req):
        pool.bind_window_pages(base + k * P, base + k * P)


# --------------------------------------------------------------------------------------------
# The port, loaded from the oracle's checkpoint, with dense reference routed experts
# --------------------------------------------------------------------------------------------
class DenseRoutedExperts:
    """``routed_forward(x, weights, ids)`` over the oracle's packed E2M1 experts with the
    oracle's own ``Expert`` math (act_quant per 32 -> ``fp4_gemm`` -> bf16, SwiGLU clamps,
    routed weight applied before ``w2``). Runs on CPU in fp32 like the oracle; returns bf16
    (the offload cache's contract), so the block's residual stays bf16."""

    def __init__(self, sd, prefix: str, n_experts: int, swiglu_limit: float, numerics: Numerics):
        self.experts = [
            Expert(sd, f"{prefix}.experts.{i}", "fp4", swiglu_limit, numerics) for i in range(n_experts)
        ]
        self.calls = 0

    def routed_forward(self, x: torch.Tensor, weights: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        dev, dtype = x.device, x.dtype
        xf = x.detach().float().cpu()
        w = weights.detach().float().cpu()
        idx = ids.detach().long().cpu()
        y = torch.zeros_like(xf)
        for i, expert in enumerate(self.experts):
            rows, top = torch.where(idx == i)
            if rows.numel() == 0:
                continue
            y[rows] += expert(xf[rows], w[rows, top, None])
        return y.to(dtype).to(dev)


def port_state_dict(state_dict: dict, args, *, with_mtp: bool) -> dict[str, torch.Tensor]:
    """The oracle checkpoint under the port's parameter names (the adapter's renames), minus the
    tensors the port does not hold (routed experts, the Engram table)."""
    from freetoken.models.deepseek_v41.weight import _dequant_fp8_block

    out: dict[str, torch.Tensor] = {}
    for name, t in state_dict.items():
        if ".ffn.experts." in name or ".engram.embed." in name:
            continue
        if name.startswith("mtp.") and not with_mtp:
            continue
        if name == "head.weight":
            out["head"] = t
        elif name.endswith(".attn.wo_a.weight"):
            base = name[: -len(".weight")]
            out[base] = _dequant_fp8_block(t, state_dict[base + ".scale"], block=None)
        elif name.endswith(".attn.wo_a.scale"):
            continue
        else:
            out[name] = t
    return out


def build_port(args, state_dict: dict, device: torch.device, prefill_mode: str, quant_act: bool = True,
               *, dspark: bool = False, pool=None, numerics: Numerics | None = None):
    """The port ``Transformer`` on ``device`` loaded from the oracle's ``state_dict``; routed
    experts replaced by :class:`DenseRoutedExperts`. ``dspark`` builds the ``mtp.*`` stages
    (``FREETOKEN_LOAD_MTP=1`` at construction). Binds to ``pool`` when given."""
    from freetoken.models.deepseek_v41.model import Transformer
    from freetoken.utils.torch_utils import torch_dtype

    apply_runtime_patches()
    # The engine constructs under ``torch.device("meta"), torch_dtype(config.dtype)`` (bf16):
    # that is what makes the dtype-less ``nn.Embedding`` bf16 (every other parameter names
    # its dtype), so the harness builds the same way.
    with _env("FREETOKEN_LOAD_MTP", "1" if dspark else "0"):
        with torch.device("meta"), torch_dtype(torch.bfloat16):
            port = Transformer(args, prefill_mode=prefill_mode, quant_act=quant_act)
    assert (len(port.mtp) > 0) == dspark, "DSpark build did not follow the flag"
    src = port_state_dict(state_dict, args, with_mtp=dspark)
    names = {n for n, _ in port.named_parameters()}
    missing = sorted(names - src.keys())
    extra = sorted(src.keys() - names)
    assert not missing and not extra, (missing, extra)
    casted = {}
    for name, p in port.named_parameters():
        t = src[name]
        assert tuple(t.shape) == tuple(p.shape), (name, tuple(t.shape), tuple(p.shape))
        casted[name] = t.to(device=device, dtype=p.dtype)
    port.load_state_dict(casted, assign=True, strict=False)
    port.eval()
    num = numerics or Numerics()
    port._dense_experts = {}
    for L, layer in enumerate(port.layers):
        n_exp, _ = args.get_moe_config(L)
        dense = DenseRoutedExperts(state_dict, f"layers.{L}.ffn", n_exp, args.swiglu_limit, num)
        layer.ffn.experts.routed_forward = dense.routed_forward
        port._dense_experts[L] = dense
    for s, stage in enumerate(port.mtp):
        n_exp, _ = args.get_moe_config(args.n_layers + s)
        dense = DenseRoutedExperts(state_dict, f"mtp.{s}.ffn", n_exp, args.swiglu_limit, num)
        stage.ffn.experts.routed_forward = dense.routed_forward
        port._dense_experts[args.n_layers + s] = dense
    if pool is not None:
        port.bind(pool, device)
    return port


# --------------------------------------------------------------------------------------------
# Engram features
# --------------------------------------------------------------------------------------------
def engram_feats_one(state_dict: dict, layer_id: int, hash_ids: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Oracle table rows for ``hash_ids`` ``[T, n_hash_cols]`` -> the port's bf16
    ``[T, n_hash_cols * head_dim]`` (fp8 code x 2^(e8m0-127) per 32-block, exact in bf16;
    identical to the oracle ``Engram.embed`` + flatten -- same column order)."""
    codes = state_dict[f"layers.{layer_id}.engram.embed.weight"].float()
    scales = state_dict[f"layers.{layer_id}.engram.embed.scale"].float()
    ids = hash_ids.long().cpu()
    vals = codes[ids].unflatten(-1, (-1, 32)) * scales[ids].unsqueeze(-1)  # [T, C, D/32, 32]
    T = ids.shape[0]
    return vals.reshape(T, -1).to(torch.bfloat16).to(device)


def engram_feats(args, state_dict: dict, hash_ids: torch.Tensor, lo: int, hi: int, device: torch.device) -> dict[int, torch.Tensor]:
    """Per Engram layer, the port features for positions ``[lo, hi)`` of ``hash_ids``
    ``[1, L, n_engram_layers, n_hash_cols]`` (the oracle's layout)."""
    feats = {}
    for i, L in enumerate(args.engram_layer_ids):
        feats[L] = engram_feats_one(state_dict, L, hash_ids[0, lo:hi, i, :], device)
    return feats


# --------------------------------------------------------------------------------------------
# Prefill / decode drivers
# --------------------------------------------------------------------------------------------
@torch.no_grad()
def prefill(port, ctx, ids: torch.Tensor, table_idx: int, start_pos: int = 0,
            engram_feats: dict[int, torch.Tensor] | None = None, all_rows: bool = False) -> torch.Tensor:
    """One-request prefill of ``ids`` ``[T]`` at positions ``start_pos ..``. Returns fp32 logits
    ``[1, V]`` (the last row) or ``[T, V]`` with ``all_rows`` (``full`` mode only: the head is
    asked for every row through ``last_indices``; CED picks its own replay row)."""
    dev = ctx.kv_cache.device
    T = int(ids.numel())
    segments = [(0, T, table_idx, start_pos)]
    flat_positions = torch.arange(start_pos, start_pos + T, device=dev)
    if all_rows:
        assert port.prefill_mode == "full", "CED prefill only yields the replay's last row"
        last = torch.arange(T, device=dev)
    else:
        last = torch.tensor([T - 1], device=dev)
    return port.prefill_batched(
        ids.view(1, -1).to(dev).long(), segments, flat_positions, last, engram_feats
    ).float()


def make_prefill_batch(ctx, all_ids: torch.Tensor, table_idx: int, cached_len: int,
                       prefill_mode: str = "ced", uid: int = 0):
    """A real prefill ``Batch`` for a request whose first ``cached_len`` tokens are cached,
    through ``backend.prepare_metadata`` (which, under CED, extends a short suffix backwards
    to a page boundary >= window before the end and rebuilds the token/position streams).
    The backend reads the prefill mode off ``FREETOKEN_DSV41_PREFILL_MODE``; it is pinned to
    ``prefill_mode`` (the port's) for the call."""
    from freetoken.core import Batch, Req, SamplingParams

    all_ids = all_ids.detach().cpu().to(torch.int32).view(-1)
    req = Req(input_ids=all_ids, table_idx=table_idx, cached_len=cached_len, output_len=1, uid=uid,
              sampling_params=SamplingParams(), cache_handle=None)
    batch = Batch(reqs=[req], phase="prefill")
    batch.padded_reqs = [req]
    batch.input_ids = all_ids[cached_len:].clone()
    batch.positions = torch.arange(cached_len, all_ids.numel(), dtype=torch.int64)
    with _env("FREETOKEN_DSV41_PREFILL_MODE", prefill_mode):
        ctx.attn_backend.prepare_metadata(batch)
    return batch


@torch.no_grad()
def prefill_via_backend(port, ctx, all_ids: torch.Tensor, table_idx: int, cached_len: int,
                        args=None, state_dict=None, hash_ids: torch.Tensor | None = None):
    """Prefill the suffix ``all_ids[cached_len:]`` the way the engine would: segments and streams
    from ``prepare_metadata`` (CED re-encode extension included), Engram rows for whatever stream
    the backend built. Returns ``(last-row logits [1, V], segments)``."""
    dev = ctx.kv_cache.device
    batch = make_prefill_batch(ctx, all_ids, table_idx, cached_len, port.prefill_mode)
    md = batch.attn_metadata
    segs = md.segments
    feats = None
    if hash_ids is not None:
        (_off, n, _ti, start), = segs
        feats = engram_feats(args, state_dict, hash_ids, start, start + n, dev)
    with ctx.forward_batch(batch):
        logits = port.prefill_batched(
            batch.input_ids.view(1, -1).to(dev).long(), segs, batch.positions.to(dev).long(),
            md.last_indices.to(dev).long(), feats,
        )
    return logits.float(), segs


def make_decode_batch(ctx, table_idx: int, pos: int, uid: int = 0):
    from freetoken.core import Batch, Req, SamplingParams

    req = Req(input_ids=torch.zeros(1, dtype=torch.int32), table_idx=table_idx, cached_len=0,
              output_len=1, uid=uid, sampling_params=SamplingParams(), cache_handle=None)
    batch = Batch(reqs=[req], phase="decode")
    batch.padded_reqs = [req]
    # The scheduler stages ``active_table_idx`` as a DEVICE tensor (the backends index the pool
    # maps with it directly); a CPU tensor only passed because the interpreter run was all-CPU.
    dev = ctx.kv_cache.device
    batch.active_table_idx = torch.tensor([table_idx], dtype=torch.int64, device=dev)
    batch.positions = torch.tensor([pos], dtype=torch.int64, device=dev)
    ctx.attn_backend.prepare_metadata(batch)
    return batch


@torch.no_grad()
def decode_step(port, ctx, token: int | torch.Tensor, pos: int, table_idx: int,
                engram_feats: dict[int, torch.Tensor] | None = None, spec=None) -> torch.Tensor:
    """One eager decode step of ``token`` at ``pos`` for request ``table_idx``: the decode
    ``Batch`` / ``DSV4AttnMetadata`` (``table_rows``, ``window_ar``) come from the backend's
    ``prepare_metadata`` exactly as in the engine; ``cmp_stage_cap = pos`` (the eager rule).
    Returns fp32 logits ``[1, V]``."""
    dev = ctx.kv_cache.device
    batch = make_decode_batch(ctx, table_idx, pos)
    tok = torch.as_tensor(token, device=dev).view(1, 1).long()
    pos_t = torch.tensor([pos], device=dev, dtype=torch.int64)
    with ctx.forward_batch(batch):
        return port.decode(tok, pos_t, cmp_stage_cap=pos, engram_feats=engram_feats, spec=spec).float()


# --------------------------------------------------------------------------------------------
# Oracle traces and TEACHER-FORCED port runs (tight, per-block parity)
#
# End-to-end logits on a random toy are chaotic: any bf16-level difference is amplified to an
# fp8 ulp (2^-3) by the next fp8-quantized linear and then flips the MoE routing (top-2 of 8),
# after which the two models compute different functions. The oracle's own prefill-vs-decode
# consistency shows the same (its bf16 bound is 2.5e-1). So the sharp test feeds every port
# block the ORACLE's block input (h, pre_mix) and compares that block's output: a wiring error
# (picks, owner, window geometry, carry, Engram gate, mHC threading) is O(1) there while the
# numerics noise stays at the bf16/fp8 level. The port's KV / carries / index keys written
# along the way come from the oracle-fed inputs, so the pools stay consistent with the oracle's.
# --------------------------------------------------------------------------------------------
class LayerTrace:
    """``picks``: for an index-source layer (Full / Reindex), the oracle's compressed picks as
    BLOCK ids ``[1, S, k]`` (``-1`` past the reach) -- the oracle's ``shared.topk_idxs`` minus its
    window offset (``kv.size(1)`` == the query count for a cold / replay call)."""

    __slots__ = ("h_pre", "h_in", "pre_in", "h_out", "pre_out", "picks", "routes")

    def __init__(self, h_pre, h_in, pre_in, h_out, pre_out, picks=None, routes=None):
        self.h_pre, self.h_in, self.pre_in, self.h_out, self.pre_out = h_pre, h_in, pre_in, h_out, pre_out
        self.picks = picks
        self.routes = routes  # the oracle's routed-expert ids ``[rows, k]`` of this block's MoE call


def oracle_routes(block) -> torch.Tensor | None:
    """The oracle's routed-expert choice ``[rows, k]`` from ``block``'s LAST MoE call (the
    reference ``Gate`` records ``last_indices``); ``None`` for a block without a gate."""
    gate = getattr(getattr(block, "ffn", None), "gate", None)
    idx = getattr(gate, "last_indices", None)
    return None if idx is None else idx.detach().clone()


def _oracle_picks(model, layer_id: int, seqlen: int):
    if model.args.mode(layer_id) not in ("full", "reindex"):
        return None
    idx = model.shared.topk_idxs
    if idx is None or idx.shape[-1] == 0:
        return None
    return torch.where(idx >= 0, idx - seqlen, idx).to(torch.int32).clone()


class OracleTrace:
    """Per-layer ``(h_pre [1,S,hc,dim], h_in, pre_in [1,S,hc], h_out, pre_out)`` (``h_pre`` is the
    block input before the Engram; ``h_in`` after it) plus the final logits ``[S, V]`` (all rows
    for ``forward``, the single replay row for CED). CED traces carry ``x_dec`` (layer ``d``'s
    hoisted attention input over ALL tokens) and ``replay_start``."""

    def __init__(self, layers, logits, start_pos=0, x_dec=None, replay_start=0, decoder_start=None):
        self.layers = layers
        self.logits = logits
        self.start_pos = start_pos
        self.x_dec = x_dec
        self.replay_start = replay_start
        self.decoder_start = decoder_start


@torch.no_grad()
def oracle_trace_forward(model, ids: torch.Tensor, start_pos: int, hash_ids: torch.Tensor | None) -> OracleTrace:
    """``ReferenceModel.forward`` with every block's input/output recorded (same cache side
    effects as the real call)."""
    from .dsv41_reference import make_identity_pre_mix

    bsz, seqlen = ids.shape
    h = model.num.bf(model.embed[ids]).unsqueeze(2).repeat(1, 1, model.hc_mult, 1)
    pre = make_identity_pre_mix(bsz, seqlen, model.hc_mult)
    layers = []
    for i, layer in enumerate(model.layers):
        h_pre = h
        h = model._apply_engram(i, h, hash_ids)
        h_out, pre_out = layer(h, start_pos, pre)
        # window offset of a cold prefill / decode call: the window rows precede the compressed
        # ones (kv.size(1) == seqlen when start_pos == 0, window_size on a decode step)
        off = seqlen if start_pos == 0 else model.args.window_size
        layers.append(LayerTrace(h_pre, h, pre, h_out, pre_out, _oracle_picks(model, i, off),
                                 oracle_routes(layer)))
        h, pre = h_out, pre_out
    _, all_logits = model.logits_from(h, pre, True)
    return OracleTrace(layers, all_logits[0], start_pos)


@torch.no_grad()
def oracle_trace_ced(model, ids: torch.Tensor, hash_ids: torch.Tensor | None) -> OracleTrace:
    """``ced_prefill`` with the encoder blocks (over all tokens), the hoisted decoder input and
    the replay blocks (over the last ``window`` tokens) recorded."""
    from .dsv41_reference import make_identity_pre_mix

    args = model.args
    d = args.decoder_start()
    bsz, seqlen = ids.shape
    win = args.window_size
    h = model.num.bf(model.embed[ids]).unsqueeze(2).repeat(1, 1, model.hc_mult, 1)
    pre = make_identity_pre_mix(bsz, seqlen, model.hc_mult)
    layers = []
    for i in range(d):
        h_pre = h
        h = model._apply_engram(i, h, hash_ids)
        h_out, pre_out = model.layers[i](h, 0, pre)
        layers.append(LayerTrace(h_pre, h, pre, h_out, pre_out, _oracle_picks(model, i, seqlen),
                                 oracle_routes(model.layers[i])))
        h, pre = h_out, pre_out
    dec = model.layers[d]
    h_d = model._apply_engram(d, h, hash_ids)
    x_d = dec.attn_norm(dec.hc_pre(h_d, pre))
    dec.attn.produce_shared_kv(x_d, 0)
    s = max(0, seqlen - win)
    h, pre = h[:, s:], pre[:, s:]
    hash_ids = hash_ids[:, s:] if hash_ids is not None else None
    for i in range(d, args.n_layers):
        h_pre = h
        h = model._apply_engram(i, h, hash_ids)
        h_out, pre_out = model.layers[i](h, s, pre, window_mode="replay", produce_kv=(i != d))
        layers.append(LayerTrace(h_pre, h, pre, h_out, pre_out, _oracle_picks(model, i, seqlen - s),
                                 oracle_routes(model.layers[i])))
        h, pre = h_out, pre_out
    logits, _ = model.logits_from(h, pre, False)
    return OracleTrace(layers, logits, 0, x_dec=x_d, replay_start=s, decoder_start=d)


def _bf(t: torch.Tensor, dev) -> torch.Tensor:
    return t.to(torch.bfloat16).to(dev)


def force_picks(port, L: int, blocks: torch.Tensor | None, dev, stats: dict | None = None) -> None:
    """Teacher-force the compressed picks of index-source layer ``L`` for its next prefill call:
    ``Indexer.score_prefill`` returns the oracle's block ids instead of its own top-k (the port
    scores in fp32 where the oracle rounds to bf16, so on the toy a few queries per 128 swap one
    pick at the top-k boundary -- a discrete difference that is not a wiring error). The port's
    own picks are still computed and, when ``stats`` is given, compared against the oracle's
    (rows whose pick SETS differ) under ``stats[f"L{L}.pick_rows_differ"]``."""
    indexer = port.layers[L].attn.indexer
    if blocks is None or indexer is None or not indexer.is_index_source:
        return
    forced = blocks.to(dev)
    orig = type(indexer).score_prefill

    def score_prefill(x, qr, start_pos, ti, runtime):
        own = orig(indexer, x, qr, start_pos, ti, runtime)
        if stats is not None:
            a, b = own.cpu().sort(-1).values, forced.cpu().sort(-1).values
            n = min(a.shape[-1], b.shape[-1])
            a, b = a[..., -n:], b[..., -n:]
            differ = (a != b).any(-1)
            stats[f"L{L}.pick_rows"] = int(differ.numel())
            stats[f"L{L}.pick_rows_differ"] = int(differ.sum())
            # per row, how many of the oracle's picks the port did not pick (a boundary swap
            # is 1; a wiring error is ~all of them)
            missing = (b.unsqueeze(-1) != a.unsqueeze(-2)).all(-1).sum(-1)
            stats[f"L{L}.pick_max_elems_differ"] = int(missing.max())
        return forced[:, : x.shape[1]]

    indexer.score_prefill = score_prefill


def force_gate_routes(gate, routes: torch.Tensor | None, dev, stats: dict | None = None,
                      key: str = "") -> None:
    """Teacher-force the routed-expert choice of ``gate``'s NEXT call: it returns the oracle's
    expert ids with the port's own unbiased scores gathered at them (normalised and scaled
    exactly as ``Gate.forward`` does). Top-2-of-8 routing on random toy weights has near-ties
    -- measured on a 5090 vs the interpreter: 1-2 rows per 200 flip, each with a selected-vs-
    rejected margin of 1e-4 against a median of 9e-2 -- which any bf16/fp8-level kernel
    difference flips; a flipped row moves that block output by 0.13-0.3, a discrete difference
    that is not a wiring error. The port's own routing is still computed and, with ``stats``,
    the rows whose expert SETS differ are counted under ``stats[f"{key}.route_rows_differ"]``.
    One-shot: the patch removes itself after the call, so later (decode) calls route on their
    own unless forced again."""
    if routes is None or gate is None:
        return
    forced = routes.to(dev, torch.int64)
    orig = type(gate).forward

    def forward(x):
        del gate.forward  # one-shot: back to the class method
        own_w, own_idx = orig(gate, x)
        assert x.shape[0] == forced.shape[0], (key, x.shape, forced.shape)
        if stats is not None:
            a, b = own_idx.cpu().sort(-1).values, forced.cpu().sort(-1).values
            stats[f"{key}.route_rows_differ"] = int((a != b).any(-1).sum())
        w = gate.scores(x).gather(1, forced)
        if gate.norm_topk_prob and gate.topk > 1:
            w = w / (w.sum(dim=-1, keepdim=True) + 1e-20)
        return w * gate.route_scale, forced

    gate.forward = forward


def force_routes(port, L: int, routes: torch.Tensor | None, dev, stats: dict | None = None) -> None:
    """:func:`force_gate_routes` for the main-tower block ``L`` (stats key ``L{L}``)."""
    force_gate_routes(getattr(port.layers[L].ffn, "gate", None), routes, dev, stats, f"L{L}")


def _port_block(port, L: int, tr: LayerTrace, attn_call, engram_feat, dev, errs: dict,
                routes: torch.Tensor | None = None) -> None:
    """Run port block ``L`` on the oracle's block input; record its errors into ``errs``.
    ``routes`` (the oracle's expert ids) teacher-forces this block's MoE routing."""
    layer = port.layers[L]
    if routes is not None:
        force_routes(port, L, routes, dev, errs)
    eng = getattr(layer, "engram", None)
    if eng is not None:
        h_in_p = eng(_bf(tr.h_pre, dev), engram_feat)
        errs[f"L{L}.engram"] = rel_max_err(h_in_p, tr.h_in)
    h_out, pre_out = layer(_bf(tr.h_in, dev), tr.pre_in[0].float().to(dev), attn_call=attn_call)
    errs[f"L{L}.h"] = rel_max_err(h_out, tr.h_out)
    errs[f"L{L}.pre"] = rel_max_err(pre_out, tr.pre_out[0])


@torch.no_grad()
def port_prefill_teacher_forced(port, ctx, table_idx: int, trace: OracleTrace,
                                engram_feats: dict[int, torch.Tensor] | None = None,
                                teacher_force_picks: bool = False,
                                teacher_force_routes: bool = False) -> dict[str, float]:
    """Teacher-forced prefill of the port (mode = ``port.prefill_mode``, matching the trace kind)
    over one request. Returns ``{"L{i}.h", "L{i}.pre", "L{i}.engram", "x_dec", "logits"}`` rel
    errors (plus ``L{i}.pick_rows_differ`` counts with ``teacher_force_picks``, see
    :func:`force_picks`, and ``L{i}.route_rows_differ`` with ``teacher_force_routes``, see
    :func:`force_gate_routes`). ``logits`` is the port head on the ORACLE's final streams."""
    from freetoken.models.deepseek_v41.runtime import SharedAttentionRuntime

    dev = ctx.kv_cache.device
    args = port.args
    engram_feats = engram_feats or {}
    errs: dict[str, float] = {}
    T = trace.layers[0].h_in.shape[1]
    start = trace.start_pos
    segments = [(0, T, table_idx, start)]
    flat = torch.arange(start, start + T, device=dev)
    runtime = SharedAttentionRuntime()
    if teacher_force_picks:
        for L in range(args.n_layers):
            force_picks(port, L, trace.layers[L].picks, dev, errs)
    if port.prefill_mode == "full":
        assert trace.x_dec is None, "a CED trace needs a CED port"
        for L in range(args.n_layers):
            attn = port.layers[L].attn
            _port_block(port, L, trace.layers[L],
                        lambda y, attn=attn: attn.forward_ragged(y, segments, flat, runtime, "causal"),
                        engram_feats.get(L), dev, errs,
                        routes=trace.layers[L].routes if teacher_force_routes else None)
        last = trace.layers[-1]
        rows = torch.arange(T, device=dev)
        lg = port.logits(_bf(last.h_out, dev), last.pre_out[0].float().to(dev), rows)
        errs["logits"] = rel_max_err(lg, trace.logits)
        return errs
    assert trace.x_dec is not None, "a full trace needs a full port"
    d = port.decoder_start
    assert d == trace.decoder_start
    for L in range(d):
        attn = port.layers[L].attn
        _port_block(port, L, trace.layers[L],
                    lambda y, attn=attn: attn.forward_ragged(y, segments, flat, runtime, "causal"),
                    engram_feats.get(L), dev, errs,
                    routes=trace.layers[L].routes if teacher_force_routes else None)
    enc_last = trace.layers[d - 1]
    dec_layer = port.layers[d]
    x_dec = dec_layer.attn_input(_bf(enc_last.h_out, dev), enc_last.pre_out[0].float().to(dev))
    errs["x_dec"] = rel_max_err(x_dec, trace.x_dec)
    slots = dec_layer.attn.attn.window_slots_of(table_idx, start, start + T)
    dec_layer.attn.produce_shared_kv(_bf(trace.x_dec, dev), start, table_idx, slots)
    s = trace.replay_start
    m = T - s
    segs_r = [(0, m, table_idx, start + s)]
    flat_r = torch.arange(start + s, start + T, device=dev)
    runtime_r = SharedAttentionRuntime(produce_kv=False)
    for L in range(d, args.n_layers):
        attn = port.layers[L].attn
        _port_block(port, L, trace.layers[L],
                    lambda y, attn=attn: attn.forward_ragged(y, segs_r, flat_r, runtime_r, "replay"),
                    None, dev, errs,
                    routes=trace.layers[L].routes if teacher_force_routes else None)
    last = trace.layers[-1]
    lg = port.logits(_bf(last.h_out[:, -1:], dev), last.pre_out[0, -1:].float().to(dev))
    errs["logits"] = rel_max_err(lg, trace.logits)
    return errs


@torch.no_grad()
def port_decode_teacher_forced(port, ctx, table_idx: int, pos: int, trace: OracleTrace,
                               engram_feats: dict[int, torch.Tensor] | None = None,
                               teacher_force_routes: bool = False) -> dict[str, float]:
    """Teacher-forced decode step at ``pos`` (trace from ``oracle_trace_forward(ids[:, pos:pos+1],
    pos, ...)``): every block on the oracle's input through the port's decode addressing."""
    from freetoken.models.deepseek_v41.runtime import SharedAttentionRuntime

    dev = ctx.kv_cache.device
    engram_feats = engram_feats or {}
    errs: dict[str, float] = {}
    batch = make_decode_batch(ctx, table_idx, pos)
    pos_t = torch.tensor([pos], device=dev, dtype=torch.int64)
    rows = torch.arange(1, device=dev)
    with ctx.forward_batch(batch):
        wctx = batch.attn_metadata.window_ctx(pos_t, rows)
        runtime = SharedAttentionRuntime()
        for L in range(port.args.n_layers):
            attn = port.layers[L].attn
            _port_block(port, L, trace.layers[L],
                        lambda y, attn=attn: attn.decode_step(y, pos_t, rows, pos, wctx, runtime, None),
                        engram_feats.get(L), dev, errs,
                        routes=trace.layers[L].routes if teacher_force_routes else None)
        last = trace.layers[-1]
        lg = port.logits(_bf(last.h_out, dev), last.pre_out[0].float().to(dev))
        errs["logits"] = rel_max_err(lg, trace.logits)
    return errs


def worst(errs: dict[str, float], key: str) -> tuple[str, float]:
    """The largest ``errs`` entry whose name ends with ``key`` (e.g. ``".h"``)."""
    sel = {k: v for k, v in errs.items() if k.endswith(key)}
    k = max(sel, key=sel.get)
    return k, sel[k]


# --------------------------------------------------------------------------------------------
# DSpark draft addressing (engine/spec_dspark.py::_window_topk / _draft_block)
# --------------------------------------------------------------------------------------------
def window_topk(ctx, table_idx: int, last_pos: int, m: int, window_floor: int = 0) -> torch.Tensor:
    """``[1, m, win]`` int32 global window slots of the main positions ``<= last_pos`` in ring
    order (slot ``j`` holds ``p = last_pos - ((last_pos - j) % win)``), ``-1`` where ``p < 0``
    or ``p < window_floor`` (CED: main KV exists only for the replayed tail)."""
    pool = ctx.kv_cache
    win = ctx.attn_backend.window_size
    dev = pool.device
    j = torch.arange(win, device=dev)
    lp = torch.tensor([last_pos], device=dev, dtype=torch.int64)
    p = lp[:, None] - ((lp[:, None] - j[None, :]) % win)
    row = pool.full_loc_map[table_idx]
    ws = pool.translate_full_to_window(row.index_select(0, p.clamp(min=0).view(-1)))
    ws = torch.where((p.view(-1) >= 0) & (p.view(-1) >= window_floor), ws, torch.full_like(ws, -1))
    return ws.view(1, 1, -1).expand(1, m, -1).to(torch.int32)


@torch.no_grad()
def store_stage_main_kv(port, ctx, table_idx: int, main_hidden: torch.Tensor, lo: int, hi: int) -> None:
    """Write the DSpark stages' main-sequence window KV for positions ``[lo, hi)`` from
    ``main_hidden`` ``[hi - lo, n_targets * dim]`` (``Transformer._dspark_prefill`` /
    ``DsparkExecutor._store_main_kv`` on a caller-supplied context, e.g. the oracle's)."""
    dev = ctx.kv_cache.device
    stage0 = port.mtp[0]
    main_x = stage0.main_x(main_hidden.to(torch.bfloat16).to(dev))
    positions = torch.arange(lo, hi, device=dev)
    freqs = stage0.attn.freqs_cis.index_select(0, positions)
    slots = ctx.attn_backend.window_slots_of(table_idx, lo, hi)
    for stage in port.mtp:
        kv = stage.attn.main_kv(main_x, freqs)
        ctx.attn_backend.store_window(kv, stage.layer_id, slots)


@torch.no_grad()
def draft_block(port, ctx, table_idx: int, anchor: int, anchor_pos: int, scratch, window_floor: int = 0):
    """Draft one block from ``anchor`` at ``anchor_pos`` over the main window ``<= anchor_pos-1``
    (``_draft_block``'s addressing). Returns ``(hidden [m, dim], base logits [m, V] fp32)``."""
    from freetoken.models.deepseek_v41.dspark import draft_block_forward, draft_block_ids

    dev = ctx.kv_cache.device
    m = port.mtp[-1].block_size
    ids = draft_block_ids(torch.tensor([anchor], device=dev, dtype=torch.int32), m, port.mtp[-1].noise_token_id)
    pos = anchor_pos + torch.arange(m, device=dev)
    freqs = port.mtp[0].attn.freqs_cis.index_select(0, pos)
    topk = window_topk(ctx, table_idx, anchor_pos - 1, m, window_floor)
    return draft_block_forward(port, ids, freqs, topk, scratch.block_kv)


__all__ = [
    "DEV", "INTERPRET", "P", "TOY",
    "DenseRoutedExperts", "apply_runtime_patches", "argmax_agreement", "bind_request", "build_port",
    "decode_step", "device", "draft_block", "engram_feats", "engram_feats_one", "make_ctx",
    "make_decode_batch", "make_inputs", "make_prefill_batch", "oracle_args", "port_args",
    "port_state_dict", "prefill", "prefill_via_backend", "rel_max_err", "skip_reason", "window_topk",
    "LayerTrace", "OracleTrace", "oracle_trace_forward", "oracle_trace_ced", "port_prefill_teacher_forced",
    "port_decode_teacher_forced", "store_stage_main_kv", "worst", "force_picks",
    "force_gate_routes", "force_routes", "oracle_routes",
]
