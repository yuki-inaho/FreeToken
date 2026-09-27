"""DeepSeek-V4.1 port vs the pure-torch oracle on the toy geometry (``dsv41_harness``).

Two kinds of comparison, for the reason spelled out in the harness: a random toy amplifies any
bf16-level difference through the fp8-quantized linears (one fp8 ulp is 2^-3) into MoE routing
flips, so end-to-end logits are only loosely comparable (the oracle's own prefill-vs-decode bound
is 2.5e-1). Therefore:

* TEACHER-FORCED per-block parity (tight): every port block gets the oracle's block input and
  its output is compared -- prefill in ``full`` mode, the CED encoder / hoisted decoder KV /
  replay, and decode steps through the port's real decode addressing. A wiring error is O(1)
  here; the measured noise is the bf16/fp8 level.
* END-TO-END runs (loose): the port's own prefill / decode / chunked / radix-continuation /
  DSpark drafts, reported as rel error + top-1 agreement and bounded generously.

GPU vs interpreter (2026-09-10, RTX 5090): the same toy that is tight under the interpreter
showed 0.13-0.3 on ONE or TWO tokens per block on the GPU, every one a top-2-of-8 routing
near-tie (gate margin 1e-4 vs a median of 9e-2) flipped by the bf16/fp8-level differences
between the native fp8 GEMM path and the interpreter's emulation. The teacher-forced tests
therefore force the MoE routing as well as the compressed picks (``H.force_gate_routes``),
and bound how many rows the port's OWN routing / picks may differ on instead.

Oracle numerics: the port's Triton sparse-attention kernel keeps the softmax probabilities in
fp32 (``p.to(kv.dtype)`` with the KV tile upcast to fp32), where the tilelang reference kernel
casts them to bf16 before the PV product. The tight comparisons use ``Numerics(attn_probs_bf16=
False)`` (measured 2e-4 vs 3e-3 on the attention output, and bit-exact on window-only layer 0);
the bf16-faithful numbers are printed alongside. Everything else is the faithful oracle.

Device knob ``FREETOKEN_TEST_DEVICE`` (``cuda`` default / ``cpu`` under ``TRITON_INTERPRET=1
FREETOKEN_FORCE_E4M3_EMU=1``); the CPU run takes ~10-15 min (interpreter).
"""

from __future__ import annotations

import pytest
import torch

from . import dsv41_harness as H
from .dsv41_reference import Numerics, ReferenceModel, ced_prefill, make_random_checkpoint

_skip = H.skip_reason()
pytestmark = pytest.mark.skipif(_skip is not None, reason=_skip or "")

SEED = 1234
TI = 0
PROMPT = 200
DECODE_STEPS = 6

# ---- tolerances (measured under the interpreter, see the module docstring) ------------------
# Teacher-forced per block: bf16 streams through fp8-quantized linears. Measured worst block on
# the toy: prefill h ~1e-2 (fp32-probs oracle); decode a bit higher (GEMV vs GEMM reassociation).
TOL_BLOCK_H = 6e-2
TOL_BLOCK_PRE = 6e-2
# Against the bf16-probs oracle (measured worst block 0.295 -- see the test's docstring).
TOL_BLOCK_H_FAITHFUL = 0.6
TOL_ENGRAM = 2e-2
TOL_HEAD = 2e-2  # port head on the oracle's final streams
# End-to-end: loose (routing chaos); a miswired port scores ~1/64 agreement and rel err > 1.
TOL_E2E_REL = 1.0
MIN_E2E_AGREE = 0.35
# Port-vs-port (chunked vs single-shot, radix continuation vs cold): same kernels, differences
# only from M-dependent fp32 reassociation in torch's own matmuls flipping bf16 roundings.
TOL_SELF_REL = 2e-1
MIN_SELF_AGREE = 0.8
# Teacher-forced routing (``H.force_gate_routes``): rows per prefill block whose port-vs-oracle
# expert SETS differ (measured on a 5090: 1-2 of 200 on the flipped layers, all at gate margins
# of ~1e-4), the total over the decode steps, and over the DSpark draft rows (5 x 3 stages x 3).
MAX_ROUTE_ROWS_DIFFER = 4
MAX_ROUTE_ROWS_DIFFER_DECODE = 3
MAX_ROUTE_ROWS_DIFFER_DSPARK = 4
# The port's own compressed picks vs the oracle's (bf16-rounded scores): rows with a swapped
# pick (measured: 4-5/128 on the interpreter; 5/200 at the index sources and 10/200 at the
# reindex layer L6 on a 5090) and how many picks a row may differ by: a top-k boundary swap
# is 1 at an index source; at a Reindex layer the pool itself comes from the candidate
# source's own block scores, so a near-tie at the last candidate block swaps a whole block
# of ``candidate_block_size`` picks (measured 7 of 8 at L6).
MAX_PICK_ROWS_DIFFER_FRAC = 0.08
MAX_PICK_ELEMS_DIFFER = 2


def _check_picks(errs: dict, oargs, layers) -> None:
    for L in layers:
        rows, differ = errs[f"L{L}.pick_rows"], errs[f"L{L}.pick_rows_differ"]
        assert differ <= MAX_PICK_ROWS_DIFFER_FRAC * rows, (L, differ, rows)
        elems = oargs.candidate_block_size if oargs.mode(L) == "reindex" else MAX_PICK_ELEMS_DIFFER
        assert errs[f"L{L}.pick_max_elems_differ"] <= elems, (L, errs[f"L{L}.pick_max_elems_differ"], elems)


def _route_rows(errs: dict) -> int:
    return sum(int(v) for k, v in errs.items() if k.endswith("route_rows_differ"))


@pytest.fixture(scope="module")
def oargs():
    return H.oracle_args()


@pytest.fixture(scope="module")
def pargs():
    return H.port_args()


@pytest.fixture(scope="module")
def sd(oargs):
    return make_random_checkpoint(oargs, seed=SEED)


@pytest.fixture(scope="module")
def inputs(oargs):
    return H.make_inputs(oargs, 320)  # radix test needs 296 tokens


@pytest.fixture
def ctx(pargs):
    pool, backend, ctx = H.make_ctx(pargs, H.device(), max_running_req=2)
    H.bind_request(ctx, 0)
    H.bind_request(ctx, 1)
    return ctx


def _report(name: str, **kv) -> None:
    print(f"[dsv41-parity] {name}: " + ", ".join(f"{k}={v:.3e}" if isinstance(v, float) else f"{k}={v}" for k, v in kv.items()))


F32_PROBS = Numerics(attn_probs_bf16=False)


# ============================================================================================
# Engram module
# ============================================================================================
def test_engram_module_matches_oracle(oargs, pargs, sd, inputs):
    from .dsv41_reference import Engram as OracleEngram

    dev = H.device()
    ids, hash_ids = inputs
    T = 37
    port = H.build_port(pargs, sd, dev, "full")
    eng = port.layers[1].engram
    g = torch.Generator().manual_seed(11)
    h = torch.randn(1, T, oargs.hc_mult, oargs.dim, generator=g).to(torch.bfloat16)
    ref = OracleEngram(oargs, sd, "layers.1.engram", Numerics())(h.float(), hash_ids[:, :T, 0, :])
    feats = H.engram_feats_one(sd, 1, hash_ids[0, :T, 0, :], dev)
    got = eng(h.to(dev), feats)
    err = H.rel_max_err(got, ref)
    _report("engram", rel_err=err)
    assert got.dtype == torch.bfloat16 and got.shape == h.shape
    assert err <= TOL_ENGRAM


# ============================================================================================
# Teacher-forced per-block parity: full prefill, then decode steps
# ============================================================================================
def test_full_prefill_and_decode_teacher_forced(oargs, pargs, sd, inputs, ctx):
    dev = H.device()
    ids, hash_ids = inputs
    n = PROMPT
    port = H.build_port(pargs, sd, dev, "full", quant_act=True, pool=ctx.kv_cache)
    feats = H.engram_feats(pargs, sd, hash_ids, 0, n, dev)

    ref = ReferenceModel(oargs, sd, F32_PROBS)
    tr = H.oracle_trace_forward(ref, ids[:, :n], 0, hash_ids[:, :n])
    # Compressed picks and MoE routing teacher-forced (discrete near-tie flips on the toy, see
    # the module docstring); the port's own picks / routing are bounded by the row counts.
    errs = H.port_prefill_teacher_forced(port, ctx, TI, tr, feats, teacher_force_picks=True,
                                         teacher_force_routes=True)
    kh, eh = H.worst(errs, ".h")
    kp, ep = H.worst(errs, ".pre")
    _report("full prefill teacher-forced (fp32-probs oracle)", worst_h=f"{kh}={eh:.3e}",
            worst_pre=f"{kp}={ep:.3e}", engram=errs["L1.engram"], head=errs["logits"],
            per_layer_h=[round(errs[f"L{L}.h"], 4) for L in range(oargs.n_layers)],
            pick_rows_differ={k: v for k, v in errs.items() if k.endswith("pick_rows_differ")},
            pick_max_elems_differ=[errs[f"L{L}.pick_max_elems_differ"] for L in oargs.index_source_layers],
            route_rows_differ=[errs.get(f"L{L}.route_rows_differ", 0) for L in range(oargs.n_layers)])
    _check_picks(errs, oargs, oargs.index_source_layers)
    for L in range(oargs.n_layers):
        assert errs.get(f"L{L}.route_rows_differ", 0) <= MAX_ROUTE_ROWS_DIFFER, (L, errs[f"L{L}.route_rows_differ"])
    assert errs["L1.engram"] <= TOL_ENGRAM
    assert eh <= TOL_BLOCK_H, (kh, eh)
    assert ep <= TOL_BLOCK_PRE, (kp, ep)
    assert errs["logits"] <= TOL_HEAD

    # decode: the oracle continues from its own prefill, the port from the teacher-forced one
    worst_dec = 0.0
    route_diff = 0
    for j in range(DECODE_STEPS):
        pos = n + j
        trd = H.oracle_trace_forward(ref, ids[:, pos:pos + 1], pos, hash_ids[:, pos:pos + 1])
        fd = H.engram_feats(pargs, sd, hash_ids, pos, pos + 1, dev)
        ed = H.port_decode_teacher_forced(port, ctx, TI, pos, trd, fd, teacher_force_routes=True)
        kh, eh = H.worst(ed, ".h")
        _report(f"decode teacher-forced pos {pos}", worst_h=f"{kh}={eh:.3e}", head=ed["logits"],
                per_layer_h=[round(ed[f"L{L}.h"], 4) for L in range(oargs.n_layers)])
        worst_dec = max(worst_dec, eh)
        assert eh <= TOL_BLOCK_H, (pos, kh, eh)
        assert ed["logits"] <= TOL_HEAD
        route_diff += _route_rows(ed)
    assert route_diff <= MAX_ROUTE_ROWS_DIFFER_DECODE, route_diff
    _report("decode teacher-forced", worst_block_h=worst_dec)


def test_full_prefill_teacher_forced_against_faithful_oracle(oargs, pargs, sd, inputs, ctx):
    """Same as above against the bf16-faithful oracle (the reference tilelang kernel casts the
    softmax probabilities to bf16 before the PV product; the port's Triton kernel keeps them
    in fp32). The deviation is 3e-3 at the attention output, but on the toy it flips the
    top-2-of-8 MoE routing INSIDE a block (measured worst block 0.295, per layer
    [0.038, 0.211, 0.205, 0.293, 0.254, 0.023, 0.148, 0.295]), so this is a documented LOOSE
    bound: it guards against gross breakage only; the fp32-probs test above is the sharp one."""
    dev = H.device()
    ids, hash_ids = inputs
    n = 160
    port = H.build_port(pargs, sd, dev, "full", quant_act=True, pool=ctx.kv_cache)
    feats = H.engram_feats(pargs, sd, hash_ids, 0, n, dev)
    ref = ReferenceModel(oargs, sd, Numerics())
    tr = H.oracle_trace_forward(ref, ids[:, :n], 0, hash_ids[:, :n])
    errs = H.port_prefill_teacher_forced(port, ctx, TI, tr, feats)
    kh, eh = H.worst(errs, ".h")
    _report("full prefill teacher-forced (bf16-probs oracle)", worst_h=f"{kh}={eh:.3e}",
            per_layer_h=[round(errs[f"L{L}.h"], 4) for L in range(oargs.n_layers)])
    assert eh <= TOL_BLOCK_H_FAITHFUL, (kh, eh)


# ============================================================================================
# Teacher-forced CED: encoder over all tokens, hoisted decoder KV, replay of the last window
# ============================================================================================
def test_ced_prefill_and_decode_teacher_forced(oargs, pargs, sd, inputs, ctx):
    dev = H.device()
    ids, hash_ids = inputs
    n = PROMPT
    port = H.build_port(pargs, sd, dev, "ced", quant_act=True, pool=ctx.kv_cache)
    feats = H.engram_feats(pargs, sd, hash_ids, 0, n, dev)
    ref = ReferenceModel(oargs, sd, F32_PROBS)
    tr = H.oracle_trace_ced(ref, ids[:, :n], hash_ids[:, :n])
    assert tr.replay_start == n - oargs.window_size
    # The compressed picks are teacher-forced too (``force_picks``): the port scores the
    # indexer in fp32 where the oracle rounds to bf16, so a few replay queries per 128 swap ONE
    # pick at the top-16 boundary (measured un-forced: L4 4/128 rows, L6 3/128 rows, single-
    # element set differences, e.g. {26} vs {78}); a swapped pick moves that row's attention
    # output by ~0.2-0.3 on the toy (L5, which reuses L4's picks verbatim, read 0.31). The
    # port's own picks are still computed and the rows whose pick SETS differ are reported.
    errs = H.port_prefill_teacher_forced(port, ctx, TI, tr, feats, teacher_force_picks=True,
                                         teacher_force_routes=True)
    kh, eh = H.worst(errs, ".h")
    kp, ep = H.worst(errs, ".pre")
    _report("ced prefill teacher-forced", worst_h=f"{kh}={eh:.3e}", worst_pre=f"{kp}={ep:.3e}",
            x_dec=errs["x_dec"], engram=errs["L1.engram"], head=errs["logits"],
            per_layer_h=[round(errs[f"L{L}.h"], 4) for L in range(oargs.n_layers)],
            pick_rows_differ={k: v for k, v in errs.items() if k.endswith("pick_rows_differ")},
            pick_max_elems_differ=[errs[f"L{L}.pick_max_elems_differ"] for L in oargs.index_source_layers],
            route_rows_differ=[errs.get(f"L{L}.route_rows_differ", 0) for L in range(oargs.n_layers)])
    _check_picks(errs, oargs, oargs.index_source_layers)
    for L in range(oargs.n_layers):
        assert errs.get(f"L{L}.route_rows_differ", 0) <= MAX_ROUTE_ROWS_DIFFER, (L, errs[f"L{L}.route_rows_differ"])
    assert errs["L1.engram"] <= TOL_ENGRAM
    assert errs["x_dec"] <= TOL_BLOCK_H
    assert eh <= TOL_BLOCK_H, (kh, eh)
    assert ep <= TOL_BLOCK_PRE, (kp, ep)
    assert errs["logits"] <= TOL_HEAD
    # decode continues from the CED state on both sides (the oracle: forward(ids, start_pos=L))
    route_diff = 0
    for j in range(DECODE_STEPS):
        pos = n + j
        trd = H.oracle_trace_forward(ref, ids[:, pos:pos + 1], pos, hash_ids[:, pos:pos + 1])
        fd = H.engram_feats(pargs, sd, hash_ids, pos, pos + 1, dev)
        ed = H.port_decode_teacher_forced(port, ctx, TI, pos, trd, fd, teacher_force_routes=True)
        kh, eh = H.worst(ed, ".h")
        _report(f"ced decode teacher-forced pos {pos}", worst_h=f"{kh}={eh:.3e}", head=ed["logits"],
                per_layer_h=[round(ed[f"L{L}.h"], 4) for L in range(oargs.n_layers)])
        assert eh <= TOL_BLOCK_H, (pos, kh, eh)
        assert ed["logits"] <= TOL_HEAD
        route_diff += _route_rows(ed)
    assert route_diff <= MAX_ROUTE_ROWS_DIFFER_DECODE, route_diff


# ============================================================================================
# End-to-end (loose): the port's own prefill / decode vs the oracle
# ============================================================================================
@pytest.mark.parametrize("mode", ["full", "ced"])
def test_end_to_end_prefill_and_decode(oargs, pargs, sd, inputs, ctx, mode):
    dev = H.device()
    ids, hash_ids = inputs
    n = PROMPT
    port = H.build_port(pargs, sd, dev, mode, quant_act=True, pool=ctx.kv_cache)
    feats = H.engram_feats(pargs, sd, hash_ids, 0, n, dev)
    ref = ReferenceModel(oargs, sd, Numerics())
    if mode == "full":
        logits = H.prefill(port, ctx, ids[0, :n], TI, 0, feats, all_rows=True)
        out = ref.forward(ids[:, :n], 0, hash_ids[:, :n], full_logits=True)
        err_all = H.rel_max_err(logits, out.all_logits[0])
        agree = H.argmax_agreement(logits, out.all_logits[0])
        err_last = H.rel_max_err(logits[-1:], out.logits)
        _report("full prefill e2e", all_rows_rel=err_all, all_rows_top1_agree=agree, last_rel=err_last)
        assert logits.shape == (n, oargs.vocab_size) and torch.isfinite(logits).all()
        assert agree >= MIN_E2E_AGREE
        assert err_all <= TOL_E2E_REL
    else:
        logits = H.prefill(port, ctx, ids[0, :n], TI, 0, feats)
        out = ced_prefill(ref, ids[:, :n], hash_ids[:, :n])
        err_last = H.rel_max_err(logits, out.logits)
        _report("ced prefill e2e", last_rel=err_last, top1_port=int(logits.argmax()), top1_oracle=int(out.logits.argmax()))
        assert logits.shape == (1, oargs.vocab_size) and torch.isfinite(logits).all()
        assert err_last <= TOL_E2E_REL
    dec_errs, hits = [], 0
    for j in range(DECODE_STEPS):
        pos = n + j
        fd = H.engram_feats(pargs, sd, hash_ids, pos, pos + 1, dev)
        lg = H.decode_step(port, ctx, int(ids[0, pos]), pos, TI, fd)
        step = ref.forward(ids[:, pos:pos + 1], pos, hash_ids[:, pos:pos + 1])
        dec_errs.append(H.rel_max_err(lg, step.logits))
        hits += int(lg.argmax() == step.logits.argmax())
        assert torch.isfinite(lg).all()
    _report(f"{mode} decode e2e", rel_errs=[round(e, 3) for e in dec_errs], top1_hits=f"{hits}/{DECODE_STEPS}")
    assert max(dec_errs) <= TOL_E2E_REL


# ============================================================================================
# Port vs port: chunked prefill == single-shot; CED radix-style continuation == cold prefill
# ============================================================================================
@pytest.mark.parametrize("mode", ["full", "ced"])
def test_chunked_prefill_matches_single_shot(oargs, pargs, sd, inputs, ctx, mode):
    dev = H.device()
    ids, hash_ids = inputs
    length = 2 * H.P if mode == "full" else 2 * H.P + 44  # ced: a short final chunk gets extended
    port = H.build_port(pargs, sd, dev, mode, quant_act=True, pool=ctx.kv_cache)
    # single-shot on request row 0
    if mode == "full":
        feats = H.engram_feats(pargs, sd, hash_ids, 0, length, dev)
        single = H.prefill(port, ctx, ids[0, :length], 0, 0, feats, all_rows=True)
    else:
        single, _ = H.prefill_via_backend(port, ctx, ids[0, :length], 0, 0, pargs, sd, hash_ids)
    # chunks of P on request row 1, through the backend's prepare_metadata
    pieces, segs_seen = [], []
    for start in range(0, length, H.P):
        end = min(length, start + H.P)
        if mode == "full":
            feats = H.engram_feats(pargs, sd, hash_ids, start, end, dev)
            pieces.append(H.prefill(port, ctx, ids[0, start:end], 1, start, feats, all_rows=True))
        else:
            lg, segs = H.prefill_via_backend(port, ctx, ids[0, :end], 1, start, pargs, sd, hash_ids)
            segs_seen.append(segs)
            pieces.append(lg)
    if mode == "full":
        chunked = torch.cat(pieces, dim=0)
        err = H.rel_max_err(chunked, single)
        agree = H.argmax_agreement(chunked, single)
        exact = (chunked == single).float().mean().item()
        _report("full chunked vs single", rel_err=err, top1_agree=agree, exact_frac=exact)
        assert agree >= MIN_SELF_AGREE and err <= TOL_SELF_REL
    else:
        # the 44-token tail was extended back to position 128 (>= window before the end)
        assert segs_seen[-1] == [(0, length - H.P, 1, H.P)], segs_seen
        err = H.rel_max_err(pieces[-1], single)
        _report("ced chunked vs single (last row)", rel_err=err, top1_chunked=int(pieces[-1].argmax()),
                top1_single=int(single.argmax()), exact_frac=(pieces[-1] == single).float().mean().item())
        assert err <= TOL_SELF_REL


def test_ced_radix_continuation_matches_cold_prefill(oargs, pargs, sd, inputs, ctx):
    """Prefill 256 tokens, then 40 more as a continuation (``cached_len=256``): the backend
    extends the segment back to position 128 and the port re-encodes ``[128, 296)``
    (idempotent rewrites of window KV / compressed rows / index keys, replay over the last 128)
    -- compared with a cold 296-token prefill on another request row."""
    dev = H.device()
    ids, hash_ids = inputs
    port = H.build_port(pargs, sd, dev, "ced", quant_act=True, pool=ctx.kv_cache)
    first, segs1 = H.prefill_via_backend(port, ctx, ids[0, :256], 0, 0, pargs, sd, hash_ids)
    assert segs1 == [(0, 256, 0, 0)]
    cont, segs2 = H.prefill_via_backend(port, ctx, ids[0, :296], 0, 256, pargs, sd, hash_ids)
    assert segs2 == [(0, 168, 0, 128)], segs2
    cold, segs3 = H.prefill_via_backend(port, ctx, ids[0, :296], 1, 0, pargs, sd, hash_ids)
    assert segs3 == [(0, 296, 1, 0)]
    err = H.rel_max_err(cont, cold)
    _report("ced radix continuation vs cold", rel_err=err, top1_cont=int(cont.argmax()),
            top1_cold=int(cold.argmax()), exact_frac=(cont == cold).float().mean().item())
    assert torch.isfinite(cont).all() and err <= TOL_SELF_REL


# ============================================================================================
# DSpark: the draft block vs the oracle's forward_spec (teacher-forced), and the report harness
# ============================================================================================
def test_dspark_draft_block_vs_oracle(oargs, pargs, sd, inputs, ctx):
    from freetoken.models.deepseek_v41.spec_state import DSV41SpecScratch

    dev = H.device()
    ids, hash_ids = inputs
    n = PROMPT
    block = oargs.dspark_block_size
    port = H.build_port(pargs, sd, dev, "full", quant_act=True, dspark=True, pool=ctx.kv_cache)
    assert len(port.mtp) == oargs.n_mtp_layers
    feats = H.engram_feats(pargs, sd, hash_ids, 0, n, dev)
    H.prefill(port, ctx, ids[0, :n], TI, 0, feats)  # writes the stages' main KV (port's own context)
    ref = ReferenceModel(oargs, sd, F32_PROBS)
    out = ref.forward(ids[:, :n], 0, hash_ids[:, :n])
    scratch = DSV41SpecScratch(pargs, block, oargs.vocab_size, dev)
    last = port.mtp[-1]
    ids_all = ids[0]

    def port_draft(p: int):
        hidden, base = H.draft_block(port, ctx, TI, int(ids_all[p + 1]), p + 1, scratch)
        prev = ids_all[p + 1:p + 1 + block].to(dev)
        bias, emb = last.markov_head(prev)
        logits = base + bias
        conf = last.confidence_head(hidden, emb)
        return logits, conf

    def oracle_draft(p: int):
        truth = ids_all[p + 2:p + 2 + block].tolist()
        it = iter(truth)
        spec = ref.forward_spec(ids[:, p + 1], out.main_hidden[:, :p + 1], 0, draft=True,
                                sampler=lambda lg: torch.tensor([next(it)]))
        return spec.logits[0], spec.confidence[0]

    rows_e2e, rows_tf = [], []
    route_stats: dict = {}
    for p in (150, 172, 190):
        lo, lc = oracle_draft(p)
        stage_routes = [H.oracle_routes(stage) for stage in ref.mtp]
        lp, cp = port_draft(p)  # the port's own main KV
        rows_e2e.append((H.rel_max_err(lp, lo), H.argmax_agreement(lp, lo), H.rel_max_err(cp, lc)))
        # teacher-forced: the stages' main KV rewritten from the ORACLE's context rows, and the
        # stages' MoE routing forced to the oracle's (on a 5090 one near-tie flip inside a 5-row
        # draft block moved a position's confidence by 3.5x its scale)
        H.store_stage_main_kv(port, ctx, TI, out.main_hidden[0, :p + 1], 0, p + 1)
        for i, stage in enumerate(port.mtp):
            H.force_gate_routes(stage.ffn.gate, stage_routes[i], dev, route_stats, f"p{p}.S{i}")
        lp, cp = port_draft(p)
        rows_tf.append((H.rel_max_err(lp, lo), H.argmax_agreement(lp, lo), H.rel_max_err(cp, lc)))
        assert lp.shape == (block, oargs.vocab_size) and cp.shape == (block,)
        assert torch.isfinite(lp).all() and torch.isfinite(cp).all()
    _report("dspark draft e2e (port main KV)", per_pos=[tuple(round(v, 3) for v in r) for r in rows_e2e])
    _report("dspark draft teacher-forced (oracle main KV)", per_pos=[tuple(round(v, 3) for v in r) for r in rows_tf])
    _report("dspark stage routing port-vs-oracle", route_rows_differ=route_stats)
    assert _route_rows(route_stats) <= MAX_ROUTE_ROWS_DIFFER_DSPARK, route_stats
    for p, (err, agree, cerr) in zip((150, 172, 190), rows_tf):
        assert err <= TOL_E2E_REL and cerr <= TOL_E2E_REL, (p, err, cerr)
    # with the oracle's own context the drafts agree on most rows
    assert sum(a for _, a, _ in rows_tf) / len(rows_tf) >= 0.5


def test_dspark_teacher_forced_report_restricted_to_ced_window(oargs, pargs, sd, inputs, ctx):
    from freetoken.models.deepseek_v41.dspark import teacher_forced_report

    dev = H.device()
    ids, hash_ids = inputs
    n = PROMPT
    port = H.build_port(pargs, sd, dev, "ced", quant_act=True, dspark=True, pool=ctx.kv_cache)
    feats = H.engram_feats(pargs, sd, hash_ids, 0, n, dev)
    H.prefill(port, ctx, ids[0, :n], TI, 0, feats)  # stages' main KV for the replayed tail only
    rep = teacher_forced_report(port, ids[0, :n].to(dev), TI, stride=32)
    block = oargs.dspark_block_size
    assert set(rep) == {"with_bias", "without_bias", "positions"}
    assert len(rep["with_bias"]) == block and len(rep["without_bias"]) == block
    assert rep["positions"] and min(rep["positions"]) >= n - oargs.window_size
    assert all(0.0 <= v <= 1.0 for v in rep["with_bias"] + rep["without_bias"])
    _report("dspark teacher_forced_report (ced tail)", positions=rep["positions"],
            with_bias=[round(v, 3) for v in rep["with_bias"]])
