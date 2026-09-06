"""Context-limit regressions for the speculative scheduler's complete device write span."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from freetoken.core import Batch, Req, SamplingParams
from freetoken.engine import spec as spec_mod
from freetoken.engine.spec import SpecExecutor
from freetoken.scheduler.decode import DecodeManager
from freetoken.scheduler.scheduler import Scheduler


def _session(*, context=64, width=None, device_len, depth, staged=None, state="ready",
             output_len=None, block=None):
    req = Req(
        input_ids=torch.arange(device_len, dtype=torch.int32), table_idx=0,
        cached_len=device_len - 1,
        output_len=context - device_len if output_len is None else output_len,
        uid=7, sampling_params=SamplingParams(temperature=0), cache_handle=None,
    )
    req.spec_state = state
    req.spec_k_staged = depth if staged is None else staged
    executor = SimpleNamespace(scratch=None, drop=Mock())
    if block is not None:
        executor.block = block
    token_pool = torch.full((1, context if width is None else width), -1, dtype=torch.int32)
    token_pool[0, :device_len] = req.input_ids
    scheduler = SimpleNamespace(
        spec_k=depth, token_pool=token_pool,
        engine=SimpleNamespace(spec_executor=executor, max_seq_len=context),
    )
    batch = Batch(reqs=[req], phase="decode")
    return scheduler, req, batch


@pytest.mark.parametrize("depth", [1, 2, 3, 4])
@pytest.mark.parametrize("context,width", [(64, 64), (61, 64), (64, 61)])
@pytest.mark.parametrize("extra", [0, 1])
def test_verify_reserves_bonus_and_complete_next_chain(depth, context, width, extra):
    # Equality is safe: the final staged draft occupies the last available column.
    # One position later used to launch a verify that scattered beyond that column.
    limit = min(context, width)
    device_len = limit - 2 * depth - 1 + extra
    scheduler, req, batch = _session(
        context=context, width=width, device_len=device_len, depth=depth,
    )
    before = scheduler.token_pool.clone()
    Scheduler._stage_spec(scheduler, batch)
    if not extra:
        assert batch.spec.num_input_tokens == depth + 1
        assert req.device_len == device_len + depth
        scheduler.engine.spec_executor.drop.assert_not_called()
    else:
        assert batch.spec is None
        assert req.spec_state == "off"
        assert req.device_len == device_len
        scheduler.engine.spec_executor.drop.assert_called_once_with(req.uid)
        assert torch.equal(scheduler.token_pool, before)


@pytest.mark.parametrize("depth", [1, 2, 3, 4])
@pytest.mark.parametrize("extra", [0, 1])
def test_warmup_accounts_for_complete_one_before_draft_scatter(depth, extra):
    scheduler, req, batch = _session(
        device_len=64 - depth - 1 + extra, depth=depth, state="warmup",
    )
    Scheduler._stage_spec(scheduler, batch)
    assert (getattr(batch, "spec_warmup", None) is req) == (extra == 0)
    assert req.spec_state == ("warmup" if extra == 0 else "off")


def test_last_output_token_never_warms_up():
    scheduler, req, batch = _session(device_len=12, depth=3, state="warmup", output_len=1)
    Scheduler._stage_spec(scheduler, batch)
    assert getattr(batch, "spec_warmup", None) is None
    assert req.spec_state == "off"
    scheduler.engine.spec_executor.drop.assert_called_once_with(req.uid)


def test_adaptive_tail_reserves_max_depth_not_previous_staged_depth():
    scheduler, req, batch = _session(device_len=60, depth=3, staged=1)
    Scheduler._stage_spec(scheduler, batch)
    assert batch.spec is None
    assert req.spec_state == "off"


def test_shared_scheduler_reserves_entire_dspark_block():
    scheduler, req, batch = _session(device_len=56, depth=2, block=8)
    Scheduler._stage_spec(scheduler, batch)
    assert batch.spec is None
    assert req.spec_state == "off"


def test_boundary_fallback_does_not_rearm_or_change_output_budget():
    scheduler, req, batch = _session(device_len=61, depth=3)
    Scheduler._stage_spec(scheduler, batch)
    original_limit = req.max_device_len
    while req.can_decode:
        batch = Batch(reqs=[req], phase="decode")
        Scheduler._stage_spec(scheduler, batch)
        assert batch.spec is None
        token = torch.tensor([42], dtype=torch.int32)
        scheduler.token_pool[0, req.device_len] = token[0]
        req.complete_one()
        req.append_host(token)
    assert req.input_ids.numel() == original_limit == 64
    assert req.cached_len == 63
    assert req.input_ids[-3:].tolist() == [42, 42, 42]
    scheduler.engine.spec_executor.drop.assert_called_once_with(req.uid)


@pytest.mark.parametrize("state", ["ready", "warmup", "off"])
def test_overlap_tail_drains_last_token_before_finishing(state):
    scheduler, req, first_batch = _session(device_len=62, depth=3, state=state)
    scheduler.finished_reqs = set()
    scheduler.eos_token_ids = set()
    scheduler.toolcall_anchor_id = None
    scheduler.cache_manager = SimpleNamespace(lazy_free_region=nullcontext)
    scheduler.decode_manager = DecodeManager(page_size=1, running_reqs={req})
    scheduler.prefill_manager = SimpleNamespace(pending_list=[])
    scheduler.config = SimpleNamespace(page_size=1)
    scheduler.status_reporter = SimpleNamespace(report_batch=Mock())
    scheduler._free_req_resources = Mock()
    scheduler._kv_usage_pages = lambda: (0, 1)
    scheduler._mamba_slot_usage = lambda: None
    scheduler._swa_token_usage = lambda: None
    scheduler._gpu_mem_bytes = lambda: 0
    scheduler.send_result = Mock()
    Scheduler._stage_spec(scheduler, first_batch)
    assert first_batch.spec is None

    def forward(batch, token):
        req.complete_one()
        scheduler.decode_manager.filter_reqs([req])
        return (SimpleNamespace(batch=batch), (
            None, torch.tensor([token], dtype=torch.int32), SimpleNamespace(synchronize=Mock()),
        ))

    # Match overlap_loop: forward N+1 launches before forward N is drained. Both
    # batches refer to the same Req, whose device_len already includes the last token.
    first = forward(first_batch, 42)
    last = forward(Batch(reqs=[req], phase="decode"), 43)
    assert not req.can_decode
    Scheduler._process_last_data(scheduler, first)
    first_reply = scheduler.send_result.call_args.args[0][0]
    assert not first_reply.finished
    assert first_reply.finish_reason is None
    assert req.input_ids.numel() == 63
    scheduler._free_req_resources.assert_not_called()

    Scheduler._process_last_data(scheduler, last)
    last_reply = scheduler.send_result.call_args.args[0][0]
    assert last_reply.finished and last_reply.finish_reason == "length"
    assert req.input_ids.numel() == req.max_device_len == 64
    assert req.input_ids[-2:].tolist() == [42, 43]
    scheduler._free_req_resources.assert_called_once_with(req)


@pytest.mark.parametrize("depth", [1, 2, 3, 4])
@pytest.mark.parametrize("accept_all", [False, True])
def test_real_eager_verify_scatter_and_tail_fit_at_last_admitted_position(
    monkeypatch, depth, accept_all,
):
    # Exercise the real candidate scatter, accepted-count calculation, and draft tail
    # on CPU; only the model and CUDA event are replaced. This catches an off-by-one
    # in the scheduler against the actual executor writes rather than a copied formula.
    for flag in ("_SPEC_TIME", "_SPEC_HOST_TIME", "_SPEC_HOST_DETAIL", "_SPEC_DEBUG_STEPS"):
        monkeypatch.setattr(spec_mod, flag, False)
    monkeypatch.setattr(spec_mod, "_SPEC_PROFILE", None)
    monkeypatch.setattr(spec_mod, "_SPEC_NSYS", None)
    monkeypatch.setattr(torch.cuda, "Event", Mock(return_value=Mock()))
    scheduler, req, batch = _session(device_len=64 - 2 * depth - 1, depth=depth)
    Scheduler._stage_spec(scheduler, batch)
    k1 = depth + 1
    candidates = torch.arange(20, 20 + k1, dtype=torch.int32)
    logits = torch.zeros(k1, 128)
    logits.scatter_(1, candidates.long().view(-1, 1), 1)
    batch.input_ids = torch.cat((torch.tensor([5], dtype=torch.int32), candidates[:-1]))
    if not accept_all:
        batch.input_ids[1] = 99
    commit = Mock()
    executor = SimpleNamespace(
        device=torch.device("cpu"), token_pool=scheduler.token_pool, graph_sets={},
        _next_k=lambda _: depth, _gdn_slot_tensor=lambda *args, **kwargs: torch.tensor([0]),
        _draft_k_with_page_room=lambda input_end, k: k,
        _draw_draft=lambda *args, slot=0, **kwargs: torch.tensor([100 + slot], dtype=torch.int32),
        _draft_lm_head=lambda: SimpleNamespace(forward=lambda _: logits), _embed=lambda ids: ids,
        model=SimpleNamespace(
            forward=lambda: logits, mtp=SimpleNamespace(forward=lambda *args, **kwargs: logits),
        ),
        engine=SimpleNamespace(linear_state_pool=None, stream=None), ple_layer_ids=(),
        scratch=SimpleNamespace(commit=commit, premix_rows=None, r1_rows=None),
        _cand_cpu=torch.empty(k1, dtype=torch.int32), _n_cpu=torch.empty((), dtype=torch.int64),
        prefetcher=None,
    )
    prefix = scheduler.token_pool[0, :req.cached_len + 1].clone()
    output = SpecExecutor.verify_step(executor, batch, None)
    emitted = k1 if accept_all else 1
    assert int(output.num_emitted_cpu) == emitted
    assert torch.equal(output.candidates_cpu, candidates)
    assert torch.equal(scheduler.token_pool[0, :req.cached_len + 1], prefix)
    next_device_len = req.cached_len + emitted + 1
    assert scheduler.token_pool[0, next_device_len:next_device_len + depth].tolist() == list(
        range(100, 100 + depth)
    )
    assert int(commit.call_args.kwargs["num_inputs"]) == emitted
    assert int(commit.call_args.kwargs["new_device_len"]) == next_device_len
