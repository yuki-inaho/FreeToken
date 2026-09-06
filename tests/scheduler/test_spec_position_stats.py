"""Host-only MTP telemetry: exposure denominators, accepted prefixes and terminal tails."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from freetoken.core import Batch, Req, SamplingParams
from freetoken.engine.spec import SpecBatchInfo
from freetoken.message import BaseFrontendMsg, UserReply
from freetoken.message.tokenizer import BaseTokenizerMsg, DetokenizeMsg
from freetoken.scheduler import scheduler as scheduler_mod
from freetoken.scheduler.scheduler import Scheduler, _engine_stats, _record_spec_position_stats
from freetoken.server.openai_api import _usage


def test_variable_depth_counts_actual_exposure_and_retained_prefixes():
    req = SimpleNamespace()
    # Fully accepted k=1; first rejection at k=4; EOS inside an accepted k=4 prefix;
    # then a budget-clipped k=2 verify whose bonus is beyond the length limit.
    for verified, staged, emitted, kept in ((1, 1, 2, 2), (4, 4, 1, 1),
                                            (4, 4, 5, 2), (2, 4, 3, 2)):
        _record_spec_position_stats(req, verified_depth=verified, staged_depth=staged,
                                    num_emitted=emitted, kept=kept)
    stats = req.stat_spec_positions
    assert stats == {
        "rounds": 4,
        "proposed": [4, 3, 2, 2],
        "accepted": [3, 2, 1, 1],
        "useful_accepted": [3, 2, 0, 0],
        "verified_depth_histogram": {"1": 1, "4": 2, "2": 1},
        "staged_depth_histogram": {"1": 1, "4": 3},
        "budget_clipped_rounds": 1,
        "budget_clipped_drafts": 2,
        "truncated_rounds": 2,
        "truncated_tokens": 4,
        "truncated_accepted_tokens": 2,
    }


def _drain(monkeypatch, *, depth, emitted, output_len, tokens=None, enabled=True,
           eos=(), stop_after=None, staged=None):
    monkeypatch.setattr(scheduler_mod, "_SPEC_POSITION_STATS", enabled)
    # No timing diagnostics and no global GPU synchronization are needed by telemetry.
    import freetoken.engine.spec as spec_mod

    monkeypatch.setattr(spec_mod, "_SPEC_HOST_TIME", False)
    monkeypatch.setattr(torch.cuda, "synchronize", Mock(side_effect=AssertionError("extra sync")))
    req = Req(input_ids=torch.tensor([10, 11], dtype=torch.int32), table_idx=0,
              cached_len=1, output_len=output_len, uid=1,
              sampling_params=SamplingParams(stop_strs=["END"] if stop_after else []),
              cache_handle=None)
    req.spec_state = "ready"
    req.spec_k_staged = staged or depth
    executor = SimpleNamespace(scratch=None, recommit=Mock(), drop=Mock())
    running = [req]
    scheduler = SimpleNamespace(
        engine=SimpleNamespace(spec_executor=executor, max_seq_len=128),
        token_pool=torch.zeros((1, 128), dtype=torch.int32),
        spec_k=staged or depth,
        finished_reqs=set(), eos_token_ids=set(eos),
        cache_manager=SimpleNamespace(lazy_free_region=nullcontext, truncate_paged=Mock()),
        decode_manager=SimpleNamespace(running_reqs=running, remove_req=running.remove,
                                       filter_reqs=Mock()),
        prefill_manager=SimpleNamespace(pending_list=[]),
        config=SimpleNamespace(page_size=1),
        status_reporter=SimpleNamespace(report_batch=Mock()),
        _free_req_resources=Mock(), _kv_usage_pages=lambda: (0, 1),
        _mamba_slot_usage=lambda: None, _swa_token_usage=lambda: None,
        _gpu_mem_bytes=lambda: 0, send_result=Mock(),
        _match_stop_str=lambda r: "END" if len(r.input_ids) == 2 + stop_after else None,
    )
    batch = Batch(reqs=[req], phase="decode")
    Scheduler._stage_spec(scheduler, batch)
    assert batch.spec.num_input_tokens - 1 == depth
    # In the real executor, staging the NEXT round changes this before drain. The
    # diagnostic must use the immutable schedule-time snapshot on batch.spec.
    req.spec_k_staged = 1
    candidates = tokens or list(range(20, 20 + depth + 1))
    output = SimpleNamespace(num_emitted_cpu=torch.tensor(emitted),
                             candidates_cpu=torch.tensor(candidates, dtype=torch.int32),
                             copy_done=SimpleNamespace(synchronize=Mock()))
    Scheduler._process_spec_last_data(scheduler, batch, output)
    output.copy_done.synchronize.assert_called_once_with()
    torch.cuda.synchronize.assert_not_called()
    return req, scheduler, executor


@pytest.mark.parametrize("depth", [1, 2, 3, 4])
@pytest.mark.parametrize("accept_all", [False, True])
def test_all_accept_or_first_reject_across_depths(monkeypatch, depth, accept_all):
    emitted = depth + 1 if accept_all else 1
    req, _, _ = _drain(monkeypatch, depth=depth, emitted=emitted, output_len=8)
    stats = req.stat_spec_positions
    assert stats["proposed"] == [1] * depth
    assert stats["accepted"] == [int(accept_all)] * depth
    assert stats["useful_accepted"] == stats["accepted"]
    assert stats["rounds"] == req.stat_spec_steps == req.stat_decode_steps == 1
    assert sum(stats["proposed"]) == req.stat_draft_tokens == depth
    assert sum(stats["accepted"]) == req.stat_accepted_tokens == emitted - 1


@pytest.mark.parametrize("terminal", ["eos", "stop"])
def test_terminal_token_in_accepted_prefix_counts_all_kept_drafts(monkeypatch, terminal):
    req, scheduler, executor = _drain(
        monkeypatch, depth=4, emitted=5, output_len=8,
        tokens=[20, 99, 21, 22, 23], eos=[99] if terminal == "eos" else [],
        stop_after=2 if terminal == "stop" else None,
    )
    stats = req.stat_spec_positions
    assert stats["accepted"] == [1, 1, 1, 1]
    assert stats["useful_accepted"] == [1, 1, 0, 0]
    assert stats["truncated_tokens"] == 3
    assert stats["truncated_accepted_tokens"] == 2
    assert req.stat_accepted_tokens == 4  # existing aggregate keeps raw semantics
    executor.recommit.assert_called_once_with(req, 2)
    reply = scheduler.send_result.call_args.args[0][0]
    assert reply.finish_reason == "stop"
    assert reply.engine_stats["spec_positions"] == stats


def test_length_clipping_snapshots_staged_depth_and_discards_only_bonus(monkeypatch):
    req, scheduler, executor = _drain(monkeypatch, depth=2, staged=4, emitted=3, output_len=2)
    stats = req.stat_spec_positions
    assert stats["proposed"] == [1, 1, 0, 0]
    assert stats["accepted"] == stats["useful_accepted"] == [1, 1, 0, 0]
    assert stats["staged_depth_histogram"] == {"4": 1}
    assert stats["verified_depth_histogram"] == {"2": 1}
    assert stats["budget_clipped_rounds"] == 1
    assert stats["budget_clipped_drafts"] == 2
    assert stats["truncated_rounds"] == stats["truncated_tokens"] == 1
    assert stats["truncated_accepted_tokens"] == 0
    executor.recommit.assert_called_once_with(req, 2)
    reply = scheduler.send_result.call_args.args[0][0]
    assert reply.finish_reason == "length"
    assert reply.engine_stats["spec_positions"] == stats


def test_stats_disabled_preserves_aggregate_payload_and_generation(monkeypatch):
    req, _, _ = _drain(monkeypatch, depth=4, emitted=5, output_len=8, enabled=False)
    assert req.stat_spec_positions is None
    assert "spec_positions" not in _engine_stats(req, 0)
    assert req.stat_accepted_tokens == 4
    assert req.input_ids.tolist() == [10, 11, 20, 21, 22, 23, 24]


def test_nested_position_payload_survives_messages_and_openai_usage():
    req = SimpleNamespace()
    _record_spec_position_stats(req, verified_depth=2, staged_depth=4, num_emitted=3, kept=2)
    stats = _engine_stats(req, 0)
    for msg, base in (
        (DetokenizeMsg(uid=1, next_token=3, finished=True, engine_stats=stats), BaseTokenizerMsg),
        (UserReply(uid=1, incremental_output="x", finished=True, engine_stats=stats), BaseFrontendMsg),
    ):
        decoded = base.decoder(base.encoder(msg))
        assert decoded.engine_stats == stats
        assert _usage(10, 5, 0, engine_stats=decoded.engine_stats)["freetoken"] == stats


def test_capture_batches_without_staged_metadata_remain_compatible():
    info = SpecBatchInfo(scratch=None, num_input_tokens=5)
    assert info.num_staged_drafts is None
