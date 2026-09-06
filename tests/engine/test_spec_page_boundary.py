"""MTP draft chains may write only KV pages owned by their current forward."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from freetoken.core import Batch, Req, SamplingParams
from freetoken.engine import spec as spec_mod
from freetoken.engine.spec import SpecBatchInfo, SpecExecutor
from freetoken.scheduler.cache import CacheManager


@pytest.fixture(autouse=True)
def _cpu_events(monkeypatch):
    for flag in ("_SPEC_TIME", "_SPEC_HOST_TIME", "_SPEC_HOST_DETAIL", "_SPEC_DEBUG_STEPS"):
        monkeypatch.setattr(spec_mod, flag, False)
    monkeypatch.setattr(spec_mod, "_SPEC_PROFILE", None)
    monkeypatch.setattr(spec_mod, "_SPEC_NSYS", None)
    monkeypatch.setattr(torch.cuda, "Event", Mock(return_value=Mock()))


def _owned_pages(input_end):
    table = torch.full((1, 512), -1, dtype=torch.int32)
    cm = CacheManager(4, 128, table, "naive")
    cm.allocate_paged([SimpleNamespace(cached_len=0, device_len=input_end, table_idx=0)])
    return cm, table


def _req(device_len, depth):
    return Req(
        input_ids=torch.zeros(device_len, dtype=torch.int32), table_idx=0,
        cached_len=device_len - 1, output_len=512 - device_len, uid=7,
        sampling_params=SamplingParams(temperature=0), cache_handle=None,
        spec_state="ready", spec_k_staged=depth,
    )


@pytest.mark.parametrize("depth", [1, 2, 3, 4])
@pytest.mark.parametrize("input_end", [126, 127, 128, 129])
def test_real_warmup_chain_stays_on_owned_pages_and_recovers(depth, input_end):
    cm, table = _owned_pages(input_end)
    req = _req(input_end, depth)
    req.complete_one()  # engine does this before calling warmup_draft
    pool = torch.zeros_like(table)
    writes = []

    def mtp_forward(rows_r, tokens, rows, **kwargs):
        for table_idx, start, end in rows:
            assert torch.all(table[table_idx, start:end] >= 0)
            writes.extend(range(start, end))
        return rows_r

    ex = SimpleNamespace(
        engine=SimpleNamespace(config=SimpleNamespace(page_size=128)), k=depth, k_max=depth,
        device=torch.device("cpu"), token_pool=pool,
        _premix_carry={req.uid: torch.zeros(1, 2)}, _premix=lambda _: torch.zeros(1, 2),
        _shadow_batch=lambda rows: rows, _embed=lambda tokens: tokens,
        model=SimpleNamespace(mtp=SimpleNamespace(forward=mtp_forward)),
        scratch=SimpleNamespace(r1_rows=torch.zeros(2, 2)),
        _draft_lm_head=lambda: SimpleNamespace(forward=lambda hidden: hidden),
        _draw_draft=lambda *args, slot=0: torch.tensor([100 + slot], dtype=torch.int32),
    )
    ex._draft_k_with_page_room = SpecExecutor._draft_k_with_page_room.__get__(ex)
    ex._next_k = SpecExecutor._next_k.__get__(ex)
    SpecExecutor.warmup_draft(
        ex, SimpleNamespace(reqs=[req], input_ids=torch.tensor([8], dtype=torch.int32)),
        torch.tensor([9], dtype=torch.int32),
    )
    expected = min(depth, 128 - input_end + 1) if input_end <= 128 else depth
    assert req.spec_k_staged == expected
    assert len(writes) == expected + 1
    assert pool[0, req.device_len:req.device_len + expected].tolist() == list(
        range(100, 100 + expected)
    )
    # Crossing a page naturally allocates the next page. The configured depth returns;
    # the session is never switched off just because a proposal was clipped.
    cm.allocate_paged([req])
    assert ex._draft_k_with_page_room(130, ex._next_k(req)) == depth
    assert req.spec_state == "ready"


@pytest.mark.parametrize("depth,emitted", [(k, n) for k in range(1, 5) for n in range(1, k + 2)])
@pytest.mark.parametrize("input_end", [127, 128])
def test_real_graph_chain_uses_allocated_slots_for_every_acceptance(depth, emitted, input_end):
    cm, table = _owned_pages(input_end)
    req = _req(input_end - depth, depth)
    req.begin_verify(depth)
    batch = Batch(reqs=[req], phase="decode")
    batch.spec = SpecBatchInfo(scratch=None, num_input_tokens=depth + 1)
    candidates = torch.arange(20, 21 + depth, dtype=torch.int32)
    batch.input_ids = torch.cat((torch.tensor([8], dtype=torch.int32), candidates[:-1]))
    if emitted <= depth:
        batch.input_ids[emitted] = 99
    logits = torch.zeros(depth + 1, 128)
    logits.scatter_(1, candidates.long().view(-1, 1), 1)
    g = SimpleNamespace(
        stage=Mock(), graph_a=SimpleNamespace(replay=Mock()), graph_b=SimpleNamespace(replay=Mock()),
        logits=logits, dlogits=logits, dlogits2=logits[:1], cand=torch.empty_like(candidates),
        chain_tok=torch.zeros(1, dtype=torch.int32), chain_r1=torch.zeros(1, 2),
        chain_r1_out=torch.zeros(1, 2), chain_pos=torch.zeros(1, dtype=torch.int32),
        chain_out_loc=torch.zeros(1, dtype=torch.int32), md_c=None,
    )
    writes = []

    def replay_chain():
        pos = int(g.chain_pos)
        slot = int(g.chain_out_loc)
        assert slot >= 0, f"MTP wrote unallocated page at position {pos}"
        assert slot == int(table[0, pos])
        assert slot // 128 not in (cm.free_slots // 128).tolist()
        writes.append(pos)

    g.graph_c = SimpleNamespace(replay=replay_chain)
    indices = torch.zeros(512, dtype=torch.int32)
    ex = SimpleNamespace(
        engine=SimpleNamespace(
            config=SimpleNamespace(page_size=128), page_table=table, linear_state_pool=None,
            stream=None, attn_backend=SimpleNamespace(
                stage_spec=Mock(), spec_indices_buffer=lambda _: indices,
            ),
        ),
        k=depth, k_max=depth, device=torch.device("cpu"), model=SimpleNamespace(),
        token_pool=torch.zeros_like(table), graph_sets={depth: g},
        _gdn_slot_tensor=lambda *args, **kwargs: torch.tensor([0]),
        scratch=SimpleNamespace(commit=Mock(), r1_rows=torch.zeros(depth + 1, 2)),
        _draw_draft=lambda *args, slot=0, **kwargs: torch.tensor([100 + slot], dtype=torch.int32),
        ple_layer_ids=(), prefetcher=None, _cand_cpu=torch.empty_like(candidates),
        _n_cpu=torch.empty((), dtype=torch.int64),
    )
    ex._draft_k_with_page_room = SpecExecutor._draft_k_with_page_room.__get__(ex)
    ex._next_k = SpecExecutor._next_k.__get__(ex)
    output = SpecExecutor.verify_step(ex, batch, None)
    assert int(output.num_emitted_cpu) == emitted
    expected = min(depth, 128 - input_end + 1)
    assert req.spec_k_staged == expected
    assert len(writes) == expected - 1
    bonus_position = req.cached_len + emitted
    assert int(ex.token_pool[0, bonus_position]) == int(candidates[emitted - 1])
    old_device_len = req.device_len
    req.commit_verify(emitted)
    cm.truncate_paged(req, old_device_len)
    req.begin_verify(req.spec_k_staged)
    cm.allocate_paged([req])
    assert ex._draft_k_with_page_room(130, ex._next_k(req)) == depth
    # No extra reservation survives the step: all allocated pages remain unique,
    # and the untouched lifecycle returns them without a leak or duplicate free.
    allocated_end = ((req.device_len + 127) // 128) * 128
    cm._free(table[0, :allocated_end])
    cm.check_integrity()
