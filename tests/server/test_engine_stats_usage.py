"""The per-request engine-stats extension: wire round-trip and usage surfacing."""

from freetoken.message import BaseFrontendMsg, UserReply
from freetoken.message.tokenizer import BaseTokenizerMsg, DetokenizeMsg
from freetoken.server.openai_api import _usage

STATS = {
    "prefill_s": 1.234,
    "decode_s": 8.5,
    "decode_steps": 100,
    "spec_steps": 90,
    "draft_tokens": 210,
    "accepted_tokens": 151,
}


def test_detokenize_msg_engine_stats_round_trip():
    msg = DetokenizeMsg(uid=7, next_token=3, finished=True, engine_stats=dict(STATS))
    decoded = BaseTokenizerMsg.decoder(BaseTokenizerMsg.encoder(msg))
    assert isinstance(decoded, DetokenizeMsg)
    assert decoded.engine_stats == STATS


def test_detokenize_msg_engine_stats_defaults_to_none():
    msg = DetokenizeMsg(uid=7, next_token=3, finished=False)
    decoded = BaseTokenizerMsg.decoder(BaseTokenizerMsg.encoder(msg))
    assert decoded.engine_stats is None


def test_user_reply_engine_stats_round_trip():
    msg = UserReply(uid=1, incremental_output="x", finished=True, engine_stats=dict(STATS))
    decoded = BaseFrontendMsg.decoder(BaseFrontendMsg.encoder(msg))
    assert isinstance(decoded, UserReply)
    assert decoded.engine_stats == STATS


def test_usage_carries_engine_stats_extension():
    u = _usage(10, 5, 0, engine_stats=dict(STATS))
    assert u["freetoken"] == STATS
    assert u["total_tokens"] == 15


def test_usage_omits_engine_stats_when_absent():
    assert "freetoken" not in _usage(10, 5, 0)
    assert "freetoken" not in _usage(10, 5, 0, engine_stats=None)
    assert "freetoken" not in _usage(10, 5, 0, engine_stats={})
