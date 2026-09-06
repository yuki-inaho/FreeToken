"""Opt-in HTTP regression for MTP at a real server's context boundary.

Start a dedicated Qwen server with MTP enabled and a 4096-token context limit, then set
FREETOKEN_BOUNDARY_TEST_URL=http://127.0.0.1:1930. These requests intentionally
consume the full context; never point the test at a shared production server.
"""

from __future__ import annotations

import json
import os
from urllib.request import Request, urlopen

import pytest


BASE = os.environ.get("FREETOKEN_BOUNDARY_TEST_URL", "").rstrip("/")
LIMIT = int(os.environ.get("FREETOKEN_BOUNDARY_TEST_CONTEXT", "4096"))
pytestmark = pytest.mark.skipif(not BASE, reason="requires a dedicated boundary-test server")


def _request(path, payload=None):
    body = json.dumps(payload).encode() if payload is not None else None
    request = Request(BASE + path, data=body, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=45) as response:
        return json.load(response)


@pytest.fixture(scope="module")
def model():
    assert _request("/health")["status"] == "ok"
    models = _request("/v1/models")["data"]
    assert len(models) == 1
    assert models[0]["max_model_len"] == LIMIT, "server must use the dedicated test context limit"
    return models[0]["id"]


@pytest.mark.parametrize("temperature", [1.0, 0.0])
@pytest.mark.parametrize("remaining", [64, 63, 16, 8, 5, 4, 3, 2, 1])
def test_full_context_finishes_and_server_remains_healthy(model, remaining, temperature):
    # " hello" is one token in this Qwen checkpoint. Usage independently checks
    # that the served tokenizer agrees, so the test cannot silently miss the edge.
    prompt_length = LIMIT - remaining
    result = _request("/v1/completions", {
        "model": model,
        "prompt": " hello" * prompt_length,
        "max_tokens": remaining + 16,
        "temperature": temperature,
        "top_k": 20,
        "top_p": 0.95,
        "ignore_eos": True,
        "stream": False,
    })
    assert result["usage"]["prompt_tokens"] == prompt_length
    assert result["usage"]["completion_tokens"] == remaining
    assert result["choices"][0]["finish_reason"] == "length"
    # The completions endpoint exposes token usage, but not chat's per-request
    # engine counters. The launch manifest/log must record that MTP is enabled.
    assert _request("/health")["status"] == "ok"


def test_short_request_after_boundary_requests(model):
    result = _request("/v1/chat/completions", {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with READY."}],
        "reasoning_effort": "none",
        "temperature": 0,
        "max_tokens": 16,
    })
    assert result["choices"][0]["message"]["content"].strip()
    assert _request("/health")["status"] == "ok"


@pytest.mark.parametrize("prompt_length", [125, 126, 127, 128])
@pytest.mark.parametrize("temperature", [1.0, 0.0])
def test_draft_page_crossings_leave_later_requests_usable(model, prompt_length, temperature):
    result = _request("/v1/completions", {
        "model": model,
        "prompt": " hello" * prompt_length,
        "max_tokens": 96,
        "temperature": temperature,
        "top_k": 20,
        "top_p": 0.95,
        "ignore_eos": True,
    })
    assert result["usage"]["prompt_tokens"] == prompt_length
    assert result["usage"]["completion_tokens"] == 96
    assert result["choices"][0]["finish_reason"] == "length"
    assert _request("/health")["status"] == "ok"
