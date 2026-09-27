"""Measurement driver: runs one suite against a live FreeToken server and appends
machine-readable records to ``<out>/runs.jsonl`` (resumable: existing keys are skipped).

Suites (``--suite``):
    primary       253,952-in / 4,096-out, C=1, cold prefix, N runs (default 10)
    context       context-length sweep x 4,096-out (default 3 runs each)
    prefill       cold-prefill sweep (max_tokens=8; raw vs effective prefill tok/s)
    cache         cold prefix vs warm prefix vs repeated request (labelled separately)
    mtp           the speculative side of the MTP ablation (same workload; run it once on a
                  spec-enabled boot with --variant mtp_on and once on a spec-off boot with
                  --variant mtp_off; compare.py joins them)
    concurrency   fixed 16-request workload at C in {1,2,4,8} (server must allow it;
                  boot with FREETOKEN_MAX_RUNNING_REQUESTS=8)
    smoke         short versions of everything, for validating accounting before long runs
    mixed         code/English/Japanese x context x greedy/sampled, scored separately

The driver never starts or stops servers; ``run_all.sh`` sequences boots and suites.
Timing conventions (documented in the report): TTFT and decode wall are measured
client-side with a monotonic clock from the SSE stream; sustained decode tok/s =
(completion_tokens - 1) / (t_last_token - t_first_token); the engine additionally reports
in-engine prefill/decode wall time and speculative draft/accept counters in
``usage.freetoken`` (added for this benchmark).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid
from typing import Any

from .telemetry import TelemetrySampler
from .workloads import TokenFitter, load_tokenizer, make_counter

DEFAULT_BASE = os.environ.get("FREETOKEN_API_BASE", "http://127.0.0.1:1919/v1")
CONTEXT_LADDER = [4096, 16384, 32768, 65536, 131072, 196608, 253952]
PRIMARY_INPUT = 253952
PRIMARY_OUTPUT = 4096


# ------------------------------------------------------------------ HTTP client
def _get(url: str, timeout: float = 10.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _post(url: str, payload: dict, timeout: float = 30.0) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _engine_from_timings(t: dict | None) -> dict | None:
    """Normalize llama.cpp's `timings` onto the engine-stats schema this suite reports.

    `predicted_n` counts emitted tokens, not decode steps; with speculation a step emits
    several, so steps are derived as emitted - accepted (exact when every accepted draft
    rides a verify step).
    """
    if not t:
        return None
    emitted = t.get("predicted_n") or 0
    accepted = t.get("draft_n_accepted") or 0
    return {
        "prefill_s": round(t["prompt_ms"] / 1000.0, 4) if t.get("prompt_ms") else None,
        "decode_s": round(t["predicted_ms"] / 1000.0, 4) if t.get("predicted_ms") else None,
        "decode_steps": max(emitted - accepted, 0) or None,
        "spec_steps": None,
        "draft_tokens": t.get("draft_n") or 0,
        "accepted_tokens": accepted,
        "cache_n": t.get("cache_n"),
    }


def stream_chat(base: str, model: str, messages: list[dict], max_tokens: int,
                temperature: float = 0.0, ignore_eos: bool = True,
                timeout: float = 900.0, extra_body: dict | None = None,
                capture_output: bool = False) -> dict[str, Any]:
    """One streamed chat completion; returns client timings + usage + output hash.

    Engine-side counters come from whichever extension the server offers:
    FreeToken's ``usage.freetoken`` or llama.cpp's ``timings`` block on the final
    delta. Both are normalized to the same schema so one harness compares them.
    """
    payload = {
        "model": model, "messages": messages, "temperature": temperature,
        "max_tokens": max_tokens, "ignore_eos": ignore_eos,
        "stream": True, "stream_options": {"include_usage": True},
    }
    if extra_body:
        payload.update(extra_body)
    req = urllib.request.Request(f"{base}/chat/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t_send = time.monotonic()
    t_first = t_last = None
    usage = None
    timings = None
    finish_reason = None
    n_events = 0
    out_sha = hashlib.sha256()
    out_len = 0
    head = []
    captured_choices = [] if capture_output else None
    captured_text = [] if capture_output else None
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            ev = json.loads(line[6:])
            if ev.get("error"):
                raise RuntimeError(f"server error: {ev['error']}")
            for ch in ev.get("choices") or []:
                delta = ch.get("delta") or {}
                piece = delta.get("content") or delta.get("reasoning_content")
                if piece:
                    now = time.monotonic()
                    if t_first is None:
                        t_first = now
                    t_last = now
                    n_events += 1
                    out_sha.update(piece.encode())
                    out_len += len(piece)
                    if len(head) < 6:
                        head.append(piece)
                    if captured_text is not None:
                        captured_text.append(piece)
                if captured_choices is not None:
                    captured_choices.append(ch)
                if ch.get("finish_reason"):
                    finish_reason = ch["finish_reason"]
            if ev.get("usage"):
                usage = ev["usage"]
            if ev.get("timings"):
                timings = ev["timings"]
    t_done = time.monotonic()
    if usage is None:
        raise RuntimeError("stream ended without a usage block")
    comp = usage.get("completion_tokens", 0)
    ttft = (t_first - t_send) if t_first else None
    decode_wall = (t_last - t_first) if (t_first and t_last and t_last > t_first) else None
    result = {
        "t_send_wall": time.time() - (t_done - t_send),
        "ttft_s": round(ttft, 4) if ttft else None,
        "decode_wall_s": round(decode_wall, 4) if decode_wall else None,
        "e2e_s": round(t_done - t_send, 4),
        "decode_tps": round((comp - 1) / decode_wall, 2) if decode_wall and comp > 1 else None,
        "tpot_ms": round(decode_wall / (comp - 1) * 1000, 3) if decode_wall and comp > 1 else None,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": comp,
        "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
        "server_response_s": usage.get("response_time"),
        "server_tps": usage.get("response_token/s"),
        "engine": usage.get("freetoken") or _engine_from_timings(timings),
        "timings": timings,
        "finish_reason": finish_reason,
        "stream_events": n_events,
        "output_sha256": out_sha.hexdigest(),
        "output_chars": out_len,
        "output_head": "".join(head)[:160],
    }
    if captured_choices is not None:
        # Keep the existing hash/text selection (content or reasoning per delta),
        # while also retaining both channels and every raw choice delta losslessly.
        result["_captured_output"] = {
            "text": "".join(captured_text),
            "content": "".join((ch.get("delta") or {}).get("content") or "" for ch in captured_choices),
            "reasoning_content": "".join((ch.get("delta") or {}).get("reasoning_content") or ""
                                         for ch in captured_choices),
            "deltas": captured_choices,
        }
    return result


# ------------------------------------------------------------------ server info
def server_info(base: str) -> dict[str, Any]:
    root = base.rsplit("/v1", 1)[0]
    info: dict[str, Any] = {}
    for key, url in (("health", f"{root}/health"), ("stats", f"{base}/stats"),
                     ("cache_status", f"{base}/cache/status"), ("models", f"{base}/models")):
        try:
            info[key] = _get(url)
        except Exception as e:
            info[key] = {"error": str(e)}
    try:
        pid = subprocess.run(["pgrep", "-f", "ft serve --model-path"], capture_output=True,
                             text=True).stdout.split()
        if pid:
            with open(f"/proc/{pid[0]}/cmdline") as f:
                info["server_cmdline"] = f.read().replace("\0", " ").strip()
            with open(f"/proc/{pid[0]}/environ", "rb") as f:
                env = f.read().decode(errors="replace").split("\0")
            info["server_env"] = sorted(e for e in env if e.startswith(("FREETOKEN_", "PYTHONPATH=")))
    except Exception as e:
        info["server_cmdline_error"] = str(e)
    return info


def geometry(base: str) -> dict[str, Any]:
    """Live pool geometry, recorded per run so an elastic KV grow/shrink between runs is
    visible in the data (expert slots trade against KV tokens on this server)."""
    try:
        g = (_get(f"{base}/cache/status", timeout=5) or {}).get("geometry") or {}
    except Exception as e:
        return {"error": str(e)}
    return {k: g.get(k) for k in ("num_pages", "page_size", "moe_cache_size",
                                  "num_mamba_slots", "num_experts", "cache_budget_bytes")}


def flush_prefix_cache(base: str) -> dict[str, Any]:
    """De-facto radix flush: an idle-time rebuild at the current KV geometry."""
    geo = _get(f"{base}/cache/status").get("geometry") or {}
    n = geo.get("num_pages")
    if not n:
        return {"skipped": "no geometry"}
    return _post(f"{base}/cache/rebuild", {"num_pages": n, "timeout": 300.0}, timeout=330.0)


def calibrate_overhead(base: str, model: str, count) -> int:
    """Chat-template token overhead: server prompt_tokens minus content tokens."""
    text = "Calibration message: reply with one word."
    r = stream_chat(base, model, [{"role": "user", "content": text}], max_tokens=2)
    return r["prompt_tokens"] - count(text)


# ------------------------------------------------------------------ run plumbing
class Runner:
    def __init__(self, args) -> None:
        self.base = args.base
        self.out = args.out
        os.makedirs(self.out, exist_ok=True)
        os.makedirs(os.path.join(self.out, "telemetry"), exist_ok=True)
        self.runs_path = os.path.join(self.out, "runs.jsonl")
        self.done: set[str] = set()
        if os.path.exists(self.runs_path):
            with open(self.runs_path) as f:
                for line in f:
                    try:
                        self.done.add(json.loads(line)["key"])
                    except Exception:
                        pass
        self.variant = args.variant
        self.server = args.server
        self.capture_output = getattr(args, "capture_output", False)
        self.model = args.model
        if self.model in (None, "auto"):
            self.model = _get(f"{self.base}/models")["data"][0]["id"]
        self.tok = load_tokenizer(args.model_path)
        self.count = make_counter(self.tok)
        self.overhead = calibrate_overhead(self.base, self.model, self.count)
        self.fitter = TokenFitter(self.count, self.overhead)
        # Self-correcting per-target offset: if the server's count of a built prompt is off
        # by a constant (template edge merges), later builds for that target compensate.
        self.target_bias: dict[int, int] = {}
        self.stats_url = f"{self.base}/stats"
        try:
            self.server_instance = _get(self.base.rsplit("/v1", 1)[0] + "/health").get("instance_id")
        except Exception:
            self.server_instance = None

    def _body(self, *, cache: bool) -> dict | None:
        """Server-specific request fields.

        llama.cpp reuses a slot's prompt prefix unless told otherwise, which would silently
        turn a cold run warm; the cache suite's warm modes need the opposite. FreeToken
        ignores the unknown field (its request models allow extras), so one code path serves
        both servers.
        """
        return {"cache_prompt": cache} if self.server == "llamacpp" else None

    # ---------------- record helpers
    def _store_output(self, key: str, result: dict | None) -> None:
        if not result or "_captured_output" not in result:
            return
        capture = result.pop("_captured_output")
        directory = os.path.join(self.out, "outputs")
        os.makedirs(directory, exist_ok=True)
        # Hash the full key to avoid sanitized-path collisions; a unique suffix
        # preserves orphaned captures if a run stops before appending its JSONL row.
        name = f"{hashlib.sha256(key.encode()).hexdigest()}-{uuid.uuid4().hex}.json"
        relative = os.path.join("outputs", name)
        artifact = {"schema_version": 1, "record_key": key,
                    "output_sha256": result["output_sha256"], "output_chars": result["output_chars"],
                    **capture}
        with open(os.path.join(self.out, relative), "x", encoding="utf-8") as f:
            json.dump(artifact, f, ensure_ascii=False)
            f.write("\n")
        result["output_artifact"] = relative

    def _append(self, rec: dict) -> None:
        self._store_output(rec["key"], rec.get("result"))
        for index, result in enumerate(rec.get("requests") or []):
            self._store_output(f"{rec['key']}/request/{index}", result)
        with open(self.runs_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        self.done.add(rec["key"])

    def _prompt(self, family: str, target: int, seed: int, nonce: str, **kwargs):
        bias = self.target_bias.get(target, 0)
        return self.fitter.build(family, target - bias, seed, nonce, **kwargs)

    def _learn_bias(self, target: int, expected: int, actual: int) -> None:
        # ``expected`` already reflects the previous correction. The difference
        # is the server's template offset itself, not an increment to that offset.
        self.target_bias[target] = actual - expected

    def measure(self, key: str, family: str, target: int, out_tokens: int, seed: int,
                temperature: float = 0.0, extra: dict | None = None,
                telemetry: bool = True, sampling: dict | None = None,
                task: str | None = None, nonce: str | None = None,
                server_log: str | None = None) -> dict | None:
        if key in self.done:
            print(f"skip (done): {key}", flush=True)
            return None
        nonce = nonce or f"{key}-{uuid.uuid4().hex[:8]}"
        t_build = time.monotonic()
        prompt = self._prompt(family, target, seed, nonce, **({"task": task} if task else {}))
        build_s = time.monotonic() - t_build
        sampler = None
        if telemetry:
            tp = os.path.join(self.out, "telemetry", re.sub(r"[^\w.-]", "_", key) + ".jsonl")
            sampler = TelemetrySampler(tp, stats_url=self.stats_url).start()
            sampler.mark("send")
        geo_before = geometry(self.base)
        log_offset = os.path.getsize(server_log) if server_log else None
        t0 = time.time()
        try:
            res = stream_chat(self.base, self.model, prompt.messages(), out_tokens,
                              temperature=temperature,
                              extra_body={**(self._body(cache=False) or {}), **(sampling or {})},
                              timeout=max(900.0, target / 800 + out_tokens / 20),
                              capture_output=self.capture_output)
        except Exception as e:
            res = {"error": f"{type(e).__name__}: {e}"}
        if sampler:
            sampler.mark("done")
            tele = sampler.stop()
        else:
            tele = None
        rec = {
            "key": key, "variant": self.variant, "suite": key.split("/")[0],
            "family": family, "target_input_tokens": target, "requested_output_tokens": out_tokens,
            "temperature": temperature, "seed": seed, "started_wall": t0,
            "prompt": prompt.record(), "prompt_build_s": round(build_s, 2),
            "result": res, "telemetry": tele, "extra": extra or {},
            "geometry_before": geo_before, "geometry_after": geometry(self.base),
            "server_instance": self.server_instance,
        }
        if sampling is not None:
            rec["sampling"] = {"temperature": temperature, **sampling}
        if server_log:
            from .mixed import read_moe_windows
            rec["moe_windows"] = read_moe_windows(server_log, log_offset)
        if "prompt_tokens" in res and res.get("prompt_tokens"):
            self._learn_bias(target, prompt.expected_prompt_tokens, res["prompt_tokens"])
        self._append(rec)
        r = rec["result"]
        if "error" in r:
            print(f"RUN {key}: ERROR {r['error']}", flush=True)
        else:
            eng = r.get("engine") or {}
            print(
                f"RUN {key}: in={r['prompt_tokens']} out={r['completion_tokens']} "
                f"ttft={r['ttft_s']}s decode={r['decode_tps']} tok/s tpot={r['tpot_ms']}ms "
                f"prefill_s={eng.get('prefill_s')} steps={eng.get('decode_steps')} "
                f"drafts={eng.get('draft_tokens')} acc={eng.get('accepted_tokens')} "
                f"cached={r['cached_tokens']}", flush=True)
        return rec

    # ---------------- suites
    def suite_primary(self, runs: int, in_tokens: int = PRIMARY_INPUT,
                      out_tokens: int = PRIMARY_OUTPUT) -> None:
        for i in range(runs):
            self.measure(f"primary/{self.variant}/{in_tokens}/{i}", "realistic", in_tokens,
                         out_tokens, seed=1000 + i)
            time.sleep(2)

    def suite_context(self, runs: int, out_tokens: int = PRIMARY_OUTPUT) -> None:
        for ctx in CONTEXT_LADDER:
            for i in range(runs):
                self.measure(f"context/{self.variant}/{ctx}/{i}", "realistic", ctx,
                             out_tokens, seed=2000 + i)
                time.sleep(2)

    def suite_prefill(self, runs: int) -> None:
        for ctx in CONTEXT_LADDER:
            for i in range(runs):
                self.measure(f"prefill/{self.variant}/{ctx}/{i}", "synthetic", ctx,
                             8, seed=3000 + i)
                time.sleep(2)

    def _cache_run(self, key: str, prompt, text: str, out_tokens: int, mode: str,
                   ctx: int, seed: int) -> dict:
        """One labelled cache-suite request (the prompt text is supplied, not re-fitted)."""
        if key in self.done:
            print(f"skip (done): {key}", flush=True)
            return {}
        tp = os.path.join(self.out, "telemetry", re.sub(r"[^\w.-]", "_", key) + ".jsonl")
        sampler = TelemetrySampler(tp, stats_url=self.stats_url).start()
        geo_before = geometry(self.base)
        try:
            res = stream_chat(self.base, self.model, [{"role": "user", "content": text}],
                              out_tokens, timeout=900,
                              extra_body=self._body(cache=mode != "cold"),
                              capture_output=self.capture_output)
        except Exception as e:
            res = {"error": f"{type(e).__name__}: {e}"}
        tele = sampler.stop()
        self._append({
            "key": key, "variant": self.variant, "suite": "cache", "family": "realistic",
            "target_input_tokens": ctx, "requested_output_tokens": out_tokens,
            "temperature": 0.0, "seed": seed, "prompt": prompt.record(), "result": res,
            "telemetry": tele, "extra": {"mode": mode},
            "geometry_before": geo_before, "geometry_after": geometry(self.base),
        })
        print(f"RUN {key}: prompt={res.get('prompt_tokens')} cached={res.get('cached_tokens')} "
              f"ttft={res.get('ttft_s')} decode={res.get('decode_tps')}", flush=True)
        return res

    def suite_cache(self, runs: int, ctx: int = 32768, out_tokens: int = 512) -> None:
        """Four strictly separated categories, never mixed in a statistic:

        A ``cold``          a document no request has seen (unique nonce) -- no reuse possible.
        B ``extend``        multi-turn chat: each turn re-sends the whole previous transcript
                            plus a follow-up, i.e. it EXTENDS a completed request. This is the
                            case prefix caching actually serves.
        C ``repeat``        a byte-identical resend.
        D ``shared_prefix`` the same long document with a DIFFERENT question each time, so each
                            request diverges in the middle of a stored entry. Measured because
                            it is the case people assume is cached -- on this hybrid
                            (GatedDeltaNet + sparse-attention) model it is not: reuse needs a
                            recurrent-state snapshot, which exists only at completed-request
                            boundaries.
        """
        try:
            print("flush before cache suite:", flush_prefix_cache(self.base), flush=True)
        except Exception as e:
            print(f"flush failed ({e}); unique nonces still guarantee cold prefixes", flush=True)

        for i in range(runs):
            self.measure(f"cache/{self.variant}/cold/{ctx}/{i}", "realistic", ctx, out_tokens,
                         seed=4000 + i, extra={"mode": "cold"})

        # B: multi-turn extension of one growing transcript.
        base = self._prompt("realistic", ctx, 4100, f"cache-extend-{uuid.uuid4().hex[:8]}")
        transcript = base.text
        for i in range(runs + 1):
            res = self._cache_run(f"cache/{self.variant}/extend/{ctx}/{i}", base, transcript,
                                  out_tokens, "extend_fill" if i == 0 else "extend", ctx, 4100)
            reply = (res or {}).get("output_head", "")
            transcript = (f"{transcript}\n\n[assistant] {reply}\n\n"
                          f"[user] Follow-up {i + 1}: continue the report with one more section.")

        # C: byte-identical resend.
        rep = self._prompt("realistic", ctx, 4200, f"cache-repeat-{uuid.uuid4().hex[:8]}")
        for i in range(runs + 1):
            self._cache_run(f"cache/{self.variant}/repeat/{ctx}/{i}", rep, rep.text, out_tokens,
                            "repeat_fill" if i == 0 else "repeat", ctx, 4200)

        # D: same document, different question -- diverges mid-entry.
        shared = self._prompt("realistic", ctx, 4300, f"cache-shared-{uuid.uuid4().hex[:8]}")
        for i in range(runs + 1):
            text = (shared.text +
                    f"\n\nQuestion variant {i}: also mention the number {i * 7 + 1}.")
            self._cache_run(f"cache/{self.variant}/shared_prefix/{ctx}/{i}", shared, text,
                            out_tokens, "shared_prefix_fill" if i == 0 else "shared_prefix",
                            ctx, 4300)

    def suite_mtp(self, runs: int) -> None:
        """The per-variant half of the MTP ablation (identical prompts via fixed seeds)."""
        for ctx, out_tokens in ((4096, 1024), (32768, 1024)):
            for i in range(runs):
                self.measure(f"mtp/{self.variant}/{ctx}/{i}", "realistic", ctx, out_tokens,
                             seed=5000 + i)
                time.sleep(2)

    def suite_concurrency(self, levels=(1, 2, 4, 8), n_requests: int = 16,
                          ctx: int = 4096, out_tokens: int = 512) -> None:
        for c in levels:
            key = f"concurrency/{self.variant}/C{c}"
            if key in self.done:
                print(f"skip (done): {key}", flush=True)
                continue
            prompts = [self._prompt("realistic", ctx, 6000 + j, f"{key}-{j}-{uuid.uuid4().hex[:6]}")
                       for j in range(n_requests)]
            results: list[dict | None] = [None] * n_requests
            times: list[tuple[float, float]] = [(0.0, 0.0)] * n_requests
            idx_lock = threading.Lock()
            next_idx = [0]

            def worker():
                while True:
                    with idx_lock:
                        j = next_idx[0]
                        if j >= n_requests:
                            return
                        next_idx[0] += 1
                    t0 = time.monotonic()
                    try:
                        results[j] = stream_chat(self.base, self.model, prompts[j].messages(),
                                                 out_tokens, timeout=1800,
                                                 extra_body=self._body(cache=False),
                                                 capture_output=self.capture_output)
                    except Exception as e:
                        results[j] = {"error": f"{type(e).__name__}: {e}"}
                    times[j] = (t0, time.monotonic())

            tp = os.path.join(self.out, "telemetry", re.sub(r"[^\w.-]", "_", key) + ".jsonl")
            sampler = TelemetrySampler(tp, stats_url=self.stats_url).start()
            t_start = time.monotonic()
            threads = [threading.Thread(target=worker) for _ in range(c)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            t_end = time.monotonic()
            tele = sampler.stop()
            ok = [r for r in results if r and "error" not in r]
            total_out = sum(r["completion_tokens"] for r in ok)

            def pct(vals, q):
                if not vals:
                    return None
                vs = sorted(vals)
                return round(vs[min(len(vs) - 1, int(q * len(vs)))], 4)

            ttfts = [r["ttft_s"] for r in ok if r.get("ttft_s")]
            tpots = [r["tpot_ms"] for r in ok if r.get("tpot_ms")]
            lats = [round(e - s, 3) for (s, e), r in zip(times, results) if r and "error" not in r]
            rec = {
                "key": key, "variant": self.variant, "suite": "concurrency", "level": c,
                "n_requests": n_requests, "target_input_tokens": ctx,
                "requested_output_tokens": out_tokens,
                "wall_s": round(t_end - t_start, 3),
                "aggregate_tps": round(total_out / (t_end - t_start), 2),
                "per_request_tps_mean": round(statistics.mean(r["decode_tps"] for r in ok if r.get("decode_tps")), 2) if ok else None,
                "ttft_p50": pct(ttfts, 0.50), "ttft_p95": pct(ttfts, 0.95), "ttft_p99": pct(ttfts, 0.99),
                "tpot_p50": pct(tpots, 0.50), "tpot_p95": pct(tpots, 0.95), "tpot_p99": pct(tpots, 0.99),
                "latency_p50": pct(lats, 0.50), "latency_p95": pct(lats, 0.95), "latency_p99": pct(lats, 0.99),
                "completed": len(ok), "failed": n_requests - len(ok),
                "spec_steps_total": sum((r.get("engine") or {}).get("spec_steps", 0) for r in ok),
                "telemetry": tele,
                "geometry_after": geometry(self.base),
                "requests": results,
                "prompts": [p.record() for p in prompts],
            }
            self._append(rec)
            print(f"RUN {key}: aggregate={rec['aggregate_tps']} tok/s "
                  f"per-req={rec['per_request_tps_mean']} ttft_p95={rec['ttft_p95']} "
                  f"completed={len(ok)}/{n_requests} spec_steps={rec['spec_steps_total']}",
                  flush=True)

    def suite_smoke(self) -> None:
        self.measure(f"smoke/{self.variant}/4096/0", "realistic", 4096, 256, seed=1)
        self.measure(f"smoke/{self.variant}/4096/1", "synthetic", 4096, 64, seed=2)
        self.measure(f"smoke/{self.variant}/16384/0", "realistic", 16384, 256, seed=3)


def warm_up(runner: Runner) -> None:
    """JIT/graph warm-up (not measured): one short and one chunk-boundary-crossing request."""
    for target, out_tokens in ((512, 64), (9000, 64)):
        nonce = f"warmup-{uuid.uuid4().hex[:8]}"
        p = runner._prompt("synthetic", target, 99, nonce)
        r = stream_chat(runner.base, runner.model, p.messages(), out_tokens, timeout=600)
        print(f"warm-up {target}: in={r['prompt_tokens']} ttft={r['ttft_s']}s", flush=True)


def collect_metadata(args, runner: Runner) -> dict[str, Any]:
    def sh(cmd: str) -> str:
        try:
            return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                                  timeout=30).stdout.strip()
        except Exception as e:
            return f"error: {e}"

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    meta = {
        "generated_wall": time.time(),
        "generated_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "gpu": {
            "smi": sh("nvidia-smi --query-gpu=name,vbios_version,driver_version,memory.total,"
                      "power.limit,power.max_limit,clocks.max.sm,clocks.max.memory,"
                      "pcie.link.gen.current,pcie.link.width.current --format=csv,noheader"),
        },
        "cpu": {
            "model": sh("lscpu | grep 'Model name' | cut -d: -f2").strip(),
            "cores": sh("lscpu | grep '^Core(s) per socket' | cut -d: -f2").strip(),
            "threads": os.cpu_count(),
            "numa": sh("lscpu | grep 'NUMA node(s)' | cut -d: -f2").strip(),
        },
        "ram": {
            "total_gib": round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30),
            "dimms": sh("udevadm info -q property -p /sys/devices/virtual/dmi/id 2>/dev/null | "
                        "grep -E 'MEMORY_DEVICE_[0-9]+_(SIZE|SPEED_MTS|CONFIGURED_SPEED_MTS|MANUFACTURER)' | sort"),
            "channels_note": "4 of 8 DIMM slots populated (96 GiB DDR5-5600 RDIMM each, running at 5200 MT/s)",
        },
        "board": sh("cat /sys/devices/virtual/dmi/id/board_vendor /sys/devices/virtual/dmi/id/board_name 2>/dev/null | tr '\\n' ' '"),
        "os": {"platform": platform.platform(), "kernel": platform.release(),
               "distro": sh("grep PRETTY_NAME /etc/os-release | cut -d= -f2")},
        "python": platform.python_version(),
        "torch": sh(f"{os.environ.get('FREETOKEN_VENV', '/home/nuko/.freetoken/venv')}/bin/python -c 'import torch,triton;print(torch.__version__, triton.__version__)'"),
        "git": {
            "commit": sh(f"git -C '{repo}' rev-parse HEAD"),
            "dirty_files": len(sh(f"git -C '{repo}' status --porcelain").splitlines()),
            "diff_sha256": hashlib.sha256(
                subprocess.run(["git", "-C", repo, "diff"], capture_output=True).stdout
            ).hexdigest(),
        },
        "model_path": args.model_path,
        "checkpoint_bytes": sh(f"du -sb '{args.model_path}' | cut -f1"),
        "server": server_info(args.base),
        "calibrated_template_overhead_tokens": runner.overhead,
        "argv": vars(args),
    }
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", required=True,
                    choices=["primary", "context", "prefill", "cache", "mtp", "concurrency",
                             "smoke", "warmup", "mixed"])
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--model", default="auto")
    ap.add_argument("--model-path", default=os.environ.get(
        "FREETOKEN_MODEL_PATH", os.path.expanduser("~/.freetoken/models/Qwen3.8-Flash-Next-NVFP4")))
    ap.add_argument("--out", required=True)
    ap.add_argument("--variant", default="custom",
                    help="configuration label baked into record keys (custom / mtp_on / "
                         "mtp_off / upstream / concurrency8 ...)")
    ap.add_argument("--runs", type=int, default=None)
    ap.add_argument("--levels", default="1,2,4,8")
    ap.add_argument("--no-warmup", action="store_true")
    ap.add_argument("--capture-output", action="store_true",
                    help="save complete successful measured outputs and stream deltas under OUT/outputs; default off")
    ap.add_argument("--contexts", default="4096,32768,131072,253952",
                    help="comma-separated input token counts for --suite mixed")
    ap.add_argument("--kinds", default="code,english,japanese", help="mixed content families")
    ap.add_argument("--sampling-modes", default="greedy,sampled", help="mixed sampling modes")
    ap.add_argument("--output-tokens", type=int, default=4096, help="mixed generation length")
    ap.add_argument("--boot-id", default="1", help="mixed repetition/boot identity")
    ap.add_argument("--server-log", help="read existing MoE diagnostic windows; no extra engine sync")
    ap.add_argument("--server", default="freetoken", choices=["freetoken", "llamacpp"],
                    help="which engine is being measured; selects server-specific request "
                         "fields (llama.cpp needs cache_prompt=false for cold-prefix runs)")
    args = ap.parse_args()

    runner = Runner(args)
    print(f"model={runner.model} base={args.base} variant={args.variant} "
          f"template_overhead={runner.overhead}", flush=True)
    boot_suffix = f"-boot{args.boot_id}" if args.suite == "mixed" else ""
    meta_path = os.path.join(args.out, f"meta-{args.suite}-{args.variant}{boot_suffix}.json")
    with open(meta_path, "w") as f:
        json.dump(collect_metadata(args, runner), f, indent=1)
    print(f"metadata -> {meta_path}", flush=True)
    if args.suite == "warmup":
        warm_up(runner)
        return
    if not args.no_warmup:
        warm_up(runner)
    if args.suite == "primary":
        runner.suite_primary(args.runs or 10)
    elif args.suite == "context":
        runner.suite_context(args.runs or 3)
    elif args.suite == "prefill":
        runner.suite_prefill(args.runs or 3)
    elif args.suite == "cache":
        runner.suite_cache(args.runs or 3)
    elif args.suite == "mtp":
        runner.suite_mtp(args.runs or 3)
    elif args.suite == "concurrency":
        runner.suite_concurrency(tuple(int(x) for x in args.levels.split(",")))
    elif args.suite == "smoke":
        runner.suite_smoke()
    elif args.suite == "mixed":
        from .mixed import run_mixed
        run_mixed(runner, args)
    print("suite complete", flush=True)


if __name__ == "__main__":
    main()
