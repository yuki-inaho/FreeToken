"""Automatic sanity checks over ``runs.jsonl``. A run failing a hard check is marked
INVALID (excluded from summary statistics by report.py); soft checks attach warnings.

Hard checks
    output-short      completion_tokens far below requested (ignore_eos runs must decode the
                      full budget). A shortfall of 1-2 tokens is a WARNING, not a failure:
                      the non-speculative decode path returns max_tokens-1 (see the report's
                      findings), which is a token-accounting quirk, not a truncated run.
    cold-cache-hit    cached_tokens > 0 on a run labelled cold
    no-first-token    stream produced no token timings
    error             the request errored

Soft checks (warnings)
    prompt-off-target         server prompt_tokens != target (exact-fit miss; reported).
                              Warm/repeat cache runs are exempt: they deliberately append a
                              per-run suffix to a fitted prompt.
    engine-client-decode-gap  engine decode_s vs client decode wall differ > 10%
    spec-accounting           completion_tokens != 1 + decode_steps + accepted (± k slack)
    throttled                 NVML reported a HARDWARE slowdown (thermal / power brake /
                              HW slowdown). SwPowerCap (0x4) alone is NOT flagged: a card
                              held at its board power limit is the expected full-load state,
                              and is reported as `power_limited` in the summary instead.
    gpu-idle                  mean GPU utilization < 10% during a decode-heavy run
    prefill-implausible       engine prefill rate off the expected band (500..20,000 tok/s),
                              computed on UNCACHED tokens so a prefix hit is not misread
    early-eos                 finish_reason != "length" on an ignore_eos full-length run

Usage: python -m benchmark.validate <results-dir>   (rewrites runs.jsonl in place, adding
``valid`` and ``flags`` to each record; idempotent)
"""

from __future__ import annotations

import json
import os
import sys

SPEC_K_SLACK = 5  # adaptive k_max + 1: the final verify may be clipped by the output budget

# NVML clock-event bits that mean the hardware is actually being held back. SwPowerCap (0x4)
# is excluded on purpose: a GPU pinned at its board power limit under full load is the normal
# operating point, not a measurement fault.
HW_THROTTLE_MASK = 0x8 | 0x20 | 0x40 | 0x80  # HwSlowdown, SwThermal, HwThermal, PowerBrake


def _throttle_flags(tele: dict) -> list[str]:
    """Hardware slowdowns are warnings; running at the board power limit is not."""
    out: list[str] = []
    reasons = tele.get("throttle_reasons") or []
    bits = 0
    for r in reasons:
        try:
            bits |= int(r, 16) if isinstance(r, str) else int(r)
        except (TypeError, ValueError):
            continue
    if bits & HW_THROTTLE_MASK:
        out.append(f"hw-throttled:{hex(bits & HW_THROTTLE_MASK)}")
    return out


def check(rec: dict) -> tuple[bool, list[str]]:
    flags: list[str] = []
    if rec.get("suite") == "concurrency":
        failed = rec.get("failed", 0)
        if failed:
            flags.append(f"failed-requests:{failed}")
        flags += _throttle_flags(rec.get("telemetry") or {})
        return True, flags

    res = rec.get("result") or {}
    if "error" in res:
        return False, [f"error:{res['error'][:120]}"]
    req_out = rec.get("requested_output_tokens")
    comp = res.get("completion_tokens")
    mode = (rec.get("extra") or {}).get("mode", "cold")

    if comp is None or res.get("ttft_s") is None:
        return False, ["no-first-token"]
    if req_out and comp < req_out:
        if comp < req_out - 2:
            return False, [f"output-short:{comp}<{req_out}"]
        flags.append(f"output-short-by-{req_out - comp}:{comp}<{req_out}")
    if mode == "cold" and res.get("cached_tokens"):
        return False, [f"cold-cache-hit:{res['cached_tokens']}"]

    target = rec.get("target_input_tokens")
    # Every cache mode except `cold` deliberately appends a per-run suffix (or a whole
    # conversation turn) to a fitted prompt, so its length is target + a known tail.
    exact_len_expected = not (rec.get("suite") == "cache" and mode != "cold")
    if target and res.get("prompt_tokens") not in (None, target) and exact_len_expected:
        flags.append(f"prompt-off-target:{res['prompt_tokens']}!={target}")
    eng = res.get("engine") or {}
    dw, ds = res.get("decode_wall_s"), eng.get("decode_s")
    if dw and ds and comp and comp > 64 and abs(dw - ds) > 0.1 * max(dw, ds):
        flags.append(f"engine-client-decode-gap:{dw}vs{ds}")
    steps, acc = eng.get("decode_steps"), eng.get("accepted_tokens")
    if steps is not None and acc is not None and comp:
        expect = 1 + steps + acc
        if abs(comp - expect) > SPEC_K_SLACK:
            flags.append(f"spec-accounting:{comp}!=1+{steps}+{acc}")
    tele = rec.get("telemetry") or {}
    flags += _throttle_flags(tele)
    if req_out and req_out >= 256 and (tele.get("gpu_util_mean") or 100) < 10:
        flags.append(f"gpu-idle:{tele.get('gpu_util_mean')}")
    pf = eng.get("prefill_s")
    uncached = (res.get("prompt_tokens") or 0) - (res.get("cached_tokens") or 0)
    if pf and target and target >= 4096 and uncached >= 4096:
        rate = uncached / pf
        if not (500 <= rate <= 20000):
            flags.append(f"prefill-implausible:{rate:.0f}tok/s")
    if req_out and req_out >= 64 and res.get("finish_reason") not in (None, "length"):
        flags.append(f"early-eos:{res.get('finish_reason')}")
    return True, flags


def main() -> None:
    out = sys.argv[1]
    path = os.path.join(out, "runs.jsonl")
    recs = [json.loads(l) for l in open(path) if l.strip()]
    n_bad = 0
    for rec in recs:
        ok, flags = check(rec)
        rec["valid"] = ok
        rec["flags"] = flags
        if not ok:
            n_bad += 1
        if flags or not ok:
            print(f"{'INVALID' if not ok else 'warn   '} {rec['key']}: {flags}")
    with open(path, "w") as f:
        for rec in recs:
            f.write(json.dumps(rec) + "\n")
    print(f"validated {len(recs)} records: {n_bad} INVALID, "
          f"{sum(1 for r in recs if r['flags'])} with warnings")


if __name__ == "__main__":
    main()
