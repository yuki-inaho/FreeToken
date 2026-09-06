"""Compare fixed MTP depths and the adaptive control using existing mixed workloads.

Run with the serving Python environment and an immutable, space-free source root
containing the compiled native extensions. Only this invocation's servers are stopped.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

from .benchmark import Runner, collect_metadata, warm_up, _get
from .mixed import run_mixed


ARMS = {f"fixed{k}": (k, k) for k in range(1, 5)} | {"adaptive2-3": (2, 3)}
SCREEN_ORDER = "fixed2,fixed4,adaptive2-3,fixed1,fixed3"


def utcnow():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save(path, data):
    Path(path).write_text(json.dumps(data, indent=2) + "\n")


def active_group_members(pgid):
    result = []
    for item in Path("/proc").iterdir():
        if not item.name.isdigit():
            continue
        try:
            fields = (item / "stat").read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) == pgid and fields[0] != "Z":
                result.append(int(item.name))
        except (OSError, ValueError, IndexError):
            pass
    return result


def stop_owned_server(proc):
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + 45
    while active_group_members(proc.pid) and time.monotonic() < deadline:
        time.sleep(1)
    if active_group_members(proc.pid):
        os.killpg(proc.pid, signal.SIGKILL)
    proc.wait(timeout=15)
    # The driver's memory release can outlive the Python process. Give it time
    # before the next model boot; this never kills another GPU process.
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=15, check=True,
            )
            if not result.stdout.strip():
                return
        except subprocess.TimeoutExpired:
            pass
        time.sleep(2)
    raise RuntimeError("GPU compute processes remain after stopping the owned server")


def check_port(port):
    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1", port)) == 0:
            raise RuntimeError(f"Port {port} is occupied; refusing to interrupt it")
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=15, check=True,
    )
    if result.stdout.strip():
        raise RuntimeError(f"GPU already has compute processes: {result.stdout.strip()}")


def wait_ready(proc, base, timeout=600):
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"Server exited with {proc.returncode}; inspect its log")
        try:
            if _get(base.rsplit("/v1", 1)[0] + "/health", timeout=3).get("status") == "ok":
                return
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError("Server did not become healthy within ten minutes")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arms", default=SCREEN_ORDER)
    parser.add_argument("--boot-id", default="screen1")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--context", type=int, default=131072)
    parser.add_argument("--output-tokens", type=int, default=1024)
    parser.add_argument("--port", type=int, default=1930)
    parser.add_argument("--quality-arms", default="", help="Comma-separated arms to run historical gates on")
    parser.add_argument("--quality-only", action="store_true")
    parser.add_argument("--model-path", default="/media/nuko/New Volume/Qwen3.8-Flash-Next-NVFP4")
    args = parser.parse_args()
    arms = args.arms.split(",")
    quality = set(filter(None, args.quality_arms.split(",")))
    if len(arms) != len(set(arms)) or any(a not in ARMS for a in arms) or not quality <= set(arms):
        parser.error("Invalid or duplicated arms")
    if args.quality_only and quality != set(arms):
        parser.error("Quality-only runs require every arm in --quality-arms")
    if args.runs < 1 or args.context <= 0 or args.output_tokens < 2:
        parser.error("Invalid workload dimensions")
    source = args.source_root.resolve()
    if " " in str(source) or not (source / "scripts/serve-qwen38.sh").exists():
        parser.error("Expected an immutable source root without spaces")
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    logs = args.out / "server-logs"
    logs.mkdir(exist_ok=True)
    plan = {
        "started_utc": utcnow(), "status": "running", "source_root": str(source),
        "arms": arms, "quality_arms": sorted(quality), "boot_id": args.boot_id,
        "context": args.context, "output_tokens": args.output_tokens,
        "runs_per_cell": args.runs, "families": ["code", "english", "japanese"],
        "sampling_modes": ["greedy", "sampled"], "concurrency": 1,
        "long_warmup": "One full-context, full-output request excluded before measured cells",
        "geometry_policy": "Unchanged launcher auto allocation; actual per-arm geometry retained",
        "qualification": "Exploratory screen; no default changes or universal quality claims",
        "boots": [],
    }
    plan_path = args.out / f"plan-{args.boot_id}.json"
    if plan_path.exists():
        parser.error("This boot ID already has a plan; use a new output directory or boot ID")
    save(plan_path, plan)
    proc = None
    try:
        for arm in arms:
            check_port(args.port)
            log = logs / f"{arm}-{args.boot_id}.log"
            if log.exists():
                raise FileExistsError(log)
            k, kmax = ARMS[arm]
            env = {key: value for key, value in os.environ.items() if not key.startswith("FREETOKEN_")}
            env.update(
                PYTHONPATH=str(source / "python"), FREETOKEN_HOST="127.0.0.1",
                FREETOKEN_PORT=str(args.port), FREETOKEN_MODEL_PATH=args.model_path,
                FREETOKEN_SPEC_DRAFT_TOKENS=str(k), FREETOKEN_SPEC_ADAPTIVE_K_MAX=str(kmax),
                FREETOKEN_SPEC_POSITION_STATS="1", FREETOKEN_MOE_STATS_LOG="200",
            )
            command = ["bash", str(source / "scripts/serve-qwen38.sh")]
            boot = {"arm": arm, "started_utc": utcnow(), "command": command,
                    "env": {k: v for k, v in env.items() if k.startswith("FREETOKEN_") or k == "PYTHONPATH"},
                    "log": str(log), "status": "loading"}
            plan["boots"].append(boot)
            print(f"{utcnow()} Starting {arm}, base depth {k}, maximum {kmax}", flush=True)
            with log.open("w") as stream:
                proc = subprocess.Popen(command, cwd=source, env=env, stdout=stream,
                                        stderr=subprocess.STDOUT, start_new_session=True)
            boot["pid"] = proc.pid
            save(plan_path, plan)
            base = f"http://127.0.0.1:{args.port}/v1"
            wait_ready(proc, base)
            boot["status"] = "measuring"
            save(plan_path, plan)
            if not args.quality_only:
                config = argparse.Namespace(
                    base=base, out=str(args.out), variant=arm, server="freetoken",
                    capture_output=True, model="Qwen3.8-Flash-Next-NVFP4",
                    model_path=args.model_path, suite="mixed", boot_id=args.boot_id,
                    runs=args.runs, contexts=str(args.context), kinds="code,english,japanese",
                    sampling_modes="greedy,sampled", output_tokens=args.output_tokens,
                    server_log=str(log),
                )
                runner = Runner(config)
                save(args.out / f"meta-{arm}-{args.boot_id}.json", collect_metadata(config, runner))
                warm_up(runner)
                runner.measure(
                    f"warmup/{arm}/{args.boot_id}/{args.context}", "realistic", args.context,
                    args.output_tokens, seed=9001, nonce=f"mtp-warmup-{args.boot_id}",
                    telemetry=False, server_log=str(log),
                )
                run_mixed(runner, config)
            if arm in quality:
                boot["status"] = "quality"
                save(plan_path, plan)
                dest = args.out / "quality" / f"{arm}-{args.boot_id}"
                code = subprocess.run([
                    "bash", str(source / "benchmark/run_decode_quality.sh"), str(dest), base,
                    "Qwen3.8-Flash-Next-NVFP4",
                ]).returncode
                boot["quality_exit_code"] = code
                # Retain failed quality as a result without preventing the control run.
            boot["status"] = "complete"
            boot["finished_utc"] = utcnow()
            stop_owned_server(proc)
            proc = None
            save(plan_path, plan)
        plan["status"] = "complete"
    except BaseException as error:
        plan["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        plan["error"] = str(error)
        raise
    finally:
        if proc is not None:
            stop_owned_server(proc)
        plan["finished_utc"] = utcnow()
        save(plan_path, plan)


if __name__ == "__main__":
    main()
