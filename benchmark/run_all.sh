#!/usr/bin/env bash
# End-to-end benchmark orchestration: boots the right server per phase, runs the suites,
# validates and renders the report. Run from anywhere; results land in benchmark/results/.
#
#   bash benchmark/run_all.sh [results-label] [phases]
#
# phases: comma list out of 1,2,3,4 (default "1,2,3"); 4 = upstream comparison (optional,
# builds a clean-HEAD worktree server).  Requires an idle desktop: no OBS/browser video.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${FREETOKEN_VENV:-/home/nuko/.freetoken/venv}"
PY="$VENV/bin/python"
LABEL="${1:-$(date +%Y%m%d-%H%M)}"
PHASES="${2:-1,2,3}"
OUT="$REPO/benchmark/results/$LABEL"
LOGDIR="$OUT/server-logs"
mkdir -p "$OUT" "$LOGDIR"
cd "$REPO"

BASE="${FREETOKEN_API_BASE:-http://127.0.0.1:1919/v1}"

say() { printf '\n=== %s (%s)\n' "$1" "$(date +%H:%M:%S)"; }

stop_server() {
  pkill -TERM -f "[f]t serve --model-path" 2>/dev/null || true
  for _ in $(seq 1 60); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$used" -lt 2000 ] && return 0
    sleep 2
  done
  echo "GPU did not drain below 2 GB" >&2; return 1
}

boot_server() { # boot_server <logname> [ENV=VAL ...]
  local log="$LOGDIR/$1.log"; shift
  say "booting server -> $log  ($*)"
  ( cd "$REPO" && env \
      FREETOKEN_SPEC_HOST_TIME=1 FREETOKEN_MOE_STATS_LOG=200 "$@" \
      bash scripts/serve-qwen38.sh >"$log" 2>&1 & )
  for _ in $(seq 1 240); do
    if curl -fsS --max-time 2 "${BASE%/v1}/health" 2>/dev/null | grep -q '"status": *"ok"'; then
      say "server up"; return 0
    fi
    sleep 5
  done
  echo "server did not come up; tail of log:" >&2; tail -20 "$log" >&2; return 1
}

run() { say "suite $*"; "$PY" -m benchmark.benchmark --out "$OUT" "$@"; }

if [[ ",$PHASES," == *",1,"* ]]; then
  stop_server; boot_server phase1-default
  run --suite smoke       --variant custom
  run --suite primary     --variant custom
  run --suite context     --variant custom
  run --suite prefill     --variant custom
  run --suite cache       --variant custom
  run --suite mtp         --variant mtp_on
fi

if [[ ",$PHASES," == *",2,"* ]]; then
  stop_server; boot_server phase2-specoff FREETOKEN_SPECULATIVE=off
  run --suite mtp --variant mtp_off
fi

if [[ ",$PHASES," == *",3,"* ]]; then
  stop_server; boot_server phase3-mrr8 FREETOKEN_MAX_RUNNING_REQUESTS=8
  run --suite concurrency --variant concurrency8 --levels 1,2,4,8
fi

if [[ ",$PHASES," == *",4,"* ]]; then
  say "phase 4 (upstream) must be prepared once: see benchmark/README.md; skipping if absent"
  if [ -x "$HOME/ft-upstream/scripts-upstream-boot.sh" ]; then
    stop_server
    ( "$HOME/ft-upstream/scripts-upstream-boot.sh" >"$LOGDIR/phase4-upstream.log" 2>&1 & )
    sleep 5
    for _ in $(seq 1 240); do
      curl -fsS --max-time 2 "${BASE%/v1}/health" 2>/dev/null | grep -q '"status": *"ok"' && break
      sleep 5
    done
    run --suite context --variant upstream --runs 2
    run --suite prefill --variant upstream --runs 2
  fi
fi

stop_server || true
say "validate + report"
"$PY" -m benchmark.validate "$OUT"
"$PY" -m benchmark.report "$OUT"
say "done: $OUT/report.md"
