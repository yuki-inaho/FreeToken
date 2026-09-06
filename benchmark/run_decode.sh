#!/usr/bin/env bash
# Isolated decode ablations; only stops servers started by this invocation.
# bash benchmark/run_decode.sh LABEL [VARIANTS] [BOOTS] [RUNS] [CONTEXTS] [OUTPUT]
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${FREETOKEN_VENV:-/home/nuko/.freetoken/venv}"
PY="$VENV/bin/python"
SOURCE="${FREETOKEN_DECODE_PYTHONPATH:-/home/nuko/ft/python}"
PORT="${FREETOKEN_PORT:-1920}"
BASE="http://127.0.0.1:$PORT/v1"
OUT="$REPO/benchmark/results/${1:?provide a results label}"
IFS=, read -r -a variants <<< "${2:-baseline,admission,shared,gdn,qsa,combined}"
IFS=, read -r -a boots <<< "${3:-ablation}"
runs="${4:-2}"
contexts="${5:-4096,32768}"
output="${6:-1024}"
mkdir -p "$OUT/server-logs"
cd "$REPO"
server_pid=""

stop_owned_server() {
  if [[ -n "$server_pid" ]]; then
    kill -TERM -- "-$server_pid" 2>/dev/null || true
    for ((i=0; i<60; i++)); do
      kill -0 -- "-$server_pid" 2>/dev/null || break
      sleep 1
    done
    if kill -0 -- "-$server_pid" 2>/dev/null; then
      kill -KILL -- "-$server_pid" 2>/dev/null || true
    fi
    wait "$server_pid" 2>/dev/null || true
    server_pid=""
  fi
}
trap stop_owned_server EXIT
trap 'exit 130' INT TERM

for variant in "${variants[@]}"; do
  flags=(FREETOKEN_HYBRID_ADMISSION_CREDIT=0 FREETOKEN_HYBRID_SHARED_OVERLAP=0
         FREETOKEN_GDN_STRIDED_NORM=0 FREETOKEN_QSA_PARALLEL_HEADS=0 FREETOKEN_QSA_DIRECT_PERMUTE=0)
  case "$variant" in
    baseline) ;;
    admission) flags[0]=FREETOKEN_HYBRID_ADMISSION_CREDIT=1 ;;
    shared) flags[1]=FREETOKEN_HYBRID_SHARED_OVERLAP=1 ;;
    gdn) flags[2]=FREETOKEN_GDN_STRIDED_NORM=1 ;;
    qsa) flags[3]=FREETOKEN_QSA_PARALLEL_HEADS=1; flags[4]=FREETOKEN_QSA_DIRECT_PERMUTE=1 ;;
    combined) flags=(FREETOKEN_HYBRID_ADMISSION_CREDIT=1 FREETOKEN_HYBRID_SHARED_OVERLAP=1
                    FREETOKEN_GDN_STRIDED_NORM=1 FREETOKEN_QSA_PARALLEL_HEADS=1 FREETOKEN_QSA_DIRECT_PERMUTE=1) ;;
    *) echo "Unknown variant: $variant" >&2; exit 2 ;;
  esac
  for boot in "${boots[@]}"; do
    if curl -fsS --max-time 2 "${BASE%/v1}/health" >/dev/null 2>&1; then
      echo "Port $PORT already serves a model; refusing to interrupt it." >&2
      exit 1
    fi
    source_path="$SOURCE"
    if [[ "$variant" == baseline ]]; then
      source_path="${FREETOKEN_DECODE_BASELINE_PYTHON:-$SOURCE}"
    fi
    # TVM-FFI resolves symlinks before emitting ninja input paths. A space-free
    # symlink alone therefore cannot make a checkout containing spaces JIT-safe.
    # Keep an immutable source snapshot for this boot, including its built CPU .so.
    staged_source="$(mktemp -d /tmp/freetoken-decode-XXXXXX)"
    cp -a "$source_path" "$staged_source/python"
    source_path="$staged_source/python"
    log="$OUT/server-logs/$variant-$boot.log"
    if [[ -e "$log" ]]; then
      echo "Server log $log exists; use a fresh boot label/results directory." >&2
      exit 1
    fi
    echo "Starting $variant boot $boot from $source_path"
    setsid env PYTHONPATH="$source_path" FREETOKEN_PORT="$PORT" \
      FREETOKEN_SPEC_HOST_TIME=1 FREETOKEN_MOE_STATS_LOG=200 "${flags[@]}" \
      bash scripts/serve-qwen38.sh >"$log" 2>&1 &
    server_pid=$!
    ready=0
    for ((i=0; i<600; i++)); do
      if ! kill -0 "$server_pid" 2>/dev/null; then
        tail -25 "$log" >&2
        exit 1
      fi
      if curl -fsS --max-time 2 "${BASE%/v1}/health" 2>/dev/null | \
          "$PY" -c 'import json,sys; sys.exit(json.load(sys.stdin).get("status") != "ok")' 2>/dev/null; then
        ready=1
        break
      fi
      sleep 2
    done
    [[ "$ready" == 1 ]] || { tail -25 "$log" >&2; exit 1; }
    PYTHONPATH="$SOURCE" "$PY" -m benchmark.benchmark --suite mixed --base "$BASE" \
      --out "$OUT" --variant "$variant" --boot-id "$boot" --runs "$runs" \
      --contexts "$contexts" --output-tokens "$output" --server-log "$log"
    if [[ "${FREETOKEN_DECODE_QUALITY:-0}" == 1 && "$boot" == "${boots[-1]}" ]]; then
      bash benchmark/run_decode_quality.sh "$OUT/quality/$variant-$boot" "$BASE"
    fi
    stop_owned_server
    "$PY" -m benchmark.validate "$OUT"
    "$PY" -m benchmark.mixed "$OUT"
  done
done
