#!/usr/bin/env bash
# Fresh control repeats for explicitly selected cells; never resumes an old output directory.
set -euo pipefail

plan_only=0
if [[ "${1:-}" == --plan-only ]]; then plan_only=1; shift; fi
if [[ "${1:-}" == --help || "${1:-}" == -h || $# -lt 3 || $# -gt 4 ]]; then
  cat <<'USAGE'
Usage: bash benchmark/run_decode_followup.sh [--plan-only] OUTPUT_DIR REFERENCE_DIR CELLS [BOOTS]

CELLS is a comma-separated list of sampling:kind:context, for example:
  greedy:japanese:131072,sampled:english:253952
BOOTS defaults to 1,2 and must match the existing combined boot labels.

After the main benchmark has stopped, runs five 4096-output control requests
per selected cell in each fresh server boot. FREETOKEN_FOLLOWUP_MODE selects:
  baseline (default): old source, all five controls off, variant baseline-followup
  execution-only: current source, admission off and shared/GDN/QSA on,
                  variant execution-only-followup
FREETOKEN_FOLLOWUP_VARIANT overrides the default variant. One server serves all selected
cells in a boot, in ascending context order. OUTPUT_DIR must not exist.
Each cell gets its own directory to preserve every CLI metadata file.
Complete measured outputs are saved under each cell's outputs/ directory.

--plan-only validates completed reference cells and prints the selection manifest;
it performs no writes, server requests, port probes or GPU work.

Default baseline source: /tmp/ft-decode-baseline-lNjUhP/python
Default execution-only source: this repository's python/ directory.
Source overrides: FREETOKEN_DECODE_BASELINE_PYTHON, FREETOKEN_DECODE_PYTHONPATH.
Other overrides: FREETOKEN_VENV, FREETOKEN_PORT. FREETOKEN_FOLLOWUP_WARMUP=0
skips client warm-up before the first cell; calibration still runs. Default is 1.
This follow-up reuses selected candidate observations and changes cache history.
It is descriptive follow-up evidence, not independent confirmation or proof of
no regressions. Exact prompt hashes must match in the resulting comparison.
USAGE
  [[ $# -ge 1 && $# -le 4 && ( "${1:-}" == --help || "${1:-}" == -h ) ]] && exit 0
  exit 2
fi

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${FREETOKEN_VENV:-/home/nuko/.freetoken/venv}"
PY="$VENV/bin/python"
CONTROL="${FREETOKEN_FOLLOWUP_MODE:-baseline}"
WARMUP="${FREETOKEN_FOLLOWUP_WARMUP:-1}"
case "$CONTROL" in
  baseline)
    SOURCE="${FREETOKEN_DECODE_BASELINE_PYTHON:-/tmp/ft-decode-baseline-lNjUhP/python}"
    default_variant=baseline-followup ;;
  execution-only)
    SOURCE="${FREETOKEN_DECODE_PYTHONPATH:-$REPO/python}"
    default_variant=execution-only-followup ;;
  *) echo "Unsupported follow-up mode: $CONTROL" >&2; exit 2 ;;
esac
PORT="${FREETOKEN_PORT:-1920}"
BASE="http://127.0.0.1:$PORT/v1"
VARIANT="${FREETOKEN_FOLLOWUP_VARIANT:-$default_variant}"
OUT="$1"
REFERENCE="$2"
IFS=, read -r -a boots <<< "${4:-1,2}"
[[ -x "$PY" && -d "$SOURCE/freetoken" ]] || { echo "Missing Python environment or selected runtime source." >&2; exit 1; }
cd "$REPO"

manifest="$("$PY" - "$OUT" "$REFERENCE" "$3" "${4:-1,2}" "$VARIANT" "$SOURCE" "$PORT" "$CONTROL" "$WARMUP" <<'PY'
import hashlib, json, pathlib, re, sys
from benchmark.mixed import CONTEXTS, KINDS, SAMPLING, eligible
out, reference, raw_cells, raw_boots, variant, source, port, control, warmup = sys.argv[1:]
if warmup not in ('0', '1'):
    raise SystemExit('FREETOKEN_FOLLOWUP_WARMUP must be 0 or 1.')
if pathlib.Path(out).exists():
    raise SystemExit('OUTPUT_DIR already exists; choose a new directory to preserve all records.')
if not re.fullmatch(r'[A-Za-z0-9_-]+', variant) or variant in ('baseline', 'combined'):
    raise SystemExit('Use a new variant name, such as baseline-followup.')
boots = raw_boots.split(',')
if len(boots) != len(set(boots)) or not all(re.fullmatch(r'[A-Za-z0-9_-]+', b) for b in boots):
    raise SystemExit('Boot labels must be unique and contain only letters, digits, underscores or hyphens.')
if not port.isdecimal() or not 1 <= int(port) <= 65535:
    raise SystemExit('Invalid port.')
cells = []
for cell in raw_cells.split(','):
    parts = cell.split(':')
    if len(parts) != 3 or parts[0] not in SAMPLING or parts[1] not in KINDS or not parts[2].isdecimal():
        raise SystemExit(f'Invalid selected cell: {cell}')
    mode, kind, context = parts[0], parts[1], int(parts[2])
    if context not in CONTEXTS:
        raise SystemExit(f'Context must belong to the qualification matrix: {context}')
    cells.append((mode, kind, context))
if len(set(cells)) != len(cells):
    raise SystemExit('Duplicate selected cells are not allowed.')
cells.sort(key=lambda c: (c[2], c[1], c[0]))
rows = [json.loads(line) for line in (pathlib.Path(reference) / 'runs.jsonl').read_text().splitlines() if line.strip()]
references, instances = [], {}
for boot in boots:
    ids = set()
    for mode, kind, context in cells:
        for repeat in range(5):
            matches = [r for r in rows if r.get('suite') == 'mixed' and r.get('variant') == 'combined'
                       and r.get('family') == kind and r.get('target_input_tokens') == context
                       and (r.get('extra') or {}).get('boot_id') == boot
                       and r['extra'].get('sampling_mode') == mode and r['extra'].get('repeat') == repeat]
            if len(matches) != 1:
                raise SystemExit(f'Need one completed combined reference: {boot}/{mode}/{kind}/{context}/{repeat}')
            row = matches[0]
            prompt = row.get('prompt') or {}
            nonce = f'mixed-{boot}-{mode}-{kind}-{context}-{repeat}'
            if (not eligible(row) or row.get('requested_output_tokens') != 4096
                    or prompt.get('nonce') != nonce or prompt.get('target_tokens') != context
                    or prompt.get('expected_prompt_tokens') != context
                    or not isinstance(prompt.get('sha256'), str)
                    or not re.fullmatch(r'[0-9a-fA-F]{64}', prompt['sha256'])):
                raise SystemExit(f'Reference is invalid or used a token-fit correction requiring manual replay: {row.get("key")}')
            ids.add(row.get('server_instance'))
            references.append({'key': row['key'], 'prompt_sha256': prompt['sha256'],
                               'nonce': nonce, 'overhead_tokens': prompt.get('overhead_tokens')})
    if len(ids) != 1 or None in ids:
        raise SystemExit(f'Reference boot {boot} has missing or inconsistent actual server identities.')
    instances[boot] = ids.pop()
if len(set(instances.values())) != len(boots):
    raise SystemExit('Reference boot labels reuse an actual server instance.')
record = {
    'output_directory': str(pathlib.Path(out).resolve()), 'reference_directory': str(pathlib.Path(reference).resolve()),
    'baseline_source': str(pathlib.Path(source).resolve()) if control == 'baseline' else None,
    'runtime_source': str(pathlib.Path(source).resolve()), 'control': control,
    'runtime_controls': {
        'FREETOKEN_HYBRID_ADMISSION_CREDIT': '0',
        **{name: '1' if control == 'execution-only' else '0' for name in (
            'FREETOKEN_HYBRID_SHARED_OVERLAP', 'FREETOKEN_GDN_STRIDED_NORM',
            'FREETOKEN_QSA_PARALLEL_HEADS', 'FREETOKEN_QSA_DIRECT_PERMUTE')},
    },
    'variant': variant, 'boots': boots,
    'cells': [f'{mode}:{kind}:{context}' for mode, kind, context in cells],
    'runs_per_cell_per_boot': 5, 'output_tokens': 4096, 'port': int(port),
    'capture_output': True, 'warmup_first_cell': warmup == '1',
    'reference_instances': instances, 'selected_reference_records': references,
    'selection_rule': f'Explicit cells selected after the original comparison; fresh {control} control, existing combined observations reused.',
    'limitations': ['Selected observations are reused, so this is conditional descriptive evidence.',
                    'Selected-cell order changes expert-cache history relative to the full matrix.',
                    'A fresh token-fit calibration must still produce identical prompt SHA-256; mismatches remain excluded.',
                    'Five repeats in two boots provide only two independent boot clusters.'],
    'client_source_sha256': {name: hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest()
                            for name in ('benchmark/benchmark.py', 'benchmark/workloads.py', 'benchmark/mixed.py')},
}
if control == 'execution-only':
    record['limitations'].append('Attributing a difference to admission also requires the selected current runtime source to otherwise match the combined runtime.')
if warmup == '0':
    record['limitations'].append('Client warm-up is disabled; startup calibration and server-internal warm-up still run.')
print(json.dumps(record, indent=2))
PY
)"
if [[ "$plan_only" == 1 ]]; then printf '%s\n' "$manifest"; exit 0; fi

# Probe only whether the port is free; never ask an unrelated running model for work.
assert_port_free() {
  "$PY" - "$PORT" <<'PY'
import socket, sys
try:
    with socket.socket() as sock:
        # Match the server's address reuse so owned-boot TIME_WAIT does not
        # prevent the next boot. A listening socket still blocks this bind.
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('127.0.0.1', int(sys.argv[1])))
except OSError as exc:
    raise SystemExit(f'Port {sys.argv[1]} is occupied or unavailable; refusing to start: {exc}')
PY
}
assert_port_free
mkdir -p "$OUT/server-logs" "$OUT/cells"
OUT="$(cd "$OUT" && pwd)"
REFERENCE="$(cd "$REFERENCE" && pwd)"
printf '%s\n' "$manifest" > "$OUT/selection.json"
mapfile -t cells < <("$PY" -c 'import json,sys; print("\n".join(json.load(sys.stdin)["cells"]))' <<< "$manifest")
mapfile -t controls < <("$PY" -c 'import json,sys; print("\n".join(f"{k}={v}" for k,v in json.load(sys.stdin)["runtime_controls"].items()))' <<< "$manifest")
cell_dirs=()
for cell in "${cells[@]}"; do cell_dirs+=("$OUT/cells/${cell//:/-}"); done

server_pid=""
stop_owned_server() {
  if [[ -n "$server_pid" ]]; then
    kill -TERM -- "-$server_pid" 2>/dev/null || true
    for ((i=0; i<60; i++)); do
      kill -0 -- "-$server_pid" 2>/dev/null || break
      sleep 1
    done
    if kill -0 -- "-$server_pid" 2>/dev/null; then kill -KILL -- "-$server_pid" 2>/dev/null || true; fi
    wait "$server_pid" 2>/dev/null || true
    server_pid=""
  fi
}
trap stop_owned_server EXIT
trap 'exit 130' INT TERM

for boot in "${boots[@]}"; do
  assert_port_free
  staged_source="$(mktemp -d /tmp/freetoken-decode-followup-XXXXXX)"
  cp -a "$SOURCE" "$staged_source/python"
  printf '%s\n' "$staged_source/python" > "$OUT/server-logs/$VARIANT-$boot-source.txt"
  log="$OUT/server-logs/$VARIANT-$boot.log"
  echo "Starting $VARIANT boot $boot from $staged_source/python"
  setsid env PYTHONPATH="$staged_source/python" FREETOKEN_PORT="$PORT" FREETOKEN_HOST=127.0.0.1 \
    FREETOKEN_SPEC_HOST_TIME=1 FREETOKEN_MOE_STATS_LOG=200 \
    "${controls[@]}" \
    FREETOKEN_SPECULATIVE=mtp FREETOKEN_SPEC_DRAFT_TOKENS=2 FREETOKEN_SPEC_ADAPTIVE_K_MAX=3 \
    FREETOKEN_MOE_BACKEND=hybrid FREETOKEN_NVFP4_BACKEND=triton FREETOKEN_MAX_RUNNING_REQUESTS=1 \
    FREETOKEN_EXPERT_LOAD=auto FREETOKEN_DEFAULT_REASONING_EFFORT=none \
    FREETOKEN_KV_RESERVE_TOKENS=65536 FREETOKEN_KV_MAX_TOKENS=262144 FREETOKEN_KV_ELASTIC_IDLE_S=300 \
    FREETOKEN_GPU_RESERVE_MB=0 FREETOKEN_LINEAR_STATE_CACHE=1 FREETOKEN_KV_CACHE_DTYPE=fp8 \
    FREETOKEN_DENSE_QUANT=fp8 FREETOKEN_MOE_PREFETCH=off FREETOKEN_MAMBA_SSM_DTYPE=float16 \
    bash scripts/serve-qwen38.sh > "$log" 2>&1 &
  server_pid=$!
  ready=0
  for ((i=0; i<600; i++)); do
    if ! kill -0 "$server_pid" 2>/dev/null; then tail -25 "$log" >&2; exit 1; fi
    if curl -fsS --max-time 2 "${BASE%/v1}/health" 2>/dev/null | \
      "$PY" -c 'import json,sys; sys.exit(json.load(sys.stdin).get("status") != "ok")' 2>/dev/null; then
      ready=1; break
    fi
    sleep 2
  done
  [[ "$ready" == 1 ]] || { tail -25 "$log" >&2; exit 1; }
  warmup=()
  if [[ "$WARMUP" == 0 ]]; then warmup=(--no-warmup); fi
  for cell in "${cells[@]}"; do
    IFS=: read -r mode kind context <<< "$cell"
    cell_out="$OUT/cells/${cell//:/-}"
    "$PY" -m benchmark.benchmark --suite mixed --base "$BASE" --out "$cell_out" \
      --variant "$VARIANT" --boot-id "$boot" --runs 5 --contexts "$context" --kinds "$kind" \
      --sampling-modes "$mode" --output-tokens 4096 --capture-output --server-log "$log" "${warmup[@]}"
    warmup=(--no-warmup)
  done
  stop_owned_server
done
for directory in "${cell_dirs[@]}"; do "$PY" -m benchmark.validate "$directory"; done
"$PY" -m benchmark.compare_decode "$REFERENCE" "${cell_dirs[@]}" \
  --baseline "$VARIANT" --candidate combined --out-prefix "$OUT/decode-comparison"
echo "Follow-up complete; inspect matched counts, prompt exclusions, geometry and per-boot cell changes."
