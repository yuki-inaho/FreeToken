#!/usr/bin/env bash

set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Which model the FreeToken half of the stack serves (all listen on port 1919):
#   qwen38      Qwen3.8-Flash-Next-NVFP4 (default; MoE, 128K context, MTP speculation)
#   qwen38-27b  orcarouter/Qwen3.8-27B-Uncensored-NVFP4 (dense, MTP speculation)
#   dsv4-flash  deepseek-ai/DeepSeek-V4-Flash-0731 (304B MoE, experts streamed from host RAM)
case "${FREETOKEN_MODEL:-qwen38}" in
  qwen38) readonly MODEL_LAUNCHER="$SCRIPT_DIR/serve-qwen38.sh" ;;
  qwen38-27b) readonly MODEL_LAUNCHER="$SCRIPT_DIR/serve-qwen38-27b.sh" ;;
  dsv4-flash | deepseek-v4-flash) readonly MODEL_LAUNCHER="$SCRIPT_DIR/serve-dsv4-flash.sh" ;;
  *)
    printf 'Unknown FREETOKEN_MODEL=%s (expected qwen38, qwen38-27b or dsv4-flash)\n' "$FREETOKEN_MODEL" >&2
    exit 1
    ;;
esac

freetoken_pid=''
webui_pid=''

cleanup() {
  trap - EXIT INT TERM
  for pid in "$webui_pid" "$freetoken_pid"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  for pid in "$webui_pid" "$freetoken_pid"; do
    if [[ -n "$pid" ]]; then
      wait "$pid" 2>/dev/null || true
    fi
  done
}

trap cleanup EXIT INT TERM

"$MODEL_LAUNCHER" &
freetoken_pid=$!

"$SCRIPT_DIR/serve-open-webui.sh" &
webui_pid=$!

printf '\nLocal AI stack is starting. Open http://127.0.0.1:%s in a browser.\n' "${OPEN_WEBUI_PORT:-3000}"
printf 'The first Open WebUI account created will be the administrator.\n'
printf 'Press Ctrl+C to stop both services.\n\n'

set +e
wait -n "$freetoken_pid" "$webui_pid"
status=$?
set -e

if kill -0 "$freetoken_pid" 2>/dev/null && kill -0 "$webui_pid" 2>/dev/null; then
  printf 'A service exited unexpectedly.\n' >&2
fi

exit "$status"
