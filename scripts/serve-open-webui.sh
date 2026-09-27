#!/usr/bin/env bash

set -euo pipefail

readonly OPEN_WEBUI_HOME="${OPEN_WEBUI_HOME:-$HOME/.open-webui}"
readonly OPEN_WEBUI_VENV="${OPEN_WEBUI_VENV:-$OPEN_WEBUI_HOME/venv}"
readonly OPEN_WEBUI_BIN="$OPEN_WEBUI_VENV/bin/open-webui"
readonly OPEN_WEBUI_DATA_DIR="${OPEN_WEBUI_DATA_DIR:-$OPEN_WEBUI_HOME/data}"
readonly SERVER_HOST="${OPEN_WEBUI_HOST:-127.0.0.1}"
readonly SERVER_PORT="${OPEN_WEBUI_PORT:-3000}"
readonly FREETOKEN_API_URL="${OPEN_WEBUI_FREETOKEN_URL:-http://127.0.0.1:1919/v1}"

if [[ ! -x "$OPEN_WEBUI_BIN" ]]; then
  printf 'Open WebUI is not installed at %s\n' "$OPEN_WEBUI_BIN" >&2
  printf 'Run scripts/install-open-webui.sh and retry.\n' >&2
  exit 1
fi

mkdir -p "$OPEN_WEBUI_DATA_DIR"

export DATA_DIR="$OPEN_WEBUI_DATA_DIR"
export ENV="${OPEN_WEBUI_ENV:-prod}"
export ENABLE_OPENAI_API="${OPEN_WEBUI_ENABLE_OPENAI_API:-True}"
export OPENAI_API_BASE_URL="$FREETOKEN_API_URL"
# FreeToken does not require authentication, but Open WebUI expects a non-empty
# value for an OpenAI-compatible connection.
export OPENAI_API_KEY="${OPEN_WEBUI_FREETOKEN_API_KEY:-local-freetoken}"
export ENABLE_OLLAMA_API="${OPEN_WEBUI_ENABLE_OLLAMA_API:-False}"
export WEBUI_AUTH="${OPEN_WEBUI_AUTH:-True}"
export ENABLE_SIGNUP="${OPEN_WEBUI_ENABLE_SIGNUP:-True}"
export WEBUI_URL="${OPEN_WEBUI_URL:-http://127.0.0.1:$SERVER_PORT}"
export CORS_ALLOW_ORIGIN="${OPEN_WEBUI_CORS_ALLOW_ORIGIN:-http://127.0.0.1:$SERVER_PORT;http://localhost:$SERVER_PORT}"
# Leave Open WebUI's local embeddings and audio processing on CPU so it does
# not compete with FreeToken for GPU memory.
export USE_CUDA_DOCKER="${OPEN_WEBUI_USE_CUDA:-false}"
export USER_AGENT="${OPEN_WEBUI_USER_AGENT:-Open-WebUI-local}"
export SCARF_NO_ANALYTICS="${SCARF_NO_ANALYTICS:-true}"
export DO_NOT_TRACK="${DO_NOT_TRACK:-true}"
export ANONYMIZED_TELEMETRY="${ANONYMIZED_TELEMETRY:-false}"

printf 'Open WebUI: http://%s:%s\n' "$SERVER_HOST" "$SERVER_PORT"
printf 'FreeToken API: %s\n' "$FREETOKEN_API_URL"

# The CLI stores its generated authentication secret in the current directory.
# Keeping it in DATA_DIR makes sessions stable and prevents a secret file from
# appearing in the source tree.
cd "$OPEN_WEBUI_DATA_DIR"
exec "$OPEN_WEBUI_BIN" serve --host "$SERVER_HOST" --port "$SERVER_PORT" "$@"
