#!/usr/bin/env bash

set -euo pipefail

readonly OPEN_WEBUI_HOME="${OPEN_WEBUI_HOME:-$HOME/.open-webui}"
readonly OPEN_WEBUI_VENV="${OPEN_WEBUI_VENV:-$OPEN_WEBUI_HOME/venv}"
readonly OPEN_WEBUI_VERSION="${OPEN_WEBUI_VERSION:-0.11.1}"

if ! command -v uv >/dev/null 2>&1; then
  printf 'uv is required to install Open WebUI. See https://docs.astral.sh/uv/\n' >&2
  exit 1
fi

mkdir -p "$OPEN_WEBUI_HOME/data"

if [[ ! -x "$OPEN_WEBUI_VENV/bin/python" ]]; then
  uv venv "$OPEN_WEBUI_VENV" --python 3.11
fi

uv pip install --python "$OPEN_WEBUI_VENV" "open-webui==$OPEN_WEBUI_VERSION"

printf 'Open WebUI %s is installed in %s.\n' "$OPEN_WEBUI_VERSION" "$OPEN_WEBUI_VENV"
printf 'Start it with: ./scripts/serve-open-webui.sh\n'
