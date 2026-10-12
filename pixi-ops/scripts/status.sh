#!/usr/bin/env bash
# サーバーと GPU の状態 (pixi run status)
set -uo pipefail
PORT="${FREETOKEN_PORT:-1919}"
echo "-- /v1/models --"
curl -s -m 3 "http://127.0.0.1:${PORT}/v1/models" || echo "(サーバー応答なし)"
echo
echo "-- GPU --"
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
echo "-- プロセス --"
pgrep -af "ft serve" | head -3 || echo "(ft serve なし)"
