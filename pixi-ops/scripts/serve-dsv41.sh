#!/usr/bin/env bash
# DeepSeek-V4.1-Flash を配信 (pixi run serve-dsv41)
# 注: V4.1 は公式チェックポイント 475GiB + routed experts 269GiB の pinned host RAM を要求する。
#     このマシン (ディスク 250GB / cgroup 139.4GiB) では実行不可。要件と現在地を表示して停止する。
set -euo pipefail

FT_DIR="${FREETOKEN_DIR:-$HOME/Desktop/FreeToken}"
MODEL_PATH="${FREETOKEN_MODEL_PATH:-${FREETOKEN_MODELS:-$HOME/Desktop/ft-models}/DeepSeek-V4.1-Flash}"

echo "== DeepSeek-V4.1-Flash =="
echo "要件 (pentaCoxian/FreeToken scripts/serve-dsv41-flash.sh より):"
echo "  checkpoint 475 GiB / routed experts 269 GiB pinned RAM / Engram 203GB on NVMe / MemAvailable >= 300 GiB"
echo "現在:"
printf "  disk free: "; df -h "$HOME" | awk 'NR==2 {print $4}'
if [ -r /sys/fs/cgroup/memory.max ]; then
  awk '{printf "  cgroup max: %.1f GiB\n", $1/1073741824}' /sys/fs/cgroup/memory.max
fi
awk '/MemAvailable/ {printf "  MemAvailable: %.1f GiB\n", $2/1048576}' /proc/meminfo
if [ ! -d "$MODEL_PATH" ]; then
  echo "  model dir: $MODEL_PATH (未取得)"
  echo "=> 取得するには: hf download deepseek-ai/DeepSeek-V4.1-Flash --local-dir $MODEL_PATH (475GiB)"
fi

export FREETOKEN_MODEL_PATH="$MODEL_PATH"
export FREETOKEN_FT_BIN="${FREETOKEN_FT_BIN:-${CONDA_PREFIX:-/nonexistent}/bin/ft}"
echo "--- フォークのスクリプトに引き継ぎます (プリフライトで拒否されるはず) ---"
exec bash "$FT_DIR/scripts/serve-dsv41-flash.sh" "$@"
