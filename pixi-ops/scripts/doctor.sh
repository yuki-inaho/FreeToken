#!/usr/bin/env bash
# 環境の健全性チェック (pixi run doctor)
set -uo pipefail
FT_DIR="${FREETOKEN_DIR:-$HOME/Desktop/FreeToken}"
MODELS="${FREETOKEN_MODELS:-$HOME/Desktop/ft-models}"

echo "== ft-pixi doctor =="
echo "FREETOKEN_DIR = $FT_DIR"
echo "CONDA_PREFIX  = ${CONDA_PREFIX:-<pixi 外>}"

echo "-- GPU --"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader
else
  echo "nvidia-smi が見つかりません"
fi

echo "-- nvcc --"
if command -v nvcc >/dev/null 2>&1; then
  nvcc --version | tail -2
else
  echo "nvcc が見つかりません (pixi run で実行してください)"
fi

echo "-- python / torch --"
python - <<'PY' 2>/dev/null || echo "torch の import に失敗"
import torch
ok = torch.cuda.is_available()
cc = torch.cuda.get_device_capability() if ok else None
print(f"python {__import__('sys').version.split()[0]} / torch {torch.__version__} / cuda_available={ok} / sm={cc}")
PY

echo "-- モデル (DeepSeek-V4-Flash-0731) --"
for f in config.json model.safetensors.index.json inference/config.json tokenizer.json encoding/encoding_dsv4.py; do
  if [ -f "$MODELS/DeepSeek-V4-Flash-0731/$f" ]; then echo "  OK      $f"; else echo "  MISSING $f"; fi
done
echo "-- モデル (DeepSeek-V4.1-Flash; このマシンでは 475GiB 必要なため通常は無し) --"
ls "$MODELS/DeepSeek-V4.1-Flash" >/dev/null 2>&1 && echo "  present" || echo "  not present"

echo "-- wheel (dist/) --"
ls "$FT_DIR"/dist/*.whl 2>/dev/null || echo "  (まだ無い: pixi run build-wheels)"

echo "-- ディスク --"
df -h "$HOME" | tail -1

echo "-- cgroup メモリ上限 --"
awk '{printf "%.1f GiB\n", $1/1073741824}' /sys/fs/cgroup/memory.max 2>/dev/null || echo "n/a"
