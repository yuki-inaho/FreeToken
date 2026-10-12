#!/usr/bin/env bash
# FreeToken 本体 + kernel-cache wheel をビルド (pixi run build-wheels)
# - 複数アーキの既定: 8.0 8.6 8.9 9.0 10.0 12.0 (freetoken-kernel-cache/build_backend.py)
# - 上書き: FREETOKEN_KERNEL_CACHE_ARCHES="8.0 8.9" など
set -euo pipefail

FT_DIR="${FREETOKEN_DIR:-$HOME/Desktop/FreeToken}"
: "${CONDA_PREFIX:?pixi 環境で実行してください: pixi run build-wheels}"

export FREETOKEN_BUILD_PYTHON="${FREETOKEN_BUILD_PYTHON:-$CONDA_PREFIX/bin/python}"
export CUDA_HOME="${CUDA_HOME:-$CONDA_PREFIX}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$CUDA_HOME/lib:${LIBRARY_PATH:-}"
# cgroup 環境での RAM 保護のため ninja/nvcc の並列度を制限 (上書き可)
export MAX_JOBS="${MAX_JOBS:-48}"

if [ -n "${FREETOKEN_KERNEL_CACHE_ARCHES:-}" ]; then
  echo "kernel-cache arch override: $FREETOKEN_KERNEL_CACHE_ARCHES"
fi

cd "$FT_DIR"
exec bash scripts/build-release-wheels.sh "$@"
