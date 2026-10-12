#!/usr/bin/env bash
# dist/ のビルド済み wheel をこの環境へ導入 (pixi run install-wheels)
set -euo pipefail

FT_DIR="${FREETOKEN_DIR:-$HOME/Desktop/FreeToken}"
DIST="$FT_DIR/dist"

shopt -s nullglob
RUNTIME=( "$DIST"/freetoken-*.whl )
CACHE=( "$DIST"/freetoken_kernel_cache-*.whl )
if [ ${#RUNTIME[@]} -eq 0 ] || [ ${#CACHE[@]} -eq 0 ]; then
  echo "wheel が見つかりません ($DIST) — 先に: pixi run build-wheels" >&2
  exit 1
fi

# 実行時アクセラレーション (flashinfer 事前ビルド + sglang-kernel) は公式 install.sh と同じ経路。
# flashinfer-jit-cache は flashinfer-python と一致させる (0.6.18.post1+cu130)。
# flashinfer-cubin は PyPI 側の最新 (0.7.1) しか無く python 0.6.18 と一致しないため、
# FLASHINFER_DISABLE_VERSION_CHECK=1 で実行する (serve スクリプトに設定済み)。
pip install --upgrade "${RUNTIME[0]}[accel]" "${CACHE[0]}" \
  "flashinfer-jit-cache==0.6.18.post1+cu130" \
  --extra-index-url https://download.pytorch.org/whl/cu130 \
  --extra-index-url https://docs.sglang.io/whl/cu130 \
  --extra-index-url https://flashinfer.ai/whl \
  --extra-index-url https://flashinfer.ai/whl/cu130

python - <<'PY'
import freetoken
print("freetoken ok:", freetoken.__file__)
try:
    import sgl_kernel  # noqa: F401
    print("sgl_kernel ok")
except Exception as e:
    print("sgl_kernel import failed:", e)
PY
