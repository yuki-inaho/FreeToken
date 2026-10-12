#!/usr/bin/env bash
# dist/ の wheel を GitHub Releases に公開 (pixi run release)
# 既定: yuki-inaho/FreeToken に wheels-YYYY.MM.DD タグで作成。
# 上書き: FREETOKEN_RELEASE_REPO / FREETOKEN_RELEASE_TAG
set -euo pipefail

FT_DIR="${FREETOKEN_DIR:-$HOME/Desktop/FreeToken}"
OPS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="${FREETOKEN_RELEASE_REPO:-yuki-inaho/FreeToken}"
TAG="${FREETOKEN_RELEASE_TAG:-wheels-$(date +%Y.%m.%d)}"
DIST="$FT_DIR/dist"

shopt -s nullglob
WHLS=( "$DIST"/freetoken-*.whl "$DIST"/freetoken_kernel_cache-*.whl )
if [ ${#WHLS[@]} -lt 2 ]; then
  echo "wheel が揃っていません ($DIST) — 先に: pixi run build-wheels" >&2
  exit 1
fi

# 運用ツールキット (pixi.toml + scripts) も一緒に配布
OPS_TAR="$DIST/ft-pixi-ops.tar.gz"
tar -C "$(dirname "$OPS_DIR")" -czf "$OPS_TAR" "$(basename "$OPS_DIR")"

SHA="${FREETOKEN_RELEASE_SHA:-$(git -C "$FT_DIR" rev-parse --short=9 HEAD)}"
gh release create "$TAG" "${WHLS[@]}" "$OPS_DIR/scripts/install-from-release.sh" "$OPS_TAR" \
  --repo "$REPO" \
  --title "FreeToken prebuilt wheels $TAG (cu130, multi-arch 80/86/89/90/100/120)" \
  --notes "PentaCoxian/FreeToken @ $SHA の prebuilt wheel。

- freetoken ランタイム wheel (+accel 依存は pip が取得)
- freetoken-kernel-cache wheel (py3-none, SASS: 8.0 8.6 8.9 9.0 10.0 12.0)

かんたん導入 (Linux x86_64 + NVIDIA driver r580+ / CUDA 13 toolkit):
  curl -fsSL https://github.com/$REPO/releases/download/$TAG/install-from-release.sh | bash

手動:
  pip install '<freetoken wheel の URL>[accel]' '<kernel-cache wheel の URL>' \\
    flashinfer-cubin flashinfer-jit-cache \\
    --extra-index-url https://download.pytorch.org/whl/cu130 \\
    --extra-index-url https://docs.sglang.io/whl/cu130 \\
    --extra-index-url https://flashinfer.ai/whl --extra-index-url https://flashinfer.ai/whl/cu130

起動例:
  ft serve --model-path <DeepSeek-V4-Flash-0731 など> --moe-backend hybrid --moe-cache-auto"
