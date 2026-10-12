#!/usr/bin/env bash
# GitHub Releases から FreeToken wheel を取得して導入 (pixi run install-from-release / 他マシン用)
# 上書き: FREETOKEN_RELEASE_REPO (既定 yuki-inaho/FreeToken), FREETOKEN_RELEASE_TAG (既定 latest)
set -euo pipefail

REPO="${FREETOKEN_RELEASE_REPO:-yuki-inaho/FreeToken}"
TAG="${FREETOKEN_RELEASE_TAG:-latest}"

echo "== FreeToken wheels from $REPO ($TAG) =="

URLS="$(python3 - "$REPO" "$TAG" <<'PY'
import json, sys, urllib.request
repo, tag = sys.argv[1], sys.argv[2]
api = f"https://api.github.com/repos/{repo}/releases/" + ("latest" if tag == "latest" else f"tags/{tag}")
with urllib.request.urlopen(api) as r:
    rel = json.load(r)
print("\n".join(a["browser_download_url"] for a in rel["assets"] if a["name"].endswith(".whl")))
PY
)"

FT_URL=""; KC_URL=""
while IFS= read -r u; do
  case "$u" in
    *kernel_cache*) KC_URL="$u" ;;
    *freetoken-*)   FT_URL="$u" ;;
  esac
done <<< "$URLS"

[ -n "$FT_URL" ] && [ -n "$KC_URL" ] || { echo "release に wheel が見つかりません: $URLS" >&2; exit 1; }
echo "runtime: $FT_URL"
echo "kernel : $KC_URL"

pip install "${FT_URL}[accel]" "$KC_URL" \
  flashinfer-cubin flashinfer-jit-cache \
  --extra-index-url https://download.pytorch.org/whl/cu130 \
  --extra-index-url https://docs.sglang.io/whl/cu130 \
  --extra-index-url https://flashinfer.ai/whl \
  --extra-index-url https://flashinfer.ai/whl/cu130

python3 - <<'PY'
import freetoken
print("freetoken ok:", freetoken.__file__)
PY
