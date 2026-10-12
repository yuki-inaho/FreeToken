#!/usr/bin/env bash
# DeepSeek-V4-Flash-0731 を配信 (pixi run serve)
# 既定は PentaCoxian フォークの serve-dsv4-flash.sh と同じ構成:
#   hybrid MoE (GPU スロット + CPU 共計算), expert cache auto, KV 131072, 単一リクエスト。
# 環境変数で上書き可 (FREETOKEN_PORT, FREETOKEN_MOE_BACKEND, FREETOKEN_MAX_SEQ_LEN など)。
set -euo pipefail

: "${CONDA_PREFIX:?pixi 環境で実行してください: pixi run serve}"
FT_BIN="${FREETOKEN_FT_BIN:-$CONDA_PREFIX/bin/ft}"
MODELS_DIR="${FREETOKEN_MODELS:-$HOME/Desktop/ft-models}"
# 既定モデル: REAP(プルーニング)版を優先。フル V4-Flash-0731 は cgroup 139.4GiB では
# 常駐バンク(133.2GiB)+諸経費が収まらず起動できないため (fault: 40/43層で停止)。
if [ -z "${FREETOKEN_MODEL_PATH:-}" ]; then
  for cand in "DeepSeek-V4-Flash-REAP-145B" "DeepSeek-V4-Flash-0731"; do
    if [ -f "$MODELS_DIR/$cand/inference/config.json" ]; then
      FREETOKEN_MODEL_PATH="$MODELS_DIR/$cand"
      break
    fi
  done
fi
MODEL_PATH="${FREETOKEN_MODEL_PATH:-$MODELS_DIR/DeepSeek-V4-Flash-REAP-145B}"
SERVER_HOST="${FREETOKEN_HOST:-127.0.0.1}"
SERVER_PORT="${FREETOKEN_PORT:-1919}"

export CUDA_HOME="${CUDA_HOME:-$CONDA_PREFIX}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
# flashinfer-cubin (PyPI, 0.7.1) と flashinfer-python (0.6.18.post1, [accel] の pinned 範囲) は
# バージョンが一致しないため、公式が用意しているチェック無効化を使う (jit-cache は python 側に一致済み)。
export FLASHINFER_DISABLE_VERSION_CHECK="${FLASHINFER_DISABLE_VERSION_CHECK:-1}"

[ -x "$FT_BIN" ] || { echo "ft がありません ($FT_BIN) — 先に: pixi run install-wheels" >&2; exit 1; }
for required in config.json model.safetensors.index.json inference/config.json encoding/encoding_dsv4.py; do
  [ -f "$MODEL_PATH/$required" ] || { echo "モデルが不完全: $MODEL_PATH/$required が無い" >&2; exit 1; }
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ロード前に読み込み済みの page cache を掃除し、メモリガードを並走させる
# (cgroup 上限がタイトな環境でロード中のリクレーム暴走 = フリーズを防ぐ)。
python3 "$SCRIPT_DIR/evict-cache.py" "${FREETOKEN_MODELS:-$HOME/Desktop/ft-models}" >/dev/null 2>&1 || true
if [ "${FREETOKEN_GUARD:-1}" = "1" ]; then
  bash "$SCRIPT_DIR/mem-guard.sh" > /tmp/ft-mem-guard.log 2>&1 &
  echo "[serve] mem-guard 起動 (ログ: /tmp/ft-mem-guard.log, FREETOKEN_GUARD=0 で無効化)"
fi

exec "$FT_BIN" serve \
  --model-path "$MODEL_PATH" \
  --host "$SERVER_HOST" --port "$SERVER_PORT" \
  --moe-backend "${FREETOKEN_MOE_BACKEND:-hybrid}" \
  --moe-cache-auto \
  --kv-reserve-tokens "${FREETOKEN_KV_RESERVE_TOKENS:-131072}" \
  --max-seq-len-override "${FREETOKEN_MAX_SEQ_LEN:-131072}" \
  --max-running-requests "${FREETOKEN_MAX_RUNNING_REQUESTS:-1}" \
  --expert-load "${FREETOKEN_EXPERT_LOAD:-auto}" \
  --default-reasoning-effort "${FREETOKEN_DEFAULT_REASONING_EFFORT:-none}" \
  --enable-cache-report \
  "$@"
