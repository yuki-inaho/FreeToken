#!/usr/bin/env bash
# cgroup メモリ監視 + ページキャッシュ追い出し + 緊急停止 (pixi run guard / serve が自動起動)
#
# 背景: FreeToken の DSV4 ロードはホスト RAM に ~133GiB をピン留めする。
#       cgroup 上限 139.4GiB (swap なし) の下では、読み込み中に増える page cache を
#       能動的に追い出さないとリクレームが暴走してマシンが固まることがある。
#       このガードは (1) 閾値超過でモデルファイルの page cache を DONTNEED、
#       (2) 危険域が続けば ft serve を停止してマシンを守る。
#
# 環境変数: MEM_GUARD_WARN_GIB(既定131) / MEM_GUARD_KILL_GIB(既定138) /
#           MEM_GUARD_INTERVAL(既定5秒) / FREETOKEN_MODELS
set -uo pipefail

WARN_GIB="${MEM_GUARD_WARN_GIB:-131}"
KILL_GIB="${MEM_GUARD_KILL_GIB:-138}"
INTERVAL="${MEM_GUARD_INTERVAL:-5}"
MODELS="${FREETOKEN_MODELS:-$HOME/Desktop/ft-models}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cur_gib() { awk '{printf "%.2f", $1/1073741824}' /sys/fs/cgroup/memory.current 2>/dev/null || echo 0; }

echo "[mem-guard] warn=${WARN_GIB}GiB kill=${KILL_GIB}GiB interval=${INTERVAL}s models=${MODELS}"
hot=0
while :; do
  if ! pgrep -f "ft serve" >/dev/null 2>&1; then
    [ "${MEM_GUARD_STAY:-0}" = "1" ] && sleep "$INTERVAL" && continue
    echo "[mem-guard] ft serve が居ないので終了"
    exit 0
  fi
  c="$(cur_gib)"
  if awk "BEGIN{exit !($c > $WARN_GIB)}"; then
    python3 "$SCRIPT_DIR/evict-cache.py" "$MODELS" >/dev/null 2>&1 || true
    hot=$((hot + 1))
    echo "[mem-guard] ${c}GiB — page cache 追い出し (${hot}回目)"
  else
    hot=0
  fi
  if awk "BEGIN{exit !($c > $KILL_GIB)}" && [ "$hot" -ge 2 ]; then
    echo "[mem-guard] ${c}GiB > ${KILL_GIB}GiB が継続 — マシン保護のため ft serve を停止します" >&2
    pkill -f "ft serve" >/dev/null 2>&1 || true
    exit 3
  fi
  sleep "$INTERVAL"
done
