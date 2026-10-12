# 実行レポート: DeepSeek-V4-Flash-REAP-145B on CMP 170HX (2026-10-12)

FreeToken (本フォーク) を sm_80 の NVIDIA CMP 170HX 上で動かした記録。
本フォークのマルチアーキ prebuilt wheel (`wheels-2026.10.12`) と kernel-cache 修正を含む。

## 環境

| 項目 | 値 |
| --- | --- |
| GPU | NVIDIA CMP 170HX ×1 (VRAM 64 GiB, sm_80, ヘッドレス) |
| CPU / RAM | AMD EPYC 7663 56C/112T / 251 GiB (cgroup 上限 139.4 GiB, swap 0) |
| ストレージ | NVMe SSD 238 GB |
| OS / ドライバ | Linux (コンテナ) / 610.57.04 / CUDA 13.3 (pixi 環境) |
| エンジン | FreeToken (fork, branch `build-multi-arch-wheels`) |

## モデル

- **ludo-tech/DeepSeek-V4-Flash-REAP-145B-A13B** (REAP K128)
  - `deepseek-ai/DeepSeek-V4-Flash-0731` から routed experts を 256→128 / layer に REAP プルーニング
  - 78.6 GB (48 shards, safetensors, FP8 dense + MXFP4 experts)
  - 起動には公式チェックポイントの `inference/` をコピーし、`inference/config.json` の
    `n_routed_experts` を 128 に変更して同梱した

## 起動

```sh
# wheel の導入 (Release wheels-2026.10.12)
curl -fsSL https://github.com/yuki-inaho/FreeToken/releases/download/wheels-2026.10.12/install-from-release.sh | bash

# モデル
hf download ludo-tech/DeepSeek-V4-Flash-REAP-145B-A13B --local-dir DeepSeek-V4-Flash-REAP-145B

# 配信
FLASHINFER_DISABLE_VERSION_CHECK=1 ft serve \
  --model-path DeepSeek-V4-Flash-REAP-145B \
  --moe-backend hybrid --moe-cache-auto \
  --max-seq-len-override 131072 --max-running-requests 1 \
  --default-reasoning-effort none
```

## 計測結果

| 項目 | 実測 |
| --- | ---: |
| **decode (エンジン計測)** | **11.09 tok/s** (242 トークン生成, greedy) |
| decode (wall 中央値, n=3) | 10.27 tok/s (8.53 / 10.27 / 10.46) |
| **prefill** | **~447 tok/s** (14,348 トークン → 32.1 秒) |
| コンテキスト | 131,072 |
| VRAM 使用 | 63.7 GB (expert cache 自動確保) |
| ホストメモリ (cgroup) | 75.8 GiB |

参考: RTX 5090 (32 GiB) + 377 GiB RAM でフル V4-Flash を hybrid 配信した既存報告は 24–31 tok/s。
本機は sm_80 + REAP ハーフという条件で 11 tok/s。

## 補足: フル版と V4.1 が動かない理由 (記録)

- **DeepSeek-V4-Flash-0731 (フル)**: routed experts 143 GB (=133.2 GiB) を pinned host RAM に
  ロードする設計のため、cgroup 139.4 GiB 下ではロード中 (40/43 層, ~124 GiB バンク + 諸経費 ~15 GiB)
  に上限へ到達し停止する (実測 3 回)。本機では起動不可。
- **DeepSeek-V4.1-Flash**: チェックポイント 510 GB / routed experts 269 GiB pinned 要求。
  ディスク 250 GB・cgroup 139.4 GiB のため取得も常駐も不可能。NVMe expert tier (FTW コーパス)
  を使う既存報告は 1 TB 級 NVMe + 128 GB 級 RAM が前提。

## 再現性メモ

- wheel は commit `45b5882` からビルド (kernel-cache は sm_80/8.6/8.9/9.0/10.0/12.0 の 6 アーキ SASS 入り)。
- kernel-cache ビルドには本フォークの修正 (fast_index_copy の AOT 仕様から、カーネルが
  原理的にコンパイルできないサイズ 240/400B を除外) が必要。
- メモリがタイトな環境では `pixi-ops` の mem-guard (page cache の fadvise + 閾値超過で緊急停止) を併用。
