# ft-pixi — FreeToken 運用ツールキット (pixi)

FreeToken (PentaCoxian フォーク) を pixi 環境でビルド・実行・配布するためのユーティリティ集。
すべて `pixi run <task>` で実行でき、パスは環境変数で上書きできる。

## 前提
- Linux x86_64 + NVIDIA GPU (driver r580+ / CUDA 13)
- pixi (`~/.local/bin/pixi`)

## 主なタスク

| タスク | 説明 |
| --- | --- |
| `pixi run doctor` | GPU/ドライバ/nvcc/torch/モデル/ディスク/cgroup のチェック |
| `pixi run build-wheels` | FreeToken + kernel-cache wheel をビルド (既定で sm 8.0/8.6/8.9/9.0/10.0/12.0 の6アーキ) |
| `pixi run install-wheels` | ビルド済み wheel + accel 依存をこの環境へ導入 |
| `pixi run release` | dist/ の wheel と運用ツールキットを GitHub Releases へ公開 |
| `pixi run install-from-release` | Releases から wheel を取得して導入 (別マシン/クイック導入) |
| `pixi run serve` | DeepSeek-V4-Flash-0731 を配信 (OpenAI API :1919) |
| `pixi run serve-dsv41` | DeepSeek-V4.1-Flash 用 (要件未達なら理由を表示) |
| `pixi run status` / `pixi run stop` | 状態表示 / 停止 |
| `pixi run chat` / `pixi run bench` | チャット疎通 / デコード tok/s 計測 |

## 環境変数 (抜粋)

| 変数 | 既定 | 用途 |
| --- | --- | --- |
| `FREETOKEN_DIR` | `~/Desktop/FreeToken` | FreeToken のチェックアウト |
| `FREETOKEN_MODELS` | `~/Desktop/ft-models` | モデル置き場 |
| `FREETOKEN_KERNEL_CACHE_ARCHES` | (未設定=6アーキ既定) | ビルドする SASS アーキの上書き (例 `"8.0 8.9"`) |
| `FREETOKEN_PORT` | `1919` | サーバーポート |
| `FREETOKEN_MOE_BACKEND` | `hybrid` | `offload` 等に変更可 |
| `FREETOKEN_RELEASE_REPO` | `yuki-inaho/FreeToken` | リリース先 |
| `FT_BENCH_RUNS` / `FT_BENCH_TOKENS` | `3` / `128` | bench の回数 / 生成トークン |

## 標準フロー

```sh
pixi run doctor          # 環境確認
pixi run build-wheels    # 初回のみ (~10-40分, 6アーキ分の TVM カーネル)
pixi run install-wheels  # この環境へ導入
pixi run serve           # 起動 → 別ターミナルで pixi run bench
pixi run release         # 配布物を公開
```
