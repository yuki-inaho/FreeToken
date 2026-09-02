#!/usr/bin/env bash
# Serve orcarouter/Qwen3.8-27B-Uncensored-NVFP4 (dense Qwen3.8-27B, compressed-tensors
# mixed NVFP4+FP8) with MTP speculative decoding. Measured on the RTX 5090 workstation
# (2026-08-31): ~64 tok/s plain decode; k=3 speculation + the NVFP4 draft lm_head ~97 tok/s
# on the benchmark prompt (English prose ~91, code ~107, Japanese ~78), TTFT ~0.07 s at
# short prompts, ~3.5K tok/s prefill. The whole model is resident (no expert offload).

set -euo pipefail

readonly FT_VENV="${FREETOKEN_VENV:-/home/nuko/.freetoken/venv}"
readonly FT_BIN="${FREETOKEN_FT_BIN:-$FT_VENV/bin/ft}"
readonly MODEL_PATH="${FREETOKEN_MODEL_PATH:-/media/nuko/New Volume/Qwen3.8-27B-Uncensored-NVFP4}"
readonly SERVER_HOST="${FREETOKEN_HOST:-127.0.0.1}"
readonly SERVER_PORT="${FREETOKEN_PORT:-1919}"

if [[ ! -x "$FT_BIN" ]]; then
  printf 'FreeToken is not installed at %s\n' "$FT_BIN" >&2
  exit 1
fi
if [[ ! -f "$MODEL_PATH/config.json" || ! -f "$MODEL_PATH/model.safetensors.index.json" ]]; then
  printf 'The 27B checkpoint is incomplete or missing at %s\n' "$MODEL_PATH" >&2
  exit 1
fi

cuda_home_default="$FT_VENV/lib/python3.12/site-packages/nvidia/cu13"
export CUDA_HOME="${FREETOKEN_CUDA_HOME:-$cuda_home_default}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LIBRARY_PATH="$CUDA_HOME/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"

args=(
  serve
  --model-path "$MODEL_PATH"
  --host "$SERVER_HOST"
  --port "$SERVER_PORT"
  --max-running-requests "${FREETOKEN_MAX_RUNNING_REQUESTS:-1}"
  # head_dim 256 makes the paged KV ~64 KB/token (16 full-attention layers); the
  # resident weights leave ~4 GiB for it: 16K tokens by default (32K still fits, tighter).
  --kv-reserve-tokens "${FREETOKEN_KV_RESERVE_TOKENS:-16384}"
  --default-reasoning-effort "${FREETOKEN_DEFAULT_REASONING_EFFORT:-none}"
  --enable-cache-report
  # MTP speculation: extra verify rows cost ~0.3 ms each on this dense model, so deep
  # drafting pays -- four drafts measured 2.7-3.7 tokens/step (bench 107, code ~132 tok/s)
  # vs three at 97. FREETOKEN_SPECULATIVE=off disables.
  --speculative-algorithm "${FREETOKEN_SPECULATIVE:-mtp}"
  --speculative-num-draft-tokens "${FREETOKEN_SPEC_DRAFT_TOKENS:-4}"
)

exec "$FT_BIN" "${args[@]}" "$@"
