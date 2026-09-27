#!/usr/bin/env bash

set -euo pipefail

readonly FT_VENV="${FREETOKEN_VENV:-${FREETOKEN_HOME:-$HOME/.freetoken}/venv}"
readonly FT_BIN="${FREETOKEN_FT_BIN:-$FT_VENV/bin/ft}"
readonly MODEL_PATH="${FREETOKEN_MODEL_PATH:-${FREETOKEN_HOME:-$HOME/.freetoken}/models/Qwen3.8-Flash-Next-NVFP4}"
readonly SERVER_HOST="${FREETOKEN_HOST:-127.0.0.1}"
readonly SERVER_PORT="${FREETOKEN_PORT:-1919}"

if [[ ! -x "$FT_BIN" ]]; then
  printf 'FreeToken is not installed at %s\n' "$FT_BIN" >&2
  exit 1
fi

if [[ ! -f "$MODEL_PATH/config.json" || ! -f "$MODEL_PATH/model.safetensors.index.json" ]]; then
  printf 'The Qwen checkpoint is incomplete or missing at %s\n' "$MODEL_PATH" >&2
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  printf 'nvidia-smi is unavailable; an NVIDIA driver is required.\n' >&2
  exit 1
fi

driver_version="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)"
driver_major="${driver_version%%.*}"
if [[ ! "$driver_major" =~ ^[0-9]+$ || "$driver_major" -lt 580 ]]; then
  printf 'NVIDIA driver %s is too old; FreeToken requires r580 or newer.\n' "$driver_version" >&2
  printf 'Run scripts/install-nvidia-driver.sh, reboot, and retry.\n' >&2
  exit 1
fi

# The CUDA 13 compiler/runtime is installed alongside FreeToken, keeping the
# project independent of any older system CUDA toolkit.
cuda_home_default="$FT_VENV/lib/python3.12/site-packages/nvidia/cu13"
export CUDA_HOME="${FREETOKEN_CUDA_HOME:-$cuda_home_default}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# NVIDIA's pip CUDA toolkit uses `lib`, while TVM-FFI's JIT linker emits a
# `lib64` search path. LIBRARY_PATH supplies the link-time fallback; the
# unversioned libcudart symlink is created during local setup.
export LIBRARY_PATH="$CUDA_HOME/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"

args=(
  serve
  --model-path "$MODEL_PATH"
  --host "$SERVER_HOST"
  --port "$SERVER_PORT"
  # MTP speculative decoding: ~90 tok/s warm decode vs ~72 without, same 128K
  # context; the emitted distribution is exact for every sampling family (chain
  # speculative sampling). Set FREETOKEN_SPECULATIVE=off to disable.
  --speculative-algorithm "${FREETOKEN_SPECULATIVE:-mtp}"
  # Two draft tokens per step: draft 2 is a true chained MTP row (the head
  # re-consumes its own hidden state), replayed via a per-row CUDA graph.
  # Measured 2.0-2.7 tokens/step by content vs 1.7-1.9 at one draft. A third
  # draft (FREETOKEN_SPEC_DRAFT_TOKENS=3, 2.3-3.3 tokens/step) pays only when
  # the expert cache exceeds ~6,000 slots: pair it with
  # FREETOKEN_KV_RESERVE_TOKENS=32768 for the fastest measured profile
  # (~95-100 tok/s warm decode at 32K context vs ~90 at the 128K default).
  --speculative-num-draft-tokens "${FREETOKEN_SPEC_DRAFT_TOKENS:-2}"
  --speculative-adaptive-k-max "${FREETOKEN_SPEC_ADAPTIVE_K_MAX:-3}"
  # hybrid computes each decode step's expert-cache misses on the CPU in parallel
  # with the GPU cache hits. For plain one-row decode, pure GPU offload used to win
  # by ~14% here -- but the speculative verify step processes 2 rows and its larger
  # miss set flips the comparison: hybrid takes the verify forward from ~20.6 ms to
  # ~19.2 ms (measured 2026-08-30, 5,060 cache slots). Prefill is unaffected.
  --moe-backend "${FREETOKEN_MOE_BACKEND:-hybrid}"
  # The FlashInfer b12x path currently overflows a 32-bit CUTLASS descriptor
  # with the large cache auto-selected on this 32 GiB card. Triton is the
  # supported portable path and handles the full cache correctly.
  --nvfp4-backend "${FREETOKEN_NVFP4_BACKEND:-triton}"
  # Open WebUI is primarily an interactive, single-user workload here. A
  # batch size of one leaves room for roughly 12% more cached experts.
  --max-running-requests "${FREETOKEN_MAX_RUNNING_REQUESTS:-1}"
  --expert-load "${FREETOKEN_EXPERT_LOAD:-auto}"
  # Ordinary Open WebUI requests omit reasoning_effort. Disable thinking for
  # those requests while preserving explicit low/medium/high/xhigh selections.
  --default-reasoning-effort "${FREETOKEN_DEFAULT_REASONING_EFFORT:-none}"
  --enable-cache-report
  # Only 12 of 48 layers hold paged KV. With the fp8 KV cache below they cost
  # ~14 KB/token, so a 256K floor fits in what 128K used to take: measured
  # 2026-09-01, decode stays within noise of the bf16/128K control (back-to-back
  # boots), GSM8K 90.0 vs 90.7, MMLU 88.3 vs 87.7, needle retrieval 5/5 at 191K
  # depth, ~77 tok/s decoding at 200K depth. FREETOKEN_KV_RESERVE_TOKENS=131072
  # FREETOKEN_KV_CACHE_DTYPE=bf16 restores the previous configuration exactly.
  --kv-reserve-tokens "${FREETOKEN_KV_RESERVE_TOKENS:-65536}"
  --kv-elastic-max-tokens "${FREETOKEN_KV_MAX_TOKENS:-262144}"
  --kv-elastic-idle-seconds "${FREETOKEN_KV_ELASTIC_IDLE_S:-300}"
  --gpu-reserve-mb "${FREETOKEN_GPU_RESERVE_MB:-0}"
  --linear-state-cache "${FREETOKEN_LINEAR_STATE_CACHE:-1}"
  # KV storage dtype: 'fp8' stores the QSA layers' K/V as e4m3 codes with a
  # per-(token, head) scale.
  --kv-cache-dtype "${FREETOKEN_KV_CACHE_DTYPE:-fp8}"
  # Resident dense linears (GDN/attention/HC mixes/shared expert/lm_head) are bf16 in the
  # checkpoint: ~11.5 GB of weight reads per verify step, the 1.27 GB lm_head three times
  # (graphs A/B/C). FREETOKEN_DENSE_QUANT=fp8 quantizes them to per-channel FP8 at load
  # (halves those reads, frees ~4 GB -> ~1,500 more expert-cache slots). Default since
  # 2026-09-01: GSM8K-150 90.7 (bf16 90.0), MMLU-chat 87.2 (bf16 87.7-88.3, within the
  # error bar); decode ~126 tok/s prose / ~136 code vs ~88 / ~100 bf16.
  # FREETOKEN_DENSE_QUANT=auto restores the bf16 dense weights byte-identically.
  --dense-quant "${FREETOKEN_DENSE_QUANT:-fp8}"
  # Larger prefill chunks would cut long-prompt TTFT almost linearly (each
  # chunk streams every non-resident expert once), but the GDN prefill
  # kernels hold several hundred MB of transients per 8K tokens of chunk:
  # 16K-token chunks OOM this 32 GiB card even at --memory-ratio 0.88, and
  # buying more headroom would push the expert cache off its throughput
  # plateau. The default 8192-token chunk is the measured safe maximum.
  # Copy cache-resident experts device-side during prefill and stream only
  # the misses over PCIe (falls back automatically where unsupported).
  --moe-prefill-hit-d2d
  # In-graph cross-layer expert lookahead: layer J speculatively routes layer
  # J+N with the cheap xnorm proxy and streams the predicted experts in on a
  # side stream while J..J+N-1 compute. Measured 2026-09-01 at the built-in
  # optimum (N=2, top-4/row, 23 hooks, xnorm): ~+1 tok/s decode and 15-20%
  # fewer expert-cache misses vs off; never changes routing or output.
  # Layer-lookahead prefetch is OFF by default since 2026-09-01: at >= 8,000 expert slots
  # (miss rate ~0.06) its 23 hooks cost more than the misses they cover (measured -3..-6%).
  # FREETOKEN_MOE_PREFETCH=layer re-enables it; the
  # FREETOKEN_PREFETCH_* envs (see moe/prefetch.py) tune it.
  --moe-prefetch "${FREETOKEN_MOE_PREFETCH:-off}"
)

if [[ -n "${FREETOKEN_MAX_SEQ_LEN:-}" ]]; then
  args+=(--max-seq-len-override "$FREETOKEN_MAX_SEQ_LEN")
fi

# GDN recurrent-state storage dtype (2026-09-02): float16 halves the state pool + spec scratch
# (+~265 expert slots) and the per-step state traffic; gates GSM8K-150 88.7 / MMLU 87.4 /
# needle 5/5 (bf16 failed GSM8K at 87.3). FREETOKEN_MAMBA_SSM_DTYPE=float32 restores fp32.
export FREETOKEN_MAMBA_SSM_DTYPE="${FREETOKEN_MAMBA_SSM_DTYPE:-float16}"

exec "$FT_BIN" "${args[@]}" "$@"
