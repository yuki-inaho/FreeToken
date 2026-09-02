#!/usr/bin/env bash
# Serve deepseek-ai/DeepSeek-V4-Flash-0731 (304B MoE: 43 layers x 256 DeepSeek-FP4 routed
# experts, top-6 + 1 shared, MLA sliding window + learned KV compression + Lightning Indexer).
#
# The routed experts (137 GiB of packed FP4) live in pinned host RAM and stream into a VRAM
# slot cache on demand; every decoded token touches 258 experts (3.3 GB of expert weights),
# so decode on one 32 GiB card is bandwidth-bound. `ft bench bw --dtype ds_fp4` measured on
# this box (2026-08-31): CPU MoE kernel 127.8 GB/s vs PCIe gather 50.4 GB/s -> hybrid, with
# ~28% of each step's misses fetched over PCIe while the CPU computes the rest. Measured
# warm decode: ~31 tok/s (code 32-33, Japanese 35), TTFT ~2.6 s on short prompts (prefill
# streams every expert once), ~1,900 tok/s prefill at 8K; the hit rate is ~flat in cache
# size (load-balanced routing), so ~31 tok/s is this card's ceiling. Details and the
# DSpark measurements live in LOCAL_SETUP.md ("DeepSeek-V4-Flash-0731").
#
# Download (167 GB; keep the inference/ and encoding/ subdirs, FreeToken reads both):
#   hf download deepseek-ai/DeepSeek-V4-Flash-0731 \
#     --local-dir "/media/nuko/New Volume/DeepSeek-V4-Flash-0731" --max-workers 8

set -euo pipefail

readonly FT_VENV="${FREETOKEN_VENV:-/home/nuko/.freetoken/venv}"
readonly FT_BIN="${FREETOKEN_FT_BIN:-$FT_VENV/bin/ft}"
readonly MODEL_PATH="${FREETOKEN_MODEL_PATH:-/media/nuko/New Volume/DeepSeek-V4-Flash-0731}"
readonly SERVER_HOST="${FREETOKEN_HOST:-127.0.0.1}"
readonly SERVER_PORT="${FREETOKEN_PORT:-1919}"

if [[ ! -x "$FT_BIN" ]]; then
  printf 'FreeToken is not installed at %s\n' "$FT_BIN" >&2
  exit 1
fi
for required in config.json model.safetensors.index.json inference/config.json encoding/encoding_dsv4.py; do
  if [[ ! -f "$MODEL_PATH/$required" ]]; then
    printf 'The DeepSeek-V4-Flash checkpoint is incomplete or missing at %s (no %s)\n' "$MODEL_PATH" "$required" >&2
    exit 1
  fi
done

if ! command -v nvidia-smi >/dev/null 2>&1; then
  printf 'nvidia-smi is unavailable; an NVIDIA driver is required.\n' >&2
  exit 1
fi

# The CUDA 13 compiler/runtime is installed alongside FreeToken (see serve-qwen38.sh).
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
  # Routed experts are always served from the offload cache (the loader refuses to keep
  # them resident). hybrid = GPU slot cache + CPU co-compute of the misses; offload = PCIe
  # only. The bench profile above picks hybrid for this CPU.
  --moe-backend "${FREETOKEN_MOE_BACKEND:-hybrid}"
  # Fill the VRAM left after the resident dense weights (~10 GB) and the KV floor below
  # with expert slots (12.75 MiB each). Every slot raises the cache hit rate, which is the
  # whole decode budget on this model.
  --moe-cache-auto
  # DSV4 KV is cheap per full-history token (~7 KB: compressed + indexer tiers), so a 128K
  # floor costs well under 1 GB; the 128-token window tier is sized off the swa ratio.
  --kv-reserve-tokens "${FREETOKEN_KV_RESERVE_TOKENS:-131072}"
  # The model's native limit is 1M positions; the two YaRN rope tables cost 256 MB each at
  # that length. Bound the served context (and the tables) to the KV floor.
  --max-seq-len-override "${FREETOKEN_MAX_SEQ_LEN:-131072}"
  # Open WebUI is a single-user workload here; one running request keeps the decode batch
  # (and the per-row scratch) minimal so the expert cache gets the VRAM.
  --max-running-requests "${FREETOKEN_MAX_RUNNING_REQUESTS:-1}"
  --expert-load "${FREETOKEN_EXPERT_LOAD:-auto}"
  # Ordinary Open WebUI requests omit reasoning_effort: keep thinking off for those while
  # honoring explicit low/medium/high/xhigh selections.
  --default-reasoning-effort "${FREETOKEN_DEFAULT_REASONING_EFFORT:-none}"
  --enable-cache-report
)

# Hybrid tuning knobs (defaults come from the ft bench bw profile / 24 physical cores).
if [[ -n "${FREETOKEN_MOE_CPU_THREADS:-}" ]]; then
  args+=(--moe-cpu-threads "$FREETOKEN_MOE_CPU_THREADS")
fi
if [[ -n "${FREETOKEN_MOE_HYBRID_MAX_FETCH:-}" ]]; then
  args+=(--moe-hybrid-max-fetch "$FREETOKEN_MOE_HYBRID_MAX_FETCH")
fi
# Speculative decoding with DSpark (the checkpoint's own 5-token block drafter) is
# implemented and correct (3-4 accepted tokens per step) but OFF here on purpose: on this
# card the expert cache holds ~10% of the experts, so every verify row brings its own ~230
# expert misses and a 6-row verify costs ~5.7x a one-row step -- measured 15 tok/s greedy
# vs 31 plain (2026-08-31). FREETOKEN_SPECULATIVE=dspark turns it on (pays only on GPUs
# that cache most experts).
if [[ "${FREETOKEN_SPECULATIVE:-off}" != "off" ]]; then
  args+=(--speculative-algorithm "$FREETOKEN_SPECULATIVE")
  args+=(--speculative-num-draft-tokens "${FREETOKEN_SPEC_DRAFT_TOKENS:-5}")
fi

exec "$FT_BIN" "${args[@]}" "$@"
