#!/usr/bin/env bash
# Serve deepseek-ai/DeepSeek-V4.1-Flash (552B backbone + 196B Engram): a 20-layer causal
# encoder + 20-layer decoder (CED), CSA2 sparse attention with cross-layer KV/index reuse,
# 384 DeepSeek-FP4 routed experts per layer (top-6 + 1 shared), Engram n-gram memory at
# layers 1 and 14, and the 3-stage DSpark block drafter (128 experts top-3 per stage).
#
# Memory on this box (377 GB RAM, RTX 5090 32 GiB), from the checkpoint's safetensors headers:
#   routed experts 40 x 384 x 17.93 MiB = 269 GiB  -> pinned host RAM, byte-exact FP4
#   DSpark stage experts 3 x 128            = 6.7 GiB -> pinned host RAM
#   Engram tables layers 1/14 (FP8 + E8M0)  = 203 GB  -> NOT in RAM: memory-mapped from the two
#     shards (model-00047/48) on NVMe, page cache as the hot-row cache (FREETOKEN_ENGRAM_DIR)
#   dense FP8 (32x32 blocks) + heads        ~ 12 GB   -> resident on the GPU
# Every decoded token touches 240 routed experts = 4.3 GB of expert bytes; ~5% of the experts
# fit in the VRAM slot cache and DeepSeek's routing has no locality, so decode is bound by host
# expert bandwidth (hybrid: CPU MoE + PCIe). MEASURED on this box (2026-09-10/11, greedy,
# scripts/bench-decode.py --tokens 640 --runs 3): decode 24.1 (bench) / 24.0 (english) / 24.6
# (code) / 28.7 (japanese) tok/s at 770 slots; TTFT 0.8-1.4 s below 64 prompt tokens (per-route
# expert path), ~5.2 s from 64 tokens up (the whole-layer stream of 275 GB of experts over PCIe
# at ~51 GB/s), 1622 tokens 5.2 s. `ft bench bw --dtype ds_fp4 --model dsv4.1`: CPU MoE
# 123.9 GB/s vs PCIe gather 51.2 GB/s -> hybrid (30% of misses fetched). Boot is ~15 min: the
# 275 GB of expert banks read from this NTFS volume at ~345 MB/s (ext4 would be ~3 min).
# `--moe-cache-auto` gives 770 slots with the default reserve, 604 with the 3 GiB reserve below.
# Long context (2026-09-11): the pool caps prefill chunks at FREETOKEN_DSV4_PREFILL_CHUNK (8192)
# and --gpu-reserve-mb 4096 below keeps room for a chunk's fp32 hc streams (551 slots, decode
# 23-24 tok/s): prefill ~1000 tok/s from 32K up, needle 200K 5/5. Gates: GSM8K-150 0.94,
# MMLU-chat 0.905. DSpark: drafter agreement 100% (row 1) but 15-18 tok/s prose -> off.
# Native Linux only: WSL2 exposes 188 GB and pins 40% of RAM; native Windows cannot pin
# these banks (see docs). Download (475 GiB; keep inference/ and encoding/):
#   hf download deepseek-ai/DeepSeek-V4.1-Flash \
#     --local-dir "/media/nuko/New Volume/DeepSeek-V4.1-Flash" --max-workers 8
# Then put the two Engram shards on an ext4 drive (NTFS random 4 KiB reads are slow):
#   rsync -a --progress "/media/nuko/New Volume/DeepSeek-V4.1-Flash"/model-0004{7,8}-of-00048.safetensors \
#     /mnt/ftw/DeepSeek-V4.1-Flash-engram/
#   export FREETOKEN_ENGRAM_DIR=/mnt/ftw/DeepSeek-V4.1-Flash-engram

set -euo pipefail

readonly FT_VENV="${FREETOKEN_VENV:-${FREETOKEN_HOME:-$HOME/.freetoken}/venv}"
readonly FT_BIN="${FREETOKEN_FT_BIN:-$FT_VENV/bin/ft}"
readonly MODEL_PATH="${FREETOKEN_MODEL_PATH:-${FREETOKEN_HOME:-$HOME/.freetoken}/models/DeepSeek-V4.1-Flash}"
readonly ENGRAM_DIR="${FREETOKEN_ENGRAM_DIR:-$MODEL_PATH}"
readonly SERVER_HOST="${FREETOKEN_HOST:-127.0.0.1}"
readonly SERVER_PORT="${FREETOKEN_PORT:-1919}"

if [[ ! -x "$FT_BIN" ]]; then
  printf 'FreeToken is not installed at %s\n' "$FT_BIN" >&2
  exit 1
fi
for required in config.json model.safetensors.index.json inference/config.json encoding/encoding.py tokenizer.json; do
  if [[ ! -f "$MODEL_PATH/$required" ]]; then
    printf 'The DeepSeek-V4.1-Flash checkpoint is incomplete or missing at %s (no %s)\n' "$MODEL_PATH" "$required" >&2
    exit 1
  fi
done
# The Engram tables are read straight out of their safetensors shards (row lookups, not a
# full load), so those two shards must be reachable under FREETOKEN_ENGRAM_DIR (or the
# checkpoint dir). Shard numbers come from the published index (layers.1 -> 47, layers.14 -> 48).
for shard in model-00047-of-00048.safetensors model-00048-of-00048.safetensors; do
  if [[ ! -f "$ENGRAM_DIR/$shard" && ! -f "$MODEL_PATH/$shard" ]]; then
    printf 'Engram shard %s not found under FREETOKEN_ENGRAM_DIR=%s or the checkpoint dir\n' "$shard" "$ENGRAM_DIR" >&2
    exit 1
  fi
done
engram_fs="$(stat -f -c %T "$ENGRAM_DIR" 2>/dev/null || true)"
case "$engram_fs" in
  fuseblk|ntfs|ntfs3|9p)
    printf 'warning: FREETOKEN_ENGRAM_DIR=%s is on a %s filesystem; Engram row lookups are random 4 KiB reads and run an order of magnitude faster from ext4 (see the header comment)\n' \
      "$ENGRAM_DIR" "$engram_fs" >&2 ;;
esac
if [[ -r /proc/version ]] && grep -qi microsoft /proc/version; then
  if [[ "${FREETOKEN_ALLOW_WSL:-0}" != "1" ]]; then
    printf 'Refusing to serve DeepSeek-V4.1-Flash under WSL2: the pinned expert banks need ~276 GiB of host RAM and WSL pins only 40%% of its share. Boot the native Linux side, or set FREETOKEN_ALLOW_WSL=1 to try the split-residency fallback.\n' >&2
    exit 1
  fi
fi
if [[ -r /proc/meminfo ]]; then
  mem_avail_kb="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
  if [[ -n "$mem_avail_kb" && "$mem_avail_kb" -lt $((300 * 1024 * 1024)) && "${FREETOKEN_HOST_RAM_PREFLIGHT:-1}" != "0" ]]; then
    printf 'Only %d GiB of host RAM is available; the V4.1 expert banks need ~276 GiB pinned plus headroom (set FREETOKEN_HOST_RAM_PREFLIGHT=0 to override)\n' $((mem_avail_kb / 1024 / 1024)) >&2
    exit 1
  fi
fi
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

# Model-side knobs read by the V4.1 port (env, like the other DSV4/Qwen knobs):
#   FREETOKEN_DSV41_PREFILL_MODE     ced (default: encoder-only prefill + decoder SWA bounded
#                                    replay, the tech report's deployment) | full (all 40
#                                    layers, reference-exact; the quality oracle)
#   FREETOKEN_ENGRAM_DIR             where the two Engram shards live (default: checkpoint dir)
#   FREETOKEN_ENGRAM_IO              mmap (default, measured) | pread
#   FREETOKEN_ENGRAM_READERS         fetch threads (default 128, measured)
#   FREETOKEN_ENGRAM_STATS_LOG=N     print [engram-stats] every N steps
#   FREETOKEN_DSPARK_CONF_THRESHOLD  confidence-scheduled verify length (default 0.5)
export FREETOKEN_DSV41_PREFILL_MODE="${FREETOKEN_DSV41_PREFILL_MODE:-ced}"
export FREETOKEN_ENGRAM_DIR="$ENGRAM_DIR"
# Measured on the real shards (ext4, 2026-09-10, scripts/engram-bench.py): mmap is 25x faster
# than pread on the hot prefill path (67 ms vs 1.75 s per 8K-token chunk) and 128 readers
# give the best cold prefill (0.5 s per 8K chunk); decode fetches are 0.2-3 ms either way.
export FREETOKEN_ENGRAM_IO="${FREETOKEN_ENGRAM_IO:-mmap}"
export FREETOKEN_ENGRAM_READERS="${FREETOKEN_ENGRAM_READERS:-128}"

args=(
  serve
  --model-path "$MODEL_PATH"
  --host "$SERVER_HOST"
  --port "$SERVER_PORT"
  # Routed experts are always served from the offload cache. hybrid = GPU slot cache + CPU
  # co-compute of the misses (the 0731 bench profile picked hybrid for this CPU; re-run
  # `ft bench bw --dtype ds_fp4 --model dsv4.1` for the 17.9 MiB expert geometry).
  --moe-backend "${FREETOKEN_MOE_BACKEND:-hybrid}"
  # Fill the VRAM left after the resident dense weights and the KV floor below with expert
  # slots (17.93 MiB each). Every slot raises the cache hit rate, which is the decode budget.
  --moe-cache-auto
  # V4.1 global KV is ~3.2 KB/token in the bf16 v1 pools (890 B/token once FP4 code storage
  # lands), so a 128K floor costs well under 1 GB; the 128-token window tier is sized off the
  # swa ratio.
  --kv-reserve-tokens "${FREETOKEN_KV_RESERVE_TOKENS:-131072}"
  # Native limit is 1M positions; bound the served context (and the two YaRN rope tables) to
  # the KV floor.
  --max-seq-len-override "${FREETOKEN_MAX_SEQ_LEN:-131072}"
  # Single-user workload: one running request keeps the decode batch (and per-row scratch)
  # minimal so the expert cache gets the VRAM.
  --max-running-requests "${FREETOKEN_MAX_RUNNING_REQUESTS:-1}"
  # Prefill transients are not bounded by the slot-cache auto-sizing: the ratio-1 indexer
  # scores [S, T] fp32 per layer call (0.27 GB at an 8K chunk, 2.7 GB at the scheduler's
  # default ~26K chunk = half the window pool) plus the grouped-GEMM intermediates. Measured
  # (2026-09-10): an ~8K-token prompt OOM'd with the 2.4 GiB left after 770 slots. Cap the
  # chunk at 8K and keep ~3 GiB free (~170 slots); the plan's query sub-chunking is the fix
  # that gives the slots back.
  # NOTE: the engine raises --max-prefill-length to max_seq_len for DSV4 (one prompt = one CED
  # chunk when it fits); the chunk cap that actually applies is the pool's, set by
  # FREETOKEN_DSV4_PREFILL_CHUNK (default 8192). An 8K chunk's fp32 hc streams peak ~2.7 GB.
  --gpu-reserve-mb "${FREETOKEN_GPU_RESERVE_MB:-4096}"
  --expert-load "${FREETOKEN_EXPERT_LOAD:-auto}"
  # Ordinary Open WebUI requests omit reasoning_effort: keep thinking off for those while
  # honoring explicit selections (V4.1 accepts low/medium/high/xhigh or an int budget).
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
# DSpark (the checkpoint's own 5-token block drafter, 3 stages) is OFF by default: with ~6%
# of the experts cached every verify row brings its own ~34 distinct expert misses per layer,
# and the 0731 measurement lost (15 vs 31 tok/s) at a fixed 6-row verify. FREETOKEN_SPECULATIVE=
# dspark turns it on with confidence-scheduled verify length (FREETOKEN_DSPARK_CONF_THRESHOLD,
# default 0.5) up to FREETOKEN_SPEC_DRAFT_TOKENS drafts (max 5). Measure before defaulting.
if [[ "${FREETOKEN_SPECULATIVE:-off}" != "off" ]]; then
  args+=(--speculative-algorithm "$FREETOKEN_SPECULATIVE")
  args+=(--speculative-num-draft-tokens "${FREETOKEN_SPEC_DRAFT_TOKENS:-5}")
fi

exec "$FT_BIN" "${args[@]}" "$@"
