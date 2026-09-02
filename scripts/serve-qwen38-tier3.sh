#!/usr/bin/env bash
# Serve Qwen3.8-Flash-Next on a 16 GB VRAM / 48 GB RAM budget at 256K context: MoE experts AND
# the PLE n-gram table stream from NVMe (tier 3), and routing prefers experts that are in RAM.
#
# 48 GiB rather than the full 64 so a 64 GB box has headroom for everything else. That costs
# host residency (352 of 512 experts per layer instead of 458), which residency-aware routing
# then buys back: measured at 90K, +22 to +39 % decode (bench 58.1 -> 72.8, english 59.7 ->
# 72.8, code 56.5 -> 75.0, japanese 49.6 -> 69.0 tok/s) with GSM8K-150 (8-shot CoT, greedy,
# lm_eval) at 92.0 % strict / 93.3 % flexible against the 88.67 % BENCH.md record.
#
# For the older 64 GB-of-RAM shape: FREETOKEN_TIER3_HOST_SLOTS=458 FREETOKEN_MOE_RESIDENCY_ROUTING=0
#
# The expert corpus lives in an FTW conversion; host RAM holds only FREETOKEN_TIER3_HOST_SLOTS
# experts per layer and the rest are paged in on demand by the CPU MoE coordinator. Everything
# else (dense weights, the PLE n-gram table) still comes from the original HF checkpoint, so
# BOTH directories are needed -- ft checkpoint does not carry the PLE table.
#
# One-off conversion (needs a GPU, ~70 s, ~73 GiB):
#   ft checkpoint --model "$MODEL_PATH" --out "$FTW_PATH" --moe-backend offload
#
# Pool sizing: bytes = slots * num_moe_layers * bytes_per_expert (2.64 MiB for this model's
# NVFP4 banks, 48 layers) -- 355 slots ~ 44 GiB and 69% residency. The pool MUST exceed one
# step's distinct working set (max-running-requests * top_k) or experts get dropped; the
# launcher below fails at startup if it does not.
set -euo pipefail

readonly FT_VENV="${FREETOKEN_VENV:-/home/nuko/.freetoken/venv}"
readonly FT_BIN="${FREETOKEN_FT_BIN:-$FT_VENV/bin/ft}"
readonly MODEL_PATH="${FREETOKEN_MODEL_PATH:-/media/nuko/New Volume/Qwen3.8-Flash-Next-NVFP4}"
readonly FTW_PATH="${FREETOKEN_TIER3_FTW:-/media/nuko/New Volume/Qwen3.8-Flash-Next-FTW}"

[[ -d "$FTW_PATH" ]] || { echo "no FTW corpus at $FTW_PATH -- run ft checkpoint first" >&2; exit 1; }
[[ -f "$FTW_PATH/freetoken_weight.json" ]] || { echo "$FTW_PATH is not an FTW checkpoint" >&2; exit 1; }

# Emulating a smaller card honestly: --gpu-reserve-mb only steers the pool solver, so a config
# that overflows a 16 GB card still runs here. FREETOKEN_GPU_MEM_CAP_MB hard-caps the allocator,
# turning "would this fit?" into a question the 5090 can answer. Unset by default (a real 16 GB
# card needs no emulation); set it to 16384 when validating this profile's claim.
export FREETOKEN_GPU_MEM_CAP_MB="${FREETOKEN_GPU_MEM_CAP_MB:-0}"

cuda_home_default="$FT_VENV/lib/python3.12/site-packages/nvidia/cu13"
export CUDA_HOME="${FREETOKEN_CUDA_HOME:-$cuda_home_default}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LIBRARY_PATH="$CUDA_HOME/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"

export FREETOKEN_TIER3_FTW="$FTW_PATH"
# Pinned host budget at the default geometry:
#   expert pool     458 slots x 48 layers x 2.64 MiB = 56.8 GiB   (89% residency)
#   MTP draft layer 512 experts, full width, resident =  1.35 GiB
#   PLE row cache                                     =  2.0  GiB
#   prefill staging 2 x (512-458) experts             =  0.28 GiB
#   ---------------------------------------------------------------
#                                                       60.4 GiB
# plus ~2-3 GiB of process, so 64 GiB is tight -- drop to ~440 slots for margin, or raise
# FREETOKEN_TIER3_PLE_GB if the n-gram hit rate matters more than expert residency.
# Residency is the dominant knob: every point of it comes off the CPU executor's share.
export FREETOKEN_TIER3_HOST_SLOTS="${FREETOKEN_TIER3_HOST_SLOTS:-352}"   # ~48.8 GiB pinned
# The PLE n-gram table is 47.7 GiB fully resident, which would leave the expert pool ~14%
# residency and make decode NVMe-bandwidth-bound (measured 16 tok/s). A few GiB of row cache
# over the on-disk table is enough: traffic is 16 rows x 160 B per token.
export FREETOKEN_TIER3_PLE_GB="${FREETOKEN_TIER3_PLE_GB:-2}"
# Fill the pool from disk at boot (~11 s at 5.4 GB/s). Without it the residency map starts
# empty, so the ensure kernel can fetch nothing and every route falls to the CPU behind a disk
# read -- the cold-start cliff. Set 0 only to measure that.
export FREETOKEN_TIER3_PREWARM="${FREETOKEN_TIER3_PREWARM:-1}"
# NVFP4 target lm_head: halves graph A's per-step head read (636 -> 318 MB) and removes the
# separate draft copy, worth +236 expert slots here (2771 -> 3007) and ~3% on first-pass decode.
# It perturbs the emitted logits, so it stays OPT-IN until GSM8K/MMLU/needle have been run
# against it -- set FREETOKEN_LM_HEAD_NVFP4=1 to enable.
export FREETOKEN_LM_HEAD_NVFP4="${FREETOKEN_LM_HEAD_NVFP4:-0}"
export FREETOKEN_TIER3_READERS="${FREETOKEN_TIER3_READERS:-16}"
export FREETOKEN_MAMBA_SSM_DTYPE="${FREETOKEN_MAMBA_SSM_DTYPE:-float16}"

args=(
  --model-path "$MODEL_PATH"
  --host "${FREETOKEN_HOST:-127.0.0.1}" --port "${FREETOKEN_PORT:-1919}"
  # tier 3 fetches on the CPU executor's coordinator -- the only place a decode may stall
  --moe-backend hybrid
  --nvfp4-backend triton
  --moe-cache-auto
  # MTP works alongside both tiers: its draft bank layer stays full-width and resident, and the
  # verify graph publishes PLE cache slots through the same pre-replay hook as the decode graph.
  --speculative-algorithm "${FREETOKEN_SPECULATIVE:-mtp}"
  --speculative-num-draft-tokens "${FREETOKEN_SPEC_DRAFT_TOKENS:-2}"
  --speculative-adaptive-k-max "${FREETOKEN_SPEC_ADAPTIVE_K_MAX:-3}"
  --max-running-requests "${FREETOKEN_MAX_RUNNING_REQUESTS:-1}"
  --kv-reserve-tokens "${FREETOKEN_KV_RESERVE_TOKENS:-262144}"
  --kv-elastic-max-tokens "${FREETOKEN_KV_MAX_TOKENS:-262144}"
  # fp8 by default: at this profile's 16K reserve the KV pool is only 0.22 GiB out of ~14, so
  # FREETOKEN_KV_CACHE_DTYPE=nvfp4 buys 36 expert slots (2773 -> 2809) and costs ~5% TTFT --
  # not worth it. It pays only when the context is raised, because the saving scales with it:
  # at 128K, 2176 -> 2457 slots; at 256K (KV 3.50 -> 2.08 GiB), 1497 -> 2047, +37%. Decode at
  # 90K depth is then modestly better (warm mean 42.1 -> 48.6 tok/s) but inside this profile's
  # +/-15% NVMe-paging noise, so treat the slot count as the real gain, not the tok/s.
  # Prefill re-materialises every expert of every layer PER CHUNK, so the chunk size sets how
  # many times the ~54 non-resident experts per layer are re-read from NVMe -- and with it most
  # of TTFT. Raising it is a large, real TTFT win but it costs ACTIVATION VRAM, which is the
  # binding constraint on a 16 GB card. Measured at 250K context (idle ~14.7 GiB in every case;
  # the pools fit, the prefill activations are what overflow):
  #
  #   chunk    TTFT@250K   prefill tok/s   PEAK VRAM
  #    8192      114.8 s        2,177       16.2 GiB
  #   16384       93.2 s        2,683       18.1 GiB
  #   32768       82.6 s        3,026       22.0 GiB
  #
  # 8192 by default because it is the only one close to a 16 GB budget. NOTE that even it peaks
  # at 16.2 GiB at 256K, i.e. this profile's 256K claim does not actually fit a 16 GB card --
  # see PLAN-long-context.md 2e. --gpu-reserve-mb does NOT catch this: it constrains the pool
  # solver, not transient prefill activations, so a run on a 32 GiB card proves nothing here.
  # Raise it deliberately when you have the headroom: at 90K it bought TTFT 42.1 -> 29.0 s.
  --max-prefill-length "${FREETOKEN_MAX_EXTEND_TOKENS:-8192}"
  --kv-cache-dtype "${FREETOKEN_KV_CACHE_DTYPE:-nvfp4}"
  --dense-quant "${FREETOKEN_DENSE_QUANT:-fp8}"
  # Emulating a smaller card on a 32 GiB one: the reserve is subtracted AFTER --memory-ratio,
  # so reserve = ratio * (baseline_free - target). 14500 MiB ~ a 14.4 GiB card here. Drop it
  # (0) on real small hardware.
  --gpu-reserve-mb "${FREETOKEN_GPU_RESERVE_MB:-0}"
  # Prefill re-materialises every expert of every layer per chunk; --moe-prefill-hit-d2d
  # copies the GPU-resident ones device-to-device instead of re-fetching them over PCIe.
  # Measured NEUTRAL on this profile (prefill 41.9 s vs 42.0 s at 90K) and left off: with
  # ~2,000 GPU slots against 25,088 experts almost nothing is resident to copy. Kept as an
  # env switch because it is a clear win on the roomy config, where serve-qwen38.sh sets it.
  ${FREETOKEN_MOE_PREFILL_HIT_D2D:+--moe-prefill-hit-d2d}
  # NOT --moe-prefetch: the lookahead admits host bank rows by expert id, which tier 3's
  # slot-paged pool cannot serve (it crashed with an illegal memory access before the engine
  # learned to refuse it). The decode miss rate here is ~0.45, so there would be something to
  # win -- see FINDINGS.md.
  # Residency-aware routing (low-RAM only; refused without the NVMe tier). Substitutes a
  # routed expert that is not in the host pool for a near-tied one that is, when the swap
  # displaces <= TAU of the token's routed weight. CHANGES MODEL OUTPUTS -- 0 by default.
  --moe-residency-routing "${FREETOKEN_MOE_RESIDENCY_ROUTING:-0.05}"
  --expert-load auto
  --default-reasoning-effort none
  --enable-cache-report
)
exec "$FT_BIN" serve "${args[@]}"
