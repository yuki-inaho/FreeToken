#!/usr/bin/env bash
# Pick a Qwen3.8-Flash-Next serving profile interactively, from what this machine actually has.
#
# The profiles below are measured, not guessed -- see python/freetoken/tier3/FINDINGS.md.
# Anything that trades quality for speed says so before it runs.
#
#   scripts/serve-interactive.sh              # ask
#   scripts/serve-interactive.sh --profile 2  # skip the questions (scriptable)
#   scripts/serve-interactive.sh --dry-run    # print the command instead of exec'ing it
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE=""; DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --profile) PROFILE="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) sed -n '2,10p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

# ---- what is actually here ---------------------------------------------------------------
vram_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 0)
gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "no GPU")
ram_gib=$(awk '/^MemTotal:/{printf "%.0f", $2/1048576}' /proc/meminfo)
ftw="${FREETOKEN_TIER3_FTW:-/mnt/ftw/Qwen3.8-Flash-Next-FTW}"
have_ftw=0; [ -f "$ftw/freetoken_weight.json" ] && have_ftw=1

echo "Detected: ${gpu_name} (${vram_mib} MiB VRAM), ${ram_gib} GiB RAM"
if [ "$have_ftw" = 1 ]; then echo "          FTW expert corpus at $ftw"
else echo "          no FTW corpus at $ftw -- the NVMe-tier profiles need one (ft checkpoint)"; fi
echo

# ---- profiles ------------------------------------------------------------------------------
# name | needs_ftw | one-line summary
p1_desc="Resident       32 GB VRAM, 111 GiB RAM. Everything in memory, 256K ctx, ~94 tok/s."
p2_desc="16 GB / 48 GB  NVMe tier, 256K ctx, residency routing ON (DEFAULT). ~73 tok/s varied."
p3_desc="16 GB / 64 GB  NVMe tier, 256K ctx, routing OFF. More RAM, exact routing, ~91 tok/s."
p4_desc="16 GB / 48 GB  as (2) but routing OFF -- the honest-numerics low-RAM option, ~58 tok/s."
p5_desc="PLE on NVMe    32 GB VRAM, 73 GiB RAM peak (vs 119 for 1). As (1), but the 47.7 GiB PLE
                 table is read from NVMe through a 2 GiB row cache: decode ~7% slower than (1)."

if [ -z "$PROFILE" ]; then
  echo "Profiles:"
  echo "  1) $p1_desc"
  echo "  2) $p2_desc"
  echo "  3) $p3_desc"
  echo "  4) $p4_desc"
  echo "  5) $p5_desc"
  echo
  suggest=3
  # (1) peaks at 119 GiB of host RAM (measured under a cgroup cap), which a 128 GB box can hold
  # only with nothing else running; below ~150 GiB the PLE-on-NVMe profile is the safer pick.
  [ "$vram_mib" -ge 30000 ] && [ "$ram_gib" -ge 80 ] && suggest=5
  [ "$vram_mib" -ge 30000 ] && [ "$ram_gib" -ge 150 ] && suggest=1
  [ "$vram_mib" -lt 30000 ] && [ "$ram_gib" -lt 56 ] && suggest=2
  read -r -p "Profile [1-5] (suggested: $suggest): " PROFILE
  PROFILE="${PROFILE:-$suggest}"
fi

case "$PROFILE" in
  1) script="serve-qwen38.sh"; needs_ftw=0 ;;
  2) script="serve-qwen38-tier3.sh"; needs_ftw=1 ;;
  3) script="serve-qwen38-tier3.sh"; needs_ftw=1
     export FREETOKEN_TIER3_HOST_SLOTS=458 FREETOKEN_MOE_RESIDENCY_ROUTING=0 ;;
  4) script="serve-qwen38-tier3.sh"; needs_ftw=1
     export FREETOKEN_MOE_RESIDENCY_ROUTING=0 ;;
  # Experts stay resident; only the PLE table moves to disk. The rows are read from the
  # checkpoint's model-plefp8-*.safetensors (scripts/ple-to-nvme.sh relocates them to NVMe).
  5) script="serve-qwen38.sh"; needs_ftw=0
     export FREETOKEN_TIER3_PLE_GB="${FREETOKEN_TIER3_PLE_GB:-2}" ;;
  *) echo "no such profile: $PROFILE" >&2; exit 2 ;;
esac

if [ "$needs_ftw" = 1 ] && [ "$have_ftw" != 1 ]; then
  echo "Profile $PROFILE streams experts from NVMe but there is no FTW corpus at $ftw." >&2
  echo "Build one with:  ft checkpoint --model <hf-dir> --out $ftw --moe-backend offload" >&2
  exit 1
fi

# ---- consent for the one thing that changes model outputs ---------------------------------
if [ "${FREETOKEN_MOE_RESIDENCY_ROUTING:-0.05}" != "0" ] && [ "$needs_ftw" = 1 ]; then
  cat <<'WARN'

  Residency-aware routing is ON. When a routed expert is not in host RAM it is replaced by
  the best near-tied expert that is, if the swap displaces <= 0.05 of the token's routed
  weight. THIS CHANGES MODEL OUTPUTS.

  Measured: +22 to +39 % decode at 48 GiB, GSM8K-150 (8-shot CoT, greedy, lm_eval) 92.0 %
  strict / 93.3 % flexible vs the 88.67 % record. Gated on GSM8K only -- MMLU was not run.

WARN
  if [ -t 0 ]; then
    read -r -p "  Continue with routing enabled? [Y/n] " ans
    case "${ans:-y}" in [Nn]*) export FREETOKEN_MOE_RESIDENCY_ROUTING=0
        echo "  -> disabled; serving with exact routing." ;; esac
  fi
fi

# ---- constrain a bigger card to the profile it claims ---------------------------------------
# The tier-3 profiles are named for a 16 GB card, and --moe-cache-auto will size the expert
# cache to whatever the card actually has: on a 32 GiB board an unconstrained "16 GB profile"
# quietly uses 30 GB, which is not the thing being offered. Two knobs, both needed:
#
#   --gpu-reserve-mb        steers the POOL SOLVER. The reserve is subtracted after
#                           --memory-ratio (0.9), so emulating a card of TARGET on a card of
#                           TOTAL wants 0.9 * (TOTAL - TARGET) -- independent of what else is
#                           on the GPU right now, which is why it is computed from totals.
#   FREETOKEN_GPU_MEM_CAP_MB  hard-caps the ALLOCATOR, so anything the solver did not account
#                           for (transient prefill activations) also fails inside the budget
#                           rather than spilling into the extra VRAM.
#
# On a real 16 GB card the reserve computes to 0 and the cap is a no-op ceiling: no emulation.
TARGET_VRAM_MB="${FREETOKEN_TARGET_VRAM_MB:-16384}"
if [ "$needs_ftw" = 1 ] && [ "$vram_mib" -gt 0 ]; then
  if [ -z "${FREETOKEN_GPU_RESERVE_MB:-}" ]; then
    reserve=$(( (vram_mib - TARGET_VRAM_MB) * 9 / 10 ))
    [ "$reserve" -lt 0 ] && reserve=0
    export FREETOKEN_GPU_RESERVE_MB="$reserve"
  fi
  # The cap bounds the PyTorch allocator; the CUDA context (~100-400 MiB) lives outside it and
  # comes out of the same card. Capping at TARGET therefore emulates a card slightly larger
  # than TARGET -- measured 16,486 MiB of total process VRAM against a 16,384 cap. Leave the
  # context room so "fits 16 GB" means the whole process fits. Measured on this box: ~630 MiB
  # sits outside the allocator (context + cuBLAS workspaces + the CPU-MoE UVA mappings), so
  # the allowance is 832 and the whole process lands near 16.2 GiB of a 16,384 MiB card.
  # Roughly 1 GiB is allocated OUTSIDE the PyTorch allocator (CUDA context, cuBLAS workspaces,
  # the CPU-MoE extension's own cudaMallocs). Measured: lowering the cap from 16,384 to 15,552
  # left total process VRAM unchanged at ~16.5 GiB, so the cap alone cannot hold the process
  # inside a card -- only the reserve can, because it shrinks the pools. Charge the off-budget
  # GiB to the reserve so the TOTAL lands inside TARGET, and keep the cap as a backstop on the
  # part it does govern.
  NONALLOC_MB="${FREETOKEN_NONALLOC_MB:-1024}"
  if [ "${FREETOKEN_GPU_RESERVE_MB:-0}" != "0" ]; then
    export FREETOKEN_GPU_RESERVE_MB="$(( FREETOKEN_GPU_RESERVE_MB + NONALLOC_MB ))"
  fi
  if [ -z "${FREETOKEN_GPU_MEM_CAP_MB:-}" ] && [ "$vram_mib" -gt "$TARGET_VRAM_MB" ]; then
    export FREETOKEN_GPU_MEM_CAP_MB="$(( TARGET_VRAM_MB - NONALLOC_MB ))"
  fi
  if [ "${FREETOKEN_GPU_RESERVE_MB:-0}" != "0" ]; then
    echo "  This card has ${vram_mib} MiB; holding the profile to ${TARGET_VRAM_MB} MiB"
    echo "  (--gpu-reserve-mb ${FREETOKEN_GPU_RESERVE_MB}, allocator capped at ${FREETOKEN_GPU_MEM_CAP_MB})."
    echo "  FREETOKEN_TARGET_VRAM_MB changes the target; set it to ${vram_mib} to use the whole card."
  fi
fi

echo
echo "Launching profile $PROFILE: $script"
env | grep -E "^FREETOKEN_(TIER3_HOST_SLOTS|TIER3_PLE_GB|MOE_RESIDENCY_ROUTING|KV_CACHE_DTYPE|GPU_RESERVE_MB|GPU_MEM_CAP_MB)=" | sort | sed 's/^/  /' || true
if [ "$DRY" = 1 ]; then echo "  (dry run) exec $REPO/scripts/$script"; exit 0; fi
exec "$REPO/scripts/$script"
