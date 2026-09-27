#!/usr/bin/env bash
# Serve Qwen3.8-Flash-Next on a 16 GB VRAM / 48 GB RAM budget at 256K context: MoE experts AND
# the PLE n-gram table stream from NVMe (tier 3), and routing prefers experts that are in RAM.
#
# 48 GiB rather than the full 64 so a 64 GB box has headroom for everything else. That costs
# host residency (352 of 512 experts per layer instead of 458), which residency-aware routing
# then buys back: measured at 90K, +22 to +39 % decode (bench 58.1 -> 72.8, english 59.7 ->
# 72.8, code 56.5 -> 75.0, japanese 49.6 -> 69.0 tok/s) with GSM8K-150 (8-shot CoT, greedy,
# lm_eval) at 92.0 % strict / 93.3 % flexible against the 88.67 % baseline record.
#
# The alternate 64 GB-of-RAM shape: FREETOKEN_TIER3_HOST_SLOTS=458 FREETOKEN_MOE_RESIDENCY_ROUTING=0
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
# NVFP4 banks, 48 layers) -- 352 slots ~ 48.8 GiB and 69% residency. The pool MUST exceed one
# step's distinct working set (max-running-requests * top_k) or experts get dropped; the
# launcher below fails at startup if it does not.
set -euo pipefail

readonly FT_VENV="${FREETOKEN_VENV:-${FREETOKEN_HOME:-$HOME/.freetoken}/venv}"
readonly FT_BIN="${FREETOKEN_FT_BIN:-$FT_VENV/bin/ft}"
readonly MODEL_PATH="${FREETOKEN_MODEL_PATH:-${FREETOKEN_HOME:-$HOME/.freetoken}/models/Qwen3.8-Flash-Next-NVFP4}"
# The corpus is served from ext4 (/mnt/ftw) when that copy exists: same drive, same ~5.7 GB/s at
# QD>=4, but +17 % at QD1 -- and a decode-time miss is a QD1 read. The NTFS copy (linked by
# scripts/setup-local.sh) is the fallback.
ftw_default="/mnt/ftw/Qwen3.8-Flash-Next-FTW"
[[ -f "$ftw_default/freetoken_weight.json" ]] || ftw_default="${FREETOKEN_HOME:-$HOME/.freetoken}/models/Qwen3.8-Flash-Next-FTW"
readonly FTW_PATH="${FREETOKEN_TIER3_FTW:-$ftw_default}"

[[ -d "$FTW_PATH" ]] || { echo "no FTW corpus at $FTW_PATH -- run ft checkpoint first" >&2; exit 1; }
[[ -f "$FTW_PATH/freetoken_weight.json" ]] || { echo "$FTW_PATH is not an FTW checkpoint" >&2; exit 1; }

# Emulating a smaller card honestly: --gpu-reserve-mb only steers the pool solver, so a config
# that overflows a 16 GB card still runs here. FREETOKEN_GPU_MEM_CAP_MB hard-caps the allocator,
# turning "would this fit?" into a question the 5090 can answer. Unset by default (a real 16 GB
# card needs no emulation); scripts/serve-interactive.sh computes both knobs from
# FREETOKEN_TARGET_VRAM_MB when it launches this profile on a bigger card.
export FREETOKEN_GPU_MEM_CAP_MB="${FREETOKEN_GPU_MEM_CAP_MB:-0}"

cuda_home_default="$FT_VENV/lib/python3.12/site-packages/nvidia/cu13"
export CUDA_HOME="${FREETOKEN_CUDA_HOME:-$cuda_home_default}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LIBRARY_PATH="$CUDA_HOME/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"

export FREETOKEN_TIER3_FTW="$FTW_PATH"
# Pinned host budget at the default (48 GiB) geometry:
#   expert pool     352 slots x 48 layers x 2.64 MiB = 43.6 GiB   (69% residency)
#   MTP draft layer 512 experts, full width, resident =  1.35 GiB
#   PLE row cache                                     =  2.0  GiB
#   prefill staging 2 x (512-352) experts             =  0.83 GiB
#   ---------------------------------------------------------------
#                                                       47.8 GiB  (~48.8 GiB measured pinned)
# The 64 GB shape (458 slots) pins ~61.8 GiB and is one env var away (see the header).
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
# It perturbs the emitted logits and MMLU-chat missed the 87.3 floor on the 32 GiB card
# (86.84), so it stays OPT-IN -- set FREETOKEN_LM_HEAD_NVFP4=1 to enable.
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
  # Elastic KV: boot with a 64K reservation and grow on demand (64K -> 128K -> 256K, through the
  # idle-time cache rebuild; the request is parked for the ~1-2 s it takes), shrink back after
  # the idle timeout. On this card the KV pool and the expert cache trade one-for-one, and the
  # expert-slot count is what decode speed is a function of: the 2.08 GiB a 256K NVFP4 pool costs
  # at boot is ~590 expert slots (~+29 % on ~2,050) whenever the context is short.
  # FREETOKEN_KV_RESERVE_TOKENS=262144 restores the fixed-256K boot.
  --kv-reserve-tokens "${FREETOKEN_KV_RESERVE_TOKENS:-65536}"
  --kv-elastic-max-tokens "${FREETOKEN_KV_MAX_TOKENS:-262144}"
  --kv-elastic-idle-seconds "${FREETOKEN_KV_ELASTIC_IDLE_S:-300}"
  # One GDN snapshot slot (prefix reuse for the latest conversation only): the auto sizing keeps
  # 9 slots of ~110 MiB, i.e. ~125 expert slots this profile cannot spare. -1 restores auto.
  --linear-state-cache "${FREETOKEN_LINEAR_STATE_CACHE:-1}"
  # nvfp4 KV by default: the saving scales with the pool, and the pool grows to 256K here.
  # At 128K, fp8 -> nvfp4 is 2176 -> 2457 expert slots; at 256K (KV 3.50 -> 2.08 GiB) it is
  # 1497 -> 2047, +37%. At the 64K base it is worth ~150 slots. Decode at 90K depth is then
  # modestly better (warm mean 42.1 -> 48.6 tok/s) but inside this profile's +/-15 %
  # NVMe-paging noise, so treat the slot count as the real gain, not the tok/s. TTFT costs ~5 %.
  # Prefill re-materialises every expert of every layer PER CHUNK, so the chunk size sets how
  # many times the ~160 non-resident experts per layer are re-read from NVMe -- and with it most
  # of TTFT. Raising it is a large, real TTFT win but it costs ACTIVATION VRAM, which is the
  # binding constraint on a 16 GB card. Measured at 250K context (idle ~14.7 GiB in every case;
  # the pools fit, the prefill activations are what overflow):
  #
  #   chunk    TTFT@250K   prefill tok/s   PEAK VRAM
  #    8192      114.8 s        2,177       16.2 GiB
  #   16384       93.2 s        2,683       18.1 GiB
  #   32768       82.6 s        3,026       22.0 GiB
  #
  # 8192 by default: under a real FREETOKEN_GPU_MEM_CAP_MB=16384 it completes a 250K prompt
  # (the 16.2 GiB figure is the caching allocator's high-water mark on a roomy card) while
  # 32768 OOMs on a 90K one. --gpu-reserve-mb does NOT catch
  # this: it constrains the pool solver, not transient prefill activations, so a run on a 32 GiB
  # card proves nothing here. Raise it deliberately when you have the headroom: at 90K it bought
  # TTFT 42.1 -> 29.0 s.
  --max-prefill-length "${FREETOKEN_MAX_EXTEND_TOKENS:-8192}"
  --kv-cache-dtype "${FREETOKEN_KV_CACHE_DTYPE:-nvfp4}"
  --dense-quant "${FREETOKEN_DENSE_QUANT:-fp8}"
  # 0 on real small hardware. When emulating this profile on a bigger card the reserve is what
  # actually shrinks the pools (the cap above only bounds the allocator); serve-interactive.sh
  # computes it as 0.9 * (TOTAL - TARGET) + ~1 GiB of off-allocator VRAM.
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
  # displaces <= TAU of the token's routed weight. CHANGES MODEL OUTPUTS: on by default at
  # 0.05 for this profile because GSM8K-150 / MMLU-chat / needle matched the tau=0 arm;
  # FREETOKEN_MOE_RESIDENCY_ROUTING=0 is the unmodified model.
  --moe-residency-routing "${FREETOKEN_MOE_RESIDENCY_ROUTING:-0.05}"
  # A read costs one NVMe fetch per DISTINCT expert per (layer, step), so a row that keeps one
  # of its two misses still pays for it -- which is why substituting 88 % of missing ROUTES only
  # halved the reads. Substituting every missing expert in the row instead removes most of the
  # rest. The per-token perturbation budget does not grow: each swap is still gated on tau and
  # their sum is bounded by --moe-residency-routing-total (= tau), so a row may now spend that
  # budget on several experts rather than one, and the SET of rows perturbed is unchanged.
  # Measured 2026-09-03, same session, emulated 16 GB card, bench-decode 400 tok x 5 runs:
  #
  #   routing off   bench 61.8  english 61.8  code 56.2  japanese 49.5   reads/step ~20
  #   max-subs 1    bench 76.8  english 74.6  code 78.2  japanese 66.3   reads/step ~9
  #   max-subs 4    bench 90.5  english 84.3  code 98.4  japanese 84.4   reads/step ~5
  #
  # GSM8K-150 (8-shot CoT, greedy) and MMLU-chat 570q are IDENTICAL on all three arms
  # (147/150 and 504/570); that is a quality gate, not a proof of equivalence. 1 restores the one-substitution behaviour; 0 substitutions is
  # FREETOKEN_MOE_RESIDENCY_ROUTING=0, which is the unmodified model.
  --moe-residency-routing-max-subs "${FREETOKEN_MOE_RESIDENCY_MAX_SUBS:-4}"
  # The MEASURED policy, which replaces the substitution above when it is switched on.
  # `ft moe subst` ran every expert against every other over the tokens that route to it, all 48
  # layers: no expert stands in for another (best cosine +0.02..+0.19), and substituting beats
  # simply DROPPING the route for only 4 of 23,504 experts. So with a table and a non-zero eps
  # the engine drops instead, gated on the fraction of the MoE block output the drop moves --
  # the quantity that bounds the damage, rather than the displaced routing weight tau uses.
  # A dropped route also costs nothing at all: no NVMe read, no PCIe gather, no GEMV.
  # Build the table once:
  #   ft moe subst --model "$MODEL_PATH" --dumps <route dumps> --out "$FTW_PATH/expert_subst.safetensors"
  # eps 0 keeps the substitution policy; raise it to switch over.
  --moe-residency-routing-table "${FREETOKEN_MOE_SUBST_TABLE:-auto}"
  --moe-residency-routing-eps "${FREETOKEN_MOE_RESIDENCY_EPS:-0}"
  --expert-load auto
  --default-reasoning-effort none
  --enable-cache-report
)
if [[ -n "${FREETOKEN_MAX_SEQ_LEN:-}" ]]; then
  args+=(--max-seq-len-override "$FREETOKEN_MAX_SEQ_LEN")
fi
exec "$FT_BIN" serve "${args[@]}" "$@"
