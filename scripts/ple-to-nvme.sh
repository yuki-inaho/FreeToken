#!/usr/bin/env bash
#
# Move Qwen3.8-Flash-Next's PLE n-gram table (the 10 model-plefp8-*.safetensors files, 47.7 GiB)
# onto an NVMe drive, leaving symlinks in the checkpoint so every launcher still finds it.
#
# With FREETOKEN_TIER3_PLE_GB > 0 (serve-interactive.sh profile 5, or any tier-3 profile) the
# engine does not pin the table: it keeps a small RAM row cache and reads 160 B rows from these
# files with O_DIRECT on a miss (~0.8 ms per decode step, see python/freetoken/tier3/FINDINGS.md
# "B2"). Those are latency-bound QD1 reads, so the files belong on the fastest NVMe you have.
#
# usage: scripts/ple-to-nvme.sh DIR          move the PLE files into DIR and link them back
#        scripts/ple-to-nvme.sh --check      only report where the PLE files live and whether
#                                            they are on NVMe / support O_DIRECT
#        scripts/ple-to-nvme.sh --restore    move them back into the checkpoint
#
# Environment:
#   FREETOKEN_MODEL_PATH  checkpoint (default $FREETOKEN_HOME/models/Qwen3.8-Flash-Next-NVFP4)
#
# Safe to re-run: files already in DIR are skipped; each file is copied, size-checked, and only
# then replaced by a link.

set -euo pipefail

readonly MODEL_PATH="${FREETOKEN_MODEL_PATH:-${FREETOKEN_HOME:-$HOME/.freetoken}/models/Qwen3.8-Flash-Next-NVFP4}"
readonly PLE_GLOB='model-plefp8-*.safetensors'

usage() { sed -n '3,/^$/{s/^# \{0,1\}//;p}' "${BASH_SOURCE[0]}"; }
die() { printf 'ple-to-nvme: %s\n' "$*" >&2; exit 1; }

[[ $# -eq 1 ]] || { usage >&2; exit 2; }
case "$1" in -h | --help) usage; exit 0 ;; esac

ckpt="$(readlink -f "$MODEL_PATH")"
[[ -f "$ckpt/model.safetensors.index.json" ]] || die "no checkpoint at $MODEL_PATH"
mapfile -t ple_files < <(cd "$ckpt" && compgen -G "$PLE_GLOB" | sort)
(( ${#ple_files[@]} > 0 )) || die "no $PLE_GLOB in $ckpt (not a Qwen3.8-Flash-Next checkpoint?)"

# "nvme", "ssd" or "hdd" for the block device holding PATH.
drive_kind() {
  local src dev parent
  src="$(findmnt -no SOURCE -T "$1" 2>/dev/null || true)"
  dev="$(basename "${src%%[*}")"
  [[ -b "/dev/$dev" ]] || { printf 'unknown'; return; }
  parent="$(lsblk -ndo PKNAME "/dev/$dev" 2>/dev/null || true)"
  parent="${parent:-$dev}"
  if [[ "$parent" == nvme* ]]; then
    printf 'nvme'
  elif [[ "$(lsblk -ndo ROTA "/dev/$parent" 2>/dev/null | tr -d ' ')" == 0 ]]; then
    printf 'ssd'
  else
    printf 'hdd'
  fi
}

# The row reader opens every file with O_DIRECT and has no buffered fallback.
supports_odirect() {
  python3 - "$1" <<'PY' 2>/dev/null
import mmap, os, sys
fd = os.open(sys.argv[1], os.O_RDONLY | os.O_DIRECT)
buf = mmap.mmap(-1, 4096)
os.preadv(fd, [buf], 0)
PY
}

report() {
  local f real kind
  for f in "${ple_files[@]}"; do
    real="$(readlink -f "$ckpt/$f")"
    kind="$(drive_kind "$real")"
    if supports_odirect "$real"; then
      printf '  %-34s %-5s %s\n' "$f" "$kind" "$real"
    else
      printf '  %-34s %-5s %s  (O_DIRECT FAILS)\n' "$f" "$kind" "$real"
    fi
  done
}

case "$1" in
  --check)
    printf 'PLE files of %s:\n' "$ckpt"
    report
    exit 0
    ;;
  --restore)
    for f in "${ple_files[@]}"; do
      link="$ckpt/$f"
      [[ -L "$link" ]] || continue
      target="$(readlink -f "$link")"
      cp --sparse=never "$target" "$link.tmp"
      [[ "$(stat -c %s "$link.tmp")" == "$(stat -c %s "$target")" ]] || die "copy of $f is truncated"
      mv -f "$link.tmp" "$link"
      rm -f "$target"
      printf '  restored %s\n' "$f"
    done
    exit 0
    ;;
  -*) die "unknown option $1 (see --help)" ;;
esac

dir="$1"
mkdir -p "$dir"
dir="$(readlink -f "$dir")"
[[ "$dir" != "$ckpt" ]] || die "DIR is the checkpoint itself"

kind="$(drive_kind "$dir")"
case "$kind" in
  nvme) ;;
  ssd) printf 'note: %s is on a SATA SSD, not NVMe; misses will be slower\n' "$dir" >&2 ;;
  *) die "$dir is not on an SSD (detected: $kind); PLE misses are latency-bound, use an NVMe drive" ;;
esac

need=0
for f in "${ple_files[@]}"; do
  [[ -L "$ckpt/$f" ]] || need=$(( need + $(stat -c %s "$ckpt/$f") ))
done
avail=$(( $(df --output=avail -B1 "$dir" | tail -n 1) ))
(( avail > need + (1 << 30) )) || die "$dir has $(( avail >> 30 )) GiB free; the PLE files need $(( need >> 30 )) GiB"

head -c 8192 /dev/zero >"$dir/.odirect-probe"
supports_odirect "$dir/.odirect-probe" || { rm -f "$dir/.odirect-probe"; die "$dir does not support O_DIRECT reads (use ext4 or xfs)"; }
rm -f "$dir/.odirect-probe"

for f in "${ple_files[@]}"; do
  src="$ckpt/$f"
  if [[ -L "$src" && "$(readlink -f "$src")" == "$dir/$f" ]]; then
    printf '  %s already in %s\n' "$f" "$dir"
    continue
  fi
  src_real="$(readlink -f "$src")"
  cp --sparse=never "$src_real" "$dir/$f.tmp"
  [[ "$(stat -c %s "$dir/$f.tmp")" == "$(stat -c %s "$src_real")" ]] || die "copy of $f is truncated"
  mv -f "$dir/$f.tmp" "$dir/$f"
  ln -sfn "$dir/$f" "$src"
  # A previous move to another DIR leaves its copy behind; only the checkpoint's own file is ours
  # to remove.
  [[ "$src_real" == "$ckpt/$f" ]] || printf '  note: old copy left at %s\n' "$src_real"
  printf '  moved %s -> %s\n' "$f" "$dir"
done

printf '\nPLE files of %s:\n' "$ckpt"
report
printf '\nServe with the PLE on disk: scripts/serve-interactive.sh --profile 5\n'
