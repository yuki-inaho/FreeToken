#!/usr/bin/env bash
#
# Deploy FreeToken + Qwen3.8-Flash-Next-NVFP4 on a fresh RTX 5090 (32 GB) / 128 GB RAM Linux box.
#
# This is the "Resident" profile from scripts/serve-interactive.sh (profile 1): every expert in
# memory, 256K context, MTP speculation. It peaks at 119 GiB of host RAM, so a 128 GB box has
# ~3 GiB to spare -- close other large processes before serving, or use --ple-nvme (73 GiB peak,
# ~7% slower decode).
#
# What it does, in order (each step is skipped when already done, so it is safe to re-run):
#   1. Checks the hardware (RTX 5090-class GPU >= 30 GiB VRAM, >= 120 GiB RAM, driver >= r580)
#      and the build tools (g++, curl, git).
#   2. Installs uv if it is missing.
#   3. Downloads RadixArk/Qwen3.8-Flash-Next-NVFP4 (~126 GB, resumable) into
#      $FREETOKEN_HOME/models, where the launchers look for it.
#   4. Runs scripts/setup-local.sh --yes: builds FreeToken from this tree into
#      $FREETOKEN_HOME/venv with the pinned CUDA 13 toolchain, checks nvcc, runs the doctor.
#   5. With --ple-nvme, serves the PLE table from disk instead of pinning it (profile 5).
#   6. Optionally installs a systemd user service that serves on 127.0.0.1:1919.
#
# usage: scripts/deploy-rtx5090-128g.sh [--install-driver] [--service] [--no-webui]
#                                       [--skip-download] [--verify] [--serve]
#                                       [--ple-nvme[=DIR]]
#
#   --install-driver  Install the NVIDIA r610 open driver via apt (sudo) if it is missing or
#                     older than r580, then stop so you can reboot and re-run.
#   --service         Install and enable ~/.config/systemd/user/freetoken.service.
#   --no-webui        Skip Open WebUI.
#   --skip-download   Do not download the checkpoint (it must already be in place).
#   --verify          SHA-256 the checkpoint against its download hashes (~10 minutes).
#   --serve           Start serving in the foreground when setup succeeds.
#   --ple-nvme[=DIR]  Keep the 47.7 GiB PLE n-gram table on NVMe behind a 2 GiB RAM row cache
#                     (serve-interactive.sh profile 5) instead of pinning it: 73 GiB peak host RAM
#                     instead of 119, for ~7% lower decode tok/s (measured). With DIR, the
#                     PLE files are moved there (scripts/ple-to-nvme.sh) -- use it when the
#                     checkpoint sits on a slower drive than DIR. DIR must be ext4/xfs on NVMe.
#
# Environment:
#   FREETOKEN_HOME        install root (default ~/.freetoken)
#   FREETOKEN_MODEL_DIR   where the checkpoint is stored (default $FREETOKEN_HOME/models); when set
#                         to another disk, $FREETOKEN_HOME/models/<name> is linked to it
#   HF_TOKEN              Hugging Face token, if the download needs one
#   FREETOKEN_SKIP_HW_CHECK=1  continue on a GPU / RAM that does not match this profile

set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
readonly FT_HOME="${FREETOKEN_HOME:-$HOME/.freetoken}"
readonly LINK_DIR="$FT_HOME/models"
readonly STORE_DIR="${FREETOKEN_MODEL_DIR:-$LINK_DIR}"
readonly MODEL_REPO=RadixArk/Qwen3.8-Flash-Next-NVFP4
readonly MODEL_NAME=Qwen3.8-Flash-Next-NVFP4
readonly MODEL_GB=126
readonly MIN_VRAM_MIB=30000
readonly PLE_CACHE_GB=2

install_driver=0
service=0
webui=1
download=1
verify=0
serve=0
ple_nvme=0
ple_dir=''
for arg in "$@"; do
  case "$arg" in
    --install-driver) install_driver=1 ;;
    --service) service=1 ;;
    --no-webui) webui=0 ;;
    --skip-download) download=0 ;;
    --verify) verify=1 ;;
    --serve) serve=1 ;;
    --ple-nvme) ple_nvme=1 ;;
    --ple-nvme=*) ple_nvme=1; ple_dir="${arg#--ple-nvme=}" ;;
    -h | --help)
      sed -n '3,/^$/{s/^# \{0,1\}//;p}' "${BASH_SOURCE[0]}"
      exit 0
      ;;
    *)
      printf 'Unknown option: %s (see --help)\n' "$arg" >&2
      exit 2
      ;;
  esac
done

if (( ple_nvme )); then
  readonly PROFILE=5 MIN_RAM_GIB=80 SERVE_RAM_GIB=73
else
  readonly PROFILE=1 MIN_RAM_GIB=120 SERVE_RAM_GIB=119
fi

if [[ -t 1 ]]; then
  C_CYAN=$'\033[1;36m'; C_GREEN=$'\033[1;32m'; C_YELLOW=$'\033[1;33m'; C_RED=$'\033[1;31m'; C_RESET=$'\033[0m'
else
  C_CYAN=''; C_GREEN=''; C_YELLOW=''; C_RED=''; C_RESET=''
fi
step() { printf '\n%s==>%s %s\n' "$C_CYAN" "$C_RESET" "$*"; }
ok() { printf '  %sok%s    %s\n' "$C_GREEN" "$C_RESET" "$*"; }
note() { printf '  %s-%s     %s\n' "$C_YELLOW" "$C_RESET" "$*"; }
die() {
  printf '  %sFAIL%s  %s\n' "$C_RED" "$C_RESET" "$*" >&2
  exit 1
}
hw_mismatch() {
  if [[ "${FREETOKEN_SKIP_HW_CHECK:-0}" == 1 ]]; then
    note "$* (continuing: FREETOKEN_SKIP_HW_CHECK=1)"
  else
    die "$* (set FREETOKEN_SKIP_HW_CHECK=1 to continue anyway)"
  fi
}

check_hardware() {
  step "Hardware"
  [[ "$(uname -s)" == Linux ]] || die "Linux is required"
  [[ "$(uname -m)" == x86_64 ]] || hw_mismatch "x86_64 is required (found $(uname -m))"

  if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
    need_driver "no working NVIDIA driver (nvidia-smi is unavailable)"
    return
  fi
  local name vram driver
  IFS=, read -r name vram driver < <(nvidia-smi --query-gpu=name,memory.total,driver_version \
    --format=csv,noheader,nounits | head -n 1)
  name="${name# }"; vram="${vram// /}"; driver="${driver// /}"
  if (( vram >= MIN_VRAM_MIB )); then
    ok "GPU $name, $vram MiB"
  else
    hw_mismatch "GPU $name has $vram MiB; this profile needs an RTX 5090-class card (>= $MIN_VRAM_MIB MiB)"
  fi
  if [[ "${driver%%.*}" =~ ^[0-9]+$ ]] && (( ${driver%%.*} >= 580 )); then
    ok "driver $driver"
  else
    need_driver "driver $driver is older than r580"
  fi

  local ram_gib avail_gib
  ram_gib="$(awk '/^MemTotal:/{printf "%d", $2/1048576}' /proc/meminfo)"
  avail_gib="$(awk '/^MemAvailable:/{printf "%d", $2/1048576}' /proc/meminfo)"
  if (( ram_gib >= MIN_RAM_GIB )); then
    ok "RAM ${ram_gib} GiB (${avail_gib} GiB available)"
  else
    hw_mismatch "RAM ${ram_gib} GiB; profile $PROFILE needs >= $MIN_RAM_GIB GiB (--ple-nvme needs less; see scripts/serve-interactive.sh for NVMe-tier profiles)"
  fi
  if (( avail_gib < SERVE_RAM_GIB )); then
    note "only ${avail_gib} GiB is available now; serving needs ~${SERVE_RAM_GIB} GiB, so stop other large processes first"
  fi
  if (( ! ple_nvme && ram_gib < 150 )); then
    note "profile 1 peaks at ${SERVE_RAM_GIB} GiB of ${ram_gib}; --ple-nvme serves the PLE table from NVMe (73 GiB peak, ~7% slower decode)"
  fi
  if grep -qi microsoft /proc/version 2>/dev/null; then
    note "WSL detected: it sees only part of the host RAM and caps pinned memory; native Linux is recommended"
  fi
}

need_driver() {
  if (( install_driver )); then
    note "$1; installing the driver"
    "$SCRIPT_DIR/install-nvidia-driver.sh"
    printf '\nReboot (sudo reboot), then re-run: %s\n' "$0"
    exit 0
  fi
  die "$1: re-run with --install-driver, or run scripts/install-nvidia-driver.sh and reboot"
}

check_build_tools() {
  # setup.py compiles two C++ extensions; uv supplies Python itself.
  step "Build tools"
  local tool missing=()
  for tool in g++ curl git; do
    command -v "$tool" >/dev/null 2>&1 || missing+=("$tool")
  done
  (( ${#missing[@]} == 0 )) || die "missing ${missing[*]}: sudo apt-get install -y build-essential curl git"
  ok "g++ $(g++ -dumpfullversion), curl, git"
}

ensure_uv() {
  step "uv"
  if ! command -v uv >/dev/null 2>&1 && [[ -x "$HOME/.local/bin/uv" ]]; then
    export PATH="$HOME/.local/bin:$PATH"
  fi
  if command -v uv >/dev/null 2>&1; then
    ok "$(uv --version)"
    return
  fi
  command -v curl >/dev/null 2>&1 || die "curl is required to install uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  command -v uv >/dev/null 2>&1 || die "uv installed but is not on PATH"
  ok "$(uv --version) (installed)"
}

download_model() {
  step "Checkpoint $MODEL_REPO"
  local dest="$STORE_DIR/$MODEL_NAME" link="$LINK_DIR/$MODEL_NAME"
  mkdir -p "$STORE_DIR" "$LINK_DIR"
  if [[ "$dest" != "$link" ]]; then
    if [[ -e "$link" && ! -L "$link" ]]; then
      die "$link exists and is not a link; move it or unset FREETOKEN_MODEL_DIR"
    fi
    ln -sfn "$dest" "$link"
  fi

  if (( ! download )); then
    [[ -f "$link/model.safetensors.index.json" ]] || die "no checkpoint at $link (drop --skip-download)"
    ok "using existing checkpoint at $(readlink -f "$link")"
    return
  fi

  # Room for whatever is not downloaded yet, plus a margin.
  local have_gb free_gb
  have_gb=0
  [[ -d "$dest" ]] && have_gb="$(du -sL --block-size=1G "$dest" | cut -f1)"
  free_gb="$(df --output=avail --block-size=1G "$STORE_DIR" | tail -n 1 | tr -d ' ')"
  if (( free_gb + have_gb < MODEL_GB + 5 )); then
    die "$STORE_DIR has ${free_gb} GB free; the checkpoint needs ~${MODEL_GB} GB (set FREETOKEN_MODEL_DIR to another disk)"
  fi

  # hf download resumes, and skips files that already match, so re-runs are cheap.
  note "downloading to $dest (~${MODEL_GB} GB; resumable)"
  uvx --from 'huggingface_hub[hf_xet]' hf download "$MODEL_REPO" --local-dir "$dest" \
    || die "download failed; re-run to resume (set HF_TOKEN if the repo needs authentication)"
  [[ -f "$dest/model.safetensors.index.json" ]] || die "download finished without model.safetensors.index.json"
  ok "$MODEL_NAME at $dest"
}

run_setup() {
  local -a args=(--yes)
  (( webui )) || args+=(--no-webui)
  (( verify )) && args+=(--verify)
  # Search only our own model directory, so setup-local.sh never links a stray copy from /media.
  FREETOKEN_HOME="$FT_HOME" FREETOKEN_MODEL_SEARCH="$LINK_DIR" "$SCRIPT_DIR/setup-local.sh" "${args[@]}"
}

setup_ple_nvme() {
  step "PLE table on NVMe"
  if [[ -n "$ple_dir" ]]; then
    FREETOKEN_MODEL_PATH="$LINK_DIR/$MODEL_NAME" "$SCRIPT_DIR/ple-to-nvme.sh" "$ple_dir" \
      || die "could not move the PLE files to $ple_dir"
  else
    FREETOKEN_MODEL_PATH="$LINK_DIR/$MODEL_NAME" "$SCRIPT_DIR/ple-to-nvme.sh" --check
    if FREETOKEN_MODEL_PATH="$LINK_DIR/$MODEL_NAME" "$SCRIPT_DIR/ple-to-nvme.sh" --check | grep -q 'O_DIRECT FAILS'; then
      die "the PLE files do not support O_DIRECT where they are; pass --ple-nvme=DIR on an ext4/xfs NVMe drive"
    fi
    if FREETOKEN_MODEL_PATH="$LINK_DIR/$MODEL_NAME" "$SCRIPT_DIR/ple-to-nvme.sh" --check | grep -qE ' (hdd|unknown) '; then
      note "the PLE files are not on an SSD; pass --ple-nvme=DIR to move them to NVMe"
    fi
  fi
  ok "PLE served from disk with a ${PLE_CACHE_GB} GiB row cache (FREETOKEN_TIER3_PLE_GB=$PLE_CACHE_GB)"
}

install_service() {
  step "systemd user service"
  command -v systemctl >/dev/null 2>&1 || die "systemctl is unavailable"
  local unit_dir="$HOME/.config/systemd/user"
  mkdir -p "$unit_dir"
  cat >"$unit_dir/freetoken.service" <<EOF
[Unit]
Description=FreeToken ($MODEL_NAME, RTX 5090 resident profile)
After=network-online.target

[Service]
Type=simple
Environment=FREETOKEN_HOME=$FT_HOME
$( (( ple_nvme )) && printf 'Environment=FREETOKEN_TIER3_PLE_GB=%s' "$PLE_CACHE_GB" )
ExecStart=$SCRIPT_DIR/serve-qwen38.sh
Restart=on-failure
RestartSec=10

[Install]
WantedBy=default.target
EOF
  systemctl --user daemon-reload
  systemctl --user enable freetoken.service >/dev/null
  ok "enabled freetoken.service (start: systemctl --user start freetoken)"
  if command -v loginctl >/dev/null 2>&1 && [[ "$(loginctl show-user "$USER" -p Linger --value 2>/dev/null)" != yes ]]; then
    note "to keep it running after logout: sudo loginctl enable-linger $USER"
  fi
}

check_hardware
check_build_tools
ensure_uv
download_model
run_setup
(( ple_nvme )) && setup_ple_nvme
(( service )) && install_service

cat <<EOF

${C_GREEN}Deployed.${C_RESET} Serve profile $PROFILE (OpenAI-compatible API on http://127.0.0.1:1919/v1):
  $SCRIPT_DIR/serve-interactive.sh --profile $PROFILE
With Open WebUI as well:
  $( (( ple_nvme )) && printf 'FREETOKEN_TIER3_PLE_GB=%s ' "$PLE_CACHE_GB" )$SCRIPT_DIR/serve-local-ai.sh
EOF

if (( serve )); then
  exec "$SCRIPT_DIR/serve-interactive.sh" --profile "$PROFILE"
fi
