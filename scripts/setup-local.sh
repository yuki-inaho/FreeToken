#!/usr/bin/env bash
#
# One-shot local setup for this workstation.
#
# The three NTFS data drives all carry the label "New Volume", so udisks mounts them as
# "New Volume", "New Volume1" and "New Volume2" in whatever order they come up, and any
# hard-coded /media path eventually breaks. This script finds each checkpoint wherever it
# is mounted and links it at a stable path under $FREETOKEN_HOME/models, which the launchers
# use by default. It also repairs the workspace .venv link, installs whatever is missing
# (FreeToken, its CUDA 13 toolchain, Open WebUI) and finishes with doctor-qwen38.sh.
#
# Safe to re-run; re-run it whenever the drives come back under different mount names.
#
# usage: scripts/setup-local.sh [--verify] [--reinstall] [--no-webui] [--yes]
#
#   --verify     SHA-256 every checkpoint file against the hashes Hugging Face recorded at
#                download time (.cache/huggingface/download). Reads ~126 GB; ~10 minutes.
#   --reinstall  Rebuild FreeToken into the venv from this source tree even if it is already
#                installed. Replaces files that were copied into the venv by hand.
#   --no-webui   Skip the Open WebUI install check.
#   --yes, -y    Do not ask before --reinstall replaces a modified install.
#
# Environment:
#   FREETOKEN_HOME          install root (default ~/.freetoken); links go in $FREETOKEN_HOME/models
#   FREETOKEN_VENV          FreeToken venv (default $FREETOKEN_HOME/venv)
#   FREETOKEN_MODEL_SEARCH  colon-separated directories to search for checkpoints
#                           (default /mnt/ftw, /media/$USER/*, /mnt/*)

set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
readonly FT_HOME="${FREETOKEN_HOME:-$HOME/.freetoken}"
readonly FT_VENV="${FREETOKEN_VENV:-$FT_HOME/venv}"
readonly MODELS_DIR="$FT_HOME/models"
readonly PY_VERSION=3.12
readonly CUDA_HOME_DIR="${FREETOKEN_CUDA_HOME:-$FT_VENV/lib/python$PY_VERSION/site-packages/nvidia/cu13}"
# nvcc must share torch's CUDA major (13); 13.3.73 is the release the current venv was
# validated with. nvcc, its headers (crt) and its front end (nvvm) ship as separate wheels
# and must be the same release: left to float, nvvm resolves newer and emits PTX that
# nvcc's own ptxas rejects ("Unsupported .version 9.4").
readonly CUDA_TOOLKIT_VERSION="${FREETOKEN_CUDA_TOOLKIT_VERSION:-13.3.73}"
readonly TOOLKIT_SPECS=(
  "nvidia-cuda-nvcc==$CUDA_TOOLKIT_VERSION"
  "nvidia-cuda-crt==$CUDA_TOOLKIT_VERSION"
  "nvidia-nvvm==$CUDA_TOOLKIT_VERSION"
)
readonly OPEN_WEBUI_VENV="${OPEN_WEBUI_VENV:-${OPEN_WEBUI_HOME:-$HOME/.open-webui}/venv}"
readonly MAIN_MODEL=Qwen3.8-Flash-Next-NVFP4

# name | file the directory must contain | required or optional
readonly MODELS=(
  "$MAIN_MODEL|model.safetensors.index.json|required"
  "Qwen3.8-Flash-Next-FTW|freetoken_weight.json|optional"
  "Qwen3.8-27B-Uncensored-NVFP4|model.safetensors.index.json|optional"
  "DeepSeek-V4-Flash-0731|inference/config.json|optional"
  "DeepSeek-V4.1-Flash|inference/config.json|optional"
)

# Same cu130 channels as install.sh: torch and sglang-kernel come from their CUDA 13
# indexes, flashinfer's prebuilt kernels from its own.
readonly CU_INDEX_ARGS=(
  --index-strategy unsafe-best-match
  --extra-index-url https://download.pytorch.org/whl/cu130
  --extra-index-url https://docs.sglang.io/whl/cu130
  --extra-index-url https://flashinfer.ai/whl
  --extra-index-url https://flashinfer.ai/whl/cu130
)

verify=0
reinstall=0
webui=1
assume_yes=0
for arg in "$@"; do
  case "$arg" in
    --verify) verify=1 ;;
    --reinstall) reinstall=1 ;;
    --no-webui) webui=0 ;;
    -y | --yes) assume_yes=1 ;;
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

if [[ -t 1 ]]; then
  C_CYAN=$'\033[1;36m'; C_GREEN=$'\033[1;32m'; C_YELLOW=$'\033[1;33m'; C_RED=$'\033[1;31m'; C_RESET=$'\033[0m'
else
  C_CYAN=''; C_GREEN=''; C_YELLOW=''; C_RED=''; C_RESET=''
fi
problems=()
step() { printf '\n%s==>%s %s\n' "$C_CYAN" "$C_RESET" "$*"; }
ok() { printf '  %sok%s    %s\n' "$C_GREEN" "$C_RESET" "$*"; }
note() { printf '  %s-%s     %s\n' "$C_YELLOW" "$C_RESET" "$*"; }
problem() {
  printf '  %sFAIL%s  %s\n' "$C_RED" "$C_RESET" "$*" >&2
  problems+=("$*")
}

search_roots() {
  local -a roots
  if [[ -n "${FREETOKEN_MODEL_SEARCH:-}" ]]; then
    IFS=: read -ra roots <<<"$FREETOKEN_MODEL_SEARCH"
  else
    # /mnt/ftw first: its ext4 copy of the FTW corpus is faster than the NTFS one.
    roots=(/mnt/ftw /media/"$USER"/* /mnt/*)
  fi
  local root
  for root in "${roots[@]}"; do
    [[ -d "$root" ]] && printf '%s\n' "$root"
  done
}

link_models() {
  step "Model checkpoints ($MODELS_DIR)"
  mkdir -p "$MODELS_DIR"
  local roots entry name marker need link root found
  mapfile -t roots < <(search_roots)
  for entry in "${MODELS[@]}"; do
    IFS='|' read -r name marker need <<<"$entry"
    link="$MODELS_DIR/$name"
    # An existing link that still resolves is kept, so a hand-picked target survives re-runs.
    if [[ -f "$link/$marker" ]]; then
      ok "$name -> $(readlink -f "$link")"
      continue
    fi
    found=''
    for root in "${roots[@]}"; do
      if [[ -f "$root/$name/$marker" ]]; then
        found="$root/$name"
        break
      fi
    done
    if [[ -z "$found" ]]; then
      if [[ "$need" == required ]]; then
        problem "$name not found under: ${roots[*]} (set FREETOKEN_MODEL_SEARCH or mount the drive)"
      else
        note "$name not found (optional)"
      fi
      continue
    fi
    if [[ -e "$link" && ! -L "$link" ]]; then
      problem "$link exists but is not a link or a complete checkpoint; leaving it alone"
      continue
    fi
    ln -sfn "$found" "$link"
    ok "$name -> $found (linked)"
  done
}

verify_checkpoint() {
  local path="$MODELS_DIR/$MAIN_MODEL"
  step "Verifying $MAIN_MODEL against its download hashes"
  [[ -d "$path" ]] || return 0
  if python3 - "$path" <<'PY'; then
import concurrent.futures
import hashlib
import os
import sys
import time

root = os.path.realpath(sys.argv[1])
meta_root = os.path.join(root, ".cache", "huggingface", "download")
if not os.path.isdir(meta_root):
    print(f"  no Hugging Face download metadata under {meta_root}; cannot verify")
    sys.exit(1)

# Each <file>.metadata holds: commit, etag, timestamp. The etag is the SHA-256 of an LFS
# file (64 hex) or the git blob SHA-1 of a small file (40 hex).
expected = {}
commits = set()
for dirpath, _, names in os.walk(meta_root):
    for name in names:
        if name.endswith(".metadata"):
            full = os.path.join(dirpath, name)
            commit, etag = open(full).read().split("\n")[:2]
            expected[os.path.relpath(full, meta_root)[: -len(".metadata")]] = etag
            commits.add(commit)

present = set()
for dirpath, dirnames, names in os.walk(root):
    dirnames[:] = [d for d in dirnames if not (dirpath == root and d == ".cache")]
    present.update(os.path.relpath(os.path.join(dirpath, n), root) for n in names)

added = sorted(present - set(expected))
missing = sorted(set(expected) - present)
total = sum(os.path.getsize(os.path.join(root, r)) for r in expected if r in present)


def matches(rel):
    path = os.path.join(root, rel)
    etag = expected[rel]
    if len(etag) == 64:
        digest = hashlib.sha256()
    else:
        digest = hashlib.sha1(b"blob %d\0" % os.path.getsize(path))
    with open(path, "rb") as f:
        while chunk := f.read(16 << 20):
            digest.update(chunk)
    return rel, digest.hexdigest() == etag, os.path.getsize(path)


print(f"  revision {', '.join(sorted(commits))}: {len(expected)} files, {total / 1e9:.1f} GB")
started = time.monotonic()
done_bytes = 0
next_report = 0.1
changed = []
# hashlib releases the GIL on large updates, so threads keep several reads in flight.
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    for rel, same, size in pool.map(matches, sorted(r for r in expected if r in present)):
        if not same:
            changed.append(rel)
        done_bytes += size
        if total and done_bytes / total >= next_report:
            print(f"  {done_bytes / total:4.0%}  ({time.monotonic() - started:.0f} s)", flush=True)
            next_report += 0.1

for label, names in (("changed", changed), ("missing", missing), ("added", added)):
    for name in names:
        print(f"  {label}: {name}")
if changed or missing or added:
    sys.exit(1)
print(f"  all {len(expected)} files match the downloaded revision")
PY
    ok "$MAIN_MODEL is byte-identical to the downloaded revision"
  else
    problem "$MAIN_MODEL does not match its download hashes (see above)"
  fi
}

ensure_cudart_link() {
  # TVM-FFI's JIT linker and setup.py link -lcudart, but NVIDIA's pip runtime only ships
  # the versioned libcudart.so.13.
  local lib="$CUDA_HOME_DIR/lib"
  if [[ -e "$lib/libcudart.so.13" && ! -e "$lib/libcudart.so" ]]; then
    ln -s libcudart.so.13 "$lib/libcudart.so"
    ok "created $lib/libcudart.so"
  fi
}

installed_drift() {
  # Number of Python modules that differ between this source tree and the installed
  # package. Compiled extensions differ on every build and docs are not packaged, so
  # only .py files are compared.
  local installed="$FT_VENV/lib/python$PY_VERSION/site-packages/freetoken"
  [[ -d "$installed" ]] || { printf '0'; return; }
  python3 - "$REPO_DIR/python/freetoken" "$installed" <<'PY' 2>/dev/null || printf '0'
import filecmp
import os
import sys

source, installed = sys.argv[1:]


def modules(root):
    return {
        os.path.relpath(os.path.join(dirpath, name), root)
        for dirpath, _, names in os.walk(root)
        if "__pycache__" not in dirpath
        for name in names
        if name.endswith(".py")
    }


a, b = modules(source), modules(installed)
changed = [p for p in a & b if not filecmp.cmp(f"{source}/{p}", f"{installed}/{p}", shallow=False)]
print(len(changed) + len(a ^ b))
PY
}

install_freetoken() {
  # Called from an `if`, where set -e does not apply: every step checks its own status.
  if ! command -v uv >/dev/null 2>&1; then
    note "uv is required: curl -LsSf https://astral.sh/uv/install.sh | sh"
    return 1
  fi
  if [[ ! -x "$FT_VENV/bin/python" ]]; then
    uv venv "$FT_VENV" --python "$PY_VERSION" || return 1
  fi
  # setup.py compiles two C++ extensions against CUDA_HOME and checks nvcc's CUDA major
  # against torch's, so the toolchain goes in first.
  uv pip install --python "$FT_VENV" "${CU_INDEX_ARGS[@]}" "torch>=2.11,<2.12" "${TOOLKIT_SPECS[@]}" || return 1
  ensure_cudart_link
  export CUDA_HOME="$CUDA_HOME_DIR"
  export PATH="$CUDA_HOME/bin:$PATH"
  export LIBRARY_PATH="$CUDA_HOME/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
  # Non-editable, as originally installed. --reinstall-package also forces uv to rebuild
  # the wheel instead of reusing a cached build of an older tree.
  # The toolkit pins are repeated so resolving FreeToken's dependencies cannot move them.
  (cd "$REPO_DIR" && uv pip install --python "$FT_VENV" "${CU_INDEX_ARGS[@]}" \
    --reinstall-package freetoken ".[accel]" flashinfer-cubin flashinfer-jit-cache "${TOOLKIT_SPECS[@]}")
}

setup_freetoken() {
  step "FreeToken ($FT_VENV)"
  local version='' drift
  if [[ -x "$FT_VENV/bin/ft" ]]; then
    version="$("$FT_VENV/bin/ft" --version 2>/dev/null)" || version=''
  fi

  if [[ -n "$version" && "$reinstall" == 0 ]]; then
    ok "$version"
    drift="$(installed_drift)"
    if (( drift > 0 )); then
      note "installed package differs from this source tree in $drift Python files (--reinstall rebuilds it)"
    fi
  else
    if [[ -n "$version" ]]; then
      drift="$(installed_drift)"
      if (( drift > 0 )) && [[ "$assume_yes" == 0 ]]; then
        printf '  The installed package differs from this source tree in %s Python files; --reinstall\n' "$drift"
        printf '  replaces them with a fresh build of %s.\n' "$REPO_DIR"
        local answer=''
        { printf '  Continue? [y/N] ' >/dev/tty && read -r answer </dev/tty; } 2>/dev/null || answer=''
        if [[ ! "$answer" =~ ^[yY] ]]; then
          problem "reinstall cancelled (pass --yes to skip this question)"
          return
        fi
      fi
    fi
    note "installing FreeToken from $REPO_DIR (compiles two C++ extensions)"
    if install_freetoken && version="$("$FT_VENV/bin/ft" --version 2>/dev/null)"; then
      ok "$version"
    else
      problem "FreeToken install failed (see the uv output above)"
      return
    fi
  fi

  local toolkit
  toolkit="$(toolkit_versions)"
  if [[ "$toolkit" != "$CUDA_TOOLKIT_VERSION $CUDA_TOOLKIT_VERSION $CUDA_TOOLKIT_VERSION" ]]; then
    note "installing CUDA toolkit $CUDA_TOOLKIT_VERSION (found nvcc/crt/nvvm: ${toolkit:-none})"
    if ! uv pip install --python "$FT_VENV" "${TOOLKIT_SPECS[@]}"; then
      problem "could not install CUDA toolkit $CUDA_TOOLKIT_VERSION"
      return
    fi
  fi
  ensure_cudart_link
  if nvcc_compiles; then
    ok "CUDA toolkit $CUDA_TOOLKIT_VERSION compiles and links a test kernel"
  else
    problem "nvcc in $CUDA_HOME_DIR cannot compile a test kernel (see above)"
  fi
}

toolkit_versions() {
  # Installed nvcc, crt and nvvm wheel versions, space-separated ("?" when absent).
  "$FT_VENV/bin/python" - <<'PY' 2>/dev/null || true
import importlib.metadata as m
versions = []
for name in ("nvidia-cuda-nvcc", "nvidia-cuda-crt", "nvidia-nvvm"):
    try:
        versions.append(m.version(name))
    except m.PackageNotFoundError:
        versions.append("?")
print(" ".join(versions))
PY
}

nvcc_compiles() {
  # A version mismatch inside the toolkit only shows up when nvcc runs ptxas, so build a
  # real kernel rather than trusting `nvcc --version`.
  local work status=0
  work="$(mktemp -d)"
  printf '__global__ void k(float *x) { x[threadIdx.x] *= 2.0f; }\nint main() { return 0; }\n' >"$work/k.cu"
  PATH="$CUDA_HOME_DIR/bin:$PATH" "$CUDA_HOME_DIR/bin/nvcc" -o "$work/k" "$work/k.cu" \
    -L"$CUDA_HOME_DIR/lib" || status=$?
  rm -rf "$work"
  return "$status"
}

setup_workspace() {
  step "Workspace ($REPO_DIR)"
  # The workspace is on NTFS, which cannot hold FlashInfer's wheel layout, so .venv is a
  # link to the real environment. A link written by one NTFS driver (ntfs3) is unreadable
  # through the other (ntfs-3g), so it is recreated whenever it does not resolve.
  local link="$REPO_DIR/.venv"
  if [[ -L "$link" && "$(readlink "$link")" == "$FT_VENV" && -x "$link/bin/python" ]]; then
    ok ".venv -> $FT_VENV"
  elif [[ -e "$link" && ! -L "$link" ]]; then
    note ".venv is a real directory, not a link to $FT_VENV; leaving it alone"
  elif rm -f "$link" && ln -s "$FT_VENV" "$link"; then
    ok ".venv -> $FT_VENV (relinked)"
  else
    problem "could not relink $link to $FT_VENV"
  fi

  # NTFS reports every file as executable, which git would show as a mode change on every
  # tracked file.
  local fstype
  fstype="$(findmnt -no FSTYPE -T "$REPO_DIR" 2>/dev/null || true)"
  case "$fstype" in
    ntfs* | fuseblk | vfat | exfat | msdos)
      if [[ "$(git -C "$REPO_DIR" config --bool core.fileMode 2>/dev/null)" != false ]]; then
        git -C "$REPO_DIR" config core.fileMode false
        ok "git core.fileMode=false (repository is on $fstype)"
      else
        ok "git ignores file modes ($fstype)"
      fi
      ;;
  esac
}

setup_webui() {
  step "Open WebUI ($OPEN_WEBUI_VENV)"
  local version
  if [[ -x "$OPEN_WEBUI_VENV/bin/open-webui" ]]; then
    version="$("$OPEN_WEBUI_VENV/bin/python" -c 'import importlib.metadata as m; print(m.version("open-webui"))' 2>/dev/null || true)"
    ok "Open WebUI ${version:-installed}"
  elif "$SCRIPT_DIR/install-open-webui.sh"; then
    ok "Open WebUI installed"
  else
    problem "Open WebUI install failed (see above)"
  fi
}

check_driver() {
  step "NVIDIA driver"
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    problem "nvidia-smi is unavailable: run scripts/install-nvidia-driver.sh, then reboot"
    return
  fi
  local version
  version="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)"
  if [[ "${version%%.*}" =~ ^[0-9]+$ ]] && (( ${version%%.*} >= 580 )); then
    ok "driver $version"
  else
    problem "driver $version is older than r580: run scripts/install-nvidia-driver.sh, then reboot"
  fi
}

nvfp4_fetch_fraction() {
  # The benched nvfp4 hybrid fetch fraction for GPU 0 (empty when there is none), read with the
  # same loader the engine uses.
  "$FT_VENV/bin/python" - <<'PY' 2>/dev/null
import subprocess
from freetoken.moe.bench_profile import load_hybrid_fetch_fraction
name, uuid = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=name,uuid", "--format=csv,noheader", "-i", "0"], text=True
).strip().split(", ")
fraction = load_hybrid_fetch_fraction("nvfp4", gpu_name=name, gpu_uuid=uuid)
print("" if fraction is None else f"{fraction:.3f}")
PY
}

check_bench_profile() {
  # The hybrid MoE backend splits each decode step's expert misses between PCIe and the CPU by
  # the ratio `ft bench bw` measured. Without an nvfp4 entry it silently falls back to fetching
  # one expert per step: Qwen3.8 decode dropped from ~220-230 to ~150-190 tok/s when another
  # model's bench overwrote the profile (one file per GPU, rewritten by every bench).
  step "MoE bandwidth profile (ft bench bw)"
  if [[ ! -x "$FT_VENV/bin/ft" ]] || ! command -v nvidia-smi >/dev/null 2>&1; then
    note "skipped: FreeToken or the NVIDIA driver is missing"
    return
  fi
  local fraction
  fraction="$(nvfp4_fetch_fraction)" || fraction=''
  if [[ -z "$fraction" ]]; then
    note "no nvfp4 bandwidth profile for this GPU; running ft bench bw --dtype nvfp4 (~15 s)"
    CUDA_HOME="$CUDA_HOME_DIR" PATH="$CUDA_HOME_DIR/bin:$PATH" "$FT_VENV/bin/ft" bench bw --dtype nvfp4 || true
    fraction="$(nvfp4_fetch_fraction)" || fraction=''
  fi
  if [[ -n "$fraction" ]]; then
    ok "hybrid MoE fetches $(awk -v f="$fraction" 'BEGIN{printf "%.1f", f*100}')% of expert misses over PCIe"
  else
    problem "ft bench bw did not produce an nvfp4 profile; decode will use a 1-expert fetch cap"
  fi
}

run_doctor() {
  step "Doctor"
  if [[ ! -f "$MODELS_DIR/$MAIN_MODEL/model.safetensors.index.json" || ! -x "$FT_VENV/bin/ft" ]]; then
    note "skipped: fix the failures above first"
    return
  fi
  if ! FREETOKEN_VENV="$FT_VENV" FREETOKEN_MODEL_PATH="$MODELS_DIR/$MAIN_MODEL" \
    "$SCRIPT_DIR/doctor-qwen38.sh"; then
    problem "doctor-qwen38.sh failed (see above)"
  fi
}

link_models
(( verify )) && verify_checkpoint
setup_freetoken
setup_workspace
(( webui )) && setup_webui
check_driver
check_bench_profile
run_doctor

printf '\n'
if (( ${#problems[@]} )); then
  printf '%sSetup finished with %d problem(s):%s\n' "$C_RED" "${#problems[@]}" "$C_RESET" >&2
  printf '  - %s\n' "${problems[@]}" >&2
  exit 1
fi
printf '%sSetup complete.%s Start the local stack with: ./scripts/serve-local-ai.sh\n' "$C_GREEN" "$C_RESET"
