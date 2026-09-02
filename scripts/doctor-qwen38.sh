#!/usr/bin/env bash

set -euo pipefail

readonly FT_VENV="${FREETOKEN_VENV:-/home/nuko/.freetoken/venv}"
readonly PYTHON="$FT_VENV/bin/python"
readonly MODEL_PATH="${FREETOKEN_MODEL_PATH:-/media/nuko/New Volume/Qwen3.8-Flash-Next-NVFP4}"
readonly CUDA_HOME_DEFAULT="$FT_VENV/lib/python3.12/site-packages/nvidia/cu13"

failed=0

if [[ -x "$FT_VENV/bin/ft" ]]; then
  "$FT_VENV/bin/ft" --version
else
  printf 'FAIL: FreeToken executable is missing from %s\n' "$FT_VENV" >&2
  failed=1
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  gpu_line="$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -n 1)"
  printf 'GPU: %s\n' "$gpu_line"
  driver_version="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)"
  driver_major="${driver_version%%.*}"
  if [[ ! "$driver_major" =~ ^[0-9]+$ || "$driver_major" -lt 580 ]]; then
    printf 'FAIL: NVIDIA driver r580 or newer is required.\n' >&2
    failed=1
  fi
else
  printf 'FAIL: nvidia-smi is unavailable.\n' >&2
  failed=1
fi

export CUDA_HOME="${FREETOKEN_CUDA_HOME:-$CUDA_HOME_DEFAULT}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LIBRARY_PATH="$CUDA_HOME/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"

if [[ -x "$CUDA_HOME/bin/nvcc" ]]; then
  "$CUDA_HOME/bin/nvcc" --version | tail -n 1
else
  printf 'FAIL: CUDA 13 nvcc is missing from %s\n' "$CUDA_HOME" >&2
  failed=1
fi

if [[ ! -e "$CUDA_HOME/lib/libcudart.so" && ! -e "$CUDA_HOME/lib64/libcudart.so" ]]; then
  printf 'FAIL: the CUDA linker cannot find an unversioned libcudart.so.\n' >&2
  failed=1
fi

if [[ ! -x "$PYTHON" ]]; then
  printf 'FAIL: Python is missing from %s\n' "$FT_VENV" >&2
  exit 1
fi

export FREETOKEN_DOCTOR_MODEL_PATH="$MODEL_PATH"
"$PYTHON" - <<'PY' || failed=1
import json
import os
from pathlib import Path

import torch
from freetoken.models.register import _load_attr, get_model_spec
from transformers import AutoConfig, AutoTokenizer

root = Path(os.environ["FREETOKEN_DOCTOR_MODEL_PATH"])
index_path = root / "model.safetensors.index.json"
if not index_path.is_file():
    raise SystemExit(f"FAIL: missing {index_path}")

index = json.loads(index_path.read_text())
files = sorted(set(index["weight_map"].values()))
missing = [name for name in files if not (root / name).is_file()]
empty = [name for name in files if (root / name).is_file() and not (root / name).stat().st_size]
if missing or empty:
    raise SystemExit(f"FAIL: checkpoint has {len(missing)} missing and {len(empty)} empty files")

config = AutoConfig.from_pretrained(root, local_files_only=True)
architecture = config.architectures[0]
spec = get_model_spec(architecture)
model_config = _load_attr(spec.module, spec.parse_config)(config)
tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
tokenizer.apply_chat_template([{"role": "user", "content": "Hello"}], tokenize=True)

print(f"Checkpoint: {len(index['weight_map'])} tensors in {len(files)} files")
print(f"Architecture: {architecture} via {spec.module}")
print(
    f"Geometry: {model_config.num_layers} layers, "
    f"{model_config.num_experts} experts/layer, top-{model_config.num_experts_per_tok}"
)
print(f"Torch: {torch.__version__} (CUDA {torch.version.cuda})")
if not torch.cuda.is_available():
    raise SystemExit("FAIL: Torch cannot initialize CUDA; check the NVIDIA driver and reboot")
print(f"CUDA device: {torch.cuda.get_device_name(0)} {torch.cuda.get_device_capability(0)}")
PY

if (( failed )); then
  exit 1
fi

printf 'FreeToken and Qwen3.8-Flash-Next-NVFP4 are ready.\n'
