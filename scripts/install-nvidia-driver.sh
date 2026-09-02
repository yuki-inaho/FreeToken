#!/usr/bin/env bash

set -euo pipefail

if [[ "$(id -u)" -eq 0 ]]; then
  sudo_cmd=()
else
  sudo_cmd=(sudo)
fi

printf 'Installing the Ubuntu NVIDIA r610 open driver. A reboot will be required.\n'
"${sudo_cmd[@]}" apt-get update
"${sudo_cmd[@]}" apt-get install -y nvidia-driver-610-open

printf '\nDriver packages installed. Reboot with: sudo reboot\n'
