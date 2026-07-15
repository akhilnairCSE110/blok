#!/usr/bin/env bash
set -euo pipefail

require() { command -v "$1" >/dev/null || { echo "missing: $1"; exit 1; }; }

if [ "$(uname -s)" != Linux ]; then
  echo "skip: hardware check requires target Linux"
  exit 0
fi

require lscpu
lscpu | grep -qi amd || { echo "missing: AMD CPU"; exit 1; }
require nvidia-smi
nvidia-smi -L | grep -qi nvidia || { echo "missing: NVIDIA GPU"; exit 1; }
require nvcc
nvcc --version >/dev/null
require lsblk
lsblk -d -o NAME,MODEL,TRAN | grep -Eiq 'samsung|evo|nvme' || { echo "missing: Samsung EVO/NVMe SSD"; exit 1; }
test -e "${BLOK_UGDS_DEVICE:-/dev/ugds_drv0}" || { echo "missing: ${BLOK_UGDS_DEVICE:-/dev/ugds_drv0}"; exit 1; }
echo "ok: AMD CPU, NVIDIA GPU, NVMe/Samsung SSD, CUDA, and uGDS device detected"
