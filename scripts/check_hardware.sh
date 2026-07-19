#!/usr/bin/env bash
set -euo pipefail

require() { command -v "$1" >/dev/null || { echo "missing: $1"; exit 1; }; }

if [ "$(uname -s)" != Linux ]; then
  echo "skip: hardware check requires target Linux"
  exit 0
fi

require lscpu
lscpu | grep -qi 'Ryzen 9 5950X' || { echo "wrong CPU: expected Ryzen 9 5950X"; exit 1; }
require nvidia-smi
gpu="$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader)"
grep -qi 'RTX 5060 Ti.*, 12\.0' <<<"$gpu" || { echo "wrong GPU: $gpu"; exit 1; }
require nvcc
cuda="$(nvcc --version | sed -n 's/.*release \([0-9][0-9.]*\).*/\1/p' | head -1)"
test "$(printf '%s\n' 12.8 "$cuda" | sort -V | head -1)" = 12.8 || { echo "CUDA 12.8+ required, found $cuda"; exit 1; }
require modinfo
modinfo -F license nvidia | grep -Eqi 'MIT|GPL' || { echo "NVIDIA open kernel modules are required"; exit 1; }
require lsblk
lsblk -d -o MODEL,TRAN | grep -Eiq 'Samsung.*990 EVO Plus.*nvme' || { echo "missing: Samsung 990 EVO Plus NVMe"; exit 1; }
test -e "${BLOK_UGDS_DEVICE:-/dev/ugds_drv0}" || { echo "missing: ${BLOK_UGDS_DEVICE:-/dev/ugds_drv0}"; exit 1; }
echo "ok: exact CPU/GPU/NVMe, CUDA $cuda, open NVIDIA driver, and uGDS device"
