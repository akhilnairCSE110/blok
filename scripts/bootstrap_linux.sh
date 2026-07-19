#!/usr/bin/env bash
set -euo pipefail

sudo apt-get update
sudo apt-get install -y \
  build-essential \
  clang-format \
  cmake \
  git \
  linux-headers-"$(uname -r)" \
  ninja-build \
  pkg-config \
  python3-venv \
  shellcheck

python3 -m venv .venv
.venv/bin/python -m pip install --requirement requirements.txt
