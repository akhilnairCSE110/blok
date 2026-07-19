#!/usr/bin/env bash
set -euo pipefail

sudo apt-get update
sudo apt-get install -y \
  build-essential \
  clang \
  clang-format \
  clang-tidy \
  cmake \
  cppcheck \
  curl \
  git \
  libssl-dev \
  ninja-build \
  pipx \
  pkg-config \
  python3-venv \
  shellcheck \
  shfmt

python3 -m venv .venv
.venv/bin/python -m pip install --requirement requirements.txt

if ! command -v rustup >/dev/null 2>&1; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
fi

# shellcheck source=/dev/null
source "$HOME/.cargo/env"

if [ -n "${GITHUB_PATH:-}" ]; then
  printf '%s\n' "$HOME/.cargo/bin" >>"$GITHUB_PATH"
fi
rustup component add rustfmt clippy
rustup toolchain install nightly --profile minimal
rustup +nightly component add miri

if [ "${GITHUB_ACTIONS:-}" = "true" ]; then
  cargo install --locked just
  cargo install --locked cargo-audit cargo-deny cargo-geiger cargo-nextest dprint typos-cli
else
  cargo install --locked just || true
  cargo install --locked cargo-audit cargo-deny cargo-geiger cargo-nextest dprint typos-cli || true
fi

pipx ensurepath
pipx install pre-commit || true

echo "Linux bootstrap complete."
