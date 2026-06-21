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
  shellcheck \
  shfmt

if ! command -v rustup >/dev/null 2>&1; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
fi

# shellcheck source=/dev/null
source "$HOME/.cargo/env"

rustup component add rustfmt clippy
rustup toolchain install nightly --profile minimal
rustup +nightly component add miri

cargo install --locked just || true
cargo install --locked cargo-audit cargo-deny cargo-nextest dprint typos-cli || true

pipx ensurepath
pipx install pre-commit || true

echo "Linux bootstrap complete."
