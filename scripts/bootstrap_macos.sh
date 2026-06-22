#!/usr/bin/env bash
set -euo pipefail

if ! command -v brew >/dev/null 2>&1; then
  echo "Homebrew is required for macOS bootstrap: https://brew.sh" >&2
  exit 1
fi

brew update
brew install \
  clang-format \
  cmake \
  cppcheck \
  dprint \
  just \
  ninja \
  pipx \
  shellcheck \
  shfmt \
  typos-cli || true

if ! command -v rustup >/dev/null 2>&1; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
fi

# shellcheck source=/dev/null
source "$HOME/.cargo/env"

rustup component add rustfmt clippy
cargo install --locked cargo-audit cargo-deny cargo-nextest || true

pipx ensurepath
pipx install pre-commit || true

echo "macOS bootstrap complete."
