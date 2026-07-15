#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

have_files() {
  git ls-files "$@" | grep -q .
}

files() {
  git ls-files "$@"
}

read_files() {
  selected_files=()
  while IFS= read -r file; do
    selected_files+=("$file")
  done < <(files "$@")
}

run_if_available() {
  local command_name="$1"
  shift

  if command -v "$command_name" >/dev/null 2>&1; then
    "$@"
  else
    echo "skip: $command_name is not installed"
  fi
}

has_cargo_project() {
  test -f Cargo.toml
}

has_cmake_project() {
  test -f CMakeLists.txt
}

fmt() {
  local selected_files=()

  run_if_available dprint dprint fmt

  if have_files '*.cpp' '*.cc' '*.cxx' '*.cu' '*.c' '*.h' '*.hpp' '*.hxx'; then
    read_files '*.cpp' '*.cc' '*.cxx' '*.cu' '*.c' '*.h' '*.hpp' '*.hxx'
    run_if_available clang-format clang-format -i "${selected_files[@]}"
  fi

  if has_cargo_project; then
    cargo fmt --all
  fi

  if have_files '*.sh'; then
    read_files '*.sh'
    run_if_available shfmt shfmt -w "${selected_files[@]}"
  fi
}

fmt_check() {
  local selected_files=()

  run_if_available dprint dprint check
  run_if_available typos typos

  if have_files '*.cpp' '*.cc' '*.cxx' '*.cu' '*.c' '*.h' '*.hpp' '*.hxx'; then
    read_files '*.cpp' '*.cc' '*.cxx' '*.cu' '*.c' '*.h' '*.hpp' '*.hxx'
    run_if_available clang-format clang-format --dry-run --Werror "${selected_files[@]}"
  fi

  if has_cargo_project; then
    cargo fmt --all -- --check
    cargo clippy --all-targets --all-features -- -D warnings
  fi

  if have_files '*.sh'; then
    read_files '*.sh'
    run_if_available shellcheck shellcheck "${selected_files[@]}"
    run_if_available shfmt shfmt -d "${selected_files[@]}"
  fi
}

setup() {
  if command -v pre-commit >/dev/null 2>&1; then
    pre-commit install || true
  fi

  if has_cmake_project; then
    cmake -S . -B build -G Ninja -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
  fi
}

cxx_linux() {
  local selected_files=()

  if ! has_cmake_project; then
    echo "skip: no CMakeLists.txt"
    return 0
  fi

  cmake -S . -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
  cmake --build build

  if have_files '*.cpp' '*.cc' '*.cxx' '*.cu'; then
    read_files '*.cpp' '*.cc' '*.cxx' '*.cu'
    run_if_available clang-tidy clang-tidy "${selected_files[@]}" -p build
  fi

  local dirs=()
  test -d src && dirs+=(src)
  test -d include && dirs+=(include)
  if [ "${#dirs[@]}" -gt 0 ]; then
    run_if_available cppcheck cppcheck --enable=warning,style,performance,portability --error-exitcode=1 --inline-suppr "${dirs[@]}"
  fi

  ctest --test-dir build --output-on-failure
}

cxx_sanitize_linux() {
  if ! has_cmake_project; then
    echo "skip: no CMakeLists.txt"
    return 0
  fi

  cmake -S . -B build-asan -G Ninja \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer" \
    -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address,undefined"
  cmake --build build-asan
  ctest --test-dir build-asan --output-on-failure
}

rust_linux() {
  if ! has_cargo_project; then
    echo "skip: no Cargo.toml"
    return 0
  fi

  cargo clippy --all-targets --all-features -- -D warnings
  if command -v cargo-nextest >/dev/null 2>&1; then
    cargo nextest run --all-features
  else
    cargo test --all-features
  fi
}

rust_deep_linux() {
  if ! has_cargo_project; then
    echo "skip: no Cargo.toml"
    return 0
  fi

  if rustup toolchain list | grep -q '^nightly'; then
    cargo +nightly miri test
  else
    echo "skip: nightly toolchain is not installed"
  fi
}

cuda_deep_linux() {
  if ! has_cmake_project; then
    echo "skip: no CMakeLists.txt"
    return 0
  fi
  if ! command -v nvcc >/dev/null 2>&1; then
    echo "skip: nvcc is not installed"
    return 0
  fi

  cmake -S . -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Debug
  cmake --build build-cuda
  run_if_available compute-sanitizer compute-sanitizer --tool memcheck build-cuda/blok-kimi-exec --help
  run_if_available compute-sanitizer compute-sanitizer --tool synccheck build-cuda/blok-kimi-exec --help
}

deps() {
  if ! has_cargo_project; then
    echo "skip: no Cargo.toml"
    return 0
  fi

  if [ ! -f Cargo.lock ]; then
    echo "skip: no Cargo.lock"
    return 0
  fi
  run_if_available cargo-deny cargo deny check
  run_if_available cargo-audit cargo audit
  run_if_available cargo-geiger cargo geiger --all-features
}

hardware_check() {
  scripts/check_hardware.sh
}

model_contract() {
  if [ -z "${BLOK_MODEL:-}" ]; then
    echo "skip: set BLOK_MODEL to manifest.blok or model dir"
    return 0
  fi
  scripts/check_kimi_contract.py "$BLOK_MODEL"
}

check_local() {
  fmt_check
}

verify_l0() {
  check_local
}

verify_l1() {
  model_contract
}

verify_l2() {
  cuda_deep_linux
}

verify_l3() {
  echo "missing: official tokenizer/MLA/logit parity fixtures are not implemented"
  return 1
}

verify_l4() {
  hardware_check
}

verify_l5() {
  echo "missing: revision-keyed regression fixtures are not implemented"
  return 1
}

check_linux() {
  fmt_check
  cxx_linux
  rust_linux
  deps
}

deep_linux() {
  check_linux
  cxx_sanitize_linux
  rust_deep_linux
  cuda_deep_linux
  model_contract
}

release_linux() {
  if has_cmake_project; then
    cmake -S . -B build-release -G Ninja -DCMAKE_BUILD_TYPE=Release
    cmake --build build-release
  else
    echo "skip: no CMakeLists.txt"
  fi

  if has_cargo_project; then
    local lock_flag=()
    test -f Cargo.lock && lock_flag+=(--locked)
    cargo build --release "${lock_flag[@]}"
  else
    echo "skip: no Cargo.toml"
  fi

  scripts/package_release.sh
}

case "${1:-}" in
  fmt) fmt ;;
  fmt-check) fmt_check ;;
  setup) setup ;;
  cxx-linux) cxx_linux ;;
  cxx-sanitize-linux) cxx_sanitize_linux ;;
  rust-linux) rust_linux ;;
  rust-deep-linux) rust_deep_linux ;;
  cuda-deep-linux) cuda_deep_linux ;;
  deps) deps ;;
  hardware-check) hardware_check ;;
  model-contract) model_contract ;;
  verify-l0) verify_l0 ;;
  verify-l1) verify_l1 ;;
  verify-l2) verify_l2 ;;
  verify-l3) verify_l3 ;;
  verify-l4) verify_l4 ;;
  verify-l5) verify_l5 ;;
  check-local) check_local ;;
  check-linux) check_linux ;;
  deep-linux) deep_linux ;;
  release-linux) release_linux ;;
  *)
    echo "usage: $0 {fmt|fmt-check|setup|cxx-linux|cxx-sanitize-linux|rust-linux|rust-deep-linux|cuda-deep-linux|deps|hardware-check|model-contract|verify-l0|verify-l1|verify-l2|verify-l3|verify-l4|verify-l5|check-local|check-linux|deep-linux|release-linux}" >&2
    exit 2
    ;;
esac
