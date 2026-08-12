#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
target_env="${BLOK_TARGET_ENV:-$repo_root/target.env}"
# shellcheck source=/dev/null
test ! -f "$target_env" || source "$target_env"

revision=7eb5002f6aadc958aed6a9177b7ed26bb94011bb
blok_home="${BLOK_HOME:-$HOME/.blok}"
model_root="${BLOK_MODEL_ROOT:-$blok_home/models}/moonshotai/Kimi-K2.6"
model_dir="$model_root/source/hf/$revision"
index="${BLOK_META_ROOT:-$blok_home/metadata}/moonshotai/Kimi-K2.6/runtime-index.blok"
env_file="${BLOK_UGDS_ENV_OUTPUT:-$repo_root/ugds.env}"
proof_kv_bytes=99942400000

require_linux() {
  test "$(uname -s)" = Linux || { echo "target_v0.sh requires target Linux" >&2; exit 1; }
}

require_source() {
  test -f sub_dir/uGDS/CMakeLists.txt || {
    echo "missing uGDS submodule; run: git submodule update --init --recursive" >&2
    exit 1
  }
  test -x .venv/bin/python || {
    echo "missing .venv; run: scripts/bootstrap_linux.sh" >&2
    exit 1
  }
}

prepare() {
  require_linux
  require_source
  .venv/bin/python -m scripts.test_host
  scripts/model_fetch.py kimi-k2.6 materialize
  .venv/bin/python -m scripts.check_kimi_contract "$index"
  test "$(stat -c %d "$model_dir")" != "$(stat -c %d "$(dirname "$index")")" || {
    echo "model payload and Blok metadata must be on different filesystems before uGDS unbind" >&2
    exit 1
  }
  test -n "${BLOK_KV_UGDS_BASE:-}" || { echo "set BLOK_KV_UGDS_BASE" >&2; exit 1; }
  test -n "${BLOK_KV_UGDS_BYTES:-}" || { echo "set BLOK_KV_UGDS_BYTES" >&2; exit 1; }
  [[ "$BLOK_KV_UGDS_BYTES" =~ ^[0-9]+$ ]] && (( BLOK_KV_UGDS_BYTES >= proof_kv_bytes )) || {
    echo "BLOK_KV_UGDS_BYTES must reserve at least $proof_kv_bytes bytes for the 10k+10k proof" >&2
    exit 1
  }
  layout_args=("$index" --output "${BLOK_UGDS_MAP:?set BLOK_UGDS_MAP}" --env-output "$env_file")
  test -z "${BLOK_UGDS_PHYSICAL_OFFSET_ADD:-}" || layout_args+=(--physical-offset-add "$BLOK_UGDS_PHYSICAL_OFFSET_ADD")
  .venv/bin/python scripts/plan_ugds_layout.py "${layout_args[@]}"
  cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
  cmake --build build --target blok-kimi-exec
  make -C sub_dir/uGDS/drv
  echo "prepared: $index"
  echo "next: unmount the model filesystem, bind its NVMe controller to uGDS, then run: scripts/target_v0.sh run"
}

fetch() {
  require_linux
  require_source
  .venv/bin/python -m scripts.model_fetch kimi-k2.6 fetch
}

activate_runtime() {
  require_linux
  require_source
  test -f "$env_file" || { echo "missing $env_file; run prepare first" >&2; exit 1; }
  # shellcheck source=/dev/null
  source "$env_file"
  export BLOK_KIMI_EXEC_BIN="$repo_root/build/blok-kimi-exec"
  export BLOK_MODEL="$index"
  scripts/check_hardware.sh
  .venv/bin/python -m scripts.check_kimi_contract "$index"
}

run() {
  activate_runtime
  .venv/bin/python -m scripts.smoke_kimi "$index"
}

prove() {
  activate_runtime
  .venv/bin/python -m scripts.prove_kimi_10k "$index" "$@"
}

serve() {
  activate_runtime
  exec .venv/bin/python -m scripts.chat_server --model "$index" "$@"
}

action="${1:-}"
test "$#" -eq 0 || shift
case "$action" in
  fetch) fetch "$@" ;;
  prepare) prepare "$@" ;;
  run) run "$@" ;;
  prove) prove "$@" ;;
  serve) serve "$@" ;;
  *) echo "usage: $0 {fetch|prepare|run|prove|serve}" >&2; exit 2 ;;
esac
