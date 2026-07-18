#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
home="${BLOK_HOME:-$HOME/.blok}"
session="${BLOK_DOWNLOAD_SESSION:-blok-kimi-download}"
log="$home/kimi-download.log"

case "${1:-start}" in
  start)
    tmux has-session -t "$session" 2>/dev/null && { echo "running: $session"; exit 0; }
    mkdir -p "$home"
    tmux new-session -d -s "$session" "cd '$repo' && scripts/model_fetch.py kimi-k2.6 fetch >>'$log' 2>&1"
    echo "started: $session"; echo "log: tail -f $log" ;;
  status) tmux has-session -t "$session" 2>/dev/null && echo "running: $session" || echo "not-running" ;;
  attach) exec tmux attach -t "$session" ;;
  log) exec tail -f "$log" ;;
  *) echo "usage: $0 [start|status|attach|log]" >&2; exit 64 ;;
esac
