#!/usr/bin/env bash
set -euo pipefail

repo=/home/akhil-nair/Desktop/home/code/blok
py=/home/akhil-nair/Desktop/home/venvs/blok-hf/bin/python
hf=/home/akhil-nair/Desktop/home/venvs/blok-hf/bin/hf
log=/home/akhil-nair/Desktop/home/.blok/kimi-download.log
session=blok-kimi-download

case "${1:-start}" in
  start)
    tmux has-session -t "$session" 2>/dev/null && { echo "running: $session"; exit 0; }
    mkdir -p "$(dirname "$log")"
    printf '=== %s start ===\n' "$(date -Is)" >"$log"
    tmux new-session -d -s "$session" "cd '$repo' && BLOK_HF_BIN='$hf' exec '$py' download.py >>'$log' 2>&1"
    echo "started: $session"; echo "log: tail -f $log" ;;
  restart)
    tmux kill-session -t "$session" 2>/dev/null || true
    exec "$0" start ;;
  status)
    tmux has-session -t "$session" 2>/dev/null && echo "running: $session" || echo "not-running" ;;
  attach) exec tmux attach -t "$session" ;;
  log) exec tail -f "$log" ;;
  *) echo "usage: $0 [start|restart|status|attach|log]" >&2; exit 64 ;;
esac
