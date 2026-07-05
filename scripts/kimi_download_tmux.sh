#!/usr/bin/env bash
set -euo pipefail

repo=/home/akhil-nair/Desktop/home/code/blok
python_bin=/home/akhil-nair/Desktop/home/venvs/blok-hf/bin/python
hf_bin=/home/akhil-nair/Desktop/home/venvs/blok-hf/bin/hf
log=/home/akhil-nair/Desktop/home/.blok/kimi-download.log
session=blok-kimi-download
action=${1:-start}

case "$action" in
  start|restart)
    if [ "$action" = restart ]; then
      tmux kill-session -t "$session" 2>/dev/null || true
    elif tmux has-session -t "$session" 2>/dev/null; then
      echo "session already running: $session"
      echo "attach: tmux attach -t $session"
      echo "log: tail -f $log"
      exit 0
    fi
    ;;
  status)
    tmux has-session -t "$session" 2>/dev/null && tmux list-sessions | rg "^${session}:" || echo "$session:not-running"
    exit 0
    ;;
  attach)
    exec tmux attach -t "$session"
    ;;
  log)
    exec tail -f "$log"
    ;;
  *)
    echo "usage: $0 [start|restart|status|attach|log]" >&2
    exit 64
    ;;
esac

token=${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}
if [ -n "$token" ]; then
  export HF_TOKEN="$token"
  export HUGGING_FACE_HUB_TOKEN="$token"
fi

mkdir -p "$(dirname "$log")"
if "$hf_bin" auth whoami >/dev/null 2>&1; then
  echo "hugging face: logged in"
else
  echo "hugging face: not logged in"
  read -rsp "HF token: " token
  echo
  export HF_TOKEN="$token"
  export HUGGING_FACE_HUB_TOKEN="$token"
  if "$hf_bin" auth whoami >/dev/null 2>&1; then
    echo "hugging face: logged in"
  else
    echo "hugging face: invalid token"
    exit 1
  fi
fi
export BLOK_HF_BIN="$hf_bin"
tmux new-session -d -s "$session" \
  "cd '$repo' && exec '$python_bin' download.py >> '$log' 2>&1"

echo "started tmux session: $session"
echo "attach: tmux attach -t $session"
echo "log: tail -f $log"
