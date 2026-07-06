#!/usr/bin/env python3
import json, os, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HOME = Path(os.getenv("BLOK_HOME", "/home/akhil-nair/Desktop/home/.blok"))
HF = os.getenv("BLOK_HF_BIN", "/home/akhil-nair/Desktop/home/venvs/blok-hf/bin/hf")
FETCH = ROOT / "scripts/model_fetch.py"
MODEL = "kimi-k2.6"
BUFFER = 50 * 1024**3

def run(*cmd, capture=False, env=None):
    p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE if capture else None, check=False, env=env)
    if p.returncode: raise SystemExit(p.returncode)
    return p.stdout if capture else ""

def status(env):
    return json.loads(run(sys.executable, str(FETCH), MODEL, "status", capture=True, env=env))

def main():
    if Path.cwd().resolve() != ROOT: raise SystemExit(f"run from {ROOT}")
    if not Path(HF).is_file(): raise SystemExit(f"missing hf cli: {HF}")
    env = os.environ | {"BLOK_HOME": str(HOME), "BLOK_HF_BIN": HF}
    run(sys.executable, "-m", "py_compile", str(FETCH), env=env)
    s = status(env)
    free = shutil.disk_usage(HOME).free
    need = max(0, s["expected_bytes"] - s["downloaded_bytes"]) + BUFFER
    print(f"{s['repo']} {s['safetensors']}/{s['expected_safetensors']} shards {s['downloaded_bytes']}/{s['expected_bytes']} bytes free={free}")
    if s["complete"]: return
    if free < need: raise SystemExit(f"not enough disk: need {need}, free {free}")
    os.execve(sys.executable, [sys.executable, str(FETCH), MODEL, "fetch"], env)

if __name__ == "__main__": main()
