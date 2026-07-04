#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

MODEL = "kimi-k2.6"
DEFAULT_BLOK_HOME = Path("/home/akhil-nair/Desktop/home/.blok")
DEFAULT_HF_BIN = DEFAULT_BLOK_HOME / "tools/hf/bin/hf"
SCRIPT = Path(__file__).resolve().parent / "scripts/model_fetch.py"
MIN_FREE_BUFFER_BYTES = 50 * 1024 * 1024 * 1024


def fail(message):
    print(f"download.py: {message}", file=sys.stderr)
    raise SystemExit(1)


def run(cmd, env=None, capture=False):
    kwargs = {"env": env, "check": False}
    if capture:
        kwargs |= {"text": True, "stdout": subprocess.PIPE, "stderr": subprocess.PIPE}
    p = subprocess.run(cmd, **kwargs)
    if p.returncode != 0:
        if capture and p.stderr:
            print(p.stderr.strip(), file=sys.stderr)
        fail(f"command failed with exit {p.returncode}: {' '.join(str(x) for x in cmd)}")
    return p


def load_status(env):
    p = run([sys.executable, str(SCRIPT), MODEL, "status"], env=env, capture=True)
    try:
        return json.loads(p.stdout)
    except json.JSONDecodeError as e:
        fail(f"status output was not JSON: {e}")


def validate(args):
    repo = Path(__file__).resolve().parent
    if Path.cwd().resolve() != repo:
        fail(f"run from repo root: {repo}")
    if not SCRIPT.is_file():
        fail(f"missing fetch script: {SCRIPT}")

    blok_home = Path(args.blok_home).expanduser().resolve()
    token = os.environ.get("HF_TOKEN", "").strip()
    cache_token = blok_home / "hf-cache/token"
    stored_tokens = blok_home / "hf-cache/stored_tokens"
    if token and any(c.isspace() for c in token):
        fail("HF_TOKEN contains whitespace")
    if not token and not cache_token.is_file() and not stored_tokens.is_file():
        fail(
            "Hugging Face auth was not found; set HF_TOKEN or run "
            f"`HF_HOME={blok_home / 'hf-cache'} hf auth login`"
        )

    hf_bin = Path(args.hf_bin).expanduser().resolve()
    if not hf_bin.is_file():
        fail(f"missing Hugging Face CLI: {hf_bin}")
    if not os.access(hf_bin, os.X_OK):
        fail(f"Hugging Face CLI is not executable: {hf_bin}")

    blok_home.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "BLOK_HOME": str(blok_home),
            "BLOK_HF_BIN": str(hf_bin),
            "HF_HOME": str(blok_home / "hf-cache"),
            "HF_HUB_CACHE": str(blok_home / "hf-cache/hub"),
            "HF_XET_CACHE": str(blok_home / "hf-cache/xet"),
            "HF_XET_HIGH_PERFORMANCE": "1",
            "HF_XET_NUM_CONCURRENT_RANGE_GETS": os.environ.get("HF_XET_NUM_CONCURRENT_RANGE_GETS", "32"),
            "HF_HUB_DOWNLOAD_TIMEOUT": os.environ.get("HF_HUB_DOWNLOAD_TIMEOUT", "60"),
        }
    )

    run([sys.executable, "-m", "py_compile", str(SCRIPT)], env=env)
    hf_version = run([str(hf_bin), "--version"], env=env, capture=True).stdout.strip()
    status = load_status(env)

    expected = int(status["expected_bytes"])
    downloaded = int(status["downloaded_bytes"])
    remaining = max(0, expected - downloaded)
    usage = shutil.disk_usage(blok_home)
    required = remaining + MIN_FREE_BUFFER_BYTES
    if usage.free < required and not status["complete"]:
        fail(
            "not enough free disk for unattended download: "
            f"free={usage.free} required_at_least={required} remaining_model_bytes={remaining}"
        )

    print("download.py preflight ok")
    print(f"  model: {status['repo']} @ {status['revision']}")
    print(f"  destination: {status['local_dir']}")
    print(f"  cache: {status['cache_dir']}")
    print(f"  hf: {hf_bin} ({hf_version})")
    print(f"  progress: {downloaded}/{expected} bytes, safetensors {status['safetensors']}/{status['expected_safetensors']}")
    print(f"  free_disk_bytes: {usage.free}")
    print(f"  complete: {status['complete']}")
    return env, status


def main():
    parser = argparse.ArgumentParser(description="Validated one-shot Kimi K2.6 downloader for Blok.")
    parser.add_argument("--blok-home", default=str(DEFAULT_BLOK_HOME))
    parser.add_argument("--hf-bin", default=str(DEFAULT_HF_BIN))
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    env, status = validate(args)
    if args.preflight_only or status["complete"]:
        return
    print("starting fetch; leave this running inside tmux")
    run([sys.executable, str(SCRIPT), MODEL, "fetch"], env=env)
    status = load_status(env)
    if not status["complete"]:
        fail(
            "fetch exited before the model was complete: "
            f"{status['downloaded_bytes']}/{status['expected_bytes']} bytes, "
            f"safetensors {status['safetensors']}/{status['expected_safetensors']}"
        )


if __name__ == "__main__":
    main()
