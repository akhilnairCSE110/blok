#!/usr/bin/env python3
"""Run a small, parity-gated decode schedule synthesis sweep on the target."""

import argparse
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "metalblok/runs"
DEFAULT_CONFIGS = "0:4,4:1,4:2,4:4,4:8,8:4"


def config_list(value: str) -> list[tuple[int, int]]:
    result = []
    for item in value.split(","):
        ways, group = map(int, item.split(":"))
        if not 0 <= ways <= 32 or group not in (1, 2, 4, 8):
            raise argparse.ArgumentTypeError("configs require ways=0..32 and group=1,2,4,8")
        result.append((ways, group))
    if not result:
        raise argparse.ArgumentTypeError("empty config list")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", required=True, type=Path,
                        help="golden checkpoint; never modified")
    parser.add_argument("--model", type=Path)
    parser.add_argument("-n", "--tokens", type=int, default=32)
    parser.add_argument("--configs", type=config_list,
                        default=config_list(DEFAULT_CONFIGS),
                        help=f"ways:group pairs (default {DEFAULT_CONFIGS})")
    parser.add_argument("--max-cache-gb", type=float, default=6.0)
    args = parser.parse_args()
    source = args.state.expanduser().resolve()
    if not source.is_file():
        raise SystemExit(f"missing checkpoint: {source}")

    logs: list[Path] = []
    clones: list[Path] = []
    stamp = time.time_ns()
    try:
        for index, (ways, group) in enumerate(args.configs):
            clone = Path("/tmp") / f"metalblok-tune-{os.getpid()}-{stamp}-{index}.state"
            subprocess.run(["cp", "-c", str(source), str(clone)], check=True)
            clones.append(clone)
            command = [str(ROOT / "run_blok.py"), "ignored", "--state", str(clone),
                       "--continue-decode", "--mla", "--profile-layers",
                       "--expert-cache-ways", str(ways), "--expert-group-size", str(group),
                       "-n", str(args.tokens)]
            if args.model:
                command += ["--model", str(args.model.expanduser().resolve())]
            started = time.time_ns()
            print(f"\n[synthesis] ways={ways} group={group}", flush=True)
            subprocess.run(command, cwd=ROOT, check=True)
            candidates = [path for path in RUNS.glob("run-*.log")
                          if path.stat().st_mtime_ns >= started]
            if not candidates:
                raise RuntimeError("run completed without a diagnostics log")
            logs.append(max(candidates, key=lambda path: path.stat().st_mtime_ns))
    finally:
        for clone in clones:
            clone.unlink(missing_ok=True)

    subprocess.run([str(ROOT / "scripts/synthesize_decode_config.py"),
                    *map(str, logs), "--max-cache-gb", str(args.max_cache_gb)],
                   cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
