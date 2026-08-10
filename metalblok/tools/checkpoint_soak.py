#!/usr/bin/env python3
"""Crash-contained MetalBlok decode soak.

Each iteration reads the token predicted by the last committed checkpoint,
launches exactly one model step in a fresh process, and accepts success only if
the atomically replaced checkpoint advances by exactly one position.  A killed
or timed-out child cannot invalidate the previously committed state.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import pathlib
import struct
import subprocess
import sys


HEADER = struct.Struct("<8s7I")


def read_state(path: pathlib.Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        raw = handle.read(HEADER.size)
    if len(raw) != HEADER.size:
        raise RuntimeError(f"short checkpoint header: {len(raw)} bytes")
    magic, version, layers, max_seq, kv_rank, rope_dim, pos, token = HEADER.unpack(raw)
    if magic != b"MBLKSTAT" or version != 2:
        raise RuntimeError(f"unsupported checkpoint magic/version: {magic!r}/{version}")
    if layers != 61 or kv_rank != 512 or rope_dim != 64:
        raise RuntimeError(
            f"checkpoint is not the expected R1 graph: layers={layers} "
            f"kv_rank={kv_rank} rope_dim={rope_dim}"
        )
    if pos >= max_seq:
        raise RuntimeError(f"checkpoint is full: pos={pos} max_seq={max_seq}")
    return pos, token


def append_log(handle, text: str) -> None:
    handle.write(text)
    if text and not text.endswith("\n"):
        handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=pathlib.Path)
    parser.add_argument("--model", required=True, type=pathlib.Path)
    parser.add_argument("--state", required=True, type=pathlib.Path)
    parser.add_argument("--log", required=True, type=pathlib.Path)
    parser.add_argument("--steps", required=True, type=int)
    parser.add_argument("--context", type=int, default=64)
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()

    if args.steps < 1:
        parser.error("--steps must be positive")
    for label, path in (("binary", args.binary), ("model", args.model), ("state", args.state)):
        if not path.is_file():
            parser.error(f"{label} does not exist: {path}")
    partial = pathlib.Path(str(args.state) + ".partial")
    if partial.exists():
        parser.error(f"stale partial checkpoint requires inspection: {partial}")

    args.log.parent.mkdir(parents=True, exist_ok=True)
    with args.log.open("a", encoding="utf-8", buffering=1) as log:
        append_log(log, f"\n=== soak start {dt.datetime.now(dt.timezone.utc).isoformat()} ===")
        for index in range(args.steps):
            before_pos, input_token = read_state(args.state)
            command = [
                str(args.binary), "-m", str(args.model), "-p", "x",
                "--context", str(args.context), "--state", str(args.state),
                "--single-step-token", str(input_token),
            ]
            append_log(log, f"--- step={index + 1} before_pos={before_pos} input_token={input_token}")
            try:
                result = subprocess.run(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=args.timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                output = exc.stdout or ""
                if isinstance(output, bytes):
                    output = output.decode("utf-8", errors="replace")
                append_log(log, output)
                append_log(log, f"FAIL timeout={args.timeout}s; prior checkpoint remains authoritative")
                return 124
            append_log(log, result.stdout)
            if result.returncode != 0:
                append_log(log, f"FAIL exit={result.returncode}; prior checkpoint remains authoritative")
                return result.returncode or 1
            after_pos, next_token = read_state(args.state)
            if after_pos != before_pos + 1:
                append_log(log, f"FAIL checkpoint advanced {before_pos}->{after_pos}, expected +1")
                return 3
            if partial.exists():
                append_log(log, f"FAIL partial checkpoint remains after successful child: {partial}")
                return 4
            print(
                f"soak_step={index + 1}/{args.steps} pos={after_pos} "
                f"input={input_token} next={next_token}",
                flush=True,
            )
        append_log(log, f"=== soak pass steps={args.steps} final_pos={read_state(args.state)[0]} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
