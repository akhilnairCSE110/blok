#!/usr/bin/env python3
"""Run or resume the pinned DeepSeek-R1 GGUF on Apple Silicon."""

from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import platform
import re
import struct
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
BINARY = ROOT / "metalblok/build/metalblok"
MODEL = Path(
    "/Users/akhil/Desktop/AKHIL_HOME/ai/blade/hf_models/DeepSeek-R1-671B/"
    "DeepSeek-R1-UD-IQ1_S/DeepSeek-R1-UD-IQ1_S-00001-of-00003.gguf"
)
HEADER = struct.Struct("<8s7I")


class Error(RuntimeError):
    pass


def note(message: str) -> None:
    print(f"[MetalBlok] {message}", file=sys.stderr, flush=True)


def checked(command: list[str], timeout: int = 300) -> str:
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        detail = result.stderr.strip().splitlines()
        raise Error(detail[-1] if detail else f"exit {result.returncode}")
    return result.stdout


def build() -> None:
    if BINARY.is_file() and os.access(BINARY, os.X_OK):
        return
    note("building Metal runtime")
    checked(["cmake", "-S", str(ROOT / "metalblok"), "-B", str(BINARY.parent)])
    checked(["cmake", "--build", str(BINARY.parent), "-j", "8"])


def preflight(model: Path) -> None:
    first = checked([str(BINARY), "--preflight", str(model)], 30).splitlines()[0]
    if "all_resident=true" not in first or "manifest=deepseek-r1-ud-iq1_s" not in first:
        raise Error("model is incomplete or is not the pinned three-shard checkpoint")


def token_count(model: Path, prompt: str | Path, from_file: bool = False) -> int:
    source = ["--prompt-file", str(prompt)] if from_file else ["-p", str(prompt)]
    output = checked([str(BINARY), "-m", str(model), *source, "--tokenize-only"], 30)
    match = re.search(r"^token_ids=([0-9,]+)$", output, re.MULTILINE)
    if not match:
        raise Error("tokenizer returned no token IDs")
    return match.group(1).count(",") + 1


def state_context(path: Path) -> tuple[int, int]:
    try:
        with path.open("rb") as stream:
            raw = stream.read(HEADER.size)
        magic, version, layers, context, rank, rope, pos, token = HEADER.unpack(raw)
    except (OSError, struct.error) as exc:
        raise Error(f"invalid checkpoint {path}: {exc}") from exc
    if magic != b"MBLKSTAT" or version not in (3, 4) or (layers, rank, rope) != (61, 512, 64):
        raise Error(f"checkpoint does not match DeepSeek-R1: {path}")
    if pos > context or token >= 129280:
        raise Error(f"checkpoint header is corrupt: {path}")
    return context, pos


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream DeepSeek-R1 671B from SSD through Metal")
    parser.add_argument("prompt", nargs="?", help="user message")
    parser.add_argument("--prompt-file", type=Path,
                        help="read a long prompt without the shell argument-size limit")
    parser.add_argument("-n", "--max-output-tokens", type=int, default=256)
    parser.add_argument("--context", type=int, default=2_048)
    parser.add_argument("--model", type=Path, default=Path(os.getenv("METALBLOK_MODEL", MODEL)))
    parser.add_argument("--state", type=Path, help="new or existing conversation checkpoint")
    parser.add_argument("--continue-decode", action="store_true",
                        help="continue the unfinished decode loop in --state")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--checkpoint-every", type=int, default=256)
    parser.add_argument("--stall-timeout", type=int, default=300)
    parser.add_argument("--trace", action="store_true", help="log per-layer numerical telemetry")
    parser.add_argument("--profile-layers", action="store_true",
                        help="log per-layer GPU and NVMe attribution")
    parser.add_argument("--profile-ops", action="store_true",
                        help="split profiled stages and log per-operation GPU timing")
    parser.add_argument("--mla", action="store_true",
                        help="use explicit compact MLA mode for long context")
    parser.add_argument("--validate-mla", action="store_true",
                        help="dual-run reference/online MLA at context 1024 and log error")
    return parser.parse_args()


def run(args: argparse.Namespace) -> int:
    e2e_started = time.monotonic()
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise Error("MetalBlok requires Apple Silicon macOS")
    if bool(args.prompt) == bool(args.prompt_file):
        raise Error("provide exactly one of prompt or --prompt-file")
    if args.prompt_file and (not args.prompt_file.is_file() or args.prompt_file.stat().st_size == 0):
        raise Error(f"prompt file is missing or empty: {args.prompt_file}")
    if not 1 <= args.max_output_tokens <= 131_072:
        raise Error("output tokens must be in [1,131072]")
    if not 64 <= args.context <= 163_840 or not 30 <= args.stall_timeout <= 3_600:
        raise Error("context must be [64,163840] and stall timeout [30,3600] seconds")
    if not args.model.is_file():
        raise Error(f"model does not exist: {args.model}")

    phase = time.monotonic()
    build()
    build_s = time.monotonic() - phase
    note("verifying three physically resident model shards")
    phase = time.monotonic()
    preflight(args.model)
    preflight_s = time.monotonic() - phase
    phase = time.monotonic()
    inputs = token_count(args.model, args.prompt_file or args.prompt,
                         args.prompt_file is not None)
    tokenize_s = time.monotonic() - phase
    stamp = f"{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"
    state = (args.state or Path("/tmp") / f"metalblok-{stamp}.state").expanduser().resolve()
    log_path = ROOT / "metalblok/runs" / f"run-{stamp}.log"
    output_path = ROOT / "metalblok/runs" / f"run-{stamp}.txt"
    resume = state.exists()
    if args.continue_decode and not resume:
        raise Error("--continue-decode requires an existing --state checkpoint")
    if Path(f"{state}.partial").exists():
        raise Error(f"partial checkpoint needs inspection: {state}.partial")
    if resume:
        args.context, position = state_context(state)
        note(f"resuming conversation at position {position} from {state}")
    elif inputs + args.max_output_tokens - 1 > args.context:
        raise Error(f"run needs {inputs + args.max_output_tokens - 1} positions; increase --context")
    state.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    prompt_arg = (["--prompt-file", str(args.prompt_file)] if args.prompt_file
                  else ["-p", args.prompt])
    command = [
        str(BINARY), "-m", str(args.model), *prompt_arg,
        "-n", str(args.max_output_tokens), "--context", str(args.context),
        "--state", str(state), "--temperature", str(args.temperature),
        "--top-p", str(args.top_p), "--seed", str(args.seed),
        "--checkpoint-every", str(args.checkpoint_every),
    ]
    if resume:
        command.append("--continue-state" if args.continue_decode else "--resume-turn")
    if args.mla:
        command.append("--mla")
    env = os.environ.copy()
    if args.trace:
        env["METALBLOK_TRACE"] = "1"
    if args.profile_layers:
        env["METALBLOK_PROFILE_LAYERS"] = "1"
    if args.profile_ops:
        env["METALBLOK_PROFILE_LAYERS"] = "1"
        env["METALBLOK_PROFILE_OPS"] = "1"
    if args.validate_mla:
        if not args.mla:
            raise Error("--validate-mla requires --mla")
        env["METALBLOK_VALIDATE_MLA"] = "1"
    note(f"input={inputs} output_cap={args.max_output_tokens} context={args.context}")
    note(f"state={state}")
    note(f"diagnostics={log_path}")
    note(f"output={output_path}")

    with (Path("/tmp/metalblok-run.lock").open("a+") as lock,
          log_path.open("ab", buffering=0) as log,
          output_path.open("wb", buffering=0) as output):
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise Error("another MetalBlok run is active") from exc
        native_started = time.monotonic()
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=log, env=env)
        assert process.stdout is not None
        os.set_blocking(process.stdout.fileno(), False)
        first_output: float | None = None

        def drain_output() -> bool:
            nonlocal first_output
            changed = False
            while True:
                try:
                    chunk = os.read(process.stdout.fileno(), 65_536)
                except BlockingIOError:
                    break
                if not chunk:
                    break
                output.write(chunk)
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
                if first_output is None:
                    first_output = time.monotonic()
                changed = True
            return changed

        size = 0
        active = time.monotonic()
        try:
            while process.poll() is None:
                current = os.fstat(log.fileno()).st_size
                if current != size or drain_output():
                    size, active = current, time.monotonic()
                elif time.monotonic() - active > args.stall_timeout:
                    process.terminate()
                    raise Error(f"native runtime made no logged progress for {args.stall_timeout}s")
                time.sleep(0.25)
            drain_output()
        except KeyboardInterrupt:
            process.terminate()
            process.wait(10)
            raise
        if process.returncode:
            tail = log_path.read_text(errors="replace").splitlines()[-1:]
            raise Error(tail[0] if tail else f"native runtime exited {process.returncode}")
        finished = time.monotonic()
        timing = (
            f"[metalblok-wrapper] e2e_s={finished - e2e_started:.3f} "
            f"build_s={build_s:.3f} preflight_s={preflight_s:.3f} "
            f"tokenize_s={tokenize_s:.3f} startup_s={native_started - e2e_started:.3f} "
            f"ttft_s={(first_output or finished) - e2e_started:.3f} "
            f"native_s={finished - native_started:.3f}\n"
        )
        log.write(timing.encode())
    note(timing.removeprefix("[metalblok-wrapper] ").strip())
    note(f"complete; resume later with --state {state}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(run(arguments()))
    except KeyboardInterrupt:
        note("interrupted; the last complete checkpoint remains valid")
        raise SystemExit(130)
    except (Error, subprocess.TimeoutExpired) as exc:
        note(f"REFUSED: {exc}")
        raise SystemExit(2)
