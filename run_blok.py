#!/usr/bin/env python3
"""Safe one-command runner for the local MetalBlok DeepSeek-R1 checkpoint."""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import os
import pathlib
import platform
import re
import struct
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parent
BINARY = ROOT / "metalblok" / "build" / "metalblok"
DEFAULT_MODEL = pathlib.Path(
    "/Users/akhil/Desktop/AKHIL_HOME/ai/blade/hf_models/DeepSeek-R1-671B/"
    "DeepSeek-R1-UD-IQ1_S/DeepSeek-R1-UD-IQ1_S-00001-of-00003.gguf"
)
STATE_HEADER = struct.Struct("<8s7I")
VERIFIED_CONTEXT = 64
DEFAULT_OUTPUT_TOKENS = 8
MAX_VERIFIED_OUTPUT_TOKENS = 32
EOS_TOKEN = 1


class RunError(RuntimeError):
    pass


def status(message: str) -> None:
    print(f"[MetalBlok] {message}", file=sys.stderr, flush=True)


def read_state(path: pathlib.Path) -> tuple[int, int, int]:
    try:
        with path.open("rb") as handle:
            raw = handle.read(STATE_HEADER.size)
    except OSError as exc:
        raise RunError(f"cannot read checkpoint {path}: {exc}") from exc
    if len(raw) != STATE_HEADER.size:
        raise RunError(f"checkpoint has a short header: {len(raw)} bytes")
    magic, version, layers, max_seq, kv_rank, rope_dim, pos, next_token = (
        STATE_HEADER.unpack(raw)
    )
    if magic != b"MBLKSTAT" or version != 2:
        raise RunError("checkpoint magic/version is not MetalBlok state v2")
    if (layers, max_seq, kv_rank, rope_dim) != (61, VERIFIED_CONTEXT, 512, 64):
        raise RunError(
            "checkpoint architecture mismatch: "
            f"layers={layers} context={max_seq} kv_rank={kv_rank} rope_dim={rope_dim}"
        )
    if pos > max_seq or next_token >= 129280:
        raise RunError(f"checkpoint values are invalid: pos={pos} next={next_token}")
    return pos, next_token, max_seq


def remove_wrapper_newline(payload: bytes) -> bytes:
    return payload[:-1] if payload.endswith(b"\n") else payload


def run_logged(
    command: list[str],
    log,
    phase: str,
    timeout: int,
) -> bytes:
    stamp = dt.datetime.now(dt.timezone.utc).isoformat()
    log.write(f"\n=== {stamp} {phase} ===\n".encode())
    log.flush()
    os.fsync(log.fileno())
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        if exc.stdout:
            log.write(exc.stdout)
        if exc.stderr:
            log.write(exc.stderr)
        log.write(f"\nFAIL timeout after {timeout}s\n".encode())
        log.flush()
        os.fsync(log.fileno())
        raise RunError(
            f"{phase} exceeded {timeout}s; the last committed checkpoint is still valid"
        ) from exc
    log.write(result.stderr)
    log.write(result.stdout)
    log.write(f"\nexit={result.returncode}\n".encode())
    log.flush()
    os.fsync(log.fileno())
    if result.returncode != 0:
        tail = result.stderr.decode("utf-8", errors="replace").strip().splitlines()
        detail = tail[-1] if tail else "no diagnostic"
        raise RunError(f"{phase} failed with exit {result.returncode}: {detail}")
    return result.stdout


def ensure_build(log) -> None:
    if BINARY.is_file() and os.access(BINARY, os.X_OK):
        return
    status("native binary is missing; configuring the local build")
    run_logged(
        ["cmake", "-S", str(ROOT / "metalblok"), "-B", str(ROOT / "metalblok" / "build")],
        log,
        "cmake configure",
        120,
    )
    run_logged(
        ["cmake", "--build", str(ROOT / "metalblok" / "build"), "-j", "8"],
        log,
        "cmake build",
        300,
    )


def preflight(model: pathlib.Path, log) -> None:
    output = run_logged(
        [str(BINARY), "--preflight", str(model)], log, "model preflight", 30
    ).decode("utf-8", errors="replace")
    first = output.splitlines()[0] if output else ""
    if "all_resident=true" not in first:
        raise RunError("one or more GGUF shards are missing, sparse, or dataless")
    if "manifest=deepseek-r1-ud-iq1_s" not in first or "shards=3" not in first:
        raise RunError("model is not the verified three-shard DeepSeek-R1 UD-IQ1_S manifest")


def tokenize(prompt: str, model: pathlib.Path, log) -> list[int]:
    output = run_logged(
        [str(BINARY), "-m", str(model), "-p", prompt, "--tokenize-only"],
        log,
        "tokenize",
        30,
    ).decode("utf-8", errors="replace")
    match = re.search(r"^token_ids=([0-9,]+)$", output, flags=re.MULTILINE)
    if not match:
        raise RunError("native tokenizer did not return a token ID list")
    return [int(item) for item in match.group(1).split(",")]


def parse_step(output: bytes) -> tuple[int, int, int, bytes]:
    match = re.search(
        rb"state_pos=(\d+) input_token=(\d+) next_token=(\d+) piece=", output
    )
    if not match:
        raise RunError("native single-step output is malformed")
    piece = remove_wrapper_newline(output[match.end() :])
    return int(match[1]), int(match[2]), int(match[3]), piece


def consume_token(
    model: pathlib.Path,
    state: pathlib.Path,
    token: int,
    expected_pos: int,
    log,
    timeout: int,
    phase: str,
) -> tuple[int, bytes]:
    partial = pathlib.Path(str(state) + ".partial")
    if partial.exists():
        raise RunError(f"refusing to proceed while a partial checkpoint exists: {partial}")
    output = run_logged(
        [
            str(BINARY),
            "-m", str(model),
            "-p", "x",
            "--context", str(VERIFIED_CONTEXT),
            "--state", str(state),
            "--single-step-token", str(token),
        ],
        log,
        phase,
        timeout,
    )
    reported_pos, reported_input, reported_next, piece = parse_step(output)
    state_pos, state_next, _ = read_state(state)
    if reported_input != token:
        raise RunError(f"native step consumed token {reported_input}, expected {token}")
    if reported_pos != expected_pos or state_pos != expected_pos:
        raise RunError(
            f"checkpoint position invariant failed: reported={reported_pos} "
            f"stored={state_pos} expected={expected_pos}"
        )
    if reported_next != state_next:
        raise RunError(
            f"checkpoint token invariant failed: reported={reported_next} stored={state_next}"
        )
    if partial.exists():
        raise RunError(f"partial checkpoint remained after a successful step: {partial}")
    return state_next, piece


def default_run_paths() -> tuple[pathlib.Path, pathlib.Path]:
    run_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}"
    return (
        pathlib.Path("/tmp") / f"metalblok-{run_id}.state",
        ROOT / "metalblok" / "runs" / f"run-{run_id}.log",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the local 671B DeepSeek-R1 checkpoint safely on Apple Metal."
    )
    parser.add_argument("prompt", help="user prompt; DeepSeek chat markers are added automatically")
    parser.add_argument(
        "-n", "--max-output-tokens", type=int, default=DEFAULT_OUTPUT_TOKENS,
        help=f"maximum generated tokens (default {DEFAULT_OUTPUT_TOKENS}, verified maximum {MAX_VERIFIED_OUTPUT_TOKENS})",
    )
    parser.add_argument(
        "--model", type=pathlib.Path,
        default=pathlib.Path(os.environ.get("METALBLOK_MODEL", DEFAULT_MODEL)),
        help="first GGUF shard; defaults to the verified local checkpoint",
    )
    parser.add_argument(
        "--state", type=pathlib.Path,
        help="new checkpoint path; must not already exist (default: unique file in /tmp)",
    )
    parser.add_argument(
        "--log", type=pathlib.Path,
        help="native diagnostic log (default: unique file under metalblok/runs)",
    )
    parser.add_argument(
        "--timeout-per-token", type=int, default=60,
        help="kill a stuck token process while preserving the prior checkpoint (default 60s)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RunError("this runner is verified only on Apple Silicon macOS")
    if not args.prompt:
        raise RunError("prompt must not be empty")
    if not 1 <= args.max_output_tokens <= MAX_VERIFIED_OUTPUT_TOKENS:
        raise RunError(
            f"--max-output-tokens must be between 1 and {MAX_VERIFIED_OUTPUT_TOKENS}"
        )
    if not 15 <= args.timeout_per_token <= 300:
        raise RunError("--timeout-per-token must be between 15 and 300 seconds")
    if not args.model.is_file():
        raise RunError(f"model shard does not exist: {args.model}")

    default_state, default_log = default_run_paths()
    state = (args.state or default_state).expanduser().resolve()
    log_path = (args.log or default_log).expanduser().resolve()
    if state.exists() or pathlib.Path(str(state) + ".partial").exists():
        raise RunError(f"state path already exists; choose a new path: {state}")
    state.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # A second 671B process would defeat the memory ledger even if each process
    # is safe independently.  All users of this convenience runner serialize.
    lock_path = pathlib.Path("/tmp/metalblok-run.lock")
    lock = lock_path.open("a+")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RunError("another run_blok.py process is already active") from exc

    with log_path.open("ab", buffering=0) as log:
        ensure_build(log)
        status("checking that all three 140.23 GB model shards are physically resident")
        preflight(args.model, log)
        token_ids = tokenize(args.prompt, args.model, log)
        available_output = VERIFIED_CONTEXT - len(token_ids) + 1
        if len(token_ids) > VERIFIED_CONTEXT:
            raise RunError(
                f"formatted prompt is {len(token_ids)} tokens; verified context is {VERIFIED_CONTEXT}"
            )
        if args.max_output_tokens > available_output:
            raise RunError(
                f"prompt uses {len(token_ids)}/{VERIFIED_CONTEXT} context positions; "
                f"at most {available_output} output tokens are safe"
            )

        status(
            f"prompt={len(token_ids)} tokens, output cap={args.max_output_tokens}, "
            f"checkpoint={state}"
        )
        status(f"native diagnostics={log_path}")

        next_token = 0
        for index, token in enumerate(token_ids):
            status(f"prefill {index + 1}/{len(token_ids)}")
            next_token, _ = consume_token(
                args.model,
                state,
                token,
                index + 1,
                log,
                args.timeout_per_token,
                f"prefill {index + 1}/{len(token_ids)}",
            )

        status("generation started")
        if next_token == EOS_TOKEN:
            status("model returned EOS immediately")
            return 0

        # Decode the prediction already committed by the final prompt step.
        first = run_logged(
            [
                str(BINARY), "-m", str(args.model), "-p", "x",
                "--context", str(VERIFIED_CONTEXT), "--state", str(state),
                "--continue-state", "-n", "1",
            ],
            log,
            "emit first prediction",
            30,
        )
        sys.stdout.buffer.write(remove_wrapper_newline(first))
        sys.stdout.buffer.flush()
        produced = 1

        while produced < args.max_output_tokens:
            before_pos, current_token, _ = read_state(state)
            if current_token == EOS_TOKEN:
                break
            next_token, piece = consume_token(
                args.model,
                state,
                current_token,
                before_pos + 1,
                log,
                args.timeout_per_token,
                f"decode {produced + 1}/{args.max_output_tokens}",
            )
            if next_token == EOS_TOKEN:
                break
            sys.stdout.buffer.write(piece)
            sys.stdout.buffer.flush()
            produced += 1

        sys.stdout.buffer.write(b"\n")
        sys.stdout.buffer.flush()
        status(f"complete: generated {produced} token(s); checkpoint retained at {state}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[MetalBlok] interrupted; the last committed checkpoint remains valid", file=sys.stderr)
        sys.exit(130)
    except RunError as exc:
        print(f"[MetalBlok] REFUSED: {exc}", file=sys.stderr)
        sys.exit(2)
