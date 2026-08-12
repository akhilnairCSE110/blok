#!/usr/bin/env python3
"""Run the pinned DeepSeek checkpoint through exactly 1,000 input/output tokens."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import struct
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "metalblok/build/metalblok"
RUNNER = ROOT / "run_blok.py"
MODEL = Path(
    "/Users/akhil/Desktop/AKHIL_HOME/ai/blade/hf_models/DeepSeek-R1-671B/"
    "DeepSeek-R1-UD-IQ1_S/DeepSeek-R1-UD-IQ1_S-00001-of-00003.gguf"
)
HEADER = struct.Struct("<8s7I")
TARGET = 1_000


def token_count(model: Path, prompt: str) -> int:
    result = subprocess.run(
        [str(BINARY), "-m", str(model), "-p", prompt, "--tokenize-only"],
        check=True, capture_output=True, text=True,
    )
    match = re.search(r"^token_ids=([0-9,]+)$", result.stdout, re.MULTILINE)
    if not match:
        raise RuntimeError("native tokenizer emitted no token IDs")
    return match.group(1).count(",") + 1


def exact_prompt(model: Path) -> str:
    prefix = """Write a complete, production-quality Python 3 program named jsonl_report.py.
It must use only the standard library, stream an arbitrarily large JSON Lines input file without loading it into memory, reject malformed records with line-numbered diagnostics, group valid records by a required string field named category, and compute count, sum, minimum, maximum, and numerically stable mean for a required finite numeric field named value. Add argparse options for input path, output path, and strict mode. Emit deterministic UTF-8 JSON with sorted categories and keys. Use compensated summation, explicit type checks that reject booleans as numbers, atomic output replacement, useful exit codes, type hints, docstrings, and a main guard. Include self-contained unittest cases runnable with python -m unittest, covering empty input, malformed JSON, missing fields, boolean values, non-finite values, strict mode, Unicode, deterministic ordering, and a successful multi-category file. Avoid third-party packages and network access.

Additional review constraints follow. Each repeated constraint is intentional and remains binding.
"""
    constraint = (
        "Constraint: preserve streaming memory bounds, deterministic behavior, clear errors, "
        "portable standard-library semantics, and directly test every failure branch.\n"
    )
    suffix = """
Return one concise explanation followed by one complete Python code block. Do not omit tests, use placeholders, or claim behavior the code does not implement.
"""
    prompt = prefix
    while token_count(model, prompt + constraint + suffix) <= TARGET:
        prompt += constraint

    atoms = (" reliable", " typed", " tested", " clear", " safe", " exact", ".", "\n")
    while (current := token_count(model, prompt + suffix)) < TARGET:
        choices = [
            (token_count(model, prompt + atom + suffix), atom)
            for atom in atoms
        ]
        choices = [choice for choice in choices if current < choice[0] <= TARGET]
        if not choices:
            raise RuntimeError(f"could not pad prompt exactly from {current} tokens")
        _, atom = max(choices)
        prompt += atom
    prompt += suffix
    actual = token_count(model, prompt)
    if actual != TARGET:
        raise RuntimeError(f"constructed {actual}, not {TARGET}, input tokens")
    return prompt


def state_position(path: Path) -> int:
    with path.open("rb") as stream:
        magic, version, layers, context, rank, rope, pos, token = HEADER.unpack(
            stream.read(HEADER.size)
        )
    if (magic, version, layers, context, rank, rope) != (
        b"MBLKSTAT", 3, 61, 2_048, 512, 64,
    ) or token >= 129_280:
        raise RuntimeError("proof checkpoint header is invalid")
    return pos


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume-state", type=Path,
                        help="resume an exact checkpoint at position 1000..1998")
    args = parser.parse_args()
    if not BINARY.is_file():
        subprocess.run(["cmake", "-S", str(ROOT / "metalblok"), "-B", str(BINARY.parent)], check=True)
        subprocess.run(["cmake", "--build", str(BINARY.parent), "-j", "8"], check=True)

    prompt = exact_prompt(args.model)
    print(f"proof input: {token_count(args.model, prompt)} native tokens", file=sys.stderr)
    if args.dry_run:
        return 0

    stamp = time.strftime("%Y%m%d-%H%M%S")
    state = args.resume_state or ROOT / "metalblok/runs" / f"proof-1k-{stamp}.state"
    position = state_position(state) if args.resume_state else TARGET
    if args.resume_state and not TARGET <= position < 2 * TARGET - 1:
        raise RuntimeError("resume checkpoint position must be in [1000,1998]")
    output_tokens = 2 * TARGET - position if args.resume_state else TARGET
    output = ROOT / "metalblok/runs" / f"proof-1k-{stamp}.txt"
    command = [
        str(RUNNER), prompt, "-n", str(output_tokens), "--context", "2048",
        "--state", str(state), "--checkpoint-every", "256", "--temperature", "0",
    ]
    if args.resume_state:
        command.append("--continue-decode")
    with output.open("w", encoding="utf-8") as saved:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, text=True)
        assert process.stdout is not None
        while chunk := process.stdout.read(1):
            sys.stdout.write(chunk)
            sys.stdout.flush()
            saved.write(chunk)
        status = process.wait()
    if status:
        return status
    position = state_position(state)
    expected = 2 * TARGET - 1
    if position != expected:
        raise RuntimeError(
            f"proof stopped at state position {position}; exact 1k+1k requires {expected}"
        )
    print(
        f"proof passed: input={TARGET} emitted={TARGET} committed={position} "
        f"state={state} output={output}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
