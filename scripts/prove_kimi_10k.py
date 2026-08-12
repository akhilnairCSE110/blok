#!/usr/bin/env python3
import argparse
import ast
import json
import os
import re
import time
from pathlib import Path

import blok.runtime as blk

BASE_PROMPT = """Generate one substantial, self-contained Python 3.12 program.

The program must be a dependency-free command-line JSONL source analyzer. It must read UTF-8 JSON objects from stdin, validate each record, compute useful deterministic text and numeric statistics, group results, and emit canonical JSON. Include dataclasses, type hints, clear errors, streaming operation, an argparse CLI, and a comprehensive unittest suite reachable through a --self-test option. Handle malformed JSON, missing fields, Unicode, empty input, large records, and broken pipes professionally. Do not access the network, invoke subprocesses, or write outside a user-selected output path.

Return exactly one fenced ```python block and no prose. The Python source inside it must be between 9,500 and 9,800 model tokens, must parse as Python 3, and must end with the literal line '# BLOK_PROOF_COMPLETE'. Do not emit the end-of-message token until the complete marker has been written.

The following appendix exists only to make this request a long-context test. Ignore its repeated notes when designing the program.
"""
FILLER = "Appendix note {number:04d}: preserve the program contract above; this deterministic sentence carries no new requirement."
PADDING = (" context", " appendix", " note", " preserve", " program", " x", " 0", "\n", ".", " #")


def sized_prompt(tokenizer: Path, target: int) -> str:
    encoding = blk.tokenizer_encoding(tokenizer)
    count = lambda text: len(blk.encode_prompt(tokenizer, text, encoding))
    prompt, tokens, number = BASE_PROMPT, count(BASE_PROMPT), 1
    if tokens > target:
        raise SystemExit(f"base proof prompt already uses {tokens} tokens, above target {target}")
    while True:
        candidate = prompt + "\n" + FILLER.format(number=number)
        candidate_tokens = count(candidate)
        if candidate_tokens > target:
            break
        prompt, tokens, number = candidate, candidate_tokens, number + 1
    while tokens < target:
        choices = []
        for fragment in PADDING:
            candidate = prompt + fragment
            candidate_tokens = count(candidate)
            if tokens < candidate_tokens <= target:
                choices.append((candidate_tokens, candidate))
        if not choices:
            raise SystemExit(f"could not construct an exact {target}-token Kimi prompt; stopped at {tokens}")
        tokens, prompt = max(choices, key=lambda item: item[0])
    return prompt


def python_source(text: str) -> str:
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if len(blocks) != 1:
        raise SystemExit(f"proof expected exactly one Python fence, found {len(blocks)}")
    source = blocks[0].rstrip() + "\n"
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise SystemExit(f"generated program is not valid Python: {error}") from error
    if not source.rstrip().endswith("# BLOK_PROOF_COMPLETE"):
        raise SystemExit("generated program is missing the completion marker")
    if not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) for node in ast.walk(tree)):
        raise SystemExit("generated program contains no functions")
    return source


def main() -> None:
    parser = argparse.ArgumentParser(description="Prove a 10k-token Kimi input and near-10k-token Python generation")
    parser.add_argument("model", help="runtime-index.blok or its metadata directory")
    parser.add_argument("--output", default="runs/kimi-10k-proof.py")
    parser.add_argument("--input-tokens", type=int, default=10_000)
    parser.add_argument("--output-tokens", type=int, default=10_000)
    parser.add_argument("--minimum-output-tokens", type=int, default=9_500)
    parser.add_argument("--max-time", type=float, default=float(os.getenv("BLOK_PROOF_MAX_TIME", "2592000")))
    args = parser.parse_args()
    if not 0 < args.minimum_output_tokens <= args.output_tokens:
        raise SystemExit("minimum output tokens must be positive and no larger than the output limit")
    index = blk.runtime_index(args.model)
    output = Path(args.output).expanduser().resolve()
    metadata = output.with_suffix(output.suffix + ".json")
    if output.exists() or metadata.exists():
        raise SystemExit(f"refusing to overwrite proof output: {output} or {metadata}")
    prompt = sized_prompt(index.parent / "tokenizer.blok", args.input_tokens)
    required = blk.required_kv_bytes(args.input_tokens, args.output_tokens)
    print(json.dumps({"status": "starting", "input_tokens": args.input_tokens, "output_limit": args.output_tokens,
                      "required_kv_bytes": required}), flush=True)
    started = time.monotonic()
    result = blk.generate_report(model_dir=index, prompt=prompt, max_tokens=args.output_tokens, max_time=args.max_time)
    elapsed = time.monotonic() - started
    if result.input_tokens != args.input_tokens:
        raise SystemExit(f"input token mismatch: {result.input_tokens} != {args.input_tokens}")
    if result.output_tokens < args.minimum_output_tokens or result.finish_reason != "eos":
        raise SystemExit(f"incomplete generation: {result.output_tokens} tokens, finish_reason={result.finish_reason}")
    source = python_source(result.text)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(source)
    report = {"status": "ok", "input_tokens": result.input_tokens, "output_tokens": result.output_tokens,
              "finish_reason": result.finish_reason, "elapsed_seconds": elapsed, "program": str(output),
              "executed": False}
    metadata.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
