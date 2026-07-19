#!/usr/bin/env python3
import sys
from pathlib import Path

from blok.runtime import encode_kimi_prompt

FIXTURES = {
    "hello": [163587, 2482, 163601, 22931, 163586, 163588, 69702, 163601, 163606, 163607],
    "1234567890": [163587, 2482, 163601, 6694, 12972, 16242, 15, 163586, 163588, 69702, 163601, 163606, 163607],
}


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: check_kimi_tokenizer.py <tokenizer.blok>")
    path = Path(sys.argv[1])
    for prompt, expected in FIXTURES.items():
        actual = encode_kimi_prompt(path, prompt)
        if actual != expected:
            raise SystemExit(f"tokenizer mismatch for {prompt!r}: {actual} != {expected}")
    print("ok: Kimi tokenizer and single-user chat template")


if __name__ == "__main__":
    main()
