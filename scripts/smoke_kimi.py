#!/usr/bin/env python3
import json
import os
import sys

import blok.runtime as blk

PROMPT = (
    "Answer this question in one word. Do not use capitals or punctuation. "
    "What is the capital of France? Enclose your response in these braces: <>"
)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: smoke_kimi.py <manifest-or-model-directory>")
    response = blk.new_thread(
        model_dir=sys.argv[1],
        max_tokens=10,
        max_time=float(os.getenv("BLOK_SMOKE_MAX_TIME", "86400")),
        prompt=PROMPT,
    ).run()
    if response.text.asstr() != "paris":
        raise SystemExit(f"quality smoke failed: {response.text.asstr()!r}")
    print(json.dumps({"status": "ok", "text": response.text.asstr()}))


if __name__ == "__main__":
    main()
