#!/usr/bin/env python3
import blok.runtime as blk


PROMPT = (
    "Answer this question in one word. Do not use capitals or punctuation. "
    "What is the capital of France? Enclose your response in these braces: <>"
)


def main() -> None:
    text = blk.generate(
        model_dir="<kimi k2 model directory>",
        max_tokens=10,
        max_time=86400,
        prompt=PROMPT,
    )
    assert blk.answer(text) == "paris", text


if __name__ == "__main__":
    main()
