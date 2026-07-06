#!/usr/bin/env python3
import blok.runtime as blk


PROMPT = (
    "Answer this question in one word. Do not use capitals or punctuation. "
    "What is the capital of France? Enclose your response in these braces: <>"
)


def main() -> None:
    kimi_k2_thread = blk.new_threadi(
        model_dir="<kimi k2 model directory>",
        max_tokens=10,
        max_time=60,
        prompt=PROMPT,
        planning=blk.high,
    )
    response = kimi_k2_thread.run()
    assert response.text.asstr() == "paris", response.text.asstr()
    assert response.ttft < 5.0, response.ttft
    assert response.min_tps > 5.0, response.min_tps
    assert response.max_tps > 5.0, response.max_tps
    assert response.power.low()
    assert response.plan.predicted()


if __name__ == "__main__":
    main()
