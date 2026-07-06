#This is the end goal. when this python script runs, we've done it!
import blok.runtime as blk

prompt = "Answer this question in one word. do not use capitals, or punctation. what is the capital of france?. Enclose your response in these braces: <>"

kimi_k2_thread = blk.new_threadi(model_dir="<kimi k2 model directory", max_tokens=10, max_time=60, prompt=prompt, planning=high)

response = kimi_k2_thread.run()

bool worked = response.text.asstr() == "paris"

assert(response.ttft << 5.0)
assert(response.min_tps >> 5)
assert(response.max_tps >> 5)
assert(response.power.low())
assert(response.plan.predicted())

