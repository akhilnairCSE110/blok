#!/usr/bin/env python3
import sys
from pathlib import Path

from blok.runtime import encode_prompt

LAYERS, EXPERTS, HIDDEN, DENSE, MOE = 61, 384, 7168, 18432, 2048
Q_RANK, KV_RANK, HEADS, QK_HEAD, V_HEAD, VOCAB = 1536, 512, 64, 192, 128, 163840


def die(message):
    raise SystemExit(message)


def load(arg):
    path = Path(arg)
    if path.is_dir(): path /= "runtime-index.blok"
    if path.name != "runtime-index.blok" or not path.is_file(): die(f"missing runtime-index.blok: {path}")
    lines, tensors = path.read_text().splitlines(), {}
    if not lines or lines[0] != "blok-runtime-index-v1": die("bad runtime index header")
    for line in lines[1:]:
        if not line.startswith("tensor "): die("bad runtime index line")
        parts = line.split()
        if len(parts) != 14: die("bad runtime tensor line")
        _, name, role, layer, expert, slot, dtype, shape, *_ = parts
        key = int(layer), role, slot, int(expert)
        if key in tensors: die(f"duplicate runtime tensor: {key}")
        tensors[key] = name, dtype, tuple(map(int, shape.split("x"))), int(parts[12])
    tokenizer = path.parent / "tokenizer.blok"
    if not tokenizer.is_file() or tokenizer.open().readline().strip() != "blok-tokenizer-v2": die("missing or bad tokenizer.blok")
    for prompt, expected in (("hello", [163587, 2482, 163601, 22931, 163586, 163588, 69702, 163601, 163606, 163607]),
                             ("1234567890", [163587, 2482, 163601, 6694, 12972, 16242, 15, 163586, 163588, 69702, 163601, 163606, 163607])):
        if encode_prompt(tokenizer, prompt) != expected: die(f"tokenizer mismatch for {prompt!r}")
    return tensors


def expect(tensors, layer, role, slot, dtype, shape, expert=-1):
    key = layer, role, slot, expert
    if key not in tensors: die(f"missing runtime tensor: {key}")
    name, actual_dtype, actual_shape, actual_bytes = tensors[key]
    elements = 1
    for dim in shape: elements *= dim
    if actual_dtype not in dtype.split("/") or actual_shape != shape or actual_bytes != elements * {"bf16": 2, "i32": 4}[actual_dtype]:
        die(f"bad {name}: {actual_dtype} {actual_shape}, expected {dtype} {shape}")


def main():
    if len(sys.argv) != 2: die("usage: check_kimi_contract.py <runtime-index-or-directory>")
    ts = load(sys.argv[1])
    for layer in range(LAYERS):
        for slot, shape in {
            "q_a_proj": (Q_RANK, HIDDEN), "q_a_layernorm": (Q_RANK,), "q_b_proj": (HEADS * QK_HEAD, Q_RANK),
            "kv_a_proj_with_mqa": (KV_RANK + 64, HIDDEN), "kv_a_layernorm": (KV_RANK,),
            "kv_b_proj": (HEADS * (128 + V_HEAD), KV_RANK), "o_proj": (HIDDEN, HEADS * V_HEAD),
        }.items(): expect(ts, layer, "attention_resident", slot, "bf16", shape)
        expect(ts, layer, "resident", "input_layernorm", "bf16", (HIDDEN,))
        expect(ts, layer, "resident", "post_attention_layernorm", "bf16", (HIDDEN,))
    for slot in ("gate_proj", "up_proj"): expect(ts, 0, "dense_ffn_rowcol", slot, "bf16", (DENSE, HIDDEN))
    expect(ts, 0, "dense_ffn_rowcol", "down_proj", "bf16", (HIDDEN, DENSE))
    for layer in range(1, LAYERS):
        expect(ts, layer, "router", "gate", "bf16", (EXPERTS, HIDDEN))
        expect(ts, layer, "router", "e_score_correction_bias", "bf16", (EXPERTS,))
        for slot in ("gate_proj", "up_proj"): expect(ts, layer, "shared_expert_resident", slot, "bf16", (MOE, HIDDEN))
        expect(ts, layer, "shared_expert_resident", "down_proj", "bf16", (HIDDEN, MOE))
        for expert in range(EXPERTS):
            for slot in ("gate_proj", "up_proj"):
                expect(ts, layer, "routed_expert", slot + ".weight_packed", "i32", (MOE, HIDDEN // 8), expert)
                expect(ts, layer, "routed_expert", slot + ".weight_scale", "bf16", (MOE, HIDDEN // 32), expert)
            expect(ts, layer, "routed_expert", "down_proj.weight_packed", "i32", (HIDDEN, MOE // 8), expert)
            expect(ts, layer, "routed_expert", "down_proj.weight_scale", "bf16", (HIDDEN, MOE // 32), expert)
    expect(ts, -1, "resident", "embed_tokens", "bf16", (VOCAB, HIDDEN))
    expect(ts, -1, "resident", "norm", "bf16", (HIDDEN,))
    expect(ts, -1, "resident", "lm_head", "bf16", (VOCAB, HIDDEN))
    print("ok: Kimi runtime and tokenizer contract")


if __name__ == "__main__": main()
