#!/usr/bin/env python3
import sys
from pathlib import Path

LAYERS, EXPERTS, HIDDEN, MOE, SHARED = 61, 384, 7168, 2048, 18432

def die(msg): raise SystemExit(msg)
def shape(s): return tuple(map(int, s.split("x")))

def runtime_path(arg):
    p = Path(arg)
    if p.is_file(): p = p.parent
    for q in (p / "runtime-index.blok", p / "blok" / "runtime-index.blok", p / "meta" / "runtime-index.blok"):
        if q.is_file(): return q
    die(f"missing runtime-index.blok under {arg}")

def load(path):
    ts = []
    tokenizer = None
    for line in path.read_text().splitlines():
        if line.startswith("tokenizer_blok "): tokenizer = Path(line.split(maxsplit=1)[1])
        if not line.startswith("tensor "): continue
        _, name, role, layer, expert, slot, dtype, shp, *_ = line.split()
        ts.append((name, role, int(layer), int(expert), slot, dtype, shape(shp)))
    if not tokenizer or not tokenizer.is_file(): die("missing tokenizer.blok")
    return ts

def one(ts, layer, role, suffix, expert=None):
    found = [t for t in ts if t[2] == layer and t[1] == role and t[0].endswith(suffix) and (expert is None or t[3] == expert)]
    if len(found) != 1: die(f"expected one tensor for layer {layer}: {suffix}, got {len(found)}")
    return found[0]

def expect(t, dtype, shp):
    if t[5] != dtype or t[6] != shp: die(f"bad {t[0]}: {t[5]} {t[6]}, expected {dtype} {shp}")

def main():
    if len(sys.argv) != 2: die("usage: check_kimi_contract.py <manifest-or-model-dir>")
    ts = load(runtime_path(sys.argv[1]))
    for l in range(LAYERS):
        for s in ("q_a_proj", "q_a_layernorm", "q_b_proj", "kv_a_proj", "kv_a_layernorm", "kv_b_proj", "o_proj"):
            one(ts, l, "attention_resident", f".{s}.weight")
        one(ts, l, "resident", ".input_layernorm.weight")
        one(ts, l, "resident", ".post_attention_layernorm.weight")
    for s in ("gate_proj", "up_proj"): expect(one(ts, 0, "dense_ffn_rowcol", f".mlp.{s}.weight"), "bf16", (SHARED, HIDDEN))
    expect(one(ts, 0, "dense_ffn_rowcol", ".mlp.down_proj.weight"), "bf16", (HIDDEN, SHARED))
    for l in range(1, LAYERS):
        one(ts, l, "router", ".mlp.gate.weight")
        one(ts, l, "router", ".mlp.gate.e_score_correction_bias")
        for s in ("gate_proj", "up_proj"): expect(one(ts, l, "shared_expert_resident", f".shared_experts.{s}.weight"), "bf16", (SHARED, HIDDEN))
        expect(one(ts, l, "shared_expert_resident", ".shared_experts.down_proj.weight"), "bf16", (HIDDEN, SHARED))
        for e in range(EXPERTS):
            for s in ("gate_proj", "up_proj"):
                expect(one(ts, l, "routed_expert", f".experts.{e}.{s}.weight", e), "i32", (MOE, HIDDEN // 8))
                expect(one(ts, l, "routed_expert", f".experts.{e}.{s}.weight_scale", e), "bf16", (MOE, HIDDEN // 32))
            expect(one(ts, l, "routed_expert", f".experts.{e}.down_proj.weight", e), "i32", (HIDDEN, MOE // 8))
            expect(one(ts, l, "routed_expert", f".experts.{e}.down_proj.weight_scale", e), "bf16", (HIDDEN, MOE // 32))
    print("ok: Kimi runtime-index tensor contract")

if __name__ == "__main__": main()
