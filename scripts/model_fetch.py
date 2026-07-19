#!/usr/bin/env python3
import base64, json, os, re, struct, subprocess, sys
from pathlib import Path

MODELS = {"kimi-k2.6": ("moonshotai/Kimi-K2.6", "7eb5002f6aadc958aed6a9177b7ed26bb94011bb", 595148192736, 64)}
UP, DOWN = ("up_proj", "gate_proj", "w1", "w3"), ("down_proj", "w2")
DTYPE = {"BF16": "bf16", "I32": "i32"}

def die(msg=None):
    if msg: print(msg, file=sys.stderr)
    print(f"usage: {Path(sys.argv[0]).name} kimi-k2.6 {{status|fetch|materialize}}", file=sys.stderr)
    raise SystemExit(64 if msg is None else 1)

def js(x): return json.dumps(x, separators=(",", ":"))
def align(n, a): return ((n + a - 1) // a) * a
def align_down(n, a): return n // a * a

def ctx(model):
    if model not in MODELS: die(f"unknown model: {model}")
    repo, rev, size, shards = MODELS[model]
    home = Path(os.getenv("BLOK_HOME", Path.home() / ".blok"))
    local_hf = Path(__file__).parents[1] / ".venv/bin/hf"
    root = Path(os.getenv("BLOK_MODEL_ROOT", home / "models")) / repo
    meta = Path(os.getenv("BLOK_META_ROOT", home / "metadata")) / repo
    return {"model": model, "repo": repo, "revision": rev, "expected_bytes": size, "expected_safetensors": shards,
            "local_dir": root / "source/hf" / rev, "meta_dir": meta, "cache_dir": home / "hf-cache",
            "hf": os.getenv("BLOK_HF_BIN", str(local_hf) if local_hf.is_file() else "hf")}

def files(c):
    root = c["local_dir"]
    if not root.is_dir(): return []
    return [p for p in root.rglob("*") if p.is_file() and ".cache" not in p.relative_to(root).parts]

def status(c):
    fs = files(c)
    got = sum(p.stat().st_size for p in fs)
    shards = sorted(p for p in fs if re.fullmatch(r"model-\d{5}-of-\d{6}\.safetensors", p.name))
    expected = [f"model-{i:05d}-of-{c['expected_safetensors']:06d}.safetensors" for i in range(1, c["expected_safetensors"] + 1)]
    index = c["local_dir"] / "model.safetensors.index.json"
    indexed_bytes = json.loads(index.read_text()).get("metadata", {}).get("total_size") if index.is_file() else None
    return {k: str(v) if k.endswith("_dir") else v for k, v in c.items() if k not in ("hf", "meta_dir")} | {
        "downloaded_bytes": got, "indexed_weight_bytes": indexed_bytes, "safetensors": len(shards),
        "complete": [p.name for p in shards] == expected and indexed_bytes == c["expected_bytes"],
    }

def hf_env(c):
    env = os.environ.copy()
    token = env.pop("BLOK_HF_TOKEN", "")
    env.pop("HF_TOKEN", None); env.pop("HUGGING_FACE_HUB_TOKEN", None)
    env.update(HF_HOME=str(c["cache_dir"]), HF_HUB_CACHE=str(c["cache_dir"] / "hub"), HF_XET_CACHE=str(c["cache_dir"] / "xet"),
               HF_XET_HIGH_PERFORMANCE="1", HF_HUB_DOWNLOAD_TIMEOUT=env.get("HF_HUB_DOWNLOAD_TIMEOUT", "60"))
    return env, token.strip()

def fetch(c):
    c["local_dir"].mkdir(parents=True, exist_ok=True); c["cache_dir"].mkdir(parents=True, exist_ok=True)
    env, token = hf_env(c)
    cmd = [c["hf"], "download", c["repo"], "--revision", c["revision"], "--local-dir", str(c["local_dir"]),
           "--max-workers", os.getenv("BLOK_HF_WORKERS", "8")]
    if token: cmd += ["--token", token]
    rc = subprocess.run(cmd, env=env).returncode
    if rc: raise SystemExit(rc)

def role(name):
    if name.endswith(".mlp.gate.weight") or name.endswith(".mlp.gate.e_score_correction_bias"):
        return "router"
    if "shared_experts" in name: return "shared_expert_resident"
    if re.search(r"\.experts?\.\d+\.", name): return "routed_expert"
    if ".self_attn." in name: return "attention_resident"
    if any(x in name for x in UP + DOWN): return "dense_ffn_rowcol"
    return "resident"

def runtime_slot(name):
    m = re.search(r"\.layers\.(\d+)\.", name)
    layer = int(m.group(1)) if m else -1
    m = re.search(r"\.experts?\.(\d+)\.", name)
    expert = int(m.group(1)) if m else -1
    suffix = name.rsplit(".", 1)[-1]
    leaf = name.rsplit(".", 2)[-2] if suffix == "weight" else suffix
    if suffix.startswith("weight_"): leaf = name.rsplit(".", 2)[-2] + "." + suffix
    return layer, expert, leaf

def tokenizer(c):
    vocab_file = c["local_dir"] / "tiktoken.model"
    config_file = c["local_dir"] / "tokenizer_config.json"
    if not vocab_file.is_file() or not config_file.is_file():
        raise SystemExit("pinned checkpoint requires tiktoken.model and tokenizer_config.json")
    tokens = {}
    for line in vocab_file.read_bytes().splitlines():
        encoded, rank = line.split()
        tokens[int(rank)] = base64.b64decode(encoded)
    added = json.loads(config_file.read_text()).get("added_tokens_decoder", {})
    special = {int(i): meta["content"].encode() for i, meta in added.items()}
    if sorted(tokens) != list(range(163584)) or sorted(special) != list(range(163584, 163840)):
        raise SystemExit("unexpected Kimi tokenizer rank contract")
    rows = [f"tok {i} {value.hex()}" for i, value in sorted(tokens.items())]
    rows += [f"special {i} {value.hex()}" for i, value in sorted(special.items())]
    return "blok-tokenizer-v2\n" + "\n".join(rows) + "\n"

def tensors(c):
    out = []
    for path in sorted(c["local_dir"].glob("*.safetensors")):
        with path.open("rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]; header = json.loads(f.read(n)); base = 8 + n
        for name, meta in header.items():
            if name == "__metadata__": continue
            if not name.startswith("language_model."): continue
            if name.endswith(".weight_shape"): continue
            if meta["dtype"] not in DTYPE: continue
            start, end = meta["data_offsets"]
            out.append((name, role(name), DTYPE[meta["dtype"]], "x".join(map(str, meta["shape"])), str(path), base + start, end - start))
    return out

def materialize(c):
    s = status(c)
    if not s["complete"]: raise SystemExit(f"incomplete download: {s['safetensors']}/{s['expected_safetensors']} shards")
    if any(x.isspace() for x in str(c["local_dir"])): raise SystemExit("model path cannot contain whitespace")
    c["meta_dir"].mkdir(parents=True, exist_ok=True)
    runtime = ["blok-runtime-index-v1"]
    tb = c["meta_dir"] / "tokenizer.blok"
    tb.write_text(tokenizer(c))
    alignment = 4096
    for name, r, dtype, shape, file, start, size in tensors(c):
        off, end = align_down(start, alignment), align(start + size, alignment)
        layer, expert, slot = runtime_slot(name)
        runtime.append(f"tensor {name} {r} {layer} {expert} {slot} {dtype} {shape} {off} {end - off} {alignment} {start} {size} {file}")
    (c["meta_dir"] / "runtime-index.blok").write_text("\n".join(runtime) + "\n")
    return {"model": c["model"], "tensors": len(runtime) - 1, "index": str(c["meta_dir"] / "runtime-index.blok")}

def main():
    if len(sys.argv) != 3: die()
    c, mode = ctx(sys.argv[1]), sys.argv[2]
    if mode == "status": print(js(status(c)))
    elif mode == "fetch": fetch(c); print(js(status(c)))
    elif mode == "materialize": print(js(materialize(c)))
    else: die(f"unknown mode: {mode}")

if __name__ == "__main__": main()
