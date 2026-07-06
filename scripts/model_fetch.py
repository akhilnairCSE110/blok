#!/usr/bin/env python3
import json, os, re, shutil, struct, subprocess, sys
from pathlib import Path

MODELS = {"kimi-k2.6": ("moonshotai/Kimi-K2.6", "7eb5002f6aadc958aed6a9177b7ed26bb94011bb", 595421860056, 64)}
ATTN = ("q_proj", "k_proj", "v_proj", "o_proj", "q_a_proj", "q_b_proj", "kv_a_proj", "kv_b_proj")
UP, DOWN = ("up_proj", "gate_proj", "w1", "w3"), ("down_proj", "w2")
DTYPE = {"BF16": "bf16", "F16": "f16", "F32": "f32", "I8": "i8", "U8": "u8"}

def die(msg=None):
    if msg: print(msg, file=sys.stderr)
    print(f"usage: {Path(sys.argv[0]).name} kimi-k2.6 {{status|fetch|detach|layout|materialize}}", file=sys.stderr)
    raise SystemExit(64 if msg is None else 1)

def js(x): return json.dumps(x, separators=(",", ":"))
def align(n, a): return ((n + a - 1) // a) * a
def align_down(n, a): return n // a * a

def ctx(model):
    if model not in MODELS: die(f"unknown model: {model}")
    repo, rev, size, shards = MODELS[model]
    home = Path(os.getenv("BLOK_HOME", Path(__file__).resolve().parents[3] / ".blok"))
    root = Path(os.getenv("BLOK_MODEL_ROOT", home / "models")) / repo
    return {"model": model, "repo": repo, "revision": rev, "expected_bytes": size, "expected_safetensors": shards,
            "local_dir": root / "source/hf" / rev, "meta_dir": root / "blok", "cache_dir": home / "hf-cache",
            "hf": os.getenv("BLOK_HF_BIN", "hf")}

def files(c):
    root = c["local_dir"]
    if not root.is_dir(): return []
    return [p for p in root.rglob("*") if p.is_file() and ".cache" not in p.relative_to(root).parts]

def status(c):
    fs = files(c)
    got = sum(p.stat().st_size for p in fs)
    shards = sum(p.name.endswith(".safetensors") for p in fs)
    return {k: str(v) if k.endswith("_dir") else v for k, v in c.items() if k not in ("hf", "meta_dir")} | {
        "downloaded_bytes": got, "safetensors": shards,
        "complete": got >= c["expected_bytes"] and shards >= c["expected_safetensors"],
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
    c["meta_dir"].mkdir(parents=True, exist_ok=True)
    (c["meta_dir"] / "fetch-status.json").write_text(js(status(c)) + "\n")

def role(name):
    if name.endswith(".mlp.gate.weight") or name.endswith(".mlp.gate.e_score_correction_bias"):
        return "router"
    if "shared_experts" in name: return "shared_expert_resident"
    if re.search(r"\.experts?\.\d+\.", name): return "routed_expert"
    if any(x in name for x in ATTN): return "attention_resident"
    if any(x in name for x in UP + DOWN): return "dense_ffn_rowcol"
    return "resident"

def tensors(c):
    out = []
    for path in sorted(c["local_dir"].glob("*.safetensors")):
        with path.open("rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]; header = json.loads(f.read(n)); base = 8 + n
        for name, meta in header.items():
            if name == "__metadata__": continue
            if meta["dtype"] not in DTYPE: continue
            start, end = meta["data_offsets"]
            out.append((name, role(name), DTYPE[meta["dtype"]], "x".join(map(str, meta["shape"])), str(path), base + start, end - start))
    return out

def layout(c):
    ts = tensors(c)
    by_role = {}
    for _, r, _, _, _, _, n in ts: by_role[r] = by_role.get(r, 0) + n
    return {"model": c["model"], "source": str(c["local_dir"]), "layout": "sidecar", "tensors": len(ts), "bytes_by_role": by_role}

def materialize(c):
    s = status(c)
    if not s["complete"]: raise SystemExit(f"incomplete download: {s['safetensors']}/{s['expected_safetensors']} shards")
    c["meta_dir"].mkdir(parents=True, exist_ok=True)
    lines = ["blok-manifest-v1", "architecture=hybrid", "layout=sidecar"]
    index, alignment = [], 4096
    for name, r, dtype, shape, file, start, size in tensors(c):
        off, end = align_down(start, alignment), align(start + size, alignment)
        lines.append(f"tensor {name} {r} {dtype} {shape} {off} {end - off} {alignment} {file}")
        index.append({"name": name, "role": r, "dtype": dtype, "shape": shape, "file": file, "offset": off, "bytes": end - off})
    (c["meta_dir"] / "manifest.blok").write_text("\n".join(lines) + "\n")
    (c["meta_dir"] / "layout-index.json").write_text(js({"schema": "blok.layout.v1", "layout": "sidecar", "tensors": index}) + "\n")
    return {"model": c["model"], "layout": "sidecar", "tensors": len(index), "manifest": str(c["meta_dir"] / "manifest.blok")}

def detach(c):
    c["meta_dir"].mkdir(parents=True, exist_ok=True)
    log = open(c["meta_dir"] / "fetch.log", "ab")
    p = subprocess.Popen([sys.executable, __file__, c["model"], "fetch"], stdout=log, stderr=log, start_new_session=True)
    (c["meta_dir"] / "fetch.pid").write_text(f"{p.pid}\n")
    return {"model": c["model"], "pid": p.pid, "log": str(c["meta_dir"] / "fetch.log")}

def main():
    if len(sys.argv) != 3: die()
    c, mode = ctx(sys.argv[1]), sys.argv[2]
    if mode in ("status", "plan"): print(js(status(c)))
    elif mode == "fetch": fetch(c); print(js(status(c)))
    elif mode == "detach": print(js(detach(c)))
    elif mode == "layout": print(js(layout(c)))
    elif mode == "materialize": print(js(materialize(c)))
    else: die(f"unknown mode: {mode}")

if __name__ == "__main__": main()
