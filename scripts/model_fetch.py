#!/usr/bin/env python3
# 1. Avoid tiny random reads: model storage must favor large aligned sequential chunks because flash latency to first byte dominates small I/O.
# 2. Keep dense weights resident: embeddings and attention weights are DRAM/VRAM candidates, while sparse FFN and expert weights are flash-resident.
# 3. Predict before fetching: activation sparsity only helps if a cheap predictor names active neurons before their FFN weights are read.
# 4. Bundle by neuron: each active FFN neuron should map to one contiguous row-column block containing its up/gate input slice and down output slice.
# 5. Do not bundle by coactivation until measured: popular neurons create redundant reads and can make nearest-neighbor bundling slower.
# 6. Preallocate runtime slots: streamed neuron blocks move into fixed arenas with pointer maps and last-k activity state, never hot-path allocation.
# 7. Separate acquisition from execution: downloading may be naive, but layout metadata must preserve enough tensor identity for later repacking.
import json, os, re, struct, subprocess, sys
from pathlib import Path

MODELS = {
    "kimi-k2.6": ("moonshotai/Kimi-K2.6", "7eb5002f6aadc958aed6a9177b7ed26bb94011bb", 595421860056, 64)
}

def die():
    print(f"usage: {sys.argv[0]} kimi-k2.6 {{plan|status|fetch|detach|layout}}", file=sys.stderr)
    raise SystemExit(64)

def ctx(name):
    repo, rev, want_bytes, want_st = MODELS.get(name, (None, None, None, None))
    if not repo: die()
    home = Path(os.getenv("BLOK_HOME", Path.home() / ".blok"))
    root = Path(os.getenv("BLOK_MODEL_ROOT", home / "models")) / repo
    return dict(model=name, repo=repo, revision=rev, expected_bytes=want_bytes,
                expected_safetensors=want_st, local_dir=root / "source/hf" / rev,
                meta_dir=root / "blok", cache_dir=home / "hf-cache",
                hf=os.getenv("BLOK_HF_BIN", "hf"))

def status(c):
    files = list(c["local_dir"].rglob("*")) if c["local_dir"].is_dir() else []
    got = sum(p.stat().st_size for p in files if p.is_file())
    st = sum(p.name.endswith(".safetensors") for p in files if p.is_file())
    return {k: str(v) if k.endswith("_dir") else v for k, v in c.items()
            if k not in ("hf", "meta_dir")} | dict(downloaded_bytes=got,
            safetensors=st, complete=got >= c["expected_bytes"] and st >= c["expected_safetensors"])

def fetch(c):
    c["local_dir"].mkdir(parents=True, exist_ok=True); c["meta_dir"].mkdir(parents=True, exist_ok=True)
    c["cache_dir"].mkdir(parents=True, exist_ok=True)
    env = os.environ | dict(HF_HOME=str(c["cache_dir"]), HF_HUB_CACHE=str(c["cache_dir"] / "hub"),
                            HF_XET_CACHE=str(c["cache_dir"] / "xet"), HF_XET_HIGH_PERFORMANCE="1",
                            HF_XET_NUM_CONCURRENT_RANGE_GETS=os.getenv("HF_XET_NUM_CONCURRENT_RANGE_GETS", "32"),
                            HF_HUB_DOWNLOAD_TIMEOUT=os.getenv("HF_HUB_DOWNLOAD_TIMEOUT", "60"))
    subprocess.run([c["hf"], "download", c["repo"], "--revision", c["revision"],
                    "--local-dir", str(c["local_dir"])], check=True, env=env)
    (c["meta_dir"] / "fetch-status.json").write_text(json.dumps(status(c), separators=(",", ":")) + "\n")

def headers(c):
    out = {}
    for p in c["local_dir"].glob("*.safetensors"):
        with p.open("rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            h = json.loads(f.read(n))
        out |= {k: dict(file=str(p), dtype=v["dtype"], shape=v["shape"],
                        offsets=v["data_offsets"]) for k, v in h.items() if k != "__metadata__"}
    return out

def kind(t):
    if any(x in t for x in ("q_proj","k_proj","v_proj","o_proj","q_a_proj","q_b_proj","kv_a_proj","kv_b_proj")): return "attention_dram"
    if any(x in t for x in ("up_proj","gate_proj","w1","w3")): return "ffn_up_flash"
    if any(x in t for x in ("down_proj","w2")): return "ffn_down_flash"
    return "resident_or_aux"

def group(t):
    layer = re.search(r"layers?\.(\d+)", t)
    expert = re.search(r"experts?\.(\d+)", t)
    return (layer.group(1) if layer else "-", expert.group(1) if expert else "-")

def layout(c):
    h, groups = headers(c), {}
    for t in h: groups.setdefault(group(t), {}).setdefault(kind(t), []).append(t)
    bundles = [dict(layer=k[0], expert=k[1], up=v.get("ffn_up_flash", []),
                    down=v.get("ffn_down_flash", []), layout="row_column_bundle")
               for k, v in groups.items() if v.get("ffn_up_flash") and v.get("ffn_down_flash")]
    plan = dict(model=c["model"], source=str(c["local_dir"]), policy=dict(
        attention="dram_resident", ffn="flash_resident_row_column_bundle",
        experts="expert_contiguous", cache="sliding_window_static_slots"),
        tensors=len(h), bytes=sum(v["offsets"][1]-v["offsets"][0] for v in h.values()),
        bundles=bundles)
    if h:
        c["meta_dir"].mkdir(parents=True, exist_ok=True)
        (c["meta_dir"] / "layout-plan.json").write_text(json.dumps(plan, separators=(",", ":")) + "\n")
    return plan

def main():
    if len(sys.argv) != 3: die()
    c, mode = ctx(sys.argv[1]), sys.argv[2]
    if mode in ("plan", "status"): print(json.dumps(status(c), separators=(",", ":")))
    elif mode == "fetch": fetch(c); print(json.dumps(status(c), separators=(",", ":")))
    elif mode == "layout": print(json.dumps(layout(c), separators=(",", ":")))
    elif mode == "detach":
        c["meta_dir"].mkdir(parents=True, exist_ok=True)
        log = open(c["meta_dir"] / "fetch.log", "ab")
        p = subprocess.Popen([sys.executable, __file__, sys.argv[1], "fetch"], stdout=log, stderr=log, start_new_session=True)
        (c["meta_dir"] / "fetch.pid").write_text(f"{p.pid}\n")
        print(json.dumps({"model": sys.argv[1], "pid": p.pid, "log": str(c["meta_dir"] / "fetch.log")}, separators=(",", ":")))
    else: die()

if __name__ == "__main__": main()
