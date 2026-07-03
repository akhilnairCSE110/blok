#!/usr/bin/env python3
import json, os, subprocess, sys
from pathlib import Path

MODELS = {
    "kimi-k2.6": ("moonshotai/Kimi-K2.6", "7eb5002f6aadc958aed6a9177b7ed26bb94011bb", 595421860056, 64)
}

def die():
    print(f"usage: {sys.argv[0]} kimi-k2.6 {{plan|status|fetch|detach}}", file=sys.stderr)
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

def main():
    if len(sys.argv) != 3: die()
    c, mode = ctx(sys.argv[1]), sys.argv[2]
    if mode in ("plan", "status"): print(json.dumps(status(c), separators=(",", ":")))
    elif mode == "fetch": fetch(c); print(json.dumps(status(c), separators=(",", ":")))
    elif mode == "detach":
        c["meta_dir"].mkdir(parents=True, exist_ok=True)
        log = open(c["meta_dir"] / "fetch.log", "ab")
        p = subprocess.Popen([sys.executable, __file__, sys.argv[1], "fetch"], stdout=log, stderr=log, start_new_session=True)
        (c["meta_dir"] / "fetch.pid").write_text(f"{p.pid}\n")
        print(json.dumps({"model": sys.argv[1], "pid": p.pid, "log": str(c["meta_dir"] / "fetch.log")}, separators=(",", ":")))
    else: die()

if __name__ == "__main__": main()
