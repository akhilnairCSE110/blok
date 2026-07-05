#!/usr/bin/env python3
# 1. MoE sparsity makes storage bandwidth the bottleneck: keep dense/shared weights resident and stream only routed experts.
# 2. Expert MoE layout is per-layer contiguous expert blocks: offset = expert_id * expert_size, read with pread, no file lookup.
# 3. Dense sparse-FFN layout is neuron row-column bundling: up/gate neuron input slice plus matching down output slice are contiguous.
# 4. Never mmap streamed expert payloads: large random expert reads should be explicit pread/GDS/uGDS transfers into fixed arenas.
# 5. Generation must not use payload mmap or unreported buffered reads; direct I/O/CUDA gates own payload movement.
# 6. K is a quality contract, not just speed: default top_k=4 for routed experts and reject K<4 unless explicitly overridden later.
# 7. Runtime memory is static: download and layout planning may allocate, generation hot paths must use fixed slots and pointer maps.
# 8. Repacking is separate from fetching: this file may download and plan; destructive/raw layout writes require a later explicit gate.
import json, os, re, shutil, struct, subprocess, sys
from pathlib import Path

MODELS = {"kimi-k2.6": ("moonshotai/Kimi-K2.6", "7eb5002f6aadc958aed6a9177b7ed26bb94011bb", 595421860056, 64)}
ATTN = ("q_proj","k_proj","v_proj","o_proj","q_a_proj","q_b_proj","kv_a_proj","kv_b_proj")
UP, DOWN = ("up_proj","gate_proj","w1","w3"), ("down_proj","w2")
DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "I8": 1, "U8": 1}
ROLE_ORDER = {"resident": 0, "attention_resident": 1, "shared_expert_resident": 2, "dense_ffn_rowcol": 3, "routed_expert": 4}

def die():
    print(f"usage: {sys.argv[0]} kimi-k2.6 {{plan|status|fetch|detach|layout|materialize}}", file=sys.stderr); raise SystemExit(64)

def js(x): return json.dumps(x, separators=(",", ":"))
def rd(p):
    try: return Path(p).read_text().strip()
    except OSError: return ""

def ctx(m):
    if m not in MODELS: die()
    repo, rev, bytes_, st = MODELS[m]
    home = Path(os.getenv("BLOK_HOME", Path(__file__).resolve().parents[3] / ".blok"))
    root = Path(os.getenv("BLOK_MODEL_ROOT", home / "models")) / repo
    return dict(model=m, repo=repo, revision=rev, expected_bytes=bytes_, expected_safetensors=st,
                local_dir=root / "source/hf" / rev, meta_dir=root / "blok",
                layout_dir=root / "blok" / "repacked",
                cache_dir=home / "hf-cache", hf=os.getenv("BLOK_HF_BIN", "hf"),
                top_k=int(os.getenv("BLOK_TOP_K", "4")))

def hw(c):
    dev = next((p.name for p in Path("/sys/block").glob("nvme*n*")), "")
    q, link = Path("/sys/block") / dev / "queue", Path("/sys/block") / dev / "device"
    pci = next((p for p in link.resolve().parents if (p / "vendor").exists()), link)
    root = any(line.split()[4] == "/" and dev in line for line in rd("/proc/self/mountinfo").splitlines())
    return dict(host=rd("/proc/sys/kernel/hostname"), nvme=dev, nvme_role="root" if root else "model_or_unknown",
                nvme_model=rd(Path("/sys/block") / dev / "device/model"),
                block_bytes=int(rd(q / "logical_block_size") or 4096),
                read_ahead_kb=int(rd(q / "read_ahead_kb") or 128),
                nr_requests=int(rd(q / "nr_requests") or 0), scheduler=rd(q / "scheduler"),
                pcie=f"{rd(pci/'current_link_speed')} x{rd(pci/'current_link_width')}",
                pcie_max=f"{rd(pci/'max_link_speed')} x{rd(pci/'max_link_width')}",
                cuda="present" if Path("/dev/nvidiactl").exists() else "permission_blocked",
                direct_repack_allowed=not root, destructive_ops_allowed=False,
                signed_off="source_download_and_metadata_plan_only_on_root_nvme" if root else "model_nvme_layout_writes_require_operator_gate",
                transfer="direct_io_or_cuda_required_before_generate")

def status(c):
    fs = list(c["local_dir"].rglob("*")) if c["local_dir"].is_dir() else []
    files = [p for p in fs if p.is_file() and ".cache" not in p.relative_to(c["local_dir"]).parts]
    got = sum(p.stat().st_size for p in files)
    st = sum(p.name.endswith(".safetensors") for p in files)
    return {k: str(v) if k.endswith("_dir") else v for k, v in c.items() if k not in ("hf","meta_dir")} | \
           dict(downloaded_bytes=got, safetensors=st, complete=got >= c["expected_bytes"] and st >= c["expected_safetensors"], hardware=hw(c))

def fetch(c):
    c["local_dir"].mkdir(parents=True, exist_ok=True); c["meta_dir"].mkdir(parents=True, exist_ok=True); c["cache_dir"].mkdir(parents=True, exist_ok=True)
    env = os.environ | dict(HF_HOME=str(c["cache_dir"]), HF_HUB_CACHE=str(c["cache_dir"]/"hub"),
                            HF_XET_CACHE=str(c["cache_dir"]/"xet"), HF_XET_HIGH_PERFORMANCE="1",
                            HF_XET_NUM_CONCURRENT_RANGE_GETS=os.getenv("HF_XET_NUM_CONCURRENT_RANGE_GETS","32"),
                            HF_HUB_DOWNLOAD_TIMEOUT=os.getenv("HF_HUB_DOWNLOAD_TIMEOUT","60"))
    preflight_hf_auth(c, env)
    subprocess.run([c["hf"], "download", c["repo"], "--revision", c["revision"], "--local-dir", str(c["local_dir"])], check=True, env=env)
    (c["meta_dir"]/"fetch-status.json").write_text(js(status(c))+"\n")

def preflight_hf_auth(c, env):
    token = env.get("HF_TOKEN") or env.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        env["HF_TOKEN"] = token
        env["HUGGING_FACE_HUB_TOKEN"] = token
    result = subprocess.run(
        [c["hf"], "auth", "whoami"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or "hf auth whoami failed"
        raise SystemExit(f"Hugging Face authentication failed; download not started: {detail}")

def align(n, a): return ((n + a - 1) // a) * a
def align_down(n, a): return (n // a) * a
def dtype(d): return d.lower().replace("float", "f").replace("bfloat", "bf")

def tensors(c):
    out = {}
    for p in c["local_dir"].glob("*.safetensors"):
        with p.open("rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]; h = json.loads(f.read(n)); base = 8 + n
        out |= {k: dict(file=str(p), dtype=v["dtype"], shape=v["shape"],
                        offsets=v["data_offsets"], source_offset=base + v["data_offsets"][0],
                        bytes=(v["data_offsets"][1]-v["data_offsets"][0])) for k, v in h.items() if k != "__metadata__"}
    return out

def layer(t): return (re.search(r"layers?\.(\d+)", t) or [None, "-"])[1]
def expert(t): return (re.search(r"experts?\.(\d+)", t) or [None, "-"])[1]
def role(t):
    if "shared_experts" in t: return "shared_expert_resident"
    if re.search(r"\.experts?\.\d+\.", t): return "routed_expert"
    if any(x in t for x in ATTN): return "attention_resident"
    if any(x in t for x in UP): return "dense_ffn_up_flash"
    if any(x in t for x in DOWN): return "dense_ffn_down_flash"
    return "resident"

def layout(c):
    if c["top_k"] < 4: raise SystemExit("top_k below 4 is disabled: prior MoE results show quality collapse")
    h, hardware, groups = tensors(c), hw(c), {}
    for t in h: groups.setdefault((layer(t), expert(t), role(t)), []).append(t)
    expert_groups = {(l,e): v for (l,e,r), v in groups.items() if r == "routed_expert" and e != "-"}
    by_layer = {}
    for (l,e), ts in expert_groups.items(): by_layer.setdefault(l, []).append((e, ts))
    expert_layers = [dict(layer=l, experts=len(v), layout="expert_contiguous_layer_file",
                          read="pread(fd[layer], arena, expert_size, expert_id*expert_size)")
                     for l, v in sorted(by_layer.items(), key=lambda x: int(x[0]) if x[0].isdigit() else -1)]
    rowcol = [dict(layer=l, layout="row_column_bundle", up=groups.get((l,"-", "dense_ffn_up_flash"), []),
                   down=groups.get((l,"-", "dense_ffn_down_flash"), []))
              for l in sorted({k[0] for k in groups}) if groups.get((l,"-", "dense_ffn_up_flash")) and groups.get((l,"-", "dense_ffn_down_flash"))]
    bytes_by_role = {}
    for t, v in h.items(): bytes_by_role[role(t)] = bytes_by_role.get(role(t), 0) + v["bytes"]
    return dict(model=c["model"], source=str(c["local_dir"]), hardware=hardware,
                policy=dict(top_k=c["top_k"], expert_io="direct_io_required", custom_lru=False,
                            mmap_experts=False, dense_ffn="row_column_bundle", moe="expert_contiguous"),
                tensors=len(h), bytes_by_role=bytes_by_role, expert_layers=expert_layers,
                dense_ffn_bundles=rowcol, resident=[t for t in h if role(t) in ("resident","attention_resident")])

def copy_tensor(src, dst, meta, size=None):
    remaining = meta["bytes"] if size is None else size
    with open(meta["file"], "rb") as f:
        f.seek(meta["source_offset"])
        while remaining:
            chunk = f.read(min(8 * 1024 * 1024, remaining))
            if not chunk: raise OSError(f"short read from {meta['file']}")
            dst.write(chunk); remaining -= len(chunk)

def tensor_sort_key(name):
    return (int(layer(name)) if layer(name).isdigit() else -1,
            int(expert(name)) if expert(name).isdigit() else -1,
            ROLE_ORDER.get(role(name), 99), name)

def write_padding(f, count):
    if count <= 0: return
    zero = b"\0" * min(1024 * 1024, count)
    while count:
        n = min(len(zero), count); f.write(zero[:n]); count -= n

def rowcol_names(groups, l):
    ups = sorted(groups.get((l,"-", "dense_ffn_up_flash"), []))
    downs = sorted(groups.get((l,"-", "dense_ffn_down_flash"), []))
    gates = [n for n in ups if any(x in n for x in ("gate_proj", "w1"))]
    up_only = [n for n in ups if n not in gates]
    return gates, up_only, downs

def write_row(f, meta, row, elems, elem):
    with open(meta["file"], "rb") as src:
        src.seek(meta["source_offset"] + row * elems * elem)
        data = src.read(elems * elem)
        if len(data) != elems * elem: raise OSError(f"short row read from {meta['file']}")
        f.write(data)

def write_column(f, meta, col, rows, cols, elem):
    with open(meta["file"], "rb") as src:
        for r in range(rows):
            src.seek(meta["source_offset"] + ((r * cols) + col) * elem)
            data = src.read(elem)
            if len(data) != elem: raise OSError(f"short column read from {meta['file']}")
            f.write(data)

def materialize_dense_rowcol(c, h, groups, l, records, align_bytes):
    gates, ups, downs = rowcol_names(groups, l)
    names = gates + ups + downs
    if not names: return
    if len(gates) > 1 or len(ups) > 1 or len(downs) != 1:
        raise SystemExit(f"layer {l} dense FFN row-column bundling needs one gate, one up, one down tensor")
    gate, up, down = (gates[0] if gates else None), (ups[0] if ups else None), downs[0]
    tensors_ = [n for n in (gate, up, down) if n]
    dtypes = {h[n]["dtype"] for n in tensors_}
    if len(dtypes) != 1 or next(iter(dtypes)) not in DTYPE_BYTES:
        raise SystemExit(f"layer {l} dense FFN row-column bundling only supports fixed-width numeric dtypes")
    elem = DTYPE_BYTES[next(iter(dtypes))]
    out, tmp = c["layout_dir"] / "dense_ffn" / f"layer-{int(l):04d}.bin", c["layout_dir"] / "dense_ffn" / f"layer-{int(l):04d}.bin.tmp"
    out.parent.mkdir(parents=True, exist_ok=True)
    hidden = h[down]["shape"][0]; neurons = h[down]["shape"][1]
    for n in (gate, up):
        if n and h[n]["shape"] != [neurons, hidden]: raise SystemExit(f"{n} shape does not match row-column contract")
    block = align(((1 if gate else 0) + (1 if up else 0)) * hidden * elem + hidden * elem, align_bytes)
    offset = 0
    with tmp.open("wb") as f:
        for i in range(neurons):
            start = offset
            if gate: write_row(f, h[gate], i, hidden, elem); offset += hidden * elem
            if up: write_row(f, h[up], i, hidden, elem); offset += hidden * elem
            write_column(f, h[down], i, hidden, neurons, elem); offset += hidden * elem
            write_padding(f, start + block - offset); offset = start + block
            records.append(dict(name=f"layer.{l}.dense_ffn.neuron.{i}", role="dense_ffn_rowcol", dtype=dtype(h[down]["dtype"]),
                                shape=[hidden], file=str(out), offset=start, bytes=block, alignment=align_bytes,
                                layout="gate_up_down_neuron_bundle"))
    tmp.replace(out)

def materialize_resident(c, h, records, align_bytes):
    names = [n for n in sorted(h, key=tensor_sort_key) if role(n) in ("resident", "attention_resident", "shared_expert_resident")]
    if not names: return
    out, tmp = c["layout_dir"] / "resident.bin", c["layout_dir"] / "resident.bin.tmp"
    out.parent.mkdir(parents=True, exist_ok=True); offset = 0
    with tmp.open("wb") as f:
        for n in names:
            start = align(offset, align_bytes); write_padding(f, start - offset); offset = start
            copy_tensor(None, f, h[n]); offset += h[n]["bytes"]
            records.append(dict(name=n, role=role(n), dtype=dtype(h[n]["dtype"]), shape=h[n]["shape"],
                                file=str(out), offset=start, bytes=h[n]["bytes"], alignment=align_bytes,
                                source_file=h[n]["file"], source_offset=h[n]["source_offset"]))
    tmp.replace(out)

def materialize_experts(c, h, groups, records, align_bytes):
    layers = sorted({l for (l,e,r) in groups if r == "routed_expert" and e != "-"}, key=int)
    for l in layers:
        expert_ids = sorted({int(e) for (gl,e,r) in groups if gl == l and r == "routed_expert" and e != "-"})
        if expert_ids != list(range(max(expert_ids) + 1)): raise SystemExit(f"layer {l} experts are not dense from zero")
        per = {}
        for eid in expert_ids:
            ts = sorted(groups[(l, str(eid), "routed_expert")])
            per[eid] = (ts, align(sum(align(h[t]["bytes"], align_bytes) for t in ts), align_bytes))
        expert_size = max(size for _, size in per.values())
        out, tmp = c["layout_dir"] / "experts" / f"layer-{int(l):04d}.bin", c["layout_dir"] / "experts" / f"layer-{int(l):04d}.bin.tmp"
        out.parent.mkdir(parents=True, exist_ok=True)
        with tmp.open("wb") as f:
            for eid in expert_ids:
                block_start = eid * expert_size
                f.seek(block_start); offset = block_start
                for n in per[eid][0]:
                    start = align(offset, align_bytes); write_padding(f, start - offset); offset = start
                    copy_tensor(None, f, h[n]); offset += h[n]["bytes"]
                    records.append(dict(name=n, role="routed_expert", dtype=dtype(h[n]["dtype"]), shape=h[n]["shape"],
                                        file=str(out), offset=start, bytes=h[n]["bytes"], alignment=align_bytes,
                                        layer=int(l), expert=eid, expert_size=expert_size,
                                        expert_offset=block_start, source_file=h[n]["file"], source_offset=h[n]["source_offset"]))
                write_padding(f, block_start + expert_size - offset)
        tmp.replace(out)

def manifest_text(records, layout_="repacked"):
    lines = ["blok-manifest-v1", "architecture=hybrid", f"layout={layout_}"]
    for r in sorted(records, key=lambda x: (x["file"], x["offset"], x["name"])):
        shape = "x".join(str(v) for v in r["shape"])
        lines.append(f"tensor {r['name']} {r['role']} {r['dtype']} {shape} {r['offset']} {r['bytes']} {r['alignment']} {r['file']}")
    return "\n".join(lines) + "\n"

def materialize(c):
    s = status(c)
    if not s["complete"]: raise SystemExit("source download is incomplete; run fetch or detach first")
    if c["top_k"] < 4: raise SystemExit("top_k below 4 is disabled: prior MoE results show quality collapse")
    if os.getenv("BLOK_REPACK_LAYOUT") != "1":
        h, records = tensors(c), []
        align_bytes = max(4096, hw(c)["block_bytes"])
        for n in sorted(h, key=tensor_sort_key):
            start = align_down(h[n]["source_offset"], align_bytes)
            end = align(h[n]["source_offset"] + h[n]["bytes"], align_bytes)
            records.append(dict(name=n, role=role(n), dtype=dtype(h[n]["dtype"]), shape=h[n]["shape"],
                                file=h[n]["file"], offset=start, bytes=end-start, tensor_delta=h[n]["source_offset"]-start,
                                tensor_bytes=h[n]["bytes"], alignment=align_bytes,
                                source_file=h[n]["file"], source_offset=h[n]["source_offset"]))
        c["meta_dir"].mkdir(parents=True, exist_ok=True)
        index = dict(schema="blok.layout.v1", model=c["model"], repo=c["repo"], revision=c["revision"],
                     layout="sidecar", root=str(c["local_dir"]), alignment=align_bytes,
                     policy=layout(c)["policy"], tensors=records)
        (c["meta_dir"]/"layout-index.json").write_text(js(index)+"\n")
        (c["meta_dir"]/"manifest.blok").write_text(manifest_text(records, "sidecar"))
        return dict(model=c["model"], layout_dir=str(c["local_dir"]), layout="sidecar",
                    tensors=len(records), bytes=sum(r["bytes"] for r in records),
                    manifest=str(c["meta_dir"]/"manifest.blok"), index=str(c["meta_dir"]/"layout-index.json"))
    if c["layout_dir"].exists() and os.getenv("BLOK_OVERWRITE_LAYOUT") != "1":
        raise SystemExit(f"{c['layout_dir']} exists; set BLOK_OVERWRITE_LAYOUT=1 to replace Blok-owned layout files")
    if c["layout_dir"].exists(): shutil.rmtree(c["layout_dir"])
    h, groups, records = tensors(c), {}, []
    align_bytes = max(4096, hw(c)["block_bytes"])
    for t in h: groups.setdefault((layer(t), expert(t), role(t)), []).append(t)
    materialize_resident(c, h, records, align_bytes)
    for l in sorted({k[0] for k in groups if k[0] != "-"}, key=int): materialize_dense_rowcol(c, h, groups, l, records, align_bytes)
    materialize_experts(c, h, groups, records, align_bytes)
    c["meta_dir"].mkdir(parents=True, exist_ok=True)
    index = dict(schema="blok.layout.v1", model=c["model"], repo=c["repo"], revision=c["revision"],
                 layout="repacked", root=str(c["layout_dir"]), alignment=align_bytes,
                 policy=layout(c)["policy"], tensors=records)
    (c["meta_dir"]/"layout-index.json").write_text(js(index)+"\n")
    (c["meta_dir"]/"manifest.blok").write_text(manifest_text(records))
    return dict(model=c["model"], layout_dir=str(c["layout_dir"]), tensors=len(records),
                bytes=sum(r["bytes"] for r in records), manifest=str(c["meta_dir"]/"manifest.blok"),
                index=str(c["meta_dir"]/"layout-index.json"))

def main():
    if len(sys.argv) != 3: die()
    c, mode = ctx(sys.argv[1]), sys.argv[2]
    if mode in ("plan","status"): print(js(status(c)))
    elif mode == "layout": print(js(layout(c)))
    elif mode == "materialize": print(js(materialize(c)))
    elif mode == "fetch": fetch(c); print(js(status(c)))
    elif mode == "detach":
        c["meta_dir"].mkdir(parents=True, exist_ok=True); log = open(c["meta_dir"]/"fetch.log", "ab")
        p = subprocess.Popen([sys.executable, __file__, c["model"], "fetch"], stdout=log, stderr=log, start_new_session=True)
        (c["meta_dir"]/"fetch.pid").write_text(f"{p.pid}\n"); print(js(dict(model=c["model"], pid=p.pid, log=str(c["meta_dir"]/"fetch.log"))))
    else: die()

if __name__ == "__main__": main()
