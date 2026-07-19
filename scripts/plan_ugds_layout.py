#!/usr/bin/env python3
import argparse
import fcntl
import os
import struct
import sys
from pathlib import Path

FS_IOC_FIEMAP = 0xC020660B
FIEMAP_FLAG_SYNC = 0x00000001
FIEMAP_EXTENT_LAST = 0x00000001
FIEMAP_EXTENT_UNKNOWN = 0x00000002
FIEMAP_EXTENT_DELALLOC = 0x00000004
FIEMAP_EXTENT_ENCODED = 0x00000008
FIEMAP_EXTENT_NOT_ALIGNED = 0x00000100
FIEMAP_EXTENT_DATA_INLINE = 0x00000200
FIEMAP_EXTENT_DATA_TAIL = 0x00000400
FIEMAP_EXTENT_UNWRITTEN = 0x00000800
FIEMAP_EXTENT_MERGED = 0x00001000
UNUSABLE_FLAGS = (
    FIEMAP_EXTENT_UNKNOWN
    | FIEMAP_EXTENT_DELALLOC
    | FIEMAP_EXTENT_ENCODED
    | FIEMAP_EXTENT_NOT_ALIGNED
    | FIEMAP_EXTENT_DATA_INLINE
    | FIEMAP_EXTENT_DATA_TAIL
    | FIEMAP_EXTENT_UNWRITTEN
    | FIEMAP_EXTENT_MERGED
)
FIEMAP_HEADER = struct.Struct("QQIIII")
FIEMAP_EXTENT = struct.Struct("QQQQQIIII")


def die(msg: str) -> None:
    raise SystemExit(msg)


def align_down(n: int, a: int) -> int:
    return n // a * a


def align_up(n: int, a: int) -> int:
    return ((n + a - 1) // a) * a


def parse_u64(value: str, name: str) -> int:
    try:
        out = int(value, 0)
    except ValueError:
        die(f"{name} must be an integer byte value")
    if out < 0:
        die(f"{name} must be non-negative")
    return out


def runtime_path(arg: str) -> Path:
    p = Path(arg).expanduser().resolve()
    if p.is_dir(): p /= "runtime-index.blok"
    if p.is_file() and p.name == "runtime-index.blok": return p
    die(f"missing runtime-index.blok: {p}")


def read_required_ranges(path: Path, block_size: int) -> dict[Path, list[tuple[int, int]]]:
    by_file: dict[Path, list[tuple[int, int]]] = {}
    for line in path.read_text().splitlines():
        if not line.startswith("tensor "):
            continue
        parts = line.split()
        if len(parts) != 14:
            die(f"bad tensor line in {path}: {line}")
        off, size, file_name = int(parts[8]), int(parts[9]), parts[13]
        start, end = align_down(off, block_size), align_up(off + size, block_size)
        by_file.setdefault(Path(file_name).expanduser().resolve(), []).append((start, end))
    if not by_file:
        die(f"no file-backed tensor ranges found in {path}")
    return {file: merge_ranges(ranges) for file, ranges in by_file.items()}


def merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for start, end in sorted(ranges):
        if start == end:
            continue
        if not out or start > out[-1][1]:
            out.append((start, end))
        else:
            out[-1] = (out[-1][0], max(out[-1][1], end))
    return out


def fiemap(path: Path, max_extents: int) -> list[tuple[int, int, int, int]]:
    size = path.stat().st_size
    buf = bytearray(FIEMAP_HEADER.size + max_extents * FIEMAP_EXTENT.size)
    FIEMAP_HEADER.pack_into(buf, 0, 0, size, FIEMAP_FLAG_SYNC, 0, max_extents, 0)
    with path.open("rb") as f:
        fcntl.ioctl(f.fileno(), FS_IOC_FIEMAP, buf, True)
    _, _, _, mapped, _, _ = FIEMAP_HEADER.unpack_from(buf, 0)
    extents: list[tuple[int, int, int, int]] = []
    for i in range(mapped):
        at = FIEMAP_HEADER.size + i * FIEMAP_EXTENT.size
        logical, physical, length, _, _, flags, _, _, _ = FIEMAP_EXTENT.unpack_from(buf, at)
        extents.append((logical, physical, length, flags))
        if flags & FIEMAP_EXTENT_LAST:
            break
    if mapped == max_extents and extents and not (extents[-1][3] & FIEMAP_EXTENT_LAST):
        die(f"too many FIEMAP extents for {path}; rerun with --max-extents above {max_extents}")
    return extents


def intersect_extents(
    file: Path,
    needed: list[tuple[int, int]],
    extents: list[tuple[int, int, int, int]],
    physical_offset_add: int,
    block_size: int,
) -> list[tuple[int, int, int]]:
    out: list[tuple[int, int, int]] = []
    for need_start, need_end in needed:
        cursor = need_start
        for logical, physical, length, flags in extents:
            start, end = max(need_start, logical), min(need_end, logical + length)
            if start >= end:
                continue
            if flags & UNUSABLE_FLAGS:
                die(f"{file} has an unmappable FIEMAP extent at logical {logical} with flags 0x{flags:x}")
            if start != cursor:
                die(f"{file} has a FIEMAP coverage gap at logical byte {cursor}")
            device = physical_offset_add + physical + (start - logical)
            bytes_len = end - start
            if start % block_size or device % block_size or bytes_len % block_size:
                die(f"{file} produced a non-{block_size}-byte-aligned uGDS extent")
            out.append((start, bytes_len, device))
            cursor = end
            if cursor == need_end:
                break
        if cursor != need_end:
            die(f"{file} FIEMAP did not cover required logical range {need_start}:{need_end}")
    return merge_extents(out)


def merge_extents(extents: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    out: list[tuple[int, int, int]] = []
    for logical, length, device in sorted(extents):
        if out and out[-1][0] + out[-1][1] == logical and out[-1][2] + out[-1][1] == device:
            prev = out[-1]
            out[-1] = (prev[0], prev[1] + length, prev[2])
        else:
            out.append((logical, length, device))
    return out


def validate_no_overlap(model_extents: list[tuple[str, int, int]], kv_base: int, kv_bytes: int) -> None:
    if kv_bytes <= 0:
        die("KV scratch bytes must be positive")
    kv_end = kv_base + kv_bytes
    for file, device, length in model_extents:
        if max(device, kv_base) < min(device + length, kv_end):
            die(f"KV scratch overlaps model extent for {file}: device {device}:{device + length}")


def write_map(path: Path, device: str, kv_base: int, kv_bytes: int, mappings: dict[Path, list[tuple[int, int, int]]]) -> None:
    lines = [
        "blok-ugds-map-v1",
        f"device {device}",
        "block_size 4096",
        f"kv_scratch {kv_base} {kv_bytes}",
    ]
    for file in sorted(mappings):
        for logical, length, device_off in mappings[file]:
            lines.append(f"file {file} {logical} {length} {device_off}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def write_env(path: Path, device: str, map_path: Path, kv_base: int, kv_bytes: int) -> None:
    lines = [
        f"export BLOK_UGDS_DEVICE={shell_quote(device)}",
        f"export BLOK_UGDS_MAP={shell_quote(str(map_path))}",
        f"export BLOK_KV_UGDS_BASE={kv_base}",
        f"export BLOK_KV_UGDS_BYTES={kv_bytes}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate an extent-aware BLOK_UGDS_MAP for Kimi safetensor shards. "
            "Run this while the model filesystem is mounted, then unmount and bind the NVMe device to uGDS before raw reads."
        )
    )
    parser.add_argument("model", help="runtime-index.blok or metadata directory")
    parser.add_argument("--output", default=os.getenv("BLOK_UGDS_MAP", "ugds-map.blok"))
    parser.add_argument("--env-output", default=None, help="optional shell exports file")
    parser.add_argument("--device", default=os.getenv("BLOK_UGDS_DEVICE", "/dev/ugds_drv0"))
    parser.add_argument("--kv-base", default=os.getenv("BLOK_KV_UGDS_BASE"))
    parser.add_argument("--kv-bytes", default=os.getenv("BLOK_KV_UGDS_BYTES"))
    parser.add_argument("--physical-offset-add", default="0", help="bytes to add to FIEMAP physical offsets, e.g. partition start")
    parser.add_argument("--max-extents", type=int, default=262144)
    args = parser.parse_args()

    if args.kv_base is None or args.kv_bytes is None:
        die("set --kv-base and --kv-bytes, or BLOK_KV_UGDS_BASE and BLOK_KV_UGDS_BYTES")
    kv_base = parse_u64(args.kv_base, "--kv-base")
    kv_bytes = parse_u64(args.kv_bytes, "--kv-bytes")
    physical_offset_add = parse_u64(args.physical_offset_add, "--physical-offset-add")
    if kv_base % 4096 or kv_bytes % 4096:
        die("KV scratch base and bytes must be block aligned")

    rt = runtime_path(args.model)
    required = read_required_ranges(rt, 4096)
    mappings: dict[Path, list[tuple[int, int, int]]] = {}
    model_extents: list[tuple[str, int, int]] = []
    for file, ranges in required.items():
        if not file.is_file():
            die(f"missing tensor shard: {file}")
        extents = fiemap(file, args.max_extents)
        if not extents:
            die(f"FIEMAP returned no extents for {file}")
        mapped = intersect_extents(file, ranges, extents, physical_offset_add, 4096)
        mappings[file] = mapped
        model_extents.extend((str(file), device, length) for _, length, device in mapped)

    validate_no_overlap(model_extents, kv_base, kv_bytes)
    out = Path(args.output).expanduser().resolve()
    write_map(out, args.device, kv_base, kv_bytes, mappings)
    if args.env_output:
        write_env(Path(args.env_output).expanduser().resolve(), args.device, out, kv_base, kv_bytes)
    print(f"ok: wrote {out} with {len(model_extents)} extents")


if __name__ == "__main__":
    if sys.platform != "linux":
        die("plan_ugds_layout.py requires Linux FIEMAP")
    main()
