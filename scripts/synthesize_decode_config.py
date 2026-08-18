#!/usr/bin/env python3
"""Rank parity-matched decode logs on a hardware-synthesis PPA frontier."""

import argparse
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path

FIELDS = re.compile(r"(\w+)=([^ ]+)")
CACHE = re.compile(r"routed-expert cache ways=(\d+) capacity=([0-9.]+)GB")
GROUP = re.compile(r"moe-pipeline .* group_size=(\d+)")
SUMMARY = re.compile(r"decode (\d+) steps/\d+ emitted in ([0-9.]+)s")
E2E = re.compile(r"\[metalblok-wrapper\] e2e_s=([0-9.]+)")


@dataclass
class Run:
    path: Path
    ways: int
    cache_gb: float
    group: int
    rows: list[dict[str, float]]
    samples: dict[int, tuple[int, str]]
    decode_steps: int
    decode_s: float
    e2e_s: float

    def mean(self, name: str) -> float:
        values = [row[name] for row in self.rows if name in row]
        return statistics.fmean(values) if values else float("inf")

    def mean0(self, name: str) -> float:
        values = [row[name] for row in self.rows if name in row]
        return statistics.fmean(values) if values else 0.0

    def percentile(self, name: str, fraction: float) -> float:
        values = sorted(row[name] for row in self.rows if name in row)
        return values[min(len(values) - 1, int(fraction * len(values)))]

    def step_wall_us(self) -> float:
        return self.decode_s * 1e6 / self.decode_steps if self.decode_steps else self.mean("wall_us")


def parse(path: Path, warmup: int) -> Run:
    rows: list[dict[str, float]] = []
    samples: dict[int, tuple[int, str]] = {}
    ways, cache_gb, group = 0, 0.0, 4
    decode_steps, decode_s, e2e_s = 0, 0.0, float("nan")
    for line in path.read_text(errors="replace").splitlines():
        if match := CACHE.search(line):
            ways, cache_gb = int(match.group(1)), float(match.group(2))
        if match := GROUP.search(line):
            group = int(match.group(1))
        if "[metalblok] metrics pos=" in line:
            values: dict[str, float] = {}
            for key, value in FIELDS.findall(line):
                try:
                    values[key] = float(value)
                except ValueError:
                    pass
            rows.append(values)
        if "[metalblok] sample pos=" in line:
            fields = dict(FIELDS.findall(line))
            if "pos" in fields and "token" in fields:
                fingerprint = fields.get("logits_hash", "logit:" + fields.get("logit", ""))
                samples[int(fields["pos"])] = int(fields["token"]), fingerprint
        if match := SUMMARY.search(line):
            decode_steps, decode_s = int(match.group(1)), float(match.group(2))
        if match := E2E.search(line):
            e2e_s = float(match.group(1))
    rows = rows[warmup:]
    if not rows:
        raise ValueError(f"{path}: no post-warmup decode metrics")
    return Run(path, ways, cache_gb, group, rows, samples,
               decode_steps, decode_s, e2e_s)


def parity(run: Run, reference: Run) -> str:
    if run.path == reference.path:
        return "reference"
    common = run.samples.keys() & reference.samples.keys()
    if not common:
        return "unknown"
    if not all(run.samples[pos] == reference.samples[pos] for pos in common):
        return "NO"
    full_hashes = all(not run.samples[pos][1].startswith("logit:") and
                      not reference.samples[pos][1].startswith("logit:")
                      for pos in common)
    return "logits" if full_hashes else "sample"


def dominates(left: Run, right: Run) -> bool:
    # Performance, storage traffic, resident "area", and GPU work are the
    # runtime equivalents of timing, interconnect/power, area, and compute.
    a = (left.step_wall_us(), left.mean("nvme_bytes"), left.cache_gb,
         left.mean("gpu_us"))
    b = (right.step_wall_us(), right.mean("nvme_bytes"), right.cache_gb,
         right.mean("gpu_us"))
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path,
                        help="first log is the strict numerical reference")
    parser.add_argument("--warmup", type=int, default=1,
                        help="discard this many initial decode metric rows")
    parser.add_argument("--max-cache-gb", type=float, default=float("inf"))
    parser.add_argument("--allow-sample-parity", action="store_true",
                        help="permit legacy logs without full-logit hashes")
    args = parser.parse_args()
    runs = [parse(path, args.warmup) for path in args.logs]
    reference = runs[0]
    accepted_states = {"reference", "logits"}
    if args.allow_sample_parity:
        accepted_states.add("sample")
    accepted = [run for run in runs if parity(run, reference) in accepted_states]
    frontier = {run.path for run in accepted
                if not any(dominates(other, run) for other in accepted if other is not run)}

    print("parity pareto ways group cache_GB token/s decode_s e2e_s p50_ms "
          "p95_ms GPU_ms GPU_pct IOwait_ms IOwait_pct NVMe_GB NVMe_GBps "
          "cache_hit KV_KB avail_MB cmdbufs dispatches allocations pageouts "
          "compress decompress swapout prefetch_GB useful_GB false_GB "
          "drain_ms log")
    for run in sorted(runs, key=lambda item: item.step_wall_us()):
        wall = run.step_wall_us()
        hits = run.mean("expert_cache_hits")
        misses = run.mean("expert_cache_misses")
        cache_hit = (100 * hits / (hits + misses)
                     if math.isfinite(hits + misses) and hits + misses else 0)
        print(f"{parity(run, reference):>9} {str(run.path in frontier):>6} "
              f"{run.ways:>4} {run.group:>5} {run.cache_gb:>8.3f} "
              f"{1e6/wall:>7.3f} {run.decode_s:>8.3f} {run.e2e_s:>7.3f} "
              f"{run.percentile('wall_us', .50)/1e3:>7.1f} "
              f"{run.percentile('wall_us', .95)/1e3:>7.1f} "
              f"{run.mean('gpu_us')/1e3:>7.1f} {100*run.mean('gpu_us')/wall:>7.2f} "
              f"{run.mean('io_wait_us')/1e3:>9.1f} "
              f"{100*run.mean('io_wait_us')/wall:>10.2f} "
              f"{run.mean('nvme_bytes')/1e9:>7.3f} {run.mean('nvme_gbps'):>10.3f} "
              f"{cache_hit:>8.2f}% {run.mean('kv_bytes')/1e3:>7.1f} "
              f"{run.mean('available_delta')/1e6:>8.1f} "
              f"{run.mean('cmdbufs'):>7.1f} {run.mean('dispatches'):>10.1f} "
              f"{run.mean('allocations'):>11.1f} {run.mean('pageouts'):>8.1f} "
              f"{run.mean('compressions'):>8.1f} {run.mean('decompressions'):>10.1f} "
              f"{run.mean('swapouts'):>7.1f} "
              f"{run.mean0('prefetch_bytes')/1e9:>11.3f} "
              f"{run.mean0('prefetch_useful_bytes')/1e9:>9.3f} "
              f"{run.mean0('prefetch_false_bytes')/1e9:>8.3f} "
              f"{run.mean0('prefetch_drain_us')/1e3:>8.3f} "
              f"{run.path}")

    feasible = [run for run in accepted if run.cache_gb <= args.max_cache_gb]
    if feasible:
        winner = min(feasible, key=lambda item: item.step_wall_us())
        print(f"\nfastest parity-gated configuration: {winner.path} "
              f"ways={winner.ways} group={winner.group} "
              f"token/s={1e6/winner.step_wall_us():.3f}")


if __name__ == "__main__":
    main()
