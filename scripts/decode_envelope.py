#!/usr/bin/env python3
"""Compute the measured SSD/UMA/GPU throughput envelope from a decode log."""
import argparse
import re
from statistics import fmean

METRICS = re.compile(r"metrics .*?wall_us=(\d+) gpu_us=(\d+) .*?model_bytes=(\d+) .*?nvme_bytes=(\d+)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("log")
    p.add_argument("--target", type=float, default=10.0, help="target tokens/s")
    p.add_argument("--ssd-gbps", type=float, required=True)
    p.add_argument("--buffer-gb", type=float, default=25.0)
    a = p.parse_args()
    rows = []
    for line in open(a.log, errors="replace"):
        m = METRICS.search(line)
        if m:
            wall, gpu, model, nvme = map(int, m.groups())
            rows.append((wall, gpu, model, nvme))
    if not rows:
        raise SystemExit("no metrics lines found")
    wall = fmean(x[0] for x in rows) / 1e6
    gpu = fmean(x[1] for x in rows) / 1e6
    model = fmean(x[2] for x in rows) / 1e9
    nvme = fmean(x[3] for x in rows) / 1e9
    gpu_q = 1.0 / gpu if gpu else 0.0
    ssd_q = a.ssd_gbps / nvme if nvme else float("inf")
    required_hit = max(0.0, 1.0 - a.ssd_gbps / (a.target * nvme))
    deficit = max(0.0, a.target * nvme - a.ssd_gbps)
    drain = a.buffer_gb / deficit if deficit else float("inf")
    print(f"samples={len(rows)} wall_ms={wall*1e3:.3f} gpu_ms={gpu*1e3:.3f}")
    print(f"model_GB/token={model:.6f} nvme_GB/token={nvme:.6f}")
    print(f"gpu_ceiling_tok_s={gpu_q:.6f} ssd_ceiling_tok_s={ssd_q:.6f}")
    print(f"target={a.target:.3f} required_byte_hit={required_hit*100:.3f}%")
    print(f"buffer_drain_s={drain:.6f}")
    print(f"target_feasible={gpu_q >= a.target and ssd_q >= a.target}")


if __name__ == "__main__":
    main()
