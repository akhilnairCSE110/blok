# Samsung PM9C1a / 990 EVO Plus NVMe Plan

NVMe is the durable model and spill tier. Blok should read large, aligned, predictable ranges and
avoid small random I/O unless a sparse report proves it wins end to end.

## Requirements To Question

- Is the device root, model, benchmark, or uGDS-owned?
- Is a read path direct, staged, or fallback?
- Does the layout produce contiguous ranges for the current graph?
- Is a dedicated non-root NVMe available before uGDS is treated as a release gate?

## Delete / Simplify / Optimize / Automate

- Delete blind random reads from the hot path.
- Simplify first storage to uGDS/GDS for payload movement and ordinary-file `O_DIRECT` only as a
  fallback/probe path.
- Optimize with Blok sidecar layout, row-column bundling, and raw uGDS ownership on a dedicated
  non-root device.
- Automate refusal of destructive writes on root devices.

## Ownership Roles

- `root`: mounted OS/development disk, never rebound to uGDS.
- `model`: source checkpoints, Blok sidecar layouts, prompt artifacts, and KV spill files.
- `benchmark`: disposable bandwidth and write-test target.
- `ugds`: dedicated device owned by `ugds_drv`, not mounted as a filesystem.

## Gate

H1 reports baseline sequential bandwidth. H2 proves byte-exact NVMe-to-GPU movement through uGDS
without compatibility bounce; GDS is the compatibility comparison, not the preferred win. H3 proves
aligned `io_uring` plus `O_DIRECT` fallback.

## Sources

- uGDS: https://github.com/ScaleX-IO/uGDS
- NVIDIA GDS: https://docs.nvidia.com/gpudirect-storage/overview-guide/index.html
- LLM in a flash: https://arxiv.org/abs/2312.11514
