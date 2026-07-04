# Old Pitfalls To Never Fall Into

This file exists because previous Blok/ARC attempts got too broad before the runtime could emit a
real token through the target Linux path. Keep it short and enforce it in code review.

## Never Repeat These

- Do not create many crates, platforms, or parallel runtimes. One `blok` crate plus `xtask` until a
  measured boundary proves otherwise.
- Do not build Apple/Metal paths while the target is Linux + NVIDIA + NVMe. They dilute the only
  path that matters right now.
- Do not claim "ready" because metadata parsed or a plan was written. Ready means the next command
  can run the measured target path.
- Do not print fake token output or accept syntactically generated text as inference success. A run
  is only successful when tokenizer, weights, kernels, sampling, and reports all match the model
  path.
- Do not use payload `mmap` for streamed weights. Metadata-only reads are fine; payload movement
  must be explicit and reported.
- Do not silently fall back to buffered/page-cache reads, CPU bounce buffers, or GDS compatibility
  paths. Abort unless an explicit opt-in flag says otherwise, and report the backend.
- Do not allocate huge pinned arenas up front. The old 16 GB pinned plan failed against a 537 MB
  memlock limit. Arena sizes must come from measured limits and current descriptors.
- Do not duplicate 595 GB of model files by default. Sidecar manifests and indexes come first;
  physical repack is opt-in and justified by a measured read problem.
- Do not trust product names for hardware dispatch. Use probed compute capability, driver state,
  device nodes, PCIe, NVMe role, and CUDA availability.
- Do not add optimizers, speculation, prediction, cache eviction, or schedulers before the first
  direct byte-to-kernel path works.
- Do not let reports contain predictions as if they are measurements. Predictions may exist only as
  clearly labeled estimates.
- Do not hide incomplete downloads behind layout reports. Status must show bytes and safetensor
  counts, and materialization must reject incomplete sources.
- Do not let default paths split state between `~/.blok` and the repo-local `.blok`. Use the active
  Blok home consistently.

## Current Rule

Every new line must do one of these:

- make the Kimi sidecar manifest more correct;
- make direct I/O or CUDA execution real;
- make the first generated token closer;
- remove code, branches, or claims that can mislead us.

Anything else waits.
