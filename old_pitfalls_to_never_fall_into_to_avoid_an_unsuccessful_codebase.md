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
- Do not resurrect ARC's multi-crate shape. Useful ARC ideas must be converted into small typed
  descriptors in the current crate: safetensors header records, transfer windows, arena views,
  stream/event dependencies, KV page descriptors, and token-output gates.
- Do not carry ARC's final `mmap` fallback forward. It is acceptable only as a negative fixture or
  explicit unsupported-path error; payload success requires direct I/O, GDS, or uGDS reporting.
- Do not treat `mlock` failure as a warning on a path that depends on pinned host memory. If pinned
  memory is required for async behavior, failure is a capability error or the report must mark a
  deliberately staged backend.
- Do not let "preflight" names hide fake execution. A direct-read preflight is an I/O capability
  check, not D1, not GDS, and not a token path.
- Do not let tests download gated models unless authentication has already passed. Gated download
  commands must fail before resuming transfer when the Hugging Face token is absent or invalid.

## Useful Material To Preserve From ARC

- Keep the progressive validation ladder: tiny model first, enforce a minimum generated-token count,
  and fail logs that never prove real token emission.
- Keep safetensors header parsing discipline: file path, header length, data base offset, tensor
  dtype, shape, absolute offset, and byte length are metadata descriptors, not payload reads.
- Keep direct handle reporting: file size, alignment, backend, and registered-handle status belong
  in reports before any transfer is trusted.
- Keep the schedule split between transfer, dense compute, and attention/KV work. Express it as
  Blok descriptors first; only later wire CUDA streams and events.
- Keep arena high-water thinking, but size arenas from graph descriptors and measured device limits,
  not from fixed giant allocations.
- Keep storage safety rules for any future raw NVMe path: mounted devices and filesystem signatures
  are hard stops for writes.

## Current Rule

Every new line must do one of these:

- make the Kimi sidecar manifest more correct;
- make direct I/O or CUDA execution real;
- make the first generated token closer;
- remove code, branches, or claims that can mislead us.

Anything else waits.
