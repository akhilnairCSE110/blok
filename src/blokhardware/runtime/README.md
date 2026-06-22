# Runtime Scheduling Plan

The runtime schedules deterministic inference as a storage-to-compute graph. It exists to keep the
GPU doing useful math while SSD and CPU work are prepared ahead of demand.

## Requirements To Question

- Which work is truly data-dependent?
- Which work can be staged before the current token finishes?
- Which operation is memory-bound, launch-bound, storage-bound, or compute-bound?
- Which fusion removes real overhead instead of hiding complexity?

## Delete / Simplify / Optimize / Automate

- Delete runtime paths that discover payload bytes during execution.
- Simplify execution into typed descriptors: source range, destination arena, consumer op,
  dependencies, lifetime, and report fields.
- Optimize with fusion, prefetch, chunked prefill, continuous batching, and CUDA graphs only after
  descriptor reports identify the bottleneck.
- Automate regression checks for payload `mmap`, hidden page-cache reads, hot-path allocation, and
  accidental synchronization.

## Scheduler Contract

Each scheduled operation declares:

- layer, request, and token scope;
- source tensor or KV page ranges;
- destination arena view;
- consumer kernel or library call;
- dependency edges;
- release action;
- report fields.

## Gate

D1 proves one tensor fetch to one kernel result. D2 proves one full layer with stalls explained. D3
proves one full forward step without OOM, hidden cache, or unknown byte access.

## Sources

- FlexGen: https://arxiv.org/abs/2303.06865
- FlashAttention: https://arxiv.org/abs/2205.14135
- PagedAttention: https://arxiv.org/abs/2309.06180
