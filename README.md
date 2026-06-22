# Blok

Blok is an SSD-resident inference runtime. Model weights, prompts, KV state, auxiliary runtime
checkpoints, layout indexes, and benchmark artifacts live on NVMe by default. GPU VRAM and system
RAM are scratchpads for a deterministic execution schedule.

The central bet is simple: transformer inference is deterministic enough to schedule payload
movement before compute needs it. Blok should store tensors in flash-friendly layouts, move only the
bytes required by the current graph, use sparsity where it reduces total wall time, fuse operations
aggressively, and keep the GPU compute-bound instead of depending on whole-model memory residency.

The first target is:

```sh
blok generate --model <huge-moe-model> --prompt "Hi" --tokens 1
```

That command must emit one token on the Linux target with measured NVMe-to-VRAM bytes, kernel time,
memory use, and no hidden payload `mmap` or page-cache path.

## Source of Truth

- [Plan index](docs/plan-index.md): granular subsystem plans and the development method.
- [Golden implementation plan](docs/06_21_plan.md): target, invariants, research notes, milestone
  gates, and implementation order.
- [System requirements](docs/system-requirements.md): Ubuntu, CUDA, uGDS, NVMe, and build-tool
  prerequisites.
- [scripts/ci.sh](scripts/ci.sh): local and CI command surface used by `just`.

## Development State

This repository is intentionally pre-code. The current task is to iterate the granular plans until
the requirements are questioned, simplified, sourced, and measurable. The next code pass creates one
Rust crate named `blok` plus `xtask`, preserving the current `just` and `scripts/ci.sh` entry
points.

Do not add passive placeholder modules. Every new source file must define a real type, command,
report, parser, probe, or test used by the current milestone.

## Core References

- uGDS: https://github.com/ScaleX-IO/uGDS
- NVIDIA GDS: https://docs.nvidia.com/gpudirect-storage/overview-guide/index.html
- FlexGen: https://arxiv.org/abs/2303.06865
- LLM in a flash: https://arxiv.org/abs/2312.11514
- FlashAttention: https://arxiv.org/abs/2205.14135
- PagedAttention: https://arxiv.org/abs/2309.06180
