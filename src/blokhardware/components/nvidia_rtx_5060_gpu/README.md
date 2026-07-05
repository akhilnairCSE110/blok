# NVIDIA RTX 5060 Ti GPU Plan

The GPU should spend its hot path on dense math and fused kernels. Product name is advisory;
compute capability, driver state, CUDA runtime, and measured behavior decide enabled paths.

## Requirements To Question

- Does the GPU probe as `sm_120` with usable CUDA runtime support?
- Is bf16 supported and correct for the current kernel contract?
- Is a custom kernel required, or can cuBLASLt, CUDA graphs, CUDA events, or an attention library
  express the operation?
- Is the path compute-bound after transfer overlap, or only busy?

## Delete / Simplify / Optimize / Automate

- Delete product-string dispatch.
- Simplify first compute to CUDA init plus an optimized `sm_120` PTX byte-probe fed by the first
  uGDS/GDS transfer descriptor; then add bf16 cuBLASLt GEMM.
- Optimize with fused epilogues, FlashAttention-compatible kernels, CUDA graphs, and custom PTX
  only where the report names launch, memory, or fusion overhead.
- Automate checks for f16 misuse, hot-path `stream.synchronize()`, and missing event ordering.

## First Responsibilities

- CUDA context and capability probe;
- PTX module load for `blok_byte_probe` targeting `sm_120`;
- CUDA event timing for the first byte-probe kernel;
- stream and event setup;
- cuBLASLt bf16 correctness report;
- double-buffer overlap report;
- CUDA graph decode skeleton.

## Gate

C1 first reports byte-probe correctness and achieved device bandwidth, then bf16 GEMM GB/s or
TFLOPs. C2 proves transfer N+1 overlaps compute N. C3 captures one decode-step skeleton through a
single graph launch path.

## Sources

- CUDA programming guide: https://docs.nvidia.com/cuda/cuda-programming-guide/index.html
- NVIDIA GDS: https://docs.nvidia.com/gpudirect-storage/overview-guide/index.html
- FlashAttention: https://arxiv.org/abs/2205.14135
