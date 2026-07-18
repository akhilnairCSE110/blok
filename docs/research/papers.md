# Research Index

This directory tracks papers, systems, and kernel libraries relevant to Blok's target:

```text
RTX 5060 Ti 16GB sm_120 + Ryzen 9 5950X + Samsung 990 EVO Plus 1TB NVMe + uGDS
```

The references are design context, not proof that an optimization is already implemented. Hopper, Blackwell, AMD, wafer-scale, and distributed-system ideas must be translated carefully before applying them to the RTX 5060 Ti target.

## Files

- [Attention And Positional Encoding](attention.md)
- [GEMM And Low Precision](gemm-low-precision.md)
- [Kernel Libraries And DSLs](kernels-dsls.md)
- [KV Cache And Storage I/O](kv-io.md)
- [Kimi K2.6 Runtime Notes](kimi-k26-runtime-notes.md)
- [MoE And Runtime Systems](moe-runtime.md)
- [Hardware Scaling Context](hardware-scaling.md)

## Immediate Relevance To Blok

- Verify Kimi MLA numerics before performance work.
- Keep uGDS as the direct NVMe-to-GPU path.
- Use the RTX 5060 Ti `sm_120` target as the constraint for CUDA codegen and tuning.
- Avoid copying Hopper-only TMA/WGMMA assumptions into RTX 5060 Ti kernels without checking hardware support.
- Treat storage/KV-cache research as guidance for explicit ranges, alignment, reuse, and prefetching.
