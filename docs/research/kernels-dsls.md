# Kernel Libraries And DSLs

Blok's executor is CUDA/C++ because uGDS is direct and explicit. These are references for future kernels.

- Liger-Kernel: HF-compatible Triton fusions for RMSNorm, RoPE, SwiGLU, CE.
  - https://github.com/linkedin/Liger-Kernel
- ThunderKittens: warp/block/grid tile abstractions for hand-written high-utilization kernels.
  - https://arxiv.org/abs/2410.20399
  - https://github.com/HazyResearch/ThunderKittens
- Triton/Tile IR: Pythonic kernels and tile-preserving compiler lowering.
  - https://github.com/triton-lang/triton
  - https://triton-lang.org
- Tawa: automated warp specialization from Triton.
  - https://arxiv.org
- CuAsmRL: RL optimization of generated GPU SASS; only relevant after profiling.
  - https://arxiv.org
- Model2Kernel: model-aware CUDA memory-safety verification.
  - https://arxiv.org/abs/2603.24595
