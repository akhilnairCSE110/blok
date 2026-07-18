# Hardware Scaling Context

Not direct targets. Use only for architectural context.

- NVIDIA Hopper/Blackwell: TMA/WGMMA, Tensor Core, FP8/FP4, NVLink, multi-die context. Blok target is RTX 5060 Ti `sm_120`, so check compatibility.
  - https://docs.nvidia.com/cuda/hopper-tuning-guide
  - https://developer.nvidia.com/blackwell
- AMD MI300X/ROCm: large-HBM and ROCm context, not this target box.
  - https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html
  - https://github.com/ROCm
- CXL, PIM, wafer-scale, computational storage: long-term memory hierarchy context.
  - https://www.computeexpresslink.org
- CPU/edge references: useful for explicit memory movement, but Ryzen 9 5950X is not the inference hot path.
  - https://arxiv.org/abs/2312.11514
  - https://github.com/ggerganov/llama.cpp
