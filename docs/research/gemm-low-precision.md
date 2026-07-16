# GEMM And Low Precision

Current Blok priority is Kimi INT4 routed expert parity, not new quantization.

## Kernel References

- DeepGEMM: DeepSeek FP8 dense/grouped GEMMs with fine-grained scaling and JIT.
  - https://github.com/deepseek-ai/DeepGEMM
- CUTLASS 3.x: CuTe layouts, pipelined GEMM, EVT epilogues, Hopper schedules.
  - https://github.com/NVIDIA/cutlass
  - https://cutlass.readthedocs.io

## Quantization References

- FP8 workflows: E4M3/E5M2, scaling, Transformer Engine-style stability.
  - https://arxiv.org/abs/2309.16919
  - https://developer.nvidia.com/blog/nvidia-blackwell-unlocks-low-precision-floating-point/
- SmoothQuant/AWQ/GPTQ/AQLM/PV-Tuning: calibration and low-bit weight compression references.
  - https://github.com/mit-han-lab/smoothquant
  - https://arxiv.org/abs/2306.00978
  - https://github.com/Vahe1994/AQLM
  - https://arxiv.org/abs/2405.14852
- QUICK/SageAttention2: locality-aware quantized kernels and low-bit attention.
  - https://github.com/thu-ml/SageAttention
