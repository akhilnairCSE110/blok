# MoE And Runtime Systems

Kimi work should first prove tokenizer, MLA, routed INT4, and first-logit parity.

## MoE Architecture

- DeepSeek-V3/MLA: closest architectural reference class for Kimi-style MLA/MoE.
  - https://arxiv.org/abs/2412.19437
  - https://arxiv.org/html/2502.14837v1
  - https://platform.deepseek.com/docs
- Qwen3 MoE: another modern sparse activation reference.
  - https://arxiv.org/pdf/2505.09388.pdf
- MegaBlocks: block-sparse MoE kernels without token dropping/padding waste.
  - https://arxiv.org/abs/2304.05958

## Serving And Parallelism

- DistServe, Sarathi-Serve, Mooncake, vLLM, SGLang: request scheduling, disaggregation, prefix reuse.
  - https://arxiv.org/pdf/2401.09670.pdf
  - https://arxiv.org/pdf/2403.02310.pdf
  - https://www.usenix.org/conference/fast25/presentation/qin
  - https://github.com/vllm-project/vllm
  - https://github.com/sgl-project/sglang
- Megatron-LM/Core: tensor, pipeline, context, expert, and data parallelism.
  - https://github.com/nvidia/megatron-lm
  - https://docs.nvidia.com/nemo/megatron-bridge/0.2.0/parallelisms.html
- Punica/SGMV: segmented gather matvec for variable adapter workloads.
  - https://arxiv.org/pdf/2310.18547.pdf
- DualPipe/MegaScale-Infer: large MoE pipeline and expert-parallel context.
  - https://github.com/deepseek-ai/DualPipe
  - https://arxiv.org/abs/2504.02263
