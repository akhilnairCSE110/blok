# Attention And Position

Use these as design references only; Kimi correctness still requires official token/logit/layer parity.

## Core Attention Kernels

- FlashAttention-3: Hopper warp specialization, ping-pong scheduling, GEMM/softmax overlap, FP8. Relevant for rigorous attention scheduling, but Hopper TMA/WGMMA assumptions need RTX 5060 Ti review.
  - https://proceedings.neurips.cc/paper_files/paper/2024/file/7ede97c3e082c6df10a8d6103a2eebd2-Paper-Conference.pdf
  - https://github.com/Dao-AILab/flash-attention
- FlexAttention: compiler-generated custom attention with block masks and score mods.
  - https://pytorch.org/docs/stable/nn.attention.flex_attention.html
- AdaSplash: Triton alpha-entmax sparse attention.
  - https://arxiv.org

## KV Reduction And Sparse Context

- GQA: shares KV heads across query groups; useful background for KV bandwidth.
  - https://arxiv.org/abs/2305.13245
- Quest: sparse KV page selection from query/key range scores.
  - https://arxiv.org/abs/2406.10774
- MInference: million-token prefill through dynamic sparse head patterns.
  - https://www.microsoft.com/en-us/research/project/minference-million-tokens-prompt-inference-for-long-context-llms/
- StreamingLLM: attention sinks for stable streaming windows.
  - https://arxiv.org/pdf/2309.17453.pdf
- Infini-attention: compressive memory for long context.
  - https://arxiv.org/abs/2404.07143

## Kimi-Relevant Position And MLA

- YaRN: RoPE scaling; current executor still lacks exact YaRN parity.
  - https://arxiv.org/abs/2309.00071
- DeepSeek MLA analysis: most relevant background for MLA trace design.
  - https://arxiv.org/html/2502.14837v1
- Mamba/Mamba-2: linear-time sequence alternatives, not Kimi compatibility paths.
  - https://arxiv.org/abs/2312.00752
  - https://github.com/state-spaces/mamba
