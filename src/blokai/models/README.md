# Target Model Plan

Target models are the user-visible checkpoints Blok serves. They live on NVMe and are organized for
predictable local loading, caching, and execution.

## Requirements To Question

- Is the first forcing checkpoint DeepSeek R1 GGUF, Hugging Face safetensors, or a smaller MLA/MoE
  correctness ladder?
- Which metadata is required before any payload transfer?
- Which quantized dtypes are acceptable for first success?
- Must source checkpoint files remain untouched, or can Blok write a repacked physical layout?

## Delete / Simplify / Optimize / Automate

- Delete source-format quirks from runtime modules; normalize them into `Manifest`.
- Simplify initial support to the smallest model ladder that exercises the same path as the huge
  MoE target.
- Optimize physical layout only after sidecar indexes prove which reads are fragmented.
- Automate manifest validation, alignment checks, dtype checks, and checksum policy.

## Planned Data Contract

- architecture family: dense, GQA, MoE, MLA, or hybrid;
- tensor name, role, dtype, quantization, shape, source offset, size, and alignment;
- tokenizer metadata and prompt roundtrip expectations;
- KV rules, including MLA latent cache and decoupled RoPE cache separation;
- layout descriptor for source order, sidecar order, or repacked Blok order.

## Gate

`M1` succeeds only when all tensors are typed, aligned, ranged, and consumable through one manifest
API independent of source format.

## Sources

- FlexGen: https://arxiv.org/abs/2303.06865
- LLM in a flash: https://arxiv.org/abs/2312.11514
