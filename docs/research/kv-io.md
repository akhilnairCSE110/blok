# KV Cache And Storage I/O

Blok's product path is uGDS from the Samsung 990 EVO Plus NVMe into registered CUDA buffers.

## Direct Storage

- uGDS local reference:
  - ../../sub_dir/uGDS/README.md
  - ../../sub_dir/uGDS/docs/installation.md
- NVIDIA GPUDirect Storage:
  - https://docs.nvidia.com/gpudirect-storage/overview-guide/index.html
- BaM GPU-initiated storage:
  - https://arxiv.org/pdf/2203.04910.pdf

## KV Layout And Serving

- PagedAttention/vLLM: KV block tables and serving cache policy.
  - https://dl.acm.org/doi/10.1145/3600006.3613165
  - https://docs.vllm.ai/en/v0.6.0/automatic_prefix_caching/details.html
  - https://github.com/vllm-project/vllm
- vAttention/FlashInfer: virtual-contiguous KV and paged/ragged attention kernels.
  - https://arxiv.org/abs/2405.04437
  - https://github.com/flashinfer-ai/flashinfer
- SGLang RadixAttention: prefix tree KV reuse.
  - https://github.com/sgl-project/sglang

## Offload And Disaggregation

- DistServe, Sarathi-Serve, Mooncake/Kimi, LMCache, InfiniGen, FlexGen.
  - https://arxiv.org/pdf/2401.09670.pdf
  - https://arxiv.org/pdf/2403.02310.pdf
  - https://www.usenix.org/conference/fast25/presentation/qin
  - https://github.com/kvcache-ai/Mooncake
  - https://github.com/LMCache/LMCache
  - https://github.com/snu-comparch/InfiniGen/blob/main/README.md
  - https://arxiv.org/abs/2303.06865
- LLM in a Flash and ZeRO-Infinity: storage hierarchy context.
  - https://arxiv.org/abs/2312.11514
  - https://arxiv.org/pdf/2104.07857.pdf

## Adjacent I/O

- io_uring, SPDK, ZNS are fallback/adjacent references; Blok should not hide CPU/page-cache fallback.
  - https://github.com/axboe/liburing
  - https://spdk.io
  - https://zonedstorage.io
