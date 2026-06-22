# blok

Key architecture insights:
1. Transformer inference is deterministic at the graph and tensor-operation level. Runtime work
   should exploit that determinism instead of treating inference like an unpredictable control-flow
   problem.

2. The useful work in inference is multiplication of activations against model weights. The runtime
   should fuse operations, predict data movement, and create large saturated batches that maximize
   math while minimizing weight movement.

3. Input tokens, output tokens, and model size will all grow. The pipeline must optimize for prompt
   length, generation length, weight size, and KV cache size together.

4. Model data (KV cache, prompts, model weights, etc.) should be stored on NVMe by default, not
   assumed to fit in memory. The winning strategy is first-principles scheduling: model weights are
   already stored in NVMe flash, prompts can be staged there, and the KV cache can spill there.
   Linear optimization should schedule tensor placement and movement across GPU, CPU, and disk, as
   demonstrated by FlexGen.
   The GPU should stay compute-bound through up-front scheduling, and system memory plus VRAM should
   act as scratchpad capacity that supports the NVMe-backed execution plan.

5. Memory (GPU VRAM, system LPDDR/DRAM) should be scratchpad capacity for scheduled NVMe-to-GPU
   pipelines. The preferred GDS path is the open-source uGDS stack from ScaleX-IO, which provides a
   user-space GPU Direct Storage path where the CPU constructs NVMe commands and the SSD DMAs
   directly to and from GPU memory. NVIDIA GDS remains useful as a compatibility baseline, but the
   repo target is uGDS where the hardware and kernel driver state support it.

6. NVMe SSDs are strongest at large, sequential, batched reads compared with random reads. AI
   serving should not blindly generate random IO; deterministic inference lets us lay out model
   weights and runtime pipelines for predictable flash access. Apple's "LLM in a flash" paper is the
   reference for reducing transferred data and reading larger contiguous chunks from flash.

7. NVMe SSDs can use RAID-style layouts when they provide useful parallelism, but the first
   requirement is predictable device ownership and measurable sequential bandwidth.

8. CUDA is the default integration layer. PTX and inline PTX are reserved for narrow hot paths where
   generated code blocks the execution model we need.

9. The GPU has enough compute and the SSD has enough storage. The runtime should schedule work up
   front, limit memory movement, and run safely instead of depending on whole-model memory residency.

10. Keep the codebase tight. Whenever code is added, look for stale or redundant lines to remove.
    Every line should carry architectural or product weight.

Core references:
- uGDS: https://github.com/ScaleX-IO/uGDS
- FlexGen: https://arxiv.org/abs/2303.06865
- LLM in a flash: https://arxiv.org/abs/2312.11514
