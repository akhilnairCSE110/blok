# blok

Key Insights - You should stay true to these insights regarding the architecture. If these principles are followed, lower-level errors and failure paths won't distract you. When you stray off, you'll immediately start doing BS, or running into weird errors.
1. Transformer inference is fully deterministic. Every neuron, weight, activation, and calculation is known at compile-time. Algorithms, namely greedy and others, are designed for traditional computing to deal with non-deterministic algorithms. These are NON-OPTIMAL for LLMs/Transformers, which are fully deterministic static pipelines.

2. The true-work in inference IS: The multiplication/dot-product of user-data with the LLM weights. In usual pipelines, MOST of the work is done moving the LLM weights back and forth. This is a contradiction that should be optimized AROUND. Using our other intutions, we must fuse operations, predict data movement, and compute large saturated batched work that drastically reduces the movement of the weights, and maximizes the multiplication of them (the named important work).

3. All three axis: input tokens, output tokens, and model size, will ALL grow. The pipeline must be optimized for all of them. i.e. not just a large model-weight size, but large KV.

4. ALL  data (KV cache, prompts, model weights, etc) should be stored in the NVME, not memory. LLM's and their are fundamentally far too large to store in memory. Furthermore, moving model weights in and out of memory is an inefficient and power hungry process anyway. The winning strategy is a first principles one: The LLM weights are already stored in the NVME flash, the input prompt is already stored in the NVME flash, and the KV cache will be stored in the NVME flash. Using linear optimization, you can then schedule the operations (i.e. deterministic and compute bound via Principle 1 and 2) using the GPU as a dense matmul engine, achieving high utilization.
Importantly, the GPU is programmed to be massively compute bound and utilized, at the tradeoff of up-front compile-time scheduling (spent as long as needed to find the most optimal way to run the pipeline), and the system and GPU memories are used as SCRATCHPADS (lpddr and vram are not ignored in this sense, but are used in conjunction with smart scheduling so the GPU is always saturated).

5. Memory (GPU vram, system lpddr), should be used to assist as a scratchpad to assist us in our scheduled pipelines from NVMe to GPU. We will be writing the data directly from NVMe to GPU via the GPU DMA and GDS (GPU Direct Storage). However, there may be some cases where memory is required. In this case, it is used as a scratchpad. It is important to note, it will NEVER be big enough to fit into memory. It will always be huge chunks, and heavy tiling will always be needed when we use memory.

6. NVMe SSDs are massively successful at large, sequential, batched reads, as compared to random reads. People think AI requires random reads, but due to it's determinism, it does NOT.  You can intelligently store model weights and pipelines on the SSD to get the performance you want. Apple wrote an important paper noting this entitled "LLM in a Flash".

7. NVMe SSDs can be configured in various RAID configurations (RAID 0, RAID 1), to achieve increases in parallelism and concurrency.

8. Rather than using CUDA, we can use PTX to optimize around parts that are not optimal for what we want. The inline asm command can be used to write inline PTX within CUDA, which can skip some annoying bulk work.

9. The GPU has enough compute. The SSD has enough storage. I want a deterministic pipeline that schedules everything un-front, and GETS it running. The only bottleneck should essentially be speed which we can figure out later. It should never be such that the system crashes, that implies a fundamental misundestanding of the architecture we are trying to design. The key idea is, limit memory movement and GET IT RUNNING. it shouldnt be that just because people can't trivial load it ALL into memoty and run it, that is just impossible. Thats just plain lazy. I want it to RUN, safely, cleanly, optimally, scheduled, NOT randomly. 

10. Code Bloat RUINS codebases. Whenever you add code, you should be looking to remove some. Always do this intelligently. I won't give an exact ultimatum as to how to do this, but TIGHT, OPTIMIZED codebases win. Every single line of code should be working for us, and our product should be a real-conglobration of code. Each edit you make should be very well thought out.
