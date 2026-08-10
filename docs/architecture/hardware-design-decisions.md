# Blok Hardware Design Decisions

## What is being built, why each choice exists, and how to test it

**Date:** 2026-08-10

**Target machine:** Ryzen 9 5950X, 48 GB host DRAM, RTX 5060 Ti 16 GB, Samsung 990 EVO Plus 1 TB, CUDA 12.8+, Linux, uGDS

**Scope:** batch-one text inference with Kimi K2.6, plus a concrete GLM-5.2 FP8 port specification

## The idea in plain technical language

Parameter count is not the same as the number of weight bytes required by one token. A modern MoE computes a small router first; that router makes all but eight experts mathematically irrelevant for the current layer and token. Blok keeps that selection computation resident, converts its eight IDs directly into storage addresses, and fetches only the selected records. It then applies the same rule to representation and layout: keep weights compressed until multiplication, store attention state as the model's latent rather than expanded per-head K/V, and place values together when the model always consumes them together.

The hardware question is therefore specific: how much time and capacity are required to turn a small, serial selection result into the exact data needed by the next computation? The answer is determined by selected bytes, command count, queue depth, cache capacity, and the latency of the selection-to-command path. The decisions below quantify each term.

## 1. The claims

There are three main claims. Everything else in this document supports or limits them.

| Claim | Kimi K2.6 | GLM-5.2 FP8 | Why it matters |
|---|---:|---:|---|
| Fetching only router-selected experts reduces routed-weight bytes exactly | 48× | 32× | This is the main reason the models are executable from one SSD. |
| The whole cold weight step is smaller than the checkpoint | 18.04× | 18.26× | Kimi reads 32.986 GB from a 595.148 GB tensor payload; GLM reads 41.383 GB from 755.617 GB. |
| Caching MLA latents instead of expanded K/V makes the advertised context capacity-feasible | 71.11× smaller than the current FP32 cache | 56.89× smaller than expanded BF16 K/V | Without this decision, the model and full KV cache do not fit on the drive. |

These are byte-count results, not measured speedups. The cold weight bandwidth floors remain 4.61 seconds/token for Kimi and 5.79 seconds/token for GLM at the SSD's vendor-rated 7.15 GB/s. Transaction latency and KV traffic make real latency higher.

## 2. What exists today

The current Kimi executor proves the functional path but does not yet implement the performance architecture.

| Mechanism | Current state | Evidence |
|---|---|---|
| Exact Kimi tokenizer and chat formatting | Implemented | [`../../blok/runtime.py`](../../blok/runtime.py) |
| Exact top-8 routing and selected-expert execution | Implemented | [`../../src/kimi_exec.cu`](../../src/kimi_exec.cu) |
| Packed INT4 expert weights with BF16 group scales | Implemented | [`../../src/kimi_exec.cu`](../../src/kimi_exec.cu) |
| SSD-to-GPU peer DMA through uGDS | Implemented | [`../../src/kimi_exec.cu`](../../src/kimi_exec.cu), [`../../sub_dir/uGDS/README.md`](../../sub_dir/uGDS/README.md) |
| Physical extent validation and non-overlapping raw KV range | Implemented | [`../../scripts/plan_ugds_layout.py`](../../scripts/plan_ugds_layout.py) |
| Persistent registered I/O buffers | Not implemented | Every tensor load registers and deregisters a buffer. |
| Queued/batched model reads | Not implemented | The executor calls synchronous `uGDSRead`. |
| Expert-record storage layout | Not implemented | It reads rows from original safetensor shards. |
| Weight cache | Not implemented | Weights are discarded after each tile. |
| Latent MLA KV cache | Not implemented | The executor stores expanded FP32 K/V. |
| Blocked prefill | Not implemented | Prompt tokens run through the model one at a time. |
| GLM-5.2 executor | Not implemented | The current contract accepts Kimi only. |

This distinction is important: the repository has established correctness-oriented control and data paths. It has not established the final throughput claim.

### 2.1 Flash boundary

The target 990 EVO Plus is a managed TLC NAND NVMe SSD with Host Memory Buffer support; it is not NOR flash ([Samsung data sheet](https://download.semiconductor.samsung.com/resources/data-sheet/samsung_nvme_ssd_990_evo_plus_datasheet_rev.1.0.pdf)). NOR would be appropriate for controller boot code or a small immutable descriptor table, not a 595–756 GB weight payload. With the proposed affine expert address, even that table is unnecessary: `(layer, expert_id)` maps by base plus stride.

Blok's “raw” access bypasses the filesystem and host page cache. It does **not** bypass the SSD controller, its flash-translation layer, ECC, bad-block handling, or garbage collection. Likewise, the 4 KiB logical block and the controller page size used by uGDS are not claims about the NAND array's physical page size. The consumer drive does not expose its channel, die, plane, or physical-page mapping, so this work cannot claim a particular NAND placement.

The model-side mathematics supplies the flash controller with a precise access graph:

- fixed records are read in layer order;
- eight expert records become known only after each router;
- each selected expert's six payloads are always co-required;
- KV records are append-written and later read either sequentially or by selected history index.

On the present SSD, the implementation can only make these records contiguous in logical address space and measure the result. A custom flash controller could additionally stripe one record across channels, keep co-required pages on independent dies, prioritize immutable weight reads over KV writes, or accept `(layer, expert_id)` directly. Those are proposed flash implementations, not benefits already proved by the model equations. The proved benefits remain the byte reductions and capacity results in Sections 3, 4, 9, and 10.

## 3. Decision 1 — Route first, then fetch only selected experts

### Decision

Keep the router and its correction data resident. Compute the top-k expert IDs before issuing expert-weight reads. Never read an unselected expert for the current token unless it is an intentional cache prefetch.

### Alternative rejected

Streaming every expert in the layer, or predicting experts with an additional approximate model.

### Numerical benefit

Kimi has 384 routed experts and selects eight:

\[
\frac{384}{8}=48.
\]

Its complete routed bank is 570.761 GB, while one token's selected routed records total 11.891 GB.

GLM has 256 routed experts and selects eight:

\[
\frac{256}{8}=32.
\]

Its FP8 routed bank is 724.953 GB, while one token's selected routed records total 22.655 GB.

This reduction is exact after top-k selection because unselected experts have zero coefficient in the MoE output.

### Implementation

Kimi already computes the BF16 router, sigmoid, correction bias, top eight, normalized weights, and routed scale before reading experts. The current implementation then copies 16 bytes of expert IDs to the CPU with a blocking `cudaMemcpy` at each of 60 MoE layers.

The intended implementation is:

1. Keep all Kimi routers and biases resident: 330,347,520 bytes total.
2. Let the GPU write eight `uint16` expert IDs into a pinned, coherent command mailbox.
3. Derive each record address from a base, layer, and expert ID:

\[
O_{\ell,e}=O_0+(\ell E+e)S_{record}.
\]

4. Submit the eight reads together.
5. Start an expert kernel as each record slot becomes ready.

For GLM, router weights and biases total 235,968,000 bytes. Its full-indexer weights add 200,991,168 bytes. Both are selection-critical and should be resident before ordinary attention or expert weights.

### Hardware requirement

A selection-to-I/O path that accepts `(layer, expert_ids[8])` and produces storage commands without a blocking GPU-to-CPU copy. This can be a small CPU co-processor polling a coherent ring or a dedicated I/O command engine. It does not require the GPU to implement an NVMe driver.

### Falsification test

Count SSD bytes for routed experts. A cold Kimi token must read exactly 11,890,851,840 useful routed bytes; a cold GLM token must read 22,654,771,200. Compare logits and token IDs against the reference model. Any extra expert bytes must be attributed to alignment, cache prefetch, or an error.

## 4. Decision 2 — Keep weights compressed until the multiply

### Decision

DMA the checkpoint's native quantized representation into GPU memory and fuse scale application with the dot product. Do not expand weights to BF16 on SSD, in host DRAM, or in a persistent VRAM buffer.

### Alternative rejected

Dequantizing an entire matrix before execution.

### Numerical benefit

One Kimi expert contains three matrices. In BF16 they would occupy

\[
3(7168)(2048)(2)=88{,}080{,}384\ \text{bytes}.
\]

Packed INT4 plus BF16 group-32 scales occupies 24,772,608 bytes:

\[
\frac{88{,}080{,}384}{24{,}772{,}608}=3.556.
\]

One GLM expert would occupy 75,497,472 bytes in BF16. E4M3 weights plus F32 128×128 inverse-scale blocks occupy 37,757,952 bytes, almost exactly 2× smaller.

The decode weight arithmetic intensity remains low:

| Path | Weight FLOPs/token | Weight bytes/token | FLOPs/byte |
|---|---:|---:|---:|
| Kimi | 63.372 GFLOP | 32.986 GB | 1.921 |
| GLM base path | 80.596 GFLOP | 41.383 GB | 1.948 |

At 7.15 GB/s, the storage stream supplies work at only about 13.7–13.9 GFLOP/s. More matrix throughput alone cannot improve a cold batch-one step.

### Implementation

Kimi's `matvec_i4_bf16_scale_k` reads packed `uint32` words, extracts signed nibbles, applies BF16 group scales, and accumulates without creating a BF16 matrix. GLM requires an E4M3 kernel that applies F32 inverse scales per 128×128 block inside the matrix operation.

### Hardware requirement

- Signed INT4 dot products with one BF16 scale per 32 input weights.
- E4M3 dot products with one F32 inverse scale per 128×128 weight block.
- Scale and weight operands accepted directly from the I/O slab.
- Accumulation precise enough to reproduce reference logits within the agreed tolerance.

### Falsification test

Measure bytes entering VRAM and temporary VRAM allocation. There should be no full dequantized matrix. Compare kernel output against a dequantize-then-multiply reference for every tensor shape.

## 5. Decision 3 — Replace per-tensor allocation with persistent registered slabs

### Decision

Allocate a fixed GPU I/O arena at startup, align it to the GPU's 64 KiB peer-memory page, register it with uGDS once, and reuse it as a ring of non-overlapping DMA/compute slots.

### Alternative rejected

The current `cudaMalloc → uGDSBufRegister → read → uGDSBufDeregister → cudaFree` sequence for every tensor slice.

### Numerical benefit

For one sampled Kimi token, the source issues at least:

- 217,686 model-range registrations and deregistrations;
- about 217,047 GPU allocations and frees;
- additional registration cycles for two KV writes per layer and every KV read tile.

These counts follow from the loop bounds in the current executor. They are independent of SSD bandwidth and can dominate latency even when the payload is correct.

### Implementation

Use three classes of persistent slots:

1. fixed-weight stream slots;
2. eight or more expert-record slots;
3. KV read/write slots.

Each slot has four states: `free`, `DMA in flight`, `compute in flight`, and `reusable`. NVMe DMA and a GPU kernel must never access the same slot concurrently. The uGDS installation contract explicitly requires this separation.

The minimum safe design is double buffering. Triple buffering is useful when command preparation, DMA, and compute must all proceed independently.

### Hardware requirement

- A stable GPU physical mapping for the arena.
- Per-slot completion flags visible to the I/O submitter and GPU scheduler.
- Protection against reusing a slot before both DMA and compute complete.

### Falsification test

Steady-state registration, deregistration, allocation, and free counts must all be zero. A race test must alternate DMA and kernels on every slot and detect corruption with per-block checksums.

## 6. Decision 4 — Use queue depth, not synchronous reads, to reach SSD bandwidth

### Decision

Submit multiple NVMe commands per doorbell and keep enough commands in flight to cover device latency.

### Alternative rejected

One synchronous read at a time. The current executor uses this alternative.

### Numerical benefit

For command size \(s\), average completion latency \(L(s)\), queue depth \(Q\), media bandwidth \(B_m\), and link bandwidth \(B_l\),

\[
B_{eff}\le\min\left(B_m,B_l,\frac{Qs}{L(s)}\right).
\]

To target bandwidth \(B_*\), the required queue depth is

\[
Q_{sat}=\left\lceil\frac{B_*L(s)}{s}\right\rceil.
\]

At \(B_*=7.15\) GB/s:

| Command size | \(L=10\) µs | \(L=50\) µs | \(L=100\) µs |
|---:|---:|---:|---:|
| 4 KiB | 18 | 88 | 175 |
| 128 KiB | 1 | 3 | 6 |
| 1 MiB | 1 | 1 | 1 |

The uGDS project reports 5.2 µs for 4 KiB on an A100 and 990 PRO, not on this target. Even at that latency, saturating 7.15 GB/s with 4 KiB commands requires a queue depth of ten. A synchronous queue has depth one.

### Actual uGDS limits

The current library has:

- up to 128 application I/O entries per batch;
- a batch queue depth up to 512, subject to controller and hugepage limits;
- 64 PRP-list pages for commands spanning more than two controller pages;
- one batch queue pair per handle;
- one final doorbell after building a batch, with additional submit-and-drain waves if the submission queue or 64-page PRP pool fills;
- busy-polled completions.

The async stream API is a CUDA host callback that invokes the same synchronous I/O routine. It preserves stream ordering, but it is not a native deep NVMe queue. The batch API is the appropriate starting point for model reads.

### Command size is not the 4 MiB application cap

Blok currently refuses an application tensor slice larger than 4 MiB. uGDS may split that slice into smaller NVMe commands. For controller page size \(P\), one PRP-list page limits the current implementation to

\[
S_{PRP}=\left(\frac{P}{8}+1\right)P.
\]

At \(P=4096\),

\[
S_{PRP}=2{,}101{,}248\ \text{bytes}.
\]

The actual command cap is

\[
S_{cmd}=\min(S_{MDTS},S_{PRP}).
\]

If the controller reports no MDTS, this uGDS version falls back to 128 KiB. Therefore no paper should infer NVMe command count from Blok's 4 MiB application limit. The target must record `MDTS`, controller page size, and the resulting `S_cmd` at startup.

This produces concrete queue requirements. At the maximum one-list-page cap of 2,101,248 bytes, one Kimi expert record is split into 12 commands and one aligned GLM expert record into 18. At the 128 KiB fallback, the counts are 189 and 289. A batch of eight selected experts is therefore 96 or 144 commands at the larger cap, and 1,512 or 2,312 commands at the fallback. Since only 64 PRP-list pages exist, even the larger-cap case can require multiple submit-and-drain waves. Increasing queue depth alone does not remove this limit; the implementation also needs more PRP-list capacity or SGL support.

### Implementation

For each layer:

1. Build descriptors into a registered batch.
2. Sort independent expert reads by LBA.
3. Submit the queue; use additional doorbells only when command or PRP capacity forces another wave.
4. Poll completions and launch work per completed slot.
5. Refill the queue before it drains below \(Q_{sat}\).

### Hardware requirement

A command queue deep enough for the measured \(Q_{sat}\), completion delivery that does not serialize the GPU, and a command builder capable of translating record IDs to LBAs.

### Falsification test

Plot achieved bandwidth against command size and queue depth. The decision succeeds only if the measured curve approaches the lesser of SSD and PCIe bandwidth and the queue does not repeatedly empty.

## 7. Decision 5 — Store an expert as one addressable record

### Decision

Place a selected expert's gate weights, gate scales, up weights, up scales, down weights, and down scales in one contiguous, 4-KiB-aligned record. Order records by layer and expert ID.

### Alternative rejected

Keeping the six payloads in unrelated shard locations and reading them in row-sized fragments.

### Numerical benefit

One Kimi expert record is exactly 24,772,608 bytes and is already divisible by 4,096. The current executor performs 352 application tensor loads per selected expert. A record layout reduces this to one batch entry per expert:

\[
60\cdot8\cdot352=168{,}960
\quad\longrightarrow\quad
60\cdot8=480
\]

application descriptors per token.

This does **not** reduce the 11.891 GB of useful expert bytes, and it does not imply one NVMe command. uGDS splits each record into

\[
\left\lceil\frac{S_e}{S_{cmd}}\right\rceil
\]

commands. The advantage is fewer application calls, registrations, address lookups, extent boundaries, and doorbells; the NVMe-command advantage depends on MDTS and fragmentation.

One GLM expert is 37,757,952 bytes. Aligning it to 4 KiB adds 3,072 bytes. Across all 19,200 layer-experts, padding is only 58,982,400 bytes, or 0.008% of the 724.953 GB routed bank.

### Implementation

Use two model regions:

1. a deterministic fixed-weight stream in layer execution order;
2. an expert arena with address `base + (layer × experts + expert_id) × stride`.

The conversion must replace the source shard payload on the model SSD, not coexist with it. Kimi does not have capacity for both the 595 GB checkpoint and a second 571 GB expert bank; GLM has even less margin. Conversion therefore needs external temporary storage or a separately validated migration procedure.

### Hardware requirement

Only base-plus-stride address generation and batched large reads. No associative filesystem lookup is needed in the token loop.

### Falsification test

The useful bytes must remain unchanged. Record conversion must reproduce every original tensor byte and every reference logit. Device counters must show fewer submissions and extent crossings.

## 8. Decision 6 — Cache the serial dependencies before caching bulk weights

### Decision

Spend the first VRAM cache bytes on data that blocks discovery of later work, then on data that is always reused, then on experts selected by measured frequency.

### Alternative rejected

Caching the largest matrices first or using unweighted LRU over tensor tiles.

### Cache order

| Object | Kimi size | GLM size | Reason |
|---|---:|---:|---|
| Decoder norms | 2.013 MB | small | Always used and cheap to pin. |
| Routers and correction biases | 330.348 MB | 235.968 MB | Expert I/O cannot begin until these finish. |
| Full DSA indexer weights | not applicable | 200.991 MB | History selection cannot begin until these finish. |
| LM head | 2.349 GB | 1.903 GB | Used every sampled step; Kimi currently makes 640 head reads. |
| Maximum-context DSA index keys | not applicable | 5.637 GB | Otherwise scanned from SSD every generated token at 1M context. |
| Attention weights | 12.339 GB | 12.877 GB | Always used but may not fit with runtime state. |
| Expert records | 24.773 MB each | 37.758 MB each | Cache only after measuring layer/expert reuse. |

The cache value of object \(i\) is not just bytes saved. A simple score is

\[
v_i=p_i\left(\frac{s_i}{B}+r_iL_i\right)+c_i,
\]

where \(p_i\) is hit probability, \(r_i\) is avoided command count, and \(c_i\) is avoided critical-path time. Routers have a large \(c_i\) even though they are much smaller than attention weights.

### Implementation

- Pin norms, routers, and the head during initialization if the measured free VRAM budget permits.
- For GLM at long context, pin index keys before attention weights.
- Maintain cache tags at layer-expert granularity. Kimi needs 23,040 expert tags; GLM needs 19,200. The metadata is small.
- Record the selected expert IDs for every token and build a miss curve before choosing replacement policy.

### Host DRAM tier

Use host DRAM as a cache only when

\[
L_H+\frac{s}{B_H}<L_S+\frac{s}{B_S}.
\]

The 48 GB installed DRAM is not the cache budget; the OS, tokenizer, metadata, and pinned buffers consume part of it. Measure free pinned capacity and host-to-GPU bandwidth.

### Falsification test

For each cached class, report hit rate, bytes saved, commands saved, and critical-path time saved. Remove any class whose measured benefit per byte is lower than the next candidate.

## 9. Decision 7 — Cache MLA latents, not expanded per-head K/V

### Decision

Store the 512-element compressed latent and 64 rotary elements per layer/token. Reassociate the K and V projections around attention instead of materializing per-head K/V.

### Alternative rejected

The current Kimi cache: expanded FP32 keys and values. Also rejected for GLM: expanded BF16 K/V as used by the generic implementation.

### Numerical benefit

Kimi current:

\[
N_{expanded}=4{,}997{,}120\ \text{bytes/token}.
\]

Kimi BF16 latent:

\[
N_{latent}=61(512+64)(2)=70{,}272\ \text{bytes/token}.
\]

Reduction:

\[
\frac{4{,}997{,}120}{70{,}272}=71.11.
\]

At 262,144 tokens the cache falls from 1.310 TB to 18.421 GB.

GLM expanded BF16 K/V is 5,111,808 bytes/token. Its BF16 latent is 89,856 bytes/token, a 56.89× reduction. Adding 21 indexer-key streams produces 95,232 bytes/token and 99.858 GB at maximum context. The official FP8 model plus this cache fits on a nominal 1 TB device; expanded K/V does not.

### Mathematical mechanism

For cached latent \(c_j\),

\[
k_{j,h}=W_h^Kc_j,
\qquad
v_{j,h}=W_h^Vc_j.
\]

Then

\[
q_h^TW_h^Kc_j=(W_h^{K,T}q_h)^Tc_j
\]

and

\[
\sum_jp_{j,h}W_h^Vc_j
=W_h^V\left(\sum_jp_{j,h}c_j\right).
\]

This changes operation order but discards no model information.

### Implementation

1. Write `(latent[512], rotary[64])` once per layer and token.
2. Absorb the key projection into the query.
3. Use an online-softmax recurrence so each latent tile is read once.
4. Accumulate a latent value vector.
5. Apply the value projection after reduction.

The current Kimi code instead reads K three times and V once. This decision removes both expansion and redundant scans.

### Hardware requirement

A sparse/dense attention kernel that accepts latent records, keeps running max/sum/output state on chip, and applies the absorbed projections without writing expanded K/V to VRAM.

### Falsification test

Compare layer outputs and final token IDs with the reference implementation over increasing context lengths. Record cache bytes written and read. Real-arithmetic equivalence does not guarantee bitwise equality after finite-precision reordering.

## 10. Decision 8 — For GLM, lay out KV by IndexShare group

### Decision

When four GLM layers use the same top-2,048 historical positions, store those layers' latent records together by historical token.

### Alternative rejected

Four independent layer-major random gathers.

### Numerical benefit

A four-layer BF16 latent record is

\[
4(512+64)(2)=4{,}608\ \text{bytes}.
\]

Padding one record to 8 KiB changes a maximum-context decode step from 159,744 per-layer gather descriptors to 43,008 cross-layer descriptors:

\[
\frac{159{,}744}{43{,}008}=3.714.
\]

The cost is overfetch: 335.544 MB transferred instead of 184.025 MB of useful latent data. Persisting the padded format also reduces the 1 TB capacity margin from 144.525 GB to 66.947 GB.

This is a real choice, not a free optimization. The correct record size depends on measured small-read latency and top-k locality.

### Implementation

- Layers 2–5, 6–9, ..., 74–77 use 19 four-layer groups.
- Layers 0 and 1 use standalone records.
- Sort selected token indices before submission and merge adjacent physical ranges.
- Compare packed, 8-KiB-padded, and scatter/gather layouts.

### Hardware requirement

Efficient 4–8 KiB gathers or scatter/gather support that can combine several physical ranges under one submission. The current uGDS roadmap marks SGL support as not yet implemented.

### Falsification test

Measure end-to-end time, not request count alone. Reject the padded layout if the saved submission latency is smaller than the additional 151.519 MB transfer time.

## 11. Decision 9 — Use different schedules for prefill and decode

### Decision

Use layer-major matrix-matrix execution for prompt prefill. Use temporal weight caching and queued record fetches for batch-one decode.

### Alternative rejected

Running every prompt token through all layers independently, as the current Kimi executor does.

### Numerical benefit

For a matrix used by \(b\) tokens, weight arithmetic intensity grows approximately from

\[
I_1\approx2\ \text{FLOP/byte}
\]

to

\[
I_b\approx2b\ \text{FLOP/byte},
\]

ignoring activation traffic. The matrix is read once and reused \(b\) times.

For a MoE layer and token block \(B\), fetch the expert union

\[
U_{\ell,B}=\bigcup_{t\in B}A_{\ell,t},
\]

once. Its size obeys

\[
|U_{\ell,B}|\le\min(E,8|B|).
\]

### Implementation

Prefill:

1. Hold a block of token activations.
2. Stream one layer's fixed weights once.
3. Route the block.
4. Group tokens by expert.
5. Read every expert in the union once and run matrix-matrix kernels.
6. Append latent KV for the block.

Decode:

1. Keep selection data resident.
2. Fetch only misses for the current token.
3. Use weight and KV caches across time.

### Hardware requirement

Both efficient matrix-vector and matrix-matrix paths, sufficient activation SRAM/VRAM for a prefill block, and a routing scatter/gather unit that groups tokens by expert.

### Falsification test

Report weight bytes per prompt token versus block size and the expert-union size per layer. A blocked prefill that does not reduce bytes/token is incorrectly scheduled or dominated by expert-union saturation.

## 12. Decision 10 — Use direct SSD-to-GPU DMA, but do not claim it removes SSD bytes

### Decision

The SSD DMAs tensor payload directly into registered GPU memory. The CPU constructs commands and polls completions; it does not copy tensor payload.

### Alternative rejected

SSD to page cache or pinned host buffer, followed by host-to-GPU DMA.

### Numerical benefit

A staged path makes host DRAM receive \(N\) bytes from the SSD and supply \(N\) bytes to the GPU. Direct peer DMA removes approximately \(2N\) bytes of host-memory-controller traffic:

- Kimi cold weight step: about 65.973 GB of avoided host DRAM traffic;
- GLM cold weight step: about 82.767 GB.

For a serial staged path,

\[
T_{staged}=\frac{N}{B_{SSD}}+\frac{N}{B_{H2D}}.
\]

With perfect double-buffer overlap,

\[
T_{staged,overlap}\ge
\max\left(\frac{N}{B_{SSD}},\frac{N}{B_{H2D}}\right).
\]

Direct DMA obeys

\[
T_{direct}\ge\frac{N}{\min(B_{SSD},B_{P2P-link})}.
\]

Therefore direct DMA removes a copy, host traffic, and CPU/page-cache work. It does not automatically exceed SSD bandwidth, and a well-overlapped staged path can have similar steady-state bandwidth.

### Implementation

uGDS maps NVMe BARs, pins NVIDIA GPU memory in 64 KiB pages, maps those pages for the NVMe controller, constructs NVMe commands in user space, rings queue doorbells, and busy-polls completions. The current Blok path uses raw device offsets derived from a verified FIEMAP.

### Hardware requirement

- Guaranteed PCIe peer-to-peer routing between the SSD and accelerator.
- IOMMU/ACS configuration that does not force a host bounce.
- A large enough peer-memory aperture and stable page mappings.
- Error reporting for failed or invalidated mappings.

Consumer motherboard topology can invalidate this assumption. It must be checked rather than inferred from endpoint PCIe versions.

### Falsification test

Measure SSD bytes, P2P PCIe bytes, host DRAM bandwidth, CPU utilization, and latency for identical staged and direct transfers. If host DRAM sees the tensor payload during the direct test, the path is not direct.

## 13. Decision 11 — Keep control on the CPU today; expose a short path from GPU selections to commands

### Decision

Use the Ryzen CPU as the NVMe command co-processor, but communicate selections through a persistent mailbox instead of blocking copies and per-read API calls.

### Alternative rejected

Running the complete NVMe submission/completion stack on general GPU cores.

### Numerical benefit

The current control transfers are tiny but synchronizing:

- 60 blocking copies of 16 expert-ID bytes per token;
- 640 LM-head tiles, each followed by two blocking device-to-host copies, or 1,280 copies per sampled token.

The LM-head result can be reduced completely on the GPU and returned as one token ID. Expert IDs can be written to a coherent mailbox. The benefit is avoided synchronization, not avoided payload bytes.

### Implementation

1. GPU router writes eight IDs and a sequence number.
2. CPU polling core observes the sequence number.
3. CPU computes or looks up LBAs and submits one batch.
4. Completion flags release GPU work on each slot.
5. GPU computes global LM-head argmax and returns one 32-bit token ID.

### Hardware requirement

A cache-coherent or explicitly flushed command mailbox, low-latency notification, and completion flags visible to both agents. A future chip can replace the CPU poller with a small command engine using the same mailbox protocol.

### Falsification test

Count synchronization points and CPU cycles. The steady token path should contain no per-head-tile CPU comparison and no blocking expert-ID copy.

## 14. Decision 12 — Use raw immutable placement with an explicit safety boundary

### Decision

Resolve model files to physical extents while mounted, verify them, unmount the filesystem, bind the NVMe controller to uGDS, and write KV only to a separately reserved raw range.

### Alternative rejected

Using filesystem free space as if it were unallocated raw storage, or keeping the model filesystem writable while relying on stale FIEMAP addresses.

### Numerical benefit

This decision is for correctness, not speed. It prevents a KV write from corrupting model shards or filesystem data.

### Implementation

[`../../scripts/plan_ugds_layout.py`](../../scripts/plan_ugds_layout.py):

- aligns every required tensor range to 4 KiB;
- obtains physical extents with `FIEMAP_FLAG_SYNC`;
- rejects gaps, delayed allocation, unwritten extents, encoded extents, and misalignment;
- merges adjacent physical mappings;
- proves that the KV interval does not overlap a model extent;
- emits the exact device, extent map, KV base, and KV length.

The CUDA executor refuses a map/device mismatch and locks the KV range against concurrent Blok processes.

### Hardware requirement

Block-addressed reads and writes, protection/error reporting for out-of-range commands, and enough metadata integrity to fail closed.

### Falsification test

Generate adversarial maps with gaps, overlap, wrong devices, and shifted partitions. Every one must fail before the first DMA.

## 15. The layer pipeline the hardware must execute

The steady MoE-layer sequence is:

```text
known addresses                              data-dependent addresses

prefetch attention/fixed weights
        │
        ▼
attention + residual ──► norm ──► router ──► 8 expert IDs
                                          │
                                          ▼
                               one batched expert submission
                                          │
                         ┌────────────────┼────────────────┐
                         ▼                ▼                ▼
                    expert slot 0    expert slot 1    ... slot 7
                         │                │                │
                         └──── fused dequantize + matmul ──┘
                                          │
                         shared expert ───┤
                                          ▼
                                      accumulate
                                          │
                         prefetch next layer's known weights
```

There are two distinct opportunities for overlap:

1. Prefetch weights whose addresses are known before routing.
2. Execute the shared expert and completed routed experts while other expert records remain in flight.

There is one unavoidable dependency: routed-expert addresses are unknown until the router completes. Keeping the router resident and shortening selection-to-submit latency directly shortens the critical path.

## 16. What an AI chip should add

The following blocks have a direct use in this workload:

| Block | Input | Output | Quantified purpose |
|---|---|---|---|
| Selection-to-I/O command generator | layer ID and eight expert IDs | eight base-plus-stride reads | Removes 60 blocking CPU round trips and per-record address construction. |
| Deep storage queue plus PRP/SGL capacity | 4 KiB–\(S_{cmd}\) commands | completion bitmap | Maintains \(Q\ge Q_{sat}\) without the 64-list-page bottleneck; current synchronous path has \(Q=1\). |
| Persistent DMA slab manager | registered peer-memory arena | safe producer/consumer slots | Removes over 217k allocation/registration cycles per Kimi token. |
| Fused quantized matmul | INT4/BF16-scale or E4M3/F32-scale record | accumulated output | Preserves the 3.556× Kimi and 2× GLM storage reductions through execution. |
| Latent-attention datapath | 576-element records and query | attention output | Makes 256K Kimi and 1M GLM cache capacity feasible. |
| Small record cache with programmable tags | `(layer, object_id)` | hit/miss and slot | Holds routers, head, index keys, and measured hot experts. |
| Coherent CPU/GPU mailbox | selection and completion sequence numbers | commands and ready flags | Removes control-path synchronizations without putting the NVMe stack on GPU cores. |
| I/O observability counters | commands and stage events | bytes, latency, QD, stalls | Separates byte savings, transaction savings, and overlap. |

More dense TOPS without these mechanisms do not address the cold decode limit because the weight arithmetic intensity is below 2 FLOP/byte.

## 17. Measurements required before presenting a speedup

The chip presentation should show these measurements or label them pending:

1. PCIe topology and successful direct P2P path.
2. NVMe controller page size, MDTS, maximum queue depth, and resulting `S_cmd`.
3. Read latency and bandwidth for 4 KiB, 8 KiB, 128 KiB, 1 MiB, and `S_cmd` over queue depths 1–512.
4. Registration and allocation latency.
5. Selection-to-first-expert-command latency.
6. SSD bytes, PCIe bytes, and host DRAM bytes per token.
7. Queue occupancy and empty-queue time.
8. Weight-cache hit rate by object class.
9. Expert-union size during blocked prefill.
10. KV bytes written/read at each context length.
11. Per-stage GPU utilization and storage stall time.
12. Reference-logit and token-ID agreement.

The performance identity to report is:

\[
T_{token}=T_{selection}+T_{weight\ I/O}+T_{KV\ I/O}+T_{compute}
-T_{overlap}+T_{control}.
\]

Every improvement must be assigned to one of four causes:

- fewer useful bytes;
- fewer commands or submissions;
- a cache hit;
- more overlap.

If it cannot be assigned and measured, it is not yet a hardware advantage.

## 18. Implementation order

1. **Instrument first.** Print controller limits and add counters for application reads, NVMe commands, bytes, queue depth, registrations, allocations, and synchronizations.
2. **Remove software serialization.** Add persistent slabs, GPU-global LM-head argmax, and the expert-ID mailbox.
3. **Use the existing batch queue.** Replace synchronous model reads and verify that queue depth reaches the measured requirement.
4. **Change the model layout.** Build expert records and a deterministic fixed stream; verify byte-for-byte conversion.
5. **Pin the critical set.** Norms, routers, head, GLM indexer weights, and long-context index keys.
6. **Measure expert reuse.** Add the expert cache only after obtaining the miss curve.
7. **Replace expanded KV.** Implement latent MLA and online softmax.
8. **Implement blocked prefill.** Measure matrix reuse and expert unions.
9. **Port GLM FP8.** Reuse the queue, cache, record, and latent-KV mechanisms; add E4M3 and DSA/IndexShare.

This order makes each result attributable. It also prevents a correct but slow direct-storage prototype from being mistaken for the final architecture.
