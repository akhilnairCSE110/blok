# M5 TensorOps and Metal I/O engineering notes

**Recorded:** 2026-08-14  
**Target:** base Apple M5, 10-core GPU, 24 GB unified memory, internal NVMe  
**Model:** DeepSeek-R1 671B `UD-IQ1_S` GGUF  
**Status:** implementation guidance and benchmark contract; not evidence that the
accepted MetalBlok path currently dispatches TensorOps or Metal I/O

This note preserves the public M5 mechanisms that matter to MetalBlok. It also
draws a hard line between documented software tiles and Apple's private GPU
instruction set. Every proposed fast path remains subordinate to the existing
token/logit parity gate.

## 1. Supported abstraction boundary

The closest public interface to the Neural Accelerator integrated into each M5
GPU core is Metal Performance Primitives (MPP) TensorOps:

```text
MetalBlok runtime
  -> Metal 4 command and resource APIs
  -> Metal Shading Language
  -> mpp::tensor_ops / matmul2d / cooperative_tensor
  -> compiler-generated GPU and Neural Accelerator instructions
  -> private Apple GPU implementation
```

The per-GPU-core Neural Accelerator is not the separate 16-core Apple Neural
Engine (ANE). TensorOps targets the former. Arbitrary ANE execution is not a
raw Metal facility, and moving a transformer stage through Core ML would add
conversion, synchronization, and memory traffic that must be counted.

Apple does not publish a supported GPU assembler, inline GPU assembly, binary
instruction set, physical matrix-array dimensions, accumulator organization,
or matrix-instruction latency/throughput table. The supported offline path is:

```text
MSL -> AIR -> metallib -> device-specific pipeline compilation
```

`mpp::tensor_ops::matmul2d` is therefore the strongest public request for M5
matrix acceleration. A public descriptor is a logical operation, not proof of
the physical multiplier geometry.

## 2. TensorOps matrix contract

TensorOps expresses

\[
D = \alpha AB + \beta C,
\qquad
D_{ij}=\alpha\sum_{k=0}^{K-1}A_{ik}B_{kj}+\beta C_{ij}.
\]

For \(A\in\mathbb R^{M\times K}\) and
\(B\in\mathbb R^{K\times N}\), the operation performs approximately
\(2MNK\) floating-point operations. The central API pattern is conceptually:

```metal
constexpr auto desc = matmul2d_descriptor(SM, SN);
matmul2d<desc, execution_simdgroup> op;
op.run(a, b, destination);
```

`op.run()` conveys a structured matrix operation to the compiler. It is not
equivalent to manually emitting a loop of scalar FMAs and hoping the compiler
infers the same execution path.

### M5 starting geometry

Apple's March 2026 MPP guidance gives these FP16 GEMM starting points:

| Parameter | Starting point | Meaning |
|---|---:|---|
| `SM` | 32 | output rows per SIMD-group tile |
| `SN` | 32 | output columns per SIMD-group tile |
| SIMD groups/threadgroup | 2 x 2 | logical 64 x 64 output region |
| `BK` | 128 | K-dimension synchronization interval |

For one 32 x 32 output tile and a K slice of 128:

\[
2(32)(32)(128)=262{,}144\ \text{FLOPs}.
\]

With ideal single fetches of FP16 A and B tiles and one FP16 output write, the
traffic is

\[
32(128)(2)+128(32)(2)+32(32)(2)=18{,}432\ \text{bytes},
\]

for an idealized arithmetic intensity of

\[
AI\approx 262{,}144/18{,}432=14.22\ \text{FLOP/byte}.
\]

These are logical scheduling sizes. They do **not** imply a physical
32 x 32 x 128 multiplier inside M5.

## 3. Cooperative tensors and fusion

A `cooperative_tensor` distributes a TensorOps result across participating
threads using a hardware-dependent mapping. Code must not assume that lane
`i` owns matrix element `i`.

Its main value is avoiding materialization of intermediates in unified memory:

```text
matmul -> cooperative accumulator -> bias/activation/reduction -> store
```

When API compatibility permits, a cooperative tensor can feed another tensor
operation. The important attention target is:

```text
Q,K -> TensorOps QK^T -> online row max/sum/softmax
    -> TensorOps P V -> O
```

The rejected form writes and rereads full score and probability matrices.
MetalBlok should instead tile the causal score domain and keep partial maxima,
sums, and numerators local. At 128K context, avoiding an \(O(T^2)\) materialized
score matrix is a capacity requirement, not merely a speed optimization.

Fusion is accepted only if it preserves the checkpoint's finite-precision
graph. Real-number associativity alone does not prove FP32/FP16 parity after
reordering reductions or moving a quantization boundary.

## 4. Compile-time specialization

Tensor extents and slices should be static whenever model geometry makes them
known. Static extents allow compile-time bounds-check elimination and stronger
specialization:

```metal
slice<32, 32>(...);       // preferred fixed interior tile
slice<dynamic_extent, dynamic_extent>(...); // edge-only path
```

MetalBlok should compile separate kernels for:

- aligned interior tiles with static M/N/K extents;
- M, N, and K edge tiles with explicit bounds;
- DeepSeek's frozen tensor shapes and quantization families;
- prefill tiles and single-token decode, which have different rooflines.

Function constants should select a small set of proven variants. Runtime
shape-generic branching inside the hottest loops is the fallback, not the
target.

## 5. Cache scheduling and local-state pressure

MPP's Apple-Silicon guidance does not require copying every GEMM operand
through explicit threadgroup arrays. TensorOps can load device memory and use
the cache hierarchy. CUDA-style shared-memory staging must therefore earn its
place in a benchmark; it is not an automatic optimization on M5.

For tiled GEMM, use a locality-preserving threadgroup walk. Morton/Z order is
the documented M5 starting point because simultaneously executing output tiles
then share nearby A/B regions in cache.

A barrier at each `BK=128` interval can keep the four SIMD groups near the same
K position even when no threadgroup-memory payload is exchanged:

```text
without pacing: SIMD groups touch widely separated K regions
with pacing:    SIMD groups consume a compact shared cache footprint
```

Larger tiles increase reuse but also grow live accumulators approximately as
`SM * SN`. Excessive local state lowers occupancy and can spill into memory.
The 32 x 32 starting point balances reuse, accelerator efficiency, occupancy,
and register pressure; it is a benchmark seed, not a universal optimum.

## 6. Quantized TensorOps and MetalBlok's checkpoint

Current TensorOps APIs expose low-precision tensor formats and auxiliary scale
planes, including hardware-oriented 2/4/8-bit data on supported systems. A
block-scaled tensor represents approximately

\[
X_{real}=s_b Q.
\]

The scale-plane block shape describes quantization metadata, not the physical
matrix engine.

MetalBlok's checkpoint mixes Q4_K, Q5_K, Q6_K, IQ2_XXS, and IQ1_S. These GGUF
encodings are not automatically identical to MPP's supported tensor formats.
The current custom kernels decode them directly inside each dot product and do
not materialize a full dequantized matrix.

A TensorOps conversion is admissible only when all of the following are shown:

1. The repacked representation preserves every stored weight required by the
   accepted numerical contract, or its explicitly approved tolerance.
2. Repacking is offline or amortized; repack bytes and time are reported.
3. Dequantization does not create a persistent FP16/FP32 copy of the 671B model.
4. Tensor/logit/token parity is compared to the accepted custom-kernel path.
5. End-to-end wall time improves after SSD, conversion, and dispatch costs.

This makes TensorOps immediately attractive for matrix-heavy prefill and
grouped expert batches, but not an automatic replacement for bandwidth-bound
batch-one IQ1_S/IQ2_XXS GEMV.

## 7. Prefill and decode require different kernels

The roofline bound is

\[
P_{attainable}=\min(P_{compute\ peak}, BW\cdot AI),
\qquad
AI=\frac{\text{FLOPs}}{\text{bytes transferred}}.
\]

Prefill has \(M>1\), reuses weights across prompt rows, and can become a true
quantized matrix-matrix problem. It is the primary place to pursue TensorOps,
32 x 32 tiles, grouped MoE GEMM, fused causal attention, and Morton scheduling.

Single-stream decode approaches matrix-vector multiplication \(y=Wx\). With
4-bit weights, the optimistic weight-only intensity is roughly

\[
2\ \text{FLOPs}/0.5\ \text{byte}=4\ \text{FLOP/byte}.
\]

Decode therefore remains dominated by useful model bytes, SSD bandwidth when
weights are not resident, and unified-memory bandwidth when they are. More NAX
compute cannot overcome a storage byte floor. Decode priorities remain:

1. fetch only the exact routed experts;
2. keep weights compressed through multiplication;
3. overlap layer/expert I/O with current compute;
4. reuse cached fixed tensors and already selected records;
5. batch independent conversations to amortize weight reads when latency goals
   permit;
6. reduce command buffers, allocations, and synchronization per token.

## 8. Metal memory and submission

Apple Silicon has unified physical memory, but bytes still traverse the memory
fabric and caches. `MTLStorageModeShared` removes a discrete CPU-to-VRAM copy;
it does not make reads free.

The relevant persistent-resource APIs are:

| API | MetalBlok use |
|---|---|
| `MTLBuffer` | tensor and I/O slabs |
| `MTLHeap` | bulk suballocation and reusable scratch |
| `MTLResidencySet` | explicit GPU accessibility/residency |
| `MTLTensor` | tensor view presented to TensorOps |
| `MTL4ArgumentTable` | persistent tensor/scale/KV bindings |
| `MTL4CommandAllocator` | reusable encoded-command storage |
| `MTL4CommandQueue` | low-overhead submission |

Use multiple command allocators only when CPU encoding overlaps in-flight GPU
work, then reset/reuse them after completion. Argument tables should retain
stable addresses for weights, activation pools, compact KV pages, scale data,
scratch, and metadata instead of rebuilding a large binding set per operation.

## 9. Metal I/O streaming path

The relevant filesystem-to-resource APIs are:

```text
MTLIOCommandQueue
MTLIOCommandBuffer
MTLIOFileHandle
MTLIOCommandQueueDescriptor
MTLSharedEvent
```

The intended path is:

```text
APFS/NVMe -> MTLIOFileHandle -> MTLIOCommandBuffer.load()
          -> MTLBuffer -> MTLTensor/custom kernel -> GPU
```

This can remove ordinary user-space staging and can overlap I/O with GPU work.
It does not mean NAND writes into a matrix engine, nor does it bypass APFS, the
NVMe controller, flash translation, ECC, or memory-fabric traffic.

For deterministic layer execution, use at least two persistent slabs:

```text
GPU computes layer/expert n from slab A
Metal I/O loads known fixed payload n+1 into slab B
shared event signals B ready
queues swap slabs
```

Routed expert IDs are not known until the router completes. Fixed next-layer
weights can be prefetched unconditionally; expert payloads can be submitted as
soon as exact top-k IDs commit. Do not predict experts approximately when exact
routing already removes 248 of 256 experts.

Compressed Metal I/O is a benchmark candidate only. It wins when fewer SSD
bytes plus decompression cost beats reading the checkpoint encoding directly.
The GGUF quantized representation is already compact, so a second compression
layer may add latency or provide negligible reduction.

## 10. Profiling and acceptance gates

Wall time is the objective. GPU occupancy or NAX utilization is diagnostic,
not the reward by itself. Every candidate must report:

- end-to-end and prefill tokens/s;
- decode tokens/s;
- useful model bytes/token and total bytes/token;
- effective SSD GB/s and exposed I/O wait;
- GPU kernel time and GPU duty cycle;
- Neural Accelerator utilization when the tool exposes it;
- cache bandwidth/misses, occupancy, register pressure, and spills;
- command buffers, dispatches, allocations, and synchronizations/token;
- compact KV bytes/token, context capacity, and peak available-memory delta;
- joules/token when `powermetrics` is available;
- sampled intermediate errors, final logits, top token, runner-up, and margin.

Interpret the counters carefully:

- high ALU and near-zero Neural Accelerator utilization indicates a traditional
  shader kernel, not a successful TensorOps path;
- high NAX utilization with worse wall time is still a rejection;
- reduced GPU time hidden behind unchanged I/O wait is not an end-to-end win;
- register spilling can erase the reuse gained by a larger tile;
- exact token parity is mandatory, and accepted numerical tolerances must be
  stated rather than inferred from matching one token.

Use the installed Xcode tools as the version-specific authority:

```bash
xcrun -f gpucapture
xcrun -f gpudebug
xcrun -f metalperftrace
xcrun gpucapture --help
xcrun gpudebug --help
xcrun metalperftrace --help
```

## 11. SDK and MLX source inspection

Interrogate the SDK shipped on the test machine rather than copying stale API
spellings:

```bash
SDKROOT="$(xcrun --sdk macosx --show-sdk-path)"

grep -R -n "matmul2d_descriptor" \
  "$SDKROOT/System/Library/Frameworks/MetalPerformancePrimitives.framework"
grep -R -n "cooperative_tensor" \
  "$SDKROOT/System/Library/Frameworks/MetalPerformancePrimitives.framework"
grep -R -n "execution_simdgroup" \
  "$SDKROOT/System/Library/Frameworks/MetalPerformancePrimitives.framework"
grep -R -n "tensor_blockwise\|metal_fp8" "$SDKROOT" | head -100
```

Compile the exact local toolchain with:

```bash
xcrun -sdk macosx metal -c kernel.metal -o kernel.air
xcrun -sdk macosx metallib kernel.air -o kernel.metallib
```

MLX is the best open implementation reference for Apple's production Metal
backend. Relevant searches include:

```bash
git grep -n -E 'matmul2d_descriptor|tensor_ops|cooperative_tensor'
git grep -n -i -E 'nax|neural.accel|neural_accel'
git grep -n -E 'matmul|gemm|gemv|quant|fp8|int4|attention|sdpa|softmax' \
  mlx/backend/metal
```

Copying MLX's dispatch idea is not enough: MetalBlok must benchmark its exact
DeepSeek geometry, GGUF formats, storage path, and 24 GB memory pressure.

## 12. Concrete experiment ladder

Before changing the full model, establish one reproducible matrix benchmark
with identical operands and results:

```text
1. scalar MSL FMA
2. vector/SIMD MSL
3. simdgroup matrix path
4. MPP TensorOps
5. tuned M5 TensorOps:
   SM=32, SN=32, 2x2 SIMD groups, BK=128,
   static interior extents, cooperative accumulation, Morton walk
6. tuned TensorOps plus Metal 4 persistent submission/resources
7. storage-fed variant using Metal I/O and shared-event overlap
```

For every step, record effective TFLOP/s, GB/s, arithmetic intensity,
nanoseconds/MAC, NAX utilization, register spills, and wall time. Then repeat on
the model's actual projection shapes and the useful prompt batch sizes; a
4096-square benchmark alone does not predict DeepSeek decode.

The implementation priority for MetalBlok is:

1. a true quantized prefill QMM that reuses each weight block across prompt
   rows without materializing dequantized weights;
2. tiled causal MLA attention with local online softmax state;
3. grouped routed-expert prefill GEMM;
4. persistent Metal 4 resources and lower submission overhead;
5. Metal I/O A/B streaming with shared-event synchronization;
6. only then, format-specific TensorOps repacking where parity and full
   end-to-end measurements justify it.

## 13. Primary references

- [Metal Performance Primitives Programming Guide](https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf)
- [WWDC26: Optimize custom machine learning operations with Metal](https://developer.apple.com/videos/play/wwdc2026/330/)
- [Running inline ML operations in a shader with Metal 4](https://developer.apple.com/documentation/metal/running-inline-ml-operations-in-a-shader-with-metal-4)
- [Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
- [Understanding the Metal 4 core API](https://developer.apple.com/documentation/metal/understanding-the-metal-4-core-api)
- [`MTLResidencySet`](https://developer.apple.com/documentation/metal/mtlresidencyset)
- [Metal resource loading](https://developer.apple.com/documentation/metal/resource-loading)
- [`MTLIOCommandQueue`](https://developer.apple.com/documentation/metal/mtliocommandqueue)
- [Metal tools](https://developer.apple.com/metal/tools/)
- [MLX source](https://github.com/ml-explore/mlx)
- [Apple ML Research: Exploring LLMs with MLX and M5](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)
- [Apple M5 overview](https://www.apple.com/newsroom/2025/10/apple-unleashes-m5-the-next-big-leap-in-ai-performance-for-apple-silicon/)
- [Apple M5 Pro and M5 Max overview](https://www.apple.com/newsroom/2026/03/apple-debuts-m5-pro-and-m5-max-to-supercharge-the-most-demanding-pro-workflows/)

