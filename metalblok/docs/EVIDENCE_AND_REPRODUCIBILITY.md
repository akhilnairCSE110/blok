# MetalBlok evidence and reproducibility ledger

**Artifact date:** 2026-08-12

**Scope:** native Apple-Metal DeepSeek-R1 671B V0

**Normative architecture:** [V0_ARCHITECTURE.md](V0_ARCHITECTURE.md)

**Exact execution contract:** [TENSOR_HARDWARE_SPEC.md](TENSOR_HARDWARE_SPEC.md)

This ledger separates implemented behavior, measurements, algebra, and future
ideas. A saved log is evidence only for the binary and command that produced
it. A mathematical identity is not finite-precision parity. A coherent string
is not an independent reference-runtime comparison.

## 1. Evidence vocabulary

| Class | Meaning |
|---|---|
| Checkpoint-derived | Exact GGUF metadata, descriptors, types, shapes, and stored byte arithmetic. |
| Exact algebra | Identity over the explicitly stated arithmetic domain. |
| Implemented | Reachable on the active C++/Metal path. |
| Unit-verified | Checked by a deterministic CPU or CPU/Metal test. |
| Measured | Retained host log from the named model/machine. |
| Parity-checked | Same saved state and decoding mode produced the named token/logit anchors. |
| Rejected | Implemented experimentally, failed a gate, then removed. |
| Proposed | Not part of the accepted runtime. |

## 2. Frozen artifact and machine

The target is the local three-shard `DeepSeek-R1-UD-IQ1_S` GGUF on a 10-core
Apple M5 with 24 GB unified memory. Preflight derived:

| Shard | Logical bytes | Allocated bytes | Physical status |
|---:|---:|---:|---|
| 0 | 49,349,193,664 | 49,349,193,728 | resident |
| 1 | 49,397,904,416 | 49,397,907,456 | resident |
| 2 | 41,484,340,384 | 41,484,341,248 | resident |
| **Total** | **140,231,438,464** | **140,231,442,432** | **resident** |

The manifest contains 1,025 tensors and 671,026,419,200 logical parameters.
Stored representation:

| GGML type | Tensor count | Stored decimal GB | Share |
|---|---:|---:|---:|
| F32 | 361 | 0.43 | 0.3% |
| Q4_K | 190 | 6.67 | 4.8% |
| Q5_K | 116 | 1.17 | 0.8% |
| Q6_K | 184 | 2.83 | 2.0% |
| IQ2_XXS | 6 | 5.81 | 4.1% |
| IQ1_S | 168 | 123.31 | 87.9% |

Physical-residency admission uses allocated-file metadata and does not hydrate
the 140 GB payload. Inference refuses a missing/sparse/cloud-placeholder shard.

## 3. Reproduction commands

From the repository root:

```sh
export METALBLOK_MODEL=/path/to/DeepSeek-R1-UD-IQ1_S-00001-of-00003.gguf

cmake -S metalblok -B metalblok/build
cmake --build metalblok/build -j8
ctest --test-dir metalblok/build --output-on-failure

metalblok/build/metalblok --preflight "$METALBLOK_MODEL"
metalblok/build/metalblok --probe-gguf "$METALBLOK_MODEL"
metalblok/build/metalblok --validate-router
```

Ordinary inference and the exact acceptance harness are:

```sh
./run_blok.py "Write a correct Python program." -n 256
scripts/prove_metal_1k.py
```

The proof script asks the native tokenizer to construct exactly 1,000 input
tokens, uses greedy decoding, context 2,048, exact checkpointing, and requires
1,000 emitted tokens. It can resume an authoritative proof state at positions
1,000–1,998 and derives the exact remaining output count.

Host Metal access is required. A filesystem sandbox error such as `no Metal
device` is not a pass.

## 4. Model admission evidence

The loader verifies:

- `general.architecture=deepseek2` and the frozen geometry;
- three-shard manifest and all physical extents;
- 61 layers, 128 heads, 7,168 residual width, and 129,280 vocabulary;
- Q rank 1,536, KV rank 512, key width 192, value width 128, RoPE width 64;
- three dense layers and 58 sparse layers;
- 256 routed experts, exact top-8, eight groups, retained top-four groups;
- sigmoid/no-aux routing, top-k normalization, and scale 2.5;
- required YaRN factor, original context, correction range, and log multiplier;
- all 1,025 tensor families, shapes, supported descriptor types, and distinct
  output head.

Failure is early and explicit; the runtime does not silently substitute a
general model path.

## 5. Stored-type arithmetic evidence

Every stored compute type is checked on a representative real checkpoint
matrix. CPU code independently decodes the GGUF blocks and computes GEMV; the
Metal result is compared using

\[
\epsilon_{rel}=\frac{\lVert y_M-y_C\rVert_2}
{\max(\lVert y_C\rVert_2,10^{-12})}.
\]

| Type | Representative family | Observed relative error |
|---|---|---:|
| F32 | router | \(2.04\times10^{-4}\) |
| Q4_K | query projection | \(2.09\times10^{-4}\) |
| Q5_K | shared expert | \(2.06\times10^{-4}\) |
| Q6_K | KV-A | \(2.13\times10^{-4}\) |
| IQ2_XXS | routed down | \(2.06\times10^{-4}\) |
| IQ1_S | routed gate | \(2.08\times10^{-4}\) |

All remain below the declared `5e-3` kernel-oracle threshold. These tests prove
the block decoders and row reductions on real tensors; they are not a claim
that a second engine has matched every final logit.

## 6. Router evidence

The independent CPU oracle and grouped Metal router returned IDs

```text
254, 79, 163, 171, 200, 247, 180, 242
```

with identical order. Maximum displayed mixture-weight difference was
`3.0e-8`. The test covers sigmoid, correction bias, top-two group scoring,
top-four group retention, top-eight selection, lower-ID tie-breaking,
normalization on uncorrected probabilities, and scale 2.5.

This matters for storage correctness: only after this exact result may the
runtime omit the other 248 expert slices. The omitted contribution is exactly
zero by the model's sparse graph, not an approximation.

## 7. Numerical optimization parity and negative result

The accepted scheduling optimization was compared from the same v3 state at
position 1,280 under temperature zero:

| Position | Token ID | Greedy logit, old | Greedy logit, accepted V0 |
|---:|---:|---:|---:|
| 1,280 | 270 | 48.1055 | 48.1055 |
| 1,281 | 1,990 | 51.7904 | 51.7904 |
| 1,282 | 344 | 45.4884 | 45.4884 |

An experimental residual-plus-RMSNorm kernel selected the same IDs at these
early points but changed the position-1,282 maximum logit to 45.2879. That is
not accepted parity. The experimental kernels, environment switch, and host
plumbing were removed.

This result demonstrates the release rule: operation fusion is useful only
when the finite-precision boundary remains compatible. Associativity over real
numbers cannot override measured FP32 divergence.

## 8. Byte and request evidence

The previous decode path and accepted safe path, from the same model/state:

| Per-position metric | Previous | Accepted V0 | Delta |
|---|---:|---:|---:|
| model/NVMe bytes | 13,849,970,112 | 13,587,632,064 | -262,338,048 |
| logical/actual reads | 1,939 | 1,869 | -70 |
| urgent selected-expert bytes | — | 4,035,182,592 | explicit |
| urgent selected-expert reads | — | 1,392 | explicit |
| command buffers | 178 | 178 | 0 |
| hot-path allocations | 0 | 0 | 0 |
| KV bytes appended | 4,005,504 | 4,005,504 | exact |

The 256 MiB cache retains 70 of the smallest fixed projections. A 2 GiB
experiment reduced traffic to 11,709,501,376 bytes, but increased memory
compression on the 24 GB machine. It was rejected as the release default.
That experiment is evidence that bytes can be saved, not evidence that the
larger cache is safe for a long context.

## 9. Timing evidence and metric interpretation

Before late-run memory pressure, representative accepted decode positions
reported:

| Metric | Representative interval |
|---|---:|
| end-to-end step | 3.09–3.22 s |
| decode rate | 0.31–0.32 token/s |
| NVMe span | 2.96–3.07 s |
| effective aggregate NVMe | 4.4–4.6 GB/s |
| Metal GPU execution | 0.47–0.52 s |
| command buffers | 178 |
| hot allocations | 0 |

The storage span and wall time are close while GPU time is much smaller, so
the current decode is storage-bound. This is a measured bottleneck statement,
not a generic claim about Apple GPUs.

Metric semantics prevent misleading arithmetic:

- `model_bytes` is useful immutable model payload consumed by the schedule;
- `nvme_bytes` is actual submitted `pread` payload;
- `nvme_span_us` spans first I/O start through last completion and is used for
  `nvme_gbps`;
- `io_service_us` sums service time across 12 workers and may exceed wall time;
- `io_wait_us` is explicit producer blocking, not total I/O duration;
- `gpu_us` sums completed Metal command-buffer execution;
- VM counters are deltas during the position, not process-local bytes.

## 10. Memory and context evidence

The exact accepted state adds

\[
61[128(128)+128(128)+64]2=4{,}005{,}504
\]

bytes per position. Context 2,048 reserves 8,203,272,192 bytes. The measured
admission ledger included a 760.166 MB resident head, two 368 MB fixed slabs,
263 MB fixed cache, 90.833 MB expert arena, 124.95 MB prefill scratch, 33.55 MB
runtime margin, 1.07 GB host reserve, and 2.15 GB cache guard. Estimated active
runtime allocation was about 10.22 GB against 20.95 GB live available memory.

Every decode line carries pageouts, compressions, decompressions, swap-ins,
swap-outs, and available-memory delta. The acceptance continuation has shown
memory compression and pageout activity. Through position 1,848 it accumulated
2,129 system-wide swap-ins and 116 swap-outs; the swap-outs occurred at five
positions. The counters cover the whole host and cannot be attributed solely
to MetalBlok, but they are a real pressure/performance signal and explain why
the late run is slower than its early steady sample. Final totals are recorded
only after the proof reaches position 1,999.

## 11. Exact 1,000 + 1,000 acceptance chain

The native tokenizer constructed exactly 1,000 prompt tokens. The layer-major
prefill completed in 591.937584 s, or 1.69 token/s, and produced token ID
33,001 (`Okay`) as the first output.

| Artifact | Path | Current evidence |
|---|---|---|
| prefill log | `../runs/run-20260812-012449-18510.log` | exact 1,000-token prefill complete |
| authoritative state | `../runs/proof-1k-20260812-012449.state` | atomic v3 continuation |
| optimized continuation log | `../runs/run-20260812-021356-20767.log` | live per-position metrics |
| optimized continuation text | `../runs/run-20260812-021356-20767.txt` | live emitted suffix |
| acceptance position | state position 1,999 | **RUNNING** |

Why the final committed position is 1,999 rather than 2,000: prefill commits
1,000 prompt positions and computes pending output `A0`; emitting `A0` does
not advance it again. The remaining `A1..A999` require 999 decode advances.
Thus 1,000 emitted tokens correspond to positions 1,000–1,999 with final
state position 1,999.

Interruption and continuation do not restart prefill. State v3 stores the exact
KV prefix plus the already-predicted pending token. Periodic atomic checkpoints
bound lost work without changing the decode graph.

## 12. Logging and fault localization evidence

Normal metrics provide one closed-loop record per position: token/logit,
wall/GPU/I/O times, bytes, requests, queue service/tail/peak, command buffers,
allocations, KV growth, and VM pressure.

Two opt-in modes deepen diagnosis:

- `--profile-layers` records fixed-stage span/block, attention wall/GPU,
  FFN wall/GPU, expert I/O wait/bytes/reads, command buffers, and dispatches;
- `--trace` records layer/stage residual RMS, min/max, non-finite count, stable
  hash, router IDs/weights, and top logits.

Together these answer whether a regression begins at storage ownership,
quantized projection, RoPE, attention, router selection, expert accumulation,
or final sampling rather than merely reporting “bad output.”

## 13. Source-to-claim map

| Concern | Normative source/test |
|---|---|
| model contract and tensor schedule | `../src/gguf_runtime.cpp`, `../src/gguf_runtime.hpp` |
| GGUF descriptor parsing | `../src/gguf.cpp`, `../src/gguf.hpp` |
| block dequant CPU oracle | `../src/gguf_dequant.cpp` |
| Metal quant/attention/router/expert kernels | `../src/kernels.metal` |
| independent router oracle | `../src/router_ref.hpp`, `../tests/router_ref_test.cpp` |
| shared-buffer command accounting | `../src/metal_ctx.mm`, `../src/metal_ctx.hpp` |
| prioritized exact-length I/O | `../src/pread_ring.cpp`, `../src/pread_ring.hpp` |
| memory-pressure counters | `../src/memstat.hpp` |
| tokenizer/decode loop/state CLI | `../src/main.mm` |
| admission and wrapper safety | `../../run_blok.py` |
| exact acceptance harness | `../../scripts/prove_metal_1k.py` |

## 14. Claims and boundaries

| Claim | Evidence status | Boundary |
|---|---|---|
| Complete native 61-layer graph emits coherent text. | Implemented + measured | Not independent full-logit parity. |
| Only eight routed experts need reading after exact routing. | Exact graph + router verified | Does not remove unconditional/shared tensors. |
| Quant deconstruction is fused into multiplication. | Implemented + unit-verified | Kernels remain checkpoint-format-specific. |
| Layer-major prefill handles exactly 1,000 tokens. | Measured | 1.69 token/s; fixed-projection QMM remains future work. |
| Safe schedule reduces 262.338 MB and 70 reads/step. | Measured + parity-checked | Compared at named saved-state anchors. |
| Decode is currently storage-bound. | Measured | Applies to this model/machine/run regime. |
| Exact continuation preserves the same loop. | Implemented + measured | State is large because K/V is expanded. |
| One million context tokens work. | Not claimed | Current exact state would require about 4.006 TB. |
| ANE or undocumented NAX paths are used. | Not claimed | Complete active graph uses public Metal. |
| Temporal expert cache/concurrent serving works. | Not claimed | Future scheduler/format work. |

The defensible V0 statement is deliberately narrow: this exact 140.231 GB
checkpoint executes its complete native graph from an explicit SSD-backed
working set on a 24 GB M5; its accepted schedule is instrumented, resumable,
and parity-checked at named anchors; and the release closes only when the saved
state reaches position 1,999 with 1,000 emitted tokens.
