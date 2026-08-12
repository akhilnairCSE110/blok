# MetalBlok V0 closeout

**Date:** 2026-08-12  
**Host:** 10-core Apple M5, 24 GB unified memory, macOS/Metal  
**Checkpoint:** DeepSeek-R1 671B `UD-IQ1_S`, 3 GGUF shards, 1,025 tensors,
140,231,438,464 stored bytes

## Outcome

V0 is a native, model-specific inference engine. It tokenizes the DeepSeek
chat prompt, executes all 61 transformer layers, exact grouped top-8 MoE
routing, mixed GGUF quantized projections, causal attention, greedy or seeded
nucleus sampling, output projection, decoding, and exact crash-safe
continuation. The model remains on SSD; no 140 GB payload mapping or full
dequantized copy is created.

The release acceptance is a single exact run with 1,000 native input tokens
and 1,000 emitted output tokens. Its 1,000-token prefill completed on this host
at 1.69 token/s and produced first token ID 33001 (`Okay`). The output
continuation is retained in atomic state and log artifacts. The final row in
this document is filled only after the state reaches position 1,999.

## Exact model contract

| Quantity | Value |
|---|---:|
| layers | 61 |
| residual width | 7,168 |
| query heads | 128 |
| Q low-rank width | 1,536 |
| KV low-rank width | 512 |
| non-RoPE key/query width | 128/head |
| RoPE width | 64/head query, one shared key |
| value width | 128/head |
| dense layers | 3 |
| routed MoE layers | 58 |
| routed experts / selected | 256 / 8 |
| expert intermediate width | 2,048 |
| vocabulary | 129,280 |
| declared model context | 163,840 |
| V0 acceptance context | 2,048 |

The loader rejects any drift in these constants, the YaRN factor/original
context/log multiplier, required tensor shapes, quant types, output head, or
three-shard manifest.

## Forward graph and numerical decisions

For every layer, V0 performs pre-attention RMSNorm, Q-A, Q-A norm, Q-B,
DeepSeek consecutive-pair YaRN RoPE, KV-A, KV-A norm and shared RoPE, the
stored combined KV-B projection, FP16 K/V cache write, stable causal attention,
attention output projection, and the residual. The first three layers use
dense SwiGLU. The remaining 58 execute the shared expert plus exact
group-limited sigmoid routing and eight routed SwiGLU experts. Final RMSNorm
and the distinct Q6_K output head produce FP32 logits.

V0 deliberately preserves explicit FP32 activation boundaries. A fused
residual-plus-RMSNorm experiment selected the same early token IDs but changed
the third checked logit from 45.4884 to 45.2879. It failed the parity gate and
was removed. Likewise, real-arithmetic MLA weight absorption is not used for
this legacy combined-KV-B quantized checkpoint: reassociation and different
rounding points are not accepted as “exact” merely because the algebra is
associative over real numbers.

The production state therefore stores expanded non-RoPE K and V plus one
shared rotary K in FP16:

\[
b_{KV/token}=61\,[128(128+128)+64]2=4{,}005{,}504\text{ bytes}.
\]

At context 2,048, virtual KV capacity is 8,203,272,192 bytes. Lazy per-layer
buffers commit pages as positions are written. A v3 state at position `p`
contains a 36-byte header followed by exactly `p × 4,005,504` KV bytes.

## Storage and scheduling architecture

The hot path has no per-token allocation. The output head (760.166 MB), F32
norms/biases, codebooks, and a bounded fixed cache remain resident. The default
fixed cache is 256 MiB and chooses the smallest fixed projections first,
maximizing eliminated requests under a byte budget. On the acceptance host it
retains 70 projections / 263 MB.

Two 16-KiB-aligned 368 MB layer slabs implement deterministic double
buffering. While Metal computes layer `L`, the reader fetches unconditional
fixed projections for layer `L+1` into the other slab. A slab is never reused
until its current GPU consumer has completed.

Each of the three shards has four `F_NOCACHE`, no-read-ahead lanes. Every lane
has a background fixed-projection queue and an urgent queue. Once routing has
made expert IDs exact, the 24 gate/up/down slices enter the urgent queue and
are serviced before queued background work, except that a current `pread`
cannot be interrupted. Any short read or hard I/O error aborts the process;
corrupt or partial weights never reach a forward pass.

Prefill is layer-major in tiles of 128. Fixed weights are read once per tile.
At an MoE layer, tokens are grouped by exact expert ID, the tile's expert union
is loaded once, IQ1_S gate/up/SwiGLU and down execute as batched 2-D kernels,
and results scatter back to token/rank slots. Decode remains batch one.

## Measured optimization delta

The parity comparison starts from the same v3 checkpoint at position 1,280
with greedy decoding. The old and optimized paths produced identical anchors:

| Sample position | Token | Logit |
|---:|---:|---:|
| 1,280 | 270 | 48.1055 |
| 1,281 | 1,990 | 51.7904 |
| 1,282 | 344 | 45.4884 |

| Metric / decode step | Previous path | V0 optimized path |
|---|---:|---:|
| model bytes | 13,849,970,112 | 13,587,632,064 |
| logical reads | 1,939 | 1,869 |
| best steady wall interval | about 3.76–3.96 s | about 3.09–3.22 s |
| best effective NVMe interval | about 3.1–3.3 GB/s | about 4.4–4.6 GB/s |
| late long-run wall / NVMe | not instrumented equally | about 4.2–4.5 s / 3.1–3.4 GB/s |
| GPU kernel time | about 0.48–0.54 s | about 0.47–0.52 s |
| Metal command buffers | 178 | 178 |
| hot-path allocations | 0 | 0 |
| exact KV bytes added | 4,005,504 | 4,005,504 |

The 256 MiB cache removes 262,338,048 model bytes and 70 reads per decode
step. A 2 GiB experiment reduced traffic further to 11,709,501,376 bytes but
caused materially more memory compression on the 24 GB host, so it was not
chosen as the safe default. `METALBLOK_FIXED_CACHE_MB=0..2048` remains an
explicit measured tuning control, guarded by live available memory plus a
2 GiB cache margin.

The long run also found a limit that a short benchmark missed. Through
position 1,848, host-wide VM counters recorded 2,129 swap-ins and 116
swap-outs. The counters include other macOS processes and do not identify
which process owned a page, but they are still a real contention signal. The
model remained coherent and atomic state remained valid; the impact is a
slower sustained storage path. The best interval is retained as optimization
evidence, while the late interval is the conservative 24 GB demo expectation.

## Telemetry contract

Every decode step records the release metrics, not an unstructured debug dump:

- end-to-end wall time and GPU execution time;
- explicit host I/O wait;
- useful/model bytes, actual `pread` bytes, request count, span, and effective
  GB/s;
- exact routed-expert urgent bytes and reads;
- aggregate/max read service time and peak outstanding reads;
- command buffers, kernel dispatches, and hot-path allocations;
- exact KV bytes added and available-memory delta;
- pageout, compression, decompression, swap-in, and swap-out deltas;
- sampled token ID and greedy logit.

`--profile-layers` adds fixed-stage span/block time and attention/FFN GPU/I/O
attribution per layer. `--trace` adds residual fingerprints, non-finite counts,
RMS/min/max, hashes, and top logits. These controls are opt-in so ordinary
generation does not pay for deep diagnosis.

## Continuation semantics

State v3 records magic/version/model geometry/context/current position/pending
token and the exact committed KV prefix. Writes go to `.partial`, then
`fflush`, `fsync`, close, and rename. Normal completion always saves final
state. Periodic saves bound work loss during interruption. `--continue-decode`
emits the stored pending token and continues the same decode loop;
reusing a state without that flag appends a formatted user turn.

## Verification performed

- exact non-hydrating manifest and physical-residency preflight;
- 1,025-tensor family/shape/config validation;
- CPU/Metal checks for F32, Q4_K, Q5_K, Q6_K, IQ2_XXS, and IQ1_S real
  checkpoint matrices;
- independent CPU versus Metal grouped-router IDs and weights;
- native tokenizer exact-count construction;
- C++ unit tests, Python bytecode compilation, and whitespace audit;
- 1,000-token real prefill;
- multi-step old/new token-and-logit parity from an exact saved state;
- deliberate rejection and deletion of a numerically divergent fusion;
- long decode with per-token storage, GPU, VM-pressure, and atomic-state logs.

Full independent-runtime final-logit parity remains a separate stronger claim;
the release evidence establishes native internal parity across the scheduling
optimization and a real complete-generation proof.

## Acceptance artifacts

| Artifact | Path / result |
|---|---|
| exact 1,000-token prefill log | `metalblok/runs/run-20260812-012449-18510.log` |
| authoritative v3 state | `metalblok/runs/proof-1k-20260812-012449.state` |
| continuation log | `metalblok/runs/run-20260812-021356-20767.log` |
| continuation text | `metalblok/runs/run-20260812-021356-20767.txt` |
| required final state position | 1,999 |
| final result | **RUNNING — replace only after position 1,999 is verified** |

## Non-claims and next boundary

V0 does not claim 163,840 or one-million-token context validation, concurrent
requests, a web UI, ANE execution, direct NAX intrinsics, physical NAND
placement control, temporal expert caching, speculative expert correctness, or
multi-model support. The separate ANE is not a general raw-Metal target; moving
intermediate tensors through a second framework would add synchronization and
data movement without a verified win.

The next architecture boundary is not more scaffolding. It is a parity-tested
repacked expert format, larger safe fixed/cache capacity on a higher-memory
host, true quantized matrix-matrix fixed projections for prefill, and a
finite-precision-compatible compact MLA representation. Each enters only if it
improves bytes, requests, overlap, or compute while preserving token parity.
The full staged path to one-million input plus one-million output tokens is in
[`MILLION_TOKEN_SCALE_PLAN.md`](MILLION_TOKEN_SCALE_PLAN.md).
