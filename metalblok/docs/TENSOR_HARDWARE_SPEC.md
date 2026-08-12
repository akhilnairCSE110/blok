# MetalBlok tensor and hardware execution specification

## Normative contract for the accepted DeepSeek-R1 V0

**Artifact date:** 2026-08-12

**Machine:** 10-core Apple M5, 24 GB unified memory, internal NVMe/APFS

**Checkpoint:** DeepSeek-R1 671B `UD-IQ1_S`, three GGUF shards

**Numerical mode:** FP32 activations/reductions/logits, FP16 persistent K/V

**Companion:** [complete architecture and rationale](V0_ARCHITECTURE.md)

This is the precise execution contract, not a proposal. It replaces the old
absorbed-latent design. The accepted V0 expands the checkpoint's combined
KV-B projection at each position and stores exact expanded K/V in FP16. That
choice costs memory, but preserves the finite-precision graph that passed the
token/logit gate.

## 1. Conformance language and coordinate systems

“Must” is required for model correctness or buffer safety. “Current” is the
executing V0. “Future” is never evidence about the current binary.

Mathematical matrices are written `[N,K]`: `N` output rows, `K` input
coordinates. GGUF reports the contiguous dimension first, so the same matrix
appears as `[K,N]`. An expert bank reported as `[K,N,E]` is `E` independent
`[N,K]` matrices laid out in expert-major slices. Sequence position `p` means
that positions `[0,p)` have committed K/V; the forward at `p` appends one row.

## 2. Frozen geometry

| Symbol | Quantity | Value |
|---|---|---:|
| \(L\) | transformer layers | 61 |
| \(d\) | residual width | 7,168 |
| \(V\) | vocabulary | 129,280 |
| \(H\) | query heads | 128 |
| \(r_q\) | query latent width | 1,536 |
| \(r_{kv}\) | KV latent width | 512 |
| \(d_n\) | non-rotary Q/K width per head | 128 |
| \(d_r\) | rotary Q width/head and shared rotary K width | 64 |
| \(d_v\) | value width per head | 128 |
| \(d_f\) | dense FFN width | 18,432 |
| \(d_e\) | expert FFN width | 2,048 |
| \(E\) | routed experts/layer | 256 |
| \(k\) | selected routed experts | 8 |
| \(G\) | expert groups | 8 |
| \(G'\) | groups retained before top-k | 4 |
| \(\epsilon\) | RMSNorm epsilon | \(10^{-6}\) |
| \(\theta\) | RoPE base | 10,000 |
| \(f\) | YaRN extension factor | 40 |

The manifest must contain exactly

\[
3+61(9)+3(3)+58(8)=1{,}025
\]

tensors: three globals, nine attention/common tensors per layer, three dense
FFN tensors in layers 0–2, and eight MoE tensors in layers 3–60. The runtime
rejects geometry, routing metadata, YaRN metadata, tensor family, shape, type,
or shard-count drift before inference.

## 3. Tensor manifest by family

### 3.1 Global tensors

| GGUF name | GGUF shape | Mathematical shape | Stored type in artifact | Runtime role |
|---|---:|---:|---|---|
| `token_embd.weight` | `[7168,129280]` | \(V\times d\) | Q4_K | one compressed row read per input token |
| `output_norm.weight` | `[7168]` | \(d\) | F32 | resident final RMS gain |
| `output.weight` | `[7168,129280]` | \(V\times d\) | Q6_K | resident, distinct output projection |

The embedding and output matrices are not assumed tied. The output head is
760,166,400 bytes and remains resident because streaming it for every emitted
token would add a large, perfectly reusable read to the critical path.

### 3.2 Attention/common tensors, every layer

| Suffix | GGUF shape | Mathematical shape | Function |
|---|---:|---:|---|
| `attn_norm.weight` | `[7168]` | \(d\) | pre-attention RMS gain |
| `attn_q_a.weight` | `[7168,1536]` | \(r_q\times d\) | residual to query latent |
| `attn_q_a_norm.weight` | `[1536]` | \(r_q\) | query-latent RMS gain |
| `attn_q_b.weight` | `[1536,24576]` | \(H(d_n+d_r)\times r_q\) | query expansion |
| `attn_kv_a_mqa.weight` | `[7168,576]` | \((r_{kv}+d_r)\times d\) | compact content plus rotary key |
| `attn_kv_a_norm.weight` | `[512]` | \(r_{kv}\) | content RMS gain |
| `attn_kv_b.weight` | `[512,32768]` | \(H(d_n+d_v)\times r_{kv}\) | combined expanded K/V |
| `attn_output.weight` | `[16384,7168]` | \(d\times Hd_v\) | attention output projection |
| `ffn_norm.weight` | `[7168]` | \(d\) | pre-FFN RMS gain |

The descriptor type, not the suffix, selects F32/Q4_K/Q5_K/Q6_K/IQ2_XXS/
IQ1_S execution. Mixed quantization is part of this checkpoint.

### 3.3 Dense FFN, layers 0–2

| Suffix | GGUF shape | Mathematical shape |
|---|---:|---:|
| `ffn_gate.weight` | `[7168,18432]` | \(d_f\times d\) |
| `ffn_up.weight` | `[7168,18432]` | \(d_f\times d\) |
| `ffn_down.weight` | `[18432,7168]` | \(d\times d_f\) |

### 3.4 Sparse FFN, layers 3–60

| Suffix | GGUF shape | Mathematical content | Read policy |
|---|---:|---:|---|
| `ffn_gate_inp.weight` | `[7168,256]` | router \(W_R\in\mathbb R^{E\times d}\) | unconditional fixed stage/cache |
| `exp_probs_b.bias` | `[256]` | no-aux correction bias | resident F32 |
| `ffn_gate_shexp.weight` | `[7168,2048]` | shared gate | unconditional fixed stage/cache |
| `ffn_up_shexp.weight` | `[7168,2048]` | shared up | unconditional fixed stage/cache |
| `ffn_down_shexp.weight` | `[2048,7168]` | shared down | unconditional fixed stage/cache |
| `ffn_gate_exps.weight` | `[7168,2048,256]` | routed gates | exact selected slice only |
| `ffn_up_exps.weight` | `[7168,2048,256]` | routed ups | exact selected slice only |
| `ffn_down_exps.weight` | `[2048,7168,256]` | routed downs | exact selected slice only |

For expert `e`, tensor descriptor base `a`, and total expert-bank bytes `S`,

\[
a_e=a+eS/E,\qquad S_e=S/E.
\]

These offsets become actionable only after the exact router result exists.

## 4. Stored quantization and multiplication contract

For all block formats, `QK=256`. Bytes per block are:

| Type | Bytes/256 weights | Stored bytes/weight |
|---|---:|---:|
| Q4_K | 144 | 0.5625 |
| Q5_K | 176 | 0.6875 |
| Q6_K | 210 | 0.8203125 |
| IQ2_XXS | 66 | 0.2578125 |
| IQ1_S | 50 | 0.1953125 |

For row `i`, the kernel computes

\[
y_i=\sum_{b=0}^{K/256-1}\sum_{j=0}^{255}
D_q(B_{i,b})_j x_{256b+j}.
\]

`D_q` is fused into the multiply. No complete FP16 or FP32 weight matrix is
written to unified memory. A 32-lane SIMD group owns one output row; lanes
stride over blocks, accumulate in FP32, and reduce in a fixed order. This
choice follows the batch-one roofline: each weight is usually reused once, so
materialized dequantization would amplify both capacity and memory traffic.

IQ1_S and IQ2_XXS use their GGML lookup tables. Their small codebooks are
copied into Metal-owned shared buffers once. The copy is intentional: wrapping
read-only executable storage directly produced a GPU resource fault on this
host and did not provide a safe lifetime contract.

## 5. One-position forward equations and resources

All transient vectors below are FP32.

### 5.1 Embedding and normalization

Only compressed embedding row `u` is read and decoded:

\[
x^{(0)}=D_{Q4_K}(W_E[u,:]).
\]

RMSNorm uses one 256-thread group:

\[
\operatorname{RMS}(x;g)_i=x_i g_i
\left(\frac1n\sum_{j=0}^{n-1}x_j^2+\epsilon\right)^{-1/2}.
\]

The sum of squares and application remain FP32. The residual write followed by
RMSNorm remains an explicit boundary because the attempted fusion changed an
accepted downstream logit.

### 5.2 Query construction

\[
\bar x=\operatorname{RMS}(x;g_{attn}),
\quad q_a=W_{QA}\bar x,
\quad \hat q_a=\operatorname{RMS}(q_a;g_{QA}),
\]

\[
q_{full}=W_{QB}\hat q_a
\in\mathbb R^{128\times192}.
\]

Each head splits into `q_nope[128]` and `q_rope[64]`. The live GGUF path uses
consecutive rotary pairs `(2i,2i+1)`. The older split-half/NEOX kernels in
historical experiments are not dispatched by this runtime.

For pair `i`,

\[
r_i=1-\operatorname{clamp}\left(\frac{i-10}{13},0,1\right),
\]

\[
\phi_{p,i}=p\,10000^{-2i/64}
\left[\frac1{40}+\frac{39}{40}r_i\right],
\]

\[
(a',b')=(a\cos\phi-b\sin\phi,
b\cos\phi+a\sin\phi).
\]

### 5.3 Key/value construction

\[
[c^{raw};k^{r,raw}]=W_{KVA}\bar x,
\quad c=\operatorname{RMS}(c^{raw};g_{KVA}).
\]

The 64-coordinate shared rotary key is rotated and stored. The combined
projection is then evaluated exactly where the checkpoint graph places it:

\[
[k^n_{p,h};v_{p,h}]=W_{KVB,h}c.
\]

`mha_kv_store_f16` converts expanded non-RoPE keys and values to persistent
FP16. Keeping expansion before cache storage preserves quantized GEMV and
rounding order. Real-number absorption is not a license to reassociate these
finite-precision operations.

### 5.4 Stable causal attention

The scale is

\[
\beta=\frac{(1+0.1\ln40)^2}{\sqrt{192}}.
\]

For each head and history position `j <= p`,

\[
s_{h,j}=\beta\left[(q_h^n)^T k_{h,j}^n+
(q_h^r)^Tk_j^r\right].
\]

One 256-thread group owns a head and performs three deterministic passes over
one FP32 score row:

1. compute scores and reduce `m=max(s)`;
2. replace scores by `exp(s-m)` and reduce `Z=sum(exp(s-m))`;
3. accumulate `o_h=sum(exp(s-m)v_h)/Z`.

The output is

\[
a=x+W_O\operatorname{concat}(o_0,\ldots,o_{127}).
\]

The 128-head score scratch is shared across layers because layers are
sequential. K, V, and rotary K are separate lazy buffers per layer, avoiding a
single 8.2 GB resource binding.

### 5.5 Dense and sparse feed-forward paths

Dense layers compute

\[
x'=a+W_D[\operatorname{SiLU}(W_G\bar a)\odot W_U\bar a],
\quad \bar a=\operatorname{RMS}(a;g_{ffn}).
\]

For an MoE layer, router logits are `r=W_R bar(a)` and

\[
p_e=\sigma(r_e),\qquad c_e=p_e+b_e.
\]

Experts are eight contiguous groups of 32. Each group score is the sum of its
two greatest corrected scores. The four greatest groups survive; the eight
greatest corrected expert scores in their union are selected. Every tie is
resolved toward the lower ID. Mixture coefficients use the uncorrected
probability:

\[
\alpha_e=2.5\frac{p_e}{\sum_{j\in A}p_j}.
\]

The exact output is

\[
x'=a+E_s(\bar a)+\sum_{e\in A}\alpha_eE_e(\bar a).
\]

For IQ1_S experts, gate/up GEMVs and SwiGLU are fused so only the activated
2,048-vector is written. Down projection and weighted accumulation are also
fused. These are accepted fusions because they keep the established FP32
reductions and pass the token/logit anchors.

### 5.6 Output token

After layer 60:

\[
\ell=W_{out}\operatorname{RMS}(x^{(61)};g_{out})\in\mathbb R^{129280}.
\]

Greedy mode selects the lowest vocabulary ID at the maximum FP32 logit.
Sampling is optional and seeded; the release proof uses temperature zero.

## 6. Decode storage pipeline and ownership

### 6.1 Physical path

```text
GGUF shard file offset
  -> APFS pread with F_NOCACHE and no blind readahead
  -> 16-KiB-aligned MTLStorageModeShared allocation
  -> release/acquire ready handoff
  -> Metal encoder binding
  -> GPU caches, registers, and threadgroup memory
```

Unified memory removes a host-to-VRAM copy, not SSD or fabric traffic.
MetalBlok never claims raw NAND control.

### 6.2 Resident and streamed tiers

The resident tier contains the output head, norms/biases, codebooks, hot
activations, exact KV resources, and a bounded cache of small fixed
projections. The streamed tier contains two reusable fixed-layer slabs plus 24
router-selected expert slice slots.

The default fixed cache is 256 MiB. Candidates are stable-sorted smallest
first, maximizing avoided objects under a byte budget. Admission also reserves
host headroom and a 2 GiB safety guard; a 2 GiB experimental cache reduced
bytes but caused unacceptable memory compression on this 24 GB host.

### 6.3 Fixed-layer double buffer

With slabs `S0,S1`:

```text
load fixed layer 0 -> S0
for L in 0..60:
    wait until fixed L is ready in S[L & 1]
    launch fixed L+1 into S[(L+1) & 1]
    encode and complete every consumer of fixed L
```

A slab cannot be overwritten until its prior GPU consumer completes. The
ideal overlapped stage cost is `max(read,compute)`, not `read+compute`.

### 6.4 Reader topology and priority

Four file descriptors per shard give 12 workers. Each lane has a background
queue for unconditional fixed reads and an urgent queue for current
router-selected experts. A worker finishes its current `pread`, then drains
urgent work before more background work. This bounds interference from legal
lookahead without pretending an in-flight syscall can be preempted.

Read completion is published only when `pread` returns the exact requested
length. A short read or error aborts immediately; it can never become a ready
partial weight.

### 6.5 MoE overlap and serial boundary

The router depends on the current layer's residual, so exact expert addresses
cannot be known earlier. Once routing completes, all `8 x 3 = 24` slices are
submitted while the GPU evaluates the always-active shared expert:

\[
t_{moe}\approx t_{router}+\max(t_{shared},t_{expertIO})+t_{routed}.
\]

Predicting the next router is speculative and is not part of V0.

### 6.6 Command-buffer contract

Dense layers place attention and dense FFN in one command buffer. Every MoE
layer uses three command buffers:

1. attention, FFN norm, router projection, exact top-k;
2. shared expert while urgent reads execute;
3. routed experts and residual merge.

The token total is

\[
3+58(3)+1=178
\]

command buffers. Dependencies inside a phase remain on one encoder/queue.
There are zero hot-path allocations per decode step.

## 7. Layer-major 128-token prefill

Prefill is tiled because token-major execution would stream the active model
once per prompt token. For each tile, layers are outermost, so each fixed
projection is fetched once per tile.

At an MoE layer, token `t` has exact set `A_t`; the physical set is

\[
U_B=\bigcup_{t\in B}A_t,\qquad |U_B|\le\min(256,8B).
\]

Assignments are grouped by expert. The 2-D expert kernels map one axis to
output row and the other to assigned token, reusing compressed expert rows.
Each result is scattered to `[token,top-k-rank,hidden]`, then ranks 0–7 merge
in reference order. Grouping therefore changes weight reuse, not the
per-token summation order. The accepted 1,000-token prefill took 591.94 s,
1.69 tokens/s.

## 8. Exact KV and state format

Each layer/position stores

\[
128(128)\text{ K}+128(128)\text{ V}+64\text{ rotary K}
=32{,}832
\]

FP16 values. Therefore

\[
b_{KV/position}=61(32{,}832)(2)=4{,}005{,}504\text{ bytes},
\]

and context 2,048 reserves 8,203,272,192 bytes of KV address space. Buffers
are lazy; physical pages are committed as positions are written.

State v3 is a 36-byte header followed by the exact prefix, layer by layer:

```text
header(magic, version, layers, context, kv_rank, rope_dim, position, pending)
for layer 0..60:
    K_nonrope[0:position,128,128] fp16
    V[0:position,128,128]         fp16
    K_rope[0:position,64]         fp16
```

\[
S_{state}(p)=36+p(4{,}005{,}504).
\]

The pending token is the logit result already computed for the next position.
It lets continuation remain the same decode loop rather than recomputing or
silently dropping a token. State commits use write-to-`.partial`, `fflush`,
`fsync`, close, and atomic rename.

## 9. Memory admission ledger

Before allocations, V0 budgets output head, two fixed slabs, exact KV, score
scratch, expert arena, prefill scratch, runtime margin, host reserve, and
cache guard. For context 2,048 on the acceptance host:

| Object | Bytes/size |
|---|---:|
| output head | 760,166,400 |
| two fixed slabs | about 736 MB |
| fixed cache | about 263 MB |
| exact KV | 8,203,272,192 |
| expert arena | 90.833 MB |
| prefill scratch | 124.95 MB |
| runtime margin | 33.55 MB |
| estimated active total | about 10.22 GB |

Every position logs pageouts, compression/decompression, swap-ins/outs, and
available-memory delta. No swap-out is accepted as invisible performance.

## 10. Release instrumentation and invariants

Per-position metrics are:

```text
wall_us gpu_us io_wait_us
model_bytes nvme_bytes nvme_reads nvme_span_us nvme_gbps
urgent_bytes urgent_reads io_service_us io_max_us io_peak
cmdbufs dispatches allocations kv_bytes available_delta
pageouts compressions decompressions swapins swapouts
sample position mode token logit
```

`nvme_span_us` is first submitted read to final completion and is the correct
aggregate bandwidth denominator. `io_service_us` sums parallel worker service
and may exceed wall time. `io_wait_us` is only producer blocking. `gpu_us` is
Metal command-buffer execution, not wall time.

`--profile-layers` adds fixed-stage span/block, attention, FFN, expert I/O,
command-buffer, and dispatch attribution per layer. `--trace` records residual
RMS/min/max/non-finite count/hash plus routers and top logits. Required
invariants include:

- 1,025 validated tensors and all three resident shard extents;
- exact read length before ready publication;
- one owner per reusable slab/expert slot;
- selected expert ID `< 256`, eight unique selections;
- finite activations/logits and monotonic position;
- exactly 4,005,504 KV bytes added per advanced position;
- 178 command buffers and zero allocations in steady decode;
- unchanged accepted token ID/logit anchors after optimization.

## 11. Measured V0 operating point

The safe optimized path moves 13,587,632,064 model bytes in 1,869 reads per
decode step. Of those, 4,035,182,592 bytes / 1,392 reads are urgent selected
expert traffic. Before late-run memory pressure, representative positions
measured 3.09–3.22 s wall, 2.96–3.07 s NVMe span, 4.4–4.6 GB/s effective
NVMe, and 0.47–0.52 s GPU execution. This proves the decode is storage-bound:

\[
t_{token}\gtrsim\max(N_w/B_{NVMe},t_{GPU})+t_{control}.
\]

The previous schedule moved 13,849,970,112 bytes in 1,939 reads. The bounded
cache and priority pipeline remove 262,338,048 bytes and 70 reads while
preserving the checked greedy tokens/logits.

## 12. Explicit non-contracts

V0 does not claim latent-KV parity, million-token capacity, 163,840-token
validation, ANE execution, PTX, undocumented GPU matrix instructions,
physical NAND control, cross-token expert prediction, temporal expert-weight
caching, concurrent requests, or multi-model serving. Those are future work
until they pass the same byte, latency, memory-pressure, and token-parity
gates. The active execution and the reasoning for every rejection are detailed
in [V0_ARCHITECTURE.md](V0_ARCHITECTURE.md).
