# MetalBlok V0 architecture

## Exact DeepSeek-R1 inference when the checkpoint is 5.8× larger than memory

**Normative implementation:** `metalblok/src/gguf_runtime.cpp`,
`metalblok/src/kernels.metal`, `metalblok/src/pread_ring.cpp`  
**Target:** 10-core Apple M5, 24 GB unified memory, internal NVMe/APFS  
**Artifact:** DeepSeek-R1 671B `UD-IQ1_S`, 140.231 GB, three GGUF shards  
**Acceptance:** exactly 1,000 native input tokens and 1,000 emitted tokens

This document describes the executing V0, including its numerical boundaries,
storage schedule, Metal schedule, state format, performance model, negative
results, and the reasoning behind every material decision. Proposed future
work is explicitly separated from implemented behavior.

## 1. Optimization objective and release ordering

The objective is lexicographic, not a weighted average:

1. Preserve the selected token sequence and accepted logits.
2. Fit the exact working set without unsafe VM pressure.
3. Minimize immutable model bytes moved per token.
4. Maximize realized NVMe bandwidth and overlap unavoidable reads with GPU
   work.
5. Minimize command buffers, kernel dispatches, allocations, and host
   synchronization without changing rounding boundaries.
6. Reduce KV bytes per position only under a finite-precision parity proof.
7. Minimize joules/token after the preceding gates pass.

The measured reward vector is

\[
R=(\text{token parity},\;r_{decode},\;r_{prefill},\;-N_w,\;B_{NVMe},
-t_{GPU},\;-t_{IOwait},\;-n_{cb},\;-n_{dispatch},\;-n_{alloc},
-b_{KV},\;C,\;-\Delta M,\;-J/token).
\]

Token parity is a gate. A change that improves every performance coordinate
but changes an accepted logit is rejected. This is why the implementation
does not treat real-number associativity as finite-precision equivalence.

## 2. Hardware facts that shape the software

### 2.1 Capacity

The stored checkpoint is 140,231,438,464 bytes. Physical unified memory is
24 GB. Therefore neither the quantized model nor a dequantized form can be
resident. The system must execute from a moving working set:

\[
W_t=W_{fixed,t}\cup W_{router-selected,t}\cup W_{head}.
\]

The model's own router makes the dynamic part knowable. Virtual memory cannot
make the capacity mismatch disappear; mapping the entire payload merely turns
explicit scheduling into uncontrolled page faults and cache pollution.

Apple's specification for this exact class of machine reports a 10-core GPU,
24 GB as a supported unified-memory configuration, and 153 GB/s unified-memory
bandwidth. Those are hardware ceilings, not application throughput. V0 logs
about 3.2–4.6 GB/s for its file-read span and about 0.5 seconds of GPU work for
13.588 GB of streamed model payload, so the measured critical boundary is the
SSD/file schedule rather than the advertised unified-memory ceiling.

### 2.2 Unified memory

[`MTLStorageModeShared`](https://developer.apple.com/documentation/metal/mtlstoragemode/shared)
is visible to the CPU and GPU. A weight read lands in
the same allocation later bound to the Metal encoder. There is no explicit
host-to-discrete-VRAM copy:

```text
SSD -> APFS pread(F_NOCACHE) -> shared MTLBuffer -> GPU load
```

This removes one copy but not data movement. NAND/controller/filesystem DMA,
memory-fabric traffic, GPU cache fills, and quant decode still occur. “Zero
copy” means one shared allocation, not zero bytes moved.

### 2.3 Storage

V0 controls file offsets and request ordering, not NAND pages, FTL placement,
ECC, or NVMe firmware. Each shard is opened on four independent lanes with:

```text
O_RDONLY
F_NOCACHE = 1
F_RDAHEAD = 0
```

`F_NOCACHE` avoids retaining a 140 GB rolling payload in the unified buffer
cache. Readahead is disabled because the scheduler explicitly knows every
required range and blind neighboring reads consume memory bandwidth without a
model guarantee.

### 2.4 Separate ANE versus GPU matrix hardware

The separate Apple Neural Engine is not a documented raw-Metal execution
target. Moving an intermediate through Core ML would add framework conversion,
synchronization, and shared-memory traffic at every layer boundary. V0 keeps
the complete graph on the Metal command queue. Any future ANE split must show
lower end-to-end time and identical tokens, including transfer costs.

Apple separately documents a GPU Neural Accelerator and a
[Metal Performance Primitives path](https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf)
for authoring kernels that use it on M5. The current GGUF IQ1_S/IQ2_XXS/K-quant
decode is fused into custom MSL row reductions and does not claim that MPP
tensor path. Repacking an exact weight family into a supported block-scaled
tensor representation could make that a future compute experiment, but it
must include repack/storage bytes and parity. With current GPU time already
mostly hidden under a much longer SSD span, an unmeasured accelerator rewrite
is lower leverage than removing model bytes or I/O stalls.

The durable TensorOps, Metal 4, Metal I/O, tiling, and profiling research notes
are in [M5 TensorOps and Metal I/O engineering notes](M5_TENSOROPS_AND_METAL_IO.md).

## 3. Frozen model geometry

| Symbol | Meaning | Value |
|---|---|---:|
| \(L\) | transformer layers | 61 |
| \(d\) | residual width | 7,168 |
| \(V\) | vocabulary | 129,280 |
| \(H\) | query heads | 128 |
| \(r_q\) | query low-rank width | 1,536 |
| \(r_{kv}\) | KV low-rank width | 512 |
| \(d_n\) | non-RoPE Q/K width per head | 128 |
| \(d_r\) | RoPE width | 64 |
| \(d_v\) | value width per head | 128 |
| \(d_f\) | dense FFN width | 18,432 |
| \(d_e\) | routed/shared expert width | 2,048 |
| \(E\) | routed experts | 256 |
| \(k\) | selected experts | 8 |
| \(G/G'\) | expert groups / retained groups | 8 / 4 |
| \(\epsilon\) | RMSNorm epsilon | \(10^{-6}\) |
| \(\theta\) | RoPE base | 10,000 |
| \(f\) | YaRN factor | 40 |

The required tensor count is

\[
3+61(9)+3(3)+58(8)=1{,}025.
\]

The three terms are global tensors, nine common families per layer, three
dense FFN matrices in each leading dense layer, and eight MoE families in each
sparse layer. The runtime validates count, family, shape, stored type support,
architecture constants, and the distinct output head before allocation.

## 4. Stored representations and quantized multiplication

The checkpoint mixes F32, Q4_K, Q5_K, Q6_K, IQ2_XXS, and IQ1_S. A descriptor's
type selects its kernel; tensor names never imply quant type. For block types,
one row is a sequence of 256-weight blocks:

\[
y_i=\sum_b\sum_{j=0}^{255}D_q(B_{i,b})_j x_{256b+j}.
\]

`D_q` executes inside the dot-product kernel. The full dequantized matrix is
never materialized. One 32-lane SIMD group owns one output row. Lane \(p\)
processes blocks \(p,p+32,\ldots\), accumulates FP32 partials, and a fixed
SIMD/threadgroup reduction produces FP32 output.

This organization follows the decode roofline. With batch one, a stored weight
is generally consumed for one dot product, so the operation is bandwidth
limited. Expanding an IQ1_S matrix to FP16 before multiply would replace about
0.195 stored byte/weight with 2 bytes/weight and add a full write/read pass.
Fusing dequantization with multiplication is therefore both a capacity and a
bandwidth requirement.

The 16 KiB IQ1_S and IQ2_XXS codebooks are copied once into Metal-owned shared
buffers. Wrapping read-only executable/data pages directly caused a GPU fault
on this platform; the explicit copy establishes a safe resource lifetime.

### 4.1 Routed IQ1_S fusion

For the dominant expert type, gate and up projections share one input:

\[
g=W_gx,\qquad u=W_ux,\qquad z=\operatorname{SiLU}(g)\odot u.
\]

`expert_gate_up_swiglu_iq1_s` decodes both rows, performs both dot products,
and writes only \(z\). It avoids materializing two intermediate vectors and
removes separate gate, up, and SwiGLU dispatches. The down kernel computes

\[
y\leftarrow y+\alpha W_dz
\]

and fuses the router coefficient into the output accumulation. These fusions
preserve the established reduction and FP32 boundaries; they are unlike the
rejected residual/RMS fusion that changed downstream logits.

## 5. Exact forward pass

All transient activations and reductions are FP32. Only the persistent K/V
cache is FP16. For token \(u_t\), the CPU decodes one quantized embedding row:

\[
x_t^{(0)}=W_E[u_t,:]\in\mathbb{R}^{7168}.
\]

For every layer \(\ell\), the pre-normalized residual equations are

\[
\bar x=\operatorname{RMSNorm}(x;g_\ell^{attn}),
\]

\[
a=x+\operatorname{Attention}_\ell(\bar x),
\]

\[
\bar a=\operatorname{RMSNorm}(a;g_\ell^{ffn}),
\]

\[
x'=a+\operatorname{FFN}_\ell(\bar a).
\]

RMSNorm is

\[
\operatorname{RMSNorm}(x;g)_i=
x_i g_i\left(\frac1n\sum_jx_j^2+\epsilon\right)^{-1/2}.
\]

A 256-thread group computes the FP32 sum of squares and applies the F32 gain.
The explicit residual write and later norm remain separate because their
rounding/reduction schedule is part of observed token parity.

## 6. DeepSeek attention path

### 6.1 Query path

\[
q_a=W_{QA}\bar x\in\mathbb{R}^{1536},
\qquad
\hat q_a=\operatorname{RMSNorm}(q_a;g_{QA}),
\]

\[
q_{full}=W_{QB}\hat q_a
\in\mathbb{R}^{128\times(128+64)}.
\]

For each head, split \(q_{full,h}=[q^n_h;q^r_h]\). The checkpoint's active
GGUF layout uses consecutive rotary pairs, not the unused split-half helper:

\[
(a_i,b_i)=(q^r_{2i},q^r_{2i+1}),
\]

\[
(a'_i,b'_i)=(a_i\cos\phi_i-b_i\sin\phi_i,
b_i\cos\phi_i+a_i\sin\phi_i).
\]

### 6.2 YaRN frequency law

For pair \(i\in[0,31]\), V0 uses

\[
r_i=1-\operatorname{clamp}\left(\frac{i-10}{23-10},0,1\right),
\]

\[
\phi_{t,i}=t\theta^{-2i/64}
\left[\frac1{40}+\left(1-\frac1{40}\right)r_i\right].
\]

The attention scale is

\[
\beta=\frac{(1+0.1\ln40)^2}{\sqrt{128+64}}.
\]

The square is intentional: metadata-derived YaRN magnitude is squared once
for a Q/K dot product and stored as the runtime scale multiplier.

### 6.3 KV path and why the cache is expanded

\[
[c^{raw};k^{r,raw}]=W_{KVA}\bar x,
\qquad c=\operatorname{RMSNorm}(c^{raw};g_{KVA}).
\]

The shared 64-coordinate key is rotated with the same consecutive-pair law.
The checkpoint stores a combined quantized matrix

\[
W_{KVB}\in\mathbb{R}^{[128(128+128)]\times512}.
\]

V0 evaluates it at the checkpoint's established FP32 boundary:

\[
[k^n_{t,h};v_{t,h}]=W_{KVB,h}c_t,
\]

then converts the resulting K and V rows to FP16 for persistent state.

Over real numbers, one can absorb \(W_K^T\) into Q and move \(W_V\) after the
weighted sum. In this checkpoint, doing so changes quant-decode, accumulation,
and rounding order. The algebra is information-preserving, but bit/token
parity is not automatic. Earlier absorbed-state documentation described a
research path, not the accepted V0. The expanded cache is the conservative
finite-precision contract.

### 6.4 Attention kernel

For head \(h\), cached position \(j\le t\):

\[
s_{h,j}=\beta\left[(q_h^n)^Tk_{h,j}^n+(q_h^r)^Tk_j^r\right].
\]

One 256-threadgroup owns one head and performs three stable passes:

1. write FP32 scores and reduce \(m=\max_js_j\);
2. replace each score with \(e^{s_j-m}\) and reduce
   \(Z=\sum_je^{s_j-m}\);
3. accumulate \(o_h=Z^{-1}\sum_je^{s_j-m}v_{h,j}\).

Scores use one shared `[128, context]` FP32 scratch resource. K/V lives in
separate per-layer lazy resources, preventing one monolithic binding from
forcing the entire 8.2 GB capacity resident.

The output projection and residual are

\[
a=x+W_O\operatorname{concat}(o_0,\ldots,o_{127}).
\]

## 7. Dense and sparse FFNs

Layers 0–2 compute full SwiGLU:

\[
\operatorname{FFN}(x)=W_D
[\operatorname{SiLU}(W_Gx)\odot W_Ux].
\]

Layers 3–60 compute one shared expert plus eight routed experts.

### 7.1 Exact grouped router

The router produces \(r=W_Rx\in\mathbb{R}^{256}\). Define

\[
p_e=\sigma(r_e),\qquad c_e=p_e+b_e.
\]

Divide experts into eight contiguous groups of 32. A group score is the sum
of its two largest corrected scores. Retain the four groups with greatest
scores, then choose the eight largest \(c_e\) inside their union. Equal scores
choose the lower ID. Mixture coefficients use unbiased probabilities:

\[
\alpha_e=2.5\frac{p_e}{\sum_{j\in A}p_j},\qquad e\in A.
\]

The output is

\[
y=E_{shared}(x)+\sum_{e\in A}\alpha_eE_e(x).
\]

Every omitted expert has coefficient zero by the model definition. Fetching
only \(A\) is therefore exact after routing:

\[
\sum_{e=0}^{255}\alpha_eE_e(x)=\sum_{e\in A}\alpha_eE_e(x).
\]

The serial boundary is unavoidable for exact demand I/O:

```text
attention -> FFN norm -> router logits -> exact top-8 -> expert offsets
```

No exact scheduler knows the next layer's expert IDs before the residual that
feeds its router exists. V0 does not label previous-token expert prediction as
deterministic prefetch.

## 8. Decode I/O and GPU pipeline

### 8.1 Resident tier

Resident objects are:

- the 760,166,400-byte output projection;
- all F32 normalization gains and correction biases;
- IQ codebooks;
- FP32 activation and prefill scratch;
- lazy exact KV resources;
- a headroom-limited cache of small immutable projections.

The fixed-cache budget is

\[
C_{cache}=\min(C_{desired},
M_{available}-M_{base}-M_{host-reserve}-M_{cache-guard}),
\]

with default \(C_{desired}=256\) MiB and a 2 GiB cache guard. Candidates are
stably sorted by compressed byte length. Keeping the smallest objects first
maximizes requests eliminated under the budget and tends to retain serial
router/small-projection work.

### 8.2 Double-buffered fixed projections

Let `S0` and `S1` be reusable 16-KiB-aligned slabs. The layer schedule is:

```text
read fixed(0) -> S0
for L = 0..60:
    wait fixed(L) in S[L mod 2]
    start fixed(L+1) in S[(L+1) mod 2]
    execute layer L
```

Layer \(L+2\) cannot overwrite \(L\)'s slab until every command buffer using
layer \(L\) has completed. Thus overlap does not weaken ownership. For stage
span \(t_s\) and compute \(t_c\), the ideal non-router contribution changes
from \(t_s+t_c\) to approximately \(\max(t_s,t_c)\), subject to shard
contention and the first/last bubbles.

### 8.3 Priority reader

Each of 12 workers owns two SPSC queues:

- background: unconditional next-layer fixed projections;
- urgent: current router-selected gate/up/down slices.

After the current `pread` finishes, a worker drains urgent work first. This
prevents legal lookahead from increasing the current layer's critical-path
latency. It cannot cancel an in-flight system call, so maximum service time is
logged to expose head-of-line blocking.

Any failed or short read aborts immediately. Returning a ready flag for a
partial model tensor would convert storage failure into silent numerical
corruption and is forbidden.

### 8.4 Shared-expert overlap

After the router command buffer completes, V0 submits all 24 selected slices:

\[
8\text{ experts}\times(\text{gate}+\text{up}+\text{down}).
\]

Those reads run while Metal evaluates the always-active shared expert. Then
the host waits for selected slices and encodes all routed experts. The critical
path becomes approximately

\[
t_{router}+\max(t_{shared},t_{expertIO})+t_{routed},
\]

instead of

\[
t_{router}+t_{expertIO}+t_{shared}+t_{routed}.
\]

### 8.5 Command buffers

Dense layers encode attention and dense FFN in one command buffer. Each MoE
layer requires three:

1. attention, FFN norm, router projection, top-8;
2. shared expert while selected expert I/O proceeds;
3. routed experts and final residual accumulation.

The full decode count is therefore

\[
3\text{ dense}+58(3)\text{ MoE}+1\text{ final}=178
\]

command buffers. The router host-visibility boundary explains the MoE split;
all other dependent kernels stay ordered inside encoders rather than paying a
host round trip per operation.

### 8.6 Exact relationship to [*LLM in a Flash*](https://arxiv.org/abs/2312.11514)

MetalBlok shares that paper's governing inequality: when a checkpoint is
larger than memory, inference time is controlled by the minimum model bytes
that must cross the storage boundary, their physical request layout, and the
reuse that fits without displacing required state. It adopts those ideas only
where the checkpoint supplies an exact decision.

DeepSeek's router gives a stronger predicate than predicting active neurons in
a dense FFN. After exact top-8 selection, the remaining 248 routed experts have
coefficient zero, so omitting their records cannot change arithmetic. This is
the V0 form of conditional weight streaming. It is deterministic *after* the
router boundary, not before it; the residual feeding the next router is not
available early enough to justify speculative next-layer expert reads.

The paper's row-column bundling intuition maps to an expert's gate, up, and
down records: a selected expert requires all three. The current GGUF stores
them as separate banks, so V0 performs three exact slices per expert. A future
repack can place the three slices in one aligned record, reducing request and
address-translation overhead without changing useful bytes. V0 does not claim
that improvement before a repacked artifact exists.

Its temporal windowing idea maps to an expert cache across decode positions.
That cache is deliberately absent from the release default. The 24 GB machine
must already preserve 8.203 GB of exact KV address space, the resident output
head, two layer slabs, scratch, and macOS headroom. A cache policy is useful
only if

\[
P(hit)\,B_{saved}/B_{NVMe} > t_{lookup}+t_{pressure}+t_{eviction},
\]

where `t_pressure` includes compression/swap consequences, not merely cache
lookup time. The 2 GiB fixed-cache experiment demonstrated this constraint:
it removed bytes but produced unsafe pressure for the long run. Thus V0
implements exact sparsity and co-required overlap now, retains expert
repacking and measured temporal reuse as the next evidence-driven steps, and
does not turn a paper analogy into an unverified feature claim.

### 8.7 Dependency graph: what is parallel, concurrent, or necessarily serial

The decode schedule is a data-dependency graph, not a blanket request to
“parallelize everything”:

```text
fixed(L) ready
      |
      +--> attention/L --> FFN norm --> router --> exact IDs/weights
      |                                      |             |
      |                                      |             +--> 24 urgent preads --+
      |                                      +--> shared expert on GPU -------------+--> join
      |                                                                                |
      |                                                           routed experts rank 0..7
      |                                                                                |
      +== fixed(L+1) background prefetch overlaps the entire layer ====================+--> residual L
                                                                                              |
                                                                                         attention L+1
```

There are four distinct forms of parallel work:

1. The reader has 12 workers, so independent shard ranges can be in flight at
   once. Urgent and background queues determine priority, not arithmetic.
2. Fixed layer `L+1` I/O overlaps GPU execution of layer `L` through the second
   slab. It is legal because fixed addresses are unconditional.
3. After routing, selected expert I/O overlaps shared-expert GPU execution.
   Both depend on the router but not on one another.
4. Inside Metal, GEMV output rows, 128 attention heads, score lanes, and
   prefill expert assignments are data-parallel threadgroups.

The following boundaries remain serial for correctness:

- layer `L+1` consumes the final residual of layer `L`;
- the router consumes the post-attention normalized residual;
- expert file offsets are unknown until exact router completion is host-visible;
- attention's max, exponential sum, and weighted-value passes form a stable
  online-softmax dependency;
- routed expert contributions accumulate in top-k rank order, matching the
  accepted finite-precision path;
- final sampling consumes all 129,280 completed logits.

Query and KV projections are mathematically independent after the attention
RMSNorm, but V0 leaves them in one ordered command stream. Splitting them would
add command buffers and concurrent reads of the same activation while the
measured GPU term is already much smaller than storage time. Likewise, eight
decode experts could use eight activation/output arenas, but the added memory,
parallel accumulation, and changed FP32 summation order conflict with the
capacity/parity gates. Prefill obtains safe expert parallelism differently: it
writes independent `[token,rank]` slots and performs a final rank-ordered merge.

This distinction is central to the co-design: concurrency is used where
ownership and dependencies prove it safe; fusion is used where it removes a
materialized intermediate without changing accepted rounding; and serial work
is preserved when it carries model semantics or a finite-precision ordering.

## 9. Layer-major blocked prefill

Token-major prefill would stream most of the active model once per prompt
token. V0 instead uses tiles \(B\le128\) and visits layers outside tokens.
Fixed layer weights are fetched once per tile.

For an MoE layer, each token has exact selected set \(A_t\). The tile needs

\[
U_B=\bigcup_{t\in B}A_t.
\]

Each expert in \(U_B\) is loaded once. Assignments are grouped by expert;
the 2-D Metal grid maps `(output row, assignment)` so multiple token vectors
consume the same compressed rows while they are hot. Results are written to
`[token, top-k-rank, hidden]` slots. A final merge loops ranks in reference
order:

\[
x_t\leftarrow x_t+E_s(x_t)+
\sum_{r=0}^{7}\alpha_{t,r}E_{A_{t,r}}(x_t).
\]

Grouping changes schedule, not summation order. The union bound

\[
|U_B|\le\min(256,8B)
\]

explains the tradeoff: larger tiles amortize fixed weights but eventually use
nearly every expert. The implemented 128-token tile was chosen from available
activation memory and demonstrated 1.69 prefill token/s over exactly 1,000
tokens.

This is grouped quantized matrix-vector execution, not a true fixed-projection
QMM. The remaining prefill opportunity is to make Q4/Q5/Q6/IQ fixed matrices
consume a matrix of 128 activations per compressed tile, provided the new
reduction order passes parity.

## 10. Exact KV capacity and checkpointing

Per layer and sequence position, V0 stores

\[
128\cdot128\text{ non-RoPE K}
+128\cdot128\text{ V}
+64\text{ shared RoPE K}
=32{,}832
\]

FP16 values. Across 61 layers:

\[
b_{KV}=61\cdot32{,}832\cdot2
=4{,}005{,}504\text{ bytes/position}.
\]

Thus

\[
M_{KV}(C)=4{,}005{,}504C.
\]

At \(C=2048\), capacity is 8,203,272,192 bytes. This is why a 24 GB V0 can
prove 1,000+1,000 but cannot claim a million-token context with this state
representation.

State v3 has a 36-byte header and exact prefix payload:

\[
S_{state}(p)=36+4{,}005{,}504p.
\]

The header stores magic, version, layer count, context capacity, KV rank, RoPE
width, committed position, and pending predicted token. Saving uses

```text
write .partial -> fflush -> fsync -> close -> rename
```

so interruption cannot turn an incomplete write into the authoritative state.
The pending token is essential: it represents the already-computed next token
that has not yet been advanced through the model.

## 11. Memory admission and pressure accounting

Before Metal allocation, the runtime computes:

\[
M_{base}=M_{head}+2M_{layer-slab}+M_{KV}+M_{scores}
+M_{expert-arena}+M_{prefill-scratch}+M_{margin}.
\]

For the 2,048 acceptance run, the observed ledger was:

| Component | Size |
|---|---:|
| output head | 0.760 GB |
| two fixed slabs | 0.736 GB |
| fixed cache | 0.263 GB |
| exact KV capacity | 8.203 GB |
| expert arena | 90.833 MB |
| prefill scratch | 124.95 MB |
| runtime margin | 33.55 MB |
| estimated total | 10.22 GB |
| live available | 20.95 GB |
| host reserve | 1.07 GB |
| extra cache guard | 2.15 GB |

Available memory includes clean reclaimable file pages but excludes wired,
anonymous-active, and compressed memory. Every step logs deltas for pageouts,
compressions, decompressions, swap-ins, and swap-outs. Compression alone is a
pressure signal; swap-out is a stronger release concern. The 2 GiB fixed-cache
experiment saved more bytes but caused enough compression to reject it as the
24 GB default.

The long 256 MiB-cache acceptance continuation also eventually recorded
system-wide swap activity: 2,407 swap-ins and 168 swap-outs over its final 719
steps. This does not indicate a bad model tensor or checkpoint—the
forward continued coherently and state commits remained valid—but it does show
that 2,048-position expanded KV plus the host workload reaches macOS pressure.
The counters are host-wide, so causality is not process-local. Early 3.1–3.2 s
decode is therefore a best measured interval; late 4.2–4.5 s decode is the
more conservative sustained 24 GB expectation. A future release should
compare a zero fixed cache and smaller safety context from the same state, or
move to compact/paged KV, before increasing cache residency.

## 12. Performance model and measured bottleneck

Let \(N_w\) be actual model bytes, \(B_s\) realized storage bandwidth,
\(t_g\) GPU time, and \(t_o\) non-overlapped control. With double buffering,

\[
t_{token}\gtrsim\max(N_w/B_s,t_g)+t_o,
\]

with router barriers repeated within the storage span.

The optimized steady sample measured approximately:

| Metric | Value |
|---|---:|
| model/NVMe bytes | 13.588 GB/token |
| effective NVMe | 4.4–4.6 GB/s |
| NVMe span | 2.96–3.07 s |
| GPU execution | 0.47–0.52 s |
| end-to-end | 3.09–3.22 s |
| decode rate | 0.31–0.32 token/s |
| logical reads | 1,869/token |
| urgent expert bytes | 4.035 GB/token |
| urgent expert reads | 1,392/token |
| command buffers | 178/token |
| hot allocations | 0/token |

The wall time closely tracks the NVMe span, while GPU time is much smaller.
The system is storage-bound, not dense-multiply-bound. Optimizing a 0.5-second
GPU component by 2× cannot save more than 0.25 seconds and may save less when
it is already overlapped. Removing 262 MB of model traffic and improving
request priority reduced steady wall time by roughly 16–22% in the compared
anchors.

`io_service_us` is summed across 12 workers and can exceed wall time; it is a
work measure, not latency. `nvme_span_us` is first-start to last-completion and
is the correct denominator for effective aggregate GB/s. `io_wait_us` counts
only explicit producer blocking and can be smaller than the I/O span because
reads overlap Metal work.

## 13. Why specific attractive optimizations were rejected

### 13.1 Residual plus RMSNorm fusion

The fusion removed a device pass and dispatch, and its local loop appeared to
perform the same operations. It matched token IDs at the first three anchors,
but at position 1,282 the greedy maximum logit changed:

```text
baseline: 45.4884
fused:    45.2879
```

The change arises because inter-kernel storage, scheduling, and reduction
boundaries are observable in floating-point arithmetic. The kernels and all
conditional plumbing were deleted.

### 13.2 Latent MLA absorption

The identities

\[
q^TW_Kc=(W_K^Tq)^Tc,
\qquad
\sum_jp_jW_Vc_j=W_V\sum_jp_jc_j
\]

are exact over real numbers. This checkpoint stores a quantized combined KV-B
projection and the accepted path expands it per token before FP16 caching.
Pre-dequantizing/transposing absorbed matrices or moving the projection after
softmax changes quantization and FP rounding. V0 retains the larger cache until
a compact representation passes layer/logit/token parity.

### 13.3 A large fixed cache

A 2 GiB cache reduced per-token model traffic to 11.710 GB, but consumed
headroom needed by 8.2 GB KV capacity and macOS. Compression increased and the
risk-adjusted result was worse for a long proof. V0 uses 256 MiB by default and
keeps the larger setting explicit rather than pretending maximum caching is
always optimal.

### 13.4 Next-layer expert prediction

The exact next router depends on the current layer output. Predicting IDs can
prefetch useful records but is speculation, may waste bandwidth, and cannot
replace exact reads. V0 prefetches only unconditional fixed projections and
overlaps already-selected experts with shared compute.

### 13.5 Inline assembly or undocumented matrix paths

Metal does not expose PTX; inline PTX is NVIDIA-specific. Undocumented or
toolchain-fragile instructions are inappropriate before profiles show compute
as the limiting term. The current bottleneck is model bytes and request
scheduling. V0 specializes legal MSL kernels to exact shapes and quant formats.

## 14. Logging as a closed optimization loop

Each metric answers a falsifiable question:

| Field | Question |
|---|---|
| `wall_us` | Did end-to-end token latency improve? |
| `gpu_us` | Did Metal compute improve, or was time elsewhere? |
| `io_wait_us` | How much producer time was explicitly blocked? |
| `model_bytes` | Did the algorithm require fewer immutable bytes? |
| `nvme_bytes/reads/span` | Did actual submitted work and realized bandwidth improve? |
| `urgent_*` | What fraction is exact router-selected critical traffic? |
| `io_service/max/peak` | Are queue depth and tail latency healthy? |
| `cmdbufs/dispatches` | Was control overhead removed or merely hidden? |
| `allocations` | Did the hot path regress to allocator churn? |
| `kv_bytes` | Is context capacity math still exact? |
| VM deltas | Did speed come from unsafe memory pressure? |
| token/logit | Did numerical behavior change? |

`--profile-layers` decomposes each layer into fixed-stage span/block,
attention, FFN, expert I/O, GPU time, requests, command buffers, and
dispatches. `--trace` hashes residuals and reports RMS/min/max/non-finite values
plus top logits and routing. These modes turn a divergence into the earliest
layer/stage where it becomes observable.

## 15. Acceptance and scale boundary

Exactly 1,000 input plus 1,000 emitted tokens uses committed positions
0–1,998; the final state position is 1,999. It proves:

- the tokenizer and blocked prefill handle a substantial prompt;
- exact KV grows through the complete requested sequence;
- every decode iteration executes all 61 layers;
- SSD streaming and memory pressure remain stable for a long run;
- interruption/resume preserves the same decode loop;
- the output is long enough to expose cumulative numerical or ownership bugs.

It does not prove million-token capacity. With current expanded KV, one million
positions would require about 4.006 TB. Reaching that goal requires a new
finite-precision-compatible compact attention state or SSD-resident paged KV,
plus an attention algorithm whose history traffic does not grow into an
unusable scan. That is the next mathematical architecture problem, not a CLI
flag.

## 16. Future work admitted by the evidence

Only changes with a direct path to the measured reward vector are justified:

1. Repack each expert's gate/up/down as one aligned storage record, reducing
   application requests while keeping useful bytes constant.
2. Implement quantized matrix-matrix fixed projections for prefill, measuring
   expert-union growth and exact token parity.
3. Evaluate a compact KV representation against the current expanded oracle
   at every layer and position before enabling it.
4. On a higher-memory host, tune fixed/expert caching from byte saved,
   transactions avoided, hit probability, and VM pressure—not nominal size.
5. Serve multiple sequences only after a scheduler can preserve slab
   ownership, state isolation, and per-request accounting.
6. Measure joules/token with `powermetrics` and report energy together with
   tokens/s; faster work that causes disproportionate power is not assumed
   more efficient.

The V0 architecture is intentionally specific. Generality comes after one
model, one checkpoint, one machine, and one 1,000+1,000 run are correct.

That run is now complete. Its exact prompt, output, state arithmetic, segment
metrics, and limitations are in [the proof report](PROOF_1K_REPORT.md).

The staged design for replacing expanded KV, full checkpoint rewrites, and
token-by-token prefill at much larger scale is in the
[million-token scale plan](MILLION_TOKEN_SCALE_PLAN.md).
