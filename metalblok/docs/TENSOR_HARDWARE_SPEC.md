# MetalBlok Tensor and Hardware Execution Specification

## Normative tensor shapes, equations, storage addresses, kernel contracts, and machine decisions

**Status:** describes the executing DeepSeek-R1 GGUF path as of 2026-08-10.  
**Companion paper:** [`METALBLOK_PAPER.md`](METALBLOK_PAPER.md)  
**Normative code:** [`../src/gguf_runtime.cpp`](../src/gguf_runtime.cpp),
[`../src/kernels.metal`](../src/kernels.metal),
[`../src/router_ref.hpp`](../src/router_ref.hpp)

---

## 1. Purpose and conformance language

This specification answers four questions for every stage of one token:

1. What tensor exists, with what exact shape and representation?
2. What mathematical function consumes it?
3. When is its storage address knowable?
4. What CPU, GPU, memory, and storage operation implements that function?

“Must” describes a correctness requirement. “Current” describes the measured
MetalBlok implementation. “Target” describes a justified optimization that is
not yet measured.

The executing contract is batch-one greedy decode. Shapes use mathematical
row-major notation `[output,input]`; GGUF descriptors list the innermost
dimension first as `[K,N]`. A stacked expert bank is GGUF `[K,N,E]` and
mathematically $E$ independent matrices in $\mathbb{R}^{N\times K}$.

---

## 2. Symbols, dimensions, and representations

| Symbol | Meaning | Value |
|---|---|---:|
| $L$ | transformer layers | 61 |
| $d$ | residual width | 7,168 |
| $V$ | vocabulary | 129,280 |
| $H$ | query heads | 128 |
| $r_q$ | query low-rank width | 1,536 |
| $r_{kv}$ | KV latent width | 512 |
| $d_n$ | non-rotary Q/K width per head | 128 |
| $d_r$ | rotary Q width per head and shared rotary K width | 64 |
| $d_v$ | value width per head | 128 |
| $d_f$ | dense FFN intermediate | 18,432 |
| $d_e$ | expert intermediate | 2,048 |
| $E$ | routed experts per sparse layer | 256 |
| $k$ | selected routed experts | 8 |
| $G$ | expert groups | 8 |
| $G'$ | retained groups | 4 |
| $C$ | allocated runtime context | 64 verified |
| $q_a$ | activations/cache representation | FP16 |
| $q_r$ | reductions, scores, router weights | FP32 |

Define stored bytes per weight

\[
\rho_q=\frac{b_q}{256},
\]

where $b_q$ is the bytes in a 256-weight accounting block. For scalar F32 the
table uses a conceptual group of 256 scalars so the same size equation remains
valid; this does not imply that GGUF physically wraps F32 in a block header.

| GGML type | ID behavior used here | $b_q$ | $\rho_q$ |
|---|---|---:|---:|
| F32 | 256 scalar FP32 values | 1,024 | 4 |
| Q4_K | K-quant | 144 | 0.5625 |
| Q5_K | K-quant | 176 | 0.6875 |
| Q6_K | K-quant | 210 | 0.8203125 |
| IQ2_XXS | codebook I-quant | 66 | 0.2578125 |
| IQ1_S | codebook I-quant | 50 | 0.1953125 |

For a block-quantized matrix with $K$ divisible by 256 and $N$ rows,

\[
S_q(K,N)=N\frac{K}{256}b_q.
\]

---

## 3. Complete 1,025-tensor manifest by family

### 3.1 Global tensors: 3

| GGUF name | GGUF shape | Mathematical shape | Stored type | Role |
|---|---:|---:|---|---|
| `token_embd.weight` | `[7168,129280]` | $V\times d$ | Q4_K | token row lookup |
| `output_norm.weight` | `[7168]` | $d$ | F32 | final RMS gain |
| `output.weight` | `[7168,129280]` | $V\times d$ | Q6_K | logits; not tied |

Exact bytes:

\[
S_{emb}=521,256,960,
\quad
S_{out}=760,166,400,
\quad
S_{outnorm}=28,672.
\]

### 3.2 Common tensors in every layer: $61\times9=549$

For $\ell\in[0,60]$:

| Suffix | GGUF shape | Mathematical shape | Operation |
|---|---:|---:|---|
| `attn_norm.weight` | `[7168]` | $d$ | pre-attention RMS gain |
| `attn_q_a.weight` | `[7168,1536]` | $r_q\times d$ | residual to Q latent |
| `attn_q_a_norm.weight` | `[1536]` | $r_q$ | Q-latent RMS gain |
| `attn_q_b.weight` | `[1536,24576]` | $H(d_n+d_r)\times r_q$ | Q expansion |
| `attn_kv_a_mqa.weight` | `[7168,576]` | $(r_{kv}+d_r)\times d$ | latent KV + rotary K |
| `attn_kv_a_norm.weight` | `[512]` | $r_{kv}$ | latent K/V RMS gain |
| `attn_kv_b.weight` | `[512,32768]` | $H(d_n+d_v)\times r_{kv}$ | latent K/V expansion source |
| `attn_output.weight` | `[16384,7168]` | $d\times Hd_v$ | attention output projection |
| `ffn_norm.weight` | `[7168]` | $d$ | pre-FFN RMS gain |

Stored quant type is a per-descriptor property and must be dispatched from the
GGUF type ID. The runtime must not infer dtype from the tensor name. For
example, layer-0 Q-A is Q4_K, KV-A is Q6_K, and norms are F32; other large
matrices follow the mixed quantization chosen by the checkpoint.

### 3.3 Dense FFN tensors: $3\times3=9$

For $\ell\in[0,2]$:

| Suffix | GGUF shape | Mathematical shape |
|---|---:|---:|
| `ffn_gate.weight` | `[7168,18432]` | $d_f\times d$ |
| `ffn_up.weight` | `[7168,18432]` | $d_f\times d$ |
| `ffn_down.weight` | `[18432,7168]` | $d\times d_f$ |

### 3.4 MoE tensors: $58\times8=464$

For $\ell\in[3,60]$:

| Suffix | GGUF shape | Mathematical content | Dynamic? |
|---|---:|---:|---|
| `ffn_gate_inp.weight` | `[7168,256]` | router $W^R\in\mathbb R^{E\times d}$ | always |
| `exp_probs_b.bias` | `[256]` | correction bias $b$ | always |
| `ffn_gate_shexp.weight` | `[7168,2048]` | shared gate | always |
| `ffn_up_shexp.weight` | `[7168,2048]` | shared up | always |
| `ffn_down_shexp.weight` | `[2048,7168]` | shared down | always |
| `ffn_gate_exps.weight` | `[7168,2048,256]` | 256 routed gate matrices | selected slice |
| `ffn_up_exps.weight` | `[7168,2048,256]` | 256 routed up matrices | selected slice |
| `ffn_down_exps.weight` | `[2048,7168,256]` | 256 routed down matrices | selected slice |

The count identity

\[
3+549+9+464=1025
\]

is a useful completeness test. A runtime accepting fewer tensors is not the
same model unless it proves a tie, fold, or other equivalence.

---

## 4. Machine data path

### 4.1 Physical hierarchy

The current hierarchy is

```text
NAND cells
  -> SSD controller / FTL / ECC / NVMe
  -> APFS file offset
  -> pread(F_NOCACHE) into MTLStorageModeShared
  -> Apple unified-memory fabric
  -> GPU caches/registers/threadgroup memory
  -> FP16 activation buffer
```

The runtime controls file offsets and request sizes. It does not control NAND
channel, die, plane, physical page, ECC codeword, or FTL mapping. A hardware
presentation must not call this raw NAND access.

### 4.2 Address formation

For shard $s$, let $D_s$ be its GGUF data-section base. Tensor $j$ has
descriptor offset $o_j$, so

\[
a_j=D_s+o_j.
\]

For a rank-three expert bank $j$ of size $S_j$, expert $e$ is

\[
a_{j,e}=a_j+e(S_j/E),
\quad s_{j,e}=S_j/E.
\]

The three addresses for expert $e$ become knowable after the router returns
$e$. All other non-expert tensor addresses are static at model load.

### 4.3 Buffer state machine

The current one-shot weight buffer follows

```text
unallocated
  -> shared MTLBuffer allocated
  -> CPU/SSD writer owns buffer
  -> pread completion release-store
  -> GPU reader owns buffer
  -> command-buffer completion
  -> buffer released
```

The producer must not release or overwrite the allocation before command-buffer
completion. The GPU must not read before the acquire observation of the I/O
completion. A future persistent ring needs the same ownership invariant per
slot.

---

## 5. Forward schedule: normative pseudocode

For one token ID $u_t$:

```text
x <- dequant(token_embd[u_t])
for layer = 0..60:
    xn <- RMSNorm(x, attn_norm[layer])
    qa <- Q_A[layer] @ xn
    qan <- RMSNorm(qa, q_a_norm[layer])
    qfull <- Q_B[layer] @ qan
    (qnope, qrope) <- split_and_NEOX_RoPE(qfull, position)
    qeff <- absorbed_K[layer]^T @ qnope, independently per head

    kva <- KV_A[layer] @ xn
    (c[position], krope[position]) <- latent_norm_split_RoPE(kva)
    scores <- scaled_dot(qeff, c[0:position], qrope, krope[0:position])
    alpha <- stable_softmax(scores)
    olat <- alpha @ c[0:position]
    ofull <- absorbed_V[layer] @ olat, independently per head
    attn <- O[layer] @ concatenate(ofull)
    x <- x + attn

    fn <- RMSNorm(x, ffn_norm[layer])
    if layer < 3:
        f <- DenseSwiGLU(layer, fn)
    else:
        ids, weights <- ExactGroupedSigmoidRouter(layer, fn)
        f <- SharedSwiGLU(layer, fn)
        for each (expert_id, mixture_weight):
            f <- f + mixture_weight * RoutedSwiGLU(layer, expert_id, fn)
    x <- x + f

h <- RMSNorm(x, output_norm)
logits <- output @ h
next_token <- argmax(logits)
```

Every arrow is an actual dependency in the current runtime.

---

## 6. Kernel contracts

### 6.1 Quantized GEMV

Input:

- quantized $W$ as raw descriptor bytes;
- FP16 $x[K]$;
- output FP16 $y[N]$;
- $K$ divisible by 256 for K/I-quants.

Launch:

- grid: $N$ threadgroups;
- group size: 128 threads;
- one output row per group.

For row $r$, thread $p$ handles blocks

\[
b=p,p+128,p+256,\ldots.
\]

It accumulates

\[
a_p=\sum_{b\in\mathcal B_p}\sum_{j=0}^{255}
D_q(W_{r,b})_j x_{256b+j}
\]

in FP32. SIMD and threadgroup reductions compute

\[
y_r=\operatorname{FP16}\left(\sum_{p=0}^{127}a_p\right).
\]

The quant decoder lives in registers; no FP16 matrix scratch exists.

### 6.2 FP16 grouped GEMV for absorbed MLA

For $M$ rows and `group_size` $g$, row $r$ reads input group

\[
h(r)=\left\lfloor\frac{r}{g}\right\rfloor,
\]

and computes

\[
y_r=\sum_{i=0}^{K-1}W_{r,i}x_{h(r),i}.
\]

For absorbed K:

\[
M=Hr_{kv}=65,536, K=d_n=128, g=r_{kv}=512.
\]

Rows $h\cdot512\ldots(h\cdot512+511)$ read head $h$'s 128 non-rotary query
coordinates.

For absorbed V:

\[
M=Hd_v=16,384, K=r_{kv}=512, g=d_v=128.
\]

Rows $h\cdot128\ldots(h\cdot128+127)$ read head $h$'s 512 latent output.

### 6.3 RMSNorm

One threadgroup reduces FP32 sum of squares:

\[
s=\sum_i\operatorname{FP32}(x_i)^2,
\quad
c=(s/N+\epsilon)^{-1/2},
\quad
y_i=\operatorname{FP16}(x_i c g_i).
\]

### 6.4 NEOX RoPE

One threadgroup per query head; one group total for shared rotary K. Coordinates
(i) and $i+d_r/2$ form the pair. The rotation matrix must be identical for
query and key at a given position and frequency.

### 6.5 Three-pass attention

For head $h$, one 256-thread group performs:

1. dot products and global maximum;
2. exponentials and normalization sum;
3. weighted latent accumulation.

Scores are FP32 in `[H,C]`; cache and output are FP16. A device-memory barrier
is required between per-thread score writes and the third pass where every
thread reads scores written by other threads.

### 6.6 SwiGLU

One thread per element:

\[
z_i=\operatorname{FP16}
\left(\frac{g_i}{1+e^{-g_i}}u_i\right).
\]

### 6.7 AXPY residual/mixture accumulation

\[
y_i\leftarrow\operatorname{FP16}(y_i+\alpha x_i).
\]

Residual additions use $\alpha=1$. Routed experts use FP32 router weight
$\alpha_e$.

### 6.8 Argmax

Each of 1,024 threads scans a strided vocabulary subset, then performs a
SIMD/threadgroup max reduction. Strict `>` preserves the lower earlier index on
ties. Output is one `uint32` token ID.

---

## 7. Exact block decoder specification

### 7.1 Q4_K

A 256-weight block is

```text
offset 0:  fp16 d
offset 2:  fp16 dmin
offset 4:  12 bytes packed 6-bit scales/mins
offset 16: 128 bytes, two 4-bit codes per byte
```

For sub-block $j\in[0,7]$, unpack scale $s_j$ and minimum code $m_j$.
For code $q\in[0,15]$,

\[
w=d\,s_jq-d_{min}m_j.
\]

There are 32 weights per scale/min pair.

### 7.2 Q5_K

A block is

```text
offset 0:  fp16 d
offset 2:  fp16 dmin
offset 4:  12 packed scale/min bytes
offset 16: 32 high-bit bytes
offset 48: 128 low-nibble bytes
```

Combine each 4-bit nibble $q_4$ with one high bit $h$:

\[
q_5=q_4+16h,
\qquad
w=d\,s_jq_5-d_{min}m_j.
\]

### 7.3 Q6_K

A block is

```text
offset 0:   128 low-nibble bytes
offset 128: 64 two-high-bit bytes
offset 192: 16 signed int8 scales
offset 208: fp16 d
```

Reconstruct signed 6-bit code

\[
q_6=(q_{low4}\;|\;(q_{high2}\ll4))-32,
\]

then

\[
w=d\,s_jq_6.
\]

### 7.4 IQ1_S

A block is

```text
offset 0:  fp16 d
offset 2:  32 low grid-index bytes
offset 34: eight uint16 high/index/scale/sign records
```

For each 32-weight sub-block, let $h$ be its 16-bit record:

\[
s=(h\gg12)\mathbin{\&}7,
\quad
d_l=d(2s+1),
\]

\[
\delta=
\begin{cases}-0.125,&h\mathbin{\&}0x8000,\\+0.125,&\text{otherwise}.
\end{cases}
\]

For each group of eight weights,

\[
i=q_{low}\;|\;(((h\gg3g)\mathbin{\&}7)\ll8).
\]

The 2,048-entry codebook returns eight signed values $c_{i,j}$, and

\[
w_{g,j}=d_l(c_{i,j}+\delta).
\]

The codebook is 16 KiB and is copied into a Metal-owned shared buffer at
startup.

### 7.5 IQ2_XXS

A block is

```text
offset 0: fp16 d
offset 2: sixteen uint16 words / 64 payload bytes
```

Each 32-weight sub-block contains two logical 32-bit words $a_0,a_1$. Because
the block stride is 66 bytes, $a_0,a_1$ may be misaligned; the Metal kernel
assembles them bytewise.

For the sub-block,

\[
d_b=d\cdot\left(0.5+(a_1\gg28)\right)\cdot0.25.
\]

Each of four groups selects a codebook row with one byte of $a_0$ and a sign
pattern with seven bits of $a_1$. For codebook value $c_j$ and sign bit
$\eta_j$,

\[
w_j=d_bc_j(1-2\eta_j).
\]

### 7.6 F32

No quant unpack is performed:

\[
y_i=\operatorname{FP16}
\left(\sum_j\operatorname{FP32}(W_{ij})
\operatorname{FP32}(x_j)\right).
\]

---

## 8. MLA resident-layout construction

The stored `attn_kv_b` row order for each head is non-rotary K followed by V.
Let the stored mathematical matrix be

\[
B_{h,z,k},
\quad
h\in[0,127],\ z\in[0,255],\ k\in[0,511].
\]

Construct:

\[
U^K_{h,k,n}=B_{h,n,k},
\quad n\in[0,127],
\]

and

\[
U^V_{h,v,k}=B_{h,128+v,k},
\quad v\in[0,127].
\]

Current layout addresses are

\[
\operatorname{addr}(U^K_{h,k,n})=((h\cdot512+k)\cdot128+n)\cdot2,
\]

\[
\operatorname{addr}(U^V_{h,v,k})=((h\cdot128+v)\cdot512+k)\cdot2.
\]

Each layer allocates

\[
128\cdot512\cdot128\cdot2=16,777,216\text{ B}
\]

for $U^K$ and the same for $U^V$, totaling 33,554,432 bytes/layer.

Current construction reads and CPU-dequantizes the complete quantized KVB
tensor, transposes/reshapes it, rounds to FP16, and retains both buffers.

Target hardware may perform this transform offline in the model layout or once
through a dedicated transpose/dequant engine. It must preserve the equations
above.

---

## 9. Exact router state machine

### 9.1 Inputs

- `router_log_[256]`: FP16 $r_e$;
- `exp_probs_b.bias[256]`: FP32 $b_e$;
- constants $E=256,k=8,G=8,G'=4,\gamma=2.5,$ and `normalize=1`.

### 9.2 Steps

1. $p_e\leftarrow\sigma(r_e)$.
2. $c_e\leftarrow p_e+b_e$.
3. For each contiguous 32-expert group, find its top two $c_e$.
4. $S_g\leftarrow c_g^{(1)}+c_g^{(2)}$.
5. Retain four groups by descending $S_g$, lower group ID on ties.
6. Select eight experts by descending $c_e$ inside retained groups, lower
   expert ID on ties.
7. $Z\leftarrow\sum_{e\in A}p_e$.
8. $w_e\leftarrow2.5p_e/Z$.

### 9.3 Output and synchronization

- `router_idx_[8]`: `uint32` expert IDs;
- `router_wts_[8]`: FP32 mixture coefficients.

The current Metal router uses one thread. After dispatch, the CPU waits for the
command buffer because it needs IDs to form file offsets. This is a hard
correctness dependency, not evidence that the CPU must be involved in a custom
chip.

### 9.4 Selection-to-storage hardware interface

A proposed mailbox entry is

```text
layer_id:       uint8
expert_count:   uint8 (=8)
expert_ids:     8 x uint8
weights:        8 x fp32 or bf16/fp16 under a validated contract
sequence_epoch: uint32
ready:          release/acquire flag
```

An address engine can compute all 24 current tensor-slice requests or eight
relayout records without a general-purpose CPU copy.

---

## 10. Useful byte accounting by graph component

The executing steady selected graph reads exactly

\[
N_{steady}=13,770,679,744\text{ bytes/token}.
\]

It includes:

- one 4,032-byte embedding row;
- Q-A, Q-B, KV-A, and output projections for 61 layers;
- three dense FFN matrices for three layers;
- router, bias, and three shared matrices for 58 layers;
- 24 selected expert slices per MoE layer;
- the 760,166,400-byte output head.

The counterfactual all-expert graph reads

\[
N_{all}=138,861,340,096\text{ bytes/token}.
\]

Hence

\[
N_{all}-N_{steady}=125,090,660,352\text{ bytes/token}
\]

are omitted by exact routing.

The isolated process also reads startup tensors:

\[
N_{isolated}=N_{steady}+4,026,368+839,516,160
=14,614,222,272.
\]

This arithmetic is emitted by `--probe-gguf`; it is not estimated from the
nominal 671B parameter count.

---

## 11. Logical read accounting

### 11.1 Steady reads

| Component | Formula | Reads |
|---|---:|---:|
| embedding + output | (1+1) | 2 |
| attention stored projections | $61\cdot4$ | 244 |
| dense FFNs | $3\cdot3$ | 9 |
| MoE fixed/shared/router/bias | $58\cdot5$ | 290 |
| routed expert slices | $58\cdot8\cdot3$ | 1,392 |
| **Total** |  | **1,937** |

### 11.2 Isolated-process startup reads

| Component | Reads |
|---|---:|
| four resident norms per layer | 244 |
| output norm | 1 |
| one KVB source per layer | 61 |
| **Additional** | **306** |

Total isolated logical reads are (1,937+306=2,243).

### 11.3 Why queue capacity 64 does not imply queue depth 64

The SPSC ring can hold 64 outstanding requests per shard, but current
`stream_gemv` does

```text
submit -> wait -> GPU dispatch -> wait -> release
```

for each tensor. Its effective model-path storage depth is one. Queue capacity
is an ability; queue occupancy is the performance fact.

---

## 12. Command-buffer dependency accounting

Approximate current command buffers per common MLA:

| Operation | Count |
|---|---:|
| attention RMSNorm | 1 |
| Q-A GEMV | 1 |
| Q-A RMSNorm | 1 |
| Q-B GEMV | 1 |
| Q split/RoPE | 1 |
| absorbed K GEMV | 1 |
| KV-A GEMV | 1 |
| KV split/norm/RoPE/cache write | 1 |
| attention softmax/value | 1 |
| absorbed V GEMV | 1 |
| output GEMV | 1 |
| **MLA subtotal** | **11** |

Each layer adds one attention residual command.

Dense FFN uses RMSNorm, gate GEMV, up GEMV, SwiGLU, and down GEMV: five
commands, plus one residual command.

MoE FFN uses:

- RMSNorm: 1;
- shared gate/up/SwiGLU/down: 4;
- router GEMV + router select: 2;
- eight experts × (gate, up, SwiGLU, down, weighted add): 40;
- residual: 1.

Together with MLA and attention residual this is approximately 60 commands per
MoE layer. The complete token is about 3,537 command buffers.

This is correct but inefficient. A production scheduler should preserve
dependencies while reducing submission boundaries.

---

## 13. Capacity equations

### 13.1 Latent cache

For runtime context $C$,

\[
M_{KV}(C)=L C(r_{kv}+d_r)2
=70,272C.
\]

At $C=64$,

\[
M_{KV}=4,497,408\text{ B}.
\]

At $C=256$, it would be 17,989,632 bytes. This arithmetic demonstrates
capacity, not validation of 256-position numerics or behavior.

### 13.2 Scores

\[
M_{scores}(C)=HC\cdot4=512C.
\]

At 64 this is 32,768 bytes.

### 13.3 Absorbed matrices

\[
M_{absorb}=LHr_{kv}(d_n+d_v)2
=2,046,820,352\text{ B}.
\]

### 13.4 Activation buffers

Major FP16 activation sizes are:

| Buffer | Elements | Bytes |
|---|---:|---:|
| `x`, `x_norm` each | 7,168 | 14,336 |
| `q_a`, `q_a_n` each | 1,536 | 3,072 |
| `q_full` | 24,576 | 49,152 |
| `q_nope` | 16,384 | 32,768 |
| `q_rope` | 8,192 | 16,384 |
| `kv_a` | 576 | 1,152 |
| `q_eff`, `o_lat` each | 65,536 | 131,072 |
| `o_full` | 16,384 | 32,768 |
| dense FFN scratch each | 18,432 | 36,864 |
| logits | 129,280 | 258,560 |

### 13.5 Admission gate

Current admission requires

\[
M_{available}\ge
M_{absorb}+M_{KV}+M_{scores}+S_{output}+256\text{ MiB}+3\text{ GiB}.
\]

At context 64, the modeled runtime portion is 3,079,952,384 bytes. The final
3 GiB is reserved for the operating system and other applications.

---

## 14. Checkpoint specification

Version-2 header, little-endian:

```text
char[8]  magic = "MBLKSTAT"
uint32   version = 2
uint32   layers = 61
uint32   max_seq
uint32   kv_rank = 512
uint32   rope_dim = 64
uint32   pos
uint32   next_token
```

For each layer in order, store:

1. `pos * 512` FP16 latent-cache values;
2. `pos * 64` FP16 rotary-key values.

Exact file size:

\[
36+pos\cdot61\cdot576\cdot2.
\]

Atomicity protocol:

```text
write path.partial
fflush
fsync(file)
close
rename(path.partial, path)
```

No weights are checkpointed. A resumed process rebuilds resident norms and
absorbed matrices from GGUF, then restores cache and position.

---

## 15. Current decision table

| Design axis | Current decision | Exact reason | Consequence |
|---|---|---|---|
| Expert predicate | real grouped router | zero coefficients prove omission | router is serial boundary |
| Expert granularity | one matrix slice/read | compatible with existing GGUF | 3 reads/expert |
| Weight representation | native mixed GGUF | bounds bytes and memory | custom decoders required |
| Weight lifetime | one GEMV | simplest safe ownership | no reuse/overlap |
| Norm lifetime | runtime resident FP16 | tiny and repeatedly used | cold conversion cost |
| KVB lifetime | absorbed resident FP16 | scalable latent attention | 2.047 GB fixed memory |
| KV representation | FP16 latent + rotary | exact 71.111× reduction | cache tied to MLA |
| I/O caching | `F_NOCACHE`, no readahead | avoid 140 GB page-cache pressure | explicit I/O |
| GPU memory | shared UMA buffers | CPU pread and GPU consume same allocation | explicit synchronization |
| Math mode | Metal safe math | correctness baseline | lower peak speed possible |
| Scheduling | submit/wait per stage | easy dependency proof | ~3,537 command buffers |
| Sampling | greedy argmax | deterministic validation | no diversity controls |
| Demo isolation | one process/token | recoverable failure boundary | repeated cold construction |

---

## 16. Target chip decisions implied by the model

### 16.1 Resident selection plane

Keep these resident because they decide future storage:

- router matrices and correction biases;
- tensor/record base addresses and strides;
- layer state and sequence position;
- quant codebooks;
- norms where capacity permits.

The current router matrices are about 7.34 MB per MoE layer in F32. A chip may
quantize or repartition them only after proving that expert IDs and weights stay
within the required equivalence contract. Router errors change the storage
working set, not merely one arithmetic result.

### 16.2 Expert record layout

Target record:

```text
record header
gate quant blocks
up quant blocks
down quant blocks
optional per-section scale/codebook metadata
checksum/ECC boundary metadata
padding to controller-optimal alignment
```

Address:

\[
a_{\ell,e}=a_0+\ell S_{layer}+eS_{record}.
\]

This eliminates tensor-name lookup and reduces three logical reads to one. It
does not reduce useful expert bytes.

### 16.3 Persistent I/O slabs

Use at least two ownership-separated slots per stream:

```text
FREE -> IO_IN_FLIGHT -> READY -> COMPUTE_IN_FLIGHT -> FREE
```

Slot reuse requires both I/O and compute completion. Queue-depth sizing for
request size $s$, latency $\ell_0$, and target bandwidth $B$ obeys the
rough latency-hiding condition

\[
Q\gtrsim\frac{B\ell_0}{s}.
\]

This is a starting bound; controller parallelism and service-time variance must
be measured.

### 16.4 Fused quant matrix engine

The matrix unit needs:

- byte-addressable unaligned loads for 50-, 66-, 176-, and 210-byte blocks;
- codebook lookup for IQ1_S and IQ2_XXS;
- bit-field extraction and signed scale handling;
- FP16 activation conversion;
- FP32 accumulation and reductions;
- FP16 output;
- block-size-aware row addressing.

Advertising generic “INT2” support does not meet this contract.

### 16.5 Latent-attention engine

Support two head-grouped projections, a shared latent cache, per-head score
computation against the same latent rows, NEOX rotary dot products, stable
softmax, and latent weighted sums. Expanding K/V into DRAM would defeat the
71.111× cache result.

### 16.6 Scheduling and fusion

Safe fusion candidates:

- RMSNorm + following projection when gain access and reduction ordering are
  preserved;
- Q-B + split/RoPE;
- KV-A + latent RMSNorm + cache write;
- gate + up + SwiGLU when both projections can share input residency;
- expert down + weighted residual accumulation;
- final RMSNorm + output projection tiles + online argmax, eliminating the
  full 760 MB transient if output weights are tiled safely.

Each fusion must define rounding points. Removing an intermediate FP16 store
can change numerics because current semantics round at that boundary.

---

## 17. Performance model

Let $N_w$ be useful stored weight bytes, $m$ logical requests, $C_m$
matrix operations, $n_{cb}$ command buffers, and $T$ context length. A
non-overlapped upper-level model is

\[
t_{token}\approx
m\ell_{io}+\frac{N_w}{B_{io}}
+\frac{2P_{active}}{F_{eff}}
+n_{cb}\ell_{cb}
+t_{attn}(T)+t_{cpu}+t_{sync}.
\]

The current implementation serializes most terms, so adding them is reasonable.
An overlapped target is bounded instead by critical-path maxima over pipeline
stages, plus the router barriers:

\[
t_{token,target}\ge
\sum_{\ell=0}^{60}
\left[t_{pre-router,\ell}
+t_{router,\ell}
+\max(t_{expertIO,\ell},t_{expertCompute,\ell})
+t_{post,\ell}\right].
\]

No design can overlap exact expert reads before the router unless it performs
speculative prefetch. Speculation may improve latency but can increase bytes;
its hit and waste rates must be reported.

---

## 18. Verification matrix

| Property | Oracle | Acceptance |
|---|---|---|
| shard residency | `lstat`, block allocation | all three resident, exact sizes |
| tensor completeness | descriptor family/count | 1,025 and required shapes |
| quant block decode | CPU decoder | real tensor GEMV relative error < $5\times10^{-3}$ |
| router | independent CPU function | exact IDs, weights within FP tolerance |
| NEOX RoPE | scalar reference | pair-index and numeric parity |
| MLA absorption | expanded K/V reference | per-head outputs within FP16 contract |
| residual graph | layer checkpoint comparison | per-layer bounded error |
| output head | dedicated tensor | no embedding fallback |
| byte ledger | traced `pread` | exact sum or explained cache delta |
| checkpoint | forced process death | last renamed state loads and advances +1 |
| memory | ledger + OS counters | startup refusal below reserve; no unexplained growth |
| long context | separate soak | must not inherit 64-position validation label |

---

## 19. Non-claims

This specification does not establish:

- that APFS offsets map to chosen NAND pages;
- that the SSD reaches its advertised sequential bandwidth under this access
  pattern;
- that 256/8 routing creates a 32× total-token speedup;
- that bundling or window caching has been implemented in this GGUF path;
- that a 163,840-token context is correct or safe merely because its metadata
  declares that length;
- that coherent text equals reference-logit parity;
- that the current 3,537-command schedule is a recommended chip schedule;
- that IQ1_S preserves full model quality.

These exclusions are part of the engineering result. They identify exactly
what the current evidence supports and what the next experiment must prove.
