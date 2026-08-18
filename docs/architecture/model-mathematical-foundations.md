# Mathematical Foundations of Blok

> **Legacy/future Linux research scope.** The proofs below support the separate
> Kimi/GLM CUDA/uGDS design. They do not describe the active Metal execution
> graph or its finite-precision K/V choice. See the
> [MetalBlok V0 architecture](../../metalblok/docs/V0_ARCHITECTURE.md).

## From token tensors to exact MoE-selected storage reads

**Date:** 2026-08-10

**Purpose:** define every model operation that creates the storage access pattern, prove when an expert ID is known, and separate exact reductions from approximations and measurements.

## 1. Evidence labels

Every claim in this document has one of four meanings.

| Label | Meaning |
|---|---|
| Implemented | The operation exists in the current Kimi CUDA path. |
| Source-audited | The statement follows from a named implementation, configuration, or tensor header. |
| Derived | The number follows from integer tensor shapes, dtypes, or algebra. |
| Measured | A command ran on named hardware and produced a saved log. |

There is currently no completed Kimi target-hardware run. The derived claims are reproducible in [`evidence/derived-claims.md`](evidence/derived-claims.md). The absent target measurements are listed in [`evidence/target-measurement-protocol.md`](evidence/target-measurement-protocol.md).

## 2. Definitions

| Symbol | Definition |
|---|---|
| (t) | Sequence position of the token currently being processed. |
| (ell) | Decoder-layer index. |
| (d) | Hidden-state width. |
| (x_{ell,t}\in\mathbb{R}^{d}) | Residual-stream vector entering layer (ell) for token (t). |
| (H) | Number of attention heads. |
| (E) | Number of routed experts in one MoE layer. |
| (k) | Number of experts selected for one token in one layer. Both models use (k=8). |
| (d_e) | Intermediate width of one routed expert. |
| (A_{ell,t}\subset\{0,\ldots,E-1\}) | Exact set of selected expert IDs. (|A_{ell,t}|=k). |
| (W^g_e,W^u_e,W^d_e) | Gate, up, and down matrices of expert (e). |
| (q) | Bytes per scalar in a stored representation. |
| (S_e) | Stored bytes in one complete expert record. |
| (C_t) | Records already present in a cache before token (t). |
| (S_{cmd}) | Maximum payload of one physical NVMe command in the current uGDS implementation. |

A **tensor** is a typed multidimensional array. A **matrix** is a rank-two tensor. A **record** is a storage layout decision: one contiguous object containing tensors that are always needed together. A record is not necessarily one NVMe command.

## 3. One autoregressive token

Let the tokenizer map text to token IDs (u_0,\ldots,u_{T-1}). For vocabulary matrix (W_E\in\mathbb{R}^{V\times d}), the initial residual for token (t) is

\[
x_{0,t}=W_E[u_t,:].
\]

Each decoder layer applies causal attention and an FFN or MoE, with residual additions:

\[
a_{ell,t}=x_{ell,t}+
\operatorname{Attention}_{\ell}
\left(\operatorname{RMSNorm}_{\ell}^{attn}(x_{ell,t}),K_{\le t},V_{\le t}\right),
\]

\[
x_{ell+1,t}=a_{ell,t}+
\operatorname{FFN}_{\ell}
\left(\operatorname{RMSNorm}_{\ell}^{ffn}(a_{ell,t})\right).
\]

After the final layer,

\[
z_t=W_{LM}\operatorname{RMSNorm}_{final}(x_{L,t}),
\qquad
u_{t+1}=\arg\max_i z_{t,i}
\]

for the repository's greedy correctness baseline.

The critical fact for storage is that the FFN at a sparse layer is not known until its input vector has been produced by every earlier operation in that layer. This creates a dependency chain:

\[
x_{ell,t}ightarrow\text{attention}\rightarrow a_{ell,t}ightarrow
\text{FFN norm}\rightarrow\text{router}\rightarrow A_{ell,t}ightarrow
\text{expert reads}.
\]

## 4. Hardware operations used by the model

### 4.1 Matrix-vector multiply

For (W\in\mathbb{R}^{m\times n}) and (x\in\mathbb{R}^{n}),

\[
y_i=\sum_{j=0}^{n-1}W_{ij}x_j.
\]

This requires approximately (2mn) floating-point operations: one multiply and one add per weight. If (W) is read once for one vector, the weight-only arithmetic intensity is approximately

\[
I_{mv}\approx\frac{2mn}{mnq}=\frac{2}{q}\ \text{FLOP/byte}.
\]

It is about one FLOP/byte for BF16 weights and two FLOP/byte for one-byte FP8 weights, before scales. This is why batch-one decode is weight-bandwidth limited when weights come from storage.

### 4.2 RMSNorm

For learned scale (w\in\mathbb{R}^d) and (epsilon>0),

\[
\operatorname{RMSNorm}(x)_i=
w_i x_i
\left(\frac{1}{d}\sum_{j=0}^{d-1}x_j^2+\epsilon\right)^{-1/2}.
\]

The implementation is [`rmsnorm_k`](../../src/kimi_exec.cu), with (epsilon=10^{-5}).

### 4.3 SwiGLU

For input (x\in\mathbb{R}^d), intermediate width (d_e), and

\[
\operatorname{SiLU}(v)=v\sigma(v)=\frac{v}{1+e^{-v}},
\]

one expert computes

\[
E_e(x)=W_e^d
\left[
\operatorname{SiLU}(W_e^g x)\odot(W_e^u x)
\right].
\]

Therefore a selected expert always requires all three matrices. In a quantized checkpoint it also requires the scale tensor associated with each matrix. The exact co-required set is

\[
T_{ell,e}=\{
W^g_{packed},S^g,
W^u_{packed},S^u,
W^d_{packed},S^d
\}.
\]

This is not a statistical bundling claim. It follows directly from the SwiGLU dataflow.

## 5. How the model determines the exact expert

### 5.1 Router tensors

At sparse layer (ell), the normalized FFN input is

\[
h_{ell,t}=\operatorname{RMSNorm}^{ffn}_{\ell}(a_{ell,t}).
\]

The router matrix is (W^R_{\ell}\in\mathbb{R}^{E\times d}). It computes

\[
r_{ell,t}=W^R_{\ell}h_{ell,t},
\qquad
s_{ell,t,i}=\sigma(r_{ell,t,i}).
\]

Both Kimi K2.6 and GLM-5.2 use a correction bias (b_{ell,i}) for selection:

\[
c_{ell,t,i}=s_{ell,t,i}+b_{ell,i}.
\]

With the published configurations used here, `n_group = topk_group = 1`, so group filtering retains the only group. The selected set is simply

\[
A_{ell,t}=\operatorname{TopK}_k(c_{ell,t}).
\]

The mixture weight uses the unbiased sigmoid score, not the correction-biased score:

\[
\alpha_{ell,t,e}=
\begin{cases}
\displaystyle
\gamma\frac{s_{ell,t,e}}
{\sum_{j\in A_{ell,t}}s_{ell,t,j}+10^{-20}},
&e\in A_{ell,t},\\[8pt]
0,&e\notin A_{ell,t},
\end{cases}
\]

where (gamma=2.827) for the pinned Kimi path and (gamma=2.5) for GLM-5.2.

The routed output is

\[
y^{routed}_{ell,t}=\sum_{e=0}^{E-1}
\alpha_{ell,t,e}E_{ell,e}(h_{ell,t})
=\sum_{e\in A_{ell,t}}
\alpha_{ell,t,e}E_{ell,e}(h_{ell,t}).
\]

The shared expert is evaluated independently and added:

\[
y^{MoE}_{ell,t}=y^{routed}_{ell,t}+E^{shared}_{\ell}(h_{ell,t}).
\]

### 5.2 Answer to “do we know which expert to fetch?”

Yes, after the router finishes. No predictor is required.

The exact current Kimi sequence is visible at [`src/kimi_exec.cu:836`](../../src/kimi_exec.cu):

1. multiply (W^R h);
2. apply sigmoid;
3. copy the sigmoid vector to a selection vector;
4. add the BF16 correction bias to the selection vector;
5. perform deterministic top-8;
6. copy eight `uint16` IDs to the CPU;
7. resolve exactly six tensors for each ID;
8. read and evaluate those tensors.

The current top-k kernel scans IDs in ascending order and replaces the winner only on strict `>`. Equal scores therefore choose the lower remaining ID. A port must define the same tie behavior or show that different ties cannot affect the validation set.

What is not known is (A_{ell,t}) before (h_{ell,t}) exists. A system may predict or prefetch likely experts, but that is an optional performance speculation. Correct execution waits for the exact router result.

### 5.3 Exactness theorem for selected-expert fetching

Let the reference model define (alpha_e=0) outside its top-k set (A). Let the storage executor:

1. compute the same (A) and (alpha_e);
2. fetch complete tensors (T_{ell,e}) for every (e\in A);
3. evaluate the same expert function (E_e);
4. sum the selected expert outputs in an accepted numerical order.

Then in real arithmetic,

\[
\sum_{e\in A}\alpha_eE_e(x)
=\sum_{e=0}^{E-1}\alpha_eE_e(x).
\]

The proof is immediate because every omitted term has coefficient zero. This does **not** claim that top-k MoE equals a hypothetical dense mixture over all experts. It claims that fetching only top-k experts exactly implements the already sparse model definition.

### 5.4 Storage address after routing

For fixed aligned record stride (S_e'), layer-local expert count (E), and expert-arena base (O_0),

\[
O_{\ell,e}=O_0+(\ell E+e)S_e'.
\]

The router result therefore needs only \((\ell,e)\) to generate an LBA. No filesystem name lookup or learned predictor is on this path.

## 6. Exact expert-byte reductions

### 6.1 Kimi packed INT4

Kimi stores eight signed 4-bit values in one 32-bit word. The current kernel decodes nibble (q\in\{0,\ldots,15\}) as

\[
\hat q=q-8\in\{-8,\ldots,7\}.
\]

For matrix row (i), column (j), and group (g=\lfloor j/32\rfloor),

\[
\hat W_{ij}=S_{i,g}\hat q_{ij},
\]

where (S) is BF16. A matrix with (m\times n) weights uses

\[
S_4(m,n)=\frac{mn}{2}+2m\frac{n}{32}.
\]

For (d=7168,d_e=2048), gate, up, and down each use 8,257,536 bytes, so

\[
S_e^{Kimi}=24{,}772{,}608\ \text{bytes}.
\]

The BF16 equivalent is

\[
3dd_e(2)=88{,}080{,}384\ \text{bytes},
\]

giving

\[
\frac{88{,}080{,}384}{24{,}772{,}608}=3.555556.
\]

Kimi has (E=384,k=8,L_m=60). The routed bank and cold selected bytes are

\[
N_{bank}=60(384)S_e=570{,}760{,}888{,}320,
\]

\[
N_{selected}=60(8)S_e=11{,}890{,}851{,}840.
\]

Thus

\[
\frac{N_{bank}}{N_{selected}}=\frac{384}{8}=48.
\]

### 6.2 GLM E4M3 plus block scales

For an (m\times n) E4M3 matrix with one F32 inverse scale per 128×128 block,

\[
S_8(m,n)=mn+4
\left\lceil\frac{m}{128}\right\rceil
\left\lceil\frac{n}{128}\right\rceil.
\]

With (d=6144,d_e=2048), each projection uses 12,585,984 bytes:

\[
S_e^{GLM}=37{,}757{,}952\ \text{bytes}.
\]

Its 4-KiB-aligned record stride is 37,761,024 bytes. With (E=256,k=8,L_m=75),

\[
N_{bank}=75(256)S_e=724{,}952{,}678{,}400,
\]

\[
N_{selected}=75(8)S_e=22{,}654{,}771{,}200,
\]

and

\[
\frac{N_{bank}}{N_{selected}}=\frac{256}{8}=32.
\]

These 48× and 32× results are byte reductions for the routed bank. They are not end-to-end speedups because attention, shared experts, the head, KV traffic, command latency, and computation remain.

## 7. Multi-head latent attention

### 7.1 Projection path

For normalized residual \(\bar{x}_t\), MLA forms a compressed query and compressed KV state:

\[
c_t^Q=W^{DQ}\bar x_t,
\qquad
q_t=W^{UQ}\operatorname{RMSNorm}(c_t^Q),
\]

\[
[c_t^{KV};k_t^R]=W^{DKV}\bar x_t.
\]

The shared latent (c_t^{KV}\in\mathbb{R}^{r_{kv}}) is expanded for the current computation:

\[
[k_{t,h}^{C};v_{t,h}]=W_h^{UKV}
\operatorname{RMSNorm}(c_t^{KV}).
\]

After rotary embedding, causal attention is

\[
a_{t,j,h}=
\frac{
(q_{t,h}^{C})^Tk_{j,h}^{C}
+(q_{t,h}^{R})^Tk_j^R
}{\sqrt{d_q}}\,m_{rope},
\]

\[
p_{t,j,h}=\frac{e^{a_{t,j,h}}}
{\sum_{r\le t}e^{a_{t,r,h}}},
\qquad
o_{t,h}=\sum_{j\le t}p_{t,j,h}v_{j,h}.
\]

### 7.2 Why latent caching is exact in real arithmetic

Because

\[
(q_h^C)^T W_h^K c_j=(W_h^{K,T}q_h^C)^Tc_j,
\]

the key projection can be absorbed into the query. Because matrix multiplication is linear,

\[
\sum_j p_{j,h}W_h^Vc_j
=W_h^V\left(\sum_jp_{j,h}c_j\right),
\]

the value projection can occur after accumulation. Therefore expanded per-head K/V need not be persisted. The storage record is only

\[
R_{\ell,t}=[c_{\ell,t}^{KV};k_{\ell,t}^{R}].
\]

Finite-precision reassociation can change rounding. A production kernel must compare layer outputs and final logits against the reference; algebraic equivalence alone does not prove bitwise identity.

### 7.3 Capacity result

Kimi's current expanded FP32 cache is

\[
61\left[64(128+64)+64(128)\right](4)
=4{,}997{,}120\ \text{bytes/sequence token}.
\]

Its BF16 latent cache is

\[
61(512+64)(2)=70{,}272\ \text{bytes/sequence token},
\]

a 71.111111× reduction.

GLM's expanded BF16 cache is

\[
78(64)[(192+64)+256](2)
=5{,}111{,}808\ \text{bytes/sequence token}.
\]

Its latent cache is

\[
78(512+64)(2)=89{,}856\ \text{bytes/sequence token},
\]

a 56.888889× reduction.

## 8. Online softmax in one history pass

The current Kimi executor scans K three times and V once. A one-pass tile recurrence maintains maximum (m), normalizer (l), and value accumulator (o). For a new score (s) and value (v),

\[
m'=\max(m,s),
\]

\[
l'=e^{m-m'}l+e^{s-m'},
\]

\[
o'=e^{m-m'}o+e^{s-m'}v.
\]

After all history records,

\[
\operatorname{Attention}=\frac{o}{l}.
\]

For a tile, (s) and (v) are replaced by its local maximum, local exponential sum, and weighted sum. This recurrence is the mathematically stable basis for reading each latent tile once.

## 9. GLM DeepSeek Sparse Attention and IndexShare

GLM has a second exact selection operation over sequence history. At a full-indexer layer, the official implementation forms 32 index query heads of width 128 and one width-128 key per historical token.

Let

\[
q^I_{t,h}=W^{Iq}_h c_t^Q,
\qquad
k^I_j=\operatorname{LayerNorm}(W^{Ik}x_j).
\]

After interleaved RoPE on their rotary components, the per-head nonnegative score is

\[
s^I_{t,j,h}=\operatorname{ReLU}
\left(\frac{(q^I_{t,h})^Tk^I_j}{\sqrt{128}}\right).
\]

The model computes head weights

\[
w^I_{t}=\frac{W^{Iw}x_t}{\sqrt{32}}
\]

and combines them:

\[
g_{t,j}=\sum_{h=1}^{32}w^I_{t,h}s^I_{t,j,h}.
\]

After applying the causal mask, the selected history is

\[
I_t=\operatorname{TopK}_{2048}\{g_{t,j}:j\le t\}.
\]

Attention then uses only positions in (I_t). Layers marked `shared` reuse the preceding full indexer's (I_t); layers 2–5, 6–9, and so on form four-layer groups. This is why a record containing four layers' latents for one historical token is mathematically co-required.

The official source for these operations is [Hugging Face's GLM-MoE-DSA implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py); the dimensions are in the official [GLM-5.2 configuration](https://huggingface.co/zai-org/GLM-5.2/blob/main/config.json).

## 10. The selected-tensor execution algorithm

For one sparse layer and one decoded token:

```text
input: x, layer l, resident router W_R and bias b

h = RMSNorm(x_after_attention)
s = sigmoid(W_R h)
A = exact_top8(s + b)
alpha = routed_scale * s[A] / sum(s[A])

for e in A:
    offset = expert_base + (l * experts_per_layer + e) * record_stride
    submit_read(offset, record_stride, slot[e])

shared = SharedExpert(h)
routed = 0
for each completed slot e:
    (Wg,Sg,Wu,Su,Wd,Sd) = slot[e]
    routed += alpha[e] * Expert(h, Wg,Sg,Wu,Su,Wd,Sd)

return x_after_attention + shared + routed
```

The only optional operations are caching and speculative prefetch. Removing either cannot change mathematical output. A cache hit returns the same immutable record. A speculative miss must still wait for the exact selected record.

## 11. Byte model

For MoE layers (mathcal{M}), selected sets (A_{ell,t}), cache (C_t), and record size (S_{e,ell}), routed bytes are

\[
N^{routed}_t=
\sum_{\ell\in\mathcal{M}}
\sum_{e\in A_{\ell,t}\setminus C_t}S_{e,\ell}.
\]

With fixed (k), no cache, and fixed record size,

\[
N^{routed}_{cold}=|\mathcal{M}|kS_e.
\]

Reading every expert would cost

\[
N^{routed}_{all}=|\mathcal{M}|ES_e,
\]

so the exact routed-byte reduction is

\[
\frac{N^{routed}_{all}}{N^{routed}_{cold}}=\frac{E}{k}.
\]

For a window cache

\[
C_{\ell,t}^{(w)}=\bigcup_{\tau=t-w+1}^{t}A_{\ell,\tau},
\qquad
\Delta_{\ell,t+1}=A_{\ell,t+1}\setminus C_{\ell,t}^{(w)},
\]

and

\[
N^{routed}_{t+1}=S_e\sum_{\ell}|\Delta_{\ell,t+1}|.
\]

Larger (w) can reduce misses but cannot reduce the union's required cache capacity:

\[
D_{cache}=S_e\sum_{\ell}|C_{\ell,t}^{(w)}|.
\]

Reuse is therefore a measured capacity-for-bandwidth trade, not a free theorem.

## 12. Command and latency model

An application record of size (S) is split into

\[
n_{cmd}(S)=\left\lceil\frac{S}{S_{cmd}}\right\rceil
\]

physical NVMe commands. For controller page size (P), this uGDS implementation is limited to one PRP-list page:

\[
S_{PRP}=\left(\frac{P}{8}+1\right)P,
\qquad
S_{cmd}=\min(S_{MDTS},S_{PRP}).
\]

At (P=4096), (S_{PRP}=2{,}101{,}248) bytes. That makes a Kimi expert 12 commands and an aligned GLM expert 18 commands if MDTS permits the PRP cap. The 128-KiB fallback makes them 189 and 289 commands.

For command payload (s), command latency (L(s)), in-flight depth (Q), media limit (B_m), and link limit (B_l),

\[
B_{eff}\le
\min\left(B_m,B_l,\frac{Qs}{L(s)}\right).
\]

The minimum depth for target (B_*) is

\[
Q_{sat}=\left\lceil\frac{B_*L(s)}{s}\right\rceil.
\]

For a serial path,

\[
T_{IO}=n_{cmd}\ell_{cmd}+\frac{N_{physical}}{B}.
\]

For a queued path, the additive (n_{cmd}\ell_{cmd}) expression is no longer valid because latencies overlap; the queue-bound inequality above applies. End-to-end time still obeys

\[
T_{token}=T_{selection}+T_{weightIO}+T_{KVIO}+T_{compute}+T_{control}-T_{overlap}.
\]

Every claimed speedup must identify which term changed.

## 13. Prefill versus decode

If one weight matrix is applied to (b) token vectors as a matrix-matrix operation, the weight bytes are read once while useful FLOPs grow by (b):

\[
I_b\approx\frac{2mnb}{mnq}=b\frac{2}{q}.
\]

For a token block (B), a MoE layer needs the union

\[
U_{\ell,B}=\bigcup_{t\in B}A_{\ell,t}.
\]

Expert bytes per prompt token become

\[
\frac{S_e|U_{\ell,B}|}{|B|}.
\]

This is why blocked prefill can amortize fixed matrices but must measure expert-union growth. Decode at batch one has no such within-step reuse and depends more on cache hits across tokens.

## 14. Conditions required for reference correctness

1. Token IDs and chat formatting match the pinned tokenizer.
2. Every tensor shape, dtype, and scale geometry matches the checkpoint.
3. RMSNorm, RoPE/YaRN, attention masks, router correction, top-k tie behavior, and routed scaling match.
4. Every selected expert record is complete and byte-identical to its source tensors after relayout.
5. Quantized kernels apply the same signed decoding and scales.
6. Storage completion precedes every consumer kernel; no DMA slot is reused early.
7. Cache hits return immutable bytes for the same `(layer, object_id)`.
8. Latent-attention reassociation stays within the accepted numerical tolerance.
9. Final logits and greedy token IDs match the reference prompts.

## 15. What is proved and what remains experimental

### Proved by model definition and integer geometry

- The exact expert IDs are known after the router.
- Unselected experts have zero coefficient in the model's top-k MoE output.
- Kimi and GLM routed-bank byte reductions are 48× and 32×.
- Six expert payloads are deterministically co-required.
- The native expert representations use 3.555556× and 1.999512× fewer bytes than BF16.
- Latent KV uses 71.111111× and 56.888889× fewer bytes than the compared expanded caches.
- The cold useful weight steps are 32,986,459,136 and 41,383,396,416 bytes.

### Source-audited but not target-measured

- Current Kimi execution performs at least 217,686 application model reads per sampled token.
- The current path uses synchronous uGDS reads and repeated buffer registration.
- The current uGDS batch path has 128 application entries, queue depth up to 512, and 64 PRP-list pages.

### Requires experiments

- PCIe peer-to-peer routing on the specified motherboard.
- Actual MDTS, command latency, bandwidth, thermal behavior, and queue saturation.
- Expert reuse distributions for real prompts.
- Reference token agreement on the full target.
- Numerical error of latent reassociation and FP8 kernels.
- NAND internal read amplification and channel placement, which the consumer SSD does not expose.

## References

1. Blok Kimi CUDA path, [`src/kimi_exec.cu`](../../src/kimi_exec.cu).
2. Blok model contract, [`blok/runtime.py`](../../blok/runtime.py).
3. Moonshot AI, [pinned Kimi K2.6 implementation](https://huggingface.co/moonshotai/Kimi-K2.6/blob/7eb5002f6aadc958aed6a9177b7ed26bb94011bb/modeling_deepseek.py).
4. Z.ai, [GLM-5.2 configuration](https://huggingface.co/zai-org/GLM-5.2/blob/main/config.json).
5. Hugging Face, [GLM-MoE-DSA implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py).
6. Alizadeh et al., [*LLM in a Flash*](https://arxiv.org/abs/2312.11514).
