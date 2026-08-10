# MetalBlok: Exact Conditional Weight Streaming for a 671B Mixture-of-Experts Transformer on Apple Unified Memory

## A correctness-first systems study of DeepSeek-R1 inference from NAND-backed storage

**Artifact date:** 2026-08-10  
**Implemented target:** Apple Silicon M5, 24 GB unified memory, macOS, Metal  
**Model artifact:** DeepSeek-R1 671B, three-shard `UD-IQ1_S` GGUF  
**Checkpoint payload:** 140,231,438,464 bytes, 1,025 tensors  
**Implementation:** [`../src/gguf_runtime.cpp`](../src/gguf_runtime.cpp), [`../src/kernels.metal`](../src/kernels.metal)

---

## Abstract

MetalBlok executes a 671,026,419,200-parameter mixture-of-experts (MoE)
transformer from a 140.23 GB quantized checkpoint on a computer with 24 GB of
unified memory. The model cannot be made resident. The system therefore treats
storage as a level of the inference memory hierarchy and constructs the exact
weight working set only after the model's own control tensors determine it.

The implementation combines three mathematically distinct reductions. First,
DeepSeek-R1's router selects exactly eight of 256 routed experts. Once that
selection has completed, all unselected expert terms have coefficient zero;
omitting their weights is exact, not predictive. Second, Multi-head Latent
Attention (MLA) permits the key/value cache to retain a 512-dimensional latent
and a shared 64-dimensional rotary key rather than 128 independently expanded
192-dimensional keys and 128-dimensional values. This reduces cache bytes per
sequence position by exactly 71.111×, at the cost of retaining 2.047 GB of
absorbed projection matrices. Third, every stored quant block is decoded inside
the matrix-vector kernel, so no full FP16 weight matrix is materialized.

The exact GGUF descriptor ledger gives 13.771 GB of useful steady-state weight
reads per token and 1,937 logical reads. Reading every routed expert instead
would require 138.861 GB and 45,089 reads. Thus MoE selection reduces complete
token weight bytes by 10.084× and routed-expert bytes by exactly 32×; these two
numbers are intentionally not conflated. The crash-contained one-token process
mode additionally reconstructs absorbed MLA weights and resident norms, giving
14.614 GB and 2,243 reads per isolated step. On the measured M5 system, a full
61-layer isolated step takes 8.20–8.60 seconds and ends at 1.08–1.77 GB process
RSS. A 32-generated-token checkpointed soak completed at position 38.

This artifact establishes executability and a rigorous storage/compute model.
It does not claim optimized throughput, end-to-end logit equivalence to a
second runtime, direct NAND control, or a 32× end-to-end speedup. In fact, the
current implementation deliberately serializes I/O and submits thousands of
small command buffers. Those limitations expose concrete chip requirements:
selection-to-storage command generation, expert-record layout, persistent
quantized slabs, fused graph execution, and latent-attention support.

---

## 1. Claim discipline

A hardware report is only useful if the reader can tell what is a theorem,
what is a measurement, and what remains a proposal. This document uses five
labels.

| Label | Meaning |
|---|---|
| **Exact algebra** | Follows from the model equation without a distributional assumption. |
| **Derived** | Computed from exact tensor shapes, dtypes, or file descriptors. |
| **Implemented** | Exists on the executing GGUF/Metal path. |
| **Measured** | Observed in a saved run on the named M5 system. |
| **Proposed** | A hardware or software improvement not present in the measured path. |

The central claims are:

1. **Exact algebra:** after routing, only eight routed experts contribute to a
   DeepSeek-R1 MoE layer.
2. **Derived:** this is a 32× reduction for the routed bank, but only a 10.084×
   reduction in complete useful weight bytes after attention, shared experts,
   router, dense layers, and output head are included.
3. **Exact algebra:** MLA can compute the same non-rotary attention score and
   value result from a latent cache by matrix reassociation.
4. **Derived:** the implemented latent cache uses 70,272 bytes per sequence
   position across 61 layers, versus 4,997,120 bytes for expanded FP16 K/V.
5. **Implemented and measured:** the exact 140.23 GB checkpoint completes the
   61-layer graph and emits tokens under a bounded memory ledger.
6. **Not claimed:** the current runtime is not throughput-optimal and its
   coherent output is not proof of full reference-logit parity.

---

## 2. The original intuition and its precise translation

MetalBlok follows the storage-level reasoning in
[*LLM in a Flash*](https://arxiv.org/abs/2312.11514), but the predicate in this
model is different.

### 2.1 Request cost

For a storage request carrying $s$ useful bytes, use

\[
t_{io}(s)=\ell_0+\frac{s}{B_{\max}},
\qquad
T(s)=\frac{s}{t_{io}(s)}
=\frac{sB_{\max}}{\ell_0B_{\max}+s}.
\]

Here $\ell_0>0$ is fixed command, filesystem, controller, and scheduling
latency, while $B_{\max}$ is the asymptotic storage bandwidth. Then

\[
T'(s)=\frac{\ell_0 B_{\max}^2}
{(\ell_0B_{\max}+s)^2}>0,
\]

so larger requests deliver higher effective bandwidth. For total payload
$N=ms$ transferred by $m$ serial equal requests,

\[
t_{total}=m\ell_0+\frac{N}{B_{\max}}.
\]

Reducing useful bytes $N$ and reducing request count $m$ act on different
additive terms. They must be measured separately.

### 2.2 Dense activation sparsity versus routed MoE sparsity

For a dense ReLU FFN,

\[
h=\operatorname{ReLU}(W_{up}x),\qquad y=W_{down}h,
\]

zero coordinates of $h$ can make associated weights irrelevant. In the
original flash work, an auxiliary predictor estimates those coordinates before
loading them. A false negative changes the output.

DeepSeek-R1 uses a different and stronger systems predicate. Its MoE is

\[
y_{routed}=\sum_{e=0}^{255}\alpha_e E_e(x),
\]

and the model's router explicitly sets $\alpha_e=0$ for 248 experts. The
system first computes the real router and then loads the eight experts with
nonzero coefficients. There is no expert predictor in MetalBlok and therefore
no predictor false-negative approximation.

### 2.3 What carries over

Three associations remain valid:

- **Conditional bytes:** the router supplies an exact dynamic storage
  predicate.
- **Temporal reuse:** expert IDs repeat across nearby tokens and could support
  a window cache, although the correctness-first GGUF runtime does not yet
  retain expert weights.
- **Co-required layout:** gate, up, and down projections of a selected SwiGLU
  expert are deterministically co-required and form a natural storage record.

The measured implementation proves the first association. The latter two are
documented hardware directions, not measured speedups.

---

## 3. Exact artifact

The GGUF metadata declares:

| Quantity | Symbol | Value |
|---|---:|---:|
| Transformer layers | $L$ | 61 |
| Hidden width | $d$ | 7,168 |
| Vocabulary | $V$ | 129,280 |
| Attention heads | $H$ | 128 |
| Query low-rank width | $r_q$ | 1,536 |
| KV latent width | $r_{kv}$ | 512 |
| Non-rotary query/key width | $d_n$ | 128/head |
| Rotary width | $d_r$ | 64/head for Q, shared for latent K |
| Value width | $d_v$ | 128/head |
| Dense FFN width | $d_f$ | 18,432 |
| Leading dense layers | $L_d$ | 3 |
| MoE layers | $L_m$ | 58 |
| Routed experts/layer | $E$ | 256 |
| Selected experts | $k$ | 8 |
| Expert intermediate width | $d_e$ | 2,048 |
| Shared experts/layer | $E_s$ | 1 |
| Routed scale | $\gamma$ | 2.5 |
| Expert groups retained | $G'/G$ | 4/8 |
| RMS epsilon | $\epsilon$ | $10^{-6}$ |
| RoPE base | $\theta$ | 10,000 |
| YaRN factor | $f$ | 40 |
| Declared context | $C_{model}$ | 163,840 |
| Verified runtime context | $C$ | 64 |

The tensor family count is exactly

\[
3+61(9)+3(3)+58(8)=1025.
\]

The terms are three global tensors; nine common tensors per transformer layer;
three dense FFN tensors for each of three leading layers; and eight MoE tensor
families for each of 58 sparse layers. The three routed expert banks each store
256 matrices inside one rank-three tensor.

The parameter count follows directly:

\[
P_{global}=2Vd+d=1,853,365,248,
\]

\[
P_{attn+norm,layer}=187,121,664,
\]

\[
P_{dense,layer}=3dd_f=396,361,728,
\]

\[
P_{moe,layer}
=dE+E+3dd_e+3dd_eE
=11,320,164,608.
\]

Therefore

\[
P=P_{global}+61P_{attn+norm,layer}
+3P_{dense,layer}+58P_{moe,layer}
=671,026,419,200.
\]

This is the origin of “671B”; it is not the number of weights multiplied for
one token.

---

## 4. Complete autoregressive forward pass

Let $u_0,\ldots,u_{T-1}$ be tokenizer IDs. For a batch-one decode, only the
new position (t) is advanced. Activations are FP16; reductions and routing
weights accumulate in FP32 unless stated otherwise.

### 4.1 Tokenization and embedding

The implemented prompt is

\[
\texttt{BOS}\;\Vert\;\texttt{<|User|>}\;\Vert\;p\;\Vert\;
\texttt{<|Assistant|><think>\textbackslash n}.
\]

The actual checkpoint uses Unicode full-width separator forms of these special
tokens. The tokenizer treats control and user-defined tokens atomically and
prepends BOS ID 0. For embedding matrix

\[
W_E\in\mathbb{R}^{V\times d},
\]

the residual begins as

\[
x_t^{(0)}=W_E[u_t,:]\in\mathbb{R}^{7168}.
\]

Only one quantized embedding row is read and CPU-dequantized to FP16. The
embedding tensor is Q4_K and occupies 521,256,960 bytes, so one row is

\[
\frac{521,256,960}{129,280}=4,032\text{ bytes}.
\]

### 4.2 Pre-normalized residual block

For layer $\ell$,

\[
\bar{x}_{\ell,t}=\operatorname{RMSNorm}
(x_{\ell,t};g^{attn}_\ell),
\]

\[
a_{\ell,t}=x_{\ell,t}+
\operatorname{MLA}_\ell(\bar{x}_{\ell,t}),
\]

\[
\bar{a}_{\ell,t}=\operatorname{RMSNorm}
(a_{\ell,t};g^{ffn}_\ell),
\]

\[
x_{\ell+1,t}=a_{\ell,t}+
\begin{cases}
\operatorname{DenseFFN}_\ell(\bar{a}_{\ell,t}),&\ell<3,\\
\operatorname{MoE}_\ell(\bar{a}_{\ell,t}),&3\le\ell<61.
\end{cases}
\]

RMSNorm is

\[
\operatorname{RMSNorm}(x;g)_i
=g_i x_i\left(\frac{1}{n}\sum_{j=0}^{n-1}x_j^2+\epsilon\right)^{-1/2}.
\]

The sum of squares and reciprocal square root use FP32; the result is stored in
FP16.

---

## 5. Multi-head Latent Attention, tensor by tensor

### 5.1 Query low-rank projection

The normalized residual is compressed:

\[
q^a_{\ell,t}=W^{QA}_\ell\bar{x}_{\ell,t},
\quad
W^{QA}_\ell\in\mathbb{R}^{1536\times7168}.
\]

It is normalized with learned gain $g^{QA}_\ell\in\mathbb{R}^{1536}$:

\[
\hat q^a_{\ell,t}=\operatorname{RMSNorm}(q^a_{\ell,t};g^{QA}_\ell).
\]

The expansion matrix produces 128 head records of width (128+64=192):

\[
q^{full}_{\ell,t}=W^{QB}_\ell\hat q^a_{\ell,t},
\quad
W^{QB}_\ell\in\mathbb{R}^{24576\times1536}.
\]

Reshape

\[
q^{full}_{\ell,t}\mapsto
\{q^n_{\ell,t,h}\in\mathbb{R}^{128},
q^r_{\ell,t,h}\in\mathbb{R}^{64}\}_{h=0}^{127}.
\]

### 5.2 NEOX rotary transform

GGUF stores the 64 rotary coordinates in split-half NEOX order. For
$m=d_r/2=32$, pair coordinates (i) and (i+m), not adjacent coordinates.
With

\[
\omega_i=\theta^{-2i/d_r},\qquad \phi_{t,i}=t\omega_i,
\]

the transform is

\[
\begin{bmatrix}q'_{i}\\q'_{i+m}\end{bmatrix}
=
\begin{bmatrix}
\cos\phi_{t,i}&-\sin\phi_{t,i}\\
\sin\phi_{t,i}& \cos\phi_{t,i}
\end{bmatrix}
\begin{bmatrix}q_i\\q_{i+m}\end{bmatrix}.
\]

Using the interleaved-pair kernel would be a layout error even though shapes
match. MetalBlok has separate NEOX kernels for the GGUF path.

### 5.3 Latent KV projection and cache write

One multi-query projection creates a latent and shared rotary key:

\[
z^{KV}_{\ell,t}=W^{KVA}_\ell\bar{x}_{\ell,t},
\quad
W^{KVA}_\ell\in\mathbb{R}^{576\times7168}.
\]

Split

\[
z^{KV}_{\ell,t}=[c^{raw}_{\ell,t};k^{r,raw}_{\ell,t}],
\quad c^{raw}\in\mathbb{R}^{512},\ k^{r,raw}\in\mathbb{R}^{64}.
\]

Then

\[
c_{\ell,t}=\operatorname{RMSNorm}
(c^{raw}_{\ell,t};g^{KVA}_\ell),
\qquad
k^r_{\ell,t}=\operatorname{RoPE}_{NEOX}
(k^{r,raw}_{\ell,t},t).
\]

The cache stores exactly $c_{\ell,t}$ and $k^r_{\ell,t}$.

### 5.4 Absorption of $W^{KVB}$

The checkpoint contains

\[
W^{KVB}_\ell\in
\mathbb{R}^{[128\times(128+128)]\times512}.
\]

For head (h), split it into

\[
W^{KVB}_{\ell,h}
=\begin{bmatrix}W^K_{\ell,h}\\W^V_{\ell,h}\end{bmatrix},
\]

where

\[
W^K_{\ell,h}\in\mathbb{R}^{128\times512},
\qquad
W^V_{\ell,h}\in\mathbb{R}^{128\times512}.
\]

Ordinary expanded attention would form

\[
k^n_{\ell,\tau,h}=W^K_{\ell,h}c_{\ell,\tau},
\qquad
v_{\ell,\tau,h}=W^V_{\ell,h}c_{\ell,\tau}.
\]

The non-rotary score term is

\[
(q^n_{\ell,t,h})^T k^n_{\ell,\tau,h}
=(q^n_{\ell,t,h})^T W^K_{\ell,h}c_{\ell,\tau}.
\]

By associativity,

\[
(q^n)^T W^Kc
=((W^K)^Tq^n)^Tc.
\]

Define the absorbed query

\[
q^{eff}_{\ell,t,h}=(W^K_{\ell,h})^Tq^n_{\ell,t,h}
\in\mathbb{R}^{512}.
\]

After softmax, ordinary value aggregation is

\[
o_{\ell,t,h}
=\sum_{\tau\le t}\alpha_{\ell,t,h,\tau}
W^V_{\ell,h}c_{\ell,\tau}.
\]

Linearity gives

\[
o_{\ell,t,h}
=W^V_{\ell,h}
\left(\sum_{\tau\le t}\alpha_{\ell,t,h,\tau}
c_{\ell,\tau}\right).
\]

Define

\[
o^{lat}_{\ell,t,h}
=\sum_{\tau\le t}\alpha_{\ell,t,h,\tau}c_{\ell,\tau}.
\]

Then $o_{\ell,t,h}=W^V_{\ell,h}o^{lat}_{\ell,t,h}$. Both transformations
are exact in real arithmetic. MetalBlok stores the absorbed $W^K$ transpose
and $W^V$ layouts in FP16, so the implementation also introduces the normal
FP16 rounding expected by its activation contract.

### 5.5 Scores and softmax

The score for cached position $\tau\le t$ is

\[
s_{\ell,t,h,\tau}
=\beta\left[
(q^{eff}_{\ell,t,h})^T c_{\ell,\tau}
+(q^r_{\ell,t,h})^Tk^r_{\ell,\tau}
\right],
\]

with

\[
\beta=\frac{m_{YaRN}}{\sqrt{d_n+d_r}},
\quad
m_{YaRN}=(1+0.1\ln 40)^2\approx1.8739.
\]

MetalBlok computes stable three-pass softmax:

\[
m=\max_{\tau\le t}s_\tau,
\quad
z=\sum_{\tau\le t}e^{s_\tau-m},
\quad
\alpha_\tau=\frac{e^{s_\tau-m}}{z}.
\]

It then forms $o^{lat}$, expands it with $W^V$, concatenates all heads,
and applies

\[
y^{attn}_{\ell,t}=W^O_\ell
\operatorname{concat}_{h=0}^{127}(o_{\ell,t,h}),
\quad
W^O_\ell\in\mathbb{R}^{7168\times16384}.
\]

### 5.6 Cache reduction and its cost

An expanded FP16 cache stores, per position across all layers,

\[
N_{expanded}
=61\cdot128\cdot(192+128)\cdot2
=4,997,120\text{ bytes}.
\]

The latent cache stores

\[
N_{latent}
=61\cdot(512+64)\cdot2
=70,272\text{ bytes}.
\]

Thus

\[
\frac{N_{expanded}}{N_{latent}}
=\frac{128(192+128)}{512+64}
=71.111\ldots.
\]

This reduction is not free. The absorbed matrices consume

\[
N_{absorb}
=61\cdot128\cdot512\cdot(128+128)\cdot2
=2,046,820,352\text{ bytes}.
\]

Compared only on resident capacity, absorption breaks even against expanded
cache near

\[
C^*=\frac{N_{absorb}}
{N_{expanded/token}-N_{latent/token}}
\approx415.4\text{ positions}.
\]

At the verified 64-position bring-up context, absorption uses more memory than
an expanded cache would. It was selected because it implements the model's
scalable long-context dataflow and eliminates expanded K/V traffic. This is a
real capacity-versus-fixed-residency tradeoff, not a Pareto improvement.

---

## 6. Dense SwiGLU layers

For layers 0–2, with $d_f=18432$,

\[
g=W^G_\ell\bar a,
\quad
u=W^U_\ell\bar a,
\quad
g,u\in\mathbb{R}^{18432},
\]

\[
z=\operatorname{SiLU}(g)\odot u,
\qquad
\operatorname{SiLU}(v)=\frac{v}{1+e^{-v}},
\]

\[
y^{ffn}=W^D_\ell z\in\mathbb{R}^{7168}.
\]

The three matrices contain

\[
3dd_f=396,361,728
\]

weights per dense layer. No neuron predictor is implemented; all dense FFN
weights are streamed.

---

## 7. What a mixture-of-experts layer is

An MoE layer replaces one large dense FFN with a set of independent FFNs and a
small learned router. It is sparse in computation, not sparse in parameter
storage: all experts exist in the checkpoint, but only selected experts are
evaluated for a token.

### 7.1 Expert function

Each routed expert $e\in\{0,\ldots,255\}$ is a SwiGLU:

\[
E_{\ell,e}(x)=W^D_{\ell,e}
\left[
\operatorname{SiLU}(W^G_{\ell,e}x)
\odot(W^U_{\ell,e}x)
\right],
\]

where

\[
W^G_{\ell,e},W^U_{\ell,e}\in
\mathbb{R}^{2048\times7168},
\quad
W^D_{\ell,e}\in\mathbb{R}^{7168\times2048}.
\]

All three matrices are co-required. Reading only gate and up cannot produce the
7168-dimensional output; reading down without gate and up has no activation to
project.

One shared expert $E^S_\ell$ with the same dimensions is always evaluated.
Its purpose is model-defined shared capacity; it is not controlled by top-k.

### 7.2 Router logits

The router matrix is

\[
W^R_\ell\in\mathbb{R}^{256\times7168}.
\]

It produces

\[
r_e=(W^R_\ell\bar a)_e,
\qquad
p_e=\sigma(r_e)=\frac{1}{1+e^{-r_e}}.
\]

The checkpoint also contains correction bias $b_e$. Selection uses

\[
c_e=p_e+b_e,
\]

but mixture weights use the uncorrected $p_e$. Confusing these two quantities
changes both expert IDs and output coefficients.

### 7.3 Group-limited `noaux_tc` selection

Partition 256 experts into eight contiguous groups of 32:

\[
\mathcal G_g=\{32g,\ldots,32g+31\},
\quad g\in\{0,\ldots,7\}.
\]

Let $c^{(1)}_g,c^{(2)}_g$ be the largest and second-largest corrected scores
in group (g). The group score is

\[
S_g=c^{(1)}_g+c^{(2)}_g.
\]

Retain the four groups with largest $S_g$:

\[
\mathcal H=\operatorname{TopK}_4\{S_0,\ldots,S_7\}.
\]

The candidate experts are

\[
\mathcal C=\bigcup_{g\in\mathcal H}\mathcal G_g,
\quad |\mathcal C|=128.
\]

Select

\[
A_{\ell,t}=\operatorname{TopK}_8\{c_e:e\in\mathcal C\}.
\]

MetalBlok breaks equal group scores and equal expert scores by lower index. This
tie rule is shared by the CPU reference and Metal kernel.

### 7.4 Mixture coefficients

With normalization enabled and routed scale $\gamma=2.5$,

\[
\alpha_e=
\begin{cases}
\displaystyle
\gamma\frac{p_e}{\sum_{j\in A_{\ell,t}}p_j},
&e\in A_{\ell,t},\\[8pt]
0,&e\notin A_{\ell,t}.
\end{cases}
\]

The complete MoE output is

\[
y^{MoE}_{\ell,t}
=E^S_\ell(\bar a_{\ell,t})
+\sum_{e\in A_{\ell,t}}\alpha_eE_{\ell,e}(\bar a_{\ell,t}).
\]

### 7.5 Proof of exact expert omission

For every $e\notin A_{\ell,t}$, $\alpha_e=0$. Therefore

\[
\alpha_eE_{\ell,e}(x)=0
\]

for any expert parameters and any input. Removing all storage reads and
arithmetic used solely to evaluate those terms leaves $y^{MoE}$ unchanged.
The proof is conditional on having computed the exact router result first. It
does not license predicting $A_{\ell,t}$ from the previous token and treating
the prediction as exact.

### 7.6 Parameter and operation reduction

The routed bank in one MoE layer contains

\[
P_{routed}=3dd_eE=11,274,289,152
\]

parameters. Eight active experts contain

\[
P_{selected}=3dd_ek=352,321,536.
\]

Thus

\[
\frac{P_{routed}}{P_{selected}}=\frac{E}{k}=32.
\]

However, a token also evaluates the router and shared expert:

\[
P_{active,MoE}=dE+3dd_e+3dd_ek
=398,196,736.
\]

The 32× statement applies to the routed bank, not to this total and not to the
entire transformer.

### 7.7 Other MoE variants

The implementation also contains a generic top-k kernel for models whose
router uses softmax. Two common variants are:

1. **Full-softmax weights:**
   \[
   p_e=\frac{e^{r_e-m}}{\sum_j e^{r_j-m}},
   \quad A=\operatorname{TopK}_k(p),
   \]
   with selected weights taken from the full distribution.
2. **Selected-softmax renormalization:** select using $r_e+b_e$, then apply
   softmax only over unbiased logits of selected experts.

DeepSeek-R1 uses neither for its executing path. It uses sigmoid probabilities,
correction bias for selection, group limiting, top-8, selected normalization,
and scale 2.5. “MoE top-k” is therefore insufficient as an implementation
specification; the scoring function, correction, grouping, normalization,
scale, and tie rules must all be named.

---

## 8. Output projection and decoding

After layer 60,

\[
h_t=\operatorname{RMSNorm}(x_{61,t};g^{out}),
\]

\[
\ell_t=W_{out}h_t,
\quad
W_{out}\in\mathbb{R}^{129280\times7168}.
\]

The checkpoint has a distinct Q6_K `output.weight` of 760,166,400 bytes. It is
not tied to the Q4_K embedding. Substituting `token_embd.weight` would be a
mathematically different model even though the shapes match.

The correctness baseline is greedy:

\[
u_{t+1}=\operatorname*{argmax}_{v\in[0,V)}\ell_{t,v}.
\]

No temperature, top-p, repetition penalty, or stochastic sampler is part of
the measured path.

---

## 9. Active arithmetic per token

Ignoring elementwise operations and context-dependent attention for a moment,
the number of matrix weights used by one token is

\[
P_{attn,active/layer}=187,105,280,
\]

\[
P_{dense,active/layer}=396,361,728,
\]

\[
P_{moe,active/layer}=398,196,736,
\]

and the output head uses $Vd=926,679,040$ weights. Thus

\[
P_{active/token}
=61P_{attn,active/layer}
+3P_{dense,active/layer}
+58P_{moe,active/layer}
+Vd
=36,624,596,992.
\]

Counting multiply and add as two FLOPs gives a weight-projection floor of

\[
73,249,193,984\text{ FLOPs/token}.
\]

Attention over context length $T$ adds, per layer,

\[
2H T(2r_{kv}+d_r)=278,528T\text{ FLOPs}.
\]

At $T=64$, all 61 layers add about 1.087 GFLOP. The effective model is
therefore close to the DeepSeek report's “37B activated parameters” statement,
but the exact artifact-derived value includes this runtime's output and
projection accounting.

---

## 10. Quantized tensor mathematics

Each matrix-vector kernel reads quantized blocks and accumulates directly into
FP32. Let $b_q$ be stored bytes per 256 weights:

| Type | Block bytes | Bytes/weight | FP16-size reduction |
|---|---:|---:|---:|
| F32 | 1 weight / 4 bytes | 4.000000 | 0.5× |
| Q4_K | 144 | 0.562500 | 3.556× |
| Q5_K | 176 | 0.687500 | 2.909× |
| Q6_K | 210 | 0.820312 | 2.438× |
| IQ2_XXS | 66 | 0.257812 | 7.758× |
| IQ1_S | 50 | 0.195312 | 10.240× |

For row $i$, the kernel computes

\[
y_i=\operatorname{FP16}
\left(\sum_{j=0}^{K-1}D_q(B_{i,\lfloor j/256\rfloor})_j
\operatorname{FP16}(x_j)\right),
\]

where $D_q$ is the exact block decoder. No $N\times K$ dequantized matrix
is allocated.

The full checkpoint type histogram is:

| Type | Tensor count | Payload |
|---|---:|---:|
| F32 | 361 | 0.43 GB |
| Q4_K | 190 | 6.67 GB |
| Q5_K | 116 | 1.17 GB |
| Q6_K | 184 | 2.83 GB |
| IQ2_XXS | 6 | 5.81 GB |
| IQ1_S | 168 | 123.31 GB |

The exact per-format block equations and bit fields are specified in
[`TENSOR_HARDWARE_SPEC.md`](TENSOR_HARDWARE_SPEC.md). They are part of the
hardware contract; “supports INT4/INT2” is not precise enough for GGUF K- and
I-quants.

---

## 11. Storage and memory path

### 11.1 NAND boundary

The model resides on a managed NAND SSD behind APFS, an NVMe controller,
flash-translation layer, ECC, wear leveling, and controller queues. MetalBlok
does not address NAND pages or channels directly. NOR flash would be suitable
for boot firmware or a small immutable descriptor table, not this 140.23 GB
weight payload.

The system's controllable address is a file offset. For tensor descriptor
$j$,

\[
a_j=\text{shard-data-base}_j+\text{tensor-offset}_j.
\]

For expert $e$ in a stacked rank-three tensor whose bank size is $S$,

\[
a_{j,e}=a_j+e\frac{S}{256},
\qquad
s_{j,e}=\frac{S}{256}.
\]

This is the concrete selection-to-address transform.

### 11.2 Metadata versus payload

GGUF headers and tensor descriptors are mapped only through bounded metadata
windows capped at 256 MiB; the actual first-shard metadata is about 5.26 MB.
Payload pages are never mapped. After the flat index is constructed, metadata
mappings are dropped.

Before any payload read, `lstat` checks each shard for:

- exact logical size;
- regular-file status;
- macOS `SF_DATALESS` cloud placeholder status;
- allocated blocks at least 90% of logical bytes.

The exact expected shard sizes are 49,349,193,664; 49,397,904,416; and
41,484,340,384 bytes. A sparse or dataless model is refused without reading its
payload, preventing an accidental 140 GB cloud hydration.

### 11.3 Payload transfer

Each shard has one read-only descriptor configured with

- `F_NOCACHE=1`, avoiding persistent unified-buffer-cache residency;
- `F_RDAHEAD=0`, because the runtime declares its reads explicitly;
- one worker thread and a 64-entry SPSC request ring.

For a weight dispatch:

1. allocate one `MTLStorageModeShared` buffer of exactly the tensor or expert
   slice size;
2. `pread` from the shard directly into the buffer's CPU-visible address;
3. publish completion with release/acquire ordering;
4. bind the same allocation to Metal;
5. dequantize and multiply on GPU;
6. wait for command-buffer completion;
7. release the transient buffer.

Apple documents shared Metal resources as system memory accessible by both CPU
and GPU. Therefore there is no second explicit host-to-discrete-VRAM copy. This
does not mean “zero memory traffic”: SSD DMA and the filesystem populate unified
memory, then the GPU reads it.

### 11.4 Current serialization

Although the read ring can queue requests, the current `stream_gemv` submits
one request and immediately waits. It then commits one Metal command buffer and
waits before releasing the weight. Consequently the measured runtime has queue
depth approximately one on the model critical path and no I/O/compute overlap.
This was a deliberate correctness choice, not the intended final architecture.

---

## 12. Exact byte and request ledger

The GGUF probe derives traffic directly from tensor descriptors. Since all 256
expert slices within a bank have equal size, selected useful bytes are
independent of the eight IDs.

| Path | Useful bytes/token | Logical reads/token |
|---|---:|---:|
| Steady graph, selected experts | 13,770,679,744 | 1,937 |
| Counterfactual, all experts | 138,861,340,096 | 45,089 |
| Isolated process, selected + cold norms + absorbed source | 14,614,222,272 | 2,243 |

The complete-token byte reduction is

\[
\frac{138.861340096}{13.770679744}=10.084.
\]

It is lower than 32× because attention, output head, shared experts, router, and
dense layers are invariant to routed selection.

The isolated process adds:

\[
N_{cold,norm}=4,026,368\text{ bytes in 245 reads},
\]

\[
N_{cold,KVB}=839,516,160\text{ bytes in 61 reads}.
\]

The output head alone is 760,166,400 bytes every token. It is therefore both a
large bandwidth floor and the largest transient allocation.

### 12.1 Request-count interpretation

The 1,937 reads are model objects, not physical NAND transactions. APFS and the
NVMe controller may split or merge them. Still, the logical count is actionable:
58 MoE layers each issue 24 routed expert reads—gate, up, and down for eight
experts—plus shared, router, bias, and attention reads.

### 12.2 Co-required expert record

For selected expert $e$, define

\[
R_{\ell,e}=
\{W^G_{\ell,e},W^U_{\ell,e},W^D_{\ell,e}\}.
\]

The current GGUF layout issues three reads. A relaid record issues one read:

\[
m_{routed,current}=58\cdot8\cdot3=1392,
\]

\[
m_{routed,bundled}=58\cdot8=464.
\]

Useful bytes are unchanged. Under $t(s)=\ell_0+s/B$, the fixed latency term
falls by

\[
(1392-464)\ell_0=928\ell_0
\]

per token. This is the correct bundling claim. It is proposed because the
present checkpoint is not relaid.

### 12.3 Temporal expert cache

Let $A_{\ell,t}$ be the eight selected IDs and

\[
C_{\ell,t}^{(w)}=\bigcup_{\tau=t-w+1}^{t}A_{\ell,\tau}.
\]

The next-token misses are

\[
\Delta_{\ell,t+1}
=A_{\ell,t+1}\setminus C_{\ell,t}^{(w)}.
\]

A weight cache changes routed bytes to the records for $\Delta$, but resident
capacity grows monotonically with $w$. This is a strict capacity/bandwidth
tradeoff. MetalBlok currently uses process isolation and retains no expert
weights across tokens; it does not claim a window-cache speedup.

---

## 13. Metal execution model

All activations and caches are shared-mode FP16 buffers. Each quantized GEMV
launches one threadgroup per output row with 128 threads. Threads stride across
256-weight blocks, decode into registers, multiply FP16 activations, reduce
FP32 partial sums first within 32-lane SIMD groups and then through threadgroup
scratch, and have thread 0 store one FP16 output.

RMSNorm and attention use 256-thread groups; the final argmax uses 1,024
threads. Correctness builds request Metal safe math rather than fast math.

The runtime commits and waits at each graph dependency. This creates about
3,537 Metal command buffers for one full token:

- approximately 11 in each MLA block;
- residual additions;
- 5 dense-FFN stages in three layers;
- approximately 47 MoE-FFN stages in each of 58 layers;
- final norm, output GEMV, and argmax.

This count is a diagnosis. A production accelerator should fuse elementwise
stages, encode more dependent kernels in one command buffer, overlap future
weight reads, and avoid host round trips after routing.

---

## 14. Memory ledger and safety architecture

Before allocating Metal resources, the runtime computes

\[
M_{estimate}=M_{absorb}+M_{KV}+M_{scores}
+M_{largest\ transient}+M_{margin}.
\]

At context 64:

| Component | Bytes |
|---|---:|
| Absorbed $W^K/W^V$ | 2,046,820,352 |
| Latent KV cache | 4,497,408 |
| Attention score scratch | 32,768 |
| Output-head transient | 760,166,400 |
| Runtime margin | 268,435,456 |
| **Estimate** | **3,079,952,384 (3.080 GB)** |

Startup is refused unless

\[
M_{available}\ge M_{estimate}+3\text{ GiB}.
\]

The final 3 GiB is a host reserve, not model memory.

Long correctness runs use an atomic checkpoint containing position, predicted
token, and exact per-layer latent/rotary cache prefixes. For position $p$,

\[
S_{state}(p)=36+p\cdot61\cdot(512+64)\cdot2
\]

bytes, ignoring filesystem allocation rounding. The file is written to
`.partial`, flushed, `fsync`ed, and renamed. A killed process leaves the last
renamed state authoritative.

The convenience runner adds process isolation: one token per child, a timeout,
exclusive wrapper lock, exact +1 position verification, manifest verification,
and refusal if a partial checkpoint exists.

---

## 15. Validation and measured results

### 15.1 Structural validation

- all three shard sizes matched the target manifest;
- every shard was physically resident;
- 1,025 tensor descriptors and 140.23 GB payload were indexed;
- every required tensor family and shape was checked before inference;
- the dedicated output head was required.

### 15.2 CPU/Metal kernel parity

Real checkpoint tensors covered every stored compute type:

| Type | Representative tensor | Relative error |
|---|---|---:|
| F32 | `blk.3.ffn_gate_inp.weight` | $2.04\times10^{-4}$ |
| Q4_K | `blk.0.attn_q_a.weight` | $2.09\times10^{-4}$ |
| Q5_K | `blk.3.ffn_gate_shexp.weight` | $2.06\times10^{-4}$ |
| Q6_K | `blk.0.attn_kv_a_mqa.weight` | $2.13\times10^{-4}$ |
| IQ2_XXS | `blk.3.ffn_down_exps.weight` | $2.06\times10^{-4}$ |
| IQ1_S | `blk.3.ffn_gate_exps.weight` | $2.08\times10^{-4}$ |

The CPU and Metal grouped sigmoid routers produced identical expert IDs and
weights on the validation vector.

### 15.3 End-to-end execution

The exact formatted prompt for `Hi` tokenized as

\[
[0,128803,23166,128804,128798,201].
\]

The model completed all 61 layers and produced the sequence beginning

```text
Okay, so I need to solve this problem. I'm trying to solve this problem.
```

The low-bit checkpoint later became repetitive. Coherence demonstrates a live
graph, tokenizer, router, and output path; it is not a semantic-quality or
reference-logit proof.

The crash-contained soak consumed 32 generated tokens after six prompt tokens,
ending at checkpoint position 38. Across the recorded isolated steps:

- full-graph time: 8.20–8.60 s/token;
- end-of-step RSS: 1.08–1.77 GB;
- ledger estimate: 3.08 GB;
- available memory observed at startup: approximately 15.6–18.5 GB;
- every committed checkpoint advanced by exactly one;
- a process killed externally left a valid checkpoint that resumed.

The isolated useful-byte rate implied by the ledger and measured time is about

\[
\frac{14.614\text{ GB}}{8.2\text{--}8.6\text{ s}}
=1.70\text{--}1.78\text{ GB/s}.
\]

This is an end-to-end useful-byte rate, not raw SSD bandwidth: it includes
thousands of reads, CPU dequantization of absorbed matrices, Metal dispatch,
GPU compute, and synchronization.

---

## 16. Design decisions and rejected alternatives

| Decision | Alternative rejected | Mathematical or operational reason | Cost |
|---|---|---|---|
| Route before expert fetch | Predict experts | Exact router makes false-negative approximation unnecessary | Serial selection barrier |
| Fetch only eight experts | Stream all 256 | Unselected coefficients are exactly zero | Irregular reads |
| Keep checkpoint quantized through GEMV | Expand full matrices | Cuts weight bytes by format factor and bounds transient memory | Decode logic in every dot product |
| Cache MLA latent + rotary key | Expanded per-head K/V | Exact 71.111× cache reduction | 2.047 GB absorbed matrices |
| Require distinct output head | Tie to embedding | Checkpoint contains separately trained projection | 760 MB/token read floor |
| `pread` with `F_NOCACHE` | Payload `mmap` | Avoids page-cache residency of a 140 GB file | Explicit buffers and syscalls |
| Runtime source-compile Metal with safe math | Fast math/offline-only toolchain | Reproducible correctness build on installed system | Startup compilation overhead |
| One dispatch/read at a time | Pipeline immediately | Simplest dependency and lifetime proof | Thousands of sync points |
| One-token process isolation for demo | One long process | Crash containment and atomic recovery | Rebuild 839.5 MB absorbed source each step |
| Context 64 for validation | Advertised 163,840 | Bounded first proof; long context needs separate validation | Does not demonstrate full context |

---

## 17. Hardware implications

The model exposes a serial control boundary:

\[
\bar a_{\ell,t}
\rightarrow W^R_\ell\bar a_{\ell,t}
\rightarrow A_{\ell,t}
\rightarrow \{a_{\ell,e}:e\in A_{\ell,t}\}
\rightarrow\text{expert compute}.
\]

A chip cannot know the exact expert IDs before the current router input exists.
It can speculate, but correctness must wait for the exact result. The valuable
hardware primitive is therefore not “AI prefetch” in the abstract; it is a
low-latency selection-to-command path:

1. router writes eight IDs into a coherent mailbox;
2. address generator evaluates base + layer stride + expert stride;
3. storage engine submits eight bundled expert-record reads;
4. completion bits release expert kernels;
5. accumulator applies eight FP32 mixture coefficients.

For custom NAND-backed hardware, useful decisions are:

- lay one expert's gate/up/down blocks in one logical record;
- stripe that record across channels/dies without changing its logical
  contiguity;
- retain routers, norms, correction biases, output-critical metadata, and
  quant codebooks in DRAM/SRAM;
- provide persistent registered slabs with explicit DMA/compute ownership;
- maintain queue depth sufficient to cover $\ell_0$;
- implement Q4_K, Q5_K, Q6_K, IQ1_S, and IQ2_XXS decode in the matrix unit;
- support FP32 reductions and deterministic top-k tie behavior;
- cache latent KV rather than forcing expanded per-head cache formats;
- expose counters for useful bytes, physical bytes, commands, queue depth,
  cache hits, router time, storage time, and compute time.

The current Apple implementation proves the access graph and math on a managed
SSD. It cannot claim control of physical NAND placement, FTL behavior, or NVMe
firmware scheduling.

---

## 18. Limitations and falsification tests

### 18.1 Known limitations

1. No second-runtime end-to-end logit comparison has been completed.
2. IQ1_S semantic quality is visibly degraded and repetitive.
3. Context 64 is validated; the declared 163,840 context is not.
4. I/O and GPU work are serialized.
5. Expert gate/up/down tensors are not physically bundled.
6. There is no temporal expert cache.
7. Absorbed weights are rebuilt in each isolated child.
8. Router IDs cross a GPU/CPU visibility boundary before storage submission.
9. A per-request `pread` error is logged but not propagated through a rich
   request result object.
10. The runtime is batch one and greedy only.

### 18.2 Falsification tests

- **Router:** compare IDs and weights for real per-layer logits against an
  independent reference, including ties and group boundaries.
- **MLA:** compare latent/absorbed attention outputs to explicitly expanded K/V
  at multiple layers and positions.
- **Quant:** compare every kernel family against byte-identical CPU block
  decoding over randomized rows and real tensors.
- **End to end:** compare layer residuals and final logits to a known-good
  runtime before comparing only sampled text.
- **Byte ledger:** instrument `pread` offsets and sizes; their sum must equal
  descriptor-derived totals absent intentional caching.
- **Bundling:** compare identical useful bytes before/after relayout; attribute
  improvement only to request count and realized bandwidth.
- **Window cache:** report resident bytes and misses together; a byte reduction
  without cache capacity is incomplete.
- **Safety:** record memory pressure, pageouts, command-buffer faults, state
  advancement, and partial-file behavior through forced termination.

---

## 19. Conclusion

The work is not “put a huge model on disk and hope virtual memory handles it.”
The architectural idea is that the model itself produces exact, structured
working-set information:

- the MoE router identifies eight expert records;
- MLA identifies the latent state sufficient for attention;
- quant formats identify the representation sufficient for multiplication;
- tensor dependencies identify values that should be bundled;
- temporal selection history identifies potential cache reuse.

MetalBlok has implemented the first correctness path across the entire 671B
graph. The key result is not current speed. It is the conversion of a vague
“flash-backed LLM” idea into explicit equations, addresses, byte counts,
buffer lifetimes, kernel formats, validation gates, and hardware decisions.

---

## References

1. K. Alizadeh et al., [“LLM in a Flash: Efficient Large Language Model
   Inference with Limited Memory,”](https://arxiv.org/abs/2312.11514) 2023.
2. DeepSeek-AI, [“DeepSeek-V3 Technical Report,”](https://arxiv.org/abs/2412.19437)
   2024.
3. DeepSeek-AI, [“DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via
   Reinforcement Learning,”](https://arxiv.org/abs/2501.12948) 2025.
4. Apple, [“MTLResourceStorageModeShared,”](https://developer.apple.com/documentation/metal/mtlresourceoptions/storagemodeshared).
5. GGUF/GGML format behavior is grounded in the local parser, tensor metadata,
   and vendored quantization codebooks identified in the accompanying hardware
   specification.
