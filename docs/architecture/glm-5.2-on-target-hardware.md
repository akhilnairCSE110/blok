# GLM-5.2 FP8 on a 16 GB Consumer GPU with NVMe-Resident Weights

## A capacity proof, direct-storage execution specification, and extension of the Blok thesis to DSA and IndexShare

**Status:** proposed Blok port; GLM execution is not implemented in this repository

**Date:** 2026-08-09

**Target:** Ryzen 9 5950X, 48 GB host DRAM, RTX 5060 Ti 16 GB (`sm_120`), CUDA 12.8+, Samsung 990 EVO Plus 1 TB, Linux, uGDS

**Feasible checkpoint on this device:** official `zai-org/GLM-5.2-FP8`; the official BF16 checkpoint is too large for 1 TB

## Abstract

This note specifies how GLM-5.2's autoregressive text backbone could execute on the exact Blok hardware even though neither the checkpoint nor one token's active weights fit in 16 GB of VRAM. It first proves a necessary feasibility distinction. The official BF16 safetensor index contains 1,506,659,919,872 tensor bytes and cannot be stored on a nominal 1 TB device. The official FP8 index contains 755,617,140,416 tensor bytes and is storage-feasible. Its 78-layer backbone uses 256 routed experts per sparse layer, eight selected experts per token, multi-head latent attention (MLA), DeepSeek Sparse Attention (DSA), and IndexShare.

Using the official configuration, Hugging Face implementation, FP8 block geometry, and tensor headers, the stated greedy base-model contract with no weight-cache hits requires 41,383,396,416 useful weight bytes, excluding the optional multi-token-prediction layer. Of these, 22.655 GB are the 600 selected routed-expert records. At the Samsung drive's vendor-rated 7.15 GB/s ceiling, the cold weight path alone is bounded below by 5.79 seconds per token. This is an upper bound of roughly 0.173 token/s before transaction latency, DSA index scans, KV gathers, dequantization, and computation.

GLM-5.2 also reveals a deeper association with the Blok idea. IndexShare makes the same 2,048 historical token indices co-required by four adjacent attention layers. Therefore the natural direct-storage record is not merely a tensor or an expert: it can be a cross-layer latent-KV bundle keyed by historical token. An exact latent cache needs 89,856 BF16 bytes per sequence token for the backbone; the 21 full indexers need another 5,376 bytes per token. At the 1,048,576-token limit, both consume 99.858 GB, so the official FP8 weights and compressed caches fit within a nominal 1 TB capacity with 144.525 GB of payload margin. By contrast, the expanded BF16 K/V cache used by the current generic Transformers implementation would consume 5.360 TB. Thus an exact latent-KV kernel is not an optional optimization on this hardware; it is part of the capacity proof.

The decision-level hardware specification, including rejected alternatives, queue equations, implementation mechanisms, and falsification tests, is in [`hardware-design-decisions.md`](hardware-design-decisions.md).

## 1. Research question and answer

The question is not whether GLM-5.2 can be loaded into 16 GB. It cannot. The question is:

> Can a correct autoregressive step be decomposed into a small resident state and a sequence of direct-storage transfers whose byte volume, request count, cache capacity, and lower-bound latency are explicitly known?

For the official FP8 checkpoint, the answer is **yes in capacity and architecture, pending implementation and target measurement**. The BF16 answer is **no on the specified 1 TB device**, before performance is considered.

The system principle is the same as in the Kimi implementation: keep the full parameter universe on NVMe, compute the model's own selection predicates first, fetch only the resulting conditional working set, and tile every matrix operation through GPU memory. GLM adds two important structures: sparse selection over history through DSA, and deterministic cross-layer reuse of that selection through IndexShare.

## 2. Claim discipline and current repository boundary

This document distinguishes:

- **Verified source fact:** present in an official GLM artifact, official implementation, or current Blok source.
- **Derived result:** follows from verified tensor shapes, dtypes, and algebra.
- **Proposed implementation:** a concrete port design that does not yet exist in the repository.
- **Pending measurement:** requires the target Linux host and bound NVMe device.

The current repository is hard-coded for Kimi K2.6. [`../../blok/runtime.py`](../../blok/runtime.py) enforces Kimi's 61-layer text contract, [`../../scripts/model_fetch.py`](../../scripts/model_fetch.py) recognizes only the pinned Kimi model, and [`../../src/kimi_exec.cu`](../../src/kimi_exec.cu) accepts only BF16 and packed-INT4 Kimi tensors. A GLM run would require a new materializer contract, FP8 kernels, DSA/IndexShare support, and a new executor. No benchmark or correctness result in this note is presented as already observed.

## 3. Verified GLM-5.2 architecture

Z.ai describes GLM-5.2 as a `744B-A40B` model and publishes BF16 and FP8 checkpoints in the official [GLM-5 repository](https://github.com/zai-org/GLM-5). The [model card](https://huggingface.co/zai-org/GLM-5.2) reports a stable 1M-token context, IndexShare, a 2.9× reduction in per-token FLOPs at 1M context, and up to 20% greater MTP acceptance length. This note uses checkpoint byte totals, rather than rounded parameter labels, for device-capacity proofs.

The official [GLM-5.2 configuration](https://huggingface.co/zai-org/GLM-5.2/blob/main/config.json) gives:

| Quantity | Symbol | Value |
|---|---:|---:|
| Backbone layers | \(L\) | 78 |
| Hidden width | \(d\) | 6,144 |
| Dense FFN width | \(d_{dense}\) | 12,288 |
| Dense layers |  | 3 |
| Sparse MoE layers | \(L_m\) | 75 |
| Attention heads | \(H\) | 64 |
| Query LoRA rank | \(r_q\) | 2,048 |
| KV latent rank | \(r_{kv}\) | 512 |
| Non-rotary Q/K width per head | \(d_n\) | 192 |
| Rotary width per head | \(d_r\) | 64 |
| Value width per head | \(d_v\) | 256 |
| Routed experts per sparse layer | \(E\) | 256 |
| Selected experts per token | \(k\) | 8 |
| Shared experts |  | 1 |
| Expert intermediate width | \(d_e\) | 2,048 |
| Vocabulary | \(V\) | 154,880 |
| Maximum context | \(S\) | 1,048,576 |
| DSA index heads | \(H_i\) | 32 |
| DSA index width | \(d_i\) | 128 |
| DSA selected history | \(K_i\) | 2,048 |
| Full indexers | \(L_i\) | 21 |
| MTP layers |  | 1 |

The router uses sigmoid scores, a correction bias, normalized top-k weights, and routed scaling factor 2.5. RoPE is interleaved with base 8,000,000. The first three layers use dense SwiGLU FFNs and the remaining 75 use MoE FFNs.

The official [Transformers GLM-MoE-DSA implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py) exposes the exact MLA and indexer shapes. A full indexer contains:

\[
W^{iq}\in\mathbb{R}^{4096\times2048},\quad
W^{ik}\in\mathbb{R}^{128\times6144},\quad
W^{iw}\in\mathbb{R}^{32\times6144},
\]

plus a 128-dimensional LayerNorm. It emits 32-bit top-k indices of shape `[batch, sequence, 2048]`. Shared layers consume the previous full layer's indices instead of running a new indexer.

The 78-entry `indexer_types` sequence has full indexers at layers 0, 1, 2, 6, 10, ..., 74. Layers 2–5 form the first four-layer sharing group; layers 6–9 form the next; this pattern continues through layers 74–77. Layers 0 and 1 are standalone full-indexer layers. There are therefore 21 indexer key streams but 78 attention-layer KV streams.

## 4. Checkpoint feasibility on the exact 1 TB device

The official [BF16 safetensor index](https://huggingface.co/zai-org/GLM-5.2/blob/main/model.safetensors.index.json) declares

\[
S_{BF16}=1{,}506{,}659{,}919{,}872\ \text{bytes}.
\]

Since

\[
S_{BF16}>1{,}000{,}000{,}000{,}000,
\]

the BF16 checkpoint cannot reside on the specified drive. No caching or sparse execution technique changes that storage fact unless the checkpoint representation changes or storage capacity increases.

The official [FP8 safetensor index](https://huggingface.co/zai-org/GLM-5.2-FP8/blob/main/model.safetensors.index.json) declares

\[
S_{FP8}=755{,}617{,}140{,}416\ \text{bytes}.
\]

The nominal payload remainder is

\[
D_{remain}=1{,}000{,}000{,}000{,}000-S_{FP8}
=244{,}382{,}859{,}584\ \text{bytes}.
\]

This proves that the tensor payload is storage-feasible. It does not authorize a raw KV range automatically. Safetensor headers, non-weight files, filesystem allocation, partition boundaries, and safety margins must be measured. A GLM port must use the same FIEMAP and no-overlap procedure as [`../../scripts/plan_ugds_layout.py`](../../scripts/plan_ugds_layout.py), then unmount the filesystem before binding the NVMe controller to uGDS.

## 5. The modern form of the *LLM in a Flash* argument

### 5.1 Exact expert sparsity

For sparse layer \(\ell\) and token \(t\), let \(A_{\ell,t}\) be the eight router-selected experts. Then

\[
y_{\ell,t}^{routed}=
\sum_{e\in A_{\ell,t}}\alpha_{\ell,t,e}
W^d_{\ell,e}
\left[
\operatorname{SiLU}(W^g_{\ell,e}x_{\ell,t})
\odot W^u_{\ell,e}x_{\ell,t}
\right].
\]

Once \(A_{\ell,t}\) has been computed exactly, every unselected expert has zero influence on this output. The active routed fraction is

\[
\rho_{GLM}=\frac{8}{256}=3.125\%.
\]

This maps the neuron-active set in [*LLM in a Flash*](https://arxiv.org/abs/2312.11514) to a model-native expert-active set. No auxiliary activity predictor is needed.

### 5.2 Exact co-required expert bundle

The official [FP8 configuration](https://huggingface.co/zai-org/GLM-5.2-FP8/blob/main/config.json) specifies E4M3 weights, dynamic activation quantization, and 128×128 weight-scale blocks. The [official tensor headers](https://huggingface.co/zai-org/GLM-5.2-FP8/blob/main/model-00001-of-00141.safetensors) show F32 inverse-scale matrices with shape

\[
\left\lceil\frac{m}{128}\right\rceil
\times
\left\lceil\frac{n}{128}\right\rceil
\]

for a quantized \(m\times n\) weight. Thus

\[
S_8(m,n)=mn+4
\left\lceil\frac{m}{128}\right\rceil
\left\lceil\frac{n}{128}\right\rceil.
\]

For each expert projection,

\[
S_8(2048,6144)=S_8(6144,2048)=12{,}585{,}984\ \text{bytes}.
\]

Gate, up, and down projections are all necessary for a selected SwiGLU expert, so one deterministic expert record is

\[
S_e=3(12{,}585{,}984)=37{,}757{,}952\ \text{bytes}.
\]

Aligning the record to 4 KiB produces a 37,761,024-byte stride, adding 3,072 bytes. One batch descriptor can name that entire record, but uGDS splits it into physical commands at

\[
S_{cmd}=\min\left(S_{MDTS},\left(\frac{P}{8}+1\right)P\right).
\]

At a 4 KiB controller page, the current one-PRP-list-page cap is 2,101,248 bytes, so the record requires 18 commands if MDTS permits that size. With uGDS's 128 KiB fallback it requires 289. These are physical-command counts; “one expert record” means one batch descriptor and one deterministic address, not one command. The benefit over the original six separately located weight/scale tensors depends on their extent fragmentation and must be measured.

### 5.3 Temporal reuse remains a strict tradeoff

For a window of \(w\) decoded tokens,

\[
C_{\ell,t}^{(w)}=\bigcup_{\tau=t-w+1}^{t}A_{\ell,\tau},
\qquad
\Delta_{\ell,t+1}=A_{\ell,t+1}\setminus C_{\ell,t}^{(w)}.
\]

Routed miss bytes are

\[
N_{routed,t+1}=S_e\sum_{\ell=3}^{77}|\Delta_{\ell,t+1}|,
\]

while cache occupancy is

\[
D_{routed}=S_e\sum_{\ell=3}^{77}|C_{\ell,t}^{(w)}|.
\]

No GLM-5.2 routing trace is present in this repository. Reuse must be measured, not inferred from another model or from the success of IndexShare over attention indices.

## 6. Exact FP8 active-weight byte derivation

The base greedy decoder does not need the auxiliary MTP layer. This section accounts for one 78-layer autoregressive backbone pass and full LM-head argmax. Large linear matrices use E4M3 plus F32 block scales. Norms, router weights, correction biases, the indexer weighting projection, embeddings, and LM head remain BF16 as indicated by the official FP8 tensor map.

### 6.1 MLA attention

The five quantized attention matrices are:

| Matrix | Shape | Stored bytes including scales |
|---|---:|---:|
| `q_a_proj` | \(2048\times6144\) | 12,585,984 |
| `q_b_proj` | \(16384\times2048\) | 33,562,624 |
| `kv_a_proj_with_mqa` | \(576\times6144\) | 3,539,904 |
| `kv_b_proj` | \(28672\times512\) | 14,683,648 |
| `o_proj` | \(6144\times16384\) | 100,687,872 |

After adding query/KV RMSNorm weights and the two decoder RMSNorm weights,

\[
N_{attn/layer}=165{,}089{,}728\ \text{bytes},
\]

so

\[
N_{attn}=78N_{attn/layer}=12{,}876{,}998{,}784\ \text{bytes}.
\]

### 6.2 DSA indexer weights

One full indexer contains quantized \(W^{iq}\) and \(W^{ik}\), BF16 \(W^{iw}\), and BF16 LayerNorm weight and bias:

\[
N_{indexer}=9{,}571{,}008\ \text{bytes}.
\]

Only 21 layers have full indexers:

\[
N_{indexers}=21N_{indexer}=200{,}991{,}168\ \text{bytes}.
\]

### 6.3 FFN weights

One FP8 dense FFN is

\[
N_{dense/layer}=2S_8(12288,6144)+S_8(6144,12288)
=226{,}547{,}712\ \text{bytes}.
\]

The three dense layers consume 679,643,136 bytes per token.

One routed or shared expert consumes

\[
N_{expert}=37{,}757{,}952\ \text{bytes}.
\]

One BF16 router plus correction bias consumes

\[
N_{router}=(256\cdot6144+256)(2)=3{,}146{,}240\ \text{bytes}.
\]

The 75-layer routed selection traffic is

\[
N_{selected}=75\cdot8\cdot37{,}757{,}952
=22{,}654{,}771{,}200\ \text{bytes}.
\]

The complete routed bank alone occupies

\[
75\cdot256\cdot37{,}757{,}952
=724{,}952{,}678{,}400\ \text{bytes},
\]

or approximately 95.9% of the FP8 tensor payload. The router converts this 724.953 GB storage universe into a 22.655 GB cold miss set for one token.

### 6.4 Total

All base-path bytes other than selected routed experts and the LM head are

\[
\begin{aligned}
N_{fixed}={}&N_{attn}+N_{indexers}+3N_{dense/layer}\\
&+75(N_{router}+N_{expert,shared})\\
={}&16{,}825{,}447{,}488\ \text{bytes}.
\end{aligned}
\]

The BF16 LM head is

\[
N_{head}=154880\cdot6144\cdot2
=1{,}903{,}165{,}440\ \text{bytes},
\]

and one BF16 embedding row is 12,288 bytes. Therefore

\[
\boxed{N_{token,FP8}=41{,}383{,}396{,}416\ \text{bytes}}.
\]

This derivation is close to the official 40B-active label for the expected reason: approximately one byte is read for each active FP8 weight, plus F32 block scales and the BF16 exceptions.

For comparison, the same backbone path in BF16 would require about 80.598 GB of useful weights per sampled token, including indexers and head. Its ideal storage floor would be 11.27 seconds/token, but that checkpoint is not capacity-feasible on this device.

## 7. Storage and transaction lower bounds

For request size \(s\), fixed service overhead \(\ell_0\), and asymptotic bandwidth \(B\),

\[
t(s)=\ell_0+\frac{s}{B},
\qquad
T(s)=\frac{sB}{\ell_0B+s}.
\]

Since

\[
T'(s)>0,\qquad T''(s)<0,
\]

larger requests improve effective bandwidth with diminishing returns. A serial cold step with \(n_{cmd}\) physical commands obeys

\[
L_{io}=n_{cmd}\ell_0+\frac{N}{B}.
\]

Samsung rates the 1 TB 990 EVO Plus at up to 7,150 MB/s sequential read ([official data sheet](https://download.semiconductor.samsung.com/resources/data-sheet/samsung_nvme_ssd_990_evo_plus_datasheet_rev.1.0.pdf)). Taking that optimistic rate,

\[
L_{weights}\ge\frac{41.383396416}{7.15}=5.7879\ \text{s/token},
\]

or

\[
R_{decode}\le0.1728\ \text{token/s}
\]

before all other costs.

For expert-record stride \(S_e'=37{,}761{,}024\), the selected-expert command count is

\[
n_{expert}=600\left\lceil\frac{S_e'}{S_{cmd}}\right\rceil.
\]

It is 10,800 commands at the maximum 2,101,248-byte PRP cap and 173,400 commands at the 128 KiB fallback. The fixed path and head require the same per-record calculation after their physical layout is chosen; an assumed 4 MiB application slab is not a command count. This is why the port needs the native uGDS batch queue, adequate PRP/SGL capacity, and a layout created for the access graph. The current CUDA-stream “async” API wraps synchronous I/O in a host callback and does not by itself provide a deep NVMe queue.

## 8. DSA and IndexShare on NVMe

### 8.1 What the indexer computes

At a full-indexer layer, the model forms 32 query heads of width 128 and one width-128 key per historical token. It computes per-head scores, applies ReLU, combines heads using 32 learned weights, enforces causality, and returns the top 2,048 token indices. The official implementation evaluates the scoring matmuls in FP32 and returns `int32` indices. An exact port must preserve scoring, correction, top-k ordering, and tie behavior closely enough to match reference token IDs.

The full indexer key cache is

\[
N_{index-key/token}=21\cdot128\cdot2=5{,}376\ \text{bytes}.
\]

At maximum context,

\[
N_{index-key,total}=5{,}376\cdot1{,}048{,}576
=5{,}637{,}144{,}576\ \text{bytes}.
\]

If these keys remain on NVMe, every decoded token must stream approximately 5.637 GB merely to choose historical positions at maximum context. The layout is favorable—21 large sequential streams of roughly 268 MB each—but the byte floor is still 0.788 seconds at 7.15 GB/s. The 48 GB host DRAM or part of VRAM can retain all index keys, making them a high-value cache candidate.

Without IndexShare, 78 distinct key streams would be required. The key-scan reduction is exactly

\[
\frac{78}{21}=3.714\times.
\]

Z.ai reports a 2.9× reduction in total per-token FLOPs at 1M context because the rest of the model is unaffected; the 3.714× figure here applies only to the number of indexer streams.

### 8.2 Exact latent KV cache

The generic Transformers implementation currently expands and caches per-head K/V and contains a source TODO to use compressed latents for sparse attention. Expanded BF16 storage per sequence token is

\[
\begin{aligned}
N_{expanded/token}
&=78\cdot64[(192+64)+256]\cdot2\\
&=5{,}111{,}808\ \text{bytes}.
\end{aligned}
\]

At maximum context this is

\[
5{,}111{,}808\cdot1{,}048{,}576
=5{,}360{,}119{,}185{,}408\ \text{bytes}.
\]

It is impossible on the target device. The nominal space remaining after FP8 weights would hold at most 47,807 expanded-cache tokens even if nothing else consumed space.

MLA permits exact latent caching. Let \(c_j\in\mathbb{R}^{512}\) and \(k^R_j\in\mathbb{R}^{64}\) be the stored latent and rotary key. For head \(h\),

\[
k^C_{j,h}=W^K_hc_j,
\qquad
v_{j,h}=W^V_hc_j.
\]

Then

\[
(q^C_h)^TW^K_hc_j=(W_h^{K,T}q^C_h)^Tc_j
\]

and

\[
\sum_jp_{j,h}W^V_hc_j
=W^V_h\left(\sum_jp_{j,h}c_j\right).
\]

Thus a custom kernel can absorb the K projection into the query, attend over selected latent records, accumulate values in latent space, and apply the V projection after reduction. In real arithmetic this is the same function; finite-precision evaluation order may prevent bitwise identity and must be validated.

The BF16 backbone latent cache is

\[
N_{latent/token}=78(512+64)(2)=89{,}856\ \text{bytes}.
\]

Adding indexer keys gives

\[
N_{cache/token}=89{,}856+5{,}376=95{,}232\ \text{bytes}.
\]

At maximum context,

\[
N_{cache,total}=95{,}232\cdot1{,}048{,}576
=99{,}857{,}989{,}632\ \text{bytes}.
\]

Therefore

\[
S_{FP8}+N_{cache,total}
=855{,}475{,}130{,}048\ \text{bytes},
\]

leaving 144,524,869,952 nominal bytes for shard/header overhead, filesystem structures, maps, tokenizer data, raw-range safety margins, and other artifacts. This is the central capacity proof for a full-context GLM-5.2 FP8 port.

### 8.3 IndexShare implies a cross-layer storage record

After a full indexer selects token set \(I_g\), the next three shared layers reuse exactly the same historical positions. For a four-layer group \(g\), the required latent records are

\[
R_{g,j}=\{(c_{\ell,j},k^R_{\ell,j}):\ell\in g\},
\qquad |R_{g,j}|=4\cdot576\cdot2=4{,}608\ \text{bytes}.
\]

Storing these four layer records together turns four irregular gathers for token \(j\) into one. One concrete 4-KiB-safe layout pads each group record to 8 KiB. At 2,048 selected positions:

- a four-layer group transfers 16,777,216 bytes in 2,048 requests;
- 19 four-layer groups transfer 318,767,104 bytes;
- two standalone layers transfer another 16,777,216 bytes in 4,096 requests;
- the total is 335,544,320 physical bytes and 43,008 requests.

The useful latent bytes are only

\[
78\cdot2048\cdot1152=184{,}025{,}088\ \text{bytes}.
\]

The padded scheme therefore spends capacity and bandwidth to cut the naïve per-layer request count from 159,744 to 43,008. A better production layout could byte-pack records, sort selected indices, merge adjacent ranges, or use batched scatter/gather. The important association is exact: **IndexShare creates cross-layer co-access, and the storage record should follow that co-access relation.**

If the padded records were persisted for every position, backbone latent records plus index keys would occupy 169,216 bytes per sequence token, or 177,435,836,416 bytes at maximum context. Even that deliberately simple layout remains nominally capacity-feasible beside the FP8 tensor payload, but it reduces the 1 TB payload margin from 144.525 GB to 66.947 GB. This makes the same strict tradeoff visible again: fewer gather transactions consume more storage and transfer padding.

At maximum context, if index keys and selected latents both come from the same SSD, the optimistic byte-only floor becomes approximately

\[
\frac{41.383+5.637+0.184}{7.15}=6.60\ \text{s/token}
\]

using useful latent bytes, before gather overfetch and request latency. Keeping the 5.637 GB index cache in DRAM or VRAM removes the largest context-dependent SSD term.

## 9. Proposed end-to-end executor

### 9.1 Preparation

1. Add an official `GLM-5.2-FP8` model contract pinned to a revision and exact index total.
2. Materialize tensor metadata for E4M3 weights, F32 `weight_scale_inv` blocks, BF16 exceptions, 21 indexers, 78 backbone layers, and the optional MTP layer.
3. Prefer a storage-native relayout that emits controller-aware sequential weight records, contiguous six-payload expert bundles, and cross-layer latent-KV records. Split each application record at the measured `S_cmd`. If the original shards are retained, generate an extent-aware range plan exactly as Blok does for Kimi.
4. Reserve a non-overlapping raw KV region of at least 100 GB plus explicit alignment and safety margin for maximum-context BF16 latents and indexer keys.
5. Generate and verify FIEMAP coverage while mounted, then unmount and bind only the checked controller to uGDS.

### 9.2 One base-model decode step

For current token \(x_t\):

1. Fetch the BF16 embedding row.
2. For layer \(\ell=0,\ldots,77\):
   1. RMS-normalize the residual.
   2. Stream/dequantize the FP8 MLA query and KV projections and produce \(q_t\), \(c_{\ell,t}\), and \(k^R_{\ell,t}\).
   3. Append the BF16 latent record to the raw KV area.
   4. If `indexer_types[ℓ] == full`, compute and append the width-128 index key, scan that indexer's history, and select 2,048 `int32` positions. Otherwise reuse the previous full layer's indices.
   5. Gather only the selected latent records, perform exact absorbed MLA attention, stream the output projection, and add the residual.
   6. RMS-normalize again.
   7. At layers 0–2, stream one FP8 dense SwiGLU FFN. At layers 3–77, evaluate the BF16 router, issue one batch for eight FP8 expert bundles, execute the shared expert, and accumulate normalized routed outputs.
   8. Add the FFN residual.
3. Apply final normalization.
4. Stream or cache the 1.903 GB BF16 language-model head and compute the next-token logits.
5. Greedily select the next token for the correctness baseline.

The Ryzen CPU should own tokenization, metadata, setup, and coarse scheduling. It should not copy tensor payloads between NVMe and GPU. The 16 GB GPU should retain activations, persistent registered staging slabs, routing/indexing state, the LM head or another high-value fixed subset, and a measured hot-record cache.

### 9.3 Optional MTP/speculative path

The one MTP layer is not needed for base next-token semantics, but it may be disproportionately valuable when weights are storage-resident. Z.ai reports acceptance length 5.47 for a seven-step MTP experiment after IndexShare, KVShare, rejection sampling, and end-to-end TV loss, versus 4.56 at baseline ([GLM-5.2 architecture note](https://huggingface.co/blog/zai-org/glm-52-blog)).

If a verification block contains \(d\) proposed positions and accepts random length \(A\), fixed backbone matrices can be read once and applied as matrix-matrix operations. Their long-run fixed-weight bytes per accepted token approach

\[
\frac{N_{fixed}+N_{head}}{\mathbb{E}[A]}.
\]

At \(\mathbb{E}[A]=5.47\), the 18.729 GB fixed-plus-head term alone would amortize to roughly 3.42 GB per accepted token. Routed experts do not divide so simply. For layer \(\ell\), verification needs the union

\[
U_{\ell,d}=\bigcup_{i=1}^{d}A_{\ell,t+i},
\]

so dynamic expert bytes per accepted token are

\[
S_e\frac{\mathbb{E}[\sum_\ell |U_{\ell,d}|]}{\mathbb{E}[A]}.
\]

This quantity requires traces. MTP is therefore a promising storage-amortization mechanism, not an automatic 5.47× end-to-end speedup.

## 10. VRAM and DRAM policy

The no-cache active set is 41.383 GB, so one token's weights do not fit in VRAM. Useful candidate subsets are:

- BF16 LM head: 1.903 GB;
- all 21 full-indexer weights: 0.201 GB;
- full maximum-context index-key cache: 5.637 GB;
- all FP8 backbone attention weights: 12.877 GB;
- one selected expert bundle: 37.758 MB;
- one token's 600 selected expert bundles: 22.655 GB.

At long context, retaining the 5.637 GB index-key cache and LM head uses about 7.54 GB and eliminates 7.54 GB of repeated reads per output token. The remaining VRAM can hold working buffers and hot experts. At short context, pinning more attention weights may have greater value. This is a phase-dependent cache policy.

For object \(i\), size \(s_i\), predicted hit probability \(p_i\), and avoided transaction count \(r_i\), rank cache value by a measured form such as

\[
v_i=p_i\left(\frac{s_i}{B_{source}}+r_i\ell_{0,source}\right),
\]

then solve

\[
\max\sum_i c_iv_i
\quad\text{s.t.}\quad
\sum_i c_is_i\le D_{free}.
\]

The 48 GB host DRAM is large enough for all maximum-context index keys plus a substantial second-tier expert cache. A host hit still crosses PCIe, so the correct decision depends on measured PCIe and NVMe service curves. Direct storage eliminates compulsory host staging; it does not prohibit deliberate multi-tier caching.

## 11. Prefill must be layer-major

A token-at-a-time prefill would read approximately 41 GB of active weights for every prompt position and is unusable for a 1M-context model. Prefill must process a block of token activations through each matrix tile before evicting that tile.

For block \(B\), dense and attention weight traffic is amortized over \(|B|\). At a sparse layer the expert traffic depends on the union

\[
U_{\ell,B}=\bigcup_{t\in B}A_{\ell,t},
\qquad
|U_{\ell,B}|\le\min(256,8|B|).
\]

Fetch each expert in \(U_{\ell,B}\) once, group the routed tokens, execute a matrix-matrix kernel, and scatter the results. DSA prefill should likewise tile index-score computation and cache writes rather than repeatedly scan history from the start. Decode batch one, multi-request serving, and prefill are three different schedules over the same records.

## 12. What is proven, what is inferred, and what remains open

### Proven from public artifacts and algebra

- The official BF16 payload cannot fit on the specified 1 TB SSD.
- The official FP8 tensor payload fits nominally.
- Only eight of 256 routed experts affect one sparse-layer token after exact routing.
- The FP8 no-cache base step carries 41.383 GB of useful weights under the stated tensor contract.
- The cold weight bandwidth floor is at least 5.79 seconds/token at the drive's vendor ceiling.
- Expanded BF16 K/V cannot support 1M context on this device.
- BF16 latent MLA plus BF16 index keys requires 99.858 GB at the configured maximum context and is nominally capacity-feasible beside the FP8 weights.
- IndexShare makes four layers' historical positions deterministically co-accessed and therefore provides a principled cross-layer storage bundle.

### Inferred design consequences

- The index-key cache is likely more valuable in DRAM/VRAM as context grows.
- Large contiguous expert slabs should reduce transaction overhead compared with tensor-row reads.
- MTP verification may amortize fixed weight traffic unusually well in a storage-bound executor.
- FP8 tensor-core execution will likely make storage, routing dependencies, and gather latency more important than raw arithmetic throughput.

### Not yet established

- Reference-correct GLM token IDs on this executor, because no GLM executor exists yet.
- Actual 990 EVO Plus uGDS bandwidth, latency curve, queue depth, and thermal behavior on the target motherboard.
- FP8 E4M3 kernel throughput and numerical parity on the RTX 5060 Ti.
- Routed-expert temporal reuse and DSA index locality on real GLM-5.2 workloads.
- Whether model shards, cache reservation, alignment, and operational safety margins all fit the physical formatted device; only tensor-payload capacity is proven here.
- Bitwise equivalence of absorbed latent attention under finite-precision reordering.

## 13. What to tell a chip designer

GLM-5.2 creates three storage traffic classes:

1. **Large conditional expert records.** After a small BF16 router, 600 of 19,200 layer-experts are needed. Each is a 37.758 MB FP8-plus-scale record.
2. **Large sequential index scans.** At 1M context, 21 width-128 key streams total 5.637 GB per output token if not cached.
3. **Small irregular latent gathers.** The selected 2,048 historical positions are reused across four layers, enabling 8-KiB cross-layer records but still producing tens of thousands of gathers per token.

The complete decision dossier is [`hardware-design-decisions.md`](hardware-design-decisions.md). The GLM-specific requests are these:

| Decision | Concrete implementation | Proven or measured benefit | Acceptance test |
|---|---|---|---|
| Route before expert I/O | Keep 235.968 MB of routers/biases resident and translate eight IDs to record addresses. | Exactly \(256/8=32\times\) fewer routed bytes than reading every expert. | Cold routed payload is 22,654,771,200 useful bytes; logits match the reference. |
| Retain indexer state | Keep 200.991 MB of full-indexer weights and, at long context, 5.637 GB of index keys in VRAM or DRAM. | Avoids an SSD scan of 5.637 GB per token at 1M context. | Device counters show zero index-key SSD bytes on a cache hit. |
| Execute native FP8 records | Apply F32 inverse scales per 128×128 block inside the E4M3 matrix operation. | One expert is 37.758 MB instead of 75.497 MB in BF16, approximately 2× fewer bytes. | No expanded matrix is allocated; kernel agrees with a dequantize-first reference. |
| Store latent KV | Persist 512 BF16 latent and 64 rotary values per layer/token. | 89,856 bytes/token instead of 5,111,808: 56.89× smaller. | Model plus maximum-context latent/index cache stays within the derived 855.475 GB payload. |
| Follow IndexShare in storage | Store the four layers sharing an index set together; test packed and 8-KiB records. | The padded option changes 159,744 gather descriptors to 43,008, but overfetches 151.519 MB. | Use the layout only when measured latency saved exceeds overfetch time. |
| Provision commands from MDTS | Use \(n_{expert}=600\lceil37{,}761{,}024/S_{cmd}\rceil\), not a 4 MiB assumption. | Exposes the real range: 10,800 expert commands at the PRP cap or 173,400 at the fallback. | Startup reports MDTS/page size; counters equal the predicted split count. |

These requests target the actual limiting operations: selection, command generation, storage movement, quantized consumption, and sparse history access.

## 14. Validation plan for the port

1. Pin the official FP8 revision and validate all 141 shards and the exact 755,617,140,416-byte index total.
2. Generate a GLM runtime index and test every dtype, shape, FP8 scale block, layer role, full/shared indexer relation, and expert coordinate on the host.
3. Implement individual CUDA reference kernels: FP8 block-scale matvec, router, indexer, absorbed latent MLA, dense/shared/routed SwiGLU, residual/norm, and head argmax.
4. Compare every layer boundary against the official Transformers implementation on short prompts, then compare final logits and greedy IDs.
5. Add the FIEMAP/raw-range workflow and validate that no cache write can touch a model or filesystem extent.
6. Benchmark the raw I/O service curve and select slab sizes and queue depths from measurements.
7. Run one-token decode with caches disabled to validate the 41.383 GB accounting against device counters.
8. Add index-key retention, expert caching, cross-layer latent records, and async overlap one at a time; attribute each change to bytes removed, requests removed, or overlap gained.
9. Implement blocked prefill and only then attempt long prompts.
10. Treat MTP as a separate measured experiment with acceptance, expert-union, and bytes-per-accepted-token traces.

## References

1. Z.ai, [GLM-5.2 model card](https://huggingface.co/zai-org/GLM-5.2), 2026.
2. Z.ai, [GLM-5 series repository and checkpoint table](https://github.com/zai-org/GLM-5), 2026.
3. Z.ai, [GLM-5.2 configuration](https://huggingface.co/zai-org/GLM-5.2/blob/main/config.json).
4. Z.ai, [GLM-5.2 BF16 safetensor index](https://huggingface.co/zai-org/GLM-5.2/blob/main/model.safetensors.index.json).
5. Z.ai, [GLM-5.2 FP8 configuration](https://huggingface.co/zai-org/GLM-5.2-FP8/blob/main/config.json).
6. Z.ai, [GLM-5.2 FP8 safetensor index](https://huggingface.co/zai-org/GLM-5.2-FP8/blob/main/model.safetensors.index.json).
7. Z.ai, [GLM-5.2: Built for Long-Horizon Tasks](https://huggingface.co/blog/zai-org/glm-52-blog), 2026.
8. GLM-5 Team, [*GLM-5: From Vibe Coding to Agentic Engineering*](https://arxiv.org/abs/2602.15763), 2026.
9. IndexCache authors, [*IndexCache: Accelerating Sparse Attention via Cross-Layer Index Reuse*](https://arxiv.org/abs/2603.12201), 2026.
10. Hugging Face Transformers, [GLM-MoE-DSA reference implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py).
11. Alizadeh et al., [*LLM in a Flash: Efficient Large Language Model Inference with Limited Memory*](https://arxiv.org/abs/2312.11514), 2023.
12. ScaleX-IO, [uGDS: user-space GPU Direct Storage](https://github.com/ScaleX-IO/uGDS).
13. Samsung, [990 EVO Plus data sheet](https://download.semiconductor.samsung.com/resources/data-sheet/samsung_nvme_ssd_990_evo_plus_datasheet_rev.1.0.pdf).
14. NVIDIA, [GeForce RTX 5060 family specifications](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5060-family/).
