# Kimi K2.6 on a 16 GB Consumer GPU with NVMe-Resident Weights

## A direct-storage MoE execution design, mathematical audit, and statement of the Blok thesis

**Status:** systems design and implementation audit, 2026-08-09

**Target:** Ryzen 9 5950X, 48 GB host DRAM, RTX 5060 Ti 16 GB (`sm_120`), CUDA 12.8+, Samsung 990 EVO Plus 1 TB, Linux, uGDS

**Implemented model path:** text-only greedy decoding for pinned Kimi K2.6 revision `7eb5002f6aadc958aed6a9177b7ed26bb94011bb`

## Abstract

Blok asks whether a model whose checkpoint is hundreds of gigabytes can execute on a consumer GPU without pretending that the model fits in VRAM. The answer is to make storage part of the inference memory hierarchy. The SSD holds the full checkpoint; the GPU holds only the current activation state, tensor tiles, selected experts, and deliberately cached records. Direct storage moves required ranges from NVMe into GPU memory without staging the tensor payload through Python or pageable host memory.

Kimi K2.6 is unusually well matched to this experiment. Its text decoder is a 61-layer mixture-of-experts model with 384 routed experts per sparse layer but only eight selected per token. Its official checkpoint is natively INT4 for routed experts. The exact routed-expert bank represented by the text architecture is 570.761 GB, but one token requires only 11.891 GB of routed-expert records when there are no cache hits. Including attention, routers, shared experts, the first dense layer, the embedding row, and the full language-model head, the present executor's useful weight payload is 32.986 GB per sampled token. A 7.15 GB/s SSD therefore imposes a best-case weight-transfer floor of 4.61 seconds per token before request latency, computation, and KV-cache traffic.

That result is not a performance victory yet. It is a precise diagnosis. The current implementation makes at least 217,686 logical model reads per sampled token, stores expanded FP32 keys and values, scans the key cache three times, and does not retain weights across tokens. The research contribution is the association between modern MoE inference and the three independent levers in *LLM in a Flash*: exact conditional sparsity reduces bytes, temporal reuse reduces marginal bytes, and co-required record layout reduces request count. For Kimi's SwiGLU experts, the natural bundle is not a pair of FFN tensors but the six payloads—three packed matrices and their three scale matrices—that become jointly necessary after routing. Multi-head latent attention supplies a second exact reduction: cache the 512-dimensional latent and 64 rotary dimensions, not expanded per-head K/V. This reduces the present KV footprint by 71.1× without changing the mathematical output.

The decision-level hardware specification, including rejected alternatives, queue equations, implementation mechanisms, and falsification tests, is in [`hardware-design-decisions.md`](hardware-design-decisions.md).

## 1. The thesis in one sentence

**A trillion-parameter MoE need not be resident to be executable: the relevant object is the conditional working set for one inference step, and the hardware/software contract should minimize the bytes and transactions needed to materialize that working set.**

This is a memory-hierarchy project, not a claim that NVMe is as fast as HBM. Its purpose is to determine which parts of inference are fundamentally capacity-bound, which are bandwidth-bound, which are transaction-latency-bound, and which can be removed by exploiting structure already present in the model.

## 2. Claim discipline

This document uses three labels:

- **Verified source fact** means a value appears in the pinned checkpoint/configuration, an official model source, or this repository.
- **Derived result** means the value follows algebraically from verified tensor shapes and dtypes.
- **Pending measurement** means the proposition requires the target Linux machine and is not presented as an observed result.

The public Kimi model is multimodal. Blok deliberately materializes only tensors prefixed by `language_model.` and exposes a text-only path. It does not run MoonViT, images, or video. The current sampler is greedy; Moonshot's recommended stochastic settings are therefore not being claimed as reproduced. The target end-to-end hardware smoke remains pending in [`../../PLAN.md`](../../PLAN.md).

## 3. Model and checkpoint contract

The official [Kimi K2.6 model card](https://huggingface.co/moonshotai/Kimi-K2.6) describes a 1T-parameter MoE with 32B activated parameters, 61 layers, one dense layer, 384 routed experts, eight selected experts, one shared expert, 64 attention heads, MLA, SwiGLU, and a 262,144-token context. Blok pins the exact [configuration](https://huggingface.co/moonshotai/Kimi-K2.6/blob/7eb5002f6aadc958aed6a9177b7ed26bb94011bb/config.json) and [reference model implementation](https://huggingface.co/moonshotai/Kimi-K2.6/blob/7eb5002f6aadc958aed6a9177b7ed26bb94011bb/modeling_deepseek.py).

The text contract enforced by [`../../blok/runtime.py`](../../blok/runtime.py) and [`../../src/kimi_exec.cu`](../../src/kimi_exec.cu) is:

| Quantity | Symbol | Value |
|---|---:|---:|
| Decoder layers | \(L\) | 61 |
| Hidden width | \(d\) | 7,168 |
| Dense FFN width | \(d_{dense}\) | 18,432 |
| Dense layers |  | 1 |
| Sparse MoE layers | \(L_m\) | 60 |
| Attention heads | \(H\) | 64 |
| Query LoRA rank | \(r_q\) | 1,536 |
| KV latent rank | \(r_{kv}\) | 512 |
| Non-rotary query/key width per head | \(d_n\) | 128 |
| Rotary width per head | \(d_r\) | 64 |
| Value width per head | \(d_v\) | 128 |
| Routed experts per sparse layer | \(E\) | 384 |
| Selected experts per token | \(k\) | 8 |
| Shared experts |  | 1 |
| Expert intermediate width | \(d_e\) | 2,048 |
| Vocabulary | \(V\) | 163,840 |
| Maximum context | \(S\) | 262,144 |

The official safetensor index reports exactly **595,148,192,736 tensor bytes** across 64 shards. [`../../scripts/model_fetch.py`](../../scripts/model_fetch.py) refuses to declare a download complete unless the revision, shard count, and this index total all match. The Hugging Face API currently reports slightly greater repository storage because repository storage includes more than indexed tensor payload. The indexed total is the relevant number for tensor-capacity arithmetic.

The routed expert tensors use the official compressed-tensors [pack-quantized representation](https://github.com/vllm-project/compressed-tensors/blob/main/src/compressed_tensors/compressors/quantized_compressors/pack_quantized.py): eight signed 4-bit values are packed into one `uint32`; quantization groups contain 32 input values; scales are BF16. Attention, the dense layer, shared experts, routers, embeddings, norms, and the language-model head are BF16 in this checkpoint path.

## 4. From *LLM in a Flash* to a modern MoE

The original [*LLM in a Flash*](https://arxiv.org/abs/2312.11514) idea uses activation sparsity in a dense FFN: predict which neurons will be nonzero, fetch only their associated rows and columns, and exploit reuse over a temporal window. Kimi supplies a cleaner systems predicate at a coarser granularity. Its learned router explicitly selects the active experts.

For sparse layer \(\ell\), token \(t\), normalized input \(x_{\ell,t}\), router scores \(g_{\ell,t}\), and selected set \(A_{\ell,t}\),

\[
A_{\ell,t}=\operatorname{TopK}(g_{\ell,t},k), \qquad |A_{\ell,t}|=8,
\]

and the routed contribution is

\[
y_{\ell,t}^{routed}=
\sum_{e\in A_{\ell,t}} \alpha_{\ell,t,e}
W^{down}_{\ell,e}
\left(
\operatorname{SiLU}(W^{gate}_{\ell,e}x_{\ell,t})
\odot W^{up}_{\ell,e}x_{\ell,t}
\right).
\]

### Lemma 1: routed sparsity is exact after selection

Conditioned on \(A_{\ell,t}\), the parameters of every \(e\notin A_{\ell,t}\) have zero influence on \(y_{\ell,t}^{routed}\). Fetching those parameters cannot change the output.

This is stronger than an approximate neuron predictor. The router is part of the trained model and the executor first computes it exactly, applies sigmoid scores plus the correction bias, chooses the top eight, normalizes their original sigmoid weights, and multiplies by the configured routed scaling factor 2.827. The sparsity fraction is

\[
\rho_{Kimi}=\frac{k}{E}=\frac{8}{384}=2.0833\%.
\]

The router does create a serial dependency: expert addresses are unknown until router evaluation completes. The system must therefore make routing cheap and resident, then immediately issue a batch for all selected expert records.

### Lemma 2: the Kimi bundle contains three projections and three scale tensors

If expert \(e\) is selected, its gate, up, and down projections are all required. In this INT4 format each projection also requires its BF16 scales. These six tensors are a deterministic co-access set, not a correlation guess.

For a matrix with \(m\) rows and \(n\) columns, the stored bytes are

\[
S_4(m,n)=\frac{mn}{2}+2m\frac{n}{32}.
\]

For Kimi, each of the three projection records happens to have the same size:

\[
S_4(2048,7168)=S_4(7168,2048)=8{,}257{,}536\ \text{bytes}.
\]

Therefore one selected expert bundle is

\[
S_e=3(8{,}257{,}536)=24{,}772{,}608\ \text{bytes}.
\]

Relayout does not reduce these useful bytes. It changes the application interface from hundreds of tensor fragments to one addressable record. The current row-sliced implementation requires 352 application tensor loads per selected expert. A record layout requires one batch descriptor per expert, so across 480 selected layer-experts the application-level count changes from

\[
60\cdot8\cdot352=168{,}960
\quad\text{to}\quad
60\cdot8=480.
\]

This does not mean one NVMe command per expert. In the current uGDS batch path, the physical command cap is

\[
S_{cmd}=\min\left(S_{MDTS},\left(\frac{P}{8}+1\right)P\right),
\]

where \(P\) is the controller page size. At \(P=4096\), the one-PRP-list-page cap is 2,101,248 bytes, so one expert requires 12 commands if MDTS permits that size. If uGDS uses its 128 KiB fallback, it requires 189 commands. The record still removes application calls, registrations, address lookups, and shard/extent crossings; its physical-command benefit must be computed from the target MDTS and the source fragmentation.

### Lemma 3: temporal reuse is a capacity-for-bandwidth exchange

Let the layer-local cache hold the union of the last \(w\) selected sets:

\[
C_{\ell,t}^{(w)}=\bigcup_{\tau=t-w+1}^{t}A_{\ell,\tau}.
\]

The next routed miss set is

\[
\Delta_{\ell,t+1}=A_{\ell,t+1}\setminus C_{\ell,t}^{(w)},
\]

and routed bytes become

\[
N_{expert,t+1}=S_e\sum_{\ell=1}^{60}|\Delta_{\ell,t+1}|.
\]

Increasing \(w\) cannot shrink the union's resident size, so this is not a free Pareto improvement:

\[
S_e\sum_\ell |C_{\ell,t}^{(w)}|\le D_{cache}.
\]

Blok has not yet collected Kimi K2.6 routing traces, so it does **not** import the paper's reuse percentage as if it were measured here. The correct next experiment is to record \(A_{\ell,t}\), plot the miss curve versus cache bytes, and select the operating point from the actual 16 GB VRAM and 48 GB DRAM budgets.

## 5. Exact current execution path

The control plane and data plane are intentionally separated.

1. [`../../scripts/model_fetch.py`](../../scripts/model_fetch.py) downloads the pinned 64-shard checkpoint and validates its safetensor index.
2. Materialization reads safetensor headers, filters to `language_model.*`, records dtype, shape, file, byte range, role, layer, and expert, and writes `runtime-index.blok`. It also writes the exact tokenizer ranks and special tokens to `tokenizer.blok`. The model payload is not copied into a second format.
3. [`../../scripts/plan_ugds_layout.py`](../../scripts/plan_ugds_layout.py) runs while the model filesystem is mounted. Linux FIEMAP resolves every required aligned file range to physical device extents. The script rejects coverage gaps, unusable extent flags, non-4-KiB alignment, and any overlap between model extents and the reserved raw KV region.
4. The model filesystem is unmounted. Only the verified NVMe controller is rebound to uGDS. Metadata, binaries, and the extent map remain on the system disk.
5. [`../../blok/runtime.py`](../../blok/runtime.py) applies the pinned tiktoken expression and Kimi chat template, passes explicit token IDs to one CUDA process, enforces a deadline, validates its JSON response, and decodes returned IDs.
6. [`../../src/kimi_exec.cu`](../../src/kimi_exec.cu) validates every required tensor shape and dtype before opening the data path. It resolves logical tensor slices through the FIEMAP-derived map and calls uGDS against raw device offsets.

For each input or generated token at position \(t\), the CUDA executor performs:

1. Fetch one BF16 embedding row.
2. For each of 61 decoder layers:
   1. RMS-normalize the residual stream.
   2. Execute MLA query and compressed-KV projections.
   3. Apply YaRN-scaled RoPE with factor 64 and original context 4,096.
   4. Expand and write FP32 K/V for this layer and position to the reserved raw NVMe range.
   5. Read tiled history, using three K scans and one V scan for numerically stable attention.
   6. Execute the output projection and residual addition.
   7. RMS-normalize again.
   8. At layer 0, execute the BF16 dense SwiGLU FFN. At layers 1–60, execute the BF16 router, fetch eight INT4 routed experts, and execute the BF16 shared expert.
   9. Add the FFN result to the residual stream.
3. On a sampled step, apply the final RMS norm, stream all 163,840 LM-head rows, take a global greedy argmax, and return the token ID.

This is a complete architectural forward path, not a framework offload wrapper. It is also deliberately simple: batch size one, synchronous I/O, no persistent weight cache, repeated buffer registration, and no prefill batching.

## 6. Exact active-weight byte accounting

All quantities below are useful tensor payload, before 4-KiB alignment padding or filesystem fragmentation.

### 6.1 Attention and normalization

For every layer, the BF16 tensors are:

\[
\begin{aligned}
N_{attn/layer}=2[&r_qd+r_q+H(d_n+d_r)r_q\\
&+(r_{kv}+d_r)d+r_{kv}\\
&+H(d_n+d_v)r_{kv}+d(Hd_v)+2d].
\end{aligned}
\]

Substitution gives

\[
N_{attn/layer}=202{,}276{,}864\ \text{bytes},
\]

and

\[
N_{attn}=61N_{attn/layer}=12{,}338{,}888{,}704\ \text{bytes}.
\]

### 6.2 Dense, router, shared, and routed experts

The first dense SwiGLU layer requires

\[
N_{dense}=3dd_{dense}(2)=792{,}723{,}456\ \text{bytes}.
\]

For each of 60 MoE layers,

\[
N_{router/layer}=(Ed+E)(2)=5{,}505{,}792\ \text{bytes},
\]

where the second term is the correction bias, and

\[
N_{shared/layer}=3dd_e(2)=88{,}080{,}384\ \text{bytes}.
\]

The selected routed payload is

\[
N_{routed}=60\cdot 8\cdot 24{,}772{,}608
=11{,}890{,}851{,}840\ \text{bytes}.
\]

For perspective, the complete routed-expert bank implied by these shapes is

\[
60\cdot384\cdot S_e=570{,}760{,}888{,}320\ \text{bytes},
\]

or 95.9% of the indexed checkpoint. Exact routing turns that storage capacity problem into an 11.891 GB per-token miss problem before reuse.

### 6.3 Total per sampled token

Weights that do not depend on which routed experts win total

\[
\begin{aligned}
N_{fixed}={}&N_{attn}+N_{dense}\\
&+60(N_{router/layer}+N_{shared/layer})\\
={}&18{,}746{,}782{,}720\ \text{bytes}.
\end{aligned}
\]

The full LM head is

\[
N_{head}=Vd(2)=2{,}348{,}810{,}240\ \text{bytes},
\]

and the selected embedding row is 14,336 bytes. Therefore the present no-cache useful payload of a sampled token is

\[
\boxed{N_{token}=32{,}986{,}459{,}136\ \text{bytes}}.
\]

The final norm is another 14,336 bytes loaded once per generation, not once per sampled token.

## 7. Request latency is a separate bottleneck from byte volume

For a request carrying \(s\) bytes, use

\[
t(s)=\ell_0+\frac{s}{B},
\qquad
T(s)=\frac{s}{t(s)}=\frac{sB}{\ell_0B+s}.
\]

For \(\ell_0>0\),

\[
T'(s)=\frac{B^2\ell_0}{(\ell_0B+s)^2}>0,
\qquad
T''(s)=-\frac{2B^2\ell_0}{(\ell_0B+s)^3}<0.
\]

Larger transfers produce higher effective bandwidth with diminishing returns. For a synchronous schedule with total bytes \(N\) and \(n_{cmd}\) physical NVMe commands,

\[
L_{io}=n_{cmd}\ell_0+\frac{N}{B}.
\]

The current source implies the following minimum application-level model-load count per sampled token, assuming every logical range lies in one physical extent:

| Source | Loads |
|---|---:|
| Attention and four small norm tensors | 597 per layer |
| Dense FFN at layer 0 | 688 |
| Router, eight routed experts, and shared expert | 2,999 per MoE layer |
| All transformer layers | 217,045 |
| Embedding row | 1 |
| LM head, 256 rows per tile | 640 |
| **Total application calls** | **217,686** |

Physical FIEMAP fragmentation can only increase the number of `uGDSRead` calls. uGDS can then split each call into multiple commands at \(S_{cmd}\), so

\[
n_{cmd}=\sum_{i=1}^{m_{app}}
\left\lceil\frac{s_i}{S_{cmd}}\right\rceil
\ge m_{app}.
\]

Each tensor slice is 4-KiB aligned, so alignment also transfers bytes around the useful payload. The command count cannot be stated exactly until the controller MDTS, page size, and generated extent plan are known.

Samsung rates the 1 TB 990 EVO Plus at up to 7,150 MB/s sequential read; this is a vendor ceiling, not a Blok measurement ([Samsung data sheet](https://download.semiconductor.samsung.com/resources/data-sheet/samsung_nvme_ssd_990_evo_plus_datasheet_rev.1.0.pdf)). Even granting that ceiling,

\[
L_{weight}\ge\frac{32.986459136}{7.15}=4.6135\ \text{s/token}.
\]

Using the 217,686 application calls as an optimistic lower bound on physical commands gives:

| Assumed \(\ell_0\) | \(n_{cmd}\ell_0\) lower bound | Transfer + request floor, before compute/KV |
|---:|---:|---:|
| 10 µs | 2.177 s | 6.790 s |
| 50 µs | 10.884 s | 15.498 s |
| 100 µs | 21.769 s | 26.382 s |

These rows are sensitivity analysis, not measured latency, and they undercount whenever an application call is split. They show why implementing only expert sparsity is insufficient: the current fine-grained schedule can throw away the gain through transaction overhead.

## 8. KV cache: the present design and the exact latent alternative

### 8.1 Present expanded FP32 cache

The executor stores, for every layer and token,

\[
N_K=64(128+64)(4)=49{,}152\ \text{bytes},
\]

\[
N_V=64(128)(4)=32{,}768\ \text{bytes}.
\]

Across 61 layers,

\[
N_{KV,current/token}=61(49{,}152+32{,}768)
=4{,}997{,}120\ \text{bytes}.
\]

At the nominal maximum context this is 1,309,965,025,280 bytes, which cannot fit on the 1 TB device even without the 595 GB model. With a nominal 1,000,000,000,000-byte drive and no other overhead, the model leaves capacity for at most 81,017 such KV tokens. The actual safe limit is lower because shard headers, filesystem allocation, metadata, and reserved margins also consume capacity.

The current numerically stable attention implementation reads K three times and V once. At history length \(t\), its KV read volume per output token is

\[
\begin{aligned}
N_{KV-read}(t)
&=61t[3(49{,}152)+32{,}768]\\
&=10{,}993{,}664t\ \text{bytes}.
\end{aligned}
\]

KV reads exceed the 32.986 GB weight payload after approximately 3,000 prior tokens. At 262,144 tokens, the scan is 2.882 TB for one output token, implying a 407.7-second combined weight-plus-KV storage floor at 7.15 GB/s. Thus the present cache is a correctness vehicle, not a viable long-context design.

### 8.2 Exact MLA latent caching

MLA already computes a 512-dimensional latent \(c_j\) and a 64-dimensional rotary key \(k^R_j\). Per head \(h\), expanded values have the form

\[
k^C_{j,h}=W^K_hc_j,
\qquad
v_{j,h}=W^V_hc_j.
\]

The non-rotary score can be reassociated exactly:

\[
(q^C_h)^T W^K_hc_j
= (W_h^{K,T}q^C_h)^Tc_j.
\]

The value reduction can also be reassociated exactly because \(W^V_h\) is linear:

\[
\sum_j p_{j,h}W^V_hc_j
=W^V_h\left(\sum_j p_{j,h}c_j\right).
\]

Therefore an exact executor can cache only \((c_j,k^R_j)\), absorb the K projection into the query, accumulate attention in latent space, and apply the V projection after reduction. At BF16,

\[
N_{KV,latent/token}=61(512+64)(2)=70{,}272\ \text{bytes}.
\]

This is

\[
\frac{4{,}997{,}120}{70{,}272}=71.11\times
\]

smaller than the current FP32 expanded cache. The full 262,144-token latent cache is 18.421 GB and fits beside the model on the 1 TB drive.

The equivalence above is exact in real arithmetic. A CUDA implementation changes operation order and rounding, so numerical and token-level parity must still be tested; “exact” here means no learned approximation or information-discarding cache representation is introduced.

A fused online-softmax kernel can consume each latent tile once while maintaining running maximum \(m\), denominator \(l\), and weighted latent output \(o\). That changes the long-context scan from three K passes plus one V pass to one latent pass. At maximum context the latent read is 18.421 GB per output token, so the combined no-weight-cache storage floor becomes approximately

\[
\frac{32.986+18.421}{7.15}=7.19\ \text{s/token},
\]

before request latency and compute. It remains slow, but it is finite, capacity-feasible, and two orders of magnitude better than the present long-context path.

## 9. What 16 GB of VRAM should do

The GPU cannot hold the 595 GB checkpoint or even the 32.986 GB no-cache active set. It can nevertheless hold strategically valuable subsets.

- All attention and norm weights total 12.339 GB.
- The LM head totals 2.349 GB.
- Together they total 14.688 GB, theoretically below 16 GB but too close to assume safe after CUDA context, kernels, working buffers, and uGDS registrations.
- One token's selected routed experts total 11.891 GB.
- A single expert record is 24.773 MB.

This suggests two operating regimes that must be benchmarked:

1. **Pin mostly fixed weights.** Keep the LM head and as many attention matrices as fit. Stream shared and routed FFNs. This removes bytes and hundreds to thousands of repeated requests on every token.
2. **Cache expert records by observed reuse.** Keep a smaller fixed core and use the remaining space for hot layer-expert bundles. The replacement value of an object should include both saved bytes and saved transactions.

For cached object \(i\) with size \(s_i\), expected next-step hit probability \(p_i\), and avoided request count \(r_i\), a simple value estimate is

\[
v_i=p_i\left(\frac{s_i}{B}+r_i\ell_0\right).
\]

Cache selection is a capacity-constrained optimization, not merely LRU:

\[
\max_{c_i\in\{0,1\}}\sum_i c_iv_i
\quad\text{s.t.}\quad
\sum_i c_is_i\le D_{free}.
\]

Host DRAM can form a second tier, but it changes the data route. A DRAM hit must traverse PCIe rather than NVMe; its value depends on measured host-to-device bandwidth, pinning overhead, and topology. Direct storage bypasses a mandatory CPU copy; it does not make host DRAM useless as an intentional cache.

## 10. Prefill and decode are different scheduling problems

The current executor calls the complete 61-layer path separately for every prompt token. For a prompt of \(P\) tokens, it can therefore stream roughly \(P\) times the active model weights. That is mathematically correct and operationally poor.

The correct prefill schedule is layer-major and token-blocked. For a token block \(B\), read a dense matrix tile once and multiply it by a matrix of activations. In a sparse layer, fetch the union of experts used by the block:

\[
U_{\ell,B}=\bigcup_{t\in B}A_{\ell,t},
\]

so routed traffic is

\[
N_{prefill,expert}(B)=S_e\sum_\ell |U_{\ell,B}|,
\]

not

\[
S_e\sum_{t\in B}\sum_\ell |A_{\ell,t}|.
\]

Because \(|U_{\ell,B}|\le\min(E,k|B|)\), matrix reuse is guaranteed not to be worse in useful expert bytes, apart from activation buffering and any overfetch introduced by layout. Decode at batch one cannot use spatial token batching, so it depends on temporal caching, speculative verification, or request batching across independent users.

## 11. Negative results and boundaries

Several attractive statements are false or unproven:

- **Direct storage does not remove the bandwidth floor.** It removes staging and enables a cleaner path; 32.986 GB still has to cross a real storage and PCIe hierarchy on a cold step.
- **Quantization is not the same as sparsity.** INT4 makes every selected expert smaller. Routing determines which experts are selected. Their gains multiply but arise from different facts.
- **Bundling does not reduce useful bytes.** It reduces request count and can improve realized bandwidth. Bundling unrelated experts based on correlation can increase bytes when only one is used.
- **Windowing is not free.** A larger temporal union consumes more VRAM or DRAM.
- **The 24% marginal-activation number from the Apple paper is not a Kimi measurement.** Kimi routing traces must decide the reuse curve.
- **The current target has not produced the end-to-end `paris` smoke.** Host contract tests do not substitute for CUDA 12.8, `sm_120`, driver binding, FIEMAP, and physical NVMe execution.
- **The current path is text-only and greedy.** It does not establish multimodal correctness or stochastic sampling equivalence.

## 12. Implementation progress and the next experiment

The completed work is more substantial than a vague intuition:

- A single public Python API owns exact tokenization and error handling.
- A single CUDA executable owns all 61 decoder layers and greedy selection.
- The runtime validates exact tensor roles, shapes, dtypes, INT4 packing, scales, and Kimi constants before inference.
- The 595 GB checkpoint remains in its official shards; a compact metadata index maps tensors without duplicating model bytes.
- FIEMAP converts filesystem ranges into verified raw device extents.
- The raw KV reservation is explicit, locked, aligned, capacity-checked, and rejected if it overlaps any model extent.
- Model and KV payloads use uGDS rather than a Python/CPU tensor staging path. uGDS itself exposes synchronous, batched, and asynchronous facilities ([uGDS repository](https://github.com/ScaleX-IO/uGDS)); the current Blok executor uses the synchronous form.

The next target run should not begin by adding more abstractions. It should establish four measurements:

1. Correct end-to-end token IDs against a reference implementation on a short deterministic prompt.
2. The empirical request service curve \(t(s,Q)\) for 4 KiB through the measured `S_cmd`, over queue depths 1–512 on the bound 990 EVO Plus.
3. Per-token routed expert traces and the miss curve versus cache capacity.
4. A byte-accurate timeline separating model reads, KV reads/writes, GPU kernels, registration, synchronization, and idle gaps.

Only then should the implementation sequence be: persistent registered buffers and the native batch queue; expert-record relayout and batched reads; fixed-weight and temporal expert caches; exact latent MLA cache with one-pass online softmax; and blocked prefill.

## 13. What to tell a chip designer

The complete decision dossier is [`hardware-design-decisions.md`](hardware-design-decisions.md). The Kimi-specific requests are these:

| Decision | Concrete implementation | Proven or measured benefit | Acceptance test |
|---|---|---|---|
| Route before I/O | Keep 330.348 MB of routers/biases resident; pass eight IDs through a coherent mailbox to a base-plus-stride command generator. | Only \(8/384\) routed experts are needed: exactly 48× fewer routed bytes than reading the bank. | Cold routed reads equal 11,890,851,840 useful bytes and logits match the reference. |
| Use persistent DMA slabs | Register a 64-KiB-aligned GPU arena once and manage it as non-overlapping DMA/compute slots. | Removes at least 217,686 register/deregister pairs and about 217,047 allocation/free pairs per sampled token. | Steady-state registration and allocation counts are zero; checksum race test passes. |
| Batch physical commands | Size queue depth as \(Q\ge\lceil B_*L(s)/s\rceil\); enlarge PRP capacity or add SGL support. | Prevents the present synchronous \(Q=1\) path from imposing one latency payment at a time. | Bandwidth-versus-\(s,Q\) reaches the SSD/link limit without repeated queue-empty periods. |
| Consume INT4 directly | Fuse signed-nibble extraction and BF16 group-32 scaling into the matrix operation. | One expert is 24.773 MB instead of 88.080 MB in BF16: 3.556× fewer storage bytes. | No dequantized matrix exists in VRAM; output agrees with a dequantize-first reference. |
| Store latent MLA state | Persist 512 BF16 latent values plus 64 rotary values per layer/token and use online softmax. | KV storage falls from 4,997,120 to 70,272 bytes/token: 71.11×. | At 262,144 tokens KV occupies 18.421 GB, not 1.310 TB, with reference-correct outputs. |

These are the chip claims. More arithmetic throughput is useful only after the selection-to-I/O path, command queue, and memory representation stop starving the arithmetic units.

## References

1. Alizadeh et al., [*LLM in a Flash: Efficient Large Language Model Inference with Limited Memory*](https://arxiv.org/abs/2312.11514), 2023.
2. Moonshot AI, [Kimi K2.6 model card and model summary](https://huggingface.co/moonshotai/Kimi-K2.6), accessed 2026-08-09.
3. Moonshot AI, [pinned Kimi K2.6 configuration](https://huggingface.co/moonshotai/Kimi-K2.6/blob/7eb5002f6aadc958aed6a9177b7ed26bb94011bb/config.json).
4. Moonshot AI, [pinned Kimi reference implementation](https://huggingface.co/moonshotai/Kimi-K2.6/blob/7eb5002f6aadc958aed6a9177b7ed26bb94011bb/modeling_deepseek.py).
5. Moonshot AI, [pinned safetensor index](https://huggingface.co/moonshotai/Kimi-K2.6/blob/7eb5002f6aadc958aed6a9177b7ed26bb94011bb/model.safetensors.index.json).
6. vLLM Project, [compressed-tensors packed quantization implementation](https://github.com/vllm-project/compressed-tensors/blob/main/src/compressed_tensors/compressors/quantized_compressors/pack_quantized.py).
7. ScaleX-IO, [uGDS: user-space GPU Direct Storage](https://github.com/ScaleX-IO/uGDS).
8. Samsung, [990 EVO Plus data sheet](https://download.semiconductor.samsung.com/resources/data-sheet/samsung_nvme_ssd_990_evo_plus_datasheet_rev.1.0.pdf).
9. NVIDIA, [GeForce RTX 5060 family specifications](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5060-family/).
