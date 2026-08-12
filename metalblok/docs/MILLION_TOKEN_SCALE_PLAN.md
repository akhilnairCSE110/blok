# Million-input/million-output scale plan

## A physically defensible path from the V0 proof to two million positions

**Target acceptance:** 1,000,000 native input tokens plus 1,000,000 emitted
tokens, ending at committed state position 1,999,999.

**Starting point:** the accepted DeepSeek-R1 671B `UD-IQ1_S` V0 on a 10-core
M5, 24 GB unified memory, one internal SSD, exact expanded FP16 K/V, and
explicit model-weight streaming.

This plan is intentionally separate from the V0 contract. It states what must
change, what can remain exact, which changes require a new numerical reference,
and which physical lower bounds cannot be optimized away.

## 1. Definition of success

The million-token target has four independent gates:

1. **Addressability:** tokenizer, positions, RoPE, state indexes, file offsets,
   and counters must represent at least 2,000,000 positions without overflow.
2. **Capacity:** weights, live working set, and the full two-million-position
   attention state must fit across unified memory and storage with recovery
   headroom.
3. **Execution:** one million input tokens must prefill and one million output
   tokens must decode without rebuilding prior state, corrupting pages, or
   silently evicting context.
4. **Model quality:** the model must be validated at those positions. Merely
   allocating a larger cache is not evidence that a model trained/configured
   for a shorter context remains high fidelity.

For `N` committed input positions and `M` emitted tokens, the state position is

\[
p_{final}=N+M-1.
\]

The minus one has the same pending-token explanation as V0: prefill computes
the first emitted token before one more forward advance. At one million plus
one million, the acceptance state must therefore contain position 1,999,999
and exactly 1,000,000 emitted token IDs.

## 2. Hard limits exposed by V0

### 2.1 Current expanded KV cannot scale

V0 stores 4,005,504 bytes per position. At two million positions:

\[
4{,}005{,}504\times2{,}000{,}000
=8{,}011{,}008{,}000{,}000\text{ bytes},
\]

or 8.011 TB decimal. This does not fit the target machine and cannot be fixed
with lazy virtual allocation; the completed state eventually touches every
page.

### 2.2 Compact MLA changes capacity, not attention complexity

The natural DeepSeek latent record is 512 content coordinates plus 64 shared
rotary-key coordinates per layer. Stored as FP16 across 61 layers:

\[
b_{latent}=61(512+64)2=70{,}272\text{ bytes/position}.
\]

At two million positions this is 140,544,000,000 bytes, or 140.544 GB. The
capacity reduction relative to expanded V0 state is exactly

\[
4{,}005{,}504/70{,}272=57.
\]

That makes storage capacity plausible, but exact dense causal attention still
depends on every preceding key. For one million generated positions following
a one-million-token prompt, the number of historical position-record visits is

\[
\sum_{t=1{,}000{,}000}^{1{,}999{,}999}t
=1{,}499{,}999{,}500{,}000.
\]

Reading one 70,272-byte all-layer latent record for every visit would move

\[
105{,}407{,}964{,}864{,}000{,}000\text{ bytes},
\]

or 105.4 PB, before model weights. A compact cache solves capacity but not the
quadratic total history work of one million dense-attention decode steps.

### 2.3 Model weights remain a second lower bound

At the V0 operating point, one generated token consumes 13,587,632,064 model
bytes. Repeating that unchanged for one million outputs moves 13.588 PB. Even
at 4.6 GB/s sustained file bandwidth, the weight-only floor is about 34 days.
Therefore a usable million-output system also needs weight reuse across
requests/tokens, more storage bandwidth, a checkpoint with fewer active bytes,
or some combination. Kernel tuning alone cannot erase this byte floor.

### 2.4 This checkpoint declares a shorter context

The frozen GGUF declares context 163,840. Two million positions are more than
12 times that contract. YaRN address arithmetic can produce angles at larger
positions, but that does not establish attention quality. A high-fidelity
two-million-position claim requires either:

- a checkpoint trained or continued for that context and corresponding
  tokenizer/RoPE metadata; or
- an independently evaluated context-extension method with task-level and
  logit evidence at the target lengths.

Until then, MetalBlok may test storage and execution mechanics beyond 163,840,
but must label semantic quality unvalidated.

### 2.5 Million-token dense prefill is also quadratic

A one-million-token causal prompt contains

\[
\frac{N(N+1)}2=500{,}000{,}500{,}000
\]

query/key position pairs per attention head. Across 61 layers and 128 heads,
that is 3,904,003,904,000,000 head-pairs before counting score-vector width,
softmax, or value accumulation. Flash/online attention removes quadratic
*storage* for scores; it does not remove these exact query-dependent dot
products. Quantized QMM accelerates projections but does not change attention's
asymptotic work.

Consequently, a mechanically exact dense-R1 million-token prefill is a very
long systems proof on one M5, not an interactive workload. An interactive
million-token product needs a checkpoint trained for a sparse/windowed/
retrieval attention graph, or far more compute and storage bandwidth. The
project must report which target it is pursuing.

## 3. Target memory hierarchy

The scale architecture has four explicit tiers:

```text
Tier 0: registers/threadgroup memory
        quant block, online-softmax state, matrix tile

Tier 1: unified memory
        activations, current KV pages, output head or hot tiles,
        fixed/expert cache, I/O rings, page tables

Tier 2: local NVMe
        compressed model records, cold KV pages, append-only checkpoints,
        tokenizer/prompt file, trace summaries

Tier 3: optional additional local NVMe devices
        striped immutable model records and/or cold KV page sets
```

Every object must have one owner, a byte budget, an eviction rule, and a logged
hit/miss/read/write count. VM paging is not a fifth scheduler tier. MetalBlok
must continue using explicit pages so macOS compression or swap cannot silently
decide which model/state object is displaced.

## 4. Numerical fork required for compact MLA

Over real arithmetic, the non-rotary score and value equations allow

\[
q^T(W_Kc)=(W_K^Tq)^Tc,
\]

\[
\sum_jp_j(W_Vc_j)=W_V\left(\sum_jp_jc_j\right).
\]

This permits caching `c_j` rather than expanded per-head K/V. The rotary key
remains a separate 64-coordinate record. However, the current checkpoint has a
quantized combined KV-B projection, FP32 accumulation, and an FP16 expanded
cache boundary. Reassociating changes rounding and is not automatically token
identical.

The project must make one explicit choice:

### Path A: preserve V0 finite-precision parity

Keep expanded K/V as the numerical oracle and develop a lossless page codec
whose decode reconstructs the exact FP16 bit patterns. Capacity depends on
measured compressibility and may remain multiple terabytes. This path preserves
the strongest parity claim but may fail the physical-capacity gate.

### Path B: adopt compact MLA as a new canonical graph

Implement absorbed query/content and delayed value projection, compare every
layer against an independent DeepSeek MLA reference, freeze its reduction and
rounding rules, and make that graph the V1 oracle. V1 token parity then means
parity with the compact canonical graph, not bit identity with the expanded V0
implementation. This is likely the only practical 24 GB/SSD capacity path.

The decision cannot be hidden behind an optimization flag. A state header must
identify `expanded-v3` versus `latent-v1`, numerical format, kernel ABI, and
model hash so incompatible continuations are rejected.

## 5. Paged long-context state

### 5.1 Page format

Replace one monolithic state image with fixed-size, independently checksummed
pages. A latent page should contain a contiguous position interval and all
layers needed by the chosen attention traversal, for example:

```text
page header:
  magic, version, model hash, sequence ID
  first position, position count, representation, dtype
  uncompressed bytes, stored bytes, checksum
payload:
  layer-major or position-major latent/rotary records
```

Page geometry is a benchmark variable. Candidate payloads from 2–16 MiB trade
request overhead against cache pollution and cancellation granularity. Pages
must be 16-KiB aligned for the current allocator and laid out sequentially in
the order the attention kernel consumes them.

### 5.2 UMA page cache

Use a fixed unified-memory page pool split into:

- immutable model records;
- recent/hot KV pages;
- attention scan buffers;
- checkpoint write buffers.

The cache key is `(sequence, layer-range, position-page, representation)`.
Readers increment a generation/refcount before a Metal binding; eviction waits
for the consuming command buffer. CPU and GPU share the allocation, but the
release/acquire and command-completion rules remain explicit.

### 5.3 Crash-safe manifest

Current state v3 rewrites the complete prefix. At 140 GB that is unacceptable.
V1 must use:

1. immutable page files or append-only extents;
2. a journal record for newly durable page checksums;
3. a small manifest containing the ordered page set, position, pending token,
   RNG state, model/kernel hashes, and parent state;
4. `fsync` of pages, then journal, then atomic manifest replacement.

Checkpoint cost becomes proportional to newly generated pages, not total
history. Conversation continuation and branching can share immutable prefix
pages copy-on-write. Garbage collection is a separate recoverable operation
that never runs on an active manifest.

## 6. SSD-streamed exact attention

### 6.1 Online softmax without score materialization

For a history tile `b`, compute local maximum `m_b`, local denominator `l_b`,
and local weighted value `o_b`. Combine it with accumulated `(m,l,o)` using

\[
m'=\max(m,m_b),
\]

\[
l'=l e^{m-m'}+l_b e^{m_b-m'},
\]

\[
o'=o e^{m-m'}+o_b e^{m_b-m'}.
\]

The final output is `o/l`. This removes the `[heads,context]` score tensor and
requires one sequential pass over each history tile. The implementation must
freeze tile order and FP32 reduction order and compare logits against the V1
oracle; the identities are exact over real numbers but tiling is observable in
floating point.

### 6.2 Double-buffered attention pages

For each layer or layer group:

```text
read history page j+1 -> scan buffer B
GPU absorbs/scores/accumulates page j -> scan buffer A
join A, swap A/B, continue
```

The two scan buffers are reusable. Large sequential page reads replace millions
of tiny KV reads. Queue depth is tuned so `read(page)` is hidden by
`attention(page)` when possible. Logs report useful KV bytes, physical bytes,
pages, span, page-cache hits, GPU accumulation time, and final reduction time.

### 6.3 Exact dense-attention limit

No general exact algorithm can skip an arbitrary key solely because it is old:
its score depends on the current query. Hierarchical search, top-k attention,
landmarks, sliding windows, and kernel-feature linear attention can reduce work,
but they change the model unless the checkpoint was trained for that structure.

Therefore there are two product levels:

- **exact dense mode:** O(context) history bytes per emitted token, slow but
  complete;
- **accelerated long-context mode:** a model trained for sparse/windowed/
  compressed attention, with its own quality and parity suite.

The architecture must never market the second as an exact optimization of the
first. For one million outputs to be “like butter,” a long-context sparse model
or additional hardware is required; SSD tiling alone makes execution possible,
not fast.

## 7. Million-token input and prefill

### 7.1 Prompt ingestion

A million-token prompt cannot be passed safely as one shell argument. Add:

```text
--prompt-file PATH
--prompt-ids PATH
--raw-prompt-file PATH
```

The tokenizer reads/mmap-windows the file, emits IDs into a chunked spool, and
records a hash and exact token count. Prompt formatting tokens are explicit.
The runtime consumes token-ID pages; it does not retain the original text or a
million-token Python string.

### 7.2 Quantized matrix-matrix fixed projections

Current prefill reuses expert rows across 128 tokens but still dispatches many
fixed projections as grouped GEMVs. V1 needs a model-specific QMM kernel:

1. load one compressed weight tile;
2. decode it once into registers/threadgroup storage;
3. multiply by a tile of prompt activations;
4. accumulate FP32 outputs without writing a dequantized matrix.

Tile dimensions are chosen from measured occupancy, register pressure, and
threadgroup memory on the exact M5. Candidate tile sizes are benchmarked by
stored bytes decoded per activation tile and GPU time, not copied from an
unrelated GPU architecture.

### 7.3 MoE prefill scheduler

For tile `B`, retain exact per-token routing and expert union

\[
U_B=\bigcup_{t\in B}A_t.
\]

Larger `B` improves fixed-weight reuse but pushes `|U_B|` toward 256 and grows
activation scratch. The scheduler chooses `B` from a measured cost model:

\[
t(B)=t_{fixed}(B)+\sum_{e\in U_B}t_e(n_e)+t_{KVwrite}(B)+t_{pressure}(B).
\]

It can bucket multiple tiles concurrently only when their activation/state
buffers are disjoint and their final per-token expert merge remains rank
ordered. The optimal tile may differ by layer because router entropy differs.

### 7.4 Prefill checkpoints

Commit completed token-page/layer frontiers rather than restarting a
million-token prefill. A recovery record must distinguish fully materialized
KV pages from temporary activations. Partially completed layers are discarded;
the last complete layer/tile is replayed.

## 8. Weight-path acceleration

### 8.1 Repack co-required expert records

Create a versioned immutable format containing selected expert gate, up, and
down payloads in one aligned record with a compact index:

```text
(layer,expert) -> shard, offset, gate bytes, up bytes, down bytes, checksum
```

This changes 24 application reads/layer into eight larger reads without
changing useful weights. Records should not be dequantized. Repacking is
offline, deterministic, checksummed against GGUF descriptor bytes, and retains
the source GGUF as the conversion oracle.

### 8.2 Temporal expert cache

Add a bounded, frequency/recency-aware cache only with full accounting:

\[
B_{marginal}=B_{fixed-miss}+B_{expert-miss}+B_{head-miss}.
\]

Per layer log expert IDs, reuse distance, record bytes, hits, misses, eviction,
and bytes saved. Admission score should approximate

\[
score(record)=\frac{P(reuse\ before\ eviction)\,bytes\ saved}
{resident\ bytes+pressure\ penalty}.
\]

The cache budget shrinks as KV grows. It must never rely on macOS compression
to appear larger. The V0 256 MiB fixed cache is the safe seed; the rejected
2 GiB result is the warning that nominal hit rate alone is insufficient.

### 8.3 Multi-NVMe striping

One internal SSD limits aggregate bytes/s. If the target grows to multiple
local NVMe devices, stripe immutable records by layer/expert while keeping each
co-required record contiguous on one device. The scheduler submits independent
reads per device and reports bandwidth separately; simple byte interleaving
that splits every expert across devices increases joins and is not assumed
better.

### 8.4 Output head

Keep the 760 MB head resident while memory permits. At extreme page-cache
pressure, split it into vocabulary tiles and overlap tiled Q6_K logits with
the final layer's release, retaining a global `(logit,id)` reduction with
lower-ID tie-breaking. Streaming the head is a fallback because it adds a
reusable 760 MB read to every token.

## 9. Compute-path acceleration

Prioritize by measured critical-path contribution:

1. fused quant decode + multiply for every stored type;
2. QMM reuse during prefill and multi-sequence decode;
3. one-pass tiled online attention;
4. accepted gate/up/SwiGLU and down/mixture fusions;
5. model-specific M5 Metal Performance Primitives experiments after exact
   repacking into supported tensor/block-scale layouts;
6. only then smaller elementwise fusion.

Each candidate is evaluated by `(token/logit parity, bytes, wall, GPU,
command buffers, dispatches, allocations, pressure)`. The residual/RMSNorm
negative result remains the template: a local kernel speedup that changes an
accepted logit is deleted.

The separate ANE remains outside the hot graph unless a whole subgraph can be
kept in one Core ML execution with lower end-to-end time. Per-layer ANE/GPU
ping-pong is expected to lose to synchronization and shared-memory traffic.

## 10. Concurrency and batching

### 10.1 Multiple sequences

Batching is the most powerful way to amortize weight bytes. At layer `L`, group
ready sequences by selected expert and load each union record once. Fixed
projections become true QMM. KV pages and RNG/pending tokens remain per
sequence. The scheduler maximizes

\[
\frac{useful\ activation\ multiplies}{model\ bytes\ read}
\]

subject to latency and memory budgets.

### 10.2 Shared-prefix conversations

Immutable paged state makes a conversation tree cheap: child sessions point to
the parent's prefix pages and append their own turn. This is exact activation
storage, not a text cache. Page reference counts and model/kernel hashes prevent
use-after-free or cross-model reuse.

### 10.3 Pipeline boundaries

The legal overlap remains:

```text
fixed(L+1) I/O || compute(L)
selected expert I/O || shared expert compute
KV page j+1 I/O || attention page j compute
checkpoint page writes || later compute, after immutable handoff
```

Routers, layer residuals, stable-softmax combines, and per-sequence sampling
remain dependency barriers. Concurrency is introduced with disjoint buffers
and explicit joins, never by racing shared accumulation.

## 11. Metrics and release telemetry

Million-token work must aggregate without producing terabytes of logs. Keep a
compact per-tile/per-N-decode record plus opt-in samples:

| Metric | Required interpretation |
|---|---|
| input/output tokens/s | end-to-end, excluding and including checkpoint time |
| TTFT | prompt open through first emitted byte |
| model useful/physical bytes | selected model payload versus actual reads |
| KV useful/physical bytes | history payload versus page/cache reads |
| NVMe GB/s/device | first-submit to last-completion span |
| GPU kernel time | by QMM, MoE, attention, elementwise, sampling |
| overlap efficiency | `1 - exposed_IO / total_IO_span` |
| cache hit/bytes saved | fixed, expert, KV page, prefix separately |
| command buffers/dispatches/allocations | per token and per prefill tile |
| KV bytes/position | representation invariant |
| peak available-memory delta | plus compression/pageout/swap counters |
| checkpoint write amplification | bytes written / new logical state bytes |
| joules/token | `powermetrics` samples with wall/tokens alignment |
| parity | earliest divergent layer, position, token, and logit |

Histograms record p50/p95/p99 read service, queue depth, page residency, router
reuse distance, and layer time. Raw traces are ring-buffered and flushed only
around a fault or explicit capture.

## 12. Staged delivery plan

### Stage 0: freeze V0 oracle

- Complete and retain the 1,000+1,000 proof.
- Freeze model hash, tokenizer vector, state v3 parser, token/logit anchors,
  quant oracles, and router oracle.
- Add deterministic replay from token IDs so text formatting is outside the
  kernel parity loop.

**Exit:** current proof and tests are reproducible from documented commands.

### Stage 1: prompt-file and paged-state substrate

- Add 64-bit positions/offsets and prompt-ID files.
- Implement append-only page/checksum/manifest state with conversion from v3.
- Retain expanded K/V initially to isolate storage correctness.
- Test interruption at every page/journal/manifest boundary.

**Exit:** 100,000-position synthetic and real state survives forced crashes
without full-prefix rewrites or token drift.

### Stage 2: compact MLA oracle

- Implement latent+rotary state and absorbed/delayed projections.
- Compare each layer and final logits against an independent implementation.
- Decide Path A or Path B explicitly; version the state ABI.
- Measure bytes/position and conversion costs.

**Exit:** 57x state-capacity target is achieved or rejected with evidence; no
silent parity downgrade.

### Stage 3: tiled SSD attention

- Implement online-softmax page combine and double-buffered reads.
- Remove full score materialization.
- Establish fixed tile/reduction ordering and long-context parity anchors.
- Benchmark context lengths geometrically: 2K, 8K, 32K, 128K, 256K, 1M.

**Exit:** attention memory is bounded; bytes and wall scale as predicted; no
non-finite values or unaccounted reads.

### Stage 4: prefill QMM and durable progress

- Add prompt-file token spooling and quantized fixed-projection QMM.
- Auto-tune tile size from expert-union, memory, and GPU metrics.
- Checkpoint complete layer/tile frontiers.

**Exit:** one million token IDs prefill without process-size growth or restart
from zero; output at the model's validated context matches the oracle.

### Stage 5: repacked experts and measured caching

- Build checksummed gate/up/down records.
- Add temporal cache with pressure-aware admission.
- Add multi-sequence layer/expert batching.
- Re-run parity after each scheduling change.

**Exit:** model bytes/output token fall materially, p99 does not regress, and
no swap-out is hidden.

### Stage 6: two-million-position execution proof

- Use a checkpoint whose context/quality contract covers the target, or label
  the run as systems-only.
- Run 1,000,000 input plus 1,000,000 output with periodic manifest commits.
- Verify final position 1,999,999, output ID count, page checksums, no state
  holes, and complete metric summaries.

**Exit:** one reproducible command, one immutable artifact manifest, and one
evidence ledger with capacity, speed, energy, and parity results.

## 13. Scaling model size toward 50 trillion parameters

Parameter count alone is not the runtime capacity equation. For stored average
`q` bits/parameter and `P` parameters,

\[
S_{weights}\ge Pq/8.
\]

At 50 trillion parameters and an ideal one bit/parameter, payload alone is at
least 6.25 TB before block metadata, embeddings, scales, indexes, and KV state.
That does not fit a 1–4 TB internal SSD. Twenty-five independent two-trillion-
parameter models have the same aggregate storage lower bound and additionally
need isolated tokenizers, state, and scheduling metadata.

The scalable architecture is therefore a storage fabric plus conditional
execution contract, not a larger array allocation:

- shard immutable experts across multiple NVMe devices/nodes;
- keep router/fixed spine and indexes near compute;
- route exact selected records to a reusable compute arena;
- batch sequences to amortize each selected record;
- page each model's KV independently and share only hash-identical immutable
  prefixes;
- report active parameter bytes/token, not headline total parameters;
- refuse a model whose stored payload plus recoverable state exceeds physical
  storage.

On the current single M5, such models can be a correctness/slow-execution
research target only if their active bytes/token remain small and their full
payload exists on attached storage. “No capacity bottleneck” is not a software
assumption; every release must publish the byte ledger.

## 14. Final priority order

1. Finish/freeze the exact V0 evidence.
2. Replace full-state rewrites with append-only pages and prompt files.
3. Establish a compact MLA numerical oracle; this is the capacity gate.
4. Stream long-context attention with online softmax; this is the bounded-
   memory gate.
5. Add quantized QMM prefill; this is the million-input TTFT gate.
6. Repack experts and measure temporal reuse; this is the weight-byte gate.
7. Batch sequences and stripe devices; this is the throughput gate.
8. Use a model validated at two million positions; this is the quality gate.
9. Run the exact million+million acceptance and retain its immutable manifest.

The architecture can make a two-million-position run mechanically possible on
storage-constrained hardware. It cannot make dense attention sublinear, make a
163,840-context checkpoint trained for two million tokens, or move petabytes
instantly. Those constraints are the reason this plan begins with math and
versioned numerical contracts rather than a context-size constant.
