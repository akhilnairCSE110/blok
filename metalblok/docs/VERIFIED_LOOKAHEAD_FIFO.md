# Verified Lookahead Tensor Residency

**Model:** DeepSeek-R1 671B `UD-IQ1_S`  
**Machine:** target M5 Mac, unified memory plus local NVMe  
**Correctness rule:** prediction may change residency, never the forward graph

## 1. Objective

The runtime should make router-selected expert weights appear to the GPU as
resident unified-memory tensors while NVMe replenishes a bounded staging pool
in the background. The mechanism is a cache/FIFO hybrid: FIFO scheduling hides
read latency, while reuse is the only mechanism that reduces sustained NVMe
bytes. Every cache lookup is verified against the exact router result.

For layer \(\ell\) and output position \(t\), the authoritative selected set is

\[
S_{\ell,t}=\operatorname{Top8GroupedSigmoid}
             (W^R_\ell\,\operatorname{RMSNorm}(h_{\ell,t})+b_\ell).
\]

The hidden state \(h_{\ell,t}\) depends on all preceding layers. Consequently,
\(S_{\ell,t}\) cannot be known exactly before that dependency chain executes.
A lookahead policy may predict a set \(P_{\ell,t}\), but only
\(S_{\ell,t}\) determines computation.

## 2. Exactness invariant

Let \(C_{\ell,t}\) be the expert tensors resident for layer \(\ell\) when its
router completes. The verified hit and miss sets are

\[
H_{\ell,t}=S_{\ell,t}\cap C_{\ell,t},\qquad
M_{\ell,t}=S_{\ell,t}\setminus C_{\ell,t}.
\]

Experts in \(H\) execute from unified memory. Every expert in \(M\) is read
from its original GGUF offset and must complete before its exact gate/up/down
operations execute. Predicted experts outside \(S\) contribute nothing. Thus
residency cannot alter expert IDs, router coefficients, reduction order, or
logits.

The implementation obtains lookahead without speculative I/O first: tensors
read for \(S_{\ell,t}\) are admitted directly into the layer's resident cache.
They are therefore ready predictions for \(t+1\) at zero additional SSD cost.
Capacity beyond eight entries retains older routes under LRU replacement.
Pinned experts selected by the current token cannot be evicted until their GPU
work is complete.

## 3. Queue and bandwidth model

Let

- \(D=4.035\) GB be the measured uncached selected-expert bytes per token;
- \(R_s=6.388\) GB/s be measured sustained NVMe throughput;
- \(h\) be the byte-weighted verified cache-hit fraction;
- \(T_c=0.265\) s be the measured GPU time per steady token.

Unique SSD demand is

\[
D_s=(1-h)D.
\]

With perfect compute/I/O overlap, the lower bound is

\[
T_{token}\ge\max\left(T_c,\frac{D_s}{R_s}\right).
\]

The current path becomes compute-bound only when

\[
h\ge 1-\frac{R_sT_c}{D}=58.1\%.
\]

The I/O side of five tokens/s requires

\[
h\ge 1-\frac{R_s(0.2)}{D}=68.3\%.
\]

The compute side separately requires (T_c\le0.2\) s. At the measured 265 ms
GPU time, even a 100% expert-cache hit rate has a 3.77 token/s roof.

To make 60 GB/s of useful tensor consumption appear behind a 6.388 GB/s SSD,
at least

\[
1-\frac{6.388}{60}=89.4\%
\]

of useful bytes must be cache hits. The apparent bandwidth then comes from
reuse, not from relabeling preloaded SSD time.

For a pure FIFO with occupancy \(Q\), capacity \(C\), and unique consumption
rate \(R_d\),

\[
Q(t+\Delta t)=\min(C,Q(t)+R_s\Delta t)-R_d\Delta t.
\]

A finite FIFO eliminates stalls indefinitely only if \(R_d\le R_s\). When
\(R_d>R_s\), capacity controls burst duration but not steady throughput. The
resident history changes this equation by eliminating repeated source bytes.

## 4. Concrete runtime policy

1. All deterministic fixed projections remain resident. Their reuse rate is
   100%, so evicting them for speculative experts would increase traffic.
2. Each sparse layer owns a bounded exact expert-history cache. Four entries
   remain the measured default; `--expert-cache-ways 0..32` controls capacity.
3. Requested capacity is clamped to available memory while preserving a
   6.442 GB safety reserve. KV growth can evict the expert tier.
4. Each exact cache hit updates its LRU timestamp. A miss replaces the oldest
   non-pinned entry or uses the ordinary staging arena if all entries are
   pinned.
5. Gate, up, and down bytes are loaded directly into the chosen resident slot.
   There is no intermediate copy and no duplicate prefetch.
6. Layer profiling logs all eight authoritative expert IDs, cache hits/misses,
   bytes avoided, SSD bytes, I/O span, GPU time, and wall time.

This policy uses the memory pool as far-future history without allowing an
unverified prediction to affect the forward pass. A future active prefetcher
is admissible only if traces demonstrate that its precision saves more bytes
than its false predictions read.

`--profile-predictor` performs that test without issuing speculative reads. It
applies each target layer's actual resident DeepSeek router to earlier residual
states, then compares horizons 0 through `--predictor-depth - 1` with the
authoritative post-attention route. The default depth is four. The log records
exact top-8 overlap, predictor wall/GPU cost, time to router verification, and
time to the first routed-expert use for every horizon. It deliberately perturbs
timing and is not a throughput benchmark. Active speculative I/O remains
disabled until a horizon demonstrates both adequate recall and enough deadline
slack.

## 5. Predictor objective

For predicted set \(P\), exact routed set \(S\), and tensor byte size \(b(e)\),
the primary metric is verified byte recall:

\[
R_b=\frac{\sum_{e\in P\cap S}b(e)}{\sum_{e\in S}b(e)}.
\]

Prediction precision prevents an apparently successful predictor from wasting
the SSD on false positives:

\[
P_b=\frac{\sum_{e\in P\cap S}b(e)}{\sum_{e\in P}b(e)}.
\]

The runtime must also measure:

- useful resident bytes and compulsory SSD bytes per token;
- false-prefetch bytes per token;
- prediction lead time, from read completion to first GPU use;
- per-layer reuse-distance distribution and cache eviction count;
- cache occupancy and the number of bytes evicted unused;
- exposed GPU starvation time after all legal overlap;
- end-to-end, p50, and p95 decode latency;
- token IDs and logits against the strict reference.

The optimization objective is lexicographic: exact token/logit parity first,
then minimum end-to-end wall time. Subject to parity, a useful analytical
score is

\[
J = T_c + T_{starve}
    + \lambda\frac{B_{false}}{R_s}
    + \mu\,\text{pageout\_time},
\]

where \(T_c\) is compute time, \(T_{starve}\) is exposed read wait, and
\(B_{false}\) is prediction traffic never used. The coefficients are measured
time conversions, not arbitrary model-quality weights.

### Branch-predictor and hardware-prefetch discipline

Expert lookahead is structurally closer to a branch predictor plus cache
prefetcher than to text generation. The prediction output is a set of eight
layer-local expert IDs; the authoritative router is the resolution stage; an
incorrect prediction incurs bandwidth/cache-pollution cost but is squashed
before compute. The core metrics are coverage/recall, accuracy, timeliness, and
capacity. That matches the hardware-prefetch analysis in
[Classifying Memory Access Patterns for Prefetching](https://people.ucsc.edu/~hlitz/papers/asplos2020.pdf),
which also explains why low accuracy can slow a system through bandwidth and
cache pollution.

The shipped history tuner is a confidence-free baseline: it exhaustively
searches decay and previous-route-bonus parameters on the first 60% of a route
trace, then reports the frozen candidate on the held-out tail. Its 28.00%
held-out top-8 recall is a rejection, not a reason to prefetch. A future learned
history predictor may use a small per-layer perceptron, inspired by
[Jiménez and Lin's neural branch predictor](https://www.cs.utexas.edu/~lin/papers/hpca01.pdf),
but only after a long trace supports a real train/validation/test split.

The first state-conditioned candidate is cheaper and more model-aware than a
second LLM: apply each future layer's own resident router to the current
residual. `--profile-predictor` measures that exact cross-layer candidate now.
Adding a separate LLM would introduce another weight stream and recurrence
chain, so it is dominated unless measured prediction value exceeds its compute
and traffic cost.

### What is and is not adopted from learned speculation

The useful lesson from
[EAGLE-3](https://arxiv.org/abs/2503.01840) is that a prediction head can gain
signal by fusing lower-, middle-, and upper-level model features, and that the
proposal must be judged by end-to-end accepted work rather than predictor
accuracy alone. EAGLE-3 itself predicts tokens and verifies a speculative token
tree; that is a different dependency graph from selecting eight layer-local
experts. Its draft model and tree verifier are therefore not part of V0.

The directly applicable design is
[SpecPrefetch's transfer/execution separation](https://arxiv.org/abs/2607.24787):
a learned candidate may schedule an asynchronous read, but only the frozen
native router may select an expert for multiplication. A prediction error can
consume bandwidth or pollute the cache; it cannot change the forward pass.

[Pre-Attention Expert Prediction](https://arxiv.org/abs/2511.10676) reports
93.03% top-k accuracy on DeepSeek-V2-Lite using a trained two-linear-layer,
ranking-loss head over same-layer pre-attention state. That number does not
transfer automatically to this DeepSeek-R1 quantization, prompt distribution,
or M5 storage path. The current probe tests the zero-new-weight candidate first:
DeepSeek's own router applied to the pre-attention residual. A learned head is
admissible only after captured state/route pairs are divided by conversation
into train, validation, and held-out test sets and it passes all four gates:

1. higher held-out byte recall than the model-native probe;
2. positive net traffic after false reads and cache pollution;
3. read completion before first expert use, not merely route accuracy;
4. lower full-decode wall with matching full-logit fingerprints.

This keeps the smart part of the papers—the richer state signal and
transfer-only verification boundary—without importing speculative machinery
that has not paid for itself on this exact trace and hardware.

The broader research audit is intentionally selective:

| work | useful mechanism | decision for exact R1 V0 |
|---|---|---|
| [FATE](https://arxiv.org/abs/2502.12224) | apply adjacent-layer state to future-layer gates for earlier expert knowledge | adopted as the horizon-indexed, profiling-only model-native router sweep |
| [ExpertFlow](https://arxiv.org/abs/2410.17954) | predict routing paths, then correct cache state when real routing resolves | adopt the correction boundary; do not copy its throughput claims across hardware/models |
| [DuoServe-MoE](https://arxiv.org/abs/2509.07379) | specialize prefill and decode policies; train a decode predictor from traces | phase split adopted; learned predictor remains gated on this model's held-out traces |
| [MoE-Infinity](https://arxiv.org/abs/2401.14361) | prompt/trace-conditioned sparsity cache for batch-one decode | exact LRU/Belady/history replay implemented; measured history predictors rejected |
| [Pre-gated MoE](https://arxiv.org/abs/2308.12066) | move expert selection into a prior block so transfer and compute decouple | not adopted: it changes and fine-tunes the model architecture |
| [SiDA-MoE](https://proceedings.mlsys.org/paper_files/paper/2024/file/698cfaf72a208aef2e78bcac55b74328-Paper-Conference.pdf) | offline LSTM/hash predictor in a parallel thread | not adopted: it replaces routing and reports nonzero quality loss |
| [HOBBIT](https://arxiv.org/abs/2411.01433) | low-precision miss experts plus multilevel caching/prefetch | caching hierarchy is relevant; precision substitution is disallowed |

Reported paper speedups are not treated as expected results. Most compare
against different CPU-to-discrete-GPU baselines, use smaller models, tolerate
retraining or precision changes, or measure throughput under batching. The
only acceptance evidence here is a same-checkpoint M5 run with full-logit
fingerprints and measured wall/bytes/deadlines.

Active admission will require a confidence threshold \(\tau\) chosen by trace
search. For candidate \(e\), prefetch only when

\[
p_eT_{saved} > (1-p_e)\left(\frac{b_e}{R_s}+T_{pollution}\right),
\]

and the read fits the HLS bandwidth/capacity schedule. Confidence is updated
only after the authoritative router resolves, analogous to a branch-predictor
update. This prevents low-confidence speculation from consuming the resource
needed by exact misses.

## 6. Acceptance experiment

Clone one checkpoint so every run starts at the identical state and position:

```sh
cp STATE /tmp/blok-cache0.state
cp STATE /tmp/blok-cache4.state
cp STATE /tmp/blok-cache32.state

METALBLOK_EXPERT_CACHE_WAYS=0 ./run_blok.py ignored --state /tmp/blok-cache0.state \
  --continue-decode --mla --profile-layers -n 32

./run_blok.py ignored --state /tmp/blok-cache4.state --continue-decode --mla \
  --expert-cache-ways 4 --profile-layers -n 32

./run_blok.py ignored --state /tmp/blok-cache32.state --continue-decode --mla \
  --expert-cache-ways 32 --profile-layers -n 32
```

The final run automatically selects the largest capacity that fits its safety
reserve. Promotion requires identical token IDs and logged logits, lower
end-to-end decode wall time, fewer SSD bytes/token, no swap, and no material
increase in pageouts or compression. Cache-hit percentage alone is not an
acceptance criterion.

One profiled run now records all authoritative routes. Replay it through every
candidate history capacity without rerunning the model:

```sh
python3 scripts/analyze_expert_routes.py metalblok/runs/run-....log
```

The reported floor uses the measured 4.035 GB/token, 6.388 GB/s NVMe, and
265 ms GPU time. It is a scheduling bound; the final capacity must still win a
real parity-gated end-to-end run.

## 7. Formal bounds and hardware-style scheduling

### Exactness theorem

Assume a resident cache entry is populated by reading the exact GGUF byte
interval for expert \(e\), and the authoritative router selects \(S\). For
each \(e\in S\), execution reads either that byte-identical resident interval
or waits for the same interval from SSD. Experts outside \(S\) are never bound
to a compute dispatch. Therefore every kernel receives the same weight bytes,
activation bytes, router coefficient, and rank order as the uncached graph.
By induction over expert rank and transformer layer, every activation and final
logit is identical. Predictor failure can increase waiting or unused residency;
it cannot alter the result.

### Compulsory-miss theorem

For a route trace containing \(N\) selections and \(U\) distinct
`(layer, expert)` pairs, any cache that admits a tensor only after its first
exact use incurs at least \(U\) misses:

\[
h\le 1-\frac{U}{N}.
\]

The existing 23-position trace contains \(N=10{,}672\) selections and
\(U=4{,}769\) distinct pairs. Its passive-cache ceiling is therefore 55.31%,
regardless of capacity. With 32 entries per layer, Belady's future-aware
optimal eviction reaches 55.18%; measured LRU replay reaches 46.23%. This
quantifies both the compulsory traffic and the learnable eviction gap.

Active prediction can move a compulsory read before its deadline but cannot
remove its SSD bytes. Its benefit is bounded by available SSD slack and lead
time; false predictions consume the same constrained resource.

### Resource-constrained formulation

This is the same class of scheduling problem used in high-level synthesis.
For candidate tensor \(j\), define binary resident state \(x_{j,t}\), read
decision \(r_{j,t}\), verified use \(s_{j,t}\), byte size \(b_j\), and deadline
\(d_{j,t}\). A time-slotted offline oracle obeys

\[
\sum_j b_jx_{j,t}\le C,
\qquad
\sum_j b_jr_{j,t}\le R_s\Delta t,
\]

\[
x_{j,t+1}\le x_{j,t}+r_{j,t},
\qquad
s_{j,t}\le x_{j,d_{j,t}}+r_{j,d_{j,t}}.
\]

The lexicographic objective is zero correctness violations, then minimum
deadline stall, false-read bytes, and eviction cost. This can be encoded as
CP-SAT/SMT for offline proofs. The implemented cache subproblem does not need
a general solver: Belady's farthest-next-use rule is provably optimal when the
future trace is known. Runtime uses bounded LRU; a future probabilistic
lookahead scheduler should use earliest deadline first, with admission ordered
by expected saved stall per byte, and compare itself to the Belady bound.

SAT does not make future routes observable. Its role is proving schedule
feasibility and producing an oracle against which the online predictor is
measured.

### Chip-synthesis algorithms adopted

The runtime is treated as a timed dataflow graph (G=(V,E)), not as a list of
ad-hoc threads. A node is a fixed-weight read, norm, projection, attention,
router, expert read, shared-expert compute, routed compute, or residual. An
edge is a true data or buffer-lifetime dependency. For start time (t_v) and
measured latency (L_v), every edge (u\rightarrow v) imposes

\[
t_v\ge t_u+L_u.
\]

Each resource (r)—NVMe lane/bandwidth, resident bytes, Metal queue, or GPU
execution—also has a cumulative capacity constraint. This is the same
schedule/bind split used in high-level synthesis: first choose the cycle for
each operation, then bind operations and storage to finite resources. AMD's
[HLS scheduling and binding guide](https://docs.amd.com/r/2024.1-English/ug1399-vitis-hls/Understanding-High-Level-Synthesis-Scheduling-and-Binding)
describes those phases and the resource-sharing constraint directly.

For a repeated decode pipeline, the lower-bound initiation interval is

\[
II_{min}=\max(ResMII,RecMII),
\]

where `ResMII` is the busiest resource demand divided by its capacity and
`RecMII` is the longest loop-carried dependency bound. LLVM's production
[Swing Modulo Scheduler](https://llvm.org/doxygen/MachinePipeliner_8h_source.html)
uses precisely this split before constructing a lifetime-sensitive software
pipeline. In MetalBlok, the SSD resource bound is (D_s/R_s), the GPU resource
bound is measured GPU time, and autoregressive token dependence supplies the
recurrence bound. This prevents a buffer from being credited with an
impossible steady-state speedup.

The concrete mappings are:

- **ASAP/ALAP and critical-path scheduling:** fixed layer (L+1) reads issue
  as soon as their staging slot is free; exact routed reads issue immediately
  after the authoritative router. Slack is the measured deadline minus read
  service time.
- **Modulo scheduling:** the two layer slabs are rotating pipeline registers.
  Read (L+1), compute (L), and previously submitted I/O overlap subject to
  recurrence and capacity constraints.
- **Retiming:** a deterministic fixed read may move earlier across a compute
  boundary without changing the graph's values, analogous to moving pipeline
  registers while preserving circuit behavior. The formal basis is
  [Leiserson and Saxe's retiming transformation](https://www.cs.columbia.edu/~CS6861/handouts/leiserson-algorithmica-88.pdf).
  A routed expert read cannot be retimed before its router unless it is marked
  speculative and verified later.
- **Storage binding / lifetime coloring:** staging buffers are assigned by
  non-overlapping live intervals. The double buffer is the two-color solution
  for the current layer/next layer lifetime pattern; resident expert capacity
  is a byte-capacitated extension of the same problem.
- **Optimal replacement oracle:** the trace analyzer uses Belady/MIN, evicting
  the entry whose next use is farthest away. It is an exact offline bound, not
  a production predictor. Runtime LRU is compared to it to measure the
  learnable eviction gap.
- **CP-SAT schedule oracle:** optional read intervals, precedence, `NoOverlap`
  per I/O lane, and cumulative resident-byte constraints match the official
  [OR-Tools scheduling model](https://github.com/google/or-tools/blob/stable/ortools/sat/docs/scheduling.md).
  CP-SAT belongs offline on captured traces; putting a combinatorial solver in
  the decode loop would consume the latency it is meant to save.
- **Equivalence checking:** the strict graph and optimized graph form a miter.
  Weight intervals, selected IDs, router weights, kernel reduction order,
  layer fingerprints, logits, and tokens must match. Algebraic rewrites are
  admitted only under their actual floating-point semantics. In particular,
  real-number associativity does not prove bitwise FP32 equivalence. E-graph
  techniques such as
  [HEC](https://arxiv.org/abs/2506.02290) are useful for enumerating and
  checking transforms, but the target-run parity gate remains authoritative.

What ships by default is deliberately smaller than the complete solver model:
a dependency-correct double-buffer schedule, bounded exact cache, LRU online
policy, Belady offline oracle, resource lower bounds, and a state-conditioned
predictor probe. Active speculative reads were removed after the rank-one A/B
failed the full-logit and checkpoint gate. The authoritative router resolves
every multiplication; the profiler issues no model reads.

The first target predictor trace was run on the M5 over 406 sparse-layer routes
and 3,248 exact expert selections. It establishes the accuracy/lead curve:

| lookahead layers | top-8 recall | rank-1 precision | mean first-use lead |
|---:|---:|---:|---:|
| 0 | 81.19% | 98.77% | 4.003 ms |
| 1 | 68.17% | 95.81% | 16.819 ms |
| 2 | 59.88% | 94.33% | 29.538 ms |
| 3 | 53.76% | 89.66% | 42.152 ms |

This rejects all-eight prefetch and justified a rank-one A/B experiment.
At horizon one, rank one moves 0.483 GB/token of useful reads earlier, adds
0.021 GB/token of false reads, and has at most 75.7 ms/token of hideable I/O at
6.388 GB/s. Prefetch does not erase compulsory traffic: total physical bytes
remain exact bytes plus false bytes. The resulting background-read candidate
improved raw rate from 1.429 to 1.461 step/s, but its full-logit hashes diverged
after 25 matched steps and its checkpoint differed. It was rejected and
deleted. The completed predictor trace is
`metalblok/runs/run-20260814-165304-4377.log`; the rejected final A/B is
`run-20260817-220420-3862.log` versus `run-20260817-221931-4952.log`.

`scripts/synthesize_decode_config.py` is the synthesis report. It accepts a
strict reference followed by candidate logs, rejects candidates whose
overlapping full-logit hashes differ, and computes the non-dominated frontier
over whole-decode wall time, NVMe bytes/token, resident cache GB, and GPU time.
Its report also retains p50/p95 latency, end-to-end wall, command buffers,
dispatches, allocations, and pageouts. The fastest candidate within an
explicit cache budget is selected only from the full-hash parity set. This is the
runtime analogue of timing/area/power exploration; SSD bytes are a labeled
traffic/energy proxy until `powermetrics` provides actual joules.

`scripts/tune_decode.py` automates design-space exploration. It APFS-clones one
golden checkpoint for every `(cache ways, expert group size)` point, runs each
point from the identical position, and feeds the resulting logs to the strict
synthesis report. The default set is intentionally small:
`0:4,4:1,4:2,4:4,4:8,8:4`. It isolates the cache optimum already observed and
the command/I/O initiation-interval parameter without launching a wasteful
Cartesian sweep.
