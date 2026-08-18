# Exact 10-token/s UMA/SSD Decode Algorithm

This document defines the strongest parity-safe route to at least 10 decoded
tokens per second for the three-shard DeepSeek-R1-UD-IQ1_S checkpoint on the
target Apple M5 system. It is a conditional theorem and an engineering
algorithm, not a performance claim. The theorem passes only when the measured
target-run constants satisfy its inequalities.

## 1. Non-negotiable semantics

The native DeepSeek router remains authoritative. For every token and every
MoE layer, it alone determines the executed expert IDs, gate weights, tensor
order, accumulation order, and logits. Prediction may issue a read earlier;
it may not execute a predicted expert, replace a router result, change a
quantization format, or change floating-point reduction order.

Consequently, a prediction miss can cost time and bytes but cannot change the
model result. Acceptance requires identical full-logit hashes and identical
final checkpoint SHA-256 from the same saved state.

This separation is consistent with recent expert-prefetch work: SpecPrefetch
uses a frozen/native router with a transfer-only predictor and a
window-aware scheduler ([paper](https://arxiv.org/abs/2607.24787)); SP-MoE
uses deadline/cutoff scheduling and asynchronous batched I/O
([paper](https://arxiv.org/abs/2510.10302)). SpecMD reports that ordinary LRU
does not model MoE expert locality well and evaluates future-aware policies
([paper](https://arxiv.org/abs/2602.03921)). Those papers motivate the design;
their reported rates are not substituted for measurements of this checkpoint.

## 2. Target constants and measured constants

Every run writes the following constants in its header:

| Symbol | Meaning | Existing measured path | Required target hypothesis |
|---|---|---:|---:|
| (D) | unique expert/model bytes consumed per token | 4.035 GB | 3.000 GB |
| (R) | sustained SSD refill bandwidth | 6.388 GB/s | 16.000 GB/s |
| (C) | cache capacity after reserve | measured per run | 25 GB |
| (G) | complete GPU time per token | 265 ms | at most 100 ms |
| (U) | sustainable UMA read bandwidth | measure | must exceed logical rereads |
| (q) | requested decode rate | — | 10 token/s |

The “required target hypothesis” column is not evidence. The target machine
must measure these values using the actual model, actual tensor shapes, actual
reader queue, and the same memory reserve used for inference.

### Checkpoint-specific object model

This checkpoint has 61 transformer layers, of which 58 are routed MoE layers.
The optimizer operates on the actual manifest blocks rather than abstract
whole experts. Define a block as

\[
e=(l,i,p,b),
\qquad p\in\{gate,up,down\},
\]

where (l) is the layer, (i) the expert ID, (p) the stored projection, and
(b) the storage-aligned tile. The GGUF manifest supplies each block's exact
size (s_e), shard, file offset, quantization type, and reader lane.

For token (t), the native router produces the exact top-(K) set
(TopK_l(t)). The required block set is

\[
R_t=R_t^{fixed}\cup R_t^{KV}\cup
\bigcup_{l\in\mathcal M}\bigcup_{i\in TopK_l(t)}
\bigcup_{p,b}(l,i,p,b).
\]

This union is formed only after native routing. A prediction may stage a block
earlier but may never change (R_t), gate weights, quantization, tensor order,
or floating-point accumulation order.

## 3. Exact queue model

Let token (n) require (d_n) bytes that are not already resident. Let
(p_n) be false-prefetch bytes and (s_n) the bytes completed by SSD reads
before the next deadline. The SSD demand is:

\[
x_n=d_n+p_n.
\]

Let (B_n) be ready bytes in UMA before token (n), and (C) the hard cache
capacity. The exact recurrence is:

\[
B_{n+1}=\min(C,\;B_n+s_n-x_n).
\]

No stall occurs at token (n) if and only if:

\[
B_n+s_n\ge x_n.
\]

For a deterministic route trace, zero-stall execution over tokens
(0,\ldots,N-1) is therefore equivalent to:

\[
B_0+\sum_{i=0}^{k-1}s_i
\ge
\sum_{i=0}^{k-1}x_i
\quad\forall k\le N,
\]

with (0\le B_k\le C). This prefix condition is both necessary and
sufficient; it is stronger than an average-bandwidth argument because it also
catches bursts.

## 4. Long-run throughput theorem

Let (\bar x) be the average SSD bytes required per token. At rate (q), the
SSD demand rate is (q\bar x). A finite buffer can sustain an infinite decode
only if:

\[
q\bar x\le R.
\]

If (q\bar x>R), cumulative deficit diverges and every finite buffer
eventually starves. If (q\bar x<R), the average queue is stable; the prefix
condition determines the required initial headroom.

With byte-weighted residency (h), false bytes (F), and logical bytes (D):

\[
\bar x=D(1-h)+F.
\]

Therefore the storage-feasible rate is:

\[
q_{SSD}=\frac{R}{D(1-h)+F}.
\]

## 5. Compute and UMA theorem

Let (G) include every serialized GPU operation, command submission delay,
and synchronization that cannot overlap with the next token. Then:

\[
q_{GPU}=\frac1G.
\]

Let (W) be logical UMA bytes read per token, including resident rereads. UMA
is not a consumable queue: rereading a resident block does not remove it. It
does impose a bandwidth constraint:

\[
q_{UMA}=\frac{U}{W}.
\]

For a correctly overlapped pipeline, the achievable rate is bounded by:

\[
\boxed{
q_{max}=\min(q_{SSD},q_{GPU},q_{UMA}).
}
\]

The 10-token/s theorem is therefore:

\[
\boxed{
\begin{aligned}
10[D(1-h)+F]&\le R,\\
10G&\le1,\\
10W&\le U,\\
B_0+\sum_{i<k}s_i&\ge\sum_{i<k}x_i\quad\forall k.
\end{aligned}}
\]

All four inequalities must pass on the target run. No buffer size alone can
prove the result.

## 6. Numerical target proof

Under the proposed target constants (D=3) GB/token, (R=16) GB/s, and
(F=0):

### 10 tokens/s

\[
10D=30\text{ GB/s}.
\]

Required byte residency is:

\[
h\ge1-\frac{16}{30}=0.4666667.
\]

Thus at least **46.67% of logical expert bytes** must be supplied by UMA
reuse. This is a byte ratio, not a request-count ratio.

The GPU condition is:

\[
G\le100\text{ ms/token}.
\]

If all misses were unique and the queue began with 25 GB, its finite burst
cover would be:

\[
\frac{25}{30-16}=1.7857\text{ s}.
\]

If reuse makes (D(1-h)<1.6) GB/token, the long-run queue is stable and the
buffer does not drain.

### 7 tokens/s reference

At 7 token/s, SSD demand is (21) GB/s and required residency is:

\[
h\ge1-\frac{16}{21}=0.238095.
\]

The GPU condition is (G\le142.857) ms/token. With no reuse, a 25 GB buffer
would cover (25/(21-16)=5) seconds.

## 7. Offline optimal schedule

For an observed route trace, create one binary variable per block and step:

* (r_{e,t}): block (e) is resident;
* (p_{e,t}): a read starts;
* (v_{e,t}): block (e) is evicted;
* (z_{e,t}): block (e) is ready by its deadline.

Subject to:

\[
r_{e,t}\le r_{e,t-1}+p_{e,t},
\qquad
r_{e,t-1}\le r_{e,t}+v_{e,t},
\]

\[
\sum_e s_er_{e,t}\le C,
\qquad
\sum_{e,u\in[t-L,t]}s_ep_{e,u}\le RL,
\]

and the deadline constraint (z_{e,t}=1) only when the read completion is no
later than the consumer deadline. Solve lexicographically:

\[
J=\left(
\sum_{t,e\in R_t}(1-z_{e,t})L_e,
\lambda\sum_{t,e}s_ep_{e,t}\mathbf1[e\notin R_{future}],
\mu\sum_{t,e}s_ev_{e,t}
\right).
\]

The first term minimizes GPU stalls, the second false traffic, and the third
cache churn. The LP relaxation gives a lower bound; the integer solution is
the trace oracle. The online algorithm reports regret against this oracle.

For implementation, discretize time into measured scheduling quanta and use
binary variables (r_{e,t},p_{e,t},v_{e,t},z_{e,t}). Let (a_{e,t}) and
(c_{e,t}) be read start and completion times, with measured service time
(\ell_e). The following linear constraints enforce completion and readiness:

\[
c_{e,t}\ge a_{e,t}+\ell_e-M(1-p_{e,t}),
\qquad
c_{e,t}\le d_{e,t}+M(1-z_{e,t}).
\]

For each physical shard (j), impose its measured service envelope:

\[
\sum_{e\in shard(j)}s_ep_{e,t}\le R_j\Delta t.
\]

Partition UMA explicitly between fixed tensors, expert blocks, KV pages, and
reserve:

\[
\sum_{e\in fixed}s_er_{e,t}+
\sum_{e\in expert}s_er_{e,t}+
\sum_{e\in KV}s_er_{e,t}
\le M_{UMA}-M_{reserve}.
\]

Use a scalar objective only with dominance constants (A\gg B\gg C>0):

\[
J=A\,stall+B\,falseBytes+C\,churn.
\]

Choose the constants larger than the maximum possible lower-priority term.
This makes the scalar MILP exactly equivalent to lexicographically minimizing
stall, then false traffic, then churn. The LP relaxation is a certified lower
bound; the integer solution is the exact oracle for the recorded route trace.

## 8. Online algorithm

For each candidate block (e), compute:

\[
\mathrm{score}(e)=
\frac{p_e\,\max(0,L_e-\sigma_e)}{s_e}
-\lambda\frac{f_e}{s_e}
-\mu c_e,
\]

where (p_e) is calibrated use probability, (L_e) measured miss latency,
(\sigma_e=d_e-now) deadline slack, (f_e) false-read risk, and (c_e)
eviction cost.

The scheduler:

1. reserves (M_{reserve}) and never allocates into it;
2. pins blocks with a deadline inside the next protected window;
3. admits positive-score blocks while the sliding SSD budget is feasible;
4. fills a loading FIFO and a ready FIFO using independent reader lanes;
5. starts compute only after the exact native route confirms readiness;
6. evicts the lowest future-use score after the protected window;
7. records every byte, deadline, hit, false read, stall, and eviction.

The predictor is a transfer oracle only. It cannot affect logits.

## 9. Exact acceptance protocol

A target run is accepted only when all of the following pass:

1. full-logit hashes equal the serialized reference at every sampled token;
2. final state SHA-256 is identical;
3. (G\le100) ms/token;
4. (D(1-h)+F\le1.6) GB/token at 16 GB/s;
5. the prefix buffer condition passes for the entire run;
6. (10W\le U) passes with measured UMA counters;
7. no reserve violation, pageout increase, or hot-path allocation occurs;
8. end-to-end rate is at least 10.0 tokens/s over a long enough interval to
   exclude startup buffering.

The last requirement must be sustained, not achieved by draining a 25 GB
startup FIFO for a short benchmark.

## 10. What is and is not proven today

The existing repository measurements are not this target proof: the accepted
expanded path measured 4.035 GB/token, 6.388 GB/s SSD, and about 265 ms GPU
time/token. Those values do not satisfy the 10-token/s theorem. The values
(D=3), (R=16), (C=25), and (G\le100) are target hypotheses that must be
measured after implementing this scheduler and the optimized exact kernels.

The proof strategy is intentionally strict: optimize the schedule against the
trace oracle, optimize kernels against (G), then accept only if the four
throughput inequalities and full numerical parity all pass simultaneously.
