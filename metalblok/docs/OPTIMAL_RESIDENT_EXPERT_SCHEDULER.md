# Resident Expert Scheduler

This is the parity-safe algorithm for turning a large UMA buffer into a
deterministic expert working set. It separates three problems that must not be
confused:

1. the native router chooses the experts and therefore owns correctness;
2. the scheduler chooses which bytes to stage and when;
3. the GPU consumes only ready UMA-resident buffers.

The scheduler never changes token IDs, router results, floating-point order, or
the executed expert set.

## 1. State and trace

Let an expert block be (e=(l,i)), where (l) is the layer and (i) is the
expert ID. Its immutable storage size is (s_e) bytes. At decode step (t),
the authoritative router produces the exact set (R_t). For every
(ein R_t), the runtime knows a compute deadline (d_{t,e}): the latest time
by which the bytes must be ready to avoid a consumer stall.

The resident set is (C_t), with capacity

\[
\sum_{e\in C_t}s_e \le M_{UMA}-M_{reserve}.
\]

`M_reserve` is never borrowed by the cache. It protects the OS, KV state,
command allocations, and page-fault headroom.

## 2. Exact offline reference solver

For a recorded route trace, define binary variables:

* (r_{e,t}): block (e) is resident at step (t);
* (p_{e,t}): an SSD read for (e) starts at step (t);
* (v_{e,t}): block (e) is evicted at step (t);
* (z_{e,t}): (e) is ready by its deadline.

The constraints are:

\[
r_{e,t}\le r_{e,t-1}+p_{e,t},
\qquad
r_{e,t-1}\le r_{e,t}+v_{e,t},
\]

\[
\sum_e s_e r_{e,t}\le M_{UMA}-M_{reserve},
\]

\[
\sum_{(e,u):u\in[t-W,t]}s_e p_{e,u}\le R_{SSD}W,
\]

and (z_{e,t}=1) only when the read completion time is no later than
(d_{t,e}). The objective is lexicographic:

\[
\min\left(
  \sum_{t,e\in R_t}(1-z_{e,t})\,L_e,
  \lambda\sum_{t,e}s_ep_{e,t}\mathbf1[e\notin R_{future}],
  \mu\sum_{t,e}s_ev_{e,t}
\right),
\]

where (L_e) is measured miss latency. The first term minimizes GPU stall,
the second false traffic, and the third cache churn. This is the ground-truth
oracle for a trace. It is used offline to evaluate policies, not in the hot
path. With equal-size blocks and no bandwidth constraint, it reduces to
Bélády's farthest-next-use policy; variable sizes and deadlines require the
weighted constrained formulation.

### Falsifiable optimization conjecture

For a fixed route trace, measured block sizes/latencies, UMA capacity, SSD
bandwidth, and deadline sequence, the policy above is the minimum-stall
schedule subject to the residency and bandwidth constraints. This is a
conjecture about the implementation model, not a claim that arbitrary online
expert prediction is optimal. It is tested by solving the binary program and
checking that no feasible schedule has a smaller objective. The LP relaxation
provides a lower bound; the integrality gap is reported explicitly.

The online policy is accepted only with bounded regret against that oracle:

\[
 regret = J_{online}-J_{oracle},
 \qquad
 J=(stall, falseBytes, churn).
\]

The coefficients (lambda,mu,alpha,eta) are fitted from held-out traces
by coordinate search over the measured objective, then frozen. A parameter is
not admitted because it “looks good”; it must improve held-out (J), remain
within the LP lower-bound gap, and preserve the full-logit/checkpoint gate.

## 3. Live online policy

The live scheduler maintains a two-queue FIFO (loading and ready) and uses a
deadline score for each candidate block:

\[
U(e)=\frac{p_e\,L_e\,\max(0,L_e^{miss}-\sigma_e)}{s_e}
       -\alpha\,q_e-\beta\,c_e.
\]

Here:

* (p_e) is calibrated probability of use before eviction;
* (L_e^{miss}) is measured miss latency;
* (sigma_e=d_e-now) is deadline slack;
* (q_e) is false-read risk;
* (c_e) is eviction/churn cost.

The scheduler admits the highest positive (U(e)) per byte while both
conditions hold:

\[
readyBytes \ge H_{high}
\quad\text{or}\quad
\widehat{arrival}(e)\le d_e,
\]

and the sliding SSD budget remains feasible. It stops prefetching at
(H_{high}), resumes below (H_{low}), and never consumes the reserve.

Eviction uses the same score with (p_e) replaced by the measured next-use
hazard. A block needed by the next (K) deadlines is pinned; all other blocks
are evictable. (K) is tuned from the offline oracle, not guessed.

## 4. What prediction is allowed to do

Prediction may issue an earlier read. It may not execute a predicted expert,
modify router weights, substitute an expert, or alter tensor operation order.
The native router revalidates every request. A false prediction costs bytes and
cache space but cannot change output.

This follows the useful part of recent MoE work: prediction is for transfer
overlap while the frozen/native router remains authoritative. SpecPrefetch
uses the same separation and a window-aware scheduler; SP-MoE adds a cutoff
policy and asynchronous batched I/O. [SpecPrefetch](https://arxiv.org/abs/2607.24787),
[SP-MoE](https://arxiv.org/abs/2510.10302). SpecMD reports that ordinary LRU
assumptions can be poor for expert traffic and proposes a least-stale policy;
that is why this design uses measured future-use hazard rather than recency
alone. [SpecMD](https://arxiv.org/abs/2602.03921).

## 5. Acceptance metrics

Every policy is accepted only if it passes all of these gates on the same
state and route trace:

1. full logit hashes and final checkpoint SHA-256 match the reference;
2. zero false execution (predictions can only affect reads);
3. `stall_ms/token`, `useful_bytes/token`, `false_bytes/token`, and cache hit
   rate improve together;
4. no reserve violation, pageout increase, or hot-path allocation;
5. end-to-end wall time improves over the serialized baseline.

The decisive capacity condition is not UMA read bandwidth. It is the unique
miss rate:

\[
R_{unique}\le R_{SSD}.
\]

At the measured 4.035 GB/token and 6.388 GB/s SSD rate, 10 tokens/s requires
at least 84.17% byte residency before the GPU limit is considered. A large
FIFO can hide a temporary deficit, but cannot violate this long-run bound.

## 6. Implementation order

1. Export exact `(layer, expert, size, deadline, ready, hit/miss)` records.
2. Build the offline oracle and replay it against every cache capacity.
3. Add the deadline scheduler with read-only prefetch and watermarks.
4. Compare online decisions to oracle decisions and report regret in bytes and
   stall time.
5. Enable active prefetch only after full-logit and checkpoint parity passes.

This makes the algorithm auditable: every read, eviction, deadline, and speed
claim has a mathematical reason and a trace record.
