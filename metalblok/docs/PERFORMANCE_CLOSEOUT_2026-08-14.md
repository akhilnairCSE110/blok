# Strict M5 performance closeout — 2026-08-14

## Result

This pass improved the exact DeepSeek-R1 671B `UD-IQ1_S` 128-token prefill
probe without changing the greedy token. The accepted command was:

```sh
./run_blok.py --prompt-file /private/tmp/metalblok-proof-128.txt \
  --mla --profile-layers --io-lanes 8 --context 131072 -n 1
```

The saved accepted artifacts are:

- diagnostics: `metalblok/runs/run-20260814-035515-9394.log`;
- model output: `metalblok/runs/run-20260814-035515-9394.txt`;
- state: `/private/tmp/metalblok-20260814-035515-9394.state`.

The model emitted token `33001`, text `Okay`, with winning logit `57.0334`,
runner-up `54.7401`, margin `2.2933`, and zero nonfinite logits. The strict
reference run `run-20260814-025331-6910.log` emitted the same token with
winning logit `57.0330`, runner-up `54.7396`, and margin `2.29336`.

| Metric | strict reference | accepted closeout | delta |
|---|---:|---:|---:|
| prefill tile wall | 37.443902 s | 35.148059 s | -6.13% |
| prompt throughput | 3.418 token/s | 3.642 token/s | +6.55% |
| summed GPU time | 31.401902 s | 29.615172 s | -5.69% |
| exposed I/O wait | 4.620807 s | 4.158632 s | -10.00% |
| dispatches | 130,042 | 92,069 | -29.20% |
| command buffers | 1,529 | 1,529 | unchanged |
| hot allocations | 0 | 0 | unchanged |

This is a real improvement, not the requested order of magnitude. A 10x claim
would be false. The remaining strict wall floor is dominated by streamed
routed experts, scalar K-quant fixed projections, and mandatory SSD traffic.

## Final accepted performance ledger

These are the canonical measurements from
`run-20260814-035515-9394.log`. Decimal GB uses 1,000,000,000 bytes. Times
reported as GPU time are summed Metal command-buffer timestamps; I/O service
time is summed across workers and therefore can exceed wall time.

| Scope | Metric | Final value |
|---|---|---:|
| request | native input tokens | 128 |
| request | emitted tokens | 1 (`Okay`, token 33001) |
| request | decode steps | 0 (the emitted token was the prefill sample) |
| request | configured context capacity | 131,072 positions |
| model | tensors / shards / encoded size | 1,025 / 3 / 140.23 GB |
| model | layers / hidden / vocabulary | 61 / 7,168 / 129,280 |
| model | experts / active experts | 256 / 8 |
| model | MLA latent / RoPE dimensions | 512 / 64 |
| end to end | wrapper wall / TTFT | 36.299 s / 36.299 s |
| end to end | native runtime | 36.068 s |
| startup | build / preflight / tokenize / startup | 0.000 / 0.210 / 0.019 / 0.230 s |
| prefill | reported prefill | 35.15 s, 3.641 token/s |
| prefill tile | wall / throughput | 35.148059 s / 3.642 token/s |
| prefill tile | summed GPU time | 29.615172 s |
| prefill tile | GPU-time-to-wall ratio | 84.258% |
| prefill tile | exposed I/O wait | 4.158632 s (11.832% of tile wall) |
| per input token | wall / GPU / exposed I/O wait | 274.594 / 231.369 / 32.489 ms |
| submission | command buffers / dispatches / hot allocations | 1,529 / 92,069 / 0 |
| per input token | command buffers / dispatches | 11.945 / 719.289 |
| storage | bytes read / reads | 98,477,678,592 / 31,999 |
| storage | useful selected-expert bytes / reads | 91,839,283,200 / 31,695 |
| storage | bytes / useful bytes per input token | 769,356,864 / 717,494,400 |
| storage | effective stream rate | 2.810 GB/s |
| storage | summed worker service / maximum read | 103.209043 s / 54.059 ms |
| storage | peak outstanding reads / reader lanes | 24 / 24 (8 per shard) |
| storage | summed-service concurrency equivalent | 2.936 workers |
| memory | output head / layer slabs / fixed cache | 0.76 / 0.71 / 2.35 GB |
| memory | fixed cache exact payload | 2.338 GB across 309 projections |
| memory | compact MLA latent / RoPE / scores | 8.187 GB / 1.023 GB / 67.11 MB |
| memory | absorbed MLA weights | 0.840 GB |
| memory | expert arena / batch / margin | 90.833 / 192.06 / 33.55 MB |
| memory | fixed set / estimated total | 8.98 / 14.25 GB |
| memory | available / reserve / cache guard | 18.55 / 2.15 / 2.15 GB |
| memory | compact KV bytes per position | 70,272 bytes |
| memory | compact KV at 131,072 positions | 9,210,691,584 bytes (9.211 GB) |
| memory | available-memory endpoint delta | -374,865,920 bytes |
| numerical | winning / runner-up logit | 57.0334 / 54.7401 |
| numerical | margin / nonfinite logits | 2.2933 / 0 |
| parity | strict-reference token | identical: 33001 (`Okay`) |
| parity | winner / runner-up logit delta | +0.0004 / +0.0005 |

The negative available-memory delta means reported available memory increased
by 374.9 MB between tile endpoints. It is not a peak allocation measurement.
The effective stream rate is bytes divided by the measured I/O span, not the
sum of per-worker rates.

### GPU-stage ledger

| GPU component | Calls | GPU time | Share of tile GPU time |
|---|---:|---:|---:|
| attention, including MLA work | 61 | 13.396996 s | 45.24% |
| routed-expert pipeline | 58 | 12.702949 s | 42.89% |
| fixed MoE projections | 58 | 2.344523 s | 7.92% |
| dense FFN layers | 3 | 1.119437 s | 3.78% |
| MoE merge | 58 | 0.023607 s | 0.08% |
| remaining measured GPU work | - | 0.027660 s | 0.09% |

The 58 MoE layers made 59,392 routing selections. Their per-layer expert union
ranged from 148 to 209, averaged 182.155, and totaled 10,565. Routed-pipeline
I/O wait summed to 4.097615 seconds. Layer staging spanned 34.458976 seconds,
but blocked the consumer for only 50.404 milliseconds; its span overlaps the
compute timeline and must not be added to tile wall time.

### SSD-shard ledger

| Shard | Lanes | Bytes | Reads | Useful bytes | Useful reads | Worker service |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 8 | 32.470 GB | 10,180 | 29.607 GB | 9,990 | 33.590 s |
| 1 | 8 | 34.786 GB | 11,468 | 32.700 GB | 11,405 | 36.711 s |
| 2 | 8 | 31.221 GB | 10,351 | 29.532 GB | 10,300 | 32.908 s |

### Metrics not captured by this run

- Decode throughput and steady-state tokens/s were not measured: `-n 1`
  performs prefill and samples one token but executes zero decode steps.
- A 32K or 128K prefill was not run. The 131,072 value is admitted compact-KV
  capacity, not proof of 128K end-to-end wall time.
- Peak resident-memory delta was not sampled; only the endpoint available-
  memory delta above is present.
- Joules/token and average/peak package power were not captured because
  `powermetrics` was not attached.
- Neural Accelerator utilization, GPU occupancy, cache hit rates, register
  spills, stall reasons, hardware cycles, and achieved FLOP/s require a Metal
  performance trace and are intentionally not inferred from software timing.

## Accepted changes

### Exact batched router and top-k

The F32 router previously used 128 separate projection dispatches per sparse
layer, followed by 128 separate grouped top-k dispatches. The accepted path
uses one two-dimensional router dispatch and one 128-threadgroup top-k
dispatch. Every router dot retains the original 32-lane reduction tree; every
top-k threadgroup retains the exact scalar sigmoid, group selection, expert
selection, tie rule, normalization, and scaling order.

Across 58 MoE layers, operation profiling measured:

| Operation | before | after |
|---|---:|---:|
| router GPU | 385.610 ms | 108.018 ms |
| grouped top-k GPU | 620.761 ms | 8.456 ms |
| router dispatches | 7,424 | 58 |
| top-k dispatches | 7,424 | 58 |

### Exact compact-MLA preparation batching

For every layer and 128-token tile, query split/RoPE, Q6 absorbed-query
projection, and compact KV normalization/store formerly issued 384 commands.
They now issue three two-dimensional commands. Each output retains the same
Q6 coefficient order, FP32 lane accumulation, 32-lane reduction, YaRN angle,
and FP16 cache store. Aggregate `mla_prepare` GPU time fell from 2.203 seconds
to about 1.96 seconds. Q6 value reconstruction was tested in the same form but
lost about 18 ms and was reverted.

### Eight measured SSD reader lanes per shard

`--io-lanes {2,4,8}` is a bounded tuning control. Eight is the public-runner
default after the accepted probe; direct native startup uses the same default.
The reader still uses exact-length `pread`, `F_NOCACHE`, disabled readahead,
urgent/background queues, reusable aligned buffers, and fail-closed short-read
handling.

Per-lane telemetry showed all 24 lanes doing balanced work. Shard totals were
32.470, 34.786, and 31.221 GB; individual lanes processed 3.747–4.475 GB.
Maximum service latency was 54.059 ms. The result is evidence for this prompt
and current SSD state, not a universal SSD law; `--io-lanes 4` is the immediate
fallback.

### Deep profiling

`--profile-ops` implies layer profiling and emits GPU/wall/dispatch boundaries
for input norms, every fixed projection, MLA preparation, causal attention,
value reconstruction, routing, top-k, shared experts, routed expert groups,
and residuals. Sparse layer 3 additionally splits every active expert's
gate/up and down work. Each routed group records assignments, unique model
bytes, explicit I/O wait, GPU time, wall time, and dispatches.

`--profile-layers` now also prints one `profile-io` record per shard/lane with
bytes, reads, urgent bytes/reads, summed service time, and maximum read time.
Command-buffer GPU timestamps are real Metal timestamps; hardware cycle,
cache, occupancy, and Neural Accelerator counters require a Metal performance
trace and are not fabricated in the runtime log.

The routed layer-3 split established the next compute target:

| Routed component | GPU time | share |
|---|---:|---:|
| IQ1 gate + up + SwiGLU | 164.500 ms | 78.7% |
| IQ1/IQ2 down | 44.519 ms | 21.3% |

## Rejected or quarantined changes

- Four-token IQ1 dequant reuse was numerically stable but increased aggregate
  routed GPU time from 12.826 to 13.547 seconds. It was removed.
- Token-major exact Q4 batching increased tile wall to 38.879 seconds and
  moved the winning logit to 57.619. It was removed.
- Row-major exact Q4/Q5/Q6 batching raised tile wall to 41.426 seconds because
  it reread weights per token without enough reuse. It was removed.
- M5 MPP TensorOps QMM reduced the tile to about 27.05 seconds, but reassociated
  FP32 reductions moved logits and routed expert unions. It is quarantined
  behind explicit `--tensorops` and is not the strict default.
- Batched Q6 value reconstruction was slightly slower and was removed.

No rejected kernel remains on the default path. Strict token/logit behavior is
the gate, followed by end-to-end wall time—not dispatch count in isolation.

## Remaining bottlenecks

**Next-session priority: maximize steady-state decode tokens/second.** The
accepted closeout run measured prefill only, so the next performance baseline
must execute a sufficiently long multi-token decode, report warmup separately,
and optimize median steady-state output-token latency while preserving greedy
token/logit parity. No decode tokens/second claim is valid until that run is
captured.

On the strict accepted path, routed expert GPU work remains roughly 12.7
seconds per tile. Fixed Q4/Q5/Q6 projections remain scalar-order GEMV and cost
several additional seconds. The accepted run submitted 98.478 GB from SSD,
including 91.839 GB of urgent selected-expert traffic. Those bytes explain why
an order-of-magnitude speedup cannot come from host launch cleanup alone.

The next valid work is narrowly defined:

1. create a deterministic, parity-tested IQ1 gate/up kernel that reuses decode
   work without the register-pressure loss observed by tile 4;
2. make TensorOps QMM deterministic enough to pass layer/logit/routing parity,
   or keep it experimental;
3. reduce selected expert storage bytes through a verified offline record or
   quant layout—not by skipping model-required weights;
4. capture Metal performance traces for NAX utilization, cache bandwidth,
   occupancy, spills, and stalls on the M5 host;
5. validate the eight-lane default over longer prefill and decode runs while
   monitoring VM pressure and joules/token.

## Validation performed

```sh
python3 -m py_compile run_blok.py
cmake --build metalblok/build -j4
ctest --test-dir metalblok/build --output-on-failure
git diff --check
```

The acceptance model run is listed at the start of this document. No 32K or
128K-token prompt was executed during this pass; `--context 131072` proves
admission/allocation for that capacity, not end-to-end 128K prefill.
