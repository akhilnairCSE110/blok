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
