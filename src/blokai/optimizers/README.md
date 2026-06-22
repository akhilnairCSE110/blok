# Runtime Optimizer Plan

Runtime optimizers are auxiliary models and predictors used by Blok itself. They are not target
models served directly to the user.

## Requirements To Question

- Does the optimizer improve first-token correctness, steady throughput, latency, or memory use?
- Does it preserve deterministic execution reports?
- Is the overhead visible as model bytes, KV bytes, kernel time, and scheduler time?
- Can a deterministic heuristic beat the learned optimizer for the current milestone?

## Delete / Simplify / Optimize / Automate

- Delete learned optimizer paths until the measured baseline exists.
- Simplify first placement policy to a deterministic heuristic with logged constraints.
- Optimize with auxiliary models only after reports show the heuristic is the bottleneck.
- Automate accepted-token, rollback, and overhead accounting before enabling speculation by
  default.

## Initial Optimizer Classes

- Eagle 3 speculative decoding models;
- predictive memory and placement models;
- scheduler aids for prefetch, batch packing, and KV spill decisions;
- calibration data for sparse layout choices.

## Gate

No optimizer may hide work. Every optimizer must report its input bytes, output decisions, accepted
or rejected predictions, and runtime overhead.

## Sources

- EAGLE-3: https://arxiv.org/abs/2503.01840
- FlexGen: https://arxiv.org/abs/2303.06865
