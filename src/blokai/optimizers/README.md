# Optimizer Invariants

- No learned optimizer before native decode emits tokens.
- Route-aware batching is allowed only when it reduces expert diversity or improves reuse.
- Predictions must not hide bytes, kernels, or latency.
