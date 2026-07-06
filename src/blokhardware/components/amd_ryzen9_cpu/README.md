# CPU Invariants

- CPU schedules, submits I/O, launches kernels, and serializes results.
- CPU compute is not the inference hot path.
- Core pinning waits for measured stalls.
