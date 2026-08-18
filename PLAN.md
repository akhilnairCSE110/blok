# Active V0 closeout record

The active release is MetalBlok: native DeepSeek-R1 671B `UD-IQ1_S` inference
on the 24 GB Apple M5. Kimi/CUDA/uGDS remains separate legacy/future research.

The four V0 gates below are complete:

1. Build and run the complete 61-layer graph with exact tokenizer, grouped
   top-8 routing, quantized projections, causal attention, and decoding.
2. Preserve finite-precision token/logit parity while reducing model bytes,
   I/O stalls, command boundaries, allocations, and unsafe memory pressure.
3. Prove exactly 1,000 native input tokens plus 1,000 emitted tokens at context
   2,048, finishing the authoritative state at position 1,999.
4. Retain the state, output, diagnostic log, measured metrics, reproducible CLI,
   and complete architecture/evidence record.

Current status and artifacts: [MetalBlok status](metalblok/STATUS.md) and
[V0 closeout](metalblok/docs/V0_CLOSEOUT.md). Work beyond V0 is specified in
the [million-token scale plan](metalblok/docs/MILLION_TOKEN_SCALE_PLAN.md).

Post-V0 work is not silently promoted. Compact MLA, TensorOps, predictor
research, parallel expert kernels, larger contexts, and multi-request serving
each require their own numerical and performance gate. Active predictive
prefetch was removed after failing the full-logit/checkpoint gate. The authoritative
scope and reading order are in the [documentation map](docs/README.md).
