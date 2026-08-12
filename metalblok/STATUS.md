# MetalBlok status

**Active release gate:** exact 1,000-native-token coding prompt plus exactly
1,000 emitted tokens on the three-shard DeepSeek-R1 checkpoint.

| Gate | Status | Evidence |
|---|---|---|
| Native C++/Metal build | Pass | warning-clean build; runtime Metal source compiles |
| Shard/model admission | Pass | 3 physically resident shards, 1,025 tensors, 140,231,438,464 bytes |
| Stored quant kernels | Pass | F32, Q4_K, Q5_K, Q6_K, IQ2_XXS, IQ1_S CPU/Metal checks |
| Exact grouped router | Pass | CPU/Metal IDs identical; weights within declared FP tolerance |
| Complete 61-layer generation | Pass | coherent real output and exact atomic continuation |
| Exact 1,000-token prefill | Pass | 591.94 s runtime, 1.69 token/s, first token `Okay` |
| Scheduling optimization parity | Pass | positions 1,280–1,282 token IDs and greedy logits match old path |
| Fixed-weight traffic reduction | Pass | 13.850 → 13.588 GB and 1,939 → 1,869 reads/decode step |
| 1,000 emitted-token acceptance | Running | authoritative checkpoint must finish at position 1,999 |

Current steady decode is approximately 3.1–3.4 seconds/token at 4.1–4.6 GB/s
effective NVMe on the 24 GB M5, with about 0.47–0.57 seconds GPU time,
178 command buffers, and zero hot-path allocations. The long continuation has
also exposed real system pressure: 116 system-wide swap-outs across five
positions as of position 1,848. Correctness continued, but late-run sustained
decode is closer to 4.2–4.5 seconds/token at 3.1–3.4 GB/s. These VM counters
include the whole host, not just MetalBlok.

The exact live artifacts and complete implementation record are in
[`docs/V0_CLOSEOUT.md`](docs/V0_CLOSEOUT.md). CLI usage is in
[`docs/RUN_GUIDE.md`](docs/RUN_GUIDE.md).
