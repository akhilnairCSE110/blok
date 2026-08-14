# MetalBlok status

**Release gate completed:** exact 1,000-native-token coding prompt plus exactly
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
| 1,000 emitted-token acceptance | **Pass** | 1,000 IDs reconstructed; final state position 1,999, 8,007,002,532 bytes |

The final optimized 719-step segment averaged 4.179 s/step, 0.578 s GPU time,
and 3.408 GB/s aggregate effective NVMe, with 178 command buffers and zero
hot-path allocations per step. The best early interval was 3.09–3.22 s/step;
late expanded-KV pressure lowered sustained performance. Host-wide VM counters
over the segment recorded 2,407 swap-ins and 168 swap-outs.

The exact artifacts and complete implementation record are in
[`docs/V0_CLOSEOUT.md`](docs/V0_CLOSEOUT.md). CLI usage is in
[`docs/RUN_GUIDE.md`](docs/RUN_GUIDE.md).

The full prompt, decoded output, segment arithmetic, metrics, quality judgment,
and limitations are in [`docs/PROOF_1K_REPORT.md`](docs/PROOF_1K_REPORT.md).

The 2026-08-14 strict M5 pass preserved token `33001` and reference-scale
logits while reducing the 128-token/128K-capacity probe from 37.444 to 35.148
seconds (3.418 to 3.642 token/s). It cut dispatches by 29.2% and exposed I/O
wait by 10.0%. This is a measured 6.1% wall-time improvement, not a 10x claim.
Exact accepted/rejected evidence is in
[`docs/PERFORMANCE_CLOSEOUT_2026-08-14.md`](docs/PERFORMANCE_CLOSEOUT_2026-08-14.md).
