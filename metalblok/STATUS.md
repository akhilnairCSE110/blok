# MetalBlok status

**Documentation truth date:** 2026-08-17. The accepted release remains the
expanded-KV 1,000+1,000 V0. Later compact-MLA measurements are a separate,
explicitly opt-in track and do not retroactively change that proof.

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

## Current compact-decode status

`--mla` stores 70,272 bytes/position rather than 4,005,504 and admits a
131,072-position capacity with about 9.211 GB of compact KV. A 31-step decode
from a saved compact state measured 1.344 step/s without expert history and
1.374 step/s with the guarded four-way per-layer cache. That 2.2% decode-rate
gain is real but not a 10x claim. The 128-token capacity probe emitted the same
token and close reference-scale logits; its full vector was not bitwise equal
to expanded mode. No full 32K or 128K prompt has been executed.

The state-conditioned router probe measured high rank-one precision. A final
rank-one background-read trial reduced mean I/O wait from 292.7 to 240.2 ms
and raw decode wall from 20.990 to 20.540 seconds (1.429 to 1.461 step/s), but
full logits diverged after 25 matched steps and the final checkpoints differed.
The active prefetch implementation and CLI were removed; only the read-free
`--profile-predictor` probe and offline trace analyzer remain.

The numerical trace localized the separate compact-path repeatability issue.
Two cache-on runs entered position 34/layer 38 with identical layer-37 hidden
state, then produced different pre-FFN vectors in layer 38 attention. Earlier
traces first diverged at other layers only after context exceeded 32. Resource
barriers and same-command-buffer pass boundaries did not eliminate it and were
removed. Two fully serialized cache-off trace runs did match all 1,891 layer
hashes, all 30 logit hashes, and final checkpoint SHA-256. Compact mode stays
opt-in until this cross-SIMD attention boundary is resolved.

An expanded-path 1,000-output attempt on 2026-08-17 ended without a completion
record at position 39 while moving 13.588 GB/token and accumulating pageout,
compression, and swap activity. It is not an acceptance run.

`--tensorops` is likewise quarantined because faster QMM prefill changed
routing/logits. The strict rule remains: no speed result is accepted unless
the comparison starts from the same state and passes the full-logit gate.
