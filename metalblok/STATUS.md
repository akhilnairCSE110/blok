# MetalBlok status

Completion requires a real prompt to generate repeatable text from the exact
three-shard DeepSeek-R1 checkpoint without crossing the memory ledger.

| Gate | Status | Evidence |
|---|---|---|
| Native build and runtime Metal compilation | Pass | Native binary built; executing Metal kernels used in inference |
| Non-hydrating shard preflight | Pass | Exact manifest, allocation, sparse, and dataless checks |
| All three model shards resident and verified | Pass | 140,231,438,464 logical bytes; `missing_physical=0` |
| Six real GGUF quant types pass CPU/Metal parity | Pass | F32, Q4_K, Q5_K, Q6_K, IQ2_XXS, IQ1_S; relative errors 2.04–2.13e-4 |
| Exact grouped DeepSeek router passes parity | Pass | Same eight IDs; maximum displayed weight difference 3.0e-8 |
| One real generated token | Pass | Prompt `Hi` predicts `Okay` through all 61 layers |
| Five repeatable coherent tokens | Pass | Exact checkpoint chain begins `Okay, so I need...` |
| 32-token safety soak | Pass | Position 6 to 38; atomic +1 advancement; 1.08–1.77 GB final RSS |

The executability milestone is complete. Full independent-runtime logit parity,
physical expert-record bundling, temporal caching, optimized scheduling, and
contexts above 64 remain explicitly outside this milestone.
