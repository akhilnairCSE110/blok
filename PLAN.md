# Full Kimi End-to-End Plan

Done means the pinned Kimi K2.6 checkpoint produces `paris` through the public Python API, all 61 CUDA layers, INT4 experts, and uGDS model/KV I/O on the target machine.

1. Re-audit the surviving Python → CUDA → uGDS path and every generated file boundary.
2. Add deterministic host-side tests for index generation, contract validation, tokenizer formatting, layout parsing, and executor JSON handling.
3. Compile the CUDA executable and uGDS library with warnings as errors on the target toolchain.
4. Materialize and validate the complete pinned 64-shard checkpoint.
5. Generate and verify the physical uGDS extent map and reserved KV range before unmounting.
6. Bind the verified model NVMe controller to uGDS and run the public `Paris` smoke from `blok.generate`.
7. Fix every failure and repeat steps 2–6 until the observed result is `{"status":"ok","text":"paris"}`; record exact evidence below.

## Evidence

- Pending target-hardware run.
