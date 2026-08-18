# Blok documentation map

This page is the authority map for every Markdown document in the Blok
repository. It prevents historical Linux research, the accepted Apple V0,
later compact-MLA work, and future scale plans from being read as one undated
implementation claim.

## Status vocabulary

| Word | Required evidence |
|---|---|
| **Accepted** | Named model and state, completed command, retained log, numerical gate, safety gate, and measured wall time. |
| **Implemented** | Reachable code exists and builds; it may still be experimental. |
| **Measured** | A retained log supports exactly the stated metric and configuration. |
| **Parity-checked** | Compared from the same immutable state under the stated token/logit criterion. |
| **Quarantined** | Implemented experiment failed or has not passed a required release gate; off by default. |
| **Proposed** | Design, algebra, or future plan not claimed by the executing runtime. |
| **Legacy/future Linux** | Separate CUDA/uGDS target; not linked into the Apple Metal CLI. |
| **Vendored** | Upstream documentation retained for its component; not a Blok runtime claim. |

“Exact algebra” means equality over the stated mathematical domain. It does
not mean bitwise equality after quantized decode, FP16 storage, FP32 reduction,
or a changed kernel schedule. “Context admitted” means memory allocation and
addressing passed; it does not mean a prompt of that length completed.

## Active Apple-Metal documents

Read these in order:

1. [`../README.md`](../README.md) — shortest run command and repository scope.
2. [`../metalblok/docs/RUN_GUIDE.md`](../metalblok/docs/RUN_GUIDE.md) — complete
   CLI, continuation, diagnostics, and experimental-flag safety.
3. [`../metalblok/docs/V0_ARCHITECTURE.md`](../metalblok/docs/V0_ARCHITECTURE.md)
   — canonical end-to-end graph, equations, data ownership, scheduling,
   decisions, alternatives, and performance model.
4. [`../metalblok/docs/TENSOR_HARDWARE_SPEC.md`](../metalblok/docs/TENSOR_HARDWARE_SPEC.md)
   — normative tensor shapes, buffers, kernels, and accepted expanded-state
   ABI.
5. [`../metalblok/docs/EVIDENCE_AND_REPRODUCIBILITY.md`](../metalblok/docs/EVIDENCE_AND_REPRODUCIBILITY.md)
   — source-to-claim ledger and commands.
6. [`../metalblok/docs/PROOF_1K_REPORT.md`](../metalblok/docs/PROOF_1K_REPORT.md)
   — completed 1,000-input/1,000-output artifact.

Supporting records have deliberately narrower roles:

| Document | Role and authority |
|---|---|
| [`../metalblok/README.md`](../metalblok/README.md) | Component overview and mode table. |
| [`../metalblok/STATUS.md`](../metalblok/STATUS.md) | Current accepted, implemented, and quarantined status. |
| [`../metalblok/docs/V0_CLOSEOUT.md`](../metalblok/docs/V0_CLOSEOUT.md) | Historical closeout of the accepted expanded-KV V0. |
| [`../metalblok/docs/METALBLOK_PAPER.md`](../metalblok/docs/METALBLOK_PAPER.md) | Research-style explanation of the original result; not a substitute for the normative spec. |
| [`../metalblok/docs/PERFORMANCE_CLOSEOUT_2026-08-14.md`](../metalblok/docs/PERFORMANCE_CLOSEOUT_2026-08-14.md) | Later M5 profiling, accepted small wins, compact-mode results, and negative experiments. |
| [`../metalblok/docs/VERIFIED_LOOKAHEAD_FIFO.md`](../metalblok/docs/VERIFIED_LOOKAHEAD_FIFO.md) | Predictor/cache math, measured ceilings, and final rejection of active prefetch. |
| [`../metalblok/docs/M5_TENSOROPS_AND_METAL_IO.md`](../metalblok/docs/M5_TENSOROPS_AND_METAL_IO.md) | Public M5 API research and experiment contract; it does not claim the accepted path uses MPP TensorOps or Metal I/O. |
| [`../metalblok/docs/MILLION_TOKEN_SCALE_PLAN.md`](../metalblok/docs/MILLION_TOKEN_SCALE_PLAN.md) | Physically constrained future plan; not a statement that million-token execution exists. |

## Three active execution classes

| Class | CLI | Numerical contract | Evidence boundary |
|---|---|---|---|
| Accepted oracle | default | expanded FP16 non-RoPE K/V plus shared RoPE K; named V0 logit anchors | full 1K+1K completed at context 2,048 |
| Compact research | `--mla` | latent 512 plus RoPE 64 per layer/position; algebraically equivalent graph with changed finite-precision order | 128-token short probe and 31-step decode; no full 32K/128K prompt |
| Quarantined experiments | `--tensorops`; `--parallel-gate-up` until independently promoted | no release guarantee | faster or locally matching observations failed or have not completed the full-logit gate; active prefetch was removed |

## Legacy and future Linux documents

The following files describe a different Ryzen/NVIDIA/NVMe/uGDS platform and
different checkpoints. Their algebra and storage arguments are useful, but
none is evidence about the M5 runtime:

- [`architecture/model-mathematical-foundations.md`](architecture/model-mathematical-foundations.md)
- [`architecture/kimi-k2.6-on-target-hardware.md`](architecture/kimi-k2.6-on-target-hardware.md)
- [`architecture/glm-5.2-on-target-hardware.md`](architecture/glm-5.2-on-target-hardware.md)
- [`architecture/hardware-design-decisions.md`](architecture/hardware-design-decisions.md)
- [`architecture/evidence/derived-claims.md`](architecture/evidence/derived-claims.md)
- [`architecture/evidence/target-measurement-protocol.md`](architecture/evidence/target-measurement-protocol.md)

[`../sub_dir/uGDS/README.md`](../sub_dir/uGDS/README.md) and
[`../sub_dir/uGDS/docs/installation.md`](../sub_dir/uGDS/docs/installation.md)
are vendored upstream component documentation. The root CUDA/Kimi path may use
uGDS on Linux; MetalBlok does not.

[`../metalblok/vendor/llama_cpp/ATTRIBUTION.md`](../metalblok/vendor/llama_cpp/ATTRIBUTION.md)
records source provenance only. MetalBlok does not launch or link a llama.cpp
inference engine.

## Claim-conflict rule

When two records differ, use this precedence:

1. executing source and retained log for the named revision;
2. `TENSOR_HARDWARE_SPEC.md` for the accepted V0 ABI;
3. `V0_ARCHITECTURE.md` for rationale and graph semantics;
4. dated closeout/evidence records for the exact experiment they name;
5. scale and hardware-research notes as proposals only.

Never combine the best metric from one configuration with the bytes, cache,
or numerical result of another. A valid comparison holds model, starting
state, output length, decoding parameters, profiler perturbation, and memory
conditions fixed.
