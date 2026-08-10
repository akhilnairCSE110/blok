# MetalBlok Evidence and Reproducibility Ledger

**Artifact date:** 2026-08-10  
**Companions:** [`METALBLOK_PAPER.md`](METALBLOK_PAPER.md) and
[`TENSOR_HARDWARE_SPEC.md`](TENSOR_HARDWARE_SPEC.md)

This file binds the report's important claims to exact descriptor arithmetic,
executing code, or saved measurements. “Proposed” items are deliberately kept
separate from results.

## 1. Evidence vocabulary

| Class | Meaning |
|---|---|
| Exact algebra | Identity of the stated model graph. |
| Descriptor-derived | Integer arithmetic over GGUF shapes, types, and byte lengths. |
| Implemented | Present on the production GGUF/Metal path. |
| Measured | Observed on the named M5 host and retained in a log. |
| Proposed | A justified format/chip change not present in the measured path. |

## 2. Frozen artifact

The target is the local three-shard `DeepSeek-R1-UD-IQ1_S` GGUF. The runtime
discovered 1,025 tensors and 671,026,419,200 logical parameters. Preflight
reported:

| Shard | Logical bytes | Allocated bytes | Resident |
|---:|---:|---:|---|
| 0 | 49,349,193,664 | 49,349,193,728 | yes |
| 1 | 49,397,904,416 | 49,397,907,456 | yes |
| 2 | 41,484,340,384 | 41,484,341,248 | yes |
| **Total** | **140,231,438,464** | **140,231,442,432** | **yes** |

The stored tensor histogram is:

| Type | Tensors | Decimal GB | Share |
|---|---:|---:|---:|
| F32 | 361 | 0.43 | 0.3% |
| Q4_K | 190 | 6.67 | 4.8% |
| Q5_K | 116 | 1.17 | 0.8% |
| Q6_K | 184 | 2.83 | 2.0% |
| IQ2_XXS | 6 | 5.81 | 4.1% |
| IQ1_S | 168 | 123.31 | 87.9% |

## 3. Reproduction commands

From the repository root:

```sh
cmake -S metalblok -B metalblok/build
cmake --build metalblok/build -j 8
ctest --test-dir metalblok/build --output-on-failure
```

The final audit passed both registered tests:

```text
model_preflight .................. Passed
router_reference ................. Passed
100% tests passed, 0 tests failed out of 2
```

Use the first GGUF shard in each model command:

```sh
metalblok/build/metalblok --preflight /path/to/DeepSeek-R1-UD-IQ1_S-00001-of-00003.gguf
metalblok/build/metalblok --probe-gguf /path/to/DeepSeek-R1-UD-IQ1_S-00001-of-00003.gguf
metalblok/build/metalblok --validate-router
./run_blok.py "Hi" -n 8
```

Preflight must report `all_resident=true`, three shards, the exact manifest,
and `missing_physical=0`. It reads allocation metadata, not model payload, so
it does not hydrate a 140 GB cloud placeholder.

`--validate-router` needs host Metal access. A filesystem sandbox may report
`metal: no Metal device`; that is not a pass. The host-GPU audit on 2026-08-10
returned the same IDs as the independent CPU oracle:

```text
254, 79, 163, 171, 200, 247, 180, 242
```

The maximum displayed FP32 mixture-weight difference was
$3.0\times10^{-8}$.

## 4. Executable byte ledger

`--probe-gguf` derives these values from real tensor descriptor sizes:

```text
runtime_byte_ledger:
  steady_selected_useful=13770679744 (13.771 GB) reads=1937
  steady_all_experts=138861340096 (138.861 GB) reads=45089 reduction=10.084x
  cold_resident_norms=4026368 (4.026 MB) reads=245
  cold_absorbed_source=839516160 (839.516 MB) reads=61
  isolated_step_total_useful=14614222272 (14.614 GB) reads=2243
```

These are useful bytes and logical object reads for the implemented schedule,
not physical NAND transactions. APFS, NVMe, and the controller may split,
merge, or internally cache them.

## 5. Real-tensor CPU/Metal parity

For each type, the CPU independently decodes the stored blocks and computes
the same GEMV. The metric is

\[
\varepsilon_{rel}=
\frac{\lVert y_{Metal}-y_{CPU}\rVert_2}
{\max(\lVert y_{CPU}\rVert_2,10^{-12})}.
\]

| Type | Real tensor | Slice bytes | $\varepsilon_{rel}$ |
|---|---|---:|---:|
| F32 | `blk.3.ffn_gate_inp.weight` | 7,340,032 | $2.04\times10^{-4}$ |
| Q4_K | `blk.0.attn_q_a.weight` | 6,193,152 | $2.09\times10^{-4}$ |
| Q5_K | `blk.3.ffn_gate_shexp.weight` | 10,092,544 | $2.06\times10^{-4}$ |
| Q6_K | `blk.0.attn_kv_a_mqa.weight` | 3,386,880 | $2.13\times10^{-4}$ |
| IQ2_XXS | `blk.3.ffn_down_exps.weight` | 3,784,704 | $2.06\times10^{-4}$ |
| IQ1_S | `blk.3.ffn_gate_exps.weight` | 2,867,200 | $2.08\times10^{-4}$ |

All pass the declared $5\times10^{-3}$ threshold. This covers every stored
compute type on representative real matrices; it is not full reference-logit
equivalence.

## 6. End-to-end evidence

The formatted prompt `Hi` tokenized exactly as:

```text
0, 128803, 23166, 128804, 128798, 201
```

The exact checkpoint chain advanced from final prompt position 6 to position
38, producing 32 token transitions and text beginning:

```text
Okay, so I need to solve this problem. I'm trying to solve this problem.
```

The saved soak log contains 23 explicitly timed isolated steps from position
15 through 38:

| Quantity | Minimum | Maximum |
|---|---:|---:|
| 61-layer step | 8.199862 s | 8.598018 s |
| end-of-step RSS | 1,078 MB | 1,765 MB |

See [`../runs/2026-08-10-soak.log`](../runs/2026-08-10-soak.log) and the
timestamped wrapper logs in [`../runs/`](../runs/). The startup admission
ledger was 3.08 GB with a 3.22 GB reserve. Every successful child advanced the
atomic checkpoint by one.

The implied useful-byte-rate interval is:

\[
\frac{14.614222272\ \mathrm{GB}}{8.598018\ \mathrm{s}}
\le T_{useful}\le
\frac{14.614222272\ \mathrm{GB}}{8.199862\ \mathrm{s}},
\]

or 1.700–1.782 GB/s.

This includes cold MLA absorption, I/O, dispatch, compute, and synchronization;
it is not raw SSD bandwidth.

## 7. Claim matrix

| Claim | Status | Boundary |
|---|---|---|
| Router omission leaves only 8/256 routed experts. | Exact algebra | Requires exact routing first. |
| Routed-bank weight work is reduced 32×. | Exact + derived | Does not include fixed graph components. |
| Complete useful weight bytes are reduced 10.084×. | Descriptor-derived | Counterfactual is the same graph with all experts read. |
| MLA gives 71.111× smaller KV bytes/position. | Exact + derived | Costs 2.047 GB of absorbed matrices. |
| Six GGUF compute types execute on Metal. | Implemented + measured | Representative real-tensor parity, not all logits. |
| The full 61-layer 671B graph emits tokens. | Implemented + measured | Not proof of semantic quality. |
| Expert gate/up/down are physically bundled. | Proposed, not implemented | Current GGUF performs three reads/expert. |
| Temporal expert caching speeds this runtime. | Proposed, not implemented | Must report cache bytes and misses together. |
| Runtime controls physical NAND placement. | Not claimed | It controls APFS file offsets only. |
| Context 163,840 is validated. | Not claimed | Current verified context is 64. |

## 8. Source map and remaining proof obligations

| Concern | Source |
|---|---|
| tensor graph and byte dispatch | [`../src/gguf_runtime.cpp`](../src/gguf_runtime.cpp) |
| GGUF parsing | [`../src/gguf.cpp`](../src/gguf.cpp) |
| quant/MLA/router kernels | [`../src/kernels.metal`](../src/kernels.metal) |
| CPU quant oracle | [`../src/gguf_dequant.cpp`](../src/gguf_dequant.cpp) |
| CPU router oracle | [`../src/router_ref.hpp`](../src/router_ref.hpp) |
| residency refusal | [`../src/preflight.cpp`](../src/preflight.cpp) |
| I/O ownership | [`../src/pread_ring.cpp`](../src/pread_ring.cpp) |
| safe runner | [`../../run_blok.py`](../../run_blok.py) |

Before a production-equivalence or chip-speedup claim, compare expanded versus
absorbed MLA, per-layer residuals and final logits against an independent
runtime, trace `pread` offsets/lengths, measure a physically bundled checkpoint,
and report temporal-cache hits, misses, resident bytes, and latency together.

The rigorous conclusion today is narrower and concrete: an exact
router-selected, quantized, latent-attention 671B graph executes from a
140.23 GB NAND-backed file on a 24 GB unified-memory machine, with current
tensor contracts, bytes, reads, safety boundaries, and limitations accounted.
