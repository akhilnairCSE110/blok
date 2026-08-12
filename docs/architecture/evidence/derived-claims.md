# Derived Architecture Evidence

> **Legacy/future Linux evidence.** These deterministic Kimi/GLM calculations
> support the CUDA/uGDS documents only. Active M5/DeepSeek measurements are in
> the [MetalBlok evidence ledger](../../../metalblok/docs/EVIDENCE_AND_REPRODUCIBILITY.md).

**Evidence class:** deterministic integer derivation and source audit. These values are not measured latency or bandwidth.

Reproduce with:

```bash
python3 -m scripts.derive_architecture_claims --check
```

## Model results

| Quantity | Kimi K2.6 | GLM-5.2 FP8 |
|---|---:|---:|
| Checkpoint tensor bytes | 595,148,192,736 | 755,617,140,416 |
| Cold sampled-token weight bytes | 32,986,459,136 | 41,383,396,416 |
| Routed bank bytes | 570,760,888,320 | 724,952,678,400 |
| Selected routed bytes/token | 11,890,851,840 | 22,654,771,200 |
| Expert record bytes | 24,772,608 | 37,757,952 |
| Weight FLOPs/token | 63,372,132,352 | 80,595,517,440 |
| Expanded KV bytes/sequence token | 4,997,120 | 5,111,808 |
| Latent KV bytes/sequence token | 70,272 | 89,856 |

## Selection and representation ratios

- Kimi routed-bank reduction: 48; expert compression: 3.555556×; KV reduction: 71.111111×.
- GLM routed-bank reduction: 32; expert compression: 1.999512×; KV reduction: 56.888889×.

## uGDS command split

With a 4 KiB controller page, the current one-PRP-list cap is 2,101,248 bytes. This is a source-derived implementation limit, not a measured target MDTS.

| Record | Commands at PRP cap | Commands at 128 KiB fallback |
|---|---:|---:|
| Kimi expert | 12 | 189 |
| GLM expert | 18 | 289 |

## Source identities

| Source | SHA-256 |
|---|---|
| `src/kimi_exec.cu` | `1bcb74c5428c793508950e434857466519a6ac17c74596aabbe47afa9ed401c9` |
| `blok/runtime.py` | `d84783cc312980e57999b1cee0771939f381626e02a5731258eda3310f5f4fbc` |
| `scripts/model_fetch.py` | `73c4f4666a9de0412cc31f202bdb24fca3bad319c6363f2528eca24075f77bf1` |
| `sub_dir/uGDS/src/ugds_batch.cpp` | `0afc5da4155fc3bf4762b636d5122f2435a239248d147d22d1a4bc8cc6695c21` |
| `sub_dir/uGDS/src/ugds_internal.h` | `42ec976f232db969b820235e5b0fb79092b33b4c8db93c23a051398e05d7bf4b` |

## What this file does not prove

It does not prove GPU correctness, PCIe peer routing, SSD bandwidth, request latency, expert reuse, thermal stability, or end-to-end token output. Those require the target run described in `target-measurement-protocol.md`.
