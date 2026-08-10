# Vendored from llama.cpp / ggml

This directory contains data and code derived from the
[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) project.

* **Upstream commit:** `bbeb89d76c41bc250f16e4a6fefcc9b530d6e3f3` (2026-05-05)
* **License:** MIT — see `LICENSE`
* **Copyright:** © 2023-2026 The ggml authors

## Why we vendor (and why we keep it minimal)

llama.cpp is the de-facto production reference implementation of the GGUF
container format and of the IQ-series sub-2-bit quantization codebooks used
by Unsloth's "Dynamic" R1 release. The IQ codebooks in particular are
**trained data tables** (Lloyd-Max-style optimization on a calibration set)
— there is no algorithmic reproduction of them. We must use the published
tables verbatim, or our outputs will not match the quantizer's intent.

Everything else in `blade` (streaming, MLA, MoE, KV cache, runtime, CLI)
is original. We avoid pulling in `ggml.h`, the ggml graph runtime, the
backend dispatch layer, or any of the build infrastructure. We take only
the irreplaceable bits.

## What lives here

| File | Origin | Purpose |
|------|--------|---------|
| `LICENSE` | `LICENSE` (verbatim) | MIT license text |
| `iq1s_grid.h` | `ggml/src/ggml-common.h` lines 1118–1637 (data verbatim) | 2048-entry IQ1_S codebook + `IQ1S_DELTA` constant |

Each vendored file carries a header comment naming the upstream path and
commit. When we extend this directory, every new file follows the same
convention — no opaque blobs, no rewrites of upstream constants.
