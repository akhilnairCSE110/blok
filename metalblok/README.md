# MetalBlok

MetalBlok is a bounded, native Metal inference runtime for the local
DeepSeek-R1 671B `UD-IQ1_S` GGUF checkpoint. Correctness and host safety are
release gates; caching, overlap, and kernel tuning come afterward.

## Architecture package

- [`docs/METALBLOK_PAPER.md`](docs/METALBLOK_PAPER.md) gives the arXiv-style
  mathematical argument, complete transformer/MoE/MLA forward pass, measured
  results, limitations, and chip implications.
- [`docs/TENSOR_HARDWARE_SPEC.md`](docs/TENSOR_HARDWARE_SPEC.md) is the
  tensor-by-tensor normative hardware contract: shapes, addresses, block
  decoders, kernels, buffer states, byte counts, and concrete decisions.
- [`docs/EVIDENCE_AND_REPRODUCIBILITY.md`](docs/EVIDENCE_AND_REPRODUCIBILITY.md)
  separates exact, derived, implemented, measured, and proposed claims and
  supplies the reproduction commands and saved-log index.

The initial code is derived from the adjacent Blade prototype. Imported code
is treated as untrusted until it passes MetalBlok's real-model gates.

## One-command run

From the repository root:

```sh
./run_blok.py "Hi"
```

The safe default is eight output tokens. Select a different verified limit
with `-n`; the wrapper refuses values above 32:

```sh
./run_blok.py "Explain why the sky is blue." -n 16
```

The wrapper builds when necessary, performs the non-hydrating shard
preflight, enforces the 64-token verified context, serializes concurrent runs,
uses a unique atomic checkpoint, and executes one token per child process. It
prints model text on stdout and records complete native diagnostics under
`metalblok/runs/`.

## Build

```sh
cmake -S metalblok -B metalblok/build
cmake --build metalblok/build -j
```

The default build compiles `kernels.metal` through the Metal runtime. An
offline `.metallib` can be selected later with
`-DMETALBLOK_OFFLINE_METALLIB=ON` after installing Apple's Metal Toolchain.

## Safe first command

```sh
metalblok/build/metalblok --preflight /path/to/model-00001-of-00003.gguf
```

Preflight never reads model payload and refuses sparse or dataless shards.
Do not run inference until it reports every shard as resident.
