# MetalBlok

MetalBlok is the active native V0 for DeepSeek-R1 671B `UD-IQ1_S` on the
24 GB Apple M5 target. It performs real tokenization, blocked prefill, exact
continuation, all 61 Metal transformer layers, grouped top-8 MoE routing,
mixed-GGUF quantized multiplication, causal attention, sampling, and decoding.

## Fast path

From the repository root:

```sh
./run_blok.py "Explain why the sky is blue." -n 256
```

For a model outside the default local path:

```sh
export METALBLOK_MODEL=/path/to/DeepSeek-R1-UD-IQ1_S-00001-of-00003.gguf
```

The wrapper defaults to greedy decoding and context 2,048, prints model text,
and saves exact state, output, and diagnostics paths. It preflights all three
shards, enforces one native run at a time, and builds on first use.

For exactly 1,000 native input and 1,000 emitted output tokens:

```sh
scripts/prove_metal_1k.py
```

For persistent chat and interruption recovery, read the
[full run guide](docs/RUN_GUIDE.md).

## Architecture and evidence

- [Complete V0 architecture](docs/V0_ARCHITECTURE.md): deep mathematical,
  memory, NVMe, Metal, concurrency, KV, and performance rationale.
- [Million-token scale plan](docs/MILLION_TOKEN_SCALE_PLAN.md): the paged
  state, compact MLA, SSD attention, QMM prefill, and model-quality path to
  one-million input plus one-million output tokens.
- [V0 closeout](docs/V0_CLOSEOUT.md): implemented schedule, rejected
  optimizations, measured deltas, logging contract, and acceptance artifacts.
- [Tensor/hardware specification](docs/TENSOR_HARDWARE_SPEC.md): normative
  shapes, exact expanded-KV contract, buffer ownership, and kernels.
- [Evidence ledger](docs/EVIDENCE_AND_REPRODUCIBILITY.md): commands, saved
  logs, parity gates, and non-claims.
- [Architecture study](docs/METALBLOK_PAPER.md): mathematical motivation and
  the relation to conditional weight streaming.

## Build

```sh
cmake -S metalblok -B metalblok/build
cmake --build metalblok/build -j8
ctest --test-dir metalblok/build --output-on-failure
```

The standard build runtime-compiles `kernels.metal`; an offline `.metallib` is
optional. `--preflight` reads allocation metadata rather than model payload and
must report `all_resident=true` before inference.
