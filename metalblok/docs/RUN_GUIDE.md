# MetalBlok run guide

This is the complete CLI guide for the native Apple-Silicon V0. The active
model contract is the three-shard DeepSeek-R1 671B `UD-IQ1_S` GGUF. No Python
ML framework, llama.cpp process, MLX runtime, server, or model conversion is
in the inference path.

## Requirements

- Apple Silicon macOS with Metal.
- Enough free unified memory for the selected context. Context 2,048 is the
  verified 1,000-input/1,000-output setting on the 24 GB target.
- The exact three GGUF shards physically present on a fast local SSD.
- CMake and an Apple C++ compiler. The Metal source compiles at runtime, so
  the optional offline Metal toolchain is not required.

From the repository root, point the runner at shard 1 if the model is not at
the developer-machine default:

```sh
export METALBLOK_MODEL=/absolute/path/DeepSeek-R1-UD-IQ1_S-00001-of-00003.gguf
```

The other two shards must use the matching `00002-of-00003` and
`00003-of-00003` names in the same directory.

## Run one prompt

```sh
./run_blok.py "Write a correct C++ function that validates UTF-8." -n 256
```

The defaults are greedy decoding, 256 emitted tokens, and a 2,048-position
context. The wrapper builds on first use, verifies the exact model manifest
without hydrating missing cloud files, serializes native runs, streams model
text to stdout, and prints all artifact paths to stderr.

For the requested 1,000-output-token run:

```sh
./run_blok.py "Write a complete, portable C++17 JSON Lines statistics program with tests." \
  -n 1000 --context 2048 --temperature 0 \
  --state metalblok/runs/cpp-demo.state
```

The prompt plus generated positions must fit the context. For exactly 1,000
native input tokens and 1,000 emitted output tokens, use the acceptance script:

```sh
scripts/prove_metal_1k.py
```

It constructs a deterministic coding prompt of exactly 1,000 IDs with the
native tokenizer, requests exactly 1,000 emitted IDs, and accepts only a final
checkpoint position of 1,999.

## Continue after interruption

An interrupted decode retains the last atomically renamed checkpoint. Continue
the same unfinished generation with:

```sh
./run_blok.py continue -n 256 \
  --state metalblok/runs/cpp-demo.state --continue-decode
```

`continue` is a placeholder positional argument; the saved pending token and
KV state are authoritative. The checkpoint interval defaults to 256 positions.
Reduce it when recovery granularity matters more than checkpoint write cost:

```sh
./run_blok.py "prompt" -n 1000 --state demo.state --checkpoint-every 64
```

At normal completion, the final state is always saved regardless of the
interval.

## Continue a conversation

Create a persistent conversation:

```sh
./run_blok.py "My name is Max. Remember it." -n 128 \
  --state conversations/max.state --context 2048
```

Append a new user turn by reusing the state without `--continue-decode`:

```sh
./run_blok.py "What is my name?" -n 64 \
  --state conversations/max.state
```

The runner selects native `--resume-turn`, closes the pending assistant turn,
adds the new formatted user turn, and extends the exact KV cache. A state keeps
the context capacity selected when it was created; create it with enough room
for the entire conversation.

## Output and diagnostics

Every invocation prints paths such as:

```text
[MetalBlok] state=/.../conversation.state
[MetalBlok] diagnostics=/.../metalblok/runs/run-YYYYMMDD-HHMMSS-PID.log
[MetalBlok] output=/.../metalblok/runs/run-YYYYMMDD-HHMMSS-PID.txt
```

The `.txt` file contains model text only. The `.log` file includes token IDs,
greedy logits, end-to-end and GPU time, explicit I/O wait, useful/model and
NVMe bytes, request counts, effective GB/s, urgent expert traffic, service
time, peak outstanding reads, Metal command buffers and dispatches, hot-path
allocations, exact KV bytes per position, available-memory delta, pageouts,
compression, decompression, swap-ins, and swap-outs.

For layer attribution:

```sh
./run_blok.py "profile this" -n 32 --profile-layers
```

For numerical fingerprints and top logits:

```sh
./run_blok.py "trace this" -n 32 --trace
```

Tracing is diagnostic work and can slow execution. Ordinary runs retain the
compact per-token metrics line.

## Build and test directly

```sh
cmake -S metalblok -B metalblok/build
cmake --build metalblok/build -j8
ctest --test-dir metalblok/build --output-on-failure
python3 -m py_compile run_blok.py scripts/prove_metal_1k.py
```

Useful native checks are:

```sh
metalblok/build/metalblok --preflight "$METALBLOK_MODEL"
metalblok/build/metalblok --probe-gguf "$METALBLOK_MODEL"
metalblok/build/metalblok --validate-router
```

`--validate-router` needs real host Metal access. A sandbox reporting
`metal: no Metal device` has not run the check.

## Safety and failure semantics

- Preflight refuses a missing, sparse, dataless, wrongly sized, or wrong-model
  shard before reading payload.
- The memory ledger refuses a context that cannot preserve host headroom.
- Model payload reads use `F_NOCACHE`, no read-ahead, reusable aligned shared
  buffers, and fail the process on any short or failed read.
- Metal command-buffer faults abort with the last kernel name.
- Checkpoints use `.partial`, flush, `fsync`, and atomic rename. A partial file
  is never silently treated as authoritative.
- One wrapper lock prevents concurrent runs from competing for the 24 GB
  unified-memory and SSD working set. Multi-request scheduling is future work,
  not a V0 claim.

