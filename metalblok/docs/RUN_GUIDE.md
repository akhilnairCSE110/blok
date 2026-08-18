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

For synchronization-heavy operation attribution:

```sh
./run_blok.py "profile this" -n 32 --mla --profile-ops
```

`--profile-ops` logs individual projection/attention/MoE boundaries and a
representative per-expert gate/up/down split. Decode normally fuses attention
and router in one command buffer; operation profiling inserts a boundary so
their GPU time is attributed truthfully. It intentionally perturbs the Metal
schedule and is for diagnosis, not headline throughput.
`--profile-layers` also logs per-shard/per-reader I/O work. The measured M5
default is eight readers per shard; compare safely with `--io-lanes 4`.

`--profile-predictor` measures state-conditioned cross-layer lookahead. By
default, each residual is tested with the real resident DeepSeek routers for
the current and next three layers. Each target route is later compared with
the authoritative post-attention route. `--predictor-depth 1..8` controls this
diagnostic horizon. It adds work and is never a throughput run:

```sh
./run_blok.py "predictor trace" -n 32 --mla --profile-predictor
```

The analyzer reports every lookahead distance separately: top-8
recall/precision, predictor wall/GPU cost, router-verification lead,
first-expert-use lead, and false/late traffic. The probe never issues
speculative reads and cannot affect model output.

`--expert-cache-ways N` enables the exact resident expert cache. Compact MLA
defaults to the measured four-way cache; expanded V0 leaves it disabled unless
this flag is supplied, preserving the accepted baseline by default. The
runtime clamps the request to the measured UMA budget and reserve.

There is deliberately no active expert-prefetch flag. The measured rank-one
candidate improved raw wall time but failed the full-logit/checkpoint gate, so
its implementation was removed rather than left as an attractive unsafe mode.
Use the analyzer to evaluate prediction coverage without issuing reads:

```sh
python3 scripts/analyze_expert_routes.py metalblok/runs/run-....log
```

`--tensorops` enables an experimental M5 MPP QMM prefill path. It is faster,
but it reassociates FP32 reductions and has not passed strict logit/routing
parity. Do not use it for strict acceptance runs.

`--parallel-gate-up` executes each routed expert's independent IQ1_S gate and
up dot products on two 32-lane SIMD groups. Each dot retains the accepted lane
assignment and reduction order; routed down projections still accumulate in
router-rank order. Compare it against the strict default on the target before
promotion:

```sh
./run_blok.py "profile this" -n 32 --mla --parallel-gate-up --profile-layers
```

`--expert-cache-ways 0..32` controls the verified per-layer expert-history
cache. The runtime clamps the request to the largest allocation that preserves
its six-GiB reserve; `32` therefore means "use the safe available headroom, up
to 32." Entries retain exact gate/up/down tensors from earlier routes and an
authoritative router miss always falls back to SSD.

`--expert-group-size 1|2|4|8` is the routed I/O/command scheduling knob.
Smaller groups can start compute after fewer expert reads; larger groups reduce
command-buffer boundaries. Expert rank and accumulation order do not change.
The accepted default is `4`; tune only with parity-matched decode runs because
the winning value depends on the measured SSD/GPU overlap.

Compare completed experiments as a parity-gated performance/area/traffic
frontier (the first log is the strict reference):

```sh
python3 scripts/synthesize_decode_config.py REF.log CANDIDATE*.log \
  --max-cache-gb 6
```

This reports measured whole-decode tokens/s and end-to-end wall, p50/p95
forward latency, GPU/I/O time, NVMe bytes, cache capacity, command work,
allocations, and pageouts. Every new sample record contains a bitwise hash of
the full logit vector; only matching hashes enter the strict frontier. Legacy
logs are labeled `sample` and excluded unless the explicitly weaker
`--allow-sample-parity` option is supplied.

On the target, synthesize the default decode schedule automatically from one
immutable golden checkpoint:

```sh
python3 scripts/tune_decode.py --state /path/to/golden.state -n 32
```

It uses APFS copy-on-write clones, runs `0:4,4:1,4:2,4:4,4:8,8:4` as
`cache-ways:expert-group-size`, deletes only its temporary clones, then invokes
the strict full-logit-hash frontier report. Override the search with, for
example, `--configs 0:4,4:2,4:4 --max-cache-gb 3`.

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
  buffers, eight measured readers per shard, and fail the process on any short
  or failed read. `--io-lanes {2,4,8}` is the bounded fallback/tuning control.
- Metal command-buffer faults abort with the last kernel name.
- Checkpoints use `.partial`, flush, `fsync`, and atomic rename. A partial file
  is never silently treated as authoritative.
- One wrapper lock prevents concurrent runs from competing for the 24 GB
  unified-memory and SSD working set. Multi-request scheduling is future work,
  not a V0 claim.
