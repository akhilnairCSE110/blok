# 0715 Feedback Todos: Blok Architecture

## Review Summary

The current code is not a fake text generator. The native CUDA executor is wired to run an autoregressive path: tokenize prompt, read real tensor payloads, execute a 61-layer decode loop, run attention and MLP/MoE kernels, project through the LM head, choose an argmax token, and detokenize.

That is not enough to claim it runs a correct full Kimi K2.6 forward pass. The code is structurally connected, but it is not target-built, not target-run, and not numerically validated. The most important gap is not style or boilerplate. The highest-risk gap is model correctness: attention, MLA layout, tokenizer behavior, RoPE scaling, tensor shape assumptions, and INT4 dequantization all need direct verification against authoritative sources.

This file lists the exact review findings, the sources needed to verify each one, and the implementation todos that should be done before calling this a working architecture.

## Local Verification Performed

Commands run from `blok/`:

```sh
cargo build
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
cmake --build build
target/debug/blok --help
target/debug/blok generate --model Cargo.toml --prompt hello --tokens 1
```

Observed results:

- Rust build passed.
- Rust clippy passed.
- Rust tests passed, but there are zero tests, so this does not verify runtime behavior.
- `target/debug/blok --help` prints the expected CLI usage.
- Passing `Cargo.toml` as a fake model fails correctly at manifest parsing.
- CUDA executor was not built locally.
- `build/blok-kimi-exec` is absent.
- `cmake --build build` failed because the configured Ninja build tool is missing.
- `ninja` is not installed locally.
- `nvcc` is not installed locally.

Conclusion: this machine cannot verify the CUDA executor or a full forward pass.

## Architecture Currently Present

The intended execution path is:

```text
blok.runtime / CLI
  -> target/{debug,release}/blok generate
  -> src/kimi_runtime.rs
  -> build/blok-kimi-exec
  -> src/kimi_exec.cu
  -> runtime-index.blok + tokenizer.blok + safetensor payload files
  -> CUDA kernels
  -> JSON output
```

The Rust layer is a launcher and response parser. The model execution is entirely in `src/kimi_exec.cu`.

Relevant code:

- `src/blok_command_line_entrypoint.rs`: CLI process entrypoint.
- `src/blok_runtime_library.rs`: argument parsing, manifest discovery, JSON response emission.
- `src/kimi_runtime.rs`: native executor launch and JSON parsing.
- `src/tensor_manifest_parser.rs`: minimal manifest parser and validation.
- `src/primitives.rs`: Kimi K2.6 constants used by Rust planning.
- `src/kimi_exec.cu`: tokenizer, runtime index parsing, tensor I/O, CUDA kernels, decode loop.
- `CMakeLists.txt`: CUDA executor build wiring.
- `scripts/model_fetch.py`: materializes `manifest.blok`, `runtime-index.blok`, and `tokenizer.blok`.
- `scripts/check_kimi_contract.py`: current model-side contract check.
- `scripts/ci.sh`: verification levels.

## Finding 1: Rust Does Not Prove Model Execution

The Rust runtime validates CLI inputs and launches an external binary:

```rust
let bin = executor_bin();
if !bin.is_file() {
    return Err(Error::Capability("blok_kimi_exec_binary_required"));
}
```

This means Rust passing build, clippy, and tests does not prove a forward pass. The only forward implementation is the native executor.

Why this matters:

- Rust success can coexist with a missing CUDA binary.
- Rust success can coexist with invalid CUDA math.
- Rust success can coexist with wrong tensor names or shapes.

Sources needed:

- Local source: `src/kimi_runtime.rs`.
- Local build artifact: `build/blok-kimi-exec`.
- Target-machine command output from:

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
BLOK_KIMI_EXEC_BIN=build/blok-kimi-exec target/release/blok generate --model /path/to/manifest.blok --prompt hello --tokens 1
```

Todo:

- Treat Rust checks as launcher checks only.
- Add at least one test around CLI error behavior and executor JSON parsing.
- Do not mark model execution verified until `blok-kimi-exec` is built and run.

## Finding 2: CUDA Executor Is Wired, But Not Locally Build-Verified

`src/kimi_exec.cu` defines kernels and a full `generate` function. It loads embeddings, final norm, and LM head once, then performs a token-by-token loop over all 61 layers.

Important code areas:

- Tokenizer load and encode/decode: `tokenizer`, `encode`, `decode`.
- Runtime tensor metadata: `runtime_index`.
- Tensor I/O: `load_tensor`.
- Forward contract checks: `validate_forward_contract`.
- Decode loop: `generate`.
- Layer loop: lambda `run`.

Why this matters:

- The code is not a no-op.
- But CUDA compiler errors, uGDS API mismatches, C++/CUDA portability issues, and target architecture settings are unverified.

Sources needed:

- Target-machine `nvcc --version`.
- Target-machine `nvidia-smi`.
- Target-machine `cmake --build build` logs.
- Target-machine `compute-sanitizer` output for `blok-kimi-exec`.
- uGDS driver/library version and build logs.
- GPU architecture confirmation for `CMAKE_CUDA_ARCHITECTURES=120`.

Todo:

- Build on the intended Linux/CUDA/uGDS target.
- Capture full CMake configure and build logs.
- Run `compute-sanitizer --tool memcheck` and `--tool synccheck` against the executor.
- Confirm whether `CMAKE_CUDA_ARCHITECTURES=120` is correct for the deployed GPU.

## Finding 3: Attention Implementation Looks Incorrect For Multi-Head MLA

The current attention score kernel computes one scalar score per timestep by dotting flattened full-query state against flattened cached key state:

```cpp
attn_score_k<<<pos + 1, 256>>>(q.p, kc[l].p, score.p, pos + 1, HEADS * (QK_NOPE + QK_ROPE));
softmax_k<<<1, 256>>>(score.p, pos + 1);
attn_value_k<<<...>>>(score.p, vc[l].p, av.p, pos + 1, HEADS * V_HEAD);
```

This produces one softmax distribution over sequence positions, shared across all heads.

That is not the usual multi-head attention computation, where each head has separate scores and separate softmax over sequence positions. For MLA, the exact computation may differ from standard MHA, but it still needs verification against Kimi's actual forward equations.

Why this matters:

- If heads are collapsed into one score stream, final logits will diverge immediately.
- Even a one-token generation may produce a token, but it would not be the model's token.
- This is likely the highest-priority correctness risk.

Sources needed:

- Official Kimi K2.6 model config.
- Official Kimi K2.6 modeling implementation or trusted reference implementation.
- `config.json` fields for MLA dimensions, head counts, RoPE behavior, and attention scaling.
- A known-good PyTorch/Transformers forward trace for one prompt.
- Per-layer reference tensors for at least:
  - input RMSNorm output;
  - q/k/v projection output;
  - RoPE-applied q/k;
  - attention scores;
  - attention probabilities;
  - attention output;
  - post-attention residual.

Todo:

- Add a tiny trace mode to the executor that can dump selected intermediate tensors for one token and one layer.
- Compare layer 0 attention against an official/reference implementation.
- Fix attention to compute the exact Kimi MLA attention semantics, not a flattened approximation.

## Finding 4: KV Layout Assumptions Are Not Proven

The executor assumes projection output layout like this:

```cpp
cudaMemcpy(k.p, kv.p, HEADS * QK_NOPE * sizeof(float), cudaMemcpyDeviceToDevice);
k_rope_fill_k(..., kva.p + KV_RANK);
cudaMemcpy(v.p, kv.p + HEADS * QK_NOPE, HEADS * V_HEAD * sizeof(float), cudaMemcpyDeviceToDevice);
```

This assumes key-nope values occupy the first contiguous segment and values occupy the next contiguous segment.

Why this matters:

- Many model projection layouts are interleaved by head.
- If the actual tensor layout differs, all attention outputs are wrong.
- The code needs either a guaranteed repack contract or shape/layout-aware unpack kernels.

Sources needed:

- Real `runtime-index.blok` generated from the full Kimi K2.6 checkpoint.
- Real safetensor header metadata for all attention tensors.
- Official/reference Kimi attention projection output shapes and layout.
- Materializer source behavior from `scripts/model_fetch.py`.
- Any documented repack format, if one exists.

Todo:

- Verify exact shapes for `q_a_proj`, `q_b_proj`, `kv_a_proj`, `kv_b_proj`, and `o_proj`.
- Document the runtime tensor layout contract.
- Add validation that fails if runtime-index shapes do not match executor assumptions.
- If no repack exists, implement correct extraction/indexing for the official layout.

## Finding 5: RoPE/YaRN Is Simplified

The executor uses:

```cpp
constexpr float ROPE_THETA = 50000.0f;
```

and a simple RoPE kernel:

```cpp
float freq = powf(ROPE_THETA, -2.0f * p / rotary);
```

The repo docs already state YaRN long-context scaling is not implemented.

Why this matters:

- Kimi K2.6 long-context behavior depends on exact RoPE/YaRN scaling.
- Even short-context logits can differ if scaling constants or rotary placement are wrong.
- A model can emit text while still being numerically wrong.

Sources needed:

- Official Kimi K2.6 `config.json`.
- Official/reference implementation of RoPE/YaRN for this model.
- Token-position parity tests at positions 0, 1, medium context, and long context.

Todo:

- Implement exact RoPE/YaRN from the official config.
- Add a unit/parity test for rotary embeddings at fixed positions.
- Keep simple RoPE only if the config proves that is correct for the tested context.

## Finding 6: Tokenizer And Chat Template Are Not Verified

The executor builds a custom tokenizer from `tokenizer.blok` and tokenizes raw prompt bytes:

```cpp
for (unsigned char c : s) pieces.emplace_back(1, (char)c);
```

It does not apply a chat template. It only handles one EOS id:

```cpp
constexpr std::uint32_t EOS_IM_END = 163586;
```

Why this matters:

- The same human prompt can map to different tokens depending on official chat template and special-token handling.
- Incorrect tokenization invalidates every downstream comparison.
- Decoding raw token bytes may mishandle special tokens or invalid UTF-8.

Sources needed:

- Official `tokenizer.json`.
- Official `tokenizer_config.json`.
- Official chat template.
- Reference token ids for representative prompts:
  - plain ASCII;
  - Unicode text;
  - prompt with Kimi chat roles;
  - prompt containing special-token-like strings;
  - empty/whitespace prompts.

Todo:

- Add a tokenizer parity script comparing `tokenizer.blok` output against the official tokenizer.
- Decide whether `blok generate --prompt` means raw completion text or chat-formatted text.
- Implement only the chosen behavior, and document it.
- Add stop-token handling for all relevant official stop ids.

## Finding 7: Runtime Contract Validation Is Too Shallow

`validate_forward_contract` checks for the presence of broad slots. It does not fully validate all shapes/dtypes against the constants used by the kernels. It spot-checks routed expert `0`, but the route selection can choose any expert from 0 to 383.

Why this matters:

- Missing or malformed expert tensors may not fail until a specific expert is selected.
- Shape mismatches can become out-of-bounds CUDA reads/writes.
- Incorrect dtype assumptions can silently corrupt math.

Sources needed:

- Complete generated `runtime-index.blok`.
- Complete safetensor header list.
- Official Kimi K2.6 tensor naming scheme.
- Expected shape table for every tensor class.

Todo:

- Validate all 384 routed experts for all MoE layers.
- Validate exact shape and dtype for:
  - embedding;
  - final norm;
  - LM head;
  - all attention projections;
  - all attention norms;
  - dense layer 0 MLP;
  - all routers;
  - all shared experts;
  - all routed expert packed weights and scales.
- Fail before CUDA execution if any runtime-index entry violates the contract.

## Finding 8: INT4 Dequantization Needs Checkpoint-Specific Validation

The routed expert path assumes packed symmetric INT4 with:

```cpp
int q = (int)((word >> (4 * lane)) & 0xf) - 8;
v += x[...] * ((float)q * sc);
```

It validates packed shapes with group size 32 and 8 values per `i32`.

Why this matters:

- The actual quantization format may use different zero point, nibble order, signed mapping, scale dtype, or packing convention.
- If this is wrong, all routed MoE output is wrong even if shapes match.

Sources needed:

- Official checkpoint quantization documentation, if available.
- Safetensor tensor names and dtypes for routed weights and scales.
- Reference CPU/PyTorch dequantization for one expert weight.
- A known-good matvec result for one expert projection.

Todo:

- Build a small offline parity check:
  - read one routed expert weight and scale;
  - dequantize using executor logic;
  - compare against reference dequantization;
  - compare one matvec output.
- Only then trust the routed expert CUDA path.

## Finding 9: Tensor Loading Is Correctness-Oriented But Extremely Expensive

The executor reloads many tensors inside every layer and token. For each token, each layer reloads attention tensors, norms, routers, experts, and shared experts.

Why this matters:

- This can be acceptable for first correctness if it runs.
- It is too slow for production.
- Repeated allocation and free inside the decode loop can hide synchronization and lifetime bugs.

Sources needed:

- Target run timing logs.
- uGDS read bandwidth measurements.
- CUDA profiler trace.
- GPU memory capacity report.
- Expected tensor residency plan.

Todo:

- Keep on-demand loading until correctness is proven.
- After correctness, add the smallest useful cache:
  - resident embeddings/final norm/head already loaded once;
  - attention/norm/shared/router tensors should likely become resident or cached;
  - routed experts should use a real eviction policy only after routing correctness is proven.

## Finding 10: Output Metadata Is Misleading

The executor prints:

```cpp
"tokens": a.tokens
"predicted_tps": 0
"watts": null
```

This reports requested tokens, not actual generated tokens. If EOS stops early, the count is wrong. Throughput and power are placeholders.

Why this matters:

- Users may think performance and token count are measured.
- Automation may treat successful JSON as stronger evidence than it is.

Sources needed:

- Actual generated token vector length.
- Wall-clock timing around prefill/decode.
- Optional power measurement source, if power reporting remains in the protocol.

Todo:

- Emit actual generated token count.
- Rename or remove `predicted_tps` until measured throughput exists.
- Remove `watts` or explicitly emit measured power only when implemented.

## Finding 11: Verification Levels Are Honest But Incomplete

`scripts/ci.sh` already makes this explicit:

- `verify-l0`: static checks.
- `verify-l1`: model contract.
- `verify-l2`: CUDA build/sanitizer availability.
- `verify-l3`: currently fails because official tokenizer/MLA/logit parity fixtures are missing.
- `verify-l4`: hardware check.
- `verify-l5`: currently fails because regression fixtures are missing.

Why this matters:

- The repo already knows that numerical validation is missing.
- The correct next work is to implement those validation gates, not to add broad abstractions.

Sources needed:

- Official/reference tokenizer outputs.
- Official/reference one-token logits.
- Revision-keyed fixture data tied to exact model and tokenizer revisions.
- Target hardware run logs.

Todo:

- Make `verify-l3` real.
- Make `verify-l5` real.
- Require `verify-l3` before claiming model correctness.
- Require `verify-l4` before claiming target hardware viability.

## Required Evidence Before Claiming "Full Forward Pass Works"

Minimum evidence:

1. Build evidence:
   - `nvcc --version`;
   - `nvidia-smi`;
   - full CMake configure/build logs;
   - produced `build/blok-kimi-exec`.

2. Model materialization evidence:
   - full `manifest.blok`;
   - full `runtime-index.blok`;
   - generated `tokenizer.blok`;
   - list of all safetensor shards and sizes;
   - exact model revision/commit.

3. Tokenizer evidence:
   - official tokenizer revision;
   - official token ids for fixed prompts;
   - local tokenizer ids for the same prompts;
   - exact match report.

4. Forward numerical evidence:
   - one prompt;
   - exact prompt token ids;
   - reference first-token logits or top-k logits;
   - local first-token logits or top-k logits;
   - tolerance report;
   - generated greedy token match.

5. Layer-level evidence:
   - at least layer 0 attention trace;
   - one MoE layer router top-k trace;
   - one routed expert matvec trace;
   - final norm and LM head trace.

6. Target execution evidence:
   - uGDS driver loaded;
   - `/dev/ugds_drv0` present;
   - valid `BLOK_UGDS_MAP`;
   - one-token smoke run;
   - compute-sanitizer output.

## Priority Todo List

1. Build on the target Linux/CUDA/uGDS box.
2. Generate and inspect real `runtime-index.blok`.
3. Strengthen runtime contract validation for all tensors and all experts.
4. Add tokenizer parity against official tokenizer files.
5. Add first-token logits parity against a known-good reference.
6. Fix attention/MLA if the parity trace confirms the current flattened score path is wrong.
7. Verify INT4 dequantization and routed expert matvec.
8. Implement exact RoPE/YaRN behavior from official config.
9. Replace misleading output fields with measured or actual values.
10. Only after correctness is proven, add caching/performance work.

## Review Position

The architecture is cohesive enough to attempt a first target build and one-token smoke run. It is not yet proven as a correct Kimi K2.6 inference runtime. The right next step is not more boilerplate. The right next step is evidence: build logs, model contract validation, tokenizer parity, attention/MLA parity, routed expert parity, and first-token logits parity.

## High Level Feedback - where we are to our goal

From the perspective of `end_goal_prompt.py`, the goal is not "compile some kernels" or "have a plausible model architecture." The goal is:

```text
load Kimi K2.6 -> run a real prompt -> get the correct text -> meet latency/TPS/power/planning assertions
```

The current codebase is partway through building the path to that goal, but it is not at the goal. The strongest thing the repo currently has is a connected skeleton:

```text
end_goal_prompt.py
  -> blok.runtime.KimiThread.run()
  -> validate model directory/config/shard count
  -> find manifest.blok
  -> target/{release,debug}/blok generate
  -> build/blok-kimi-exec
  -> CUDA decode loop
```

That is the right product path. It is not dead code. But from an AI engineer's perspective, there are still several missing proof points before this can be called "we can load a model and run full passes."

### What `end_goal_prompt.py` Actually Requires

`end_goal_prompt.py` asserts all of this:

```python
assert response.text.asstr() == "paris"
assert response.ttft < 5.0
assert response.min_tps > 5.0
assert response.max_tps > 5.0
assert response.power.low()
assert response.plan.predicted()
```

These assertions imply six separate capabilities:

1. The model directory resolves without manual edits.
2. The complete Kimi K2.6 checkpoint is present.
3. The native runtime can load the checkpoint payloads.
4. The forward pass is numerically correct enough to answer "paris".
5. The runtime is fast enough for the latency/TPS thresholds.
6. The runtime produces truthful power and planning metadata.

Right now, only the first part of this pipeline is partially implemented. The correctness, speed, power, and planning claims are not yet proven.

### Current End-To-End State

What exists:

- `end_goal_prompt.py` calls the intended public Python API.
- `blok/runtime.py` resolves a model directory, validates `config.json`, checks safetensor shard count, finds `manifest.blok`, launches the Rust binary, parses JSON, and normalizes the text answer.
- The Rust binary parses CLI arguments and launches the native executor.
- The CUDA executor attempts a complete token-by-token forward path over all 61 layers.

What does not yet exist as proven behavior:

- A local or target run showing `end_goal_prompt.py` passes.
- A local or target run showing `blok.runtime.KimiThread.run()` returns real generated text from Kimi.
- A built `build/blok-kimi-exec` in the current checkout.
- A generated `BLOK_UGDS_MAP` path.
- A known-good first-token logits match.
- A tokenizer/chat-template match.
- A layer-level attention/MLA parity match.
- A performance measurement that supports `ttft < 5.0` or `tps > 5.0`.
- A real power measurement.
- A real planning prediction.

### The Top-Level Runtime Currently Overstates Some Results

`blok/runtime.py` computes:

```python
elapsed = max(time.perf_counter() - started, 1.0e-9)
tps = max(1.0, float(report["tokens"])) / elapsed
```

This uses the executor's reported `tokens` field. The executor currently reports requested tokens, not actual generated tokens. If EOS stops early or if the executor changes its behavior, the Python TPS can be wrong.

`PowerReport.low()` currently returns true if watts are missing:

```python
return self.watts is None or self.watts <= 250.0
```

That means `assert response.power.low()` can pass with no power measurement.

`PlanReport.predicted()` currently returns true if `predicted_tokens_per_second` is not `None`. The executor emits `predicted_tps: 0`, so this assertion can pass even though no real prediction exists.

Why this matters:

- The top-level goal file can pass metadata assertions without real power or planning.
- It cannot honestly validate performance until token counts and timing are measured correctly.
- It cannot honestly validate power until watts come from a real measurement source.

Todo:

- Change `power.low()` so missing watts does not count as a successful low-power measurement, unless the product explicitly defines "unknown" as acceptable.
- Change `PlanReport.predicted()` so `0` does not count as a useful prediction.
- Emit actual generated token count from the executor.
- Track TTFT separately from total generation time.
- Track decode TPS from actual generated tokens and decode duration.

### Model Loading Is Not Yet A Product-Quality Load Path

The current Python layer checks for safetensor shards and a manifest. That is useful, but "model load" for this project needs a stronger contract:

```text
config.json
tokenizer.json
tokenizer_config.json
all safetensor shards
manifest.blok
runtime-index.blok
tokenizer.blok
uGDS map or O_DIRECT-readable files
native executor binary
compatible CUDA/GPU/uGDS environment
```

Right now these are distributed across scripts, docs, runtime checks, and environment variables. From `end_goal_prompt.py`, a user sees only:

```python
model_dir="<kimi k2 model directory>"
```

That abstraction is fine, but the runtime has to either fully prepare the model or fail with one precise next action.

Todo:

- Add a single model readiness check that reports all missing prerequisites at once.
- Include:
  - missing checkpoint files;
  - missing tokenizer files;
  - missing materialized files;
  - missing Rust binary;
  - missing CUDA executor;
  - missing `BLOK_UGDS_DEVICE`;
  - missing `BLOK_UGDS_MAP`;
  - unsupported config values.
- Make `end_goal_prompt.py` fail with a setup diagnosis, not the first incidental missing file.

### A "Full Pass" Needs Two Definitions

There are two different meanings of "full pass" here:

1. Mechanical full pass:
   - the code enters all 61 layers;
   - reads tensors;
   - executes kernels;
   - emits a token.

2. Correct model full pass:
   - tokenization matches the official tokenizer;
   - every layer's math matches the official implementation within tolerance;
   - final logits match a reference;
   - greedy token matches a reference.

The CUDA executor is aiming at the first definition. The product goal needs the second definition.

Todo:

- Keep the mechanical smoke test, but label it honestly as `smoke`.
- Add a separate correctness test:

```text
given model revision R and prompt P:
  official tokenizer ids == blok tokenizer ids
  official first-token top-k logits ~= blok first-token top-k logits
  official greedy token == blok greedy token
```

Until that exists, "full pass" should mean "attempted mechanical pass," not "correct model inference."

### The `paris` Assertion Is Useful But Too Weak Alone

The prompt asks:

```text
Answer this question in one word. Do not use capitals or punctuation.
What is the capital of France? Enclose your response in these braces: <>
```

Then the runtime normalizes the answer by stripping braces, lowercasing, and removing punctuation.

This is a good end-to-end sanity check, but it is not enough to validate the model. A broken model path, wrong tokenizer, wrong chat template, or wrong attention can still sometimes emit "paris" by luck, bias, or prompt leakage if the model is partially functional.

Todo:

- Keep the `paris` test as a product smoke test.
- Add deterministic first-token tests for model correctness.
- Add multiple prompts whose expected outputs cannot be guessed from simple heuristics.
- Add tests that validate raw generated token ids before text normalization.

### What An AI Engineer Would Verify Next

The next useful milestone is not "more architecture." It is a target-machine proof ladder:

1. `setup-check`:
   - confirms CUDA, GPU, uGDS, model files, materialized files, and binaries.

2. `model-load-check`:
   - opens every required file;
   - parses config/tokenizer/runtime-index;
   - validates all expected tensor names, shapes, dtypes, offsets, and alignments.

3. `tokenizer-check`:
   - compares local token ids to official tokenizer ids for fixed prompts.

4. `single-layer-check`:
   - runs one token through one layer and compares intermediate tensors.

5. `first-token-check`:
   - runs full prefill and compares first-token logits/top-k against reference.

6. `decode-smoke`:
   - generates one to ten tokens and verifies the path does not crash.

7. `end-goal-smoke`:
   - runs `end_goal_prompt.py`.

8. `performance-check`:
   - measures TTFT, decode TPS, bytes read, GPU utilization, and power.

This sequence keeps the codebase small and avoids adding placeholder systems. Each check proves one real thing.

### High-Level Gap List From Product Goal

The current codebase is missing:

- A one-command readiness check for `end_goal_prompt.py`.
- A target-built CUDA executor.
- A generated and validated uGDS map.
- A verified model materialization artifact set.
- Official tokenizer parity.
- Official chat-template decision and implementation.
- First-token logits parity.
- Layer-level attention/MLA parity.
- INT4 routed expert parity.
- Accurate actual-token counting.
- Real TTFT versus total-time measurement.
- Real TPS measurement.
- Real power measurement or honest omission.
- Real planning prediction or honest omission.
- Tests that exercise the Python -> Rust -> CUDA boundary.

### Recommended Next Implementation Pass

Do the smallest work that makes the end goal falsifiable:

1. Add `blok.runtime.check_ready(model_dir)` or a CLI equivalent that reports every missing prerequisite.
2. Add a `--dry-run-contract` or `check-model` command that parses `runtime-index.blok` and validates all tensor contracts without launching generation.
3. Add tokenizer parity fixtures from the official tokenizer.
4. Add an executor debug mode that can emit first-token top-k logits.
5. Add a reference script that captures official first-token top-k logits for the exact same model revision.
6. Make `end_goal_prompt.py` depend on measured values only:
   - actual output text;
   - actual generated token count;
   - actual TTFT;
   - actual TPS;
   - actual watts if available.

After those are in place, failures will point to a specific layer of the system instead of collapsing into "generation failed" or, worse, passing with unproven metadata.

### High-Level Position

The repo is pointed in the right direction: one product path, one model target, one native executor, no generic framework detour. But from the perspective of actually loading a model and running full passes, it is still pre-validation. The current system is best described as:

```text
architecture wired, CUDA forward attempted, correctness and product readiness unproven
```

The next milestone should be:

```text
target machine can run one prompt, produce one token, and match official first-token logits/top-k
```

Only after that should the project spend serious effort on caching, batching, uGDS async paths, performance tuning, or richer API surface.

## Repo-Grounded Claim Audit Addendum

Audit date: 2026-07-15.

This addendum records the current-source verification for the material factual and technical claims in this file. Labels are intentionally limited to:

- VERIFIED IN REPO
- CONTRADICTED BY REPO
- PLAUSIBLE BUT UNPROVEN
- EXTERNAL CLAIM SUPPORTED
- EXTERNAL CLAIM NOT VERIFIED

### Read-Only Checks Run

Commands run from the repository root:

```sh
git status --short
git log --oneline -5
rg --files
rg -n "tokenizer|first-token|first token|logits|parity|MLA|attention|BLOK_UGDS|UGDS_MAP|paris|PowerReport|PlanReport|predicted_tps|watts|tokens"
find build -maxdepth 2 -type f -name 'blok-kimi-exec' -print
find . -maxdepth 4 -type f -name 'manifest.blok' -o -name 'runtime-index.blok' -o -name 'tokenizer.blok' -o -name '*ugds*map*' -o -name '*UGDS*MAP*'
find . -maxdepth 4 -type f -path '*test*' -print
find . -maxdepth 3 -type f -name '*parity*' -print
which ninja
which nvcc
cargo build
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features -- --list
target/debug/blok --help
target/debug/blok generate --model Cargo.toml --prompt hello --tokens 1
cmake --build build
```

Observed current state:

- `git status --short` was clean.
- `cargo build` passed.
- `cargo clippy --all-targets --all-features -- -D warnings` passed.
- `cargo test --all-features -- --list` reported zero Rust tests and zero doctests.
- `target/debug/blok --help` printed `usage: blok generate --model <manifest-or-dir> --prompt <text> --tokens <n>`.
- `target/debug/blok generate --model Cargo.toml --prompt hello --tokens 1` failed at manifest parsing with `manifest error: line 1 is unknown`.
- `find build ... blok-kimi-exec` found no executor artifact.
- `which ninja` and `which nvcc` failed.
- `cmake --build build` failed. The direct output was terse (`permission denied`), and `build/CMakeCache.txt` records `CMAKE_GENERATOR:INTERNAL=Ninja` with `CMAKE_MAKE_PROGRAM:FILEPATH=CMAKE_MAKE_PROGRAM-NOTFOUND`, which supports the missing-Ninja explanation.

### External Sources Consulted

External paper-search was used only for methodological claims about attention correctness and inference benchmarking.

- Vaswani et al., "Attention Is All You Need", arXiv:1706.03762. Used to support the general claim that standard multi-head attention runs separate attention functions per head, then concatenates/project outputs. This supports the concern that one flattened shared softmax is suspicious, but it does not by itself prove Kimi MLA semantics.
- Reddi et al., "MLPerf Inference Benchmark", arXiv:1911.02549. Used to support the general claim that inference benchmarking requires reproducible latency/throughput measurement methodology.
- Banbury et al., "MLPerf Tiny Benchmark", arXiv:2106.07597. Used to support the general claim that low-power inference claims require measured accuracy, latency, and energy/power-style evidence.

### Claim-By-Claim Verification

| Claim | Label | Repo/source evidence |
|---|---|---|
| The native CUDA executor is wired to run an autoregressive path: tokenize prompt, read tensor payloads, run a 61-layer decode loop, run attention/MLP/MoE kernels, project through LM head, choose argmax, and detokenize. | VERIFIED IN REPO | `src/kimi_exec.cu:115-120`, `src/kimi_exec.cu:660-683`, `src/kimi_exec.cu:687-775`, `src/kimi_exec.cu:778-783`. |
| The repo does not prove a correct full Kimi K2.6 forward pass. | VERIFIED IN REPO | `README.md:3-5`, `README.md:32-39`, `scripts/ci.sh:222-234`, `docs/kimi-forward-ugds-status.md:87-103`. |
| Local Rust build, clippy, help, fake-manifest behavior, and zero-test state are as reported. | VERIFIED IN REPO | Re-run commands above; zero tests from `cargo test --all-features -- --list`; CLI implementation in `src/blok_runtime_library.rs:20-34`, `src/blok_runtime_library.rs:81-100`. |
| CUDA executor artifact `build/blok-kimi-exec` is absent in the current checkout. | VERIFIED IN REPO | `find build -maxdepth 2 -type f -name 'blok-kimi-exec' -print` returned no paths. |
| `ninja` and `nvcc` are not installed locally. | VERIFIED IN REPO | `which ninja` and `which nvcc` both failed. `build/CMakeCache.txt:23-24` records missing CMake make program. |
| `cmake --build build` failed because the configured Ninja build tool is missing. | VERIFIED IN REPO | Direct `cmake --build build` output was `permission denied`; `build/CMakeCache.txt:23-24` records `CMAKE_MAKE_PROGRAM-NOTFOUND`, and `build/CMakeCache.txt:69` records generator `Ninja`. |
| The intended product path is Python runtime or CLI to Rust `blok generate` to `src/kimi_runtime.rs` to `build/blok-kimi-exec` to `src/kimi_exec.cu` and JSON output. | VERIFIED IN REPO | `blok/runtime.py:200-227`, `src/blok_runtime_library.rs:81-100`, `src/kimi_runtime.rs:79-120`, `src/kimi_exec.cu:786-798`. |
| The Rust layer is a launcher and parser; native model execution is in `src/kimi_exec.cu`. | VERIFIED IN REPO | `src/kimi_runtime.rs:79-120`, `src/kimi_runtime.rs:123-136`, `src/kimi_exec.cu:660-798`. |
| Rust success can coexist with missing CUDA executor. | VERIFIED IN REPO | `src/kimi_runtime.rs:89-92` returns `blok_kimi_exec_binary_required` if the executor binary is absent. |
| `src/kimi_exec.cu` defines tokenizer load/encode/decode, runtime index parsing, tensor loading, contract validation, decode loop, and layer loop. | VERIFIED IN REPO | `src/kimi_exec.cu:405-455`, `src/kimi_exec.cu:457-481`, `src/kimi_exec.cu:562-585`, `src/kimi_exec.cu:593-634`, `src/kimi_exec.cu:660-783`, `src/kimi_exec.cu:684-775`. |
| The executor loads embeddings, final norm, and LM head once before per-token/layer execution. | VERIFIED IN REPO | `src/kimi_exec.cu:665-667`. |
| The executor performs a loop over all 61 layers. | VERIFIED IN REPO | `src/kimi_exec.cu:115`, `src/kimi_exec.cu:687`. |
| `CMAKE_CUDA_ARCHITECTURES` is set to `120`. | VERIFIED IN REPO | `CMakeLists.txt:8`. |
| Attention currently computes one scalar score per timestep by dotting flattened query/key state and uses one shared softmax over sequence positions. | VERIFIED IN REPO | `src/kimi_exec.cu:241-253`, `src/kimi_exec.cu:255-274`, `src/kimi_exec.cu:709-711`. |
| Standard multi-head attention usually computes separate per-head attention distributions. | EXTERNAL CLAIM SUPPORTED | Vaswani et al., arXiv:1706.03762, section 3.2.2 defines separate `head_i = Attention(...)` functions run in parallel and concatenated. |
| The flattened attention path is likely wrong for Kimi MLA. | PLAUSIBLE BUT UNPROVEN | Repo evidence proves flattened behavior, but no official Kimi MLA trace/config implementation is present locally to conclusively prove divergence. See `docs/specs/kimi_k2_forward.yaml:79-88` for required missing MLA parity. |
| KV layout assumes key-nope values occupy the first contiguous projected segment and values occupy the next segment. | VERIFIED IN REPO | `src/kimi_exec.cu:703-705`. |
| Actual Kimi KV layout may differ and needs shape/layout verification. | PLAUSIBLE BUT UNPROVEN | No complete real `runtime-index.blok`, safetensor headers, or official Kimi attention projection trace were found in repo. |
| RoPE uses `ROPE_THETA = 50000.0f` and a simple frequency formula. | VERIFIED IN REPO | `src/kimi_exec.cu:119`, `src/kimi_exec.cu:217-228`. |
| YaRN long-context scaling is not implemented. | VERIFIED IN REPO | `docs/kimi-forward-ugds-status.md:92`, `docs/specs/kimi_k2_forward.yaml:34-40`. |
| The executor builds a custom tokenizer from `tokenizer.blok` and tokenizes raw prompt bytes. | VERIFIED IN REPO | `src/kimi_exec.cu:405-455`, especially `src/kimi_exec.cu:425-427`. |
| The executor does not apply a chat template. | VERIFIED IN REPO | No chat-template application exists in `src/kimi_exec.cu`; tokenizer config/chat template is only referenced as required/missing in `docs/specs/kimi_k2_forward.yaml:57-78`. |
| The executor only handles one EOS id, `EOS_IM_END = 163586`. | VERIFIED IN REPO | `src/kimi_exec.cu:120`, `src/kimi_exec.cu:778`. |
| Tokenizer parity tests, first-token-logits parity tests, and layer-level attention/MLA parity tests do not exist. | VERIFIED IN REPO | `find ... '*parity*'` found no files; `scripts/ci.sh:222-225` hard-fails `verify-l3`; `README.md:34-35` marks tokenizer/chat-template and MLA/YaRN parity unverified. |
| `validate_forward_contract` is shallow and spot-checks routed expert 0. | VERIFIED IN REPO | CUDA executor contract checks broad slots and expert 0 only at `src/kimi_exec.cu:593-634`, especially `src/kimi_exec.cu:620-625`. |
| A separate model-side contract script validates all 384 routed experts for all MoE layers. | VERIFIED IN REPO | `scripts/check_kimi_contract.py:46-56`. This does not contradict the executor spot-check claim; it means stronger validation exists as a separate script when `verify-l1` is run with `BLOK_MODEL`. |
| INT4 routed expert path assumes packed symmetric INT4 with `q = nibble - 8`, group size 32, and 8 values per `i32`. | VERIFIED IN REPO | `src/kimi_exec.cu:167-180`, `src/kimi_exec.cu:192-205`, `src/kimi_exec.cu:555-560`; local spec mirrors it at `docs/specs/kimi_k2_forward.yaml:139-161`. |
| The actual checkpoint quantization convention needs checkpoint-specific validation. | EXTERNAL CLAIM NOT VERIFIED | The repo has a local spec, but no official quantization document, official checkpoint dequantization script, or reference matvec result was found. |
| Executor reloads many tensors inside every layer and token. | VERIFIED IN REPO | Tensor loads are inside the `run` lambda and layer loop: `src/kimi_exec.cu:684-695`, `src/kimi_exec.cu:714-728`, `src/kimi_exec.cu:737-758`. |
| Executor prints `"tokens": a.tokens`, `"predicted_tps": 0`, and `"watts": null`. | VERIFIED IN REPO | `src/kimi_exec.cu:797-798`. |
| Reported `tokens` are requested tokens, not actual generated tokens. | VERIFIED IN REPO | Generation can stop before `a.tokens` at `src/kimi_exec.cu:778-781`, but output still reports `a.tokens` at `src/kimi_exec.cu:798`. |
| `scripts/ci.sh` verification levels are honest but incomplete; `verify-l3` and `verify-l5` currently fail because fixtures are missing. | VERIFIED IN REPO | `scripts/ci.sh:210-234`; README also documents this at `README.md:41-50`. |
| `end_goal_prompt.py` asserts text, TTFT, TPS, power, and plan. | VERIFIED IN REPO | `end_goal_prompt.py:20-25`. |
| `end_goal_prompt.py` asks for Kimi K2 via public Python API and expects `"paris"`. | VERIFIED IN REPO | `end_goal_prompt.py:5-20`. |
| `blok/runtime.py` resolves a model directory, validates `config.json`, checks safetensor shard count, finds `manifest.blok`, launches Rust, parses JSON, and normalizes text. | VERIFIED IN REPO | `blok/runtime.py:88-105`, `blok/runtime.py:142-161`, `blok/runtime.py:164-177`, `blok/runtime.py:179-197`, `blok/runtime.py:200-227`, `blok/runtime.py:230-234`. |
| Python TPS uses executor-reported `tokens`. | VERIFIED IN REPO | `blok/runtime.py:93-104`, especially `blok/runtime.py:97`. |
| `PowerReport.low()` returns true if watts are missing. | VERIFIED IN REPO | `blok/runtime.py:53-59`. |
| `PlanReport.predicted()` returns true if `predicted_tokens_per_second` is not `None`; `predicted_tps: 0` therefore passes. | VERIFIED IN REPO | `blok/runtime.py:61-68`, `blok/runtime.py:103-104`, `src/kimi_exec.cu:798`. |
| The model-load path is not product-quality because readiness is distributed across runtime checks, scripts, docs, env vars, and artifacts. | VERIFIED IN REPO | Python checks config/shards/manifest/Rust binary only: `blok/runtime.py:89-92`, `blok/runtime.py:200-205`; CUDA executor checks `BLOK_UGDS_DEVICE` and `BLOK_UGDS_MAP` only when compiled with uGDS: `src/kimi_exec.cu:48-63`; hardware script checks target environment separately: `scripts/check_hardware.sh:6-20`. |
| `BLOK_UGDS_DEVICE` and `BLOK_UGDS_MAP` are required when the executor is compiled with uGDS. | VERIFIED IN REPO | `src/kimi_exec.cu:44-72`, especially `src/kimi_exec.cu:49-50`; documented at `docs/kimi-forward-ugds-status.md:55-57`. |
| `BLOK_UGDS_MAP` is not generated by the repo. | VERIFIED IN REPO | No map artifact found; `docs/kimi-forward-ugds-status.md:91` states it directly; `docs/kimi-forward-ugds-status.md:120-126` describes manual map requirements. |
| `scripts/model_fetch.py` materializes `manifest.blok`, `runtime-index.blok`, and `tokenizer.blok`. | VERIFIED IN REPO | `scripts/model_fetch.py:112-138`, especially `scripts/model_fetch.py:122-127` and `scripts/model_fetch.py:135-137`. |
| A mechanical full pass and a numerically correct model full pass are different definitions. | VERIFIED IN REPO | Repo capability matrix distinguishes wired/unverified decode from unverified tokenizer/MLA/INT4 paths at `README.md:28-39`; specs require parity artifacts at `docs/specs/kimi_k2_forward.yaml:168-202`. |
| The `"paris"` assertion is useful but insufficient as a correctness test. | EXTERNAL CLAIM SUPPORTED | Repo normalizes output text at `blok/runtime.py:230-234`, and no raw token-id or first-token-logit parity tests exist. Standard validation practice for model/runtime correctness requires token/logit or numerical parity evidence, not a single normalized answer. |
| Inference latency/TPS/power claims require measured evidence. | EXTERNAL CLAIM SUPPORTED | MLPerf Inference and MLPerf Tiny papers support reproducible measurement of latency/throughput and energy/power-related metrics; repo currently emits placeholder `predicted_tps` and `watts` at `src/kimi_exec.cu:798`. |

### Contradictions Found

No material claim in this Markdown was directly contradicted by the current repository.

Two caveats should be addressed in the next edit:

1. The CMake failure wording should mention the exact current observation: `cmake --build build` prints `permission denied`, while `build/CMakeCache.txt` confirms the configured Ninja build tool is missing.
2. There is a repo-internal model naming/spec inconsistency outside this Markdown: `scripts/model_fetch.py:5`, `README.md:3`, and this document target `moonshotai/Kimi-K2.6`, but `docs/specs/kimi_k2_forward.yaml:2` says `moonshotai/Kimi-K2-Thinking`. The spec should be reconciled before treating it as authoritative source material for Kimi K2.6.
