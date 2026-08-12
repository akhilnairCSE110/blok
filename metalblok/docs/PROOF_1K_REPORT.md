# MetalBlok 1,000-input / 1,000-output proof report

## Final result

**PASS — native execution completed normally on 2026-08-12.**

The run did not crash. It stopped because it reached the requested maximum
token count. The final native log says:

```text
[metalblok] gguf: SUMMARY prefill 6 tok in 0.79s (7.563 tok/s, includes TTFT) |
decode 719 steps/720 emitted in 3016.18s (0.238 step/s) | stopped: max-tokens
```

The `6 tok` prefill in that line is the harmless positional placeholder passed
to a continuation command; the existing 1,000-token state was loaded instead
of prefilling those six IDs. The original prefill log independently records
exactly 1,000 native prompt tokens.

The authoritative state header is:

| Field | Final value |
|---|---:|
| magic | `MBLKSTAT` |
| state version | 3 |
| layers | 61 |
| context capacity | 2,048 |
| KV latent metadata rank | 512 |
| RoPE width | 64 |
| committed position | **1,999** |
| pending token | 343 |

The file is exactly

\[
36+1{,}999(4{,}005{,}504)=8{,}007{,}002{,}532\text{ bytes},
\]

matching its on-disk size byte for byte. Both wrapper and native child exited;
there is no inference process left to kill.

## 1. Model, machine, and decoding contract

| Property | Value |
|---|---|
| model | DeepSeek-R1 671B `UD-IQ1_S` |
| checkpoint | three GGUF shards, 140,231,438,464 stored bytes |
| tensors | 1,025 |
| target | 10-core Apple M5, 24 GB unified memory, internal NVMe |
| inference engine | native MetalBlok C++/Objective-C++/Metal |
| external engine | none; no llama.cpp, MLX, PyTorch, or server process |
| context allocated | 2,048 positions |
| sampling | greedy, temperature 0 |
| requested input | exactly 1,000 native formatted tokens |
| requested output | exactly 1,000 emitted tokens |
| EOS behavior | no EOS; stopped at the exact output cap |

## 2. The exact prompt

The acceptance harness did not estimate prompt length from words or another
tokenizer. It repeatedly called MetalBlok's native DeepSeek tokenizer until the
normally formatted chat prompt contained exactly 1,000 IDs, including BOS and
the DeepSeek user/assistant/thinking markers.

The visible prompt contains 6,119 ASCII characters / bytes. Its exact compact
definition is:

```text
Write a complete, production-quality Python 3 program named jsonl_report.py.
It must use only the standard library, stream an arbitrarily large JSON Lines input file without loading it into memory, reject malformed records with line-numbered diagnostics, group valid records by a required string field named category, and compute count, sum, minimum, maximum, and numerically stable mean for a required finite numeric field named value. Add argparse options for input path, output path, and strict mode. Emit deterministic UTF-8 JSON with sorted categories and keys. Use compensated summation, explicit type checks that reject booleans as numbers, atomic output replacement, useful exit codes, type hints, docstrings, and a main guard. Include self-contained unittest cases runnable with python -m unittest, covering empty input, malformed JSON, missing fields, boolean values, non-finite values, strict mode, Unicode, deterministic ordering, and a successful multi-category file. Avoid third-party packages and network access.

Additional review constraints follow. Each repeated constraint is intentional and remains binding.
```

It then contains this exact line **30 consecutive times**:

```text
Constraint: preserve streaming memory bounds, deterministic behavior, clear errors, portable standard-library semantics, and directly test every failure branch.
```

It ends with exactly:

```text
.
Return one concise explanation followed by one complete Python code block. Do not omit tests, use placeholders, or claim behavior the code does not implement.
```

The apparently redundant constraint is deliberate padding with meaningful
requirements. After ordinary chat formatting, the native tokenizer verified:

```text
prompt 1000 tokens (bos=0 eos=1 vocab=129280)
```

The prompt constructor and exact-count algorithm are in
[`scripts/prove_metal_1k.py`](../../scripts/prove_metal_1k.py).

## 3. How the 1,000 output tokens were proven

The proof was resumed rather than recomputing the expensive prompt. State v3
stores the exact committed KV prefix and the already-predicted pending token.
The complete output index mapping is:

| Output IDs | Prediction positions | Source log | Count |
|---|---:|---|---:|
| `A0` | 999 | `run-20260812-012449-18510.log` | 1 |
| `A1..A280` | 1,000–1,279 | `run-20260812-014259-19219.log` | 280 |
| `A281..A999` | 1,280–1,998 | `run-20260812-021356-20767.log` | 719 |
| **total** | **999–1,998** | three exact chains | **1,000** |

The state relationship is:

\[
1{,}000\text{ prompt positions}+999\text{ decode advances}
=1{,}999\text{ committed positions}.
\]

Only 999 forward advances are needed for 1,000 emitted tokens because prefill
already computes `A0`. The final continuation began from state position 1,280,
emitted stored pending token `A280`, performed 719 forward advances, and
emitted through `A999`.

The complete text was reconstructed by extracting precisely those 1,000 token
IDs and passing them to the same native tokenizer's `--decode-ids` path. A
byte-for-byte `diff` against the retained complete-output artifact passed. The
result is 4,407 UTF-8 bytes, 68 lines, and 733 whitespace-delimited words.

## 4. What the model output

The exact full output is retained at
[`proof-1k-complete-output.txt`](../runs/proof-1k-complete-output.txt).

It begins:

```text
Okay, I need to write a Python program called jsonl_report.py that meets a lot
of specific requirements. Let's break down the problem step by step.
```

The generated reasoning then correctly identifies and discusses:

1. line-by-line JSONL processing rather than loading the input file;
2. line-numbered `json.loads` failures and strict-mode early termination;
3. grouping by required `category` values;
4. count, sum, minimum, maximum, and mean per category;
5. compensated/Kahan summation for numerical stability;
6. rejecting `bool` even though Python makes it an `int` subclass;
7. rejecting NaN and infinity;
8. `argparse` input/output/strict options;
9. deterministic sorted UTF-8 JSON;
10. atomic output replacement;
11. exit codes, typing, docstrings, and a main guard;
12. the requested self-contained unit-test cases.

It moves from a numbered requirement analysis into a proposed execution flow,
then into the per-category aggregate representation and Kahan update. The last
tokens are:

```text
So, for each value in the 'value' field:

- Check it's a number (int or float), not a boolean.
- Check it's finite (
```

The final parenthesis is incomplete because token 1,000 hit the requested cap.
The engine did not emit EOS and did not reach the requested Python code block.

### Output-quality conclusion

The output is coherent, on-topic, and maintains the prompt's detailed
constraints over 1,000 generated tokens. It does not exhibit the earlier
unrelated “rock paper scissors” failure. That is strong evidence that the
native forward/state/tokenizer path generates a stable model sequence.

It is **not** a semantic pass for the requested programming task: the model
spent the entire budget in DeepSeek's thinking stream, repeated part of its
Kahan-summation reasoning, and was cut off before writing executable code or
tests. The system acceptance proves exactly 1,000 output tokens; it does not
claim that a 1,000-token cap is sufficient for DeepSeek-R1 to finish this
particular long specification. A product demo should either request more
output tokens, use a shorter verifiable task, or stop/suppress the thinking
stream under a separately validated chat template.

## 5. Timing and data-movement results

### Prefill

| Metric | Result |
|---|---:|
| native input tokens | 1,000 |
| tiles | 8, maximum tile 128 |
| forward time | 591.937584 s |
| effective prefill rate | 1.689 tokens/s |
| GPU time | 352.506568 s |
| explicit I/O wait | 115.514659 s |
| streamed model bytes | 709,792,466,432 |
| logical/actual reads | 246,088 |
| command buffers | 11,863 |
| first predicted token | ID 33,001, `Okay` |
| first-token logit | 58.0137 |

### First 280 decode advances

This prefix used the earlier schedule retained by the authoritative checkpoint:

| Metric | Result |
|---|---:|
| advances | 280 |
| summed forward time | 1,079.288677 s |
| average | 3.854602 s/advance |
| GPU time | 131.594933 s |
| explicit I/O wait | 516.797114 s |
| NVMe span | 1,026.693836 s |
| model/NVMe bytes | 3,877,991,631,360 |
| aggregate effective NVMe | 3.777 GB/s |
| command buffers | 49,840 |
| hot-path allocations | 0 |

### Final optimized 719 decode advances

| Metric | Result |
|---|---:|
| advances / emitted tokens | 719 / 720 |
| native summary time, including final save | 3,016.18 s |
| summed forward-step time | 3,004.818888 s |
| average step time | 4.179164 s |
| minimum / maximum step | 3.087547 / 8.425160 s |
| summed GPU time | 415.814452 s |
| average GPU time | 0.578323 s |
| explicit I/O wait | 633.080369 s |
| summed NVMe span | 2,866.623024 s |
| model/NVMe bytes | 9,769,507,454,016 |
| aggregate effective NVMe | 3.408 GB/s |
| model bytes per step | 13,587,632,064 |
| reads per step | 1,869 |
| urgent selected-expert bytes/step | 4,035,182,592 |
| urgent selected-expert reads/step | 1,392 |
| command buffers | 127,982 = 719 × 178 |
| hot-path allocations | 0 |
| exact KV bytes added | 2,879,957,376 |

The final segment was storage-bound: average GPU work was only 0.578 seconds,
while each step consumed 13.588 GB and its aggregate read span dominated wall
time. This is why reducing/fusing small elementwise GPU kernels would have had
far less impact than model-byte reuse and storage scheduling.

### Whole accepted chain

| Metric | Result |
|---|---:|
| input / output | 1,000 / 1,000 tokens |
| decode advances | 999 |
| summed decode forward time | 4,084.107565 s |
| average decode forward time | 4.088196 s |
| prefill + decode forward time | 4,676.045149 s = 77.934 min |
| decode model bytes | 13,647,499,085,376 |
| prefill + decode model bytes | 14,357,291,551,808 = 14.357 TB |
| decode aggregate NVMe rate | 3.505 GB/s |
| exact total KV growth | 4,001,498,496 bytes |
| prefill + decode command buffers | 189,685 |
| recorded hot-path allocations | 0 |

The 77.934 minutes is summed forward work from the exact accepted segments. It
does not include human pauses between resumptions. It also excludes some model
initialization and checkpoint-write overhead; the final continuation summary
shows about 11.36 additional seconds beyond its per-step forward sum.

## 6. Memory-pressure result

The final 719-step log records host-wide VM deltas of:

| Counter | Sum |
|---|---:|
| pageouts | 100,236 |
| compressions | 420,506,654 |
| decompressions | 418,546,803 |
| swap-ins | 2,407 |
| swap-outs | 168 |

These are macOS system counters, not process-attributed events. They cannot all
be assigned to MetalBlok, but they demonstrate real host pressure as expanded
KV grows. The state stayed valid and output remained coherent; pressure
affected performance rather than numerical correctness. It also explains why
late-run throughput was lower than the best early 3.09–3.22 s interval.

## 7. Correctness evidence and boundaries

This proof establishes:

- exact 1,000-ID native prompt construction and prefill;
- all 61 layers for every committed position;
- exact grouped sigmoid top-8 MoE routing;
- F32/Q4_K/Q5_K/Q6_K/IQ2_XXS/IQ1_S kernels in the real graph;
- deterministic greedy continuation through interruptions;
- 1,999-position exact expanded FP16 K/V state;
- 1,000 reconstructed emitted IDs and coherent decoded text;
- normal maximum-token termination rather than a crash;
- complete per-step NVMe, GPU, command-buffer, allocation, KV, and VM logs.

Separate evidence compared the optimized schedule at positions 1,280–1,282
against the prior path and matched token IDs/logits. The deliberately tested
residual/RMS fusion changed an accepted logit and was removed.

This proof does not establish:

- independent full-logit equality with another DeepSeek runtime;
- completion or unit-test correctness of the requested Python program;
- EOS generation inside 1,000 output tokens;
- the model's declared 163,840-token context, let alone one million;
- zero swap pressure, concurrent serving, ANE execution, or multi-model use.

## 8. Final artifacts

| Artifact | Path |
|---|---|
| this report | `metalblok/docs/PROOF_1K_REPORT.md` |
| complete reconstructed output | `metalblok/runs/proof-1k-complete-output.txt` |
| authoritative final state | `metalblok/runs/proof-1k-20260812-012449.state` |
| exact 1,000-token prefill log | `metalblok/runs/run-20260812-012449-18510.log` |
| prefix decode log | `metalblok/runs/run-20260812-014259-19219.log` |
| final optimized continuation log | `metalblok/runs/run-20260812-021356-20767.log` |
| final 720-token stdout segment | `metalblok/runs/run-20260812-021356-20767.txt` |
| prompt/proof harness | `scripts/prove_metal_1k.py` |

The decisive acceptance tuple is:

```text
input_tokens=1000
output_tokens=1000
final_position=1999
final_state_bytes=8007002532
termination=max-tokens
native_exit=0
```
