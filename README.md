# Blok

Blok is an experimental, text-only, Kimi K2.6-specific native inference runtime. It is
not yet numerically validated against the official implementation or proven on its
intended Linux/uGDS hardware.

Path:

```sh
python3 end_goal_prompt.py
```

That calls:

```text
blok.runtime -> target/{debug,release}/blok generate -> build/blok-kimi-exec
```

The native executor owns the model path: manifest parsing, uGDS payload movement, CUDA
kernels, and token emission. vLLM/Transformers are not part of the product path.

Required external state:

- complete `moonshotai/Kimi-K2.6` download;
- `scripts/model_fetch.py kimi-k2.6 materialize`;
- CUDA/CMake build producing `build/blok-kimi-exec`.

## Capability Matrix

| Capability | Status |
|---|---|
| Safetensor materialization and tensor index | Implemented |
| Text prefill/decode path | Wired, unverified |
| Official tokenizer/chat-template parity | Unverified |
| MLA/YaRN long-context correctness | Unverified |
| Routed INT4 path | Wired, unverified |
| uGDS execution | Wired, unproven |
| Image/video inputs | Not implemented |
| Sampling and production metrics | Not implemented |

## Verification Levels

| Level | Gate | Command |
|---|---|---|
| 0 | Static checks | `just verify-l0` |
| 1 | Model header/index contract | `BLOK_MODEL=/path/to/manifest.blok just verify-l1` |
| 2 | CUDA build/sanitizer availability | `just verify-l2` |
| 3 | End-to-end official parity | `just verify-l3` fails until fixtures exist |
| 4 | Target uGDS hardware run | `just verify-l4` plus one-token smoke |
| 5 | Revision-keyed regression fixtures | `just verify-l5` fails until fixtures exist |
