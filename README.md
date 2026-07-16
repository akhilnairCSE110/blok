# Blok

Experimental, text-only, Kimi K2.6-specific native inference runtime. It is not yet numerically validated or target-proven.

## Target

- GPU: NVIDIA RTX 5060 Ti, GB206, `sm_120`.
- CPU: AMD Ryzen 9 5950X.
- Storage: Samsung 9100 Pro NVMe, PCIe 5.0 x4.
- Board: MSI MAG X870 Tomahawk.
- OS/I/O: Ubuntu/Linux bare metal, uGDS-owned NVMe to registered CUDA buffers.

```sh
python3 end_goal_prompt.py
```

```text
blok.runtime -> target/{debug,release}/blok generate -> build/blok-kimi-exec
```

The native executor owns manifest parsing, uGDS payload movement, CUDA kernels, and token emission. vLLM/Transformers are not product-path dependencies.

## Required State

- Complete `moonshotai/Kimi-K2.6` download.
- `scripts/model_fetch.py kimi-k2.6 materialize`.
- CUDA/CMake build producing `build/blok-kimi-exec`.
- uGDS driver/library for the target kernel and NVIDIA open driver.
- `BLOK_UGDS_DEVICE`, `BLOK_UGDS_MAP`, `BLOK_KV_UGDS_BASE`, optional `BLOK_KV_UGDS_BYTES`.

## Status

| Capability | Status |
|---|---|
| Safetensor materialization/index | Implemented |
| Text prefill/decode path | Wired, unverified |
| Routed INT4 path | Wired, unverified |
| uGDS execution | Wired, unproven |
| Tokenizer/chat-template, MLA/YaRN, logits parity | Missing |
| Image/video, sampling, metrics | Not implemented |

## Docs

- [System requirements](docs/system-requirements.md)
- [Kimi over uGDS status](docs/kimi-forward-ugds-status.md)
- [Forward contract](docs/specs/kimi_k2_forward.yaml)
- [Research index](docs/research/papers.md)

## Verification

| Level | Gate | Command |
|---|---|---|
| 0 | Static | `just verify-l0` |
| 1 | Model contract | `BLOK_MODEL=/path/to/manifest.blok just verify-l1` |
| 2 | CUDA build | `just verify-l2` |
| 3 | Official parity | `just verify-l3` currently missing |
| 4 | Target hardware | `just verify-l4` plus smoke |
| 5 | Regression fixtures | `just verify-l5` currently missing |
