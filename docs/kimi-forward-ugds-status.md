# Kimi Forward Over uGDS

Goal path:

```text
end_goal_prompt.py -> blok.runtime -> blok generate -> blok-kimi-exec -> uGDS -> CUDA kernels
```

Target box:

- GPU: NVIDIA RTX 5060 Ti, GB206, `sm_120`.
- CPU: AMD Ryzen 9 5950X.
- RAM: 48 GB.
- Primary uGDS storage: Samsung 990 EVO Plus 1TB NVMe, PCIe 4.0 x4 / 5.0 x2.
- Excluded storage: Kingston SA400S37240G 240GB SATA SSD; Seagate ST2000DM008-2FR102 2TB SATA HDD.
- Board: MSI MAG X870 Tomahawk.

Runtime contract:

- Model: `moonshotai/Kimi-K2.6`, text-only.
- Materialized files: `manifest.blok`, `runtime-index.blok`, `tokenizer.blok`.
- Model reads: `BLOK_UGDS_DEVICE` + `BLOK_UGDS_MAP`.
- KV scratch: `BLOK_KV_UGDS_BASE`, optional `BLOK_KV_UGDS_BYTES`.
- Executor: `build/blok-kimi-exec`, launched by Rust.

Hardware consequence:

- The Kimi K2.6 download is roughly 595 GB, so the 1TB Samsung NVMe must hold model shards plus a deliberately bounded KV scratch range.
- Do not plan uGDS model reads or KV scratch on the SATA SSD/HDD.
- 48 GB RAM is not enough for whole-model staging; V0 must stream tensors from NVMe/uGDS.

Implemented:

- Manifest/runtime-index/tokenizer sidecars.
- Rust launcher and JSON parser.
- uGDS-only CUDA executor.
- Extent-aware model shard map generation through Linux FIEMAP.
- Executor reads tensor slices across mapped file extents instead of assuming contiguous shards.
- Batch-1 greedy prefill/decode loop over 61 layers.
- RMSNorm, RoPE, bf16 matvec, attention, router top-k, shared expert, routed INT4 expert, residual, LM head, argmax.
- Kimi dense layer 0 and routed top-8 MoE layers 1-60.
- Generated-token count in executor JSON; `predicted_tps` and `watts` are `null`.

Not proven:

- Target CUDA/uGDS compile and run.
- Target validation of generated `BLOK_UGDS_MAP`.
- KV scratch placement safety.
- Official tokenizer/chat-template parity.
- MLA, RoPE/YaRN, first-token logits, and routed INT4 numerical parity.
- Performance, power, or production sampling.

Target smoke:

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
cargo build --release

BLOK_UGDS_DEVICE=/dev/ugds_drv0 \
BLOK_UGDS_MAP=/path/to/ugds-map.txt \
BLOK_KV_UGDS_BASE=<scratch_byte_offset> \
BLOK_KV_UGDS_BYTES=<scratch_bytes> \
BLOK_KIMI_EXEC_BIN=build/blok-kimi-exec \
target/release/blok generate --model /path/to/manifest.blok --prompt "hello" --tokens 1
```

Next:

1. Build uGDS and executor on the target box.
2. Generate the extent-aware model shard map and non-overlapping KV scratch range:
   ```sh
   BLOK_MODEL=/path/to/manifest.blok \
   BLOK_UGDS_DEVICE=/dev/ugds_drv0 \
   BLOK_KV_UGDS_BASE=<scratch_byte_offset> \
   BLOK_KV_UGDS_BYTES=<scratch_bytes> \
   BLOK_UGDS_MAP=/path/to/ugds-map.blok \
   BLOK_UGDS_ENV_OUTPUT=/path/to/ugds.env \
   just ugds-layout
   ```
3. Run one-token smoke.
4. Patch real tensor shape/name/runtime issues.
5. Add tokenizer, first-logit, MLA, YaRN, and INT4 parity checks.
