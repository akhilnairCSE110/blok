# Blok

Minimal text-only inference for the pinned `moonshotai/Kimi-K2.6` revision `7eb5002f6aadc958aed6a9177b7ed26bb94011bb` on an RTX 5060 Ti. Python performs the exact Kimi chat-template/tokenizer work; one CUDA executable performs all 61 transformer layers, greedy decoding, and direct uGDS model/KV I/O.

## Implementation rule

Apply Musk's five-step algorithm in order:

1. Challenge every requirement and identify its owner.
2. Delete every part or process not required for a full Kimi forward pass.
3. Simplify and optimize only what survives.
4. Shorten the test and execution cycle.
5. Automate only after the path is correct.

The requirement is one public, reference-correct Kimi text-generation path. Languages, wrappers, duplicate indexes, fallback tokenizers, and generic tooling are not requirements; every surviving line must serve the forward pass. Sources: [Starbase interview](https://www.youtube.com/watch?v=t705r8ICkRw) and [transcript/excerpt](https://www.startuparchive.org/p/elon-musk-explains-his-5-step-algorithm-for-running-companies-1eae).

Forward semantics are pinned to Kimi's [configuration](https://huggingface.co/moonshotai/Kimi-K2.6/blob/7eb5002f6aadc958aed6a9177b7ed26bb94011bb/config.json), [reference implementation](https://huggingface.co/moonshotai/Kimi-K2.6/blob/7eb5002f6aadc958aed6a9177b7ed26bb94011bb/modeling_deepseek.py), and the [INT4 packing format](https://github.com/vllm-project/compressed-tensors/blob/main/src/compressed_tensors/compressors/quantized_compressors/pack_quantized.py).

## Target

- Ubuntu/Linux, Ryzen 9 5950X, 48 GB RAM
- RTX 5060 Ti 16 GB (`sm_120`), CUDA 12.8+, NVIDIA open modules
- Samsung 990 EVO Plus 1 TB bound to uGDS
- Model payload on the detachable NVMe filesystem
- Metadata, repository, binaries, and `ugds.env` on the system filesystem
- Dedicated 4 KiB-aligned raw KV range outside every filesystem and model extent

## Run

```sh
git submodule update --init --recursive
scripts/bootstrap_linux.sh

export BLOK_MODEL_ROOT=/mnt/kimi-models
export BLOK_META_ROOT="$HOME/.blok/metadata"
export BLOK_UGDS_DEVICE=/dev/ugds_drv0
export BLOK_UGDS_MAP="$PWD/ugds-map.blok"
export BLOK_UGDS_ENV_OUTPUT="$PWD/ugds.env"
export BLOK_KV_UGDS_BASE=<reserved-raw-byte-offset>
export BLOK_KV_UGDS_BYTES=<reserved-raw-byte-count>
export BLOK_UGDS_PHYSICAL_OFFSET_ADD=<partition-start-or-zero>

scripts/model_fetch.py kimi-k2.6 fetch
scripts/target_v0.sh prepare
```

After `prepare`, bind only the verified, now-unmounted model NVMe controller and run the complete public-path smoke:

```sh
sub_dir/uGDS/scripts/env_switch.sh ugds <verified-pci-slot>
scripts/target_v0.sh run
```

Success is `{"status": "ok", "text": "paris"}`. Never use filesystem free space for raw KV writes, and regenerate the uGDS map after any shard movement or rewrite.
