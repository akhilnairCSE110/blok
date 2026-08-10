# MetalBlok

MetalBlok is a bounded, native Metal inference runtime for the local
DeepSeek-R1 671B `UD-IQ1_S` GGUF checkpoint. Correctness and host safety are
release gates; caching, overlap, and kernel tuning come afterward.

The initial code is derived from the adjacent Blade prototype. Imported code
is treated as untrusted until it passes MetalBlok's real-model gates.

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
