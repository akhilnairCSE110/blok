# Blok

Blok's active V0 is MetalBlok: a native Apple-Silicon inference engine for the
exact three-shard DeepSeek-R1 671B `UD-IQ1_S` GGUF. It streams the model from
SSD through reusable unified-memory slabs and executes the complete 61-layer
forward pass with custom Metal kernels. There is no llama.cpp, MLX, PyTorch,
or server process in the inference path.

## Run

On the target Mac, from this directory:

```sh
./run_blok.py "Write a correct C++17 program that validates UTF-8." -n 256
```

The local developer model path is the default. For another location:

```sh
export METALBLOK_MODEL=/absolute/path/DeepSeek-R1-UD-IQ1_S-00001-of-00003.gguf
./run_blok.py "Say hello to Max." -n 64
```

The default is deterministic greedy decoding with a 2,048-position context.
For a persistent conversation:

```sh
./run_blok.py "Remember that my name is Max." -n 128 \
  --state conversations/max.state
./run_blok.py "What is my name?" -n 64 \
  --state conversations/max.state
```

For the exact 1,000-input/1,000-output coding acceptance:

```sh
scripts/prove_metal_1k.py
# equivalent project-level entry point:
./end_goal_prompt.py
```

See the [complete CLI guide](metalblok/docs/RUN_GUIDE.md),
[deep architecture](metalblok/docs/V0_ARCHITECTURE.md),
[million-token scale plan](metalblok/docs/MILLION_TOKEN_SCALE_PLAN.md),
[V0 closeout](metalblok/docs/V0_CLOSEOUT.md), and
[current status](metalblok/STATUS.md). Output text, native timing, token/logit,
NVMe, Metal, memory-pressure, and checkpoint evidence is saved under
`metalblok/runs/`.

## Build and verify

```sh
cmake -S metalblok -B metalblok/build
cmake --build metalblok/build -j8
ctest --test-dir metalblok/build --output-on-failure
python3 -m py_compile run_blok.py scripts/prove_metal_1k.py
```

The runner builds automatically when the binary is absent and refuses a
missing, sparse, dataless, wrong-sized, or wrong-model shard before inference.

## Repository scope

`metalblok/` and `run_blok.py` are the active M5/DeepSeek V0. The root CUDA,
Kimi, GLM, and vendored `sub_dir/uGDS` material records a separate Linux
research target; it is not linked into or required by the MetalBlok CLI.
