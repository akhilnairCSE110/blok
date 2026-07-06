# Blok

Kimi K2.6-only native inference runtime.

Path:

```sh
python3 end_goal_prompt.py
```

That calls:

```text
blok.runtime -> target/{debug,release}/blok generate -> build/blok-kimi-exec
```

The native executor owns the model path: manifest parsing, O_DIRECT/uGDS payload movement, CUDA
kernels, and token emission. vLLM/Transformers are not part of the product path.

Required external state:

- complete `moonshotai/Kimi-K2.6` download;
- `scripts/model_fetch.py kimi-k2.6 materialize`;
- CUDA/CMake build producing `build/blok-kimi-exec`.
