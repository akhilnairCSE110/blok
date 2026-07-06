# Source Invariants

- Kimi K2.6 only.
- Rust owns CLI, manifest validation, and native process launch.
- CUDA/C++ owns payload movement, kernels, and token emission.
- No descriptor-only success path.
- No generic graph compiler.
- No fake token output.
