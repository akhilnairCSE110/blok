# GPU Invariants

- Target SM: `sm_120`.
- CUDA kernels are the execution truth.
- PTX/inline assembly is welcome where it removes measurable overhead.
- Host issues storage I/O; kernels consume GPU buffers.
