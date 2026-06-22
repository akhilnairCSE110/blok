# System Requirements

Target OS: Ubuntu 26.04 on bare metal.

## CPU: AMD Ryzen 9

Required:
- `amd64-microcode`
- `linux-tools-generic`
- `linux-tools-$(uname -r)`
- `cpufrequtils` or the Ubuntu-supported CPU frequency tooling available for the target kernel

Recommended:
- AMD uProf for CPU profiling
- AOCC only if compiler benchmarking shows a measurable win over Clang/GCC

## GPU: NVIDIA RTX 5060 Ti

Required:
- NVIDIA GPU driver with open kernel modules, version 550 or newer
- CUDA Toolkit 12 or newer
- CUDA compiler/runtime libraries, including `nvcc` and `cudart`
- Kernel headers matching the running Ubuntu kernel

Required for uGDS:
- ScaleX-IO uGDS from https://github.com/ScaleX-IO/uGDS
- uGDS kernel module built against the running kernel
- NVIDIA driver source under `/usr/src/nvidia-*`
- An NVMe device that can be unbound from the kernel `nvme` driver and rebound to `ugds_drv`

Compatibility baseline:
- NVIDIA GDS/cuFile for side-by-side validation against uGDS

## Storage: Samsung PM9C1a / 990 EVO Plus NVMe

Required:
- `nvme-cli`
- `smartmontools`
- `fwupd`

Operational notes:
- Keep firmware current before benchmarking.
- Use `nvme-cli` and SMART data for health, namespace, queue, and thermal checks.
- If Samsung Magician is needed for a consumer EVO firmware path, run it outside the Linux
  CI/provisioning flow and record the resulting firmware version.
- Do not bind the OS/root disk to uGDS. Use a dedicated benchmark/model NVMe device.

## Base Linux Build Tools

Required:
- `build-essential`
- `clang`
- `clang-format`
- `clang-tidy`
- `cmake`
- `cppcheck`
- `curl`
- `git`
- `libssl-dev`
- `ninja-build`
- `pipx`
- `pkg-config`
- `shellcheck`
- `shfmt`

Rust/dev tools are installed by `scripts/bootstrap_linux.sh`.

## Provisioning References

- NVIDIA GDS overview and O_DIRECT behavior:
  https://docs.nvidia.com/gpudirect-storage/overview-guide/index.html
- CUDA documentation:
  https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
- ScaleX-IO uGDS:
  https://github.com/ScaleX-IO/uGDS
- Ubuntu packages:
  https://packages.ubuntu.com/
- Rust installation:
  https://rustup.rs/
