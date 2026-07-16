# System Requirements

## Exact Target Hardware

- GPU: NVIDIA RTX 5060 Ti, GB206, `sm_120`.
- CPU: AMD Ryzen 9 5950X.
- Storage: Samsung 9100 Pro NVMe, PCIe 5.0 x4.
- Motherboard: MSI MAG X870 Tomahawk.
- OS: Ubuntu/Linux bare metal.

## Required Software

- NVIDIA open kernel driver supported by uGDS.
- CUDA toolkit with `nvcc`.
- CMake and Ninja.
- Rust toolchain for the launcher.
- uGDS kernel module and userspace library.

## Storage Contract

- Model shards must live on an NVMe region that can be mapped by uGDS.
- Runtime reads use explicit shard-to-block offsets from `BLOK_UGDS_MAP`.
- KV scratch uses a separate uGDS block-device region configured by `BLOK_KV_UGDS_BASE` and optionally bounded by `BLOK_KV_UGDS_BYTES`.
- Root-mounted devices are allowed only as read-only sources, not as uGDS-owned write scratch.
