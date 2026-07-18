# System Requirements

## Exact Target Hardware

- GPU: NVIDIA RTX 5060 Ti, GB206, `sm_120`.
- CPU: AMD Ryzen 9 5950X.
- RAM: 48 GB.
- Primary uGDS storage: Samsung 990 EVO Plus 1TB NVMe, PCIe 4.0 x4 / 5.0 x2.
- Non-uGDS storage: Kingston SA400S37240G 240GB SATA SSD; Seagate ST2000DM008-2FR102 2TB SATA HDD.
- Motherboard: MSI MAG X870 Tomahawk.
- OS: Ubuntu/Linux bare metal for the target run. Windows host state is not sufficient for uGDS.

## Required Software

- NVIDIA open kernel driver supported by uGDS.
- CUDA toolkit with `nvcc`.
- CMake and Ninja.
- Rust toolchain for the launcher.
- uGDS kernel module and userspace library.

## Storage Contract

- Model shards must live on an NVMe region that can be mapped by uGDS.
- On this machine, use the Samsung 990 EVO Plus NVMe for model shards and KV scratch.
- The Kingston SATA SSD and Seagate HDD are not uGDS model/KV targets.
- Runtime reads use explicit shard-to-block extents from `BLOK_UGDS_MAP`.
- Generate `BLOK_UGDS_MAP` with FIEMAP while the model filesystem is mounted, then unmount and bind the NVMe device to uGDS before raw uGDS reads.
- KV scratch uses a separate uGDS block-device region configured by `BLOK_KV_UGDS_BASE` and optionally bounded by `BLOK_KV_UGDS_BYTES`.
- Root-mounted devices are not valid uGDS raw-read or write-scratch targets.
