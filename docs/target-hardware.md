# Target Hardware

User-reported Windows system summary, vendor-checked where possible.

## Confirmed From User

- Device name: SIDPC
- CPU: AMD Ryzen 9 5950X 16-core, 3.40 GHz base.
- RAM: 48 GB installed, 47.9 GB usable.
- GPU: NVIDIA GeForce RTX 5060 Ti, 16 GB.
- Primary SSD: Samsung SSD 990 EVO Plus 1TB, shown as 932 GB usable.
- Secondary SSD: Kingston SA400S37240G, shown as 224 GB usable.
- HDD: Seagate ST2000DM008-2FR102, shown as 1.82 TB usable.
- OS type: 64-bit Windows, x64-based processor.

## Vendor-Checked Facts

- AMD Ryzen 9 5950X: 16 cores, 32 threads, 3.4 GHz base, up to 4.9 GHz boost, 105 W TDP, AM4 socket, DDR4, PCIe 4.0.
  Source: https://www.amd.com/en/products/processors/desktops/ryzen/5000-series/amd-ryzen-9-5950x.html
- NVIDIA GeForce RTX 5060 Ti: Blackwell, 4608 CUDA cores, 16 GB or 8 GB GDDR7 variants, 448 GB/s memory bandwidth, CUDA capability 12.0, 180 W total graphics power.
  Source: https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5060-family/
- CUDA capability 12.0 corresponds to `sm_120`.
  Source: https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/cuda-platform.html
- Samsung 990 EVO Plus 1TB: PCIe 4.0 x4 / 5.0 x2 NVMe 2.0, M.2 2280, TLC, up to 7,150 MB/s read and 6,300 MB/s write.
  Source: https://news.samsung.com/us/samsung-launches-990-evo-plus-ssd-with-improved-performance-speeds-supported-by-pcie-4-0/
- Kingston SA400S37/240G: 2.5-inch SATA Rev 3.0 SSD, 240 GB, up to 500 MB/s read and 350 MB/s write.
  Source: https://www.kingston.com/unitedkingdom/en/memory/search?partid=SA400S37%2F240G
- Seagate ST2000DM008: BarraCuda 3.5-inch SATA HDD, 2 TB, 256 MB cache.
  Source: https://www.seagate.com/support/internal-hard-drives/desktop-hard-drives/barracuda-3-5/

## Exactness Notes

- The motherboard is not in the user-provided system summary. Treat it as unknown until verified from firmware.
- The old MSI MAG X870 Tomahawk assumption is rejected: X870 is AM5, while Ryzen 9 5950X is AM4.
- The Samsung 990 EVO Plus is the only reported NVMe drive and therefore the only candidate uGDS model/KV target.
- The Kingston SATA SSD and Seagate SATA HDD are not uGDS model/KV targets.
- Windows is useful for inventory, but the target runtime still requires Ubuntu/Linux bare metal for uGDS.
