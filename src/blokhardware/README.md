# Hardware Plan

Blok treats GPU, CPU, RAM, and NVMe as one scheduled execution system. Hardware plans define what
must be measured before runtime code assumes a path exists.

## Requirements To Question

- Is the observed hardware the target host or a sandboxed shell with missing device visibility?
- Which device owns each byte: root SSD, model SSD, benchmark SSD, uGDS SSD, system RAM, or VRAM?
- Which path is direct, staged, or fallback?
- Which measurement is required before enabling a hardware-specific path?

## Delete / Simplify / Optimize / Automate

- Delete product-string assumptions; use measured capability probes.
- Simplify first hardware support to the current Linux target before portability work.
- Optimize topology, pinning, and queue depth only after baseline reports exist.
- Automate capability failures with exact fixes.

## Interfaces

- `hardware` reports topology and capability.
- `io` reports direct, staged, and fallback movement.
- `cuda` reports kernel, stream, event, graph, and dtype capability.
- `observe` normalizes hardware reports for remote comparison.

## Gate

Unsupported hardware, driver, format, layout, or dtype paths fail with typed capability errors.
