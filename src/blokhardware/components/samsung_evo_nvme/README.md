# NVMe Invariants

- NVMe is backing store, not the working set.
- Reads must be large, aligned, and explicit.
- uGDS is preferred when a device can be dedicated.
- Root-mounted devices are read-only runtime sources.
