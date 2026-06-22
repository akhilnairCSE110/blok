# AMD Ryzen 9 CPU Plan

The CPU owns orchestration, preprocessing, scheduling, memory preparation, I/O polling, and runtime
bookkeeping. It should keep GPU queues full instead of competing with GPU math.

## Requirements To Question

- Which CPU work is required for correctness before GPU launch?
- Which CPU work can run ahead of the current token?
- Which cores should own CUDA submission, I/O polling, tokenization, sampling, and reporting?
- Does pinning improve measured throughput on this host?

## Delete / Simplify / Optimize / Automate

- Delete hard-coded CPU core numbers.
- Simplify topology to measured sockets, NUMA nodes, cache groups, and logical CPUs.
- Optimize pinning only after reports show submission or polling stalls.
- Automate CPU topology reports and governor checks.

## First Responsibilities

- hardware probe data collection;
- deterministic scheduler bookkeeping;
- aligned transfer descriptor construction;
- tokenizer and sampling path after manifest support;
- report serialization.

## Gate

H0 reports CPU model, vendor, sockets, physical cores, logical CPUs, NUMA nodes, scaling driver,
frequency governor, and a provisional CCD/cache-group plan.
