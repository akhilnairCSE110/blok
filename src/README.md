# Source Plan

`src/` now contains the first executable Rust surface. The current code owns CLI validation,
configuration, typed errors, deterministic reports, normalized manifest parsing, and first-token
graph descriptors. Subsystem directories still hold structured plans that must be converted into
module docs, types, probes, reports, and tests only as implementation lands.

## Requirements To Question

- Does a new module need to exist, or can the behavior live in an existing module?
- Does the module own a real runtime boundary: manifest, layout, graph, arena, I/O, CUDA, KV,
  decode, observation, or hardware probing?
- Can the module expose a typed contract now, or is it a vague placeholder?
- Can the module be tested without the full huge-model path?

## Delete / Simplify / Optimize / Automate

- Delete source files that do not define a real type, command, report, parser, probe, or test.
- Simplify ownership before adding abstractions; one concept should have one home.
- Optimize only after reports show the bottleneck.
- Automate static rejection of payload `mmap`, unbounded hot-path allocation, accidental f16 on
  `sm_120`, and hidden staged I/O.

## Current Code Shape

- `src/main.rs`: CLI only.
- `src/lib.rs`: crate invariants and module exports.
- `src/config.rs`: typed constants and environment parsing.
- `src/error.rs`: capability, validation, I/O, CUDA, graph, and layout errors.
- `src/observe.rs`: deterministic report schemas.
- `src/manifest.rs`: normalized sidecar manifest parser and validator.
- `src/graph.rs`: first-token declared working-set descriptors.

Do not create CUDA, I/O, layout, arena, or decode files until the descriptor boundary they own is
real and tested.

## Gate

`just check-local` must pass immediately after the scaffold lands, and every new source file must be
referenced by either the CLI, a public library export, or a test.
