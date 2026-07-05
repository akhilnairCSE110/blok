# Source Plan

`src/` now contains the first executable Rust surface. The current code owns CLI validation,
configuration, typed errors, deterministic reports, normalized manifest parsing, and first-token
graph, arena, uGDS-first transfer descriptors, and a first `sm_120` PTX byte-probe descriptor.
Subsystem directories still hold structured
plans that must be converted into module docs, types, probes, reports, and tests only as
implementation lands.

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

- `src/blok_command_line_entrypoint.rs`: CLI only.
- `src/blok_runtime_library.rs`: crate invariants and module exports.
- `src/runtime_environment_config.rs`: typed constants and environment parsing.
- `src/blok_runtime_error.rs`: capability, validation, I/O, CUDA, graph, and layout errors.
- `src/command_report_json.rs`: deterministic report schemas.
- `src/tensor_manifest_parser.rs`: normalized sidecar manifest parser and validator.
- `src/first_token_execution_graph.rs`: first-token declared working-set descriptors.
- `src/arena.rs`: first-token VRAM arena views derived from the graph.
- `src/io.rs`: uGDS-first aligned transfer windows and the first Linux direct-I/O fallback probe.
- `src/cuda.rs`: optimized `sm_120` PTX byte-probe descriptor for the first transfer-to-kernel
  gate.

Do not create layout, KV, kernel registry, or decode files until the descriptor boundary they own is
real and tested. CUDA code is currently limited to the first byte-probe contract; the next CUDA
change must load and launch that PTX through the driver API or delete it.

## Gate

`just check-local` must pass immediately after the scaffold lands, and every new source file must be
referenced by either the CLI, a public library export, or a test.
