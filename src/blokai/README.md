# AI Asset Plan

`src/blokai` defines ownership for model bytes and auxiliary runtime checkpoints. It does not own
inference logic; runtime logic belongs in the Rust crate modules created during scaffold.

## Requirements To Question

- Which assets must persist across process restarts?
- Which assets are source checkpoints, Blok sidecar layouts, repacked Blok layouts, KV spill files,
  or optimizer checkpoints?
- Which assets are required for first-token correctness, and which are throughput layers?
- Can an asset be regenerated deterministically instead of stored?

## Delete / Simplify / Optimize / Automate

- Delete asset categories that do not serve the current milestone.
- Simplify all model formats into one `Manifest` contract before adding format-specific runtime
  behavior.
- Optimize layout only after read reports show source-checkpoint order is inefficient.
- Automate asset validation after the tiny fixture and first real model path agree.

## Interfaces

- Target models flow through `manifest -> layout -> graph`.
- Auxiliary models flow through optimizer ownership and must report their runtime overhead.
- Runtime code consumes typed manifests and layout descriptors, never raw directory assumptions.

## Gate

The first implementation must load metadata without touching payload bytes. Payload movement starts
only after offset, size, alignment, destination arena, lifetime, and consumer operation are known.
