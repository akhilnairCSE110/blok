# Plan Index

Blok planning is intentionally distributed. The golden plan is the spine; subsystem plans live next
to the future code they will govern.

## Spine

- [Golden implementation plan](06_21_plan.md): target, invariants, research interpretation,
  milestone order, report schema, and anti-drift rules.
- [System requirements](system-requirements.md): machine provisioning, driver requirements, and
  hardware setup boundaries.

## Development Method

Every subsystem plan follows the same loop:

1. Question the requirement: identify whether the requirement is real, inherited, accidental, or
   measurable.
2. Delete: remove the path, module, dependency, transfer, allocation, or benchmark if it does not
   directly serve the current milestone.
3. Simplify: collapse duplicate concepts and make the smallest typed interface that can express the
   real requirement.
4. Optimize: optimize only after the report identifies a bottleneck.
5. Automate: automate checks, reports, and regressions only after the manual path is proven.
6. Iterate: repeat the loop whenever a new measurement contradicts the plan.

## Subsystem Plans

- [Source layout](../src/README.md)
- [AI assets](../src/blokai/README.md)
- [Target models](../src/blokai/models/README.md)
- [Runtime optimizers](../src/blokai/optimizers/README.md)
- [Hardware plan](../src/blokhardware/README.md)
- [Operating system](../src/blokhardware/os/README.md)
- [Runtime scheduling](../src/blokhardware/runtime/README.md)
- [CPU orchestration](../src/blokhardware/components/amd_ryzen9_cpu/README.md)
- [GPU execution](../src/blokhardware/components/nvidia_rtx_5060_gpu/README.md)
- [NVMe storage](../src/blokhardware/components/samsung_evo_nvme/README.md)

## Rule

If a plan becomes passive or stale, either update it with a new measurement or delete it. Planning
documents should reduce ambiguity for the next code change; they are not archival notes.
