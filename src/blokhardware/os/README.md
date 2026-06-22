# Operating System Plan

Ubuntu 26.04 bare metal is the target operating system. Linux is part of the control plane, not an
obstacle to bypass blindly.

## Requirements To Question

- Does the operation require driver or firmware provisioning, or is it a normal build/test action?
- Does the operation need root privileges?
- Is the shell sandbox hiding hardware, or is the hardware actually absent?
- Can the runtime use Linux scheduling, filesystem, and driver behavior without weakening direct
  control of payload movement?

## Delete / Simplify / Optimize / Automate

- Delete provisioning from normal CI.
- Simplify setup scripts to developer/build tools only.
- Optimize CPU affinity and I/O polling only after topology probes run.
- Automate sandbox-vs-absent failure classification in hardware reports.

## Boundaries

- `docs/system-requirements.md` owns machine provisioning.
- `scripts/bootstrap_linux.sh` owns developer/build tooling.
- Runtime probes report missing prerequisites; they do not install drivers or reconfigure hardware.

## Gate

Hardware probes must distinguish `absent`, `permission_blocked`, `driver_unreachable`, and
`unsupported`.

## Sources

- Ubuntu packages: https://packages.ubuntu.com/
- Rust install path: https://rustup.rs/
