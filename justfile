set shell := ["bash", "-cu"]

setup:
    scripts/ci.sh setup

fmt:
    scripts/ci.sh fmt

fmt-check:
    scripts/ci.sh fmt-check

check-local:
    scripts/ci.sh check-local

check-linux:
    scripts/ci.sh check-linux

deep-linux:
    scripts/ci.sh deep-linux

cxx-linux:
    scripts/ci.sh cxx-linux

cxx-sanitize-linux:
    scripts/ci.sh cxx-sanitize-linux

rust-linux:
    scripts/ci.sh rust-linux

rust-deep-linux:
    scripts/ci.sh rust-deep-linux

cuda-deep-linux:
    scripts/ci.sh cuda-deep-linux

deps:
    scripts/ci.sh deps

hardware-check:
    scripts/ci.sh hardware-check

model-contract:
    scripts/ci.sh model-contract

release-linux:
    scripts/ci.sh release-linux

clean:
    rm -rf build build-asan build-release dist
