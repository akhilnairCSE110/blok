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

deps:
    scripts/ci.sh deps

release-linux:
    scripts/ci.sh release-linux

clean:
    rm -rf build build-asan build-release dist
