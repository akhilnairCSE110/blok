#!/usr/bin/env bash
set -euo pipefail

rm -rf dist
mkdir -p dist/bin

if [ -d target/release ]; then
  find target/release -maxdepth 1 -type f -perm -111 -exec cp {} dist/bin/ \;
fi

if [ -d build-release ]; then
  find build-release -type f -perm -111 -exec cp {} dist/bin/ \;
fi

if [ -z "$(find dist/bin -type f -print -quit)" ]; then
  echo "No release binaries found; creating metadata-only artifact."
  rmdir dist/bin
fi

git rev-parse HEAD >dist/commit.txt

archive="$(mktemp)"
tar -czf "$archive" -C dist .
mv "$archive" dist/release-linux-x86_64.tar.gz
echo "Release package created in dist/"
