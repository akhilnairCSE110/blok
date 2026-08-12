#!/usr/bin/env python3
"""Run the active MetalBlok 1,000-input/1,000-output acceptance proof."""

from pathlib import Path
import subprocess


def main() -> None:
    proof = Path(__file__).resolve().parent / "scripts/prove_metal_1k.py"
    subprocess.run([str(proof)], check=True)


if __name__ == "__main__":
    main()
