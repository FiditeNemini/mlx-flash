#!/usr/bin/env python3
"""
compile_kernels.py — AOT-compile Flash Metal kernels to .metallib.

Usage:
    python mlx_flash/kernels/compile_kernels.py [--verbose]

Requires Xcode Command Line Tools (xcrun xcoderun metallib).
Produces: mlx_flash/kernels/flash_kernels.metallib
"""

import argparse
import subprocess
import sys
from pathlib import Path

KERNELS_DIR = Path(__file__).parent
SOURCES = ["flash_dequant.metal", "swiglu_fused.metal", "moe_dispatch.metal"]
OUTPUT = KERNELS_DIR / "flash_kernels.metallib"


def compile_metal(verbose: bool = False) -> int:
    """Compile all .metal sources to a single .metallib."""
    air_files = []
    print("Compiling Flash Metal kernels...")

    for src_name in SOURCES:
        src = KERNELS_DIR / src_name
        air = KERNELS_DIR / (src_name.replace(".metal", ".air"))
        cmd = [
            "xcrun", "-sdk", "macosx", "metal",
            "-c", str(src),
            "-o", str(air),
            "-O2",
        ]
        if verbose:
            print(f"  {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=not verbose)
        if result.returncode != 0:
            print(f"ERROR compiling {src_name}:")
            print(result.stderr.decode())
            return 1
        air_files.append(str(air))
        print(f"  ✓  {src_name} → {air.name}")

    # Link .air files into .metallib
    cmd = ["xcrun", "-sdk", "macosx", "metallib"] + air_files + ["-o", str(OUTPUT)]
    if verbose:
        print(f"  {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=not verbose)
    if result.returncode != 0:
        print("ERROR linking metallib:")
        print(result.stderr.decode())
        return 1

    # Clean up intermediate .air files
    for af in air_files:
        Path(af).unlink(missing_ok=True)

    print(f"\n✅  {OUTPUT.name} created ({OUTPUT.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()
    sys.exit(compile_metal(args.verbose))
