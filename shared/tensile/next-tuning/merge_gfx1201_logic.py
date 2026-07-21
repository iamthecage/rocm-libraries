#!/usr/bin/env python3
"""
Merge gfx1201 wide-run benchmark results into rocBLAS logic files.

This script takes the Tensile benchmark outputs from /home/iamthecage/out-v2/
and produces merged logic files suitable for placement into the rocBLAS
logic directory:
  projects/rocblas/library/src/blas3/Tensile/Logic/asm_full/gfx1201/

Workflow:
  1. Collect all 3_LibraryLogic/*.yaml files from completed benchmark runs
  2. Group by actual Tensile output filename (the YAML basename defines the
     problem type — e.g. Cijk_Ailk_Bljk_HHS_BH.yaml)
  3. When multiple tuning runs target the same problem type (WMMA + VALU + GSU),
     merge them with TensileMergeLibrary so the best kernel wins per size
  4. Rename from navi48_* to gfx1201_* (rocBLAS convention)
  5. Output to staging dir ready for copy into rocBLAS source tree

Usage:
  python3 merge_gfx1201_logic.py [--output-dir OUTDIR] [--dry-run]
  python3 merge_gfx1201_logic.py --status   # just show what's available
"""

import argparse
import os
import sys
import shutil
import subprocess
import tempfile
import yaml
from collections import defaultdict
from pathlib import Path

# Where benchmark outputs live
BENCHMARK_DIRS = [
    Path("/home/iamthecage/out-v2")
]

# Where merged logic goes
DEFAULT_OUTPUT = Path("/home/iamthecage/rocm-libraries/shared/tensile/next-tuning/gfx1201-merged-logic-v2")

# rocBLAS logic target
ROCBLAS_LOGIC_DIR = Path("/home/iamthecage/rocm-libraries/projects/rocblas/library/src/blas3/Tensile/Logic/asm_full")

# TensileMergeLibrary location
TENSILE_DIR = Path("/home/iamthecage/rocm-libraries/shared/tensile")
MERGE_TOOL = TENSILE_DIR / "Tensile" / "bin" / "TensileMergeLibrary"

# ---------------------------------------------------------------------------
# Valid transpose suffixes (including complex conjugate variants)
# ---------------------------------------------------------------------------
VALID_TRANSPOSES = {"nn", "nt", "tn", "tt", "nc", "cn", "cc", "tc", "ct"}

# NOTE: The '_gb' suffix in config names (e.g. hgemm_valu_gb_nn) is a human label
# for "optimized for batched workloads" but still produces StridedBatched=True kernels.
# The rocBLAS _GB YAML family (StridedBatched=False / grouped-batch) is a separate
# ProblemType that requires dedicated configs — not present in current tuning runs.


def parse_config_name(name):
    """Parse a benchmark output dir name into components.

    Handles all naming patterns in out-v2:
      hgemm_wmma_nn, hgemm_valu_gb_nn, hgemm_native_nn, hgemm_wmma_native_nn,
      bf16gemm_wmma_gsu_tt, bss_valu_tn, b8_to_s_wmma_nn,
      b8f8_to_b8_wmma_tn, f8b8_to_f8_wmma_tt,
      cgemm_cc, zgemm_nc, dgemm_nn, sgemm_tn, i8gemm_wmma_nn

    Returns: (dtype, method, is_gb, transpose) or None if unparsable.
    """
    parts = name.split("_")

    # Last part must be a valid transpose
    transpose = parts[-1]
    if transpose not in VALID_TRANSPOSES:
        return None

    # --- Determine dtype prefix ---
    # FP8/BF8 mixed-precision: b8f8_to_X, f8b8_to_X, b8_to_X, f8_to_X
    if parts[0] in ("b8f8", "f8b8", "b8", "f8") and len(parts) > 2 and parts[1] == "to":
        dtype = f"{parts[0]}_to_{parts[2]}"
        rest = parts[3:-1]
    elif parts[0] in ("bf16gemm",):
        dtype = "bf16gemm"
        rest = parts[1:-1]
    elif parts[0] in ("hgemm", "sgemm", "dgemm", "cgemm", "zgemm", "i8gemm"):
        dtype = parts[0]
        rest = parts[1:-1]
    elif parts[0] in ("hss", "bss"):
        dtype = parts[0]
        rest = parts[1:-1]
    else:
        return None

    # Determine method from remaining parts (strip 'gb' — it's a label, not a problem type flag)
    rest_clean = [p for p in rest if p != "gb"]
    method = "_".join(rest_clean) if rest_clean else "default"

    return (dtype, method, transpose)


# ---------------------------------------------------------------------------
# Mapping from our config dtype to rocBLAS logic family (for status display)
# ---------------------------------------------------------------------------
DTYPE_TO_FAMILY = {
    "hgemm":       "HHS_BH / HB",
    "bf16gemm":    "BBS_BH",
    "hss":         "HSS_BH",
    "bss":         "BSS_BH",
    "sgemm":       "SB",
    "dgemm":       "DB",
    "cgemm":       "CB",
    "zgemm":       "ZB",
    "i8gemm":      "I8II_BH",
    "b8_to_s":     "B8SS_BH",
    "b8_to_b8":    "B8B8S_BH",
    "b8_to_h":     "B8HS_BH",
    "b8f8_to_s":   "B8F8SS_BH",
    "b8f8_to_b8":  "B8F8B8S_BH",
    "b8f8_to_f8":  "B8F8F8S_BH",
    "b8f8_to_h":   "B8F8HS_BH",
    "f8b8_to_s":   "F8B8SS_BH",
    "f8b8_to_b8":  "F8B8B8S_BH",
    "f8b8_to_f8":  "F8B8F8S_BH",
    "f8b8_to_h":   "F8B8HS_BH",
    "f8_to_s":     "F8SS_BH",
    "f8_to_f8":    "F8F8S_BH",
    "f8_to_h":     "F8HS_BH",
}


def collect_logic_files():
    """Scan benchmark dirs and group by the actual YAML filename Tensile produced.

    The YAML basename (e.g. navi48_Cijk_Ailk_Bljk_HHS_BH.yaml) is the
    canonical problem-type identifier.  Multiple tuning configs that produce
    the same basename need to be merged.
    """
    groups = defaultdict(list)

    for bench_dir in BENCHMARK_DIRS:
        if not bench_dir.exists():
            print(f"  SKIP: {bench_dir} (directory does not exist)")
            continue
        for entry in sorted(bench_dir.iterdir()):
            if not entry.is_dir():
                continue

            parsed = parse_config_name(entry.name)
            if not parsed:
                print(f"  SKIP: {entry.name} (unparsable name)")
                continue

            dtype, method, transpose = parsed

            logic_dir = entry / "3_LibraryLogic"
            if not logic_dir.exists():
                print(f"  SKIP: {entry.name} (no 3_LibraryLogic yet)")
                continue

            yamls = list(logic_dir.glob("*.yaml"))
            if not yamls:
                print(f"  SKIP: {entry.name} (empty 3_LibraryLogic)")
                continue

            src_yaml = yamls[0]

            # Use Tensile's output filename directly as the canonical target name
            target_name = src_yaml.name.replace("navi48_", "gfx1201_")

            groups[target_name].append((dtype, transpose, method, src_yaml))

    return groups


def show_status(groups):
    """Print status of all collected logic files."""
    print("\n=== gfx1201 Benchmark Logic Files ===\n")

    # Restructure for pretty printing
    by_dtype = defaultdict(dict)
    for target_name, items in groups.items():
        for dtype, transpose, method, src_yaml in items:
            key = (transpose, target_name)
            if key not in by_dtype[dtype]:
                by_dtype[dtype][key] = []
            by_dtype[dtype][key].append(method)

    total_logic = 0

    for dtype in sorted(by_dtype.keys()):
        family = DTYPE_TO_FAMILY.get(dtype, "???")
        print(f"\n{dtype} -> {family}:")

        for (transpose, target), methods in sorted(by_dtype[dtype].items()):
            merge_note = " ← WILL MERGE" if len(methods) > 1 else ""
            print(f"  {transpose.upper()}: {','.join(methods)} -> {target}{merge_note}")
            total_logic += len(methods)

    total_merged = len(groups)

    print("\n\n=== Coverage Summary ===")
    print(f"  Logic files collected:     {total_logic}")
    print(f"  Unique output targets:     {total_merged}")
    needs_merge = sum(1 for items in groups.values() if len(items) > 1)
    print(f"  Targets requiring merge:   {needs_merge}")

    # Deduce actual families from the filenames Tensile generated
    our_families = set()
    for target_name in groups.keys():
        parts = target_name.replace(".yaml", "").split("_")
        # Family is everything after gfx1201_Cijk_A*_B*_
        family = "_".join(parts[4:])
        our_families.add(family)

    print(f"\n  Our families: {sorted(our_families)}")


def merge_logic_files(groups, output_dir, dry_run=False):
    """Merge WMMA + VALU + GSU logic files for each problem type.

    When there's only one method for a problem type, just copy the file.
    When there are multiple (e.g. WMMA + VALU + GSU), chain TensileMergeLibrary.
    """
    output_dir = Path(output_dir)

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Merging to {output_dir} ===\n")

    results = {"copied": 0, "merged": 0, "failed": 0}

    for target_name, items in sorted(groups.items()):
        target_path = output_dir / target_name

        if len(items) == 1:
            # Single source — just copy with arch rename
            dtype, transpose, method, src = items[0]
            print(f"  COPY: {src.name} -> {target_name} ({method})")
            if not dry_run:
                with open(src, "r") as f:
                    content = f.read()
                
                # Fast string replacement instead of slow PyYAML parse/dump
                content = content.replace("- navi48", "- gfx1201")
                content = content.replace("ArchitectureName: \"navi48\"", "ArchitectureName: \"gfx1201\"")
                
                with open(target_path, "w") as f:
                    f.write(content)
            results["copied"] += 1

        else:
            # Multiple sources — chain-merge them
            method_names = [item[2] for item in items]
            print(f"  MERGE: {target_name} <- {method_names}")

            if dry_run:
                results["merged"] += 1
                continue

            with tempfile.TemporaryDirectory() as tmpdir:
                # The YAML filename TensileMergeLibrary matches on
                std_name = target_name.replace("gfx1201_", "navi48_")

                # Start: copy first source as the "original"
                orig_dir = os.path.join(tmpdir, "original")
                os.makedirs(orig_dir)
                shutil.copy2(items[0][3], os.path.join(orig_dir, std_name))

                current_dir = orig_dir
                merge_ok = True

                for i, (dtype, transpose, method, src) in enumerate(items[1:], 1):
                    inc_dir = os.path.join(tmpdir, f"inc_{i}_{method}")
                    os.makedirs(inc_dir)
                    shutil.copy2(src, os.path.join(inc_dir, std_name))

                    out_dir = os.path.join(tmpdir, f"merged_{i}")
                    os.makedirs(out_dir)

                    cmd = [
                        sys.executable,
                        str(MERGE_TOOL),
                        current_dir,
                        inc_dir,
                        out_dir,
                        "--force_merge", "1",
                        "-v", "0",
                    ]

                    try:
                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
                        if result.returncode != 0:
                            print(f"    MERGE FAILED (step {i}):\n{result.stderr[-1500:]}")
                            merge_ok = False
                            break
                    except subprocess.TimeoutExpired:
                        print(f"    MERGE TIMEOUT (step {i})")
                        merge_ok = False
                        break

                    current_dir = out_dir

                if merge_ok:
                    merged_file = os.path.join(current_dir, std_name)
                    if os.path.exists(merged_file):
                        with open(merged_file, "r") as f:
                            content = f.read()
                        
                        # Fast string replacement
                        content = content.replace("- navi48", "- gfx1201")
                        content = content.replace("ArchitectureName: \"navi48\"", "ArchitectureName: \"gfx1201\"")
                        
                        with open(target_path, "w") as f:
                            f.write(content)
                        results["merged"] += 1
                    else:
                        print(f"    MERGE produced no output file")
                        results["failed"] += 1
                else:
                    results["failed"] += 1

    print(f"\n=== Results ===")
    print(f"  Copied:  {results['copied']}")
    print(f"  Merged:  {results['merged']}")
    print(f"  Failed:  {results['failed']}")
    print(f"  Total:   {results['copied'] + results['merged']} logic files ready")

    return results


def main():
    parser = argparse.ArgumentParser(description="Merge gfx1201 benchmark results into rocBLAS logic files")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT),
                       help=f"Output directory for merged logic (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--status", action="store_true",
                       help="Just show status of collected logic files")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be done without writing files")
    parser.add_argument("--install", action="store_true",
                       help="After merging, copy files to rocBLAS logic dir")
    args = parser.parse_args()

    print("Scanning benchmark outputs in", [str(d) for d in BENCHMARK_DIRS])
    groups = collect_logic_files()

    if not groups:
        print("\nNo logic files found! Check that tuning runs have completed.")
        return

    if args.status:
        show_status(groups)
        return

    show_status(groups)
    results = merge_logic_files(groups, args.output_dir, args.dry_run)

    if args.install and not args.dry_run and results["failed"] == 0:
        dest = ROCBLAS_LOGIC_DIR / "gfx1201"
        print(f"\n=== Installing to {dest} ===")
        dest.mkdir(parents=True, exist_ok=True)
        for f in Path(args.output_dir).glob("*.yaml"):
            shutil.copy2(f, dest / f.name)
            print(f"  {f.name}")
        print(f"Done. rocBLAS can now be rebuilt with Tensile_LOGIC=asm_full to include gfx1201 kernels.")


if __name__ == "__main__":
    main()