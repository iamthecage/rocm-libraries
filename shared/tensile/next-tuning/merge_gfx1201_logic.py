#!/usr/bin/env python3
"""
Merge gfx1201 wide-run benchmark results into rocBLAS logic files.

This script takes the Tensile benchmark outputs from /home/iamthecage/out/
and produces merged logic files suitable for placement into the rocBLAS
logic directory:
  projects/rocblas/library/src/blas3/Tensile/Logic/asm_full/gfx1201/

Workflow:
  1. Collect all 3_LibraryLogic/*.yaml files from completed benchmark runs
  2. Group by problem type (Cijk layout + dtype family)
  3. Merge WMMA + VALU logic for same problem type using TensileMergeLibrary
  4. Rename from navi48_* to gfx1201_* (rocBLAS convention)
  5. Separate batched (BH) from grouped-batch (BH_GB) variants
  6. Output to staging dir ready for copy into rocBLAS source tree

Usage:
  python3 merge_gfx1201_logic.py [--output-dir OUTDIR] [--dry-run]
  python3 merge_gfx1201_logic.py --status   # just show what's available
"""

import argparse
import os
import sys
import shutil
import subprocess
import yaml
from collections import defaultdict
from pathlib import Path

# Where benchmark outputs live
BENCHMARK_DIRS = [
    Path("/home/iamthecage/out"),
    Path("/home/iamthecage/out_gsu"),
]

# Where merged logic goes
DEFAULT_OUTPUT = Path("/home/iamthecage/rocm-libraries/shared/tensile/next-tuning/gfx1201-merged-logic")

# rocBLAS logic target
ROCBLAS_LOGIC_DIR = Path("/home/iamthecage/rocm-libraries/projects/rocblas/library/src/blas3/Tensile/Logic/asm_full")

# TensileMergeLibrary location
TENSILE_DIR = Path("/home/iamthecage/rocm-libraries/shared/tensile")
MERGE_TOOL = TENSILE_DIR / "Tensile" / "bin" / "TensileMergeLibrary"

# ---------------------------------------------------------------------------
# Mapping from our config names to rocBLAS logic families
# ---------------------------------------------------------------------------
# rocBLAS logic filename convention:
#   {arch}_Cijk_{A_layout}_{B_layout}_{dtype_family}[_GB].yaml
#
# Our config names encode: {dtype}_{method}[_gb]_{transpose}
# where method = wmma|valu|native and transpose = nn|nt|tn|tt
#
# Multiple methods (wmma + valu) for the same {dtype}_{transpose} produce
# logic files for the SAME rocBLAS family — they must be merged.
# ---------------------------------------------------------------------------

# Map config dtype prefix -> rocBLAS dtype family
DTYPE_TO_FAMILY = {
    "hgemm":    "HHS_BH",     # f16 in, f32 accum, f16 out (High Precision Accumulate)
    "bf16gemm": "BBS_BH",     # bf16 in, f32 accum, bf16 out
    "hss":      "HB",         # f16 in, f32 out (Half B)
    "bss":      "BSS_BH",     # bf16 in, f32 out
    "sgemm":    "SB",         # f32 in, f32 out (Single B)
    "dgemm":    "DB",         # f64 in, f64 out (Double B) -- may not exist in navi31
    "i8gemm":   "I8II_BH",    # int8 in, int32 out
}

# Transpose -> Cijk layout
TRANSPOSE_TO_LAYOUT = {
    "nn": "Cijk_Ailk_Bljk",   # A=N, B=N
    "nt": "Cijk_Ailk_Bjlk",   # A=N, B=T
    "tn": "Cijk_Alik_Bljk",   # A=T, B=N
    "tt": "Cijk_Alik_Bjlk",   # A=T, B=T
}


def parse_config_name(name):
    """Parse a benchmark output dir name into components.
    
    Returns: (dtype, method, is_gb, transpose) or None if unparsable.
    
    Examples:
      hgemm_wmma_nn     -> (hgemm, wmma, False, nn)
      bf16gemm_valu_gb_nt -> (bf16gemm, valu, True, nt)
      sgemm_nn          -> (sgemm, valu, False, nn)  # implicit VALU
      hgemm_wmma_native_tt -> (hgemm, wmma_native, False, tt)
    """
    parts = name.split("_")
    transpose = parts[-1]
    if transpose not in ("nn", "nt", "tn", "tt"):
        return None
    
    # Check for grouped batch
    is_gb = "gb" in parts
    
    # Determine dtype prefix
    if name.startswith("bf16gemm"):
        dtype = "bf16gemm"
        rest = parts[1:-1]  # skip bf16gemm and transpose
    elif name.startswith("hgemm"):
        dtype = "hgemm"
        rest = parts[1:-1]
    elif name.startswith("sgemm"):
        dtype = "sgemm"
        rest = parts[1:-1]
    elif name.startswith("dgemm"):
        dtype = "dgemm"
        rest = parts[1:-1]
    elif name.startswith("i8gemm"):
        dtype = "i8gemm"
        rest = parts[1:-1]
    elif name.startswith("hss"):
        dtype = "hss"
        rest = parts[1:-1]
    elif name.startswith("bss"):
        dtype = "bss"
        rest = parts[1:-1]
    else:
        return None
    
    # Determine method from remaining parts
    rest_clean = [p for p in rest if p not in ("gb",)]
    if not rest_clean:
        method = "valu"  # e.g. sgemm_nn -> implicit VALU
    else:
        method = "_".join(rest_clean)  # e.g. "wmma", "valu", "wmma_native"
    
    return (dtype, method, is_gb, transpose)


def get_logic_filename(dtype, transpose, is_gb, arch="gfx1201"):
    """Build the rocBLAS logic filename for a dtype/transpose/gb combination."""
    family = DTYPE_TO_FAMILY.get(dtype)
    if not family:
        return None
    layout = TRANSPOSE_TO_LAYOUT.get(transpose)
    if not layout:
        return None
    gb_suffix = "_GB" if is_gb else ""
    return f"{arch}_{layout}_{family}{gb_suffix}.yaml"


def collect_logic_files():
    """Scan benchmark output dirs for 3_LibraryLogic/*.yaml files.
    
    Returns: dict mapping (dtype, transpose, is_gb) -> [(method, yaml_path), ...]
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
        
            dtype, method, is_gb, transpose = parsed
        
            logic_dir = entry / "3_LibraryLogic"
            if not logic_dir.exists():
                print(f"  SKIP: {entry.name} (no 3_LibraryLogic/)")
                continue
        
            yamls = list(logic_dir.glob("*.yaml"))
            if not yamls:
                print(f"  SKIP: {entry.name} (no yaml in 3_LibraryLogic/)")
                continue
        
            if len(yamls) > 1:
                print(f"  WARN: {entry.name} has {len(yamls)} logic files, using first")
        
            key = (dtype, transpose, is_gb)
            groups[key].append((method, yamls[0]))
    
    return groups


def show_status(groups):
    """Print status of all collected logic files."""
    print("\n=== gfx1201 Benchmark Logic Files ===\n")
    
    # Group by dtype
    by_dtype = defaultdict(list)
    for (dtype, transpose, is_gb), methods in sorted(groups.items()):
        by_dtype[dtype].append((transpose, is_gb, methods))
    
    total_logic = 0
    total_merged = 0
    
    for dtype in sorted(by_dtype.keys()):
        family = DTYPE_TO_FAMILY.get(dtype, "???")
        print(f"\n{dtype} -> {family}:")
        
        for transpose, is_gb, methods in sorted(by_dtype[dtype]):
            gb_tag = " [GB]" if is_gb else ""
            method_names = [m[0] for m in methods]
            target = get_logic_filename(dtype, transpose, is_gb)
            merge_note = " ← WILL MERGE" if len(methods) > 1 else ""
            print(f"  {transpose.upper()}{gb_tag}: {','.join(method_names)} -> {target}{merge_note}")
            total_logic += len(methods)
            total_merged += 1
    
    # What's missing for navi31 parity?
    print("\n\n=== Coverage Summary ===")
    print(f"  Logic files collected: {total_logic}")
    print(f"  Merged families:      {total_merged}")
    
    # navi31 has 10 families × 4 transposes = 40 files
    # gfx1201 target: navi31 parity + DB, DB_GB, BSS_BH for completeness
    target_families = {"HHS_BH", "HHS_BH_GB", "BBS_BH", "BBS_BH_GB", "HB", "HB_GB", "SB", "SB_GB", "I8II_BH", "I8II_BH_GB",
                       "DB", "DB_GB", "BSS_BH"}
    our_families = set()
    for (dtype, transpose, is_gb), _ in groups.items():
        family = DTYPE_TO_FAMILY.get(dtype, "")
        gb_suffix = "_GB" if is_gb else ""
        our_families.add(f"{family}{gb_suffix}")
    
    print(f"\n  Our families:    {sorted(our_families)}")
    print(f"  Target families: {sorted(target_families)}")
    
    missing = target_families - our_families
    extra = our_families - target_families
    if missing:
        print(f"\n  MISSING (target has, we don't): {sorted(missing)}")
    if extra:
        print(f"  EXTRA (we have beyond navi31): {sorted(extra)}")
    
    # Size coverage advantage
    print(f"\n=== Size Coverage vs AMD ===")
    print(f"  navi31: N/K max 8192 — NO coverage for dims > 8192")
    print(f"  gfx942: sparse, misses 14336, 18432 entirely")
    print(f"  Ours:   N/K includes 3584,7168,14336,16384,18432,28672")
    print(f"          ALL critical LLM dims covered (Llama-3, DeepSeek, Qwen, Mixtral)")


def merge_logic_files(groups, output_dir, dry_run=False):
    """Merge WMMA + VALU logic files for each problem type.
    
    When there's only one method for a problem type, just copy the file.
    When there are multiple (e.g. WMMA + VALU), use TensileMergeLibrary.
    """
    output_dir = Path(output_dir)
    
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n=== Merging to {output_dir} ===\n")
    
    results = {"copied": 0, "merged": 0, "failed": 0, "skipped": 0}
    
    for (dtype, transpose, is_gb), methods in sorted(groups.items()):
        target_name = get_logic_filename(dtype, transpose, is_gb)
        if not target_name:
            print(f"  SKIP: no target name for {dtype}/{transpose}")
            results["skipped"] += 1
            continue
        
        target_path = output_dir / target_name
        
        if len(methods) == 1:
            # Single source — just copy with rename
            method, src = methods[0]
            print(f"  COPY: {src.name} -> {target_name} ({method})")
            if not dry_run:
                # Read, update arch name, write
                with open(src) as f:
                    doc = yaml.safe_load(f)
                # Update arch name to gfx1201 convention (Tensile outputs "navi48")
                if isinstance(doc, list) and len(doc) > 1:
                    doc[1] = "gfx1201"
                with open(target_path, "w") as f:
                    yaml.dump(doc, f, default_flow_style=None, width=200)
            results["copied"] += 1
        
        else:
            # Multiple sources — need to merge
            # Strategy: use the first as base, merge the rest incrementally
            method_names = [m[0] for m in methods]
            print(f"  MERGE: {target_name} <- {method_names}")
            
            if dry_run:
                results["merged"] += 1
                continue
            
            # Create temp dirs for merge workflow
            import tempfile
            with tempfile.TemporaryDirectory() as tmpdir:
                # Put first source as "original"
                orig_dir = os.path.join(tmpdir, "original")
                os.makedirs(orig_dir)
                
                base_method, base_src = methods[0]
                # Standardize filename for merge tool (it matches by Cijk pattern)
                std_name = target_name.replace("gfx1201_", "navi48_")
                shutil.copy2(base_src, os.path.join(orig_dir, std_name))
                
                # Merge each additional source incrementally
                current_dir = orig_dir
                for method, src in methods[1:]:
                    inc_dir = os.path.join(tmpdir, f"inc_{method}")
                    os.makedirs(inc_dir)
                    shutil.copy2(src, os.path.join(inc_dir, std_name))
                    
                    out_dir = os.path.join(tmpdir, f"merged_{method}")
                    os.makedirs(out_dir)
                    
                    cmd = [
                        sys.executable,
                        str(MERGE_TOOL),
                        current_dir,
                        inc_dir,
                        out_dir,
                        "--force_merge", "true",
                        "-v", "0",
                    ]
                    
                    try:
                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                        if result.returncode != 0:
                            print(f"    MERGE FAILED: {result.stderr[:200]}")
                            results["failed"] += 1
                            break
                    except subprocess.TimeoutExpired:
                        print(f"    MERGE TIMEOUT")
                        results["failed"] += 1
                        break
                    
                    current_dir = out_dir
                else:
                    # All merges succeeded — copy result
                    merged_file = os.path.join(current_dir, std_name)
                    if os.path.exists(merged_file):
                        # Read, rename arch, write
                        with open(merged_file) as f:
                            doc = yaml.safe_load(f)
                        if isinstance(doc, list) and len(doc) > 1:
                            doc[1] = "gfx1201"
                        with open(target_path, "w") as f:
                            yaml.dump(doc, f, default_flow_style=None, width=200)
                        results["merged"] += 1
                    else:
                        print(f"    MERGE produced no output file")
                        results["failed"] += 1
    
    print(f"\n=== Results ===")
    print(f"  Copied:  {results['copied']}")
    print(f"  Merged:  {results['merged']}")
    print(f"  Failed:  {results['failed']}")
    print(f"  Skipped: {results['skipped']}")
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
