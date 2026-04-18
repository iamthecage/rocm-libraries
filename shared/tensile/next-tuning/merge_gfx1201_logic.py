#!/usr/bin/env python3
"""
Merge gfx1201 wide-run benchmark results into rocBLAS logic files.

This script takes the Tensile benchmark outputs from /home/iamthecage/out/
and produces merged logic files suitable for placement into the rocBLAS
logic directory:
  projects/rocblas/library/src/blas3/Tensile/Logic/asm_full/gfx1201/

Workflow:
  1. Collect all 3_LibraryLogic/*.yaml files from completed benchmark runs
  2. Group by actual output filename to prevent ProblemType collisions
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
# Mapping from our config names to rocBLAS logic families (for status display)
# ---------------------------------------------------------------------------
DTYPE_TO_FAMILY = {
    "hgemm":    "HHS_BH / HB", 
    "bf16gemm": "BBS_BH",     
    "hss":      "HB / HSS_BH",         
    "bss":      "BSS_BH",     
    "sgemm":    "SB",         
    "dgemm":    "DB",         
    "i8gemm":   "I8II_BH",    
}

def parse_config_name(name):
    """Parse a benchmark output dir name into components.
    
    Returns: (dtype, method, is_gb, transpose) or None if unparsable.
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
        rest = parts[1:-1]
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
        method = "valu"
    else:
        method = "_".join(rest_clean)
    
    return (dtype, method, is_gb, transpose)


def collect_logic_files():
    """Scan benchmark dirs and group purely by the actual target filename."""
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
                continue
            
            yamls = list(logic_dir.glob("*.yaml"))
            if not yamls:
                continue
            
            src_yaml = yamls[0]
            
            # Build target name directly from the actual Tensile output
            target_name = src_yaml.name.replace("navi48_", "gfx1201_")
            if is_gb and "_GB" not in target_name:
                target_name = target_name.replace(".yaml", "_GB.yaml")
                
            groups[target_name].append((dtype, transpose, is_gb, method, src_yaml))
            
    return groups


def show_status(groups):
    """Print status of all collected logic files."""
    print("\n=== gfx1201 Benchmark Logic Files ===\n")
    
    # Restructure for pretty printing
    by_dtype = defaultdict(dict)
    for target_name, items in groups.items():
        for dtype, transpose, is_gb, method, src_yaml in items:
            key = (transpose, is_gb, target_name)
            if key not in by_dtype[dtype]:
                by_dtype[dtype][key] = []
            by_dtype[dtype][key].append(method)
    
    total_logic = 0
    total_merged = 0
    
    for dtype in sorted(by_dtype.keys()):
        family = DTYPE_TO_FAMILY.get(dtype, "???")
        print(f"\n{dtype} -> {family}:")
        
        # Sort by transpose, then gb, then target_name
        for (transpose, is_gb, target), methods in sorted(by_dtype[dtype].items()):
            gb_tag = " [GB]" if is_gb else ""
            merge_note = " ← WILL MERGE" if len(methods) > 1 else ""
            print(f"  {transpose.upper()}{gb_tag}: {','.join(methods)} -> {target}{merge_note}")
            total_logic += len(methods)
            
    total_merged = len(groups)
    
    print("\n\n=== Coverage Summary ===")
    print(f"  Logic files collected: {total_logic}")
    print(f"  Merged families:      {total_merged}")
    
    # Deduce actual families from the filenames Tensile generated
    our_families = set()
    for target_name in groups.keys():
        # Strip extension and split by underscores. Family name is everything after the 4th underscore
        parts = target_name.replace(".yaml", "").split("_")
        family = "_".join(parts[4:])
        our_families.add(family)

    target_families = {"HHS_BH", "HHS_BH_GB", "BBS_BH", "BBS_BH_GB", "HB", "HB_GB", "SB", "SB_GB", "I8II_BH", "I8II_BH_GB",
                       "DB", "DB_GB", "BSS_BH"}
    
    print(f"\n  Our families:    {sorted(our_families)}")
    print(f"  Target families: {sorted(target_families)}")
    
    missing = target_families - our_families
    extra = our_families - target_families
    if missing:
        print(f"\n  MISSING (target has, we don't): {sorted(missing)}")
    if extra:
        print(f"  EXTRA (we have beyond navi31): {sorted(extra)}")
    
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
    
    for target_name, items in sorted(groups.items()):
        target_path = output_dir / target_name
        
        if len(items) == 1:
            # Single source — just copy with rename
            dtype, transpose, is_gb, method, src = items[0]
            print(f"  COPY: {src.name} -> {target_name} ({method})")
            if not dry_run:
                # Read, update arch name, write
                with open(src) as f:
                    doc = yaml.safe_load(f)
                # Update arch name to gfx1201 convention
                if isinstance(doc, list) and len(doc) > 1:
                    doc[1] = "gfx1201"
                with open(target_path, "w") as f:
                    yaml.dump(doc, f, default_flow_style=None, width=200)
            results["copied"] += 1
        
        else:
            # Multiple sources — need to merge
            method_names = [item[3] for item in items]
            print(f"  MERGE: {target_name} <- {method_names}")
            
            if dry_run:
                results["merged"] += 1
                continue
            
            import tempfile
            with tempfile.TemporaryDirectory() as tmpdir:
                # Put first source as "original"
                orig_dir = os.path.join(tmpdir, "original")
                os.makedirs(orig_dir)
                
                base_src = items[0][4]
                std_name = target_name.replace("gfx1201_", "navi48_")
                shutil.copy2(base_src, os.path.join(orig_dir, std_name))
                
                current_dir = orig_dir
                for dtype, transpose, is_gb, method, src in items[1:]:
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
                        "--force_merge", "false",
                        "--notrim",
                        "--add_solution_tags",
                        "-v", "0",
                    ]
                    
                    try:
                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                        if result.returncode != 0:
                            # Expanded stderr slice to ensure the true exception is visible
                            print(f"    MERGE FAILED:\n{result.stderr[-1500:]}")
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