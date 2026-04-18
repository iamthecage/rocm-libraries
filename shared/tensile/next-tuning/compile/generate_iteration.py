#!/usr/bin/env python3
"""
Generate optimized WMMA library logic YAMLs for compile→decompile→analyze loop.

Takes the existing hgemm_wmma_nt library logic YAML, creates two variants:
1. baseline/ - Original solutions (PGR=0, PLR=0, no scheduling)
2. optimized/ - Same solutions but with scheduling enabled (PGR=1, PLR=1, SIA=1, etc.)

Then TensileCreateLibrary compiles both, and we can decompile and compare.
"""

import yaml
import copy
import os
import sys

COMPILE_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_YAML = os.path.join(
    COMPILE_DIR,
    "hgemm_wmma_nt/3_LibraryLogic/navi48_Cijk_Ailk_Bjlk_HHS_BH.yaml"
)

def load_library_logic(path):
    """Load Tensile library logic YAML (multi-document)."""
    with open(path, 'r') as f:
        docs = list(yaml.safe_load_all(f))
    # Tensile uses a single-document YAML with a list of items
    # but some YAMLs have the structure as a list within one document
    return docs

def create_optimized_yaml(source_path, output_dir, variant_name, param_overrides):
    """
    Create a new library logic YAML with modified solution parameters.
    Only includes a few solutions (first 4) for fast compile.
    """
    with open(source_path, 'r') as f:
        content = f.read()

    # Parse the YAML - Tensile uses a single list with specific index meanings
    data = yaml.safe_load(content)

    # data structure:
    # [0] = {MinimumRequiredVersion: ...}
    # [1] = schedule name (str)
    # [2] = architecture (str)
    # [3] = device IDs (list)
    # [4] = ProblemType (dict)
    # [5] = solutions (list of dicts)
    # [6] = priority index order
    # [7] = ExactLogic (list of [size, [sol_idx, gflops]])
    # [8] = RangeLogic (null)
    # [9] = FallbackLogic (null)
    # [10] = efficiency name

    # Take first 4 solutions for fast iteration
    solutions = data[5][:4]

    # Apply parameter overrides to each solution
    for sol in solutions:
        for key, value in param_overrides.items():
            sol[key] = value
        # Force re-derivation
        sol['AssignedDerivedParameters'] = False
        sol['AssignedProblemIndependentDerivedParameters'] = False

    # Re-index solutions
    for i, sol in enumerate(solutions):
        sol['SolutionIndex'] = i

    # Create minimal ExactLogic - just a few representative sizes
    test_sizes = [
        [256, 4096, 1, 4096, 256, 256, 256, 4096],
        [1024, 4096, 1, 4096, 1024, 1024, 1024, 4096],
        [4096, 4096, 1, 4096, 4096, 4096, 4096, 4096],
    ]
    exact_logic = []
    for size in test_sizes:
        exact_logic.append([size, [0, 0.0]])  # Map all to solution 0

    # Build new YAML structure
    # architectureMap requires scheduleName == mapped arch name
    # For gfx1201, architectureMap["gfx1201"] = "gfx1201", so scheduleName must be "gfx1201"
    arch_name = data[2]  # "gfx1201"

    new_data = [
        data[0],        # version
        arch_name,      # schedule name (must match architectureMap[arch])
        data[2],        # architecture
        data[3],        # device IDs
        data[4],        # problem type
        solutions,      # modified solutions
        list(range(len(solutions))),  # priority order
        exact_logic,    # exact logic
        None,           # range logic
        None,           # fallback logic
        'DeviceEfficiency'
    ]

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{arch_name}_Cijk_Ailk_Bjlk_HHS_BH.yaml")

    with open(output_path, 'w') as f:
        yaml.dump(new_data, f, default_flow_style=None, width=200)

    print(f"[{variant_name}] Wrote {len(solutions)} solutions to {output_path}")
    return output_path


def main():
    print(f"Source YAML: {SOURCE_YAML}")

    # Create baseline (no scheduling - original params)
    baseline_dir = os.path.join(COMPILE_DIR, "iteration1", "baseline_logic")
    create_optimized_yaml(
        SOURCE_YAML, baseline_dir, "baseline",
        {
            'PrefetchGlobalRead': 0,
            'PrefetchLocalRead': 0,
            'ScheduleGlobalRead': 0,
            'ScheduleLocalWrite': 0,
            'ScheduleIterAlg': 0,
        }
    )

    # Create optimized variant 1: Enable all scheduling
    opt1_dir = os.path.join(COMPILE_DIR, "iteration1", "opt1_scheduling_logic")
    create_optimized_yaml(
        SOURCE_YAML, opt1_dir, "opt1_scheduling",
        {
            'PrefetchGlobalRead': 1,
            'PrefetchLocalRead': 1,
            'ScheduleGlobalRead': 1,
            'ScheduleLocalWrite': 1,
            'ScheduleIterAlg': 1,
        }
    )

    # Create optimized variant 2: Scheduling + DepthU=32
    opt2_dir = os.path.join(COMPILE_DIR, "iteration1", "opt2_deep_unroll_logic")
    create_optimized_yaml(
        SOURCE_YAML, opt2_dir, "opt2_deep_unroll",
        {
            'PrefetchGlobalRead': 1,
            'PrefetchLocalRead': 1,
            'ScheduleGlobalRead': 1,
            'ScheduleLocalWrite': 1,
            'ScheduleIterAlg': 1,
            'DepthU': 32,
        }
    )

    # Create optimized variant 3: SIA=3 (more aggressive interleaving)
    opt3_dir = os.path.join(COMPILE_DIR, "iteration1", "opt3_sia3_logic")
    create_optimized_yaml(
        SOURCE_YAML, opt3_dir, "opt3_sia3",
        {
            'PrefetchGlobalRead': 1,
            'PrefetchLocalRead': 1,
            'ScheduleGlobalRead': 1,
            'ScheduleLocalWrite': 1,
            'ScheduleIterAlg': 3,
        }
    )

    print("\nDone! Now compile each variant with TensileCreateLibrary:")
    print("  TENSILE_ROOT=/home/iamthecage/rocm-libraries/shared/tensile")
    print("  export PYTHONPATH=$TENSILE_ROOT")
    for name in ['baseline', 'opt1_scheduling', 'opt2_deep_unroll', 'opt3_sia3']:
        logic = os.path.join(COMPILE_DIR, f"iteration1/{name}_logic")
        build = os.path.join(COMPILE_DIR, f"iteration1/{name}_build")
        print(f"\n  # {name}")
        print(f"  $TENSILE_ROOT/Tensile/bin/TensileCreateLibrary \\")
        print(f"    {logic} \\")
        print(f"    {build} \\")
        print(f"    HIP \\")
        print(f"    --architecture=gfx1201 \\")
        print(f"    --no-short-file-names \\")
        print(f"    --code-object-version=default \\")
        print(f"    --cxx-compiler=/opt/rocm/bin/amdclang++ \\")
        print(f"    --jobs=$(nproc)")


if __name__ == '__main__':
    main()
