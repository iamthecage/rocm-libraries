import re

target_file = '/home/iamthecage/rocm-libraries/shared/tensile/next-tuning/gfx1201-wide-v2/generate_configs.py'

with open(target_file, 'r') as f:
    content = f.read()

# We want to add missing configs for bss_wmma_gb, bss_valu_gb, i8gemm_wmma_gb,
# as well as their _gb_fixed variants.

new_blocks = """
        # --- Newly added missing GB blocks ---
        yield (
            f"bss_wmma_gb_{tsuffix}.yaml",
            build_yaml(
                problem_type_wmma_bss(ta, tb, tdesc + " [grouped-batch / MoE]"),
                COMMON_PARAMS_WMMA,
                wmma_fork_params(),
                sizes_gb,
                f"bss_wmma_gb_{tsuffix}",
            )
        )

        yield (
            f"bss_valu_gb_{tsuffix}.yaml",
            build_yaml(
                problem_type_valu_bss(ta, tb, tdesc + " [grouped-batch / MoE]"),
                COMMON_PARAMS_VALU,
                valu_fork_params_hpa(),
                sizes_gb,
                f"bss_valu_gb_{tsuffix}",
            )
        )

        yield (
            f"i8gemm_wmma_gb_{tsuffix}.yaml",
            build_yaml(
                problem_type_wmma_i8gemm(ta, tb, tdesc + " [grouped-batch / MoE]"),
                COMMON_PARAMS_WMMA,
                wmma_fork_params_i8(),
                sizes_i8_gb,
                f"i8gemm_wmma_gb_{tsuffix}",
            )
        )

        # --- Newly added missing GB FIXED blocks (StridedBatched: False) ---
        yield (
            f"bss_wmma_gb_fixed_{tsuffix}.yaml",
            build_yaml(
                problem_type_wmma_bss(ta, tb, tdesc + " [grouped-batch / MoE]", strided_batched=False),
                COMMON_PARAMS_WMMA,
                wmma_fork_params(),
                sizes_gb,
                f"bss_wmma_gb_fixed_{tsuffix}",
            )
        )

        yield (
            f"bss_valu_gb_fixed_{tsuffix}.yaml",
            build_yaml(
                problem_type_valu_bss(ta, tb, tdesc + " [grouped-batch / MoE]", strided_batched=False),
                COMMON_PARAMS_VALU,
                valu_fork_params_hpa(),
                sizes_gb,
                f"bss_valu_gb_fixed_{tsuffix}",
            )
        )

        yield (
            f"i8gemm_wmma_gb_fixed_{tsuffix}.yaml",
            build_yaml(
                problem_type_wmma_i8gemm(ta, tb, tdesc + " [grouped-batch / MoE]", strided_batched=False),
                COMMON_PARAMS_WMMA,
                wmma_fork_params_i8(),
                sizes_i8_gb,
                f"i8gemm_wmma_gb_fixed_{tsuffix}",
            )
        )
"""

# Find where to inject. Right before "    # Complex GEMM layouts include conjugate-transpose variants."
injection_point = "    # Complex GEMM layouts include conjugate-transpose variants."

if injection_point in content:
    content = content.replace(injection_point, new_blocks + "\n" + injection_point)
    with open(target_file, 'w') as f:
        f.write(content)
    print("Successfully patched generate_configs.py")
else:
    print("Failed to find injection point.")

