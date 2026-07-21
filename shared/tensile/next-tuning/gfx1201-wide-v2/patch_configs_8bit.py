import re
import os

target_file = '/home/iamthecage/rocm-libraries/shared/tensile/next-tuning/gfx1201-wide-v2/generate_configs_8bit.py'

with open(target_file, 'r') as f:
    content = f.read()

# 1. Update function signatures to add strided_batched=True
content = re.sub(r'(def problem_type_[a-zA-Z0-9_]*\([^)]*)(\)):', r'\1, strided_batched=True\2:', content)

# 2. Update Batched: True""" to include StridedBatched
content = re.sub(r'(\s+Batched:\s+True)"""', r'\1\n      StridedBatched: {str(strided_batched)}"""', content)

# 3. We need to duplicate the 8-bit loop for _gb and _gb_fixed.
# The loop looks like:
#         # ── 8-bit WMMA types (F8/B8/F8B8/B8F8 in, F8/B8/S/H out) ─────────────────
#         for in_type in ["f8", "B8", "F8B8", "B8F8"]:
#             ...
#             yield (...)

start_marker = "# ── 8-bit WMMA types (F8/B8/F8B8/B8F8 in, F8/B8/S/H out) ─────────────────"

if start_marker in content:
    # Find the end of this block, which is just before the end of the configs() function.
    # We can just search for the end of the configs function.
    end_marker = "def main():"
    
    start_idx = content.find(start_marker)
    end_idx = content.find(end_marker)
    
    loop_block = content[start_idx:end_idx]
    
    # We will generate three versions: standard, gb, gb_fixed
    # The standard is already in loop_block.
    
    gb_block = loop_block.replace(
        'cfg_name = f"{name_in}_to_{name_out}_wmma_{tsuffix}"',
        'cfg_name = f"{name_in}_to_{name_out}_wmma_gb_{tsuffix}"'
    ).replace(
        'sizes_i8,',
        'problem_sizes_block(M_I8_GB, N_I8_GB, K_I8_GB),'
    ).replace(
        'desc = f"{in_type} -> {out_type} HPA WMMA V2"',
        'desc = f"{in_type} -> {out_type} HPA WMMA V2 [grouped-batch / MoE]"'
    ).replace(
        '# ── 8-bit WMMA types',
        '# ── 8-bit WMMA GB types'
    )
    
    gb_fixed_block = loop_block.replace(
        'cfg_name = f"{name_in}_to_{name_out}_wmma_{tsuffix}"',
        'cfg_name = f"{name_in}_to_{name_out}_wmma_gb_fixed_{tsuffix}"'
    ).replace(
        'sizes_i8,',
        'problem_sizes_block(M_I8_GB, N_I8_GB, K_I8_GB),'
    ).replace(
        'desc = f"{in_type} -> {out_type} HPA WMMA V2"',
        'desc = f"{in_type} -> {out_type} HPA WMMA V2 [grouped-batch / MoE]"'
    ).replace(
        '# ── 8-bit WMMA types',
        '# ── 8-bit WMMA GB FIXED types'
    )
    # add strided_batched=False to problem_type_wmma_8bit calls in gb_fixed_block
    gb_fixed_block = re.sub(
        r'(problem_type_wmma_8bit\([^)]*)\)',
        r'\1, strided_batched=False)',
        gb_fixed_block
    )
    
    new_content = content[:start_idx] + loop_block + "\n" + gb_block + "\n" + gb_fixed_block + "\n\n" + content[end_idx:]
    
    with open(target_file, 'w') as f:
        f.write(new_content)
