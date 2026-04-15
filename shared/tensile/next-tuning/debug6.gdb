set pagination off
set amdgpu precise-memory on

set args --config-file /home/iamthecage/rocm-libraries/shared/tensile/next-tuning/output_valu/1_BenchmarkProblems/Cijk_Ailk_Bljk_HHS_BH_00/00_Final/source/ClientParameters.ini

# Use a breakpoint at s_wait_loadcnt *before* any crash
# Label ShadowInitStart_10 is at VMA 0x2198 relative to start of kernel code
# s_wait_loadcnt at VMA 0x2194 = ShadowInitStart_10 - 4

# Actually: break just before the first buffer_load (VMA 0x208C)
# ShadowInitStart_10 is at global VMA 0x7ffff5e6a198
# VMA 0x208C = ShadowInitStart_10 - 0x10C

# Instead of VMA math, let's break at kernel entry and step to the loads
# Try breakpoint at full kernel function name
break Cijk_Ailk_Bljk_HHS_BH_MT128x128x16_MI16x16x16x1_SN_K1

commands 1
  echo \n=== AT KERNEL ENTRY (no step) ===\n
  printf "PC = %p\n", $pc
  # Print initial SGPRs
  printf "s0=%x s1=%x s5=%x s7=%x s8=%x s9=%x s10=%x s11=%x\n", $s0,$s1,$s5,$s7,$s8,$s9,$s10,$s11
  continue
end

run

echo \n=== CRASH INFO ===\n
printf "PC = %p\n", $pc

echo \n=== SRD A ===\n
printf "s8  = %08x (SrdA[0] base lo)\n", $s8
printf "s9  = %08x (SrdA[1] base hi)\n", $s9
printf "s10 = %08x (SrdA[2] NumRecords)\n", $s10
printf "s11 = %08x (SrdA[3] format)\n", $s11

echo \n=== ShadowLimitA ===\n
printf "s0 = %08x\n", $s0
printf "s1 = %08x\n", $s1

echo \n=== GRO for A (v165-v172, lane 0) ===\n
printf "v165[0]=%x\n", $v165.s[0]
printf "v166[0]=%x\n", $v166.s[0]
printf "v167[0]=%x\n", $v167.s[0]
printf "v168[0]=%x\n", $v168.s[0]
printf "v169[0]=%x\n", $v169.s[0]
printf "v170[0]=%x\n", $v170.s[0]
printf "v171[0]=%x\n", $v171.s[0]
printf "v172[0]=%x\n", $v172.s[0]

echo \n=== GRO for A lane 31 (highest lane) ===\n
printf "v165[31]=%x\n", $v165.s[31]
printf "v166[31]=%x\n", $v166.s[31]
printf "v172[31]=%x\n", $v172.s[31]

echo \n=== excp_flag_priv ===\n
p $excp_flag_priv

quit
