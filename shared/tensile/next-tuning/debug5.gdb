set pagination off
set amdgpu precise-memory on
set args --config-file /home/iamthecage/rocm-libraries/shared/tensile/next-tuning/output_valu/1_BenchmarkProblems/Cijk_Ailk_Bljk_HHS_BH_00/00_Final/source/ClientParameters.ini

# Break at the buffer_load_b32 for A[0] (VMA 0x208C in preamble)
# kernel is: Cijk_Ailk_Bljk_HHS_BH_MT128x128x16_MI16x16x16x1_SN_K1
# We use a symbol+offset break
# ShadowInitStart_10 is at 0x2198
# buffer_loads start at 0x208C = ShadowInitStart_10 - (0x2198 - 0x208C) = ShadowInitStart_10 - 0x10C
# = ShadowInitStart_10 - 268

break ShadowInitStart_10-268

commands 1
  echo \n=== PRE-LOAD STATE (lane 0) ===\n
  echo === SrdA s[8:11] ===\n
  p/x $s8
  p/x $s9
  p/x $s10
  p/x $s11
  echo === SrdB s[12:15] ===\n
  p/x $s12
  p/x $s13
  p/x $s14
  p/x $s15
  echo === ShadowLimitA s[0:1] ===\n
  p/x $s0
  p/x $s1
  echo === StaggerUIter s7 ===\n
  p/x $s7
  echo === LoopCounterL s5 ===\n
  p/x $s5
  echo === GRO A vgpr165:172 (offsets for A loads) ===\n
  p/x $v165
  p/x $v166
  p/x $v167
  p/x $v168
  p/x $v169
  p/x $v170
  p/x $v171
  p/x $v172
  echo === GRO B vgpr173:180 (offsets for B loads) ===\n
  p/x $v173
  p/x $v174
  p/x $v175
  p/x $v176
  p/x $v177
  p/x $v178
  p/x $v179
  p/x $v180
  continue
end

run

echo \n=== CRASH INFO ===\n
p $pc
echo \n=== PC offset from ShadowInitStart_10 ===\n
p/x (unsigned long long)$pc - (unsigned long long)&ShadowInitStart_10

echo \n=== ALL SGPRS ===\n
info registers
