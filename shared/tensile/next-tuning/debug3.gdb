set pagination off
set print pretty on
set args --config-file /home/iamthecage/rocm-libraries/shared/tensile/next-tuning/output_valu/1_BenchmarkProblems/Cijk_Ailk_Bljk_HHS_BH_00/00_Final/source/ClientParameters.ini

set amdgpu precise-memory on
set amdgpu precise-alu-exceptions on

run

echo \n=== CRASH PC (all waves) ===\n
thread apply all bt 1

echo \n=== FAULTING WAVE REGISTERS ===\n
p $pc
p/x $trapsts
p/x $status
p/x $mode
echo \n=== SRD A (s8-s11) ===\n
p/x $s8
p/x $s9
p/x $s10
p/x $s11
echo \n=== SRD B (s12-s15) ===\n
p/x $s12
p/x $s13
p/x $s14
p/x $s15
echo \n=== SRD C (s20-s23) ===\n
p/x $s20
p/x $s21
p/x $s22
p/x $s23
echo \n=== SRD D (s16-s19) ===\n
p/x $s16
p/x $s17
p/x $s18
p/x $s19
echo \n=== GRO_A (v165-v172) max per lane ===\n
p/x $v165
p/x $v172
echo \n=== LWA (v163-v164) ===\n
p/x $v163
p/x $v164
