set pagination off
set args --config-file /home/iamthecage/rocm-libraries/shared/tensile/next-tuning/output_valu/1_BenchmarkProblems/Cijk_Ailk_Bljk_HHS_BH_00/00_Final/source/ClientParameters.ini

run

echo \n=== CRASH INFO ===\n
thread apply all bt 1
echo \n=== TRAP STATUS ===\n
p $trapsts
echo \n=== STATUS ===\n
p $status
echo \n=== SRD A base (s8:s9) ===\n
p/x $s8
p/x $s9
echo \n=== SRD LIMIT s10 ===\n
p/x $s10
echo \n=== ShadowLimitA[0] s0 ===\n
p/x $s0
p/x $s1
