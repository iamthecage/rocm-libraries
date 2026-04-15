set pagination off
set print pretty on
set args --config-file /home/iamthecage/rocm-libraries/shared/tensile/next-tuning/output_valu/1_BenchmarkProblems/Cijk_Ailk_Bljk_HHS_BH_00/00_Final/source/ClientParameters.ini

set amdgpu precise-memory on

run

# When we stop (GPU fault or signal), gather info
echo \n=== CRASH PC ===\n
where
echo \n=== ALL SGPRS ===\n
info all-registers
echo \n=== GRO VPGRS (v165-v180) for ALL waves ===\n
thread apply all p/x $v165
thread apply all p/x $v166
thread apply all p/x $v167
thread apply all p/x $v168
thread apply all p/x $v169
thread apply all p/x $v170
thread apply all p/x $v171
thread apply all p/x $v172
thread apply all p/x $v173
thread apply all p/x $v174
thread apply all p/x $v175
thread apply all p/x $v176
echo \n=== SrdA s[8:11] SrdB s[12:15] ===\n
p/x $s8
p/x $s9
p/x $s10
p/x $s11
p/x $s12
p/x $s13
p/x $s14
p/x $s15
