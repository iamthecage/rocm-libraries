set pagination off
set print pretty on
set args --config-file /home/iamthecage/rocm-libraries/shared/tensile/next-tuning/output_valu/1_BenchmarkProblems/Cijk_Ailk_Bljk_HHS_BH_00/00_Final/source/ClientParameters.ini

# Enable AMD GPU precise memory fault tracking
set amdgpu precise-memory on

run

# When we stop (GPU fault or signal), gather full info
bt full
echo \n=== GPU THREADS ===\n
info threads
echo \n=== FAULTING GPU THREAD ===\n
thread apply all bt
echo \n=== GPU REGISTERS ===\n
info registers
echo \n=== CURRENT FRAME LOCALS ===\n
info locals

