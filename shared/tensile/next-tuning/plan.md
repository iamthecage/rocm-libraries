# gfx1201 rocBLAS Tuning Pipeline — End-to-End Plan
# =================================================
#
# Dirs:
#   configs:  ~/rocm-libraries/shared/tensile/next-tuning/gfx1201-wide/
#   runner:   ~/rocm-libraries/shared/tensile/next-tuning/run_gfx1201_wide.sh
#   outputs:  ~/out/<config_name>/3_LibraryLogic/*.yaml
#   merge:    ~/rocm-libraries/shared/tensile/next-tuning/merge_gfx1201_logic.py
#   tensile:  ~/rocm-libraries/shared/tensile/
#   rocblas logic: ~/rocm-libraries/projects/rocblas/library/src/blas3/Tensile/Logic/asm_full/
#   installed lib: /opt/rocm/lib/rocblas/library/

# ============================================================
# PHASE 1: Run remaining benchmark configs (55 of 92 remaining)
# ============================================================
#
# Completed (37): all wmma standard + gb, bf16 valu standard + gb, bss_valu_nn, bss_wmma_*
# Known failures (skip these — 8 total):
#   bf16gemm_native_*     — Tensile rejects (B,B,B) "not supported yet"
#   bf16gemm_wmma_native_* — same rejection
#   hgemm_native_*        — KWA assertion: numVgprPerValuC = 0
#   hgemm_wmma_native_*   — same assertion
#
# That leaves 39 actionable configs to run.

# --- Step 1a: Finish VALU runs (--filter valu covers 32 configs, 24 remaining) ---
cd ~/rocm-libraries/shared/tensile/next-tuning
./run_gfx1201_wide.sh --filter valu

# Remaining VALU after current run completes:
#   bss_valu_{nt,tn,tt}           (3)
#   hgemm_valu_{nn,nt,tn,tt}     (4)
#   hgemm_valu_gb_{nn,nt,tn,tt}  (4)
#   hss_valu_{nn,nt,tn,tt}       (4)
#   hss_valu_gb_{nn,nt,tn,tt}    (4)
#   i8gemm_valu_gb_{nn,nt,tn,tt} (4)
# Total VALU remaining: 23 (+ bss_valu_nn already done = 24 done after)

# --- Step 1b: Run WMMA GB configs that weren't in the original wmma pass ---
# These are new (hss_wmma_gb_*) — added after the initial WMMA run
./run_gfx1201_wide.sh --filter hss_wmma_gb

# --- Step 1c: Run the "other" configs (no valu/wmma in name = implicit VALU) ---
# sgemm, dgemm, i8gemm (no method suffix = VALU-only families)
./run_gfx1201_wide.sh --filter sgemm
./run_gfx1201_wide.sh --filter dgemm
./run_gfx1201_wide.sh --filter hgemm_native
./run_gfx1201_wide.sh --filter i8gemm_n
./run_gfx1201_wide.sh --filter i8gemm_t
./run_gfx1201_wide.sh --filter bf16gemm_native
   # matches i8gemm_{nn,nt,tn,tt} but not i8gemm_wmma/valu_gb

# --- Step 1d: (optional) Re-run any failures ---
./run_gfx1201_wide.sh --rerun-failed

# ============================================================
# PHASE 2: Merge logic files (WMMA + VALU per problem type)
# ============================================================
#
# merge_gfx1201_logic.py:
#   - Collects ~/out/*/3_LibraryLogic/*.yaml
#   - Groups by (dtype, transpose, is_gb)
#   - Calls TensileMergeLibrary to merge WMMA+VALU for same problem type
#   - Renames navi48_* -> gfx1201_*
#   - Outputs to gfx1201-merged-logic/
#
# NOTE: Currently uses --force_merge true (incremental always wins).
#       Consider changing to --force_merge false to let per-size efficiency
#       comparison pick the genuinely faster kernel (WMMA vs VALU) at each size.

# --- Step 2a: Check status first ---
cd ~/rocm-libraries/shared/tensile/next-tuning
python3 merge_gfx1201_logic.py --status

# --- Step 2b: Dry run to preview ---
python3 merge_gfx1201_logic.py --dry-run

# --- Step 2c: Run the merge ---
python3 merge_gfx1201_logic.py --output-dir ~/rocm-libraries/shared/tensile/next-tuning/gfx1201-merged-logic

# Output: ~/rocm-libraries/shared/tensile/next-tuning/gfx1201-merged-logic/*.yaml
# Expected files (matching navi31 parity + extras):
#   gfx1201_Cijk_A*_B*_HHS_BH.yaml      (4 transposes)
#   gfx1201_Cijk_A*_B*_HHS_BH_GB.yaml   (4 transposes)
#   gfx1201_Cijk_A*_B*_BBS_BH.yaml      (4)
#   gfx1201_Cijk_A*_B*_BBS_BH_GB.yaml   (4)
#   gfx1201_Cijk_A*_B*_HB.yaml          (4)  — hss -> HB family
#   gfx1201_Cijk_A*_B*_HB_GB.yaml       (4)  — hss gb -> HB_GB
#   gfx1201_Cijk_A*_B*_BSS_BH.yaml      (4)  — bss -> BSS_BH
#   gfx1201_Cijk_A*_B*_SB.yaml          (4)  — sgemm -> SB
#   gfx1201_Cijk_A*_B*_I8II_BH.yaml     (4)  — i8gemm -> I8II_BH
#   gfx1201_Cijk_A*_B*_I8II_BH_GB.yaml  (4)  — i8gemm gb -> I8II_BH_GB
#   gfx1201_Cijk_A*_B*_DB.yaml          (4)  — dgemm -> DB (extra vs navi31)

# ============================================================
# PHASE 3: Compile library (TensileCreateLibrary)
# ============================================================
#
# TensileCreateLibrary takes logic YAMLs -> .co (code objects) + .dat (metadata)
# Must include hip/ fallback logic alongside our tuned logic so the output
# includes _fallback_ .hsaco files wired into the master lazy index.

# --- Step 3a: Assemble combined logic directory ---
mkdir -p /tmp/gfx1201_combined_logic/gfx1201
mkdir -p /tmp/gfx1201_combined_logic/hip

# Tuned logic (our merged WMMA+VALU output from Phase 2)
cp ~/rocm-libraries/shared/tensile/next-tuning/gfx1201-merged-logic/*.yaml \
   /tmp/gfx1201_combined_logic/gfx1201/

# Fallback logic (HIP source kernels — arch-independent, 108 files)
cp ~/rocm-libraries/projects/rocblas/library/src/blas3/Tensile/Logic/asm_full/hip/*.yaml \
   /tmp/gfx1201_combined_logic/hip/

# --- Step 3b: Run TensileCreateLibrary ---
cd ~/rocm-libraries/shared/tensile
python3 Tensile/bin/TensileCreateLibrary \
  /tmp/gfx1201_combined_logic/ \
  /tmp/gfx1201_library_out/ \
  HIP \
  --architecture=gfx1201 \
  --merge-files \
  --lazy-library-loading \
  --separate-architectures \
  --code-object-version=V5 \
  --library-format=msgpack \
  -j $(nproc)

# Output: /tmp/gfx1201_library_out/library/
#   TensileLibrary_lazy_gfx1201.dat         — master index (replaces existing)
#   TensileLibrary_Type_*_gfx1201.co        — tuned code objects (NEW)
#   TensileLibrary_Type_*_gfx1201.dat       — per-type metadata (NEW)
#   TensileLibrary_Type_*_fallback_gfx1201.hsaco — fallback kernels (replaces existing)

# ============================================================
# PHASE 4: Install to /opt/rocm (no rocBLAS rebuild needed)
# ============================================================
#
# rocBLAS runtime-loads the Tensile library. Just replacing the gfx1201
# files in /opt/rocm/lib/rocblas/library/ is sufficient.
# Existing fallback .hsaco files get replaced with our freshly-compiled set.
# New tuned .co/.dat pairs get added.
# The master TensileLibrary_lazy_gfx1201.dat index gets replaced with one
# that references both tuned kernels AND fallbacks.

# --- Step 4a: Backup existing gfx1201 files ---
sudo mkdir -p /opt/rocm/lib/rocblas/library/backup_gfx1201
sudo cp /opt/rocm/lib/rocblas/library/*gfx1201* \
        /opt/rocm/lib/rocblas/library/backup_gfx1201/

# --- Step 4b: Install new library files ---
sudo cp /tmp/gfx1201_library_out/library/TensileLibrary_lazy_gfx1201.dat \
        /opt/rocm/lib/rocblas/library/
sudo cp /tmp/gfx1201_library_out/library/*gfx1201*.co \
        /opt/rocm/lib/rocblas/library/
sudo cp /tmp/gfx1201_library_out/library/*gfx1201*.dat \
        /opt/rocm/lib/rocblas/library/
sudo cp /tmp/gfx1201_library_out/library/*fallback*gfx1201*.hsaco \
        /opt/rocm/lib/rocblas/library/

# --- Step 4c: Verify ---
ls /opt/rocm/lib/rocblas/library/*gfx1201* | grep -v fallback | grep -v backup
# Should now show TensileLibrary_Type_*_gfx1201.co + .dat pairs

# ============================================================
# PHASE 5: Validation
# ============================================================
#
# Quick smoke test with rocblas-bench or any GEMM workload to confirm
# tuned kernels are being dispatched instead of fallbacks.
# ROCBLAS_LAYER=2 will print kernel selection info.
#
# ROCBLAS_LAYER=2 rocblas-bench -f gemm -r h --transposeA N --transposeB N \
#   -m 4096 -n 4096 -k 4096 2>&1 | head -20
#
# If things break, restore backup:
# sudo cp /opt/rocm/lib/rocblas/library/backup_gfx1201/* \
#         /opt/rocm/lib/rocblas/library/

# ============================================================
# PHASE 6: Commit to source tree (for reproducible builds)
# ============================================================
#
# Copy merged logic to rocBLAS source tree for future builds:
#
# python3 merge_gfx1201_logic.py --install
#
# This copies to:
#   projects/rocblas/library/src/blas3/Tensile/Logic/asm_full/gfx1201/
#
# Then a standard rocBLAS build with Tensile_LOGIC=asm_full will include
# the gfx1201 tuned kernels alongside all other architectures.