# gfx1201 (RX 9070 XT) Tensile Kernel Optimization Analysis

## Hardware Profile
- **GPU**: RX 9070 XT (navi48), gfx1201, 32 CUs
- **Wave**: Wave32
- **VGPRs**: 256 per SIMD
- **LDS**: 65536 bytes per workgroup
- **ROCm**: 7.2.1
- **Tensile**: 4.47.0

---

## Current Tuned Kernel Summary (32 kernels, 8 dtype families)

| Type | Tile | VGPRs | Occ | Primary Compute | WMMA % |
|------|------|-------|-----|-----------------|--------|
| HH_HPA | MT128x64x8, MT64x64x8 | 38-58 | 50-80% | v_wmma_f32_16x16x16_f16 | 100% |
| HH native | MT32x32x8 - MT64x64x8 | 20-30 | 100% | v_pk_fma_f16 | 0% |
| SS | MT32x32x8 | 26 | 100% | v_fmac_f32 | 0% |
| DD | MT64x64x4 | 79 | 38% | v_fma_f64 | 0% |
| BB_HPA | MT32x32x8, MT32x64x8 | 26-34 | 88-100% | v_fma_f32 + v_wmma_bf16 | ~2% |
| BS_HPA | MT128x64x8, MT32x64x8 | 34-70 | 43-88% | v_fma_f32 + v_wmma_bf16 | ~2% |
| HS_HPA | MT32x32x8 - MT64x64x8 | 22-38 | 80-100% | v_fmac_f32 + v_wmma_f16 | ~10% |
| I8I_HPA | MT64x64x16 | 49-50 | 63% | v_mad_i32_i24 + v_wmma_iu8 | ~6% |

**ALL 32 kernels share**: PGR1, PLR1, SLW1, SGR1, SIA1, WG16_16_1, WS32, GSU1, zero spills, _preloaded

---

## Critical Finding: WMMA Fork is Handicapped

### The Problem
The `generate_configs.py` WMMA fork disables ALL scheduling/prefetch:
```
WMMA fork:  PGR=0, PLR=0, SLW=0, SGR=0, SIA=0  ← "still under investigation"
VALU fork:  PGR=1, PLR=1, SLW=1, SGR=1, SIA=1  ← fully enabled
```

### Why It Matters
- **NO code-level rejections** in SolutionStructs.py for WMMA + PGR/PLR/scheduling
- Tensile freely accepts WMMA solutions with scheduling enabled
- The WMMA fork was benchmarked without the features that make the VALU fork fast
- VALU wins every time because it hides memory latency; WMMA doesn't try to

### Evidence
- All 25 solutions in `hgemm_wmma_nt/3_LibraryLogic/` have PGR=0, PLR=0, SIA=0
- These are pure WMMA solutions but with no scheduling — they stall on every memory op
- The winning VALU kernels (from production tuning) ALL have PGR=1, PLR=1, SIA=1

---

## Optimization Opportunities

### HIGH CONFIDENCE — Enable scheduling for WMMA
**Change**: Set PGR=1, PLR=1, SLW=1, SGR=1, SIA=1 for WMMA solutions
**Expected**: 20-50% improvement. WMMA compute is fast but memory stalls dominate without scheduling.
**Risk**: Low. No rejections in code. The VALU fork proves these params work on gfx12.
**Status**: 🔄 TESTING (Iteration 1)

### HIGH CONFIDENCE — Try larger WMMA tiles
**Change**: MT128x64, MT64x128 with WMMA + scheduling
**Expected**: Better compute-to-memory ratio. More work per wave = less scheduling overhead.
**Risk**: Low. MT128x64 already works for BS_HPA and HH_HPA VALU kernels.

### MEDIUM CONFIDENCE — ScheduleIterAlg variations
**Change**: Try SIA=2 and SIA=3 for WMMA (different interleaving strategies)
**Expected**: SIA=3 keeps instructions in order but fills bubbles more aggressively.
**Risk**: Medium. May increase VGPR pressure. SIA>1 needs more register file.

### MEDIUM CONFIDENCE — Deeper DepthU for WMMA
**Change**: Try DepthU=32, 64 (currently 16)
**Expected**: More unrolling = more compute per global load = better latency hiding.
**Risk**: Medium. Increases LDS usage and register pressure.

### LOW CONFIDENCE — WMMA V2 non-HPA fix
**Change**: Fix SolutionStructs.py:3155 to allow native f16/bf16 WMMA accumulation
**Expected**: Halve accumulator VGPRs, enable denser compute.
**Risk**: High. Store path bug ("numVgprPerValuC=0") needs proper fix.
**Blocked**: Requires Tensile code changes, not just config tuning.

---

## Experiment Log

### Iteration 1: WMMA + Scheduling (baseline vs optimized)
- **Date**: (in progress)
- **Config**: HH_HPA NT, MI16x16x16x1, MT64x64
- **Baseline**: PGR=0, PLR=0, SLW=0, SGR=0, SIA=0
- **Optimized**: PGR=1, PLR=1, SLW=1, SGR=1, SIA=1
- **Method**: TensileCreateLibrary → RGA decompile → instruction comparison
- **Results**: (pending)

---

## File References
- generate_configs.py: `shared/tensile/next-tuning/gfx1201-wide/generate_configs.py`
- Baseline YAML: `compile/hgemm_wmma_nt/3_LibraryLogic/navi48_Cijk_Ailk_Bjlk_HHS_BH.yaml`
- KernelWriter.py: `shared/tensile/Tensile/KernelWriter.py` (waitcnt conversion at L5374)
- KernelWriterAssembly.py: `shared/tensile/Tensile/KernelWriterAssembly.py` (mfmaIter at L7298)
- SolutionStructs.py: `shared/tensile/Tensile/SolutionStructs.py` (WMMA V2 rejection at L3155)
- AsmCaps.py: `shared/tensile/Tensile/AsmCaps.py` (gfx12 caps, KernargPreloading=False stale)
