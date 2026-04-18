# Tensile/rocBLAS Tuning Gap Analysis — gfx1201 (RX 9070 XT)

**Date:** 2026-04-17  
**Data source:** `tensile_ab_results/20260417_181144/` (48-shape expanded benchmark)  
**Libraries compared:**
- **Library A (original):** `/home/iamthecage/library_backup/original` — default ROCm 6.4 fallback library (56 gfx1201 .hsaco files)
- **Library B (new):** `/home/iamthecage/library_backup/new` — custom tuned library (32 tuned .co + 120 gfx1201 files total)

**Hardware:** RX 9070 XT, 32 CU, gfx1201, ROCm 7.2.1, ~107 TFlops bf16 theoretical peak  
**Config generator:** `~/rocm-libraries/shared/tensile/next-tuning/gfx1201-wide/generate_configs.py`

---

## Executive Summary

48 shapes tested. Library B wins 46/48, mean speedup +864%. Two regressions found.  
**Three critical tuning gaps identified that leave 30–90% of peak performance on the table.**

| Gap | Severity | Affected shapes | Root cause |
|-----|----------|----------------|------------|
| M=2048 cliff | **CRITICAL** | bf16/f16/i8 all transposes | M=2048 absent from `M_STD` / `M_I8` |
| M=8192 cliff | **HIGH** | bf16 NN | M=8192 absent from `M_STD` |
| i8 M=2048 regression | **HIGH** | i8_r NN 2048×4096×4096 | Bad kernel selection — Library B -27.3% vs A |
| Large-K asymmetry | **MEDIUM** | 256×8192×28672 vs swapped | DepthU too small for K=28672 |

---

## 1. Full Results Table

### Library B (tuned) — absolute performance and efficiency

Theoretical peaks: bf16/f16 ~107 TF, i8 estimated ~107+ TOPS (WMMA), f32 via VALU only.

| M | N | K | tr | dtype | GFlops B | TFlops | % peak | µs B | vs Lib A |
|---|---|---|-----|-------|----------|--------|--------|------|----------|
| 1 | 4096 | 4096 | NN | bf16 | 182 | 0.2 | — | 184 | +294% |
| 1 | 4096 | 14336 | NN | bf16 | 76 | 0.1 | — | 1544 | +66% |
| 8 | 4096 | 4096 | NN | bf16 | 2170 | 2.2 | 2% | 124 | +487% |
| 8 | 4096 | 14336 | NN | bf16 | 1154 | 1.2 | 1% | 814 | +214% |
| 16 | 4096 | 4096 | NN | bf16 | 4336 | 4.3 | 4% | 124 | +486% |
| 16 | 14336 | 14336 | NN | bf16 | 4839 | 4.8 | 5% | 1359 | +493% |
| 32 | 4096 | 4096 | NN | bf16 | 8927 | 8.9 | 8% | 120 | +504% |
| 32 | 4096 | 14336 | NN | bf16 | 4590 | 4.6 | 4% | 819 | +214% |
| 64 | 4096 | 4096 | NN | bf16 | 12349 | 12.3 | 12% | 174 | +319% |
| 64 | 4096 | 14336 | NN | bf16 | 5656 | 5.7 | 5% | 1329 | +102% |
| 96 | 4096 | 4096 | NN | bf16 | 25931 | 25.9 | 24% | 124 | +288% |
| 128 | 4096 | 4096 | NN | bf16 | 34961 | 35.0 | 33% | 123 | +294% |
| 128 | 4096 | 4096 | NT | bf16 | 44182 | 44.2 | 41% | 97 | **+6719%** |
| 256 | 4096 | 4096 | NN | bf16 | 41042 | 41.0 | 38% | 209 | +326% |
| 256 | 4096 | 4096 | TT | bf16 | 40414 | 40.4 | 38% | 213 | +520% |
| 256 | 7168 | 7168 | NN | bf16 | 44387 | 44.4 | 41% | 593 | +340% |
| 256 | 8192 | 28672 | NN | bf16 | 12926 | 12.9 | 12% | 9304 | +29% |
| 256 | 14336 | 4096 | NN | bf16 | 46598 | 46.6 | 44% | 645 | +351% |
| 256 | 28672 | 8192 | NN | bf16 | 39366 | 39.4 | 37% | 3055 | +280% |
| 384 | 4096 | 4096 | NN | bf16 | 43138 | 43.1 | 40% | 299 | +332% |
| 512 | 4096 | 4096 | NN | bf16 | 44337 | 44.3 | 41% | 387 | +338% |
| 512 | 4096 | 4096 | NT | bf16 | 55858 | 55.9 | 52% | 308 | +6160% |
| 512 | 4096 | 14336 | NN | bf16 | 38600 | 38.6 | 36% | 1558 | +288% |
| 1024 | 3584 | 3584 | NN | bf16 | 48420 | 48.4 | 45% | 543 | +368% |
| 1024 | 4096 | 4096 | NN | bf16 | 49086 | 49.1 | 46% | 700 | +374% |
| 1024 | 4096 | 4096 | NT | bf16 | 62202 | 62.2 | 58% | 552 | +6442% |
| 1024 | 4096 | 4096 | TN | bf16 | 40244 | 40.2 | 38% | 854 | +348% |
| 1024 | 4096 | 4096 | TT | bf16 | 48722 | 48.7 | 46% | 705 | +633% |
| **2048** | **4096** | **4096** | **NN** | **bf16** | **12711** | **12.7** | **12%** | **5407** | **+21%** |
| **2048** | **4096** | **4096** | **NT** | **bf16** | **13013** | **13.0** | **12%** | **5281** | **+1381%** |
| 2048 | 2048 | 2048 | NN | bf16 | 47667 | 47.7 | 45% | 360 | +363% |
| 4096 | 4096 | 4096 | NN | bf16 | 54210 | 54.2 | 51% | 2535 | +412% |
| **8192** | **4096** | **4096** | **NN** | **bf16** | **13152** | **13.2** | **12%** | **20901** | **+25%** |
| 32 | 4096 | 4096 | NN | f16 | 11623 | 11.6 | 11% | 92 | +585% |
| 256 | 4096 | 4096 | NN | f16 | 42214 | 42.2 | 39% | 204 | +297% |
| 1024 | 4096 | 4096 | NN | f16 | 49156 | 49.2 | 46% | 699 | +341% |
| 1024 | 4096 | 4096 | NT | f16 | 62864 | 62.9 | 59% | 547 | +6709% |
| **2048** | **4096** | **4096** | **NN** | **f16** | **18939** | **18.9** | **18%** | **3628** | **+68%** |
| 4096 | 4096 | 4096 | NN | f16 | 51061 | 51.1 | 48% | 2692 | +351% |
| 256 | 4096 | 4096 | NN | f32 | 12643 | 12.6 | — | 679 | +54% |
| 1024 | 1024 | 1024 | NN | f32 | 13456 | 13.5 | — | 160 | +28% |
| 2048 | 2048 | 2048 | NN | f32 | 11290 | 11.3 | — | 1522 | **-4.4%** |
| 256 | 4096 | 4096 | NN | i8 | 49291 | 49.3 | — | 174 | +804% |
| 256 | 14336 | 4096 | NN | i8 | 59510 | 59.5 | — | 505 | +697% |
| 1024 | 4096 | 4096 | NN | i8 | 52841 | 52.8 | — | 650 | +603% |
| 1024 | 4096 | 4096 | NT | i8 | 71801 | 71.8 | — | 479 | +446% |
| **2048** | **4096** | **4096** | **NN** | **i8** | **5845** | **5.8** | **—** | **11757** | **-27.3%** |
| 4096 | 4096 | 4096 | NN | i8 | 66151 | 66.2 | — | 2078 | +698% |

**Bold rows = performance anomalies / tuning gaps.**

---

## 2. Critical Gap: M=2048 Performance Cliff

The M=2048 cliff is the single largest tuning deficiency. It affects **every dtype and transpose** tested:

| Shape (M=2048, N=4096, K=4096) | Library B TFlops | Expected TF (interpolated M=1024↔4096) | Actual % of expected |
|-------------------------------|-----------------|----------------------------------------|---------------------|
| bf16 NN | 12.7 TF | ~51 TF | **25%** |
| bf16 NT | 13.0 TF | ~58 TF | **22%** |
| f16 NN | 18.9 TF | ~50 TF | **38%** |
| i8 NN | 5.8 TOPS | ~60 TOPS | **10%** |

**Note:** 2048×2048×2048 NN bf16 runs at 47.7 TF, which is excellent. The problem is specifically when M=2048 and N×K are large (4096×4096). This suggests the kernel selection heuristic is choosing a poorly-suited tile for this specific M×N×K ratio.

**Root cause:** `M_STD = [16, 64, 256, 1024, 4096]` — M=2048 was never tuned. The heuristic falls back to a kernel optimized for M=1024 or M=4096, and neither tile size maps efficiently to M=2048 with large NK.

**Fix:** Add M=2048 to `M_STD` and `M_I8` in `generate_configs.py`.

---

## 3. Critical Gap: M=8192 Performance Cliff

| Shape | Library B TFlops | Expected | Actual % |
|-------|-----------------|----------|----------|
| 8192×4096×4096 NN bf16 | 13.2 TF | ~54 TF (same as M=4096) | **24%** |

Same root cause: M=8192 not in `M_STD`. The kernel tuned for M=4096 doesn't generalize to 2× the M dimension.

**Fix:** Add M=8192 to `M_STD` and `M_I8` in `generate_configs.py`.

---

## 4. Regression: i8 M=2048 (-27.3%)

| Shape | Library A (fallback) | Library B (tuned) | Delta |
|-------|---------------------|-------------------|-------|
| 2048×4096×4096 NN i8 | 8043 GOPS | 5845 GOPS | **-27.3%** |

Library B's tuned selection logic is actively picking a **worse** kernel than the fallback for this shape. This is a selection regression — the logic file maps M=2048 i8 to a kernel optimized for a different shape that happens to be terrible here.

**Fix:** Adding M=2048 to `M_I8` will tune a proper kernel for this shape and fix the selection.

---

## 5. Secondary Gap: Large-K Asymmetry (256×8192×28672)

| Shape | Library B TFlops | Comparison | Notes |
|-------|-----------------|------------|-------|
| 256×8192×28672 NN bf16 | 12.9 TF | — | K=28672 (large reduction) |
| 256×28672×8192 NN bf16 | 39.4 TF | 3× faster | Swapped N,K |

Same total FLOPs but swapping N and K drops performance 3×. This suggests the kernel selection for large-K shapes is suboptimal — possibly failing to apply adequate DepthU or split-K strategies.

This N×K pair (8192, 28672) IS in the tuning grid (`N_STD` and `K_STD` both contain these values), so the issue is likely that the fork combinations don't include a DepthU large enough for K=28672, or the selection logic is choosing wrong.

**Possible fix:** Increase `DepthU` options in fork parameters beyond `[16, 32]`, or add GlobalSplitU forks for large-K shapes.

---

## 6. Minor Regression: f32 2048×2048×2048 (-4.4%)

| Shape | Library A | Library B | Delta |
|-------|----------|----------|-------|
| 2048×2048×2048 NN f32 | 11808 GF | 11290 GF | -4.4% |

Minor regression in the VALU (non-WMMA) f32 path. Low priority — f32 GEMMs at this scale are uncommon in production, and the delta is within noise margin.

---

## 7. Performance Ladder Analysis

### bf16 NN M-sweep at fixed N=4096 K=4096

```
M        TFlops     % Peak    Status
------   ------     ------    ------
1        0.2        —         Memory-bound (expected)
8        2.2        2%        Memory-bound (expected)
16       4.3        4%        Memory-bound (expected)
32       8.9        8%        Transitioning
64       12.3       12%       Transitioning
96       25.9       24%       OK (non-PoT)
128      35.0       33%       OK
256      41.0       38%       OK
384      43.1       40%       OK (non-PoT)
512      44.3       41%       OK
1024     49.1       46%       OK
2048     12.7       12%       ██ CLIFF — 74% drop from M=1024
4096     54.2       51%       OK — best measured
8192     13.2       12%       ██ CLIFF — 76% drop from M=4096
```

The M=2048 and M=8192 cliffs are unambiguous. Every other M value shows smooth scaling.

### bf16 NT transpose sweep

```
M        TFlops     Status
------   ------     ------
128      44.2       Excellent (41% peak)
512      55.9       Excellent (52% peak)
1024     62.2       PEAK (58% of theoretical)
2048     13.0       ██ CLIFF — 79% drop from M=1024
```

### i8 NN M-sweep at N=4096 K=4096

```
M        TOPS       Status
------   ------     ------
256      49.3       OK
1024     52.8       OK
2048      5.8       ██ CLIFF + REGRESSION (-27.3% vs fallback)
4096     66.2       OK — best measured
```

---

## 8. Positive Findings

### Non-power-of-2 M: No gaps
- M=96: 25.9 TF — interpolates smoothly between M=64 (12.3) and M=128 (35.0)
- M=384: 43.1 TF — interpolates smoothly between M=256 (41.0) and M=512 (44.3)
- Tensile tile selection handles non-PoT M well for these sizes.

### TT transpose: Well optimized
- 256×4096×4096 TT: 40.4 TF (38% peak)
- 1024×4096×4096 TT: 48.7 TF (46% peak)
- No gaps in TT path.

### Real model dimensions: Good coverage
- 256×7168×7168 (DeepSeek): 44.4 TF (41%)
- 1024×3584×3584 (Qwen): 48.4 TF (45%)
- 256×14336×4096 (Llama-8B): 46.6 TF (44%)
- 256×28672×8192 (Llama-70B): 39.4 TF (37%)

### i8 performance: Excellent (except M=2048)
- 1024×4096×4096 NT i8: 71.8 TOPS — highest measured throughput
- 4096×4096×4096 NN i8: 66.2 TOPS
- 256×14336×4096 NN i8: 59.5 TOPS

### NT transpose: Dramatic improvement
- NT transposes saw the largest speedup over Library A (6000–6700%) due to the fallback library's extremely poor NT handling. Library B's tuned NT kernels are now competitive with NN and often faster.

---

## 9. Recommended Config Changes for `generate_configs.py`

### Priority 1: M_STD expansion (CRITICAL)

```python
# BEFORE (line ~111)
M_STD  = [16, 64, 256, 1024, 4096]

# AFTER
M_STD  = [16, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
```

**Rationale:** M=128 and M=512 already perform well (kernels from M=64/256 generalize), but including them ensures optimal kernels rather than relying on heuristic luck. M=2048 and M=8192 are the critical additions — they cannot be interpolated from neighbors.

Adding 4 new M values to 8 N values and 8 K values = **256 new problem sizes per dtype/transpose config file**.

### Priority 2: M_I8 expansion (HIGH)

```python
# BEFORE (line ~140)
M_I8   = [16, 64, 256, 1024, 4096]

# AFTER
M_I8   = [16, 64, 256, 512, 1024, 2048, 4096, 8192]
```

**Rationale:** The i8 M=2048 regression (-27.3%) is the only case where the new library is actively worse at a critical shape. This must be fixed.

### Priority 3: DepthU / GSU investigation for large-K

The 3× performance asymmetry between 256×8192×28672 (12.9 TF) and 256×28672×8192 (39.4 TF) suggests the fork parameters don't cover large-K reduction well.

```python
# Current fork params (WMMA path)
DepthU = [16, 32]

# Consider adding for large-K shapes:
DepthU = [16, 32, 64]
```

Or add GlobalSplitU configurations for the specific large-K cases.

### Priority 4: GSU configs sync

If separate GSU tuning configs exist at `~/rocm-libraries/shared/tensile/next-tuning/gfx1201-gsu/`, their `M_STD` should be updated in parallel with the same expansion.

---

## 10. Current M_STD Coverage Map

```
M value    In M_STD?    bf16 NN TFlops    Status
--------   ---------    --------------    ------
16         YES          4.3               OK (bandwidth-limited)
64         YES          12.3              OK (bandwidth-limited)
96         no           25.9              OK (heuristic works)
128        no           35.0              OK (heuristic works, not optimal)
256        YES          41.0              OK
384        no           43.1              OK (heuristic works)
512        no           44.3              OK (heuristic works, not optimal)
1024       YES          49.1              OK
2048       no           12.7              ██ BROKEN — heuristic fails catastrophically
4096       YES          54.2              OK — best measured
8192       no           13.2              ██ BROKEN — heuristic fails catastrophically
```

The heuristic works fine for M values between tuned points (128, 384, 512) but fails completely at M=2048 and M=8192 — exactly 2× above the nearest tuned M. The tile size jump from the M=1024 kernel to the M=4096 kernel creates a dead zone at M=2048 where neither is efficient.

---

## 11. Estimated Tuning Cost

| Change | New shapes added | Config files affected | Estimated wall time |
|--------|------------------|-----------------------|--------------------|
| M_STD += [128, 512, 2048, 8192] | +256 per file | 28 WMMA configs (7 types × 4 transposes) | ~2–4 hours |
| M_I8 += [512, 2048, 8192] | +72 per file | 8 i8 configs (WMMA+VALU × 4 transposes) | ~30–60 min |
| DepthU += [64] | +~50% more forks | Same 28 WMMA configs | ~2–3 hours additional |
| **Total** | | | **~5–8 hours** |

**Recommended approach:** Run M_STD and M_I8 expansion first (Priority 1 and 2), validate with this benchmark, then evaluate whether Priority 3 (DepthU) is needed based on results.

---

## 12. Validation Shapes — Retest After Retuning

After updating configs and rebuilding the library, re-run the benchmark and specifically validate:

```bash
# M=2048 cliff shapes (must improve from 12–19 TF to 45+ TF)
"2048 4096 4096 N N bf16_r"   # was 12.7 TF — target: 50+ TF
"2048 4096 4096 N T bf16_r"   # was 13.0 TF — target: 55+ TF
"2048 4096 4096 N N f16_r"    # was 18.9 TF — target: 50+ TF
"2048 4096 4096 N N i8_r"     # was 5.8 TOPS — target: 55+ TOPS (REGRESSION FIX)

# M=8192 cliff shape (must improve from 13.2 TF to 50+ TF)
"8192 4096 4096 N N bf16_r"   # was 13.2 TF — target: 50+ TF

# Large-K asymmetry (investigate if DepthU helps)
"256 8192 28672 N N bf16_r"   # was 12.9 TF — target: 35+ TF

# f32 regression (verify no worse)
"2048 2048 2048 N N f32_r"    # was -4.4% — should not regress further

# Already-good shapes (verify no regression from wider grid)
"1024 4096 4096 N T bf16_r"   # was 62.2 TF — must not regress
"1024 4096 4096 N T i8_r"     # was 71.8 TOPS — must not regress
"4096 4096 4096 N N bf16_r"   # was 54.2 TF — must not regress
```

---

## 13. Library A vs B Aggregate Statistics

```
Tests compared:      48
B wins (>1%):        46
A wins (>1%):        2
Ties (within ±1%):   0
Mean speedup (B/A):  +863.70%
Aggregate GF A:      322,776
Aggregate GF B:      1,524,129
Best case for B:     +6719%   (128×4096×4096 NT bf16)
Best case for A:     -27.3%   (2048×4096×4096 NN i8)
```

---

## Appendix: Raw Data File Locations

| File | Path |
|------|------|
| Comparison report | `tensile_ab_results/20260417_181144/comparison.txt` |
| Library A raw data | `tensile_ab_results/20260417_181144/lib_A.csv` |
| Library B raw data | `tensile_ab_results/20260417_181144/lib_B.csv` |
| Config generator | `~/rocm-libraries/shared/tensile/next-tuning/gfx1201-wide/generate_configs.py` |
| Benchmark script | `~/benchmarks/tensile_ab_bench.sh` |
