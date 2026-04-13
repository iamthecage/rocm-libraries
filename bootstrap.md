# RDNA4 Legacy Tensile Port — Agent Bootstrap

> **Read this first.** This file gives a new agent complete context to continue the RDNA4/gfx1201 WMMA V2 port of legacy Tensile. All code changes are complete. GPU testing is the next step.

---

## Project Overview

We are porting **legacy Tensile** (the GEMM code generator used by rocBLAS) to support **AMD RDNA4 / gfx1201** (RX 9070 XT). The GPU uses **WMMA V2** matrix instructions (not MFMA), runs at **wave32**, and has a different ISA wait model than prior architectures.

**TensileLite** (the modern fork used by hipBLASLt) already supports gfx12 and serves as the reference implementation. We are backporting the critical fixes to the legacy codebase.

## Repository Layout

| Path | Description |
|------|-------------|
| `shared/tensile/Tensile/` | **Legacy Tensile** — the codebase being modified |
| `projects/hipblaslt/tensilelite/Tensile/` | **TensileLite** — reference implementation (read-only) |
| `rdna4/` | RDNA4 reference docs, ISA PDF, audit agent, optimization guide |
| `shared/tensile/next-tuning/` | Test configs, kernel outputs, tuning YAMLs |
| `linux.md` | Native Ubuntu 24.04 setup guide (ROCm, Python, deps) |

## All Completed Code Changes

Every change below is **committed** to the `develop` branch (commit `3682eedfcb`) except the wait instruction fix which is **uncommitted** in KernelWriter.py.

### 1. SRD Encoding Fix — `Code.py`
- **Problem**: GFX12 was using GFX11's SRD Word3 value (`0x31004000`), corrupting every buffer_load/buffer_store
- **Fix**: Added `SrdUpperValue12XX = 0x30020000` (format=32, resource_level=0, oob_select=3)
- **Severity**: P0 — would cause GPU hang/crash

### 2. WMMA V1/V2 Capability Flags — `AsmCaps.py`
- **Problem**: Only `HasWMMA` existed; no distinction between RDNA3 (V1) and RDNA4 (V2)
- **Fix**: Added `HasWMMA_V1` and `HasWMMA_V2` to all 28 ISA entries. GFX12: V1=False, V2=True. GFX11: V1=True, V2=False.

### 3. V1/V2 Detection — `Common.py`
- **Problem**: No runtime WMMA version detection
- **Fix**: Added `tryAssembler` probes for V1 vs V2 WMMA instruction variants

### 4. Tail-loop & enableReverseInner — `KernelWriterAssembly.py`
- **Problem**: Tail-loop masking used V1 code path for all WMMA; `enableReverseInner` crashed on V2
- **Fix**: Added `is_wmma_v1`/`is_wmma_v2` properties, separate V2 tail-loop routing, fixed enableReverseInner guard

### 5. Per-operand MIInputPerThread — `KernelWriter.py`, `SolutionStructs.py`, `LraTileAssignment.py`, `LocalRead.py`
- **Problem**: `MIInputPerThread` was a single scalar; V2 needs different values for A vs B operands
- **Fix**: Changed to `MIInputPerThreadA` / `MIInputPerThreadB` throughout

### 6. neg_lo Modifier — `MFMA.py`
- **Problem**: neg_lo used V1 format `[1,1,1]` (3 sources) for all WMMA; V2 needs `[1,1]` (2 sources)
- **Fix**: Conditional format based on V1 vs V2

### 7. Wait Instruction Conversion — `KernelWriter.py` (**UNCOMMITTED**)
- **Problem**: GFX12's `s_waitcnt` is aliased to `S_WAIT_IDLE` — the operand is IGNORED. So `s_waitcnt lgkmcnt(8)` becomes a **full pipeline flush** (waits ALL counters to 0). Correct but extremely slow.
- **Fix**: Added `_convertWaitcntForGfx12()` static method — a regex post-processor that converts:
  - `s_waitcnt lgkmcnt(N)` → `s_wait_dscnt N` + `s_wait_kmcnt N`
  - `s_waitcnt vmcnt(N)` → `s_wait_loadcnt N` (+ `s_wait_storecnt 0` when N==0)
  - Combined and `s_waitcnt 0` patterns also handled
- Hooked into `getKernelObjectAssemblyFile()` for `version[0] >= 12`
- **Verified**: 0 old `s_waitcnt` remain in generated kernel, 110 new split wait instructions present
- **ISA source**: Line 18601 of `rdna4/rdna4-instruction-set-architecture.pdf`

## What Has NOT Been Changed

- `WaitCnt` class in Code.py — still emits old-style `s_waitcnt`, converted by post-processor
- `SeparateVscnt` pattern (36 sites in KernelWriterAssembly.py) — GFX10/11 only, not triggered for GFX12
- No rocBLAS logic YAML files for gfx1201 yet
- No performance tuning (tile sizes, occupancy targets)

## Current Testing Status

### What Works
- Kernel **generates** successfully (assembly source produced)
- Kernel **assembles** without errors (valid ELF binary)
- Kernel **disassembles** correctly (all GFX12 opcodes verified)
- VALU (non-WMMA) kernel generated as initial test target

### What's Blocked
- **GPU execution hangs** — tested twice on WSL2, both times caused system-level hangs (Windows BSOD via TDR timeout)
- Root cause unclear: could be WSL2/WDDM thunk issue, or a remaining code bug
- **Moving to native Ubuntu 24.04** to eliminate WSL as a variable. On native Linux, a hung kernel can be `kill -9`'d without crashing the system.

## CRITICAL SAFETY RULE

**Do NOT run tensile-client, benchmark, or any GPU kernel execution command without explicit user approval.** The kernel may hang the GPU. On native Linux this is recoverable (kill the process), but always ask first.

## Key ISA Facts for GFX12

| Property | Value |
|----------|-------|
| Wave size | 32 |
| Matrix unit | WMMA 16×16×16 (V2) |
| VGPRs/SIMD | 1024, allocated in 16-reg blocks |
| Max VGPRs/wave | 256 |
| SGPRs | 106 usable (s0-s105) |
| LDS | 128 KB/WGP |
| FP8 format | OCP (`float8_e4m3fn`), NOT FNUZ |
| XCDs | 1 (single die) |
| WGPs | 32 on gfx1201 |
| `s_waitcnt` | = `S_WAIT_IDLE` (operand IGNORED, full flush) |
| Split waits | `s_wait_dscnt`, `s_wait_kmcnt`, `s_wait_loadcnt`, `s_wait_storecnt` |
| No MFMA | No LDS DMA, no async copy |

## WMMA V1 vs V2

| Property | V1 (RDNA3/gfx11) | V2 (RDNA4/gfx12) |
|----------|-------------------|-------------------|
| Input VGPRs (A, B) | 8 each | 4 each |
| Output VGPRs (D) | 8 | 8 (packed) |
| Accumulator packing | Unpacked (1 per VGPR) | Packed (2 f16 per VGPR) |
| neg_lo format | `[1,1,1]` (3 sources) | `[1,1]` (2 sources) |
| MIInputPerThread | 16 (same A/B) | Differs per operand |
| bpeCinternal | 4 (f32 in each VGPR) | 2 (f16 packed) |

## Reference Documents

| File | Contents |
|------|----------|
| `rdna4/rdna4-instruction-set-architecture.pdf` | Official 707-page ISA spec |
| `rdna4/rdna4-optimization-reference.md` | Hardware specs, VGPR tables, WMMA throughput, tuning guidelines |
| `rdna4/rdna4-swmmac-reference.md` | SWMMAC sparse matrix reference |
| `rdna4/rdna4-audit.agent.md` | Full 19-category audit agent with ISA validation tools |
| `rdna4/rdna4-bootstrap-prompt.md` | Self-contained audit prompt for any repo |
| `rdna4/matrix_calculator.py` | AMD matrix instruction calculator (execution details) |

## Environment Setup

See `linux.md` for complete native Ubuntu 24.04 setup instructions including:
- ROCm 7.2.1 apt installation
- Python 3.12 via uv
- Tensile pip dependencies
- PyTorch ROCm stack
- bashrc exports

Key env vars (native Linux — no WSL-specific vars needed):
```bash
export ROCM_PATH=/opt/rocm
export HIP_PATH=/opt/rocm
export PATH=$ROCM_PATH/bin:$PATH
export HSA_ENABLE_SDMA=0   # Only needed if SDMA queue issues occur
```

## Next Steps (Priority Order)

1. **Set up native Ubuntu 24.04** — follow `linux.md`
2. **Smoke test GPU** — `rocm-smi`, `rocminfo`, PyTorch CUDA check
3. **Commit the wait instruction fix** — the `_convertWaitcntForGfx12()` change in KernelWriter.py
4. **Test VALU kernel** — the non-WMMA kernel (simplest test, no matrix instructions)
   - Config: `shared/tensile/next-tuning/gfx1201_sgemm_valu.yaml`
   - If it hangs on native Linux, can safely `kill -9` the process
   - If VALU works → the infrastructure (SRD, waits, barriers) is correct
5. **Test WMMA kernel** — generate and run a WMMA-enabled kernel
   - This exercises V2 instruction emission, accumulator layout, tail-loop masking
6. **Performance tuning** — tile sizes, occupancy, split-K for gfx1201

## Files Modified (Quick Reference)

All in `shared/tensile/Tensile/`:

| File | Changes | Status |
|------|---------|--------|
| `Code.py` | SRD Word3 fix (0x30020000) | Committed |
| `AsmCaps.py` | HasWMMA_V1/V2 (28 ISA entries) | Committed |
| `Common.py` | V1/V2 detection via tryAssembler | Committed |
| `KernelWriterAssembly.py` | is_wmma_v1/v2, tail-loop, enableReverseInner | Committed |
| `KernelWriter.py` | Per-operand MIInputPerThread | Committed |
| `KernelWriter.py` | `_convertWaitcntForGfx12()` + `import re` | **Uncommitted** |
| `SolutionStructs.py` | MIInputPerThreadA/B | Committed |
| `Components/LraTileAssignment.py` | Per-operand MIInputPerThread | Committed |
| `Components/LocalRead.py` | Per-operand MIInputPerThread | Committed |
| `Components/MFMA.py` | neg_lo V2 format | Committed |

## Accumulated Audit Findings

See `context/` directory for detailed session notes:
- `context/rdna4-tensile-audit-findings.md` — SRD bug details, V1/V2 difference table, correctness issues
- `context/rdna4_gfx12_tensile_exploration.md` — ISA support details, AsmCaps, architecture capabilities
- `context/tensile-waitcnt-audit.md` — Wait model gap analysis, all files with wait code
