# RDNA4/gfx12 Support Status in Tensile

## Key Findings

### 1. ISA Support
- **SupportedISA list** (Common.py):
  - (12,0,0) - gfx1200
  - (12,0,1) - gfx1201
  - Both are in globalParameters["SupportedISA"]

### 2. Architecture Mapping
- Common.py:
  - 'gfx1200':'gfx1200'
  - 'gfx1201':'gfx1201'

### 3. AsmCaps (Assembler Capabilities) for gfx12
Both (12,0,0) and (12,0,1) have identical caps:

**AsmCaps.py**:
- **HasMFMA**: False (NO MFMA support)
- **HasWMMA**: True (HAS WMMA - Wave Matrix Multiply Accumulate)
- **HasAddLshl**: True
- **HasExplicitCO**: True
- **HasExplicitNC**: True 
- **HasSMulHi**: True
- **HasLshlOr**: True
- **v_fma_f16**: True
- **v_fma_f32**: True
- **v_fma_f64**: True
- **v_fmac_f32**: True
- **v_pk_fma_f16**: True
- **v_dot2_f32_f16**: True
- **SupportedISA**: True
- **SupportedSource**: True
- MaxLgkmcnt: 15, MaxVmcnt: 63
- **HasGLCModifier**: False (different from some architectures)
- **HasNTModifier**: False (different from some architectures)

### 4. Architectural Capabilities
Common.py:
- **HasWave32**: True (isaVersion[0] in (10, 11, 12))
- **VgprBank**: True (isaVersion[0] in (10, 11, 12))
- **InstRename**: True (isaVersion[0]>=11)

### 5. Matrix Instruction Support
Common.py:
- **validWMMA**: [[16,16,16,1]]  (Fixed WMMA instruction format)
- **validMFMA**: Not applicable for gfx12 (HasMFMA = False)

### 6. GFX12 Specific Implementation Notes
- KernelWriterAssembly.py: "GFX12: s_barrier replaced by split barrier (s_barrier_signal + s_barrier_wait)"
- KernelWriterAssembly.py: "GFX12: s_cmpk_* removed, use s_cmp_* instead"
- KernelWriterAssembly.py: "GFX12 VBUFFER encoding requires 'null' instead of literal 0 for soffset"
- KernelWriterAssembly.py: "RDNA4 WMMA stores f16 accumulators packed (2 per VGPR), so keep bpeCinternal = 2" vs RDNA3 unpacked

### 7. WMMA Integration in SolutionStructs
SolutionStructs.py: validWMMA imported, WMMA selection logic present

## Summary
✅ **RDNA4/gfx12 is supported in Tensile with the following characteristics:**
- WMMA-only (Wave Matrix Multiply Accumulate) instruction support
- No MFMA (Matrix Fused Multiply-Add) support
- Wave32 and Wave64 operation modes
- Split barrier operations
- Specific encoding requirements for some instructions
- f16/bf16 accumulator packing (2 per VGPR) for WMMA
