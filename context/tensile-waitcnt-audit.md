# Tensile Legacy s_waitcnt Audit - Complete Findings

## Critical GFX12 Gap
- `SeparateVscnt` is set for GFX10/11 ONLY, NOT GFX12
- GFX12 uses completely different wait model (s_wait_loadcnt, s_wait_storecnt, s_wait_kmcnt, s_wait_dscnt)
- Legacy Tensile has ZERO references to GFX12-style wait instructions
- The `WaitCnt` class in Code.py only knows about s_waitcnt + s_waitcnt_vscnt

## Files Containing Wait Instruction Code
1. Code.py - WaitCnt class (lines 279-320) - THE central abstraction
2. KernelWriterAssembly.py - ~80+ emission sites
3. KernelWriter.py - scheduling/placeholder logic
4. Common.py - SeparateVscnt cap, MaxVmcnt detection
5. AsmCaps.py - MaxLgkmcnt/MaxVmcnt defaults per ISA
6. Components/LocalRead.py - debug CheckValue1 waits
7. Components/ShiftVectorComponents.py - ds_bpermute waits
8. Components/MAC_F32.py - performance wait injection
9. KernelWriterSource.py - HIP source waits (stub)

## Wait Instruction Categories
### 1. WaitCnt class (Code.py:279) - structured emission
### 2. Direct string emission via inst() or addInst
### 3. Direct string concatenation (addText, kStr +=)
### 4. Placeholder mechanism (__placeholder__ + vmcnt replacement)

## SeparateVscnt Pattern (36 sites in KernelWriterAssembly.py)
Every s_waitcnt vmcnt(0) is followed by:
```python
if self.archCaps["SeparateVscnt"]:
    kStr += inst("s_waitcnt_vscnt", "null", "0", "writes")
```
This pattern is GFX10/11 only and is WRONG for GFX12.

## Resolution
Added `_convertWaitcntForGfx12()` post-processor in KernelWriter.py that regex-converts
all old-style s_waitcnt to split wait instructions after kernel source generation.
This avoids modifying the ~80+ emission sites and the WaitCnt class.
