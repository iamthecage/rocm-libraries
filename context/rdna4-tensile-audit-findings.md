# RDNA4 Tensile Audit Findings

## CRITICAL BUG #1: SRD Encoding (CRASH CAUSE)
- Legacy Tensile `Code.py` uses `SrdUpperValue11XX` for `isa[0] >= 11` — GFX12 gets GFX11 encoding
- **Wrong value**: `0x31004000` (format=4, resource_level=1, oob_select=3)
- **Correct value**: `0x30020000` (format=32, resource_level=0, oob_select=3)
- Tensilelite has a separate `SrdUpperValue12XX` struct in `rocisa/rocisa/include/code.hpp`
- THIS CORRUPTS EVERY BUFFER_LOAD AND BUFFER_STORE → likely cause of BSOD/GPU hang
- Fix: Add `SrdUpperFields12XX` and `SrdUpperValue12XX` classes to `Code.py`, update `SrdUpperValue()` function

## Key Tensilelite vs Legacy Differences
| Feature | Tensilelite | Legacy |
|---------|-------------|--------|
| WMMA caps | HasWMMA_V1, HasWMMA_V2 | HasWMMA only |
| SRD format field | 32 (GFX12) | 4 (GFX11, used for GFX12) |
| SRD resource_level | 0 (GFX12) | 1 (GFX11) |
| MIInputPerThread | Per-operand A/B | Single value |
| outputVectorWidth | 8 (V2), 1 (V1) | Implicit via accs_per_wave |
| Tail loop (shiftK) | Separate V1/V2/MFMA paths | V1 path handles all WMMA |
| Local read packing | ECC-half for V2 | Same for all WMMA |
| neg_lo modifier | [1,1,1] V1, [1,1] V2 | [1,1,1] always (V1 style) |

## What Looks Correct
- WMMA instruction register widths: A=4 VGPRs, B=4 VGPRs, D/C=8 VGPRs ✓ (matches RDNA4 spec)
- accs_per_wave=8 for f32 compute ✓
- MIInputPerThread=8 for RDNA4 ✓ (16*16*1/32)
- null soffset for GFX12 buffer ops ✓
- s_barrier macro (s_barrier_signal -1 + s_barrier_wait -1) ✓
- s_cmpk removal (replaced with s_cmp) ✓

## Potential Correctness Issues (won't crash but may give wrong results)
- Accumulator output lane mapping (v_mov_b32 reordering) may not match V2 packed output layout
- Tail loop masking uses WMMA V1 code path (won't cause crash, may give wrong edge-case results)
- neg_lo format may be wrong for V2 (using V1's 3-element format)
