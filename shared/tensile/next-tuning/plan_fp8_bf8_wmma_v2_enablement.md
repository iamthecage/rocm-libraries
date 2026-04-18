# Plan: Enable FP8/BF8 (E4M3/E5M2) Support for WMMA_V2 on RDNA4 (gfx1200/gfx1201) in Tensile

## Overview
This plan details the steps required to add support for FP8/BF8 (E4M3/E5M2) GEMM kernels using WMMA_V2 instructions on RDNA4 (gfx1200/gfx1201) in Tensile. It is intended for future implementation once current benchmarks are complete.

---

## 1. Audit and Patch DataType Handling
- **File:** `Tensile/SolutionStructs.py`
  - Locate all logic that restricts WMMA/WMMA_V2 to only half, bf16, and i8 types.
  - Remove or adjust the rejection logic so that F8/B8 (and mixed F8B8/B8F8) are accepted for WMMA_V2 (gfx12).
  - Ensure `numRegisters()` and related checks in DataType logic correctly handle F8/B8 as valid for WMMA_V2.
- **File:** `Tensile/DataType.py`
  - Confirm F8/B8 (and F8B8/B8F8) are defined with correct properties for WMMA_V2.
  - Add or adjust any missing attributes (e.g., `miInput`, `reg`, etc.) as needed for kernel emission.

## 2. Kernel Emission and Instruction Selection
- **File:** `Tensile/SolutionStructs.py` and related kernel generation modules
  - Update kernel emission logic to generate WMMA_V2 kernels for F8/B8 types.
  - Add or extend code paths that currently only emit for half/bf16/i8 to also emit for F8/B8.
  - Ensure correct mapping to hardware instructions (`__hip_fp8_e4m3`, `__hip_fp8_e5m2`).
  - Validate that the correct instruction encoding and register allocation is used for F8/B8.

## 3. Update Generator and ProblemType Definitions
- **File:** `next-tuning/gfx1201-wide/generate_configs.py` (and/or similar)
  - Add F8/B8 and mixed F8/B8 families for all layouts (NN, NT, TN, TT, etc.).
  - Add new YAMLs for `f8gemm_wmma_v2`, `b8gemm_wmma_v2`, and mixed input/output types (e.g., F8B8, B8F8).
  - Ensure correct `DataType`, `DestDataType`, and `ComputeDataType` fields are set for all new configs.
  - Add grouped-batch and GSU variants as appropriate.

## 4. Test and Validation
- **Unit/Integration Tests:**
  - Add or extend tests to cover F8/B8 WMMA_V2 kernels.
  - Validate kernel correctness (numerical, edge cases) and performance on real RDNA4 hardware.
  - Add tests for all supported layouts and mixed input/output combinations.

## 5. Documentation
- **User/Developer Docs:**
  - Update internal and user-facing documentation to reflect new support.
  - Document any new generator options, YAML fields, or tuning strategies for FP8/BF8.

---

## Extended Notes & Considerations
  - Confirm that the target RDNA4 hardware (gfx1200/gfx1201) supports WMMA_V2 for FP8/BF8 at the ISA level.
  - Validate instruction encoding and register usage for these types.
- **Reference Materials:**
  - Extensive ISA, validation, and RDNA4-specific reference information is available in `/home/iamthecage/rocm-libraries/rdna4` and its subdirectories. Consult these resources for hardware details, instruction listings, validation results, and architecture-specific notes as needed during implementation.
  - Official documentation for rocBLAS and Tensile is also available at `/home/iamthecage/rdna4_kb/rocblas/INDEX.md`.

---

**RDNA4 Reference Material and ISA Inspector Tooling**

- The `rdna4` base folder contains:
  - **ISA documentation and specs**:  
    - `rdna4-instruction-set-architecture.pdf` — Full PDF ISA reference.
    - `rdna4-optimization-reference.md`, `rdna4-swmmac-reference.md` — Machine-verified instruction catalogs, tuning, and sparse matrix details.
    - `matrix_calculator.py` (+ `.readme.md`) — Interactive tool for register layouts, throughput, and matrix instruction details.
    - `documentation/` — API docs and tutorials for the IsaDecoder and IsaExplorer C++ APIs, XML spec schema, and CLI usage.
    - `info.txt` — Notes on where to find the built XML spec and CLI/test binaries.
  - **ISA spec and decoder source**:  
    - `source/` and `include/` — C++ source for IsaDecoder, IsaExplorer, CLI, and examples.
    - `test/` — Unit tests for the decoder and explorer APIs.
    - `build/` — Build scripts for Linux/Windows.

- **ISA Inspector Tooling**:
  - The main CLI tool is built under `rdna4/build/linux` (after running `prebuild_linux.sh` and `make`).
    - The CLI binary is typically named `isa_spec_cli`.
    - The machine-readable XML spec (e.g., `amdgpu_isa_rdna4.xml`) is also generated here.
    - Example usage:
      - `./isa_spec_cli -x amdgpu_isa_rdna4.xml -d <hex>` — Decode a binary instruction.
      - `./isa_spec_cli -x amdgpu_isa_rdna4.xml -i <inst_name>` — Query instruction details.
    - Unit test binaries (e.g., `amdisa_test`) are also produced here.
  - See `info.txt` for exact output paths and invocation examples.

- **What’s in the documentation subfolder**:
  - API docs and tutorials for both IsaDecoder and IsaExplorer.
  - XML schema and spec documentation.
  - CLI usage and unit test documentation.

- **How to use**:
  - Build with the provided scripts, then use the CLI and XML spec for instruction validation, decoding, and architecture queries.
  - Use the Python matrix calculator for register mapping and instruction throughput validation.

---
  - This plan assumes Tensile 4.47.0 or later. If upstream adds native support, rebase/merge as needed.
- **Backward Compatibility:**
  - Ensure changes do not break existing half/bf16/i8 WMMA_V2 or MFMA paths.
- **Naming Conventions:**
  - Follow existing YAML and kernel naming conventions for new types (e.g., `f8gemm_wmma_v2_nt.yaml`).
- **Performance Tuning:**
  - Initial parameter sweeps may be borrowed from f16/bf16, but optimal tile sizes and fork parameters may differ for FP8/BF8.
- **Mixed Precision:**
  - Support for F8B8/B8F8 (mixed input) should be included if hardware and Tensile logic allow.
- **Testing:**
  - Prioritize correctness and stability before wide performance sweeps.
  - Compare against MFMA FP8/BF8 results (if available on other architectures) for sanity.

---

## Summary Table of Required Code Changes

| Area                        | Action                                                      |
|-----------------------------|-------------------------------------------------------------|
| SolutionStructs.py          | Remove/adjust FP8/BF8 rejection for WMMA_V2                 |
| DataType.py                 | Ensure F8/B8 properties are correct for WMMA_V2             |
| Kernel emission logic       | Add F8/B8 support for WMMA_V2 kernel generation             |
| Config generator            | Add F8/B8 and mixed families for all layouts                |
| Tests                       | Add/extend for F8/BF8 WMMA_V2 kernels                       |
| Documentation               | Update for new support                                      |

---

## Next Steps
- Wait until current benchmarks are complete.
- Review this plan and update for any upstream changes.
- Begin implementation with DataType and SolutionStructs.py patches, then generator and tests.
