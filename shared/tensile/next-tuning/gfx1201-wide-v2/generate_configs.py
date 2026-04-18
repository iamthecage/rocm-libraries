#!/usr/bin/env python3
"""
gfx1201 (Navi48 / RX 9070) wide tuning YAML generator.

Creates comprehensive benchmark configs covering all dtype/transpose combinations
needed to give rocBLAS + TunableOp the best possible kernel selection for modern
LLM, MoE, and multimodal workloads.

Generated files:
  WMMA V2 path — gfx1201-native instructions:
    hgemm_wmma_{nn,nt,tn,tt}.yaml         — f16 HPA    (V_WMMA_F32_16X16X16_F16,   D=f16)
    hgemm_wmma_native_{nn,nt,tn,tt}.yaml  — f16 native (V_WMMA_F16_16X16X16_F16,   D=f16 acc)
    hss_wmma_{nn,nt,tn,tt}.yaml           — f16→f32    (V_WMMA_F32_16X16X16_F16,   D=f32)
    bf16gemm_wmma_{nn,nt,tn,tt}.yaml      — bf16 HPA   (V_WMMA_F32_16X16X16_BF16,  D=bf16)
    bss_wmma_{nn,nt,tn,tt}.yaml           — bf16→f32   (V_WMMA_F32_16X16X16_BF16,  D=f32)
    i8gemm_wmma_{nn,nt,tn,tt}.yaml        — int8 WMMA  (V_WMMA_I32_16X16X16_IU8)
    {hgemm,bf16gemm,i8gemm}_wmma_gsu_{nn,nt,tn,tt}.yaml — minimal GSU sweep

  VALU path (fallback + non-WMMA types):
    hgemm_valu_{nn,nt,tn,tt}.yaml         — f16 HPA VALU
    hgemm_native_{nn,nt,tn,tt}.yaml       — f16 native (non-HPA) VALU
    hss_valu_{nn,nt,tn,tt}.yaml           — f16→f32 VALU (HSS_BH)
    bf16gemm_valu_{nn,nt,tn,tt}.yaml      — bf16 HPA VALU
    bss_valu_{nn,nt,tn,tt}.yaml           — bf16→f32 VALU (BSS_BH)
    sgemm_{nn,nt,tn,tt}.yaml              — f32 VALU
    i8gemm_{nn,nt,tn,tt}.yaml             — int8/int32 VALU
    dgemm_{nn,nt,tn,tt}.yaml              — f64 VALU
    cgemm_{nn,nt,tn,tt,nc,cn,cc,tc,ct}.yaml — f32 complex VALU
    zgemm_{nn,nt,tn,tt,nc,cn,cc,tc,ct}.yaml — f64 complex VALU

  Grouped-batch (MoE per-expert):
    hgemm_wmma_gb_{nn,nt,tn,tt}.yaml
    bf16gemm_wmma_gb_{nn,nt,tn,tt}.yaml
    hgemm_valu_gb_{nn,nt,tn,tt}.yaml
    bf16gemm_valu_gb_{nn,nt,tn,tt}.yaml
    i8gemm_valu_gb_{nn,nt,tn,tt}.yaml
    hss_wmma_gb_{nn,nt,tn,tt}.yaml
    hss_valu_gb_{nn,nt,tn,tt}.yaml

WMMA instruction coverage (gfx1201 / Navi48):
  covered:   V_WMMA_F32_16X16X16_F16/BF16 (D=f16/bf16 and D=f32 outputs)
             V_WMMA_F16_16X16X16_F16 (native f16 accumulate)
             V_WMMA_I32_16X16X16_IU8 (int8)
  not covered: IU4 variants (Tensile has no INT4 DataType)
               FP8/BF8 WMMA (Tensile 4.47 only has MFMA-based FP8; gfx1201 HasMFMA=False)

Problem grid rationale:
  M — decode batches (16-128) → continuous batching (256-1024) → prefill (2048-8192)
       M=2048 and M=8192 added after A/B benchmarking revealed critical performance
       cliffs (12% of peak vs 51% expected) when these M values were absent.
  N — real per-layer output dims from Llama-2/3, Mixtral, Qwen2.5, DeepSeek-V3/R1, Phi-4
  K — real per-layer reduction dims from the same models

Usage:
  python3 generate_configs.py          # generates YAML files alongside this script
  python3 generate_configs.py --list   # just print file list, don't write
"""

import os
import sys
import textwrap

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Common YAML header blocks
# ---------------------------------------------------------------------------
MARKS = (
    "  marks: [skip-gfx900, skip-gfx906, skip-gfx908, skip-gfx90a, skip-gfx942, skip-gfx950, "
    "skip-gfx1010, skip-gfx1011, skip-gfx1012, skip-gfx1030, skip-gfx1031, skip-gfx1032, "
    "skip-gfx1033, skip-gfx1034, skip-gfx1035, skip-gfx1036, skip-gfx1100, skip-gfx1101, skip-gfx1102]"
)

GLOBAL_PARAMS = """\
GlobalParameters:
  NumElementsToValidate: 0
  KernelTime: True
  MaxWorkspaceSize: 67108864
  PrintSolutionRejectionReason: True
  MergeFiles: False
  DataInitTypeAlpha: 1
  DataInitTypeBeta: 0
  EnqueuesPerSync: 1
  SyncsPerBenchmark: 1
  SleepPercent: 0
"""

LIBRARY_LOGIC = """\
LibraryLogic:
    ScheduleName: "navi48"
    DeviceNames: ["Device 7550"]
    ArchitectureName: "gfx1201"
"""

# ---------------------------------------------------------------------------
# Transpose configurations
#   key: suffix used in filename
#   value: (TransposeA, TransposeB, description)
# ---------------------------------------------------------------------------
TRANSPOSES = {
    "nn": (False, False, "NN — neither transposed (Ailk_Bljk)"),
    "nt": (False, True,  "NT — B transposed  (Ailk_Bjlk)"),
    "tn": (True,  False, "TN — A transposed  (Alik_Bljk)"),
    "tt": (True,  True,  "TT — both transposed (Alik_Bjlk)"),
}

# Complex types also support conjugate transpose (C = transpose + conjugate).
# For real types C≡T, but for complex (cgemm/zgemm) they are distinct.
# key: suffix, value: (TransposeA, TransposeB, ConjA, ConjB, description)
TRANSPOSES_COMPLEX = {
    "nn": (False, False, False, False, "NN — neither transposed"),
    "nt": (False, True,  False, False, "NT — B transposed"),
    "tn": (True,  False, False, False, "TN — A transposed"),
    "tt": (True,  True,  False, False, "TT — both transposed"),
    "nc": (False, True,  False, True,  "NC — B conjugate-transposed"),
    "cn": (True,  False, True,  False, "CN — A conjugate-transposed"),
    "cc": (True,  True,  True,  True,  "CC — both conjugate-transposed"),
    "tc": (True,  True,  False, True,  "TC — A transposed, B conj-transposed"),
    "ct": (True,  True,  True,  False, "CT — A conj-transposed, B transposed"),
}

# ---------------------------------------------------------------------------
# Problem size grids
# ---------------------------------------------------------------------------
# Reduced grid for first pass — covers the most critical dims from each model.
# Full grid (10M×16N×15K=2400) can be run later on winning kernels.
# M: decode (16) → cont-batching (64,256) → prefill (1024,2048,4096,8192)
# NOTE: 2048 and 8192 added after A/B benchmarking showed critical performance
# cliffs at these M values — kernels tuned for M=1024/4096 don't generalize.
M_STD  = [16, 64, 256, 1024, 2048, 4096, 8192]

# N/K: most critical per-layer dims from production models
#   Llama-3.1-8B/70B : 4096, 8192, 14336, 28672
#   DeepSeek-V3/R1   : 7168, 18432
#   Mixtral-8×22B    : 16384
#   Qwen2.5          : 3584
N_STD  = [3584, 4096, 7168, 8192, 14336, 16384, 18432, 28672]

K_STD  = [1024, 4096, 7168, 8192, 14336, 16384, 18432, 28672]

# Grouped-batch (MoE per-expert) grid
M_GB   = [1, 8, 32, 128, 512, 1024]
N_GB   = [4096, 7168, 14336, 18432, 28672]
K_GB   = [4096, 7168, 14336, 18432, 28672]

# ── Pruned grids for types where LLM-scale sizes are wasteful ─────────────
# sgemm (f32): consumer workloads don't run pure FP32 matmuls at LLM scale
M_SGEMM = [16, 64, 256, 1024, 2048]
N_SGEMM = [64, 256, 1024, 2048, 4096]
K_SGEMM = [64, 256, 1024, 2048, 4096]

# f64 DGEMM: 1/32 of f16 rate on RDNA4 — keep small
M_F64  = [64, 256, 1024, 2048]
N_F64  = [64, 256, 1024, 2048]
K_F64  = [64, 256, 1024, 2048]

# int8 GEMM (VALU + WMMA): realistic INT8 inference sizes
# NOTE: 2048/8192 added — M=2048 i8 showed a -27% regression without a tuned kernel.
# NOTE: N=14336 added — Llama/Mistral MLP output dim, confirmed working at 59.5 TOPS
#       but only covered incidentally via bf16 configs, not i8-specific tuning.
M_I8   = [16, 64, 256, 1024, 2048, 4096, 8192]
N_I8   = [1024, 4096, 8192, 14336]
K_I8   = [1024, 4096, 8192, 14336]

# int8 grouped-batch (MoE per-expert)
M_I8_GB = [1, 8, 32, 128, 512]
N_I8_GB = [1024, 4096, 8192]
K_I8_GB = [1024, 4096, 8192]

# cgemm (f32_c): complex float — used in signal processing, physics sims
M_CGEMM = [64, 256, 1024, 2048]
N_CGEMM = [64, 256, 1024, 2048, 4096]
K_CGEMM = [64, 256, 1024, 2048, 4096]

# zgemm (f64_c): complex double — rare on consumer GPU, minimal coverage
M_ZGEMM = [64, 256, 1024]
N_ZGEMM = [64, 256, 1024, 2048]
K_ZGEMM = [64, 256, 1024, 2048]

# GSU (GlobalSplitU) configs: tall-skinny K-heavy problems where
# splitting the K-reduction across workgroups can help.
# GSU=1 wins most shapes, but GSU>1 may help very large K with small M×N.
M_GSU = [16, 64, 256]
N_GSU = [16, 64, 256]
K_GSU = [4096, 8192, 16384, 28672]


def problem_sizes_block(m_vals, n_vals, k_vals, batch=1, indent=10):
    """Return YAML lines for all M × N × K Exact entries."""
    pad = " " * indent
    lines = []
    for m in m_vals:
        for n in n_vals:
            for k in k_vals:
                lines.append(f"{pad}- Exact: [{m}, {n}, {batch}, {k}]")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Fork parameter blocks
# ---------------------------------------------------------------------------
def wmma_fork_params():
    """
    WMMA V2 fork parameters.
    Only MT64×64 ([16,16,16,1,1,4,4,1,1] / WG=[32,1,1]) is validated.
    Vary: SourceSwap, DepthU, GRVW, WaveSeparateGlobalRead.
    PGR=0 / PLR=0 — WMMA prefetch is still under investigation on gfx1201.
    """
    return """\
      ForkParameters:
        - MIArchVgpr: [True]
        - WavefrontSize: [32]
        - MatrixInstruction:
          - [16, 16, 16, 1,  1,  2,2,  1,1]   # MT32×32
          - [16, 16, 16, 1,  1,  4,4,  1,1]   # MT64×64
          - [16, 16, 16, 1,  1,  4,8,  1,1]   # MT64×128
          - [16, 16, 16, 1,  1,  8,4,  1,1]   # MT128×64
        - WorkGroup:
          - [32, 1, 1]
        - SourceSwap: [0, 1]
        - PrefetchGlobalRead: [0]
        - PrefetchLocalRead: [0]
        - DepthU: [16]
        - StaggerU: [0, 16]
        - TransposeLDS: [0]
        - VectorWidth: [1]
        - GlobalReadVectorWidth: [4]
        - WaveSeparateGlobalReadA: [0, 1]
        - WaveSeparateGlobalReadB: [0, 1]
        - OptNoLoadLoop: [0]
        - ScheduleLocalWrite: [0]
        - ScheduleGlobalRead: [0]
        - ScheduleIterAlg: [0]"""


def wmma_fork_params_i8():
    """
    WMMA V2 fork parameters for int8 GEMM (V_WMMA_I32_16X16X16_IU8).
    Same MatrixInstruction as f16 WMMA.  I8 packs 4 elements per VGPR so
    GlobalReadVectorWidth can reach 16 (= full DepthU=16 K-slice in one read).
    """
    return """\
      ForkParameters:
        - MIArchVgpr: [True]
        - WavefrontSize: [32]
        - MatrixInstruction:
          - [16, 16, 16, 1,  1,  4,4,  1,1]   # MT64×64, single wave
        - WorkGroup:
          - [32, 1, 1]
        - SourceSwap: [0, 1]
        - PrefetchGlobalRead: [0]
        - PrefetchLocalRead: [0]
        - DepthU: [16, 32, 64]
        - TransposeLDS: [0]
        - VectorWidth: [1]
        - GlobalReadVectorWidth: [4, 8, 16]
        - WaveSeparateGlobalReadA: [0, 1]
        - WaveSeparateGlobalReadB: [0, 1]
        - OptNoLoadLoop: [0]
        - ScheduleLocalWrite: [0]
        - ScheduleGlobalRead: [0]
        - ScheduleIterAlg: [0]"""


def valu_fork_params_hpa():
    """
    VALU fork parameters for HPA (f16/bf16 input, f32 accumulate).
    Multiple tile sizes + varied DU/VW/GRVW for comprehensive sweep.
    """
    return """\
      ForkParameters:
        - ThreadTile:
          - [2, 2]   # MT32×32
          - [4, 2]   # MT64×32
          - [2, 4]   # MT32×64
          - [4, 4]   # MT64×64
          - [8, 4]   # MT128×64
          - [4, 8]   # MT64×128
        - WorkGroup:
          - [16, 16, 1]
        - PrefetchGlobalRead: [1]
        - PrefetchLocalRead: [1]
        - DepthU: [8, 16, 32]
        - VectorWidth: [1, 2, 4]
        - GlobalReadVectorWidth: [2, 4, 8]
        - StaggerU: [0, 32]
        - ScheduleLocalWrite: [1]
        - ScheduleGlobalRead: [1]
        - ScheduleIterAlg: [1]
        - OptNoLoadLoop: [1]"""


def valu_fork_params_native():
    """
    VALU fork parameters for native f16 (no HPA, f16 accumulate).
    Tile shapes similar to HPA but VW options differ slightly.
    """
    return """\
      ForkParameters:
        - ThreadTile:
          - [2, 2]
          - [4, 2]
          - [2, 4]
          - [4, 4]
          - [8, 4]
          - [4, 8]
        - WorkGroup:
          - [16, 16, 1]
        - PrefetchGlobalRead: [1]
        - PrefetchLocalRead: [1]
        - DepthU: [8, 16, 32]
        - VectorWidth: [2, 4]
        - GlobalReadVectorWidth: [4, 8]
        - StaggerU: [0, 32]
        - ScheduleLocalWrite: [1]
        - ScheduleGlobalRead: [1]
        - ScheduleIterAlg: [1]
        - OptNoLoadLoop: [1]"""


def valu_fork_params_sgemm():
    """
    VALU fork parameters for f32 SGEMM.
    VW is smaller (f32 is 2× wider than f16) so pack fewer lanes.
    """
    return """\
      ForkParameters:
        - ThreadTile:
          - [2, 2]
          - [4, 2]
          - [2, 4]
          - [4, 4]
          - [8, 4]
          - [4, 8]
        - WorkGroup:
          - [16, 16, 1]
        - PrefetchGlobalRead: [1]
        - PrefetchLocalRead: [1]
        - DepthU: [8, 16, 32]
        - VectorWidth: [1, 2, 4]
        - GlobalReadVectorWidth: [1, 2, 4]
        - StaggerU: [0, 32]
        - ScheduleLocalWrite: [1]
        - ScheduleGlobalRead: [1]
        - ScheduleIterAlg: [1]
        - OptNoLoadLoop: [1]"""


def valu_fork_params_i8gemm():
    """
    VALU fork parameters for int8 GEMM (int8 input, int32 compute/output).
    Int8x4 packing allows higher GRVW.
    """
    return """\
      ForkParameters:
        - ThreadTile:
          - [2, 2]
          - [4, 2]
          - [2, 4]
          - [4, 4]
          - [8, 4]
          - [4, 8]
        - WorkGroup:
          - [16, 16, 1]
        - PrefetchGlobalRead: [1]
        - PrefetchLocalRead: [1]
        - DepthU: [8, 16, 32]
        - VectorWidth: [1, 2, 4]
        - GlobalReadVectorWidth: [4, 8, 16]
        - StaggerU: [0, 32]
        - ScheduleLocalWrite: [1]
        - ScheduleGlobalRead: [1]
        - ScheduleIterAlg: [1]
        - OptNoLoadLoop: [1]"""


def valu_fork_params_dgemm():
    """
    VALU fork parameters for f64 DGEMM.
    Smaller tiles and lower VW (f64 = 4× f32 = 8× f16 in terms of bytes).
    """
    return """\
      ForkParameters:
        - ThreadTile:
          - [2, 2]
          - [4, 2]
          - [2, 4]
          - [4, 4]
        - WorkGroup:
          - [16, 16, 1]
        - PrefetchGlobalRead: [1]
        - PrefetchLocalRead: [1]
        - DepthU: [4, 8, 16]
        - VectorWidth: [1, 2]
        - GlobalReadVectorWidth: [1, 2]
        - StaggerU: [0, 32]
        - ScheduleLocalWrite: [1]
        - ScheduleGlobalRead: [1]
        - ScheduleIterAlg: [1]
        - OptNoLoadLoop: [1]"""


def valu_fork_params_cgemm():
    """
    VALU fork parameters for f32_c CGEMM (complex float).
    Complex doubles the register pressure per element so keep tiles modest.
    """
    return """\
      ForkParameters:
        - ThreadTile:
          - [2, 2]
          - [4, 2]
          - [2, 4]
          - [4, 4]
        - WorkGroup:
          - [16, 16, 1]
        - PrefetchGlobalRead: [1]
        - PrefetchLocalRead: [1]
        - DepthU: [4, 8, 16]
        - VectorWidth: [1, 2]
        - GlobalReadVectorWidth: [1, 2]
        - StaggerU: [0, 32]
        - ScheduleLocalWrite: [1]
        - ScheduleGlobalRead: [1]
        - ScheduleIterAlg: [1]
        - OptNoLoadLoop: [1]"""


def valu_fork_params_zgemm():
    """
    VALU fork parameters for f64_c ZGEMM (complex double).
    Most register-hungry type — keep tiles and VW minimal.
    """
    return """\
      ForkParameters:
        - ThreadTile:
          - [2, 2]
          - [4, 2]
          - [2, 4]
        - WorkGroup:
          - [16, 16, 1]
        - PrefetchGlobalRead: [1]
        - PrefetchLocalRead: [1]
        - DepthU: [4, 8]
        - VectorWidth: [1]
        - GlobalReadVectorWidth: [1]
        - StaggerU: [0, 32]
        - ScheduleLocalWrite: [1]
        - ScheduleGlobalRead: [1]
        - ScheduleIterAlg: [1]
        - OptNoLoadLoop: [1]"""


def wmma_fork_params_gsu():
    """
    WMMA fork parameters with GlobalSplitU > 1 for tall-skinny-K problems.
    GSU splits the K-reduction across workgroups — rarely a win, but can
    help when M×N is small and K is very large.
    Minimal tile set since this is a niche case.
    """
    return """\
      ForkParameters:
        - MIArchVgpr: [True]
        - WavefrontSize: [32]
        - MatrixInstruction:
          - [16, 16, 16, 1,  1,  2,2,  1,1]   # MT32×32
          - [16, 16, 16, 1,  1,  4,4,  1,1]   # MT64×64
        - WorkGroup:
          - [32, 1, 1]
        - SourceSwap: [0]
        - PrefetchGlobalRead: [0]
        - PrefetchLocalRead: [0]
        - DepthU: [16]
        - GlobalSplitU: [1, 2, 4]
        - TransposeLDS: [0]
        - VectorWidth: [1]
        - GlobalReadVectorWidth: [4]
        - WaveSeparateGlobalReadA: [1]
        - WaveSeparateGlobalReadB: [1]
        - OptNoLoadLoop: [0]
        - ScheduleLocalWrite: [0]
        - ScheduleGlobalRead: [0]
        - ScheduleIterAlg: [0]"""


# ---------------------------------------------------------------------------
# Problem type blocks
# ---------------------------------------------------------------------------
def problem_type_wmma_hgemm(ta, tb, desc):
    return f"""\
    - # {desc} — f16 HPA WMMA V2 (v_wmma_f32_16x16x16_f16)
      OperationType: GEMM
      DataType: h
      ComputeDataType: s
      DestDataType: h
      HighPrecisionAccumulate: True
      TransposeA: {str(ta)}
      TransposeB: {str(tb)}
      UseBeta: True
      Batched: True"""


def problem_type_wmma_bf16gemm(ta, tb, desc):
    return f"""\
    - # {desc} — bf16 HPA WMMA V2 (v_wmma_f32_16x16x16_bf16)
      OperationType: GEMM
      DataType: B
      ComputeDataType: s
      DestDataType: B
      HighPrecisionAccumulate: True
      TransposeA: {str(ta)}
      TransposeB: {str(tb)}
      UseBeta: True
      Batched: True"""


def problem_type_valu_hgemm(ta, tb, desc):
    return f"""\
    - # {desc} — f16 HPA VALU (fallback for sizes/shapes where WMMA is not selected)
      OperationType: GEMM
      DataType: h
      ComputeDataType: s
      DestDataType: h
      HighPrecisionAccumulate: True
      TransposeA: {str(ta)}
      TransposeB: {str(tb)}
      UseBeta: True
      Batched: True"""


def problem_type_valu_hgemm_native(ta, tb, desc):
    return f"""\
    - # {desc} — f16 native (non-HPA, f16 accumulate)
      OperationType: GEMM
      DataType: h
      ComputeDataType: h
      DestDataType: h
      HighPrecisionAccumulate: False
      TransposeA: {str(ta)}
      TransposeB: {str(tb)}
      UseBeta: True
      Batched: True"""


def problem_type_valu_bf16gemm(ta, tb, desc):
    return f"""\
    - # {desc} — bf16 HPA VALU
      OperationType: GEMM
      DataType: B
      ComputeDataType: s
      DestDataType: B
      HighPrecisionAccumulate: True
      TransposeA: {str(ta)}
      TransposeB: {str(tb)}
      UseBeta: True
      Batched: True"""


def problem_type_sgemm(ta, tb, desc):
    return f"""\
    - # {desc} — f32 VALU
      OperationType: GEMM
      DataType: s
      TransposeA: {str(ta)}
      TransposeB: {str(tb)}
      UseBeta: True
      Batched: True"""


def problem_type_i8gemm(ta, tb, desc):
    # DataType: I8 (int8, char='I8', index 8 in Tensile DataType enum)
    # ComputeDataType/DestDataType: I (int32, char='I', index 6)
    return f"""\
    - # {desc} — int8 input / int32 compute+output (I8II_BH)
      OperationType: GEMM
      DataType: I8
      ComputeDataType: I
      DestDataType: I
      HighPrecisionAccumulate: True
      TransposeA: {str(ta)}
      TransposeB: {str(tb)}
      UseBeta: True
      Batched: True"""


def problem_type_dgemm(ta, tb, desc):
    return f"""\
    - # {desc} — f64 VALU
      OperationType: GEMM
      DataType: d
      TransposeA: {str(ta)}
      TransposeB: {str(tb)}
      UseBeta: True
      Batched: True"""


def problem_type_cgemm(ta, tb, ca, cb, desc):
    return f"""\
    - # {desc} — f32 complex VALU
      OperationType: GEMM
      DataType: c
      DestDataType: c
      ComputeDataType: c
      TransposeA: {str(ta)}
      TransposeB: {str(tb)}
      ComplexConjugateA: {str(ca)}
      ComplexConjugateB: {str(cb)}
      UseBeta: True
      Batched: True"""


def problem_type_zgemm(ta, tb, ca, cb, desc):
    return f"""\
    - # {desc} — f64 complex VALU
      OperationType: GEMM
      DataType: z
      DestDataType: z
      ComputeDataType: z
      TransposeA: {str(ta)}
      TransposeB: {str(tb)}
      ComplexConjugateA: {str(ca)}
      ComplexConjugateB: {str(cb)}
      UseBeta: True
      Batched: True"""


def problem_type_wmma_hgemm_gsu(ta, tb, desc):
    return f"""\
    - # {desc} — f16 HPA WMMA V2 with GSU sweep
      OperationType: GEMM
      DataType: h
      ComputeDataType: s
      DestDataType: h
      HighPrecisionAccumulate: True
      TransposeA: {str(ta)}
      TransposeB: {str(tb)}
      UseBeta: True
      Batched: True"""


def problem_type_wmma_bf16gemm_gsu(ta, tb, desc):
    return f"""\
    - # {desc} — bf16 HPA WMMA V2 with GSU sweep
      OperationType: GEMM
      DataType: B
      ComputeDataType: s
      DestDataType: B
      HighPrecisionAccumulate: True
      TransposeA: {str(ta)}
      TransposeB: {str(tb)}
      UseBeta: True
      Batched: True"""


def problem_type_wmma_i8gemm_gsu(ta, tb, desc):
    return f"""\
    - # {desc} — int8 WMMA with GSU sweep (I8II_BH)
      OperationType: GEMM
      DataType: I8
      ComputeDataType: I
      DestDataType: I
      HighPrecisionAccumulate: True
      TransposeA: {str(ta)}
      TransposeB: {str(tb)}
      UseBeta: True
      Batched: True"""


def problem_type_wmma_hgemm_native(ta, tb, desc):
    return f"""\
    - # {desc} — f16 native WMMA (V_WMMA_F16_16X16X16_F16 — f16 accumulate)
      OperationType: GEMM
      DataType: h
      ComputeDataType: h
      DestDataType: h
      HighPrecisionAccumulate: False
      TransposeA: {str(ta)}
      TransposeB: {str(tb)}
      UseBeta: True
      Batched: True"""


def problem_type_wmma_hss(ta, tb, desc):
    """V_WMMA_F32_16X16X16_F16 with D stored as F32 (HSS_BH).
    hipBLASLT serves this with MFMA kernels — WMMA V2 path is uncovered there.
    """
    return f"""\
    - # {desc} — f16→f32 WMMA (V_WMMA_F32_16X16X16_F16, D=f32 — HSS_BH)
      OperationType: GEMM
      DataType: h
      ComputeDataType: s
      DestDataType: s
      HighPrecisionAccumulate: True
      TransposeA: {str(ta)}
      TransposeB: {str(tb)}
      UseBeta: True
      Batched: True"""


def problem_type_wmma_bss(ta, tb, desc):
    """V_WMMA_F32_16X16X16_BF16 with D stored as F32 (BSS_BH).
    hipBLASLT serves this with MFMA kernels — WMMA V2 path is uncovered there.
    """
    return f"""\
    - # {desc} — bf16→f32 WMMA (V_WMMA_F32_16X16X16_BF16, D=f32 — BSS_BH)
      OperationType: GEMM
      DataType: B
      ComputeDataType: s
      DestDataType: s
      HighPrecisionAccumulate: True
      TransposeA: {str(ta)}
      TransposeB: {str(tb)}
      UseBeta: True
      Batched: True"""


def problem_type_wmma_i8gemm(ta, tb, desc):
    """V_WMMA_I32_16X16X16_IU8 — int8 in, int32 accumulate/out.
    hipBLASLT serves I8II_BH with MFMA kernels — this is the WMMA V2 path.
    """
    return f"""\
    - # {desc} — int8 WMMA (V_WMMA_I32_16X16X16_IU8 — I8II_BH)
      OperationType: GEMM
      DataType: I8
      ComputeDataType: I
      DestDataType: I
      HighPrecisionAccumulate: True
      TransposeA: {str(ta)}
      TransposeB: {str(tb)}
      UseBeta: True
      Batched: True"""


def problem_type_valu_hss(ta, tb, desc):
    return f"""\
    - # {desc} — f16→f32 VALU (f16 in, f32 out — HSS_BH, VALU fallback)
      OperationType: GEMM
      DataType: h
      ComputeDataType: s
      DestDataType: s
      HighPrecisionAccumulate: True
      TransposeA: {str(ta)}
      TransposeB: {str(tb)}
      UseBeta: True
      Batched: True"""


def problem_type_valu_bss(ta, tb, desc):
    return f"""\
    - # {desc} — bf16→f32 VALU (bf16 in, f32 out — BSS_BH, VALU fallback)
      OperationType: GEMM
      DataType: B
      ComputeDataType: s
      DestDataType: s
      HighPrecisionAccumulate: True
      TransposeA: {str(ta)}
      TransposeB: {str(tb)}
      UseBeta: True
      Batched: True"""


# ---------------------------------------------------------------------------
# Common parameters block
# ---------------------------------------------------------------------------
COMMON_PARAMS_WMMA = """\
    - InitialSolutionParameters:
      BenchmarkCommonParameters:
        - KernelLanguage: ["Assembly"]
        - EdgeType: ["ShiftPtr"]"""

COMMON_PARAMS_VALU = """\
    - InitialSolutionParameters:
      BenchmarkCommonParameters:
        - KernelLanguage: ["Assembly"]
        - EdgeType: ["ShiftPtr"]
        - WavefrontSize: [32]"""


# ---------------------------------------------------------------------------
# Full YAML assembly
# ---------------------------------------------------------------------------
def build_yaml(problem_type_block, common_params_block, fork_params_block,
               sizes_block, label):
    return f"""\
TestParameters:
{MARKS}

{GLOBAL_PARAMS}
BenchmarkProblems:
  -
{problem_type_block}

{common_params_block}
{fork_params_block}
      BenchmarkFinalParameters:
        - ProblemSizes:
{sizes_block}

{LIBRARY_LOGIC}"""


# ---------------------------------------------------------------------------
# Config family definitions
# ---------------------------------------------------------------------------
def configs():
    """
    Yield (filename, yaml_content) for every config in the wide run.
    """
    for tsuffix, (ta, tb, tdesc) in TRANSPOSES.items():

        # ── WMMA f16 HPA ────────────────────────────────────────────────────
        sizes = problem_sizes_block(M_STD, N_STD, K_STD)
        yield (
            f"hgemm_wmma_{tsuffix}.yaml",
            build_yaml(
                problem_type_wmma_hgemm(ta, tb, tdesc),
                COMMON_PARAMS_WMMA,
                wmma_fork_params(),
                sizes,
                f"hgemm_wmma_{tsuffix}",
            )
        )

        # ── WMMA bf16 HPA ───────────────────────────────────────────────────
        yield (
            f"bf16gemm_wmma_{tsuffix}.yaml",
            build_yaml(
                problem_type_wmma_bf16gemm(ta, tb, tdesc),
                COMMON_PARAMS_WMMA,
                wmma_fork_params(),
                sizes,
                f"bf16gemm_wmma_{tsuffix}",
            )
        )
        # ── WMMA f16 native (V_WMMA_F16_16X16X16_F16) ────────────────────
        yield (
            f"hgemm_wmma_native_{tsuffix}.yaml",
            build_yaml(
                problem_type_wmma_hgemm_native(ta, tb, tdesc),
                COMMON_PARAMS_WMMA,
                wmma_fork_params(),
                sizes,
                f"hgemm_wmma_native_{tsuffix}",
            )
        )

        # ── WMMA f16→f32 (V_WMMA_F32_16X16X16_F16, D=f32 — HSS_BH) ────────
        yield (
            f"hss_wmma_{tsuffix}.yaml",
            build_yaml(
                problem_type_wmma_hss(ta, tb, tdesc),
                COMMON_PARAMS_WMMA,
                wmma_fork_params(),
                sizes,
                f"hss_wmma_{tsuffix}",
            )
        )

        # ── WMMA bf16→f32 (V_WMMA_F32_16X16X16_BF16, D=f32 — BSS_BH) ──────
        yield (
            f"bss_wmma_{tsuffix}.yaml",
            build_yaml(
                problem_type_wmma_bss(ta, tb, tdesc),
                COMMON_PARAMS_WMMA,
                wmma_fork_params(),
                sizes,
                f"bss_wmma_{tsuffix}",
            )
        )

        # ── WMMA int8 (V_WMMA_I32_16X16X16_IU8) ────────────────────────
        sizes_i8 = problem_sizes_block(M_I8, N_I8, K_I8)
        yield (
            f"i8gemm_wmma_{tsuffix}.yaml",
            build_yaml(
                problem_type_wmma_i8gemm(ta, tb, tdesc),
                COMMON_PARAMS_WMMA,
                wmma_fork_params_i8(),
                sizes_i8,
                f"i8gemm_wmma_{tsuffix}",
            )
        )
        # ── VALU f16 HPA ────────────────────────────────────────────────────
        yield (
            f"hgemm_valu_{tsuffix}.yaml",
            build_yaml(
                problem_type_valu_hgemm(ta, tb, tdesc),
                COMMON_PARAMS_VALU,
                valu_fork_params_hpa(),
                sizes,
                f"hgemm_valu_{tsuffix}",
            )
        )

        # ── VALU f16 native (non-HPA) ───────────────────────────────────────
        yield (
            f"hgemm_native_{tsuffix}.yaml",
            build_yaml(
                problem_type_valu_hgemm_native(ta, tb, tdesc),
                COMMON_PARAMS_VALU,
                valu_fork_params_native(),
                sizes,
                f"hgemm_native_{tsuffix}",
            )
        )

        # ── VALU bf16 HPA ───────────────────────────────────────────────────
        yield (
            f"bf16gemm_valu_{tsuffix}.yaml",
            build_yaml(
                problem_type_valu_bf16gemm(ta, tb, tdesc),
                COMMON_PARAMS_VALU,
                valu_fork_params_hpa(),
                sizes,
                f"bf16gemm_valu_{tsuffix}",
            )
        )
        # ── VALU f16→f32 (HSS_BH) ─────────────────────────────────
        yield (
            f"hss_valu_{tsuffix}.yaml",
            build_yaml(
                problem_type_valu_hss(ta, tb, tdesc),
                COMMON_PARAMS_VALU,
                valu_fork_params_hpa(),
                sizes,
                f"hss_valu_{tsuffix}",
            )
        )

        # ── VALU bf16→f32 (BSS_BH) ────────────────────────────────
        yield (
            f"bss_valu_{tsuffix}.yaml",
            build_yaml(
                problem_type_valu_bss(ta, tb, tdesc),
                COMMON_PARAMS_VALU,
                valu_fork_params_hpa(),
                sizes,
                f"bss_valu_{tsuffix}",
            )
        )
        # ── SGEMM f32 ───────────────────────────────────────────────────────
        sizes_sgemm = problem_sizes_block(M_SGEMM, N_SGEMM, K_SGEMM)
        yield (
            f"sgemm_{tsuffix}.yaml",
            build_yaml(
                problem_type_sgemm(ta, tb, tdesc),
                COMMON_PARAMS_VALU,
                valu_fork_params_sgemm(),
                sizes_sgemm,
                f"sgemm_{tsuffix}",
            )
        )

        # ── int8 VALU ───────────────────────────────────────────────────────
        yield (
            f"i8gemm_{tsuffix}.yaml",
            build_yaml(
                problem_type_i8gemm(ta, tb, tdesc),
                COMMON_PARAMS_VALU,
                valu_fork_params_i8gemm(),
                sizes_i8,
                f"i8gemm_{tsuffix}",
            )
        )

        # ── f64 VALU ────────────────────────────────────────────────────────
        sizes_f64 = problem_sizes_block(M_F64, N_F64, K_F64)
        yield (
            f"dgemm_{tsuffix}.yaml",
            build_yaml(
                problem_type_dgemm(ta, tb, tdesc),
                COMMON_PARAMS_VALU,
                valu_fork_params_dgemm(),
                sizes_f64,
                f"dgemm_{tsuffix}",
            )
        )

        # ── Minimal GSU coverage (WMMA) ───────────────────────────────────
        sizes_gsu = problem_sizes_block(M_GSU, N_GSU, K_GSU)
        yield (
          f"hgemm_wmma_gsu_{tsuffix}.yaml",
          build_yaml(
            problem_type_wmma_hgemm_gsu(ta, tb, tdesc),
            COMMON_PARAMS_WMMA,
            wmma_fork_params_gsu(),
            sizes_gsu,
            f"hgemm_wmma_gsu_{tsuffix}",
          )
        )

        yield (
          f"bf16gemm_wmma_gsu_{tsuffix}.yaml",
          build_yaml(
            problem_type_wmma_bf16gemm_gsu(ta, tb, tdesc),
            COMMON_PARAMS_WMMA,
            wmma_fork_params_gsu(),
            sizes_gsu,
            f"bf16gemm_wmma_gsu_{tsuffix}",
          )
        )

        yield (
          f"i8gemm_wmma_gsu_{tsuffix}.yaml",
          build_yaml(
            problem_type_wmma_i8gemm_gsu(ta, tb, tdesc),
            COMMON_PARAMS_WMMA,
            wmma_fork_params_gsu(),
            sizes_gsu,
            f"i8gemm_wmma_gsu_{tsuffix}",
          )
        )

        # ── Grouped-batch (MoE) configs ─────────────────────────────────────
        sizes_gb = problem_sizes_block(M_GB, N_GB, K_GB)

        yield (
            f"hgemm_wmma_gb_{tsuffix}.yaml",
            build_yaml(
                problem_type_wmma_hgemm(ta, tb, tdesc + " [grouped-batch / MoE]"),
                COMMON_PARAMS_WMMA,
                wmma_fork_params(),
                sizes_gb,
                f"hgemm_wmma_gb_{tsuffix}",
            )
        )

        yield (
            f"bf16gemm_wmma_gb_{tsuffix}.yaml",
            build_yaml(
                problem_type_wmma_bf16gemm(ta, tb, tdesc + " [grouped-batch / MoE]"),
                COMMON_PARAMS_WMMA,
                wmma_fork_params(),
                sizes_gb,
                f"bf16gemm_wmma_gb_{tsuffix}",
            )
        )

        yield (
            f"hgemm_valu_gb_{tsuffix}.yaml",
            build_yaml(
                problem_type_valu_hgemm(ta, tb, tdesc + " [grouped-batch / MoE]"),
                COMMON_PARAMS_VALU,
                valu_fork_params_hpa(),
                sizes_gb,
                f"hgemm_valu_gb_{tsuffix}",
            )
        )

        yield (
            f"bf16gemm_valu_gb_{tsuffix}.yaml",
            build_yaml(
                problem_type_valu_bf16gemm(ta, tb, tdesc + " [grouped-batch / MoE]"),
                COMMON_PARAMS_VALU,
                valu_fork_params_hpa(),
                sizes_gb,
                f"bf16gemm_valu_gb_{tsuffix}",
            )
        )

        sizes_i8_gb = problem_sizes_block(M_I8_GB, N_I8_GB, K_I8_GB)
        yield (
            f"i8gemm_valu_gb_{tsuffix}.yaml",
            build_yaml(
                problem_type_i8gemm(ta, tb, tdesc + " [grouped-batch / MoE]"),
                COMMON_PARAMS_VALU,
                valu_fork_params_i8gemm(),
                sizes_i8_gb,
                f"i8gemm_valu_gb_{tsuffix}",
            )
        )

        yield (
            f"hss_wmma_gb_{tsuffix}.yaml",
            build_yaml(
                problem_type_wmma_hss(ta, tb, tdesc + " [grouped-batch / MoE]"),
                COMMON_PARAMS_WMMA,
                wmma_fork_params(),
                sizes_gb,
                f"hss_wmma_gb_{tsuffix}",
            )
        )

        yield (
            f"hss_valu_gb_{tsuffix}.yaml",
            build_yaml(
                problem_type_valu_hss(ta, tb, tdesc + " [grouped-batch / MoE]"),
                COMMON_PARAMS_VALU,
                valu_fork_params_hpa(),
                sizes_gb,
                f"hss_valu_gb_{tsuffix}",
            )
        )

    # Complex GEMM layouts include conjugate-transpose variants.
    for tsuffix, (ta, tb, ca, cb, tdesc) in TRANSPOSES_COMPLEX.items():
        sizes_cgemm = problem_sizes_block(M_CGEMM, N_CGEMM, K_CGEMM)
        yield (
            f"cgemm_{tsuffix}.yaml",
            build_yaml(
                problem_type_cgemm(ta, tb, ca, cb, tdesc),
                COMMON_PARAMS_VALU,
                valu_fork_params_cgemm(),
                sizes_cgemm,
                f"cgemm_{tsuffix}",
            )
        )

        sizes_zgemm = problem_sizes_block(M_ZGEMM, N_ZGEMM, K_ZGEMM)
        yield (
            f"zgemm_{tsuffix}.yaml",
            build_yaml(
                problem_type_zgemm(ta, tb, ca, cb, tdesc),
                COMMON_PARAMS_VALU,
                valu_fork_params_zgemm(),
                sizes_zgemm,
                f"zgemm_{tsuffix}",
            )
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    list_only = "--list" in sys.argv

    all_configs = list(configs())
    if list_only:
        for fname, _ in all_configs:
            print(os.path.join(SCRIPT_DIR, fname))
        return

    os.makedirs(SCRIPT_DIR, exist_ok=True)
    written = 0
    for fname, content in all_configs:
        path = os.path.join(SCRIPT_DIR, fname)
        with open(path, "w") as f:
            f.write(content)
        written += 1
        print(f"  wrote: {fname}")

    print(f"\n[done] {written} YAML configs written to {SCRIPT_DIR}/")
    total_sizes_std   = len(M_STD)   * len(N_STD)   * len(K_STD)
    total_sizes_gb    = len(M_GB)    * len(N_GB)    * len(K_GB)
    total_sizes_f64   = len(M_F64)   * len(N_F64)   * len(K_F64)
    total_sizes_sgemm = len(M_SGEMM) * len(N_SGEMM) * len(K_SGEMM)
    total_sizes_i8    = len(M_I8)    * len(N_I8)    * len(K_I8)
    total_sizes_i8_gb = len(M_I8_GB) * len(N_I8_GB) * len(K_I8_GB)
    total_sizes_cgemm = len(M_CGEMM) * len(N_CGEMM) * len(K_CGEMM)
    total_sizes_zgemm = len(M_ZGEMM) * len(N_ZGEMM) * len(K_ZGEMM)
    total_sizes_gsu   = len(M_GSU)   * len(N_GSU)   * len(K_GSU)
    print(f"  standard grid     : {len(M_STD)}M × {len(N_STD)}N × {len(K_STD)}K = {total_sizes_std:,} sizes/file")
    print(f"  grouped-batch grid: {len(M_GB)}M × {len(N_GB)}N × {len(K_GB)}K = {total_sizes_gb:,} sizes/file")
    print(f"  sgemm grid        : {len(M_SGEMM)}M × {len(N_SGEMM)}N × {len(K_SGEMM)}K = {total_sizes_sgemm:,} sizes/file")
    print(f"  f64 grid          : {len(M_F64)}M × {len(N_F64)}N × {len(K_F64)}K = {total_sizes_f64:,} sizes/file")
    print(f"  i8 grid           : {len(M_I8)}M × {len(N_I8)}N × {len(K_I8)}K = {total_sizes_i8:,} sizes/file")
    print(f"  i8 grouped-batch  : {len(M_I8_GB)}M × {len(N_I8_GB)}N × {len(K_I8_GB)}K = {total_sizes_i8_gb:,} sizes/file")
    print(f"  cgemm grid        : {len(M_CGEMM)}M × {len(N_CGEMM)}N × {len(K_CGEMM)}K = {total_sizes_cgemm:,} sizes/file")
    print(f"  zgemm grid        : {len(M_ZGEMM)}M × {len(N_ZGEMM)}N × {len(K_ZGEMM)}K = {total_sizes_zgemm:,} sizes/file")
    print(f"  gsu grid          : {len(M_GSU)}M × {len(N_GSU)}N × {len(K_GSU)}K = {total_sizes_gsu:,} sizes/file")


if __name__ == "__main__":
    main()
