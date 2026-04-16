#!/usr/bin/env python3
"""
rocblas_ab_bench.py — Real-world A/B benchmark + smoke test for gfx1201 rocBLAS kernels.

Simulates the GEMM workload of real transformer / MoE model layers using PyTorch,
which delegates to rocBLAS for all matrix multiplications on ROCm.  Also serves as
a numerical correctness smoke test: every GEMM result is verified against a known-good
reference (fp32 CPU or fp64 CPU), catching silent-corruption bugs in new kernels.

Benchmark workflow:
  1. Baseline:   python3 rocblas_ab_bench.py --tag baseline --output results/
  2. Install tuned kernels to /opt/rocm/lib/rocblas/library/
  3. Tuned:      python3 rocblas_ab_bench.py --tag tuned --output results/
  4. Compare:    python3 rocblas_ab_bench.py --compare results/baseline.json results/tuned.json

Smoke-test-only (fast, skips timing, just validates correctness):
  python3 rocblas_ab_bench.py --smoke-test
  python3 rocblas_ab_bench.py --smoke-test --models llama3-8b --scenarios prefill-512

Additional options:
  --models llama3-8b,deepseek-v3    Only benchmark specific models
  --scenarios prefill-2k,decode-128 Only benchmark specific scenarios
  --dtype fp16                      fp16 (default), bf16, fp32
  --layers 10                       Simulate N layers (default: 1)
  --warmup 2                        Warmup iterations (default: 2)
  --iters  5                        Timed iterations (default: 5)
  --min-sample-ms 100               Min ms per sample — auto-repeats tiny GEMMs (default: 100)
  --no-verify                       Skip numerical verification during benchmark
  --atol 0.05                       Absolute tolerance for verification (default: auto)
  --rtol 0.02                       Relative tolerance for verification (default: auto)
  --rocblas-log                     Set ROCBLAS_LAYER=2 to log rocBLAS calls

Requires: PyTorch with ROCm support
"""

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Model definitions — real architectures, real shapes
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Defines the GEMM shapes for one transformer layer of a real model."""
    name: str
    hidden: int
    num_heads: int
    head_dim: int
    kv_heads: int
    intermediate: int
    vocab_size: int
    num_experts: int = 1
    experts_active: int = 1
    shared_expert: bool = False
    shared_intermediate: int = 0
    description: str = ""

    @property
    def qkv_out(self) -> int:
        return (self.num_heads + 2 * self.kv_heads) * self.head_dim

    @property
    def attn_out(self) -> int:
        return self.hidden

    def gemm_shapes(self, M: int) -> list[tuple[str, int, int, int, str]]:
        """
        Return (name, M, N, K, transpose_label) for one transformer layer.
        All expressed as C[M,N] = A[M,K] @ B[K,N].
        """
        ops = []
        ops.append(("qkv_proj", M, self.qkv_out, self.hidden, "NT"))
        ops.append(("attn_out", M, self.hidden, self.num_heads * self.head_dim, "NT"))

        if self.num_experts > 1:
            expert_M = max(1, (M * self.experts_active) // self.num_experts)
            ops.append(("moe_gate", M, self.num_experts, self.hidden, "NT"))
            for i in range(self.experts_active):
                ops.append((f"expert{i}_gate_up", expert_M,
                            self.intermediate * 2, self.hidden, "NT"))
                ops.append((f"expert{i}_down", expert_M,
                            self.hidden, self.intermediate, "NT"))
            if self.shared_expert and self.shared_intermediate > 0:
                ops.append(("shared_gate_up", M,
                            self.shared_intermediate * 2, self.hidden, "NT"))
                ops.append(("shared_down", M,
                            self.hidden, self.shared_intermediate, "NT"))
        else:
            ops.append(("ffn_gate_up", M, self.intermediate * 2, self.hidden, "NT"))
            ops.append(("ffn_down", M, self.hidden, self.intermediate, "NT"))

        return ops


MODELS = {
    "llama3-8b": ModelConfig(
        name="llama3-8b",
        hidden=4096, num_heads=32, head_dim=128, kv_heads=8,
        intermediate=14336, vocab_size=128256,
        description="Llama-3 8B — standard dense, GQA",
    ),
    "llama3-70b": ModelConfig(
        name="llama3-70b",
        hidden=8192, num_heads=64, head_dim=128, kv_heads=8,
        intermediate=28672, vocab_size=128256,
        description="Llama-3 70B — large dense, GQA, big N dims",
    ),
    "deepseek-v2-lite": ModelConfig(
        name="deepseek-v2-lite",
        hidden=2048, num_heads=16, head_dim=128, kv_heads=2,
        intermediate=1408,
        vocab_size=102400,
        num_experts=64, experts_active=6,
        shared_expert=True, shared_intermediate=5632,
        description="DeepSeek-V2-Lite MoE — 64 experts, 6 active",
    ),
    "deepseek-v3": ModelConfig(
        name="deepseek-v3",
        hidden=7168, num_heads=128, head_dim=128, kv_heads=1,
        intermediate=2048,
        vocab_size=129280,
        num_experts=256, experts_active=8,
        shared_expert=True, shared_intermediate=18432,
        description="DeepSeek-V3 MoE — 256 experts, 8 active",
    ),
    "qwen2.5-7b": ModelConfig(
        name="qwen2.5-7b",
        hidden=3584, num_heads=28, head_dim=128, kv_heads=4,
        intermediate=18944, vocab_size=152064,
        description="Qwen-2.5 7B — dense, GQA, odd hidden dim",
    ),
    "mistral-7b": ModelConfig(
        name="mistral-7b",
        hidden=4096, num_heads=32, head_dim=128, kv_heads=8,
        intermediate=14336, vocab_size=32000,
        description="Mistral 7B — sliding window attn, same shapes as Llama",
    ),
    "mixtral-8x7b": ModelConfig(
        name="mixtral-8x7b",
        hidden=4096, num_heads=32, head_dim=128, kv_heads=8,
        intermediate=14336,
        vocab_size=32000,
        num_experts=8, experts_active=2,
        description="Mixtral 8x7B — 8 experts, 2 active per token",
    ),
}

SCENARIOS = {
    "prefill-512":   {"M": 512,  "desc": "Short prompt prefill (512 tokens)"},
    "prefill-2k":    {"M": 2048, "desc": "Standard prefill (2K context)"},
    "prefill-4k":    {"M": 4096, "desc": "Long prefill (4K context)"},
    "decode-1":      {"M": 1,    "desc": "Single-request decode (M=1)"},
    "decode-32":     {"M": 32,   "desc": "Batched decode, 32 requests"},
    "decode-128":    {"M": 128,  "desc": "Batched decode, 128 requests"},
    "decode-256":    {"M": 256,  "desc": "Large batch decode, 256 requests"},
}


# ---------------------------------------------------------------------------
# Verification helpers
# ---------------------------------------------------------------------------

def _default_tolerances(dtype_name: str) -> tuple[float, float]:
    """Return (atol, rtol) appropriate for the given dtype."""
    # These are calibrated for matmul accumulation error at typical sizes.
    # fp16 accumulates in fp16 on ROCm (unless using HPA), so error grows with K.
    return {
        "fp16": (0.05, 0.01),
        "bf16": (0.1, 0.02),      # bf16 has less mantissa precision
        "fp32": (1e-5, 1e-5),
    }[dtype_name]


def verify_gemm(M: int, N: int, K: int, dtype, dtype_name: str,
                atol: float, rtol: float, device: str = "cuda") -> dict:
    """
    Verify a single GEMM by comparing GPU result against fp32 CPU reference.

    Returns dict with:
      pass: bool, max_abs_err, max_rel_err, mean_abs_err,
      num_mismatched (elements exceeding tolerance), total_elements
    """
    import torch

    # Generate inputs on CPU in the target dtype, then cast to fp32 for reference
    A_cpu = torch.randn(M, K, dtype=dtype)
    B_cpu = torch.randn(K, N, dtype=dtype)

    # Reference: upcast to fp32 on CPU for ground-truth
    ref = torch.mm(A_cpu.float(), B_cpu.float())

    # GPU result in native dtype
    A_gpu = A_cpu.to(device)
    B_gpu = B_cpu.to(device)
    result_gpu = torch.mm(A_gpu, B_gpu)
    torch.cuda.synchronize()

    # Compare on CPU in fp32
    result_cpu = result_gpu.float().cpu()
    abs_err = (result_cpu - ref).abs()
    # For relative error, normalize by max(|ref|, 1) to avoid div-by-zero on small values
    rel_err = abs_err / (ref.abs().clamp(min=1.0))

    max_abs = abs_err.max().item()
    max_rel = rel_err.max().item()
    mean_abs = abs_err.mean().item()

    # Count mismatched elements (outside tolerance)
    mismatched = ((abs_err > atol) & (rel_err > rtol)).sum().item()
    total = M * N

    passed = mismatched == 0

    del A_cpu, B_cpu, A_gpu, B_gpu, result_gpu, result_cpu, ref, abs_err, rel_err

    return {
        "pass": passed,
        "max_abs_err": round(max_abs, 8),
        "max_rel_err": round(max_rel, 8),
        "mean_abs_err": round(mean_abs, 8),
        "mismatched": mismatched,
        "total_elements": total,
        "atol": atol,
        "rtol": rtol,
    }


# ---------------------------------------------------------------------------
# Benchmark engine
# ---------------------------------------------------------------------------

def get_torch_dtype(name: str):
    import torch
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[name]


def benchmark_gemm(M: int, N: int, K: int, dtype, warmup: int, iters: int,
                   device: str = "cuda", min_sample_ms: float = 100.0) -> dict:
    """Time a single GEMM: C[M,N] = A[M,K] @ B[K,N].

    Auto-calibrates repeat count so each timing sample takes at least
    min_sample_ms, preventing tiny GEMMs (e.g. M=1 decode) from finishing
    in microseconds where measurement noise dominates.
    """
    import torch

    A = torch.randn(M, K, dtype=dtype, device=device)
    B = torch.randn(K, N, dtype=dtype, device=device)

    # Warmup
    for _ in range(warmup):
        torch.mm(A, B)
    torch.cuda.synchronize()

    # --- Auto-calibrate repeat count ---
    # Probe with a single call to estimate per-GEMM time
    probe_start = torch.cuda.Event(enable_timing=True)
    probe_end = torch.cuda.Event(enable_timing=True)
    probe_start.record()
    torch.mm(A, B)
    probe_end.record()
    torch.cuda.synchronize()
    probe_ms = probe_start.elapsed_time(probe_end)

    # Calculate repeats needed so each sample >= min_sample_ms
    if probe_ms > 0:
        repeats = max(1, int(min_sample_ms / probe_ms))
    else:
        repeats = max(1, int(min_sample_ms / 0.001))  # fallback: assume 1μs

    # --- Timed iterations ---
    times_ms = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            torch.mm(A, B)
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end) / repeats)

    times_ms.sort()
    median_ms = times_ms[len(times_ms) // 2]
    mean_ms = sum(times_ms) / len(times_ms)
    min_ms = times_ms[0]
    flops = 2.0 * M * N * K
    tflops = flops / (median_ms * 1e-3) / 1e12 if median_ms > 0 else 0.0

    del A, B
    return {
        "M": M, "N": N, "K": K,
        "repeats": repeats,
        "median_ms": round(median_ms, 4),
        "mean_ms": round(mean_ms, 4),
        "min_ms": round(min_ms, 4),
        "tflops": round(tflops, 4),
        "times_ms": [round(t, 4) for t in times_ms],
    }


def benchmark_layer(model: ModelConfig, M: int, dtype, dtype_name: str,
                    warmup: int, iters: int,
                    num_layers: int = 1, device: str = "cuda",
                    do_verify: bool = True,
                    atol: float = 0.05, rtol: float = 0.01,
                    min_sample_ms: float = 100.0) -> dict:
    """
    Benchmark + optionally verify all GEMMs in a transformer layer.

    Auto-calibrates repeat count for both individual GEMMs and the
    full-layer composite so each timing sample is substantial.
    """
    import torch

    gemm_shapes = model.gemm_shapes(M)
    individual_results = []

    # --- Individual GEMM benchmarks + verification ---
    for op_name, m, n, k, trans in gemm_shapes:
        result = benchmark_gemm(m, n, k, dtype, warmup, iters, device,
                                min_sample_ms=min_sample_ms)
        result["op"] = op_name
        result["transpose"] = trans

        if do_verify:
            vr = verify_gemm(m, n, k, dtype, dtype_name, atol, rtol, device)
            result["verify"] = vr

        individual_results.append(result)

    # --- Full layer timing ---
    gemm_pairs = []
    for _, m, n, k, _ in gemm_shapes:
        A = torch.randn(m, k, dtype=dtype, device=device)
        B = torch.randn(k, n, dtype=dtype, device=device)
        gemm_pairs.append((A, B))

    for _ in range(warmup):
        for A, B in gemm_pairs:
            torch.mm(A, B)
    torch.cuda.synchronize()

    # --- Auto-calibrate layer repeats ---
    # Probe one pass of all layers to estimate timing
    probe_start = torch.cuda.Event(enable_timing=True)
    probe_end = torch.cuda.Event(enable_timing=True)
    probe_start.record()
    for _ in range(num_layers):
        for A, B in gemm_pairs:
            torch.mm(A, B)
    probe_end.record()
    torch.cuda.synchronize()
    probe_ms = probe_start.elapsed_time(probe_end)
    layer_repeats = max(1, int(min_sample_ms / probe_ms)) if probe_ms > 0 else 1

    layer_times_ms = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(layer_repeats):
            for _ in range(num_layers):
                for A, B in gemm_pairs:
                    torch.mm(A, B)
        end.record()
        torch.cuda.synchronize()
        layer_times_ms.append(start.elapsed_time(end) / layer_repeats)

    layer_times_ms.sort()
    layer_median_ms = layer_times_ms[len(layer_times_ms) // 2]
    total_flops = sum(2.0 * m * n * k for _, m, n, k, _ in gemm_shapes) * num_layers
    layer_tflops = total_flops / (layer_median_ms * 1e-3) / 1e12 if layer_median_ms > 0 else 0.0

    del gemm_pairs

    return {
        "model": model.name,
        "M": M,
        "num_layers": num_layers,
        "layer_repeats": layer_repeats,
        "individual_gemms": individual_results,
        "layer_median_ms": round(layer_median_ms, 4),
        "layer_mean_ms": round(sum(layer_times_ms) / len(layer_times_ms), 4),
        "layer_min_ms": round(layer_times_ms[0], 4),
        "layer_tflops": round(layer_tflops, 4),
        "layer_times_ms": [round(t, 4) for t in layer_times_ms],
    }


def benchmark_lm_head(model: ModelConfig, M: int, dtype, dtype_name: str,
                      warmup: int, iters: int,
                      device: str = "cuda",
                      do_verify: bool = True,
                      atol: float = 0.05, rtol: float = 0.01,
                      min_sample_ms: float = 100.0) -> dict:
    """Benchmark the final lm_head projection."""
    result = benchmark_gemm(M, model.vocab_size, model.hidden, dtype, warmup, iters,
                             device, min_sample_ms=min_sample_ms)
    result["op"] = "lm_head"
    result["transpose"] = "NT"
    if do_verify:
        result["verify"] = verify_gemm(
            M, model.vocab_size, model.hidden, dtype, dtype_name, atol, rtol, device)
    return result


# ---------------------------------------------------------------------------
# Smoke test mode — correctness only, minimal timing
# ---------------------------------------------------------------------------

def run_smoke_test(model_names: list[str], scenario_names: list[str],
                   dtype, dtype_name: str, atol: float, rtol: float,
                   device: str = "cuda") -> tuple[int, int, list[dict]]:
    """
    Run correctness-only verification for every GEMM shape.
    Returns (pass_count, fail_count, failure_details).
    """
    passes = 0
    fails = 0
    failures = []

    for model_name in model_names:
        model = MODELS[model_name]
        print(f"\n  {model.name}: {model.description}")

        for scenario_name in scenario_names:
            M = SCENARIOS[scenario_name]["M"]
            gemms = model.gemm_shapes(M)

            for op_name, m, n, k, trans in gemms:
                vr = verify_gemm(m, n, k, dtype, dtype_name, atol, rtol, device)
                status = "PASS" if vr["pass"] else "FAIL"
                if vr["pass"]:
                    passes += 1
                    sym = "✓"
                else:
                    fails += 1
                    sym = "✗"
                    failures.append({
                        "model": model_name, "scenario": scenario_name,
                        "op": op_name, "shape": f"{m}x{n}x{k}", **vr,
                    })

                # Compact output: only print details on failure or first few
                err_str = f"max_abs={vr['max_abs_err']:.6f} max_rel={vr['max_rel_err']:.6f}"
                if not vr["pass"]:
                    err_str += f" MISMATCHED={vr['mismatched']}/{vr['total_elements']}"
                print(f"    {sym} {op_name:<25} {m:>5}×{n:>5}×{k:>5}  {status}  {err_str}")

            # Also verify lm_head for prefill scenarios
            if "prefill" in scenario_name:
                m2, n2, k2 = M, model.vocab_size, model.hidden
                vr2 = verify_gemm(m2, n2, k2, dtype, dtype_name, atol, rtol, device)
                if vr2["pass"]:
                    passes += 1
                    print(f"    ✓ {'lm_head':<25} {m2:>5}×{n2:>5}×{k2:>5}  PASS")
                else:
                    fails += 1
                    failures.append({
                        "model": model_name, "scenario": scenario_name,
                        "op": "lm_head", "shape": f"{m2}x{n2}x{k2}", **vr2,
                    })
                    print(f"    ✗ {'lm_head':<25} {m2:>5}×{n2:>5}×{k2:>5}  FAIL  "
                          f"mismatched={vr2['mismatched']}/{vr2['total_elements']}")

    return passes, fails, failures


# ---------------------------------------------------------------------------
# Edge-case stress tests — shapes that historically break kernels
# ---------------------------------------------------------------------------

STRESS_SHAPES = [
    # (label, M, N, K) — shapes that are tricky for tiled kernels
    ("tiny_square",       1,    1,    1),
    ("tiny_rect",         1,   16,    1),
    ("single_row",        1, 4096, 4096),
    ("single_col",     4096,    1, 4096),
    ("not_aligned_16",   17,   33,   65),
    ("not_aligned_32",   31,   63,  127),
    ("not_aligned_64",   65,  129,  257),
    ("prime_dims",       67,   71,   73),
    ("large_prime",     127,  131,  137),
    ("tall_skinny",    8192,   16,  128),
    ("short_wide",       16, 8192,  128),
    ("deep_k",          128,  128, 16384),
    ("power_of_2",     1024, 1024, 1024),
    ("llm_decode_1",      1, 14336, 4096),  # Llama FFN gate at decode
    ("llm_decode_32",    32, 14336, 4096),  # Llama FFN gate batched decode
    ("llm_prefill",    2048, 28672, 8192),  # Llama-70B FFN
    ("moe_expert",       12,  2816, 7168),  # DeepSeek expert-sized
    ("odd_hidden",      512, 18944, 3584),  # Qwen FFN (odd dims)
    ("very_wide",       256, 128256, 4096), # lm_head with large vocab
]


def run_stress_test(dtype, dtype_name: str, atol: float, rtol: float,
                    device: str = "cuda") -> tuple[int, int, list[dict]]:
    """
    Run edge-case shapes that historically break tiled GEMM kernels.
    Tests alignment, boundary conditions, and extreme aspect ratios.
    """
    passes = 0
    fails = 0
    failures = []

    for label, m, n, k in STRESS_SHAPES:
        vr = verify_gemm(m, n, k, dtype, dtype_name, atol, rtol, device)
        if vr["pass"]:
            passes += 1
            sym = "✓"
        else:
            fails += 1
            sym = "✗"
            failures.append({"label": label, "shape": f"{m}x{n}x{k}", **vr})

        err_str = f"max_abs={vr['max_abs_err']:.6f} max_rel={vr['max_rel_err']:.6f}"
        if not vr["pass"]:
            err_str += f" MISMATCHED={vr['mismatched']}/{vr['total_elements']}"
        print(f"    {sym} {label:<20} {m:>5}×{n:>5}×{k:>5}  {'PASS' if vr['pass'] else 'FAIL'}  {err_str}")

    return passes, fails, failures


# ---------------------------------------------------------------------------
# Compare mode
# ---------------------------------------------------------------------------

def compare_results(baseline_path: str, tuned_path: str):
    """Load two result JSONs and print a comparison table."""
    with open(baseline_path) as f:
        baseline = json.load(f)
    with open(tuned_path) as f:
        tuned = json.load(f)

    print("=" * 100)
    print(f"  rocBLAS A/B Comparison: {baseline['tag']} vs {tuned['tag']}")
    print(f"  Baseline: {baseline_path}")
    print(f"  Tuned:    {tuned_path}")
    print("=" * 100)

    def make_key(r):
        return (r["model"], r["scenario"])

    tuned_map = {make_key(r): r for r in tuned["results"]}

    for br in baseline["results"]:
        key = make_key(br)
        tr = tuned_map.get(key)
        if tr is None:
            continue

        print(f"\n{'─' * 100}")
        print(f"  Model: {br['model']}  |  Scenario: {br['scenario']} (M={br['M']})")
        print(f"  Layer: {br['layer_median_ms']:.3f}ms → {tr['layer_median_ms']:.3f}ms  "
              f"({_speedup_str(br['layer_median_ms'], tr['layer_median_ms'])})")
        print(f"{'─' * 100}")
        hdr = (f"  {'Operation':<25} {'Shape (M×N×K)':<22} "
               f"{'Baseline ms':>12} {'Tuned ms':>12} {'Speedup':>10} {'ΔTFLOPS':>10}")
        print(hdr)
        print(f"  {'─'*25} {'─'*22} {'─'*12} {'─'*12} {'─'*10} {'─'*10}")

        for bg, tg in zip(br["individual_gemms"], tr["individual_gemms"]):
            shape = f"{bg['M']}×{bg['N']}×{bg['K']}"
            delta_tflops = tg["tflops"] - bg["tflops"]
            # Verification status if available
            v_base = bg.get("verify", {})
            v_tune = tg.get("verify", {})
            v_str = ""
            if v_base and v_tune:
                v_str = "  " + ("✓✓" if v_base.get("pass") and v_tune.get("pass")
                                else "✓✗" if v_base.get("pass")
                                else "✗✓" if v_tune.get("pass")
                                else "✗✗")
            print(f"  {bg['op']:<25} {shape:<22} {bg['median_ms']:>12.4f} "
                  f"{tg['median_ms']:>12.4f} "
                  f"{_speedup_str(bg['median_ms'], tg['median_ms']):>10} "
                  f"{delta_tflops:>+10.2f}{v_str}")

    # LM head
    if "lm_head" in baseline and "lm_head" in tuned:
        print(f"\n{'─' * 100}")
        print(f"  LM Head Projections")
        print(f"{'─' * 100}")
        for bk, blh in baseline["lm_head"].items():
            tlh = tuned.get("lm_head", {}).get(bk)
            if tlh:
                print(f"  {bk}: {blh['median_ms']:.4f}ms → {tlh['median_ms']:.4f}ms  "
                      f"({_speedup_str(blh['median_ms'], tlh['median_ms'])})")

    # Geometric mean
    all_speedups = []
    for br in baseline["results"]:
        tr = tuned_map.get(make_key(br))
        if tr is None:
            continue
        for bg, tg in zip(br["individual_gemms"], tr["individual_gemms"]):
            if tg["median_ms"] > 0:
                all_speedups.append(bg["median_ms"] / tg["median_ms"])

    print(f"\n{'=' * 100}")
    if all_speedups:
        print(f"  Geometric mean speedup across {len(all_speedups)} GEMMs: "
              f"{_geomean(all_speedups):.3f}x")
    print("=" * 100)


def _speedup_str(old_ms: float, new_ms: float) -> str:
    if new_ms <= 0:
        return "∞"
    ratio = old_ms / new_ms
    return f"{ratio:.2f}x ↑" if ratio >= 1.0 else f"{ratio:.2f}x ↓"


def _geomean(values: list[float]) -> float:
    if not values:
        return 1.0
    log_sum = sum(math.log(max(v, 1e-12)) for v in values)
    return math.exp(log_sum / len(values))


# ---------------------------------------------------------------------------
# TunableOp configuration
# ---------------------------------------------------------------------------

def setup_tunableop(output_dir: str, tag: str) -> str:
    """
    Configure PyTorch TunableOp to record which BLAS kernels are dispatched.

    MUST be called BEFORE `import torch` — the env vars are read at import time.

    Returns the path to the tunableop results CSV file.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    tunableop_file = out / f"tunableop_{tag}_{timestamp}.csv"

    # Enable TunableOp so PyTorch records kernel selections
    os.environ["PYTORCH_TUNABLEOP_ENABLED"] = "1"
    # Write results to our timestamped file
    os.environ["PYTORCH_TUNABLEOP_FILENAME"] = str(tunableop_file)
    # Enable tuning so it actually tries kernels and records the winner
    os.environ["PYTORCH_TUNABLEOP_TUNING"] = "1"
    # Record full results (not just best)
    os.environ["PYTORCH_TUNABLEOP_RECORD_FULL"] = "1"

    return str(tunableop_file)


# ---------------------------------------------------------------------------
# System info
# ---------------------------------------------------------------------------

def collect_system_info(tunableop_file: str = "") -> dict:
    import torch
    info = {
        "pytorch_version": torch.__version__,
        "rocm_version": getattr(torch.version, 'hip', 'unknown'),
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_count": torch.cuda.device_count(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tunableop_file": tunableop_file,
    }
    rocblas_lib = Path("/opt/rocm/lib/rocblas/library")
    if rocblas_lib.exists():
        tuned_count = len(list(rocblas_lib.glob("TensileLibrary_Type_*_gfx1201.co")))
        fallback_count = len(list(rocblas_lib.glob("*_fallback_gfx1201.hsaco")))
        info["rocblas_gfx1201_tuned_co"] = tuned_count
        info["rocblas_gfx1201_fallback_hsaco"] = fallback_count
        info["rocblas_library_state"] = "tuned" if tuned_count > 0 else "fallback-only"
    return info


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="rocBLAS A/B benchmark + smoke test for gfx1201 tuned kernels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--tag", type=str, default="run",
                        help="Label for this run (e.g. 'baseline' or 'tuned')")
    parser.add_argument("--output", type=str, default=".",
                        help="Output directory for result JSON")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model names (default: all)")
    parser.add_argument("--scenarios", type=str, default=None,
                        help="Comma-separated scenario names (default: all)")
    parser.add_argument("--dtype", type=str, default="fp16",
                        choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--layers", type=int, default=1,
                        help="Number of layers for full-layer timing")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--min-sample-ms", type=float, default=100.0,
                        help="Minimum ms per timing sample — auto-repeats small GEMMs (default: 100)")
    parser.add_argument("--no-lm-head", action="store_true",
                        help="Skip lm_head benchmark")
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip numerical verification during benchmark")
    parser.add_argument("--atol", type=float, default=None,
                        help="Absolute tolerance (default: auto per dtype)")
    parser.add_argument("--rtol", type=float, default=None,
                        help="Relative tolerance (default: auto per dtype)")
    parser.add_argument("--rocblas-log", action="store_true",
                        help="Set ROCBLAS_LAYER=2 to log rocBLAS calls")
    parser.add_argument("--compare", nargs=2, metavar=("BASELINE", "TUNED"),
                        help="Compare two result JSONs")
    parser.add_argument("--list", action="store_true",
                        help="List available models and scenarios")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Correctness-only mode: verify all GEMMs, skip timing")
    parser.add_argument("--stress-test", action="store_true",
                        help="Run edge-case stress shapes (alignment, primes, extremes)")
    args = parser.parse_args()

    # --- List mode ---
    if args.list:
        print("Models:")
        for name, m in MODELS.items():
            shapes = m.gemm_shapes(2048)
            moe_str = (f" MoE({m.num_experts}e/{m.experts_active}a)"
                       if m.num_experts > 1 else "")
            print(f"  {name:<20} h={m.hidden}, ffn={m.intermediate}, "
                  f"heads={m.num_heads}/{m.kv_heads}{moe_str} — {len(shapes)} GEMMs/layer")
        print("\nScenarios:")
        for name, s in SCENARIOS.items():
            print(f"  {name:<20} M={s['M']:<6} {s['desc']}")
        print(f"\nStress shapes: {len(STRESS_SHAPES)} edge-case configurations")
        return

    # --- Compare mode ---
    if args.compare:
        compare_results(args.compare[0], args.compare[1])
        return

    # --- Common setup (env vars BEFORE torch import) ---
    if args.rocblas_log:
        os.environ["ROCBLAS_LAYER"] = "2"

    # Configure TunableOp to log kernel selections to a timestamped CSV
    tunableop_file = setup_tunableop(args.output, args.tag)

    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch not found. Install with ROCm support:")
        print("  pip install torch --index-url https://download.pytorch.org/whl/rocm6.3")
        sys.exit(1)

    if not torch.cuda.is_available():
        print("ERROR: No CUDA/ROCm GPU available.")
        sys.exit(1)

    dtype = get_torch_dtype(args.dtype)
    default_atol, default_rtol = _default_tolerances(args.dtype)
    atol = args.atol if args.atol is not None else default_atol
    rtol = args.rtol if args.rtol is not None else default_rtol

    model_names = args.models.split(",") if args.models else list(MODELS.keys())
    scenario_names = args.scenarios.split(",") if args.scenarios else list(SCENARIOS.keys())

    for name in model_names:
        if name not in MODELS:
            print(f"ERROR: Unknown model '{name}'. Use --list to see options.")
            sys.exit(1)
    for name in scenario_names:
        if name not in SCENARIOS:
            print(f"ERROR: Unknown scenario '{name}'. Use --list to see options.")
            sys.exit(1)

    sys_info = collect_system_info(tunableop_file)
    print(f"GPU: {sys_info['gpu_name']}")
    print(f"PyTorch: {sys_info['pytorch_version']}")
    print(f"ROCm: {sys_info['rocm_version']}")
    print(f"rocBLAS state: {sys_info.get('rocblas_library_state', 'unknown')}")
    print(f"  tuned .co: {sys_info.get('rocblas_gfx1201_tuned_co', '?')}, "
          f"fallback .hsaco: {sys_info.get('rocblas_gfx1201_fallback_hsaco', '?')}")
    print(f"TunableOp log: {tunableop_file}")
    print(f"Dtype: {args.dtype} | Tolerances: atol={atol}, rtol={rtol}")

    # --- Smoke test mode ---
    if args.smoke_test:
        print(f"\n{'━' * 80}")
        print(f"  SMOKE TEST — Numerical correctness verification")
        print(f"  Comparing GPU rocBLAS results against fp32 CPU reference")
        print(f"{'━' * 80}")

        print(f"\n  Model-layer GEMMs:")
        layer_p, layer_f, layer_failures = run_smoke_test(
            model_names, scenario_names, dtype, args.dtype, atol, rtol)

        stress_p = stress_f = 0
        stress_failures = []
        if args.stress_test or True:  # Always run stress in smoke-test mode
            print(f"\n  Edge-case stress shapes:")
            stress_p, stress_f, stress_failures = run_stress_test(
                dtype, args.dtype, atol, rtol)

        total_p = layer_p + stress_p
        total_f = layer_f + stress_f
        all_failures = layer_failures + stress_failures

        print(f"\n{'━' * 80}")
        print(f"  SMOKE TEST RESULT: {total_p} PASSED, {total_f} FAILED "
              f"({'ALL CLEAR ✓' if total_f == 0 else 'FAILURES DETECTED ✗'})")
        print(f"{'━' * 80}")

        if all_failures:
            print("\n  Failed shapes:")
            for f in all_failures:
                print(f"    {f.get('model', f.get('label', '?'))}: "
                      f"{f.get('op', '')}: {f['shape']} — "
                      f"mismatched={f['mismatched']}/{f['total_elements']} "
                      f"max_abs={f['max_abs_err']:.6f}")

        # Save smoke test results
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        smoke_file = output_dir / f"smoke_test_{args.dtype}_{args.tag}.json"
        with open(smoke_file, "w") as fh:
            json.dump({
                "tag": args.tag, "system": sys_info,
                "tunableop_file": tunableop_file,
                "dtype": args.dtype, "atol": atol, "rtol": rtol,
                "passed": total_p, "failed": total_f,
                "failures": all_failures,
            }, fh, indent=2)
        print(f"\n  Results saved to {smoke_file}")
        tunableop_path = Path(tunableop_file)
        if tunableop_path.exists():
            line_count = sum(1 for _ in open(tunableop_path))
            print(f"  TunableOp log: {tunableop_file} ({line_count} entries)")

        sys.exit(1 if total_f > 0 else 0)

    # --- Stress test only mode ---
    if args.stress_test and not args.smoke_test:
        print(f"\n  Edge-case stress test:")
        s_p, s_f, s_failures = run_stress_test(dtype, args.dtype, atol, rtol)
        print(f"\n  STRESS TEST: {s_p} PASSED, {s_f} FAILED")
        if s_failures:
            for f in s_failures:
                print(f"    {f['label']}: {f['shape']} — mismatched={f['mismatched']}")
        sys.exit(1 if s_f > 0 else 0)

    # --- Benchmark mode ---
    do_verify = not args.no_verify
    min_sample_ms = args.min_sample_ms
    print(f"Warmup: {args.warmup} | Iters: {args.iters} | Layers: {args.layers} | Min sample: {min_sample_ms}ms")
    print(f"Verification: {'ON' if do_verify else 'OFF'}")
    print(f"Tag: {args.tag}")
    print()

    results = []
    verify_summary = {"passed": 0, "failed": 0, "failures": []}
    total_combos = len(model_names) * len(scenario_names)
    combo_idx = 0

    for model_name in model_names:
        model = MODELS[model_name]
        print(f"━━━ {model.name}: {model.description} ━━━")

        for scenario_name in scenario_names:
            scen = SCENARIOS[scenario_name]
            M = scen["M"]
            combo_idx += 1
            print(f"  [{combo_idx}/{total_combos}] {scenario_name} (M={M})...",
                  end="", flush=True)

            layer_result = benchmark_layer(
                model, M, dtype, args.dtype,
                args.warmup, args.iters,
                num_layers=args.layers, device="cuda",
                do_verify=do_verify, atol=atol, rtol=rtol,
                min_sample_ms=min_sample_ms,
            )
            layer_result["scenario"] = scenario_name
            layer_result["dtype"] = args.dtype

            # Summarize verification
            v_status = ""
            if do_verify:
                for g in layer_result["individual_gemms"]:
                    vr = g.get("verify", {})
                    if vr.get("pass") is True:
                        verify_summary["passed"] += 1
                    elif vr.get("pass") is False:
                        verify_summary["failed"] += 1
                        verify_summary["failures"].append({
                            "model": model_name, "scenario": scenario_name,
                            "op": g["op"],
                            "shape": f"{g['M']}x{g['N']}x{g['K']}",
                        })
                all_ok = all(g.get("verify", {}).get("pass", True)
                             for g in layer_result["individual_gemms"])
                v_status = " ✓" if all_ok else " ✗ VERIFY FAIL"

            print(f" layer={layer_result['layer_median_ms']:.3f}ms, "
                  f"{layer_result['layer_tflops']:.2f} TFLOPS{v_status}")

            for g in layer_result["individual_gemms"]:
                v_sym = ""
                if do_verify:
                    vr = g.get("verify", {})
                    v_sym = " ✓" if vr.get("pass") else f" ✗ err={vr.get('max_abs_err','?')}"
                reps = g.get('repeats', 1)
                rep_str = f" r={reps}" if reps > 1 else ""
                print(f"    {g['op']:<25} {g['M']:>5}×{g['N']:>5}×{g['K']:>5}  "
                      f"{g['median_ms']:>8.4f}ms  {g['tflops']:>7.2f} TF{rep_str}{v_sym}")

            results.append(layer_result)
        print()

    # --- LM Head ---
    lm_head_results = {}
    if not args.no_lm_head:
        print("━━━ LM Head (vocab projection) ━━━")
        for model_name in model_names:
            model = MODELS[model_name]
            for sname in [s for s in scenario_names if "prefill" in s]:
                M = SCENARIOS[sname]["M"]
                key = f"{model_name}_{sname}"
                print(f"  {key}: [{M}×{model.vocab_size}×{model.hidden}]...",
                      end="", flush=True)
                lmh = benchmark_lm_head(
                    model, M, dtype, args.dtype,
                    args.warmup, args.iters,
                    do_verify=do_verify, atol=atol, rtol=rtol,
                    min_sample_ms=min_sample_ms)
                lm_head_results[key] = lmh
                v_sym = ""
                if do_verify:
                    vr = lmh.get("verify", {})
                    if vr.get("pass") is True:
                        verify_summary["passed"] += 1
                        v_sym = " ✓"
                    elif vr.get("pass") is False:
                        verify_summary["failed"] += 1
                        v_sym = f" ✗ err={vr.get('max_abs_err','?')}"
                print(f" {lmh['median_ms']:.4f}ms, {lmh['tflops']:.2f} TF{v_sym}")
        print()

    # --- Verification summary ---
    if do_verify:
        total_v = verify_summary["passed"] + verify_summary["failed"]
        status = "ALL CLEAR ✓" if verify_summary["failed"] == 0 else "FAILURES DETECTED ✗"
        print(f"Verification: {verify_summary['passed']}/{total_v} passed — {status}")
        if verify_summary["failures"]:
            for f in verify_summary["failures"]:
                print(f"  ✗ {f['model']}/{f['scenario']}: {f['op']} {f['shape']}")
        print()

    # --- Save ---
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{args.tag}.json"

    output_data = {
        "tag": args.tag,
        "system": sys_info,
        "config": {
            "dtype": args.dtype,
            "warmup": args.warmup,
            "iters": args.iters,
            "layers": args.layers,
            "models": model_names,
            "scenarios": scenario_names,
            "verify": do_verify,
            "atol": atol,
            "rtol": rtol,
        },
        "results": results,
        "lm_head": lm_head_results,
        "verify_summary": verify_summary if do_verify else None,
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Results saved to {output_file}")

    # Report tunableop file
    tunableop_path = Path(tunableop_file)
    if tunableop_path.exists():
        line_count = sum(1 for _ in open(tunableop_path))
        print(f"TunableOp log: {tunableop_file} ({line_count} entries)")
    else:
        print(f"TunableOp log: {tunableop_file} (not written — may require PyTorch ≥2.3)")

    if do_verify and verify_summary["failed"] > 0:
        print(f"\n⚠ {verify_summary['failed']} verification failures detected!")
        print("  New kernels may be producing incorrect results.")
        sys.exit(1)

    print(f"\nNext steps:")
    if "baseline" in args.tag:
        print(f"  1. Install tuned kernels to /opt/rocm/lib/rocblas/library/")
        print(f"  2. python3 {sys.argv[0]} --tag tuned --output {args.output}")
        print(f"  3. python3 {sys.argv[0]} --compare {output_file} {output_dir}/tuned.json")
    else:
        print(f"  python3 {sys.argv[0]} --compare <baseline.json> {output_file}")


if __name__ == "__main__":
    main()
