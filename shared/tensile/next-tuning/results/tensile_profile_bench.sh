#!/usr/bin/env bash
# tensile_profile_bench.sh — Single-library benchmark for RDTS profiling
# Usage: ./tensile_profile_bench.sh [library_path] [results_dir]
#
# Runs all 48 GEMM shapes against one Tensile library.
# Designed to be wrapped with RGP/RRA capture from Radeon Developer Tool Suite.
# Use fewer iterations (TARGET_US=200000) to keep total runtime manageable
# while still getting meaningful kernel traces.

set -euo pipefail

LIB="${1:-/home/iamthecage/library_backup/new}"
RESULTS_DIR="${2:-/home/iamthecage/benchmarks/tensile_profile}"

# Lower target for profiling — enough iterations to see kernel behavior,
# not so many that RDTS captures become enormous.
TARGET_US=200000  # 0.2s of GPU compute per test
COLD=5            # fewer cold iters for profiling

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="$RESULTS_DIR/$TIMESTAMP"
mkdir -p "$RUN_DIR"
CSV="$RUN_DIR/profile.csv"

TESTS=(
    # ── LLM single-token decode (tiny M, large NK) ──
    "1 4096 4096 N N bf16_r 25"
    "1 4096 14336 N N bf16_r 60"
    "8 4096 4096 N N bf16_r 30"
    "8 4096 14336 N N bf16_r 65"

    # ── LLM small-batch decode ──
    "32 4096 4096 N N bf16_r 60"
    "32 4096 14336 N N bf16_r 180"
    "64 4096 4096 N N bf16_r 90"
    "64 4096 14336 N N bf16_r 350"

    # ── LLM prefill (medium-large M) ──
    "256 4096 4096 N N bf16_r 250"
    "512 4096 14336 N N bf16_r 1100"
    "1024 4096 4096 N N bf16_r 1500"
    "2048 4096 4096 N N bf16_r 3000"

    # ── Square (general compute baseline) ──
    "2048 2048 2048 N N bf16_r 500"
    "4096 4096 4096 N N bf16_r 6000"

    # ── Transposed (training / backward pass) ──
    "1024 4096 4096 N T bf16_r 1500"
    "1024 4096 4096 T N bf16_r 1500"
    "2048 4096 4096 N T bf16_r 3000"

    # ── f16 (same key shapes, different type path) ──
    "32 4096 4096 N N f16_r 60"
    "256 4096 4096 N N f16_r 250"
    "1024 4096 4096 N N f16_r 1500"
    "4096 4096 4096 N N f16_r 6000"
    "1024 4096 4096 N T f16_r 1500"

    # ── f32 (fallback path, non-WMMA) ──
    "256 4096 4096 N N f32_r 500"
    "1024 1024 1024 N N f32_r 200"
    "2048 2048 2048 N N f32_r 2500"

    # ── M-ladder diagnostic ──
    "16 4096 4096 N N bf16_r 25"
    "128 4096 4096 N N bf16_r 150"
    "512 4096 4096 N N bf16_r 700"
    "8192 4096 4096 N N bf16_r 12000"

    # ── Real model dimensions ──
    "256 7168 7168 N N bf16_r 900"
    "1024 3584 3584 N N bf16_r 900"
    "256 8192 28672 N N bf16_r 4500"

    # ── f16 at M=2048 cliff ──
    "2048 4096 4096 N N f16_r 3000"

    # ── NT transpose at untuned M ──
    "128 4096 4096 N T bf16_r 150"
    "512 4096 4096 N T bf16_r 700"

    # ── TT transpose ──
    "1024 4096 4096 T T bf16_r 1500"
    "256 4096 4096 T T bf16_r 250"

    # ── Large N shapes ──
    "256 14336 4096 N N bf16_r 900"
    "256 28672 8192 N N bf16_r 4500"

    # ── GSU diagnostic ──
    "16 14336 14336 N N bf16_r 1500"

    # ── Non-power-of-2 M ──
    "96 4096 4096 N N bf16_r 100"
    "384 4096 4096 N N bf16_r 400"

    # ── i8 (quantized inference) ──
    "256 4096 4096 N N i8_r 120"
    "1024 4096 4096 N N i8_r 700"
    "2048 4096 4096 N N i8_r 1500"
    "4096 4096 4096 N N i8_r 3000"
    "1024 4096 4096 N T i8_r 700"
    "256 14336 4096 N N i8_r 450"
)

NUM_TESTS=${#TESTS[@]}
echo "═══════════════════════════════════════════════════════════════"
echo " Tensile Single-Library Profile Benchmark"
echo " Library: $LIB"
echo " Tests: $NUM_TESTS shapes"
echo " Target GPU time per test: ${TARGET_US} us"
echo " Results: $RUN_DIR"
echo "═══════════════════════════════════════════════════════════════"

export ROCBLAS_TENSILE_LIBPATH="$LIB"

echo "M,N,K,transA,transB,dtype,iters,Gflops,us" > "$CSV"

TOTAL_START=$(date +%s)
count=0

for test in "${TESTS[@]}"; do
    read -r M N K tA tB dtype est_us <<< "$test"
    count=$((count + 1))

    iters=$(( TARGET_US / (est_us > 0 ? est_us : 100) ))
    [[ $iters -lt 20 ]] && iters=20
    [[ $iters -gt 5000 ]] && iters=5000

    compute_type="f32_r"
    cd_type="$dtype"
    if [[ "$dtype" == "i8_r" ]]; then
        compute_type="i32_r"
        cd_type="i32_r"
    fi

    result=$(rocblas-bench -f gemm_ex \
        --a_type "$dtype" --b_type "$dtype" --c_type "$cd_type" --d_type "$cd_type" \
        --compute_type "$compute_type" \
        -m "$M" -n "$N" -k "$K" \
        --transposeA "$tA" --transposeB "$tB" \
        --use_hipblaslt 0 \
        -i "$iters" -j "$COLD" 2>&1 | grep -E "^[NT]," | head -1) || true

    if [[ -n "$result" ]]; then
        gflops=$(echo "$result" | awk -F',' '{print $(NF-1)}' | tr -d ' ')
        us=$(echo "$result" | awk -F',' '{print $NF}' | tr -d ' ')
        echo "$M,$N,$K,$tA,$tB,$dtype,$iters,$gflops,$us" >> "$CSV"
        printf "  [%2d/%d] M=%-5s N=%-5s K=%-5s %s%s %-5s  i=%-5d → %10s GF  %10s us\n" \
            "$count" "$NUM_TESTS" "$M" "$N" "$K" "$tA" "$tB" "$dtype" "$iters" "$gflops" "$us"
    else
        echo "$M,$N,$K,$tA,$tB,$dtype,$iters,FAIL,FAIL" >> "$CSV"
        printf "  [%2d/%d] M=%-5s N=%-5s K=%-5s %s%s %-5s  i=%-5d → FAILED\n" \
            "$count" "$NUM_TESTS" "$M" "$N" "$K" "$tA" "$tB" "$dtype" "$iters"
    fi
done

TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$((TOTAL_END - TOTAL_START))

unset ROCBLAS_TENSILE_LIBPATH

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo " Profile Summary"
echo "═══════════════════════════════════════════════════════════════"
echo "  Library:  $LIB"
echo "  Runtime:  ${TOTAL_ELAPSED}s"
echo "  Results:  $CSV"
echo ""

# Peak efficiency calculation (bf16 theoretical: 107000 GFlops)
awk -F',' '
NR > 1 && $8 != "FAIL" {
    n++
    gf = $8 + 0
    sum_gf += gf
    if (n == 1 || gf > max_gf) { max_gf = gf; max_key = $1"x"$2"x"$3" "$4$5" "$6 }
    if (n == 1 || gf < min_gf) { min_gf = gf; min_key = $1"x"$2"x"$3" "$4$5" "$6 }
    # Estimate peak for dtype
    if ($6 == "bf16_r" || $6 == "f16_r") peak = 107000
    else if ($6 == "f32_r") peak = 3344
    else if ($6 == "i8_r") peak = 214000
    else peak = 107000
    eff = gf / peak * 100
    sum_eff += eff
}
END {
    printf "  Tests run:       %d\n", n
    printf "  Mean GFlops:     %.0f\n", (n>0 ? sum_gf/n : 0)
    printf "  Mean efficiency: %.1f%%\n", (n>0 ? sum_eff/n : 0)
    printf "  Best kernel:     %.0f GF  (%s)\n", max_gf, max_key
    printf "  Worst kernel:    %.0f GF  (%s)\n", min_gf, min_key
}' "$CSV"

echo "═══════════════════════════════════════════════════════════════"
