#!/usr/bin/env bash
# tensile_ab_bench.sh — A/B benchmark of two Tensile libraries via rocblas-bench
# Usage: ./tensile_ab_bench.sh [results_dir]
# Runtime target: ~10 minutes total (42 shapes × 2 libraries)
#
# Compares ROCBLAS_TENSILE_LIBPATH=A vs B across real-world GEMM shapes.
# Shapes are workload-representative, not tailored to any specific kernel set.

set -euo pipefail

LIB_A="/home/iamthecage/library_backup/original"
LIB_B="/home/iamthecage/library_backup/new"
RESULTS_DIR="${1:-/home/iamthecage/benchmarks/tensile_ab_results}"

# Target ~1s of GPU time per test. We scale iters per shape so small
# problems get enough iterations to amortize noise, big ones don't take forever.
# Rough per-iter timings (bf16 NN baseline from calibration):
#   128x4096x4096  ~ 100 us    → 5000 iters for 0.5s
#   512x4096x4096  ~ 625 us    → 1000 iters
#   1024x4096x4096 ~ 1500 us   → 500 iters
#   4096x4096x4096 ~ 6000 us   → 150 iters
# We use a compute-budget approach: target_us / estimated_us = iters
TARGET_US=750000  # 0.75s of GPU compute per test
COLD=20           # enough cold iters to stabilize clocks

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="$RESULTS_DIR/$TIMESTAMP"
mkdir -p "$RUN_DIR"
CSV_A="$RUN_DIR/lib_A.csv"
CSV_B="$RUN_DIR/lib_B.csv"
REPORT="$RUN_DIR/comparison.txt"

# ──────────────────────────────────────────────────────────────────────
# Test matrix: real-world GEMM shapes, agnostic to library contents
# Format: "M N K transA transB dtype est_us_per_iter"
#
# est_us is a rough guess for iteration scaling only — not used in results.
# Shapes from: LLM inference/prefill, vision, MoE, general compute.
# ──────────────────────────────────────────────────────────────────────
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
    # ══════════════════════════════════════════════════════════════
    # EXPANDED DIAGNOSTIC SHAPES — mapping tuning coverage gaps
    # ══════════════════════════════════════════════════════════════

    # ── M-ladder: fixed N=4096 K=4096, sweep M to find cliffs ──
    # M=128 and M=512 are absent from M_STD=[16,64,256,1024,4096]
    "16 4096 4096 N N bf16_r 25"
    "128 4096 4096 N N bf16_r 150"
    "512 4096 4096 N N bf16_r 700"
    "8192 4096 4096 N N bf16_r 12000"

    # ── Real model dimensions (DeepSeek, Qwen, Llama-70B) ──
    "256 7168 7168 N N bf16_r 900"
    "1024 3584 3584 N N bf16_r 900"
    "256 8192 28672 N N bf16_r 4500"

    # ── f16 at the M=2048 cliff (was it dtype-specific?) ──
    "2048 4096 4096 N N f16_r 3000"

    # ── NT transpose at untuned M values ──
    "128 4096 4096 N T bf16_r 150"
    "512 4096 4096 N T bf16_r 700"

    # ── TT transpose (never tested, worst-case layout) ──
    "1024 4096 4096 T T bf16_r 1500"
    "256 4096 4096 T T bf16_r 250"

    # ── Large N shapes (MoE / wide projection) ──
    "256 14336 4096 N N bf16_r 900"
    "256 28672 8192 N N bf16_r 4500"

    # ── GSU diagnostic: tiny M, huge NK (needs GlobalSplitU) ──
    "16 14336 14336 N N bf16_r 1500"

    # ── Non-power-of-2 M (misalignment stress test) ──
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
echo " Tensile Library A/B Benchmark"
echo " Library A: $LIB_A"
echo " Library B: $LIB_B"
echo " Tests: $NUM_TESTS shapes × 2 libraries"
echo " Target GPU time per test: ${TARGET_US} us ($(echo "scale=2; $TARGET_US/1000000" | bc)s)"
echo " Results: $RUN_DIR"
echo "═══════════════════════════════════════════════════════════════"

# ──────────────────────────────────────────────────────────────────────
# run_pass: benchmark all shapes against one library
# Args: $1=library_path $2=output_csv $3=label
# ──────────────────────────────────────────────────────────────────────
run_pass() {
    local libpath="$1" csv="$2" label="$3"

    echo ""
    echo "─── Pass: $label ───"
    echo "    ROCBLAS_TENSILE_LIBPATH=$libpath"

    export ROCBLAS_TENSILE_LIBPATH="$libpath"

    echo "M,N,K,transA,transB,dtype,iters,Gflops,us" > "$csv"

    local start_time=$(date +%s)
    local count=0

    for test in "${TESTS[@]}"; do
        read -r M N K tA tB dtype est_us <<< "$test"
        count=$((count + 1))

        # Scale iterations to hit target GPU time
        local iters=$(( TARGET_US / (est_us > 0 ? est_us : 100) ))
        [[ $iters -lt 50 ]] && iters=50
        [[ $iters -gt 10000 ]] && iters=10000

        local compute_type="f32_r"
        local cd_type="$dtype"
        if [[ "$dtype" == "i8_r" ]]; then
            compute_type="i32_r"
            cd_type="i32_r"
        fi

        local result
        result=$(rocblas-bench -f gemm_ex \
            --a_type "$dtype" --b_type "$dtype" --c_type "$cd_type" --d_type "$cd_type" \
            --compute_type "$compute_type" \
            -m "$M" -n "$N" -k "$K" \
            --transposeA "$tA" --transposeB "$tB" \
            --use_hipblaslt 0 \
            -i "$iters" -j "$COLD" 2>&1 | grep -E "^[NT]," | head -1) || true

        if [[ -n "$result" ]]; then
            local gflops us
            gflops=$(echo "$result" | awk -F',' '{print $(NF-1)}' | tr -d ' ')
            us=$(echo "$result" | awk -F',' '{print $NF}' | tr -d ' ')
            echo "$M,$N,$K,$tA,$tB,$dtype,$iters,$gflops,$us" >> "$csv"
            printf "  [%2d/%d] M=%-5s N=%-5s K=%-5s %s%s %-5s  i=%-5d → %10s GF  %10s us\n" \
                "$count" "$NUM_TESTS" "$M" "$N" "$K" "$tA" "$tB" "$dtype" "$iters" "$gflops" "$us"
        else
            echo "$M,$N,$K,$tA,$tB,$dtype,$iters,FAIL,FAIL" >> "$csv"
            printf "  [%2d/%d] M=%-5s N=%-5s K=%-5s %s%s %-5s  i=%-5d → FAILED\n" \
                "$count" "$NUM_TESTS" "$M" "$N" "$K" "$tA" "$tB" "$dtype" "$iters"
        fi
    done

    local end_time=$(date +%s)
    echo "    Pass complete: $((end_time - start_time))s"
    unset ROCBLAS_TENSILE_LIBPATH
}

# ──────────────────────────────────────────────────────────────────────
# Run both passes
# ──────────────────────────────────────────────────────────────────────
TOTAL_START=$(date +%s)
run_pass "$LIB_A" "$CSV_A" "Library A (original)"
run_pass "$LIB_B" "$CSV_B" "Library B (new)"
TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$((TOTAL_END - TOTAL_START))

# ──────────────────────────────────────────────────────────────────────
# Comparison report
# ──────────────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo " COMPARISON REPORT"
echo "═══════════════════════════════════════════════════════════════"

{
    echo "Tensile Library A/B Comparison — $(date)"
    echo "Library A (original): $LIB_A"
    echo "Library B (new):      $LIB_B"
    echo "Total runtime: ${TOTAL_ELAPSED}s"
    echo ""
    printf "%-6s %-6s %-6s %-3s %-6s %12s %12s %12s %12s %8s\n" \
        "M" "N" "K" "tr" "dtype" "GF_A" "GF_B" "us_A" "us_B" "B_vs_A"
    printf "%s\n" "────── ────── ────── ─── ────── ──────────── ──────────── ──────────── ──────────── ────────"

    paste -d'|' <(tail -n+2 "$CSV_A") <(tail -n+2 "$CSV_B") | while IFS='|' read -r lineA lineB; do
        IFS=',' read -r M N K tA tB dtype iA gf_A us_A <<< "$lineA"
        IFS=',' read -r _ _ _ _ _ _ iB gf_B us_B <<< "$lineB"

        trans="${tA}${tB}"

        if [[ "$gf_A" == "FAIL" || "$gf_B" == "FAIL" ]]; then
            printf "%-6s %-6s %-6s %-3s %-6s %12s %12s %12s %12s %8s\n" \
                "$M" "$N" "$K" "$trans" "$dtype" "$gf_A" "$gf_B" "$us_A" "$us_B" "N/A"
            continue
        fi

        speedup=$(awk "BEGIN { printf \"%+.1f%%\", ($us_A/$us_B - 1)*100 }")

        printf "%-6s %-6s %-6s %-3s %-6s %12s %12s %12s %12s %8s\n" \
            "$M" "$N" "$K" "$trans" "$dtype" "$gf_A" "$gf_B" "$us_A" "$us_B" "$speedup"
    done

    echo ""
    echo "(B_vs_A: positive% = B faster, negative% = A faster)"
} | tee "$REPORT"

# ──────────────────────────────────────────────────────────────────────
# Summary stats via awk
# ──────────────────────────────────────────────────────────────────────
echo ""
echo "─── Summary ───"
awk -F',' '
NR==FNR && FNR>1 {
    key = $1","$2","$3","$4","$5","$6
    us_a[key] = $9
    gf_a[key] = $8
    next
}
FNR>1 {
    key = $1","$2","$3","$4","$5","$6
    if (key in us_a && $9 != "FAIL" && us_a[key] != "FAIL") {
        ua = us_a[key] + 0; ub = $9 + 0
        ga = gf_a[key] + 0; gb = $8 + 0
        if (ua > 0 && ub > 0) {
            n++
            spd = (ua/ub - 1) * 100
            sum += spd
            sum_ga += ga; sum_gb += gb
            if (spd > 1) wins_b++
            else if (spd < -1) wins_a++
            else ties++
            if (n == 1 || spd > max_spd) { max_spd = spd; max_key = key }
            if (n == 1 || spd < min_spd) { min_spd = spd; min_key = key }
        }
    }
}
END {
    printf "  Tests compared:      %d\n", n
    printf "  B wins (>1%%):        %d\n", wins_b
    printf "  A wins (>1%%):        %d\n", wins_a
    printf "  Ties (within ±1%%):   %d\n", ties
    printf "  Mean speedup (B/A):  %+.2f%%\n", (n>0 ? sum/n : 0)
    printf "  Aggregate GF A:      %.0f\n", sum_ga
    printf "  Aggregate GF B:      %.0f\n", sum_gb
    printf "  Best case for B:     %+.1f%%  (%s)\n", max_spd, max_key
    printf "  Best case for A:     %+.1f%%  (%s)\n", min_spd, min_key
}' "$CSV_A" "$CSV_B" | tee -a "$REPORT"

echo ""
echo "Total runtime: ${TOTAL_ELAPSED}s"
echo ""
echo "Results:  $RUN_DIR/"
echo "  CSVs:   lib_A.csv, lib_B.csv"
echo "  Report: comparison.txt"
echo "═══════════════════════════════════════════════════════════════"
