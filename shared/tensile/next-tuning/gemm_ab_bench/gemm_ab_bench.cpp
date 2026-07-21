// gemm_ab_bench.cpp — A/B Benchmark: rocBLAS vs hipBLASLt
// Direct API comparison, no framework overhead.
// Build: hipcc -O3 -o gemm_ab_bench gemm_ab_bench.cpp -lrocblas -lhipblaslt -std=c++17

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <rocblas/rocblas.h>
#include <hipblaslt/hipblaslt.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <chrono>

// ─── Error Checking ─────────────────────────────────────────────────────────
#define HIP_CHECK(x) do { \
    hipError_t err = (x); \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP error %d at %s:%d: %s\n", err, __FILE__, __LINE__, hipGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#define ROCBLAS_CHECK(x) do { \
    rocblas_status s = (x); \
    if (s != rocblas_status_success) { \
        fprintf(stderr, "rocBLAS error %d at %s:%d\n", s, __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

#define HIPBLASLT_CHECK(x) do { \
    hipblasStatus_t s = (x); \
    if (s != HIPBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "hipBLASLt error %d at %s:%d\n", s, __FILE__, __LINE__); \
        return -1.0; \
    } \
} while(0)

// ─── Config ─────────────────────────────────────────────────────────────────
static constexpr double MIN_BENCH_SECS  = 10.0;
static constexpr int    WARMUP_ITERS    = 10;
static constexpr int    CALIBRATE_ITERS = 20;
static constexpr size_t WORKSPACE_SIZE  = 256ULL * 1024 * 1024; // 256 MB

// ─── Data Type Descriptor ───────────────────────────────────────────────────
struct DTypeConfig {
    const char*        name;
    rocblas_datatype   rb_ab;       // rocblas A/B type
    rocblas_datatype   rb_cd;       // rocblas C/D type
    rocblas_datatype   rb_compute;  // rocblas compute type
    hipDataType        hl_ab;       // hipblaslt A/B type
    hipDataType        hl_cd;       // hipblaslt C/D type
    hipblasComputeType_t hl_compute;
    int elem_bytes_ab;
    int elem_bytes_cd;
};

static const DTypeConfig DTYPES[] = {
    { "FP16_F32",
      rocblas_datatype_f16_r,  rocblas_datatype_f16_r, rocblas_datatype_f32_r,
      HIP_R_16F, HIP_R_16F, HIPBLAS_COMPUTE_32F,
      2, 2 },
    { "BF16_F32",
      rocblas_datatype_bf16_r, rocblas_datatype_bf16_r, rocblas_datatype_f32_r,
      HIP_R_16BF, HIP_R_16BF, HIPBLAS_COMPUTE_32F,
      2, 2 },
    { "FP32",
      rocblas_datatype_f32_r,  rocblas_datatype_f32_r, rocblas_datatype_f32_r,
      HIP_R_32F, HIP_R_32F, HIPBLAS_COMPUTE_32F,
      4, 4 },
};
static constexpr int NUM_DTYPES = sizeof(DTYPES) / sizeof(DTYPES[0]);

// ─── Problem Definition ─────────────────────────────────────────────────────
struct Problem {
    int M, N, K;
    bool transA, transB;
};

static const Problem PROBLEMS[] = {
    // Large square — compute-bound stress tests
    { 4096,  4096,  4096,  false, false },
    { 4096,  4096,  4096,  true,  false },
    { 8192,  8192,  8192,  false, false },
    { 8192,  8192,  8192,  true,  false },
    // LLM inference shapes  (small M = batch tokens, large N/K = model dims)
    {  256,  4096,  4096,  false, false },
    { 1024,  4096,  4096,  false, false },
    { 2048,  4096, 14336,  false, false },
    { 4096, 14336,  4096,  false, false },
    // LLM training shapes  (large M = sequence*batch)
    { 4096,  4096,  8192,  true,  false },
    { 8192,  4096,  4096,  true,  false },
    { 4096,  8192, 14336,  true,  false },
};
static constexpr int NUM_PROBLEMS = sizeof(PROBLEMS) / sizeof(PROBLEMS[0]);

// ─── Helpers ────────────────────────────────────────────────────────────────
static double tflops(int M, int N, int K, double us) {
    double flops = 2.0 * M * N * K;
    return flops / (us * 1e6);  // TFLOPS
}

static void fill_random(void* dst, size_t bytes) {
    // Fill host buffer with pseudo-random pattern, then copy
    std::vector<uint8_t> buf(bytes);
    uint32_t seed = 42;
    for (size_t i = 0; i < bytes; i += 4) {
        seed = seed * 1664525u + 1013904223u;
        size_t n = std::min((size_t)4, bytes - i);
        memcpy(buf.data() + i, &seed, n);
    }
    HIP_CHECK(hipMemcpy(dst, buf.data(), bytes, hipMemcpyHostToDevice));
}

static void get_dims(const Problem& p,
                     int& rowA, int& colA, int& lda,
                     int& rowB, int& colB, int& ldb) {
    // Column-major: stored dim is rows x cols, lda = rows
    if (!p.transA) { rowA = p.M; colA = p.K; }
    else           { rowA = p.K; colA = p.M; }
    lda = rowA;

    if (!p.transB) { rowB = p.K; colB = p.N; }
    else           { rowB = p.N; colB = p.K; }
    ldb = rowB;
}

// ─── rocBLAS benchmark ─────────────────────────────────────────────────────
static double bench_rocblas(rocblas_handle handle, const Problem& p,
                            const DTypeConfig& dt, int iters,
                            void* dA, void* dB, void* dC, void* dD) {
    float alpha = 1.0f, beta = 0.0f;
    auto opA = p.transA ? rocblas_operation_transpose : rocblas_operation_none;
    auto opB = p.transB ? rocblas_operation_transpose : rocblas_operation_none;
    int rowA, colA, lda, rowB, colB, ldb;
    get_dims(p, rowA, colA, lda, rowB, colB, ldb);

    // Warmup
    for (int i = 0; i < WARMUP_ITERS; i++) {
        ROCBLAS_CHECK(rocblas_gemm_ex(handle, opA, opB,
            p.M, p.N, p.K, &alpha,
            dA, dt.rb_ab, lda,
            dB, dt.rb_ab, ldb, &beta,
            dC, dt.rb_cd, p.M,
            dD, dt.rb_cd, p.M,
            dt.rb_compute, rocblas_gemm_algo_standard, 0, 0));
    }
    HIP_CHECK(hipDeviceSynchronize());

    // Timed run
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start));
    for (int i = 0; i < iters; i++) {
        ROCBLAS_CHECK(rocblas_gemm_ex(handle, opA, opB,
            p.M, p.N, p.K, &alpha,
            dA, dt.rb_ab, lda,
            dB, dt.rb_ab, ldb, &beta,
            dC, dt.rb_cd, p.M,
            dD, dt.rb_cd, p.M,
            dt.rb_compute, rocblas_gemm_algo_standard, 0, 0));
    }
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    float ms = 0;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    return (double)ms * 1000.0 / iters;  // avg microseconds
}

// ─── hipBLASLt benchmark ───────────────────────────────────────────────────
static double bench_hipblaslt(hipblasLtHandle_t handle, const Problem& p,
                              const DTypeConfig& dt, int iters,
                              void* dA, void* dB, void* dC, void* dD,
                              void* workspace, hipStream_t stream) {
    float alpha = 1.0f, beta = 0.0f;
    int rowA, colA, lda, rowB, colB, ldb;
    get_dims(p, rowA, colA, lda, rowB, colB, ldb);

    auto opA = p.transA ? HIPBLAS_OP_T : HIPBLAS_OP_N;
    auto opB = p.transB ? HIPBLAS_OP_T : HIPBLAS_OP_N;

    // Create descriptors
    hipblasLtMatmulDesc_t matmulDesc = nullptr;
    HIPBLASLT_CHECK(hipblasLtMatmulDescCreate(&matmulDesc, dt.hl_compute, HIP_R_32F));
    HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(
        matmulDesc, HIPBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA)));
    HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(
        matmulDesc, HIPBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB)));

    hipblasLtMatrixLayout_t layoutA, layoutB, layoutC, layoutD;
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&layoutA, dt.hl_ab, rowA, colA, lda));
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&layoutB, dt.hl_ab, rowB, colB, ldb));
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&layoutC, dt.hl_cd, p.M, p.N, p.M));
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&layoutD, dt.hl_cd, p.M, p.N, p.M));

    // Get best algorithm
    hipblasLtMatmulPreference_t pref;
    HIPBLASLT_CHECK(hipblasLtMatmulPreferenceCreate(&pref));
    size_t wsSize = WORKSPACE_SIZE;
    HIPBLASLT_CHECK(hipblasLtMatmulPreferenceSetAttribute(
        pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsSize, sizeof(wsSize)));

    hipblasLtMatmulHeuristicResult_t heurResult;
    int returnedAlgoCount = 0;
    HIPBLASLT_CHECK(hipblasLtMatmulAlgoGetHeuristic(
        handle, matmulDesc, layoutA, layoutB, layoutC, layoutD,
        pref, 1, &heurResult, &returnedAlgoCount));

    if (returnedAlgoCount == 0) {
        fprintf(stderr, "    hipBLASLt: no algorithm found, skipping\n");
        hipblasLtMatmulPreferenceDestroy(pref);
        hipblasLtMatrixLayoutDestroy(layoutA);
        hipblasLtMatrixLayoutDestroy(layoutB);
        hipblasLtMatrixLayoutDestroy(layoutC);
        hipblasLtMatrixLayoutDestroy(layoutD);
        hipblasLtMatmulDescDestroy(matmulDesc);
        return -1.0;
    }

    // Warmup
    for (int i = 0; i < WARMUP_ITERS; i++) {
        hipblasLtMatmul(handle, matmulDesc, &alpha,
            dA, layoutA, dB, layoutB, &beta,
            dC, layoutC, dD, layoutD,
            &heurResult.algo, workspace, wsSize, stream);
    }
    HIP_CHECK(hipStreamSynchronize(stream));

    // Timed run
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start, stream));
    for (int i = 0; i < iters; i++) {
        hipblasLtMatmul(handle, matmulDesc, &alpha,
            dA, layoutA, dB, layoutB, &beta,
            dC, layoutC, dD, layoutD,
            &heurResult.algo, workspace, wsSize, stream);
    }
    HIP_CHECK(hipEventRecord(stop, stream));
    HIP_CHECK(hipEventSynchronize(stop));
    float ms = 0;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));

    // Cleanup
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    hipblasLtMatmulPreferenceDestroy(pref);
    hipblasLtMatrixLayoutDestroy(layoutA);
    hipblasLtMatrixLayoutDestroy(layoutB);
    hipblasLtMatrixLayoutDestroy(layoutC);
    hipblasLtMatrixLayoutDestroy(layoutD);
    hipblasLtMatmulDescDestroy(matmulDesc);

    return (double)ms * 1000.0 / iters;  // avg microseconds
}

// ─── Calibration: figure out iterations for target runtime ──────────────────
static int calibrate(rocblas_handle rb_handle, const Problem& p,
                     const DTypeConfig& dt,
                     void* dA, void* dB, void* dC, void* dD) {
    float alpha = 1.0f, beta = 0.0f;
    auto opA = p.transA ? rocblas_operation_transpose : rocblas_operation_none;
    auto opB = p.transB ? rocblas_operation_transpose : rocblas_operation_none;
    int rowA, colA, lda, rowB, colB, ldb;
    get_dims(p, rowA, colA, lda, rowB, colB, ldb);

    // Quick warmup
    for (int i = 0; i < 5; i++) {
        rocblas_gemm_ex(rb_handle, opA, opB, p.M, p.N, p.K, &alpha,
            dA, dt.rb_ab, lda, dB, dt.rb_ab, ldb, &beta,
            dC, dt.rb_cd, p.M, dD, dt.rb_cd, p.M,
            dt.rb_compute, rocblas_gemm_algo_standard, 0, 0);
    }
    HIP_CHECK(hipDeviceSynchronize());

    // Time a few iterations
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start));
    for (int i = 0; i < CALIBRATE_ITERS; i++) {
        rocblas_gemm_ex(rb_handle, opA, opB, p.M, p.N, p.K, &alpha,
            dA, dt.rb_ab, lda, dB, dt.rb_ab, ldb, &beta,
            dC, dt.rb_cd, p.M, dD, dt.rb_cd, p.M,
            dt.rb_compute, rocblas_gemm_algo_standard, 0, 0);
    }
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    float ms = 0;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));

    double per_call_ms = (double)ms / CALIBRATE_ITERS;
    int iters = std::max(100, (int)ceil(MIN_BENCH_SECS * 1000.0 / per_call_ms));
    return iters;
}

// ─── Usage ──────────────────────────────────────────────────────────────────
static void print_usage(const char* prog) {
    printf("Usage: %s [OPTIONS]\n", prog);
    printf("  --tensile-lib PATH   Override rocBLAS Tensile library path\n");
    printf("                       (sets ROCBLAS_TENSILE_LIBPATH before handle creation)\n");
    printf("  --help               Show this help\n");
}

// ─── Main ───────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    // Parse CLI args
    const char* tensile_lib_path = nullptr;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--tensile-lib") == 0 && i + 1 < argc) {
            tensile_lib_path = argv[++i];
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    // Set Tensile library override BEFORE creating rocBLAS handle
    if (tensile_lib_path) {
        setenv("ROCBLAS_TENSILE_LIBPATH", tensile_lib_path, 1);
        printf("*** ROCBLAS_TENSILE_LIBPATH = %s\n\n", tensile_lib_path);
    }

    // GPU info
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));

    // Check current tensile lib path (may come from env or CLI)
    const char* active_tensile = getenv("ROCBLAS_TENSILE_LIBPATH");

    // Print header
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗\n");
    printf("║  GEMM A/B Benchmark — rocBLAS vs hipBLASLt (direct API, zero framework overhead)                  ║\n");
    printf("║  GPU: %-40s  Arch: %-20s  ║\n", prop.name, prop.gcnArchName);
    printf("║  Min runtime per test: %.0fs    Warmup: %d iters                                                  ║\n",
           MIN_BENCH_SECS, WARMUP_ITERS);
    if (active_tensile) {
        printf("║  Tensile lib: %-80s  ║\n", active_tensile);
    } else {
        printf("║  Tensile lib: (default — system rocBLAS)                                                          ║\n");
    }
    printf("╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝\n\n");

    // CSV header to stderr for easy redirection
    fprintf(stderr, "dtype,transA,transB,M,N,K,iters,rocblas_us,hipblaslt_us,rocblas_tflops,hipblaslt_tflops,winner,delta_pct\n");

    // Create handles
    rocblas_handle rb_handle;
    ROCBLAS_CHECK(rocblas_create_handle(&rb_handle));

    hipblasLtHandle_t hl_handle;
    if (hipblasLtCreate(&hl_handle) != HIPBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "Failed to create hipBLASLt handle\n");
        return 1;
    }

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    ROCBLAS_CHECK(rocblas_set_stream(rb_handle, stream));

    // Allocate workspace for hipBLASLt
    void* workspace = nullptr;
    HIP_CHECK(hipMalloc(&workspace, WORKSPACE_SIZE));

    // Iterate over dtypes and problems
    for (int di = 0; di < NUM_DTYPES; di++) {
        const auto& dt = DTYPES[di];
        printf("━━━ %s ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n", dt.name);
        printf("%-5s %-5s %8s %8s %8s  %6s  %12s %12s  %10s %10s  %-10s %s\n",
               "trA", "trB", "M", "N", "K", "iters",
               "rocBLAS(μs)", "hipBLAS(μs)", "rocBLAS TF", "hipBLAS TF", "winner", "Δ%");
        printf("─────────────────────────────────────────────────────────────────────────────────────────────────────\n");

        for (int pi = 0; pi < NUM_PROBLEMS; pi++) {
            const auto& prob = PROBLEMS[pi];

            // Allocate matrices
            int rowA, colA, lda, rowB, colB, ldb;
            get_dims(prob, rowA, colA, lda, rowB, colB, ldb);
            size_t sizeA = (size_t)rowA * colA * dt.elem_bytes_ab;
            size_t sizeB = (size_t)rowB * colB * dt.elem_bytes_ab;
            size_t sizeCD = (size_t)prob.M * prob.N * dt.elem_bytes_cd;

            void *dA, *dB, *dC, *dD;
            HIP_CHECK(hipMalloc(&dA, sizeA));
            HIP_CHECK(hipMalloc(&dB, sizeB));
            HIP_CHECK(hipMalloc(&dC, sizeCD));
            HIP_CHECK(hipMalloc(&dD, sizeCD));
            fill_random(dA, sizeA);
            fill_random(dB, sizeB);
            HIP_CHECK(hipMemset(dC, 0, sizeCD));
            HIP_CHECK(hipMemset(dD, 0, sizeCD));

            // Calibrate iterations
            int iters = calibrate(rb_handle, prob, dt, dA, dB, dC, dD);

            // Benchmark rocBLAS
            double rb_us = bench_rocblas(rb_handle, prob, dt, iters, dA, dB, dC, dD);

            // Benchmark hipBLASLt
            double hl_us = bench_hipblaslt(hl_handle, prob, dt, iters, dA, dB, dC, dD,
                                           workspace, stream);

            // Results
            double rb_tf = tflops(prob.M, prob.N, prob.K, rb_us);
            double hl_tf = (hl_us > 0) ? tflops(prob.M, prob.N, prob.K, hl_us) : 0;

            const char* winner = "N/A";
            double delta = 0;
            if (hl_us > 0) {
                if (rb_us < hl_us) {
                    winner = "rocBLAS";
                    delta = (hl_us - rb_us) / hl_us * 100.0;
                } else {
                    winner = "hipBLASLt";
                    delta = (rb_us - hl_us) / rb_us * 100.0;
                }
            }

            const char* tA = prob.transA ? "T" : "N";
            const char* tB = prob.transB ? "T" : "N";

            if (hl_us > 0) {
                printf("%-5s %-5s %8d %8d %8d  %6d  %12.2f %12.2f  %10.3f %10.3f  %-10s %+.1f%%\n",
                       tA, tB, prob.M, prob.N, prob.K, iters,
                       rb_us, hl_us, rb_tf, hl_tf, winner, delta);
                fprintf(stderr, "%s,%s,%s,%d,%d,%d,%d,%.2f,%.2f,%.3f,%.3f,%s,%.1f\n",
                        dt.name, tA, tB, prob.M, prob.N, prob.K, iters,
                        rb_us, hl_us, rb_tf, hl_tf, winner, delta);
            } else {
                printf("%-5s %-5s %8d %8d %8d  %6d  %12.2f %12s  %10.3f %10s  %-10s\n",
                       tA, tB, prob.M, prob.N, prob.K, iters,
                       rb_us, "N/A", rb_tf, "N/A", "skip");
                fprintf(stderr, "%s,%s,%s,%d,%d,%d,%d,%.2f,-1,%.3f,-1,N/A,0\n",
                        dt.name, tA, tB, prob.M, prob.N, prob.K, iters, rb_us, rb_tf);
            }
            fflush(stdout);
            fflush(stderr);

            HIP_CHECK(hipFree(dA));
            HIP_CHECK(hipFree(dB));
            HIP_CHECK(hipFree(dC));
            HIP_CHECK(hipFree(dD));
        }
        printf("\n");
    }

    // Cleanup
    HIP_CHECK(hipFree(workspace));
    HIP_CHECK(hipStreamDestroy(stream));
    hipblasLtDestroy(hl_handle);
    rocblas_destroy_handle(rb_handle);

    printf("Done. CSV data written to stderr (redirect with 2>results.csv)\n");
    return 0;
}
