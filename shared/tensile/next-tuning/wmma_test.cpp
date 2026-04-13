#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Standalone WMMA test using inline asm for gfx1201
// v_wmma_f32_16x16x16_f16 needs contiguous VGPR ranges:
//   D[v0:v7], A[v8:v11], B[v12:v15], C[v0:v7]
// We use a flat array + explicit register allocation via asm

__global__ void wmma_ones_kernel(float* D_out) {
    unsigned lane = __lane_id();

    // We'll use a struct to get contiguous VGPR allocation
    // A = 4 VGPRs of packed f16 (1.0h, 1.0h) = 0x3C003C00
    // B = 4 VGPRs of packed f16 (1.0h, 1.0h) = 0x3C003C00
    // D = 8 VGPRs of f32 (zero-init, accumulator)

    float result[8];

    // Use raw asm with explicit VGPR numbering via clobber-based approach
    // Initialize all A, B in registers, D=0, then run WMMA
    asm volatile(
        // Load A regs (v8-v11) with packed 1.0h
        "v_mov_b32 v8, 0x3C003C00\n"
        "v_mov_b32 v9, 0x3C003C00\n"
        "v_mov_b32 v10, 0x3C003C00\n"
        "v_mov_b32 v11, 0x3C003C00\n"
        // Load B regs (v12-v15) with packed 1.0h
        "v_mov_b32 v12, 0x3C003C00\n"
        "v_mov_b32 v13, 0x3C003C00\n"
        "v_mov_b32 v14, 0x3C003C00\n"
        "v_mov_b32 v15, 0x3C003C00\n"
        // Zero D regs (v0-v7) - accumulator
        "v_mov_b32 v0, 0\n"
        "v_mov_b32 v1, 0\n"
        "v_mov_b32 v2, 0\n"
        "v_mov_b32 v3, 0\n"
        "v_mov_b32 v4, 0\n"
        "v_mov_b32 v5, 0\n"
        "v_mov_b32 v6, 0\n"
        "v_mov_b32 v7, 0\n"
        // Execute WMMA: D[0:7] = A[8:11] * B[12:15] + C[0:7]
        "v_wmma_f32_16x16x16_f16 v[0:7], v[8:11], v[12:15], v[0:7]\n"
        // Copy results out
        "v_mov_b32 %0, v0\n"
        "v_mov_b32 %1, v1\n"
        "v_mov_b32 %2, v2\n"
        "v_mov_b32 %3, v3\n"
        "v_mov_b32 %4, v4\n"
        "v_mov_b32 %5, v5\n"
        "v_mov_b32 %6, v6\n"
        "v_mov_b32 %7, v7\n"
        : "=v"(result[0]), "=v"(result[1]), "=v"(result[2]), "=v"(result[3]),
          "=v"(result[4]), "=v"(result[5]), "=v"(result[6]), "=v"(result[7])
        :
        : "v0","v1","v2","v3","v4","v5","v6","v7",
          "v8","v9","v10","v11","v12","v13","v14","v15"
    );

    for (int i = 0; i < 8; i++)
        D_out[lane * 8 + i] = result[i];
}

int main() {
    float *d_D, *h_D;
    int total = 32 * 8;
    h_D = (float*)calloc(total, sizeof(float));
    hipMalloc(&d_D, total * sizeof(float));
    hipMemset(d_D, 0, total * sizeof(float));

    printf("Launching WMMA inline asm test on gfx1201...\n");
    printf("Instruction: v_wmma_f32_16x16x16_f16 v[0:7], v[8:11], v[12:15], v[0:7]\n");
    printf("A = all 1.0h (0x3C003C00), B = all 1.0h, C = all 0.0f\n\n");

    wmma_ones_kernel<<<1, 32>>>(d_D);
    hipDeviceSynchronize();

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        printf("HIP error: %s\n", hipGetErrorString(err));
        free(h_D); hipFree(d_D);
        return 1;
    }

    hipMemcpy(h_D, d_D, total * sizeof(float), hipMemcpyDeviceToHost);

    printf("D-matrix output (32 lanes x 8 VGPRs):\n");
    int nz = 0;
    for (int l = 0; l < 32; l++) {
        for (int i = 0; i < 8; i++) {
            float v = h_D[l * 8 + i];
            if (v != 0.0f) nz++;
        }
        if (l < 8 || l == 16 || l == 31) {
            printf("  lane %2d:", l);
            for (int i = 0; i < 8; i++)
                printf(" %8.2f", h_D[l * 8 + i]);
            printf("\n");
        }
    }

    printf("\nNon-zero elements: %d / %d\n", nz, total);
    if (nz == 0)
        printf("FAIL: All zeros - WMMA instruction produced no output!\n");
    else
        printf("PASS: WMMA instruction produced non-zero results.\n");

    // Expected: For 16x16x16 with all 1.0h, each D element should be 16.0
    // (dot product of 16 ones * ones = 16.0)
    printf("\nExpected value per element: 16.0 (dot16 of all 1.0h)\n");

    free(h_D); hipFree(d_D);
    return 0;
}
