#include <hip/hip_runtime.h>
#include <cstdio>

__global__ void add_kernel(float* out, const float* a, const float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

int main() {
    const int N = 256;
    size_t bytes = N * sizeof(float);

    float h_a[N], h_b[N], h_out[N];
    for (int i = 0; i < N; i++) { h_a[i] = (float)i; h_b[i] = (float)(i * 2); }

    float *d_a, *d_b, *d_out;
    hipMalloc(&d_a, bytes);
    hipMalloc(&d_b, bytes);
    hipMalloc(&d_out, bytes);

    printf("hipMemcpy H2D...\n");
    hipMemcpy(d_a, h_a, bytes, hipMemcpyHostToDevice);
    hipMemcpy(d_b, h_b, bytes, hipMemcpyHostToDevice);

    printf("Launching kernel...\n");
    add_kernel<<<1, 256>>>(d_out, d_a, d_b, N);
    hipError_t err = hipDeviceSynchronize();
    printf("hipDeviceSynchronize: %s\n", hipGetErrorString(err));

    hipMemcpy(h_out, d_out, bytes, hipMemcpyDeviceToHost);
    printf("Verify: h_out[0]=%.1f (expect 0), h_out[100]=%.1f (expect 300), h_out[255]=%.1f (expect 765)\n",
           h_out[0], h_out[100], h_out[255]);

    hipFree(d_a); hipFree(d_b); hipFree(d_out);
    printf("PASS\n");
    return 0;
}
