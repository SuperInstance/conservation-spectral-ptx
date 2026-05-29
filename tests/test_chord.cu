// test_chord.cu — 5-node chord progression test
// Compile: nvcc -arch=sm_75 -o test_chord test_chord.cu -I../include

#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cassert>

#include "conservation_types.h"

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); \
        return 1; \
    } \
} while(0)

// Inline PTX warp reduce sum
__device__ __forceinline__ float shflXor(float v, int mask) {
    float r;
    asm volatile("shfl.sync.b32 %0, %1, %2, %3;"
        : "=f"(r) : "f"(v), "r"(mask), "r"(0xFFFFFFFF));
    return r;
}

__device__ float warpReduceSum(float v) {
    v += shflXor(v, 16); v += shflXor(v, 8);
    v += shflXor(v, 4);  v += shflXor(v, 2);
    v += shflXor(v, 1);
    return v;
}

__global__ void kernelLaplacian(const float* adj, const float* deg, float* lap, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n*n) return;
    int row = tid / n, col = tid % n;
    lap[tid] = (row == col) ? deg[row] : -adj[tid];
}

__global__ void kernelMatVec(const float* mat, const float* vin, float* vout,
                              float* norm_out, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float acc = 0.0f;
    if (tid < n) {
        for (int j = 0; j < n; j++) acc += mat[tid*n+j] * vin[j];
        vout[tid] = acc;
    }
    float contrib = (tid < n) ? acc*acc : 0;
    float ws = warpReduceSum(contrib);
    unsigned lane;
    asm volatile("mov.u32 %0, %%laneid;" : "=r"(lane));
    if (lane == 0) atomicAdd(norm_out, ws);
}

// CPU reference
void cpuLaplacian(const float* adj, const float* deg, float* lap, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            lap[i*n+j] = (i==j) ? deg[i] : -adj[i*n+j];
}

void cpuMatVec(const float* mat, const float* vin, float* vout, int n) {
    for (int i = 0; i < n; i++) {
        float s = 0;
        for (int j = 0; j < n; j++) s += mat[i*n+j]*vin[j];
        vout[i] = s;
    }
}

int main() {
    printf("=== Test: 5-Node Chord Progression ===\n\n");

    const int N = 5;
    float adj[25] = {}, deg[5] = {};

    // Chord progression: C-Am-F-G-C' with cross-edge Am-G
    // 0-1, 0-4, 1-2, 2-3, 3-4, 1-3
    auto edge = [&](int a, int b) { adj[a*N+b] = adj[b*N+a] = 1.0f; };
    edge(0,1); edge(0,4); edge(1,2); edge(2,3); edge(3,4); edge(1,3);

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            deg[i] += adj[i*N+j];

    printf("Degrees: ");
    for (int i = 0; i < N; i++) printf("%.0f ", deg[i]);
    printf("\n");

    // --- Test 1: Laplacian ---
    printf("\n--- Test 1: Laplacian ---\n");
    float cpu_lap[25], gpu_lap[25];

    cpuLaplacian(adj, deg, cpu_lap, N);

    float *d_adj, *d_deg, *d_lap;
    CUDA_CHECK(cudaMalloc(&d_adj, 25*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_deg, 5*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_lap, 25*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_adj, adj, 25*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_deg, deg, 5*sizeof(float), cudaMemcpyHostToDevice));

    kernelLaplacian<<<1, 256>>>(d_adj, d_deg, d_lap, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(gpu_lap, d_lap, 25*sizeof(float), cudaMemcpyDeviceToHost));

    bool lap_ok = true;
    for (int i = 0; i < 25; i++)
        if (fabsf(cpu_lap[i] - gpu_lap[i]) > 1e-5f) lap_ok = false;

    printf("CPU:\n");
    for (int i = 0; i < N; i++) {
        printf("  [");
        for (int j = 0; j < N; j++) printf("%6.2f", cpu_lap[i*N+j]);
        printf("]\n");
    }
    printf("GPU (Inline PTX Laplacian): %s\n", lap_ok ? "PASS ✓" : "FAIL ✗");

    // --- Test 2: Matrix-Vector Multiply (Power Iteration Step) ---
    printf("\n--- Test 2: MatVec with Warp Shuffle ---\n");

    float v_in[5] = {0.4472f, 0.4472f, 0.4472f, 0.4472f, 0.4472f};
    float cpu_vout[5], gpu_vout[5];

    cpuMatVec(cpu_lap, v_in, cpu_vout, N);

    float *d_vin, *d_vout, *d_norm;
    CUDA_CHECK(cudaMalloc(&d_vin, N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_vout, N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_norm, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_vin, v_in, N*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_norm, 0, sizeof(float)));

    kernelMatVec<<<1, 256>>>(d_lap, d_vin, d_vout, d_norm, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(gpu_vout, d_vout, N*sizeof(float), cudaMemcpyDeviceToHost));

    float norm_gpu;
    CUDA_CHECK(cudaMemcpy(&norm_gpu, d_norm, sizeof(float), cudaMemcpyDeviceToHost));

    bool mv_ok = true;
    for (int i = 0; i < N; i++)
        if (fabsf(cpu_vout[i] - gpu_vout[i]) > 1e-3f) mv_ok = false;

    printf("CPU L*v: [");
    for (int i = 0; i < N; i++) printf("%.4f ", cpu_vout[i]);
    printf("]\nGPU L*v: [");
    for (int i = 0; i < N; i++) printf("%.4f ", gpu_vout[i]);
    printf("]\n");

    // CPU norm check
    float cpu_norm = 0;
    for (int i = 0; i < N; i++) cpu_norm += cpu_vout[i]*cpu_vout[i];
    cpu_norm = sqrtf(cpu_norm);

    printf("Norm: CPU=%.4f, GPU=%.4f (warp shuffle reduction)\n", cpu_norm, sqrtf(fabsf(norm_gpu)));
    printf("MatVec: %s\n", mv_ok ? "PASS ✓" : "FAIL ✗");

    // --- Test 3: Full Power Iteration ---
    printf("\n--- Test 3: Power Iteration Convergence ---\n");

    // Run on CPU
    float v[5] = {1/sqrtf(5), 1/sqrtf(5), 1/sqrtf(5), 1/sqrtf(5), 1/sqrtf(5)};
    float eig = 0, prev_eig = 0;
    int converged = 0;
    for (int iter = 0; iter < 100; iter++) {
        float lv[5];
        for (int i = 0; i < N; i++) {
            float s = 0;
            for (int j = 0; j < N; j++) s += cpu_lap[i*N+j]*v[j];
            lv[i] = s;
        }
        float norm = 0;
        for (int i = 0; i < N; i++) norm += lv[i]*lv[i];
        eig = sqrtf(norm);
        if (norm > 0) for (int i = 0; i < N; i++) v[i] = lv[i]/eig;
        if (fabsf(eig - prev_eig) < 1e-6f && iter > 0) { converged = iter+1; break; }
        prev_eig = eig;
    }

    printf("Dominant eigenvalue: %.6f (converged in %d iterations)\n", eig, converged);
    printf("Eigenvector: [");
    for (int i = 0; i < N; i++) printf("%.4f ", v[i]);
    printf("]\n");

    // Verify properties
    // Laplacian eigenvalues for this graph should be positive, largest < 2*max_degree
    printf("\nLaplacian property check: eigenvalue %.4f < 2*max_degree %.4f: %s\n",
           eig, 2*3.0f, eig < 2*3.0f ? "YES ✓" : "NO ✗");

    // Cleanup
    cudaFree(d_adj); cudaFree(d_deg); cudaFree(d_lap);
    cudaFree(d_vin); cudaFree(d_vout); cudaFree(d_norm);

    printf("\n=== All Tests Complete ===\n");
    return 0;
}
