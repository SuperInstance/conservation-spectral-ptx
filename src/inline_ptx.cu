// inline_ptx.cu — Inline PTX assembly within CUDA C++
// Uses __device__ functions with asm volatile("...") for warp-level operations
// Compile: nvcc -arch=sm_75 -o inline_ptx inline_ptx.cu -I../include

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>

#include "conservation_types.h"

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

//============================================================================
// Inline PTX Device Functions — Warp Shuffle Intrinsics
//============================================================================

// Warp shuffle: get value from another lane
__device__ __forceinline__ float shfl_sync_float(unsigned mask, float var, int srcLane) {
    float result;
    asm volatile(
        "shfl.sync.b32 %0, %1, %2, %3;"
        : "=f"(result)
        : "f"(var), "r"(srcLane), "r"(mask)
    );
    return result;
}

// Warp shuffle XOR: exchange with lane ^ mask
__device__ __forceinline__ float shfl_xor_sync_float(unsigned mask, float var, int laneMask) {
    float result;
    asm volatile(
        "shfl.sync.b32 %0, %1, %2, %3;"
        : "=f"(result)
        : "f"(var), "r"(laneMask), "r"(mask)
    );
    return result;
}

// Warp-level all-reduce sum using butterfly (XOR) pattern
__device__ __forceinline__ float warpReduceSum(float val) {
    const unsigned FULL_MASK = 0xFFFFFFFF;
    // Butterfly reduction using shfl.xor
    val += shfl_xor_sync_float(FULL_MASK, val, 16);
    val += shfl_xor_sync_float(FULL_MASK, val, 8);
    val += shfl_xor_sync_float(FULL_MASK, val, 4);
    val += shfl_xor_sync_float(FULL_MASK, val, 2);
    val += shfl_xor_sync_float(FULL_MASK, val, 1);
    return val;
}

// Warp-level all-reduce max using butterfly pattern
__device__ __forceinline__ float warpReduceMax(float val) {
    const unsigned FULL_MASK = 0xFFFFFFFF;
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = shfl_xor_sync_float(FULL_MASK, val, offset);
        val = fmaxf(val, other);
    }
    return val;
}

// Warp vote: check if any lane has predicate true
__device__ __forceinline__ unsigned warpAny(int predicate) {
    unsigned result;
    asm volatile(
        "vote.sync.any %0, %1, %2;"
        : "=r"(result)
        : "r"(predicate), "r"(0xFFFFFFFF)
    );
    return result;
}

// Warp ballot: get mask of lanes with true predicate
__device__ __forceinline__ unsigned warpBallot(int predicate) {
    unsigned result;
    asm volatile(
        "vote.sync.ballot %0, %1, %2;"
        : "=r"(result)
        : "r"(predicate), "r"(0xFFFFFFFF)
    );
    return result;
}

// Lane ID via PTX (no threadIdx dependency)
__device__ __forceinline__ unsigned laneId() {
    unsigned id;
    asm volatile("mov.u32 %0, %%laneid;" : "=r"(id));
    return id;
}

// Fast reciprocal via PTX
__device__ __forceinline__ float fastRcp(float x) {
    float result;
    asm volatile("rcp.rn.ftz.f32 %0, %1;" : "=f"(result) : "f"(x));
    return result;
}

// FMA with explicit FTZ (flush-to-zero)
__device__ __forceinline__ float fma_ftz(float a, float b, float c) {
    float result;
    asm volatile("fma.rn.ftz.f32 %0, %1, %2, %3;"
        : "=f"(result) : "f"(a), "f"(b), "f"(c));
    return result;
}

//============================================================================
// CUDA C++ Kernels (using inline PTX for warp ops)
//============================================================================

// Laplacian kernel
__global__ void kernelLaplacianInline(
    const float* __restrict__ adj,
    const float* __restrict__ deg,
    float* __restrict__ lap,
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * n;
    if (tid >= total) return;

    int row = tid / n;
    int col = tid % n;

    float a = adj[tid];
    float d = deg[row];

    // L[i][j] = D[i] if i==j, else -A[i][j]
    lap[tid] = (row == col) ? d : -a;
}

// Power iteration kernel with inline PTX warp reduction
__global__ void kernelPowerIterInline(
    const float* __restrict__ mat,
    const float* __restrict__ vin,
    float* __restrict__ vout,
    float* __restrict__ norm_out,
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread computes one element of mat * vin
    float acc = 0.0f;

    if (tid < n) {
        for (int j = 0; j < n; j++) {
            acc += mat[tid * n + j] * vin[j];
        }
        vout[tid] = acc;
    }

    // Norm reduction via inline PTX warp shuffles
    float contrib = (tid < n) ? acc * acc : 0.0f;

    // Warp-level reduce
    float warp_sum = warpReduceSum(contrib);

    // Lane 0 of each warp atomics to global
    if (laneId() == 0) {
        atomicAdd(norm_out, warp_sum);
    }
}

// Conservation operator kernel with inline PTX
__global__ void kernelConservationInline(
    const float* __restrict__ adj,
    const float* __restrict__ inv_deg,
    const float* __restrict__ vin,
    float* __restrict__ vout,
    float alpha,
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    // Compute (A * v)[tid]
    float av = 0.0f;
    for (int j = 0; j < n; j++) {
        av += adj[tid * n + j] * vin[j];
    }

    // D^{-1} A v
    float dav = av * inv_deg[tid];

    // Conservation: alpha * D^{-1}Av + (1-alpha) * v
    float v = vin[tid];
    float one_minus_alpha = 1.0f - alpha;

    // Use FMA via inline PTX for the final computation
    vout[tid] = fma_ftz(alpha, dav, one_minus_alpha * v);
}

//============================================================================
// Host-side driver
//============================================================================

void cpuComputeLaplacian(const ConservationGraph& graph, float* laplacian) {
    int n = graph.num_nodes;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            laplacian[i * n + j] = (i == j) ? graph.degree[i] : -graph.adjacency[i * n + j];
}

ConservationResult cpuPowerIteration(const float* laplacian, int n,
                                      int max_iters = 100, float tol = 1e-6f) {
    ConservationResult result = {0};
    std::vector<float> v(n, 1.0f / sqrtf((float)n));
    std::vector<float> lv(n);
    float prev = 0;

    for (int iter = 0; iter < max_iters; iter++) {
        for (int i = 0; i < n; i++) {
            float sum = 0;
            for (int j = 0; j < n; j++) sum += laplacian[i * n + j] * v[j];
            lv[i] = sum;
        }
        float norm = 0;
        for (int i = 0; i < n; i++) norm += lv[i] * lv[i];
        norm = sqrtf(norm);
        result.eigenvalue = norm;
        if (norm > 1e-10f)
            for (int i = 0; i < n; i++) v[i] = lv[i] / norm;
        result.residual = fabsf(result.eigenvalue - prev);
        result.iterations = iter + 1;
        if (result.residual < tol && iter > 0) { result.converged = 1; break; }
        prev = result.eigenvalue;
    }
    return result;
}

ConservationResult inlinePTXPowerIteration(const float* d_laplacian, int n,
                                            int max_iters = 100, float tol = 1e-6f) {
    ConservationResult result = {0};
    float *d_vin, *d_vout, *d_norm;

    CUDA_CHECK(cudaMalloc(&d_vin, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_vout, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_norm, sizeof(float)));

    std::vector<float> v_init(n, 1.0f / sqrtf((float)n));
    CUDA_CHECK(cudaMemcpy(d_vin, v_init.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    int blocks = (n + 255) / 256;
    float prev = 0;

    for (int iter = 0; iter < max_iters; iter++) {
        CUDA_CHECK(cudaMemset(d_norm, 0, sizeof(float)));

        kernelPowerIterInline<<<blocks, 256>>>(d_laplacian, d_vin, d_vout, d_norm, n);
        CUDA_CHECK(cudaDeviceSynchronize());

        float norm_sq;
        CUDA_CHECK(cudaMemcpy(&norm_sq, d_norm, sizeof(float), cudaMemcpyDeviceToHost));
        float norm = sqrtf(fabsf(norm_sq));
        result.eigenvalue = norm;

        // Normalize on host, copy back
        std::vector<float> vout(n);
        CUDA_CHECK(cudaMemcpy(vout.data(), d_vout, n * sizeof(float), cudaMemcpyDeviceToHost));
        if (norm > 1e-10f)
            for (int i = 0; i < n; i++) vout[i] /= norm;
        CUDA_CHECK(cudaMemcpy(d_vin, vout.data(), n * sizeof(float), cudaMemcpyHostToDevice));

        result.residual = fabsf(result.eigenvalue - prev);
        result.iterations = iter + 1;
        if (result.residual < tol && iter > 0) { result.converged = 1; break; }
        prev = result.eigenvalue;
    }

    CUDA_CHECK(cudaFree(d_vin));
    CUDA_CHECK(cudaFree(d_vout));
    CUDA_CHECK(cudaFree(d_norm));
    return result;
}

//============================================================================
// Benchmark
//============================================================================

void runBenchmark(const float* cpu_lap, const float* d_lap, int n) {
    printf("\n╔══════════════════════════════════════════════════╗\n");
    printf("║          BENCHMARK: CPU vs CUDA C vs Inline PTX ║\n");
    printf("╚══════════════════════════════════════════════════╝\n\n");

    const int RUNS = 1000;

    // CPU benchmark
    auto t0 = std::chrono::high_resolution_clock::now();
    ConservationResult cpu_r;
    for (int i = 0; i < RUNS; i++) cpu_r = cpuPowerIteration(cpu_lap, n);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / RUNS;

    // Inline PTX benchmark
    auto t2 = std::chrono::high_resolution_clock::now();
    ConservationResult ptx_r;
    for (int i = 0; i < RUNS; i++) ptx_r = inlinePTXPowerIteration(d_lap, n);
    auto t3 = std::chrono::high_resolution_clock::now();
    double ptx_ms = std::chrono::duration<double, std::milli>(t3 - t2).count() / RUNS;

    printf("┌────────────┬──────────────┬─────────┬──────────┐\n");
    printf("│ Method     │ Eigenvalue   │ Iters   │ Time(ms) │\n");
    printf("├────────────┼──────────────┼─────────┼──────────┤\n");
    printf("│ CPU        │ %12.6f │ %7d │ %8.3f │\n", cpu_r.eigenvalue, cpu_r.iterations, cpu_ms);
    printf("│ Inline PTX │ %12.6f │ %7d │ %8.3f │\n", ptx_r.eigenvalue, ptx_r.iterations, ptx_ms);
    printf("└────────────┴──────────────┴─────────┴──────────┘\n");

    printf("\nSpeedup (Inline PTX vs CPU): %.2fx\n", cpu_ms / ptx_ms);
    printf("Eigenvalue match: %s (diff=%.2e)\n",
           fabsf(cpu_r.eigenvalue - ptx_r.eigenvalue) < 1e-3f ? "YES ✓" : "NO ✗",
           fabsf(cpu_r.eigenvalue - ptx_r.eigenvalue));
}

//============================================================================
// Test: 5-node chord progression
//============================================================================

void runChordTest(const float* cpu_lap, const float* d_lap, int n) {
    printf("\n=== Test: 5-Node Chord Progression ===\n\n");

    printf("Adjacency matrix:\n");
    printf("  Nodes: C(0) Am(1) F(2) G(3) C'(4)\n");
    printf("  Edges: C-Am, C-C', Am-F, F-G, G-C', Am-G\n\n");

    printf("Laplacian matrix:\n");
    for (int i = 0; i < n; i++) {
        printf("  [");
        for (int j = 0; j < n; j++) printf("%6.2f", cpu_lap[i*n+j]);
        printf(" ]\n");
    }

    // CPU result
    ConservationResult cpu_r = cpuPowerIteration(cpu_lap, n);
    printf("\nCPU:  eigenvalue=%.6f, iters=%d, residual=%.2e, converged=%s\n",
           cpu_r.eigenvalue, cpu_r.iterations, cpu_r.residual,
           cpu_r.converged ? "YES" : "NO");

    // Inline PTX result
    ConservationResult ptx_r = inlinePTXPowerIteration(d_lap, n);
    printf("PTX:  eigenvalue=%.6f, iters=%d, residual=%.2e, converged=%s\n",
           ptx_r.eigenvalue, ptx_r.iterations, ptx_r.residual,
           ptx_r.converged ? "YES" : "NO");

    bool pass = fabsf(cpu_r.eigenvalue - ptx_r.eigenvalue) < 1e-3f;
    printf("\nTest result: %s\n", pass ? "PASS ✓" : "FAIL ✗");

    // Test conservation operator
    printf("\n--- Conservation Operator Test ---\n");
    float *d_adj, *d_inv_deg, *d_vin, *d_vout;
    float inv_deg[5] = {1.0f/2, 1.0f/3, 1.0f/2, 1.0f/3, 1.0f/2};
    float v_in[5] = {0.447f, 0.447f, 0.447f, 0.447f, 0.447f}; // ~1/sqrt(5)

    float adj[25];
    for (int i = 0; i < 25; i++) adj[i] = 0;
    adj[0*5+1]=1; adj[1*5+0]=1; adj[0*5+4]=1; adj[4*5+0]=1;
    adj[1*5+2]=1; adj[2*5+1]=1; adj[2*5+3]=1; adj[3*5+2]=1;
    adj[3*5+4]=1; adj[4*5+3]=1; adj[1*5+3]=1; adj[3*5+1]=1;

    CUDA_CHECK(cudaMalloc(&d_adj, 25 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_inv_deg, 5 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_vin, 5 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_vout, 5 * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_adj, adj, 25*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_inv_deg, inv_deg, 5*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vin, v_in, 5*sizeof(float), cudaMemcpyHostToDevice));

    kernelConservationInline<<<1, 256>>>(d_adj, d_inv_deg, d_vin, d_vout, 0.85f, 5);
    CUDA_CHECK(cudaDeviceSynchronize());

    float v_out[5];
    CUDA_CHECK(cudaMemcpy(v_out, d_vout, 5*sizeof(float), cudaMemcpyDeviceToHost));

    printf("  alpha=0.85, input=uniform\n");
    printf("  Conservation output: [");
    for (int i = 0; i < 5; i++) printf("%.4f%s", v_out[i], i<4?" ":"");
    printf("]\n");

    // Verify conservation: sum should be preserved (approx)
    float sum_in = 0, sum_out = 0;
    for (int i = 0; i < 5; i++) { sum_in += v_in[i]; sum_out += v_out[i]; }
    printf("  Sum in=%.4f, Sum out=%.4f, conservation error=%.2e\n",
           sum_in, sum_out, fabsf(sum_in - sum_out));

    CUDA_CHECK(cudaFree(d_adj));
    CUDA_CHECK(cudaFree(d_inv_deg));
    CUDA_CHECK(cudaFree(d_vin));
    CUDA_CHECK(cudaFree(d_vout));
}

//============================================================================
// Main
//============================================================================

int main() {
    printf("=== Conservation Spectral SDK — Inline PTX ===\n\n");

    // 5-node chord progression
    ConservationGraph graph;
    memset(&graph, 0, sizeof(graph));
    graph.num_nodes = 5;
    graph.conservation_parameter = 0.85f;

    auto setEdge = [&](int i, int j) {
        graph.adjacency[i*5+j] = 1.0f;
        graph.adjacency[j*5+i] = 1.0f;
    };
    setEdge(0,1); setEdge(0,4); setEdge(1,2);
    setEdge(2,3); setEdge(3,4); setEdge(1,3);

    for (int i = 0; i < 5; i++) {
        graph.degree[i] = 0;
        for (int j = 0; j < 5; j++) graph.degree[i] += graph.adjacency[i*5+j];
    }

    // CPU laplacian
    float cpu_lap[25];
    cpuComputeLaplacian(graph, cpu_lap);

    // GPU allocations
    float *d_adj, *d_deg, *d_lap;
    CUDA_CHECK(cudaMalloc(&d_adj, 25 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_deg, 5 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_lap, 25 * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_adj, graph.adjacency, 25*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_deg, graph.degree, 5*sizeof(float), cudaMemcpyHostToDevice));

    // Compute laplacian on GPU
    kernelLaplacianInline<<<1, 256>>>(d_adj, d_deg, d_lap, 5);
    CUDA_CHECK(cudaDeviceSynchronize());

    float gpu_lap[25];
    CUDA_CHECK(cudaMemcpy(gpu_lap, d_lap, 25*sizeof(float), cudaMemcpyDeviceToHost));

    // Verify
    bool match = true;
    for (int i = 0; i < 25; i++)
        if (fabsf(cpu_lap[i] - gpu_lap[i]) > 1e-4f) match = false;
    printf("Laplacian CPU/GPU match: %s\n\n", match ? "YES ✓" : "NO ✗");

    // Run test
    runChordTest(cpu_lap, d_lap, 5);

    // Run benchmark
    runBenchmark(cpu_lap, d_lap, 5);

    CUDA_CHECK(cudaFree(d_adj));
    CUDA_CHECK(cudaFree(d_deg));
    CUDA_CHECK(cudaFree(d_lap));

    printf("\nDone.\n");
    return 0;
}
