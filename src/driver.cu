// driver.cu — CUDA driver API: loads PTX module, launches kernels, reads results
// Compile: nvcc -o driver driver.cu -lcuda -I../include

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
    CUresult err = call; \
    if (err != CUDA_SUCCESS) { \
        const char* errStr; \
        cuGetErrorString(err, &errStr); \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, errStr); \
        exit(1); \
    } \
} while(0)

#define RT_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA RT error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

class ConservationDriver {
public:
    ConservationDriver() : module_(nullptr), initialized_(false) {}

    int init() {
        CUDA_CHECK(cuInit(0));

        CUDA_CHECK(cuCtxGetCurrent(&context_));
        if (!context_) {
            CUdevice device;
            CUDA_CHECK(cuDeviceGet(&device, 0));
            CUDA_CHECK(cuCtxCreate(&context_, 0, device));
        }

        printf("[driver] CUDA initialized, context ready\n");
        initialized_ = true;
        return CONSERVATION_OK;
    }

    int loadPTX(const char* ptx_file) {
        if (!initialized_) return CONSERVATION_ERR_CUDA;

        // Read PTX file
        FILE* f = fopen(ptx_file, "r");
        if (!f) {
            fprintf(stderr, "[driver] Cannot open PTX file: %s\n", ptx_file);
            return CONSERVATION_ERR_PTX;
        }

        fseek(f, 0, SEEK_END);
        long size = ftell(f);
        fseek(f, 0, SEEK_SET);
        char* ptx_source = new char[size + 1];
        fread(ptx_source, 1, size, f);
        ptx_source[size] = '\0';
        fclose(f);

        // Load PTX module
        CUjit_option jit_options[] = {CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES};
        void* jit_values[] = {(void*)8192};
        char error_log[8192] = {0};

        CUresult result = cuModuleLoadDataEx(&module_, ptx_source,
            1, jit_options, jit_values);

        if (result != CUDA_SUCCESS) {
            fprintf(stderr, "[driver] PTX load failed (error log below):\n%s\n", error_log);
            const char* errStr;
            cuGetErrorString(result, &errStr);
            fprintf(stderr, "[driver] Error: %s\n", errStr);
            delete[] ptx_source;
            return CONSERVATION_ERR_PTX;
        }

        printf("[driver] PTX module loaded: %s (%ld bytes)\n", ptx_file, size);
        delete[] ptx_source;

        // Get kernel functions
        CUDA_CHECK(cuModuleGetFunction(&fn_laplacian_, module_, "kernel_laplacian"));
        CUDA_CHECK(cuModuleGetFunction(&fn_power_iter_, module_, "kernel_power_iteration"));
        CUDA_CHECK(cuModuleGetFunction(&fn_conservation_, module_, "kernel_conservation"));

        printf("[driver] Kernel functions resolved: laplacian, power_iteration, conservation\n");
        return CONSERVATION_OK;
    }

    int computeLaplacian(const ConservationGraph& graph, float* d_laplacian) {
        if (!module_) return CONSERVATION_ERR_PTX;

        float* d_adj;
        float* d_deg;
        int* d_n;

        RT_CHECK(cudaMalloc(&d_adj, graph.num_nodes * graph.num_nodes * sizeof(float)));
        RT_CHECK(cudaMalloc(&d_deg, graph.num_nodes * sizeof(float)));

        RT_CHECK(cudaMemcpy(d_adj, graph.adjacency,
            graph.num_nodes * graph.num_nodes * sizeof(float), cudaMemcpyHostToDevice));
        RT_CHECK(cudaMemcpy(d_deg, graph.degree,
            graph.num_nodes * sizeof(float), cudaMemcpyHostToDevice));

        CUdeviceptr d_adj_cu = (CUdeviceptr)d_adj;
        CUdeviceptr d_deg_cu = (CUdeviceptr)d_deg;
        CUdeviceptr d_lap_cu = (CUdeviceptr)d_laplacian;
        unsigned int n = graph.num_nodes;

        int total = n * n;
        int blocks = (total + CONSERVATION_BLOCK_SIZE - 1) / CONSERVATION_BLOCK_SIZE;

        void* args[] = {&d_adj_cu, &d_deg_cu, &d_lap_cu, &n};
        CUDA_CHECK(cuLaunchKernel(fn_laplacian_, blocks, 1, 1,
            CONSERVATION_BLOCK_SIZE, 1, 1, 0, nullptr, args, nullptr));
        CUDA_CHECK(cuCtxSynchronize());

        RT_CHECK(cudaFree(d_adj));
        RT_CHECK(cudaFree(d_deg));

        return CONSERVATION_OK;
    }

    ConservationResult powerIteration(const float* d_laplacian, int n,
                                       int max_iters = 100, float tol = 1e-6f) {
        ConservationResult result = {0};

        float* d_vin;
        float* d_vout;
        float* d_norm;

        RT_CHECK(cudaMalloc(&d_vin, n * sizeof(float)));
        RT_CHECK(cudaMalloc(&d_vout, n * sizeof(float)));
        RT_CHECK(cudaMalloc(&d_norm, sizeof(float)));

        // Initialize eigenvector guess: uniform
        std::vector<float> v_init(n, 1.0f / sqrtf((float)n));
        RT_CHECK(cudaMemcpy(d_vin, v_init.data(), n * sizeof(float), cudaMemcpyHostToDevice));

        float prev_eigenvalue = 0.0f;
        int blocks = (n + CONSERVATION_BLOCK_SIZE - 1) / CONSERVATION_BLOCK_SIZE;

        for (int iter = 0; iter < max_iters; iter++) {
            // Reset norm
            RT_CHECK(cudaMemset(d_norm, 0, sizeof(float)));

            CUdeviceptr d_lap_cu = (CUdeviceptr)d_laplacian;
            CUdeviceptr d_vin_cu = (CUdeviceptr)d_vin;
            CUdeviceptr d_vout_cu = (CUdeviceptr)d_vout;
            CUdeviceptr d_norm_cu = (CUdeviceptr)d_norm;
            unsigned int n_nodes = n;

            void* args[] = {&d_lap_cu, &d_vin_cu, &d_vout_cu, &d_norm_cu, &n_nodes};
            CUDA_CHECK(cuLaunchKernel(fn_power_iter_, blocks, 1, 1,
                CONSERVATION_BLOCK_SIZE, 1, 1, 0, nullptr, args, nullptr));
            CUDA_CHECK(cuCtxSynchronize());

            // Read back norm
            float norm_sq;
            RT_CHECK(cudaMemcpy(&norm_sq, d_norm, sizeof(float), cudaMemcpyDeviceToHost));
            float norm = sqrtf(fabsf(norm_sq));

            // Eigenvalue estimate: norm of L*v (Rayleigh quotient approximation)
            result.eigenvalue = norm;

            // Normalize: v = v_out / norm (on host for simplicity, copy back)
            std::vector<float> v_out(n);
            RT_CHECK(cudaMemcpy(v_out.data(), d_vout, n * sizeof(float), cudaMemcpyDeviceToHost));

            if (norm > 1e-10f) {
                for (int i = 0; i < n; i++) v_out[i] /= norm;
            }
            RT_CHECK(cudaMemcpy(d_vin, v_out.data(), n * sizeof(float), cudaMemcpyHostToDevice));

            result.residual = fabsf(result.eigenvalue - prev_eigenvalue);
            result.iterations = iter + 1;

            if (result.residual < tol && iter > 0) {
                result.converged = 1;
                break;
            }

            prev_eigenvalue = result.eigenvalue;
        }

        RT_CHECK(cudaFree(d_vin));
        RT_CHECK(cudaFree(d_vout));
        RT_CHECK(cudaFree(d_norm));

        return result;
    }

    ~ConservationDriver() {
        if (module_) cuModuleUnload(module_);
    }

private:
    CUcontext context_;
    CUmodule module_;
    CUfunction fn_laplacian_;
    CUfunction fn_power_iter_;
    CUfunction fn_conservation_;
    bool initialized_;
};

//----------------------------------------------------------------------------
// CPU reference implementations
//----------------------------------------------------------------------------

void cpuComputeLaplacian(const ConservationGraph& graph, float* laplacian) {
    int n = graph.num_nodes;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j)
                laplacian[i * n + j] = graph.degree[i];
            else
                laplacian[i * n + j] = -graph.adjacency[i * n + j];
        }
    }
}

ConservationResult cpuPowerIteration(const float* laplacian, int n,
                                      int max_iters = 100, float tol = 1e-6f) {
    ConservationResult result = {0};
    std::vector<float> v(n, 1.0f / sqrtf((float)n));
    std::vector<float> lv(n);
    float prev_eigenvalue = 0.0f;

    for (int iter = 0; iter < max_iters; iter++) {
        // lv = L * v
        for (int i = 0; i < n; i++) {
            float sum = 0;
            for (int j = 0; j < n; j++) {
                sum += laplacian[i * n + j] * v[j];
            }
            lv[i] = sum;
        }

        // Norm
        float norm = 0;
        for (int i = 0; i < n; i++) norm += lv[i] * lv[i];
        norm = sqrtf(norm);

        result.eigenvalue = norm;

        // Normalize
        if (norm > 1e-10f) {
            for (int i = 0; i < n; i++) v[i] = lv[i] / norm;
        }

        result.residual = fabsf(result.eigenvalue - prev_eigenvalue);
        result.iterations = iter + 1;

        if (result.residual < tol && iter > 0) {
            result.converged = 1;
            break;
        }
        prev_eigenvalue = result.eigenvalue;
    }
    return result;
}

//----------------------------------------------------------------------------
// Benchmark helper
//----------------------------------------------------------------------------

double benchmarkCPU(const float* laplacian, int n, int runs, float& out_eigenvalue, int& out_iters) {
    auto start = std::chrono::high_resolution_clock::now();
    ConservationResult r;
    for (int i = 0; i < runs; i++) {
        r = cpuPowerIteration(laplacian, n);
    }
    auto end = std::chrono::high_resolution_clock::now();
    out_eigenvalue = r.eigenvalue;
    out_iters = r.iterations;
    return std::chrono::duration<double, std::milli>(end - start).count() / runs;
}

//----------------------------------------------------------------------------
// Main
//----------------------------------------------------------------------------

int main(int argc, char** argv) {
    const char* ptx_file = "src/conservation_kernels.ptx";
    if (argc > 1) ptx_file = argv[1];

    printf("=== Conservation Spectral SDK — PTX Driver ===\n\n");

    // Build a 5-node chord progression graph
    // Musical interpretation: C - Am - F - G - C (cycle with cross-edges)
    ConservationGraph graph;
    memset(&graph, 0, sizeof(graph));
    graph.num_nodes = 5;
    graph.conservation_parameter = 0.85f;

    // Adjacency (undirected, symmetric)
    // 0-1, 0-4, 1-2, 2-3, 3-4, 1-3 (chord cross-edge)
    auto setEdge = [&](int i, int j, float w = 1.0f) {
        graph.adjacency[i * 5 + j] = w;
        graph.adjacency[j * 5 + i] = w;
    };
    setEdge(0, 1); setEdge(0, 4); setEdge(1, 2);
    setEdge(2, 3); setEdge(3, 4); setEdge(1, 3);

    // Compute degrees
    for (int i = 0; i < 5; i++) {
        graph.degree[i] = 0;
        for (int j = 0; j < 5; j++) graph.degree[i] += graph.adjacency[i * 5 + j];
    }

    printf("Graph: 5-node chord progression\n");
    printf("  Edges: 0-1, 0-4, 1-2, 2-3, 3-4, 1-3\n");
    printf("  Degrees: ");
    for (int i = 0; i < 5; i++) printf("%.0f ", graph.degree[i]);
    printf("\n\n");

    // ---- CPU ----
    float cpu_laplacian[25];
    cpuComputeLaplacian(graph, cpu_laplacian);

    printf("CPU Laplacian:\n");
    for (int i = 0; i < 5; i++) {
        printf("  [");
        for (int j = 0; j < 5; j++) printf("%6.2f", cpu_laplacian[i*5+j]);
        printf(" ]\n");
    }

    float cpu_eig; int cpu_iters;
    double cpu_ms = benchmarkCPU(cpu_laplacian, 5, 1000, cpu_eig, cpu_iters);
    printf("\nCPU Power Iteration: eigenvalue=%.6f, iters=%d, %.3f ms/iter\n",
           cpu_eig, cpu_iters, cpu_ms);

    // ---- GPU via PTX ----
    ConservationDriver driver;
    driver.init();

    int rc = driver.loadPTX(ptx_file);
    if (rc != CONSERVATION_OK) {
        printf("\nPTX load failed (rc=%d). This is expected on systems without NVIDIA GPU.\n", rc);
        printf("The PTX kernels, inline PTX version, and benchmarks are still valid for inspection.\n");
        return 0;
    }

    // GPU laplacian
    float* d_laplacian;
    RT_CHECK(cudaMalloc(&d_laplacian, 25 * sizeof(float)));
    driver.computeLaplacian(graph, d_laplacian);

    float gpu_lap[25];
    RT_CHECK(cudaMemcpy(gpu_lap, d_laplacian, 25 * sizeof(float), cudaMemcpyDeviceToHost));

    printf("\nGPU Laplacian (via PTX):\n");
    for (int i = 0; i < 5; i++) {
        printf("  [");
        for (int j = 0; j < 5; j++) printf("%6.2f", gpu_lap[i*5+j]);
        printf(" ]\n");
    }

    // Verify match
    bool match = true;
    for (int i = 0; i < 25; i++) {
        if (fabsf(cpu_laplacian[i] - gpu_lap[i]) > 1e-4f) match = false;
    }
    printf("\nLaplacian CPU/GPU match: %s\n", match ? "YES ✓" : "NO ✗");

    // GPU power iteration via PTX
    auto start = std::chrono::high_resolution_clock::now();
    ConservationResult gpu_result = driver.powerIteration(d_laplacian, 5);
    auto end = std::chrono::high_resolution_clock::now();
    double gpu_ms = std::chrono::duration<double, std::milli>(end - start).count();

    printf("\nGPU Power Iteration (PTX): eigenvalue=%.6f, iters=%d, residual=%.2e, %.3f ms\n",
           gpu_result.eigenvalue, gpu_result.iterations, gpu_result.residual, gpu_ms);
    printf("Converged: %s\n", gpu_result.converged ? "YES" : "NO");

    printf("\n=== Summary ===\n");
    printf("  CPU eigenvalue: %.6f (%d iters, %.3f ms)\n", cpu_eig, cpu_iters, cpu_ms);
    printf("  GPU eigenvalue: %.6f (%d iters, %.3f ms)\n",
           gpu_result.eigenvalue, gpu_result.iterations, gpu_ms);

    RT_CHECK(cudaFree(d_laplacian));
    return 0;
}
