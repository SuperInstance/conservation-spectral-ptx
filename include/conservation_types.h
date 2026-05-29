#pragma once
#include <cstdint>

// Conservation Spectral SDK - Core Types
// PTX-native kernel definitions for GPU-accelerated spectral graph theory

static const int CONSERVATION_MAX_NODES = 256;
static const int CONSERVATION_WARP_SIZE = 32;
static const int CONSERVATION_BLOCK_SIZE = 256;

// Error codes
#define CONSERVATION_OK           0
#define CONSERVATION_ERR_CUDA    -1
#define CONSERVATION_ERR_PTX     -2
#define CONSERVATION_ERR_MEMORY  -3
#define CONSERVATION_ERR_LAUNCH  -4

// Kernel result structure - matches PTX layout expectations
struct ConservationResult {
    float eigenvalue;       // Dominant eigenvalue
    int iterations;         // Power iteration count
    float residual;         // Final residual norm
    int converged;          // 1 if converged, 0 otherwise
};

// Graph structure for GPU
struct ConservationGraph {
    float adjacency[CONSERVATION_MAX_NODES * CONSERVATION_MAX_NODES];
    float degree[CONSERVATION_MAX_NODES];
    float laplacian[CONSERVATION_MAX_NODES * CONSERVATION_MAX_NODES];
    int num_nodes;
    float conservation_parameter;  // alpha in conservation law
};

// Benchmark result
struct BenchmarkResult {
    double cpu_ms;
    double cuda_ms;
    double ptx_ms;
    float cpu_eigenvalue;
    float cuda_eigenvalue;
    float ptx_eigenvalue;
    int cpu_iters;
    int cuda_iters;
    int ptx_iters;
};
