# Conservation Spectral SDK — PTX-Native Kernels

PTX-native GPU kernels for spectral graph theory with conservation laws. Hand-written PTX assembly, CUDA driver API loading, inline PTX intrinsics, and benchmarks.

## Architecture

```
conservation-spectral-ptx/
├── include/
│   └── conservation_types.h      # Core types, constants, structures
├── src/
│   ├── conservation_kernels.ptx  # Hand-written PTX assembly kernels
│   ├── driver.cu                 # CUDA driver API: loads PTX, launches, benchmarks
│   └── inline_ptx.cu             # CUDA C++ with inline PTX asm for warp ops
├── tests/
│   └── test_chord.cu             # 5-node chord progression test suite
├── docs/
│   └── PTX_NOTES.md              # PTX programming notes
├── Makefile
└── README.md
```

## Kernels

### 1. Laplacian (`kernel_laplacian`)
Computes L = D - A for an adjacency matrix. Each thread handles one element.

### 2. Power Iteration (`kernel_power_iteration`)
One step of power iteration: v_out = Lv / ‖Lv‖ with warp shuffle reduction for norm computation.

### 3. Conservation (`kernel_conservation`)
Conservation spectral operator: C(v) = α·D⁻¹Av + (1-α)·v with FMA optimization.

## Register Allocation Strategy (PTX)

| Registers    | Purpose                            |
|-------------|-------------------------------------|
| R0-R7       | Loop counters, indices, addresses   |
| R8-R15      | Accumulators (float)                |
| R16-R23     | Temporary computation               |
| R24-R31     | Warp shuffle operands, predicates   |

## Warp Shuffle Reductions

Butterfly (XOR) pattern for warp-level all-reduce:
```
val += shfl.xor(val, 16)
val += shfl.xor(val, 8)
val += shfl.xor(val, 4)
val += shfl.xor(val, 2)
val += shfl.xor(val, 1)
```
5 steps, O(log₂ 32) = O(5), no shared memory needed.

## Build

```bash
make all          # Build everything
make driver       # Build PTX driver
make inline       # Build inline PTX version
make test         # Build and run tests
make bench        # Run benchmarks
```

Requires: nvcc 11.5+, CUDA toolkit, NVIDIA GPU (sm_75+)

## Test: 5-Node Chord Progression

Musical interpretation: **C → Am → F → G → C'** with cross-edge Am→G

```
  0 (C) ──── 1 (Am) ──── 2 (F)
  │           │ ╲           │
  │           │   ╲         │
  4 (C') ─── 3 (G) ────────┘
```

Edges: 0-1, 0-4, 1-2, 2-3, 3-4, 1-3

The chord progression graph tests:
- Laplacian computation correctness
- Power iteration convergence
- Conservation operator mass preservation
- CPU/GPU eigenvalue agreement

## Benchmark

CPU vs CUDA C vs Inline PTX comparison measuring:
- Eigenvalue accuracy
- Iteration count
- Wall-clock time per iteration

## License

MIT
