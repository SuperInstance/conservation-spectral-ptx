# PTX Programming Notes

## Target Architecture

- **PTX ISA version:** 7.5 (matches CUDA 11.5)
- **Target:** sm_75 (NVIDIA Turing)
- **Address size:** 64-bit

## Key PTX Instructions Used

### Warp Shuffle
```
shfl.sync.b32  %dst, %src, %lane, %membermask;
```
- `.sync` ensures all lanes participate
- `membermask` = 0xFFFFFFFF for full warp
- `lane` can be absolute (idx) or relative via XOR

### Warp Vote
```
vote.sync.any    %dst, %pred, %mask;
vote.sync.ballot %dst, %pred, %mask;
```

### Special Registers
```
mov.u32 %r, %laneid;   // Lane within warp (0-31)
mov.u32 %r, %tid.x;    // Thread index in block
mov.u32 %r, %ntid.x;   // Block dimension
mov.u32 %r, %ctaid.x;  // Block index
```

### Fast Math
```
rcp.rn.ftz.f32  %dst, %src;     // Fast reciprocal
fma.rn.ftz.f32  %dst, %a, %b, %c;  // Fused multiply-add
```

## Register Pressure

Hand-written PTX allows explicit register control:
- nvcc's register allocator is good but sometimes wastes
- For small kernels, we can keep everything in registers
- Avoid local memory spills at all costs

## ABI Considerations

- PTX functions using `.entry` must follow CUDA kernel ABI
- Parameters passed via `.param` space
- Driver API uses `cuLaunchKernel` with void** args array
- Inline PTX (`asm volatile`) avoids ABI issues entirely

## Debugging Tips

1. Use `cuModuleLoadDataEx` with JIT error logging
2. `cuda-memcheck` for memory errors
3. `nvprof` / Nsight for warp divergence analysis
4. Compare against CPU reference at every step

## Conservation Law Implementation

The conservation operator preserves the L1 norm (mass):
```
C(v) = α·D⁻¹·A·v + (1-α)·v
```

Properties:
- Stochastic when A is adjacency, D is degree, α ∈ [0,1]
- Related to PageRank / random walk with restart
- Eigenvalue spectrum bounded by [0, 2·max_degree] for Laplacian
