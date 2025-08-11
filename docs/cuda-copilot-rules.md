# CUDA Programming Rules for Copilot Agents

## Core Principles

### 1. Memory Hierarchy Optimization
- **Always prioritize memory coalescing**: Ensure global memory accesses by threads in a warp are coalesced into as few transactions as possible
- **Use shared memory for frequently accessed data**: Copy data from global to shared memory when it will be accessed multiple times
- **Minimize global memory accesses**: Global memory has 100s of cycles of latency
- **Prefer structure of arrays (SoA) over array of structures (AoS)** for better coalescing
- **Align data structures to 32-byte boundaries** for optimal memory transactions

### 2. Thread Organization Rules
- **Thread block size must be a multiple of 32** (warp size) for optimal efficiency
- **Use between 128-256 threads per block** as a starting point
- **Never exceed 1024 threads per block** (hardware limit)
- **Ensure grid size > number of SMs** to keep all multiprocessors busy
- **Launch thousands of blocks** to scale across different GPU architectures

### 3. Kernel Design Patterns
```cuda
// CORRECT: Coalesced memory access pattern
__global__ void goodKernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 2.0f;  // Adjacent threads access adjacent memory
    }
}

// INCORRECT: Strided access pattern
__global__ void badKernel(float* data, int n, int stride) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * stride;
    if (idx < n) {
        data[idx] = data[idx] * 2.0f;  // Causes uncoalesced access
    }
}
```

### 4. Memory Management Best Practices
- **Use pinned memory for host-device transfers**: `cudaHostAlloc()` instead of `malloc()`
- **Batch small transfers into larger ones** to minimize overhead
- **Use asynchronous transfers with streams** to overlap computation and data transfer
- **Prefer `cudaMallocAsync()` and `cudaFreeAsync()`** over `cudaMalloc()` and `cudaFree()`
- **Never allocate memory inside kernels** - do all allocations on host

### 5. Synchronization and Atomics
```cuda
// Use __syncthreads() only when necessary
__global__ void reductionKernel(float* data, float* result) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load to shared memory
    sdata[tid] = data[idx];
    __syncthreads();  // Essential synchronization
    
    // Reduction in shared memory
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();  // Essential synchronization
    }
    
    if (tid == 0) atomicAdd(result, sdata[0]);
}
```

### 6. Compute Capability Awareness
- **Always check compute capability**: Use `cudaGetDeviceProperties()`
- **Use architecture-specific features conditionally**:
```cuda
#if __CUDA_ARCH__ >= 700  // Volta and newer
    // Use tensor cores or other newer features
#endif
```
- **Compile for multiple architectures**: Use `-gencode` flags for broader compatibility

### 7. Occupancy Optimization
- **Balance register usage**: Use `__launch_bounds__` to limit registers per thread
- **Monitor shared memory usage**: Don't exceed 48KB per block
- **Aim for at least 50% occupancy** but remember higher isn't always better
- **Use the occupancy calculator** or runtime API to tune launch parameters

### 8. Performance Critical Rules
- **Avoid divergent branches within warps**:
```cuda
// GOOD: Warp-aligned branching
if ((threadIdx.x / 32) == 0) { /* All threads in warp take same path */ }

// BAD: Thread-divergent branching  
if (threadIdx.x % 2 == 0) { /* Threads in same warp diverge */ }
```

- **Use fast math functions when precision allows**: `__sinf()`, `__expf()`, etc.
- **Avoid integer division and modulo** with non-power-of-2 divisors
- **Use texture memory for spatial locality** in 2D/3D data access patterns

### 9. Error Handling
```cuda
// Always check CUDA API calls
#define CUDA_CHECK(call) \
do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(error)); \
        exit(1); \
    } \
} while(0)

// Check kernel launches
kernel<<<blocks, threads>>>();
CUDA_CHECK(cudaGetLastError());
CUDA_CHECK(cudaDeviceSynchronize());
```

### 10. Stream Management
```cuda
// Use multiple streams for concurrent operations
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

// Overlap transfers and computation
cudaMemcpyAsync(d_a, h_a, size, cudaMemcpyHostToDevice, stream1);
kernel1<<<grid, block, 0, stream1>>>(d_a);

cudaMemcpyAsync(d_b, h_b, size, cudaMemcpyHostToDevice, stream2);
kernel2<<<grid, block, 0, stream2>>>(d_b);

// Clean up
cudaStreamDestroy(stream1);
cudaStreamDestroy(stream2);
```

## Common Pitfalls to Avoid

1. **Never access host memory directly from device code** (except with unified memory)
2. **Don't use `malloc()` or `new` in device code**
3. **Avoid recursive device functions** (limited stack space)
4. **Don't ignore the 5-second kernel timeout** on Windows/Linux with display
5. **Never assume memory is zeroed** after allocation
6. **Don't mix float and double unnecessarily** (causes implicit conversions)
7. **Avoid `printf()` in kernels** for production code (performance impact)
8. **Don't forget to free device memory** (causes memory leaks)

## Memory Optimization Checklist

- [ ] Data aligned to 32-byte boundaries
- [ ] Coalesced global memory access patterns
- [ ] Shared memory used for data reuse
- [ ] Bank conflicts minimized in shared memory
- [ ] Constant memory used for read-only data accessed uniformly
- [ ] Texture memory considered for 2D spatial locality
- [ ] L2 cache persistence configured for frequently accessed data

## Launch Configuration Guidelines

```cuda
// Recommended launch configuration pattern
void launchKernel(float* data, int n) {
    int blockSize;  // Threads per block
    int minGridSize;  // Minimum grid size for max occupancy
    int gridSize;  // Actual grid size
    
    // Get occupancy-based launch configuration
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, 
                                       myKernel, 0, 0);
    
    // Round up grid size
    gridSize = (n + blockSize - 1) / blockSize;
    
    myKernel<<<gridSize, blockSize>>>(data, n);
    cudaGetLastError();  // Check for launch errors
}
```

## Performance Metrics to Monitor

1. **Effective bandwidth**: Should approach theoretical maximum
2. **Occupancy**: Aim for >50% but not at the expense of other optimizations
3. **Memory throughput**: Monitor coalescing efficiency
4. **Warp execution efficiency**: Minimize divergence
5. **Instruction throughput**: Balance memory and compute operations

## Advanced Optimization Techniques

### Warp-Level Primitives (CC 3.0+)
```cuda
// Use warp shuffle for intra-warp communication
int value = __shfl_sync(0xffffffff, data, srcLane);
```

### Cooperative Groups (CC 3.5+)
```cuda
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void kernel() {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    // Use for flexible synchronization
}
```

### Unified Memory Best Practices
```cuda
// Use prefetching for better performance
cudaMallocManaged(&data, size);
cudaMemPrefetchAsync(data, size, deviceId);
// Perform computation
cudaMemPrefetchAsync(data, size, cudaCpuDeviceId);  // Prefetch back to CPU
```

## Code Review Checklist for CUDA

When reviewing CUDA code, ensure:

1. ✓ All CUDA API calls have error checking
2. ✓ Kernel launches check for errors with `cudaGetLastError()`
3. ✓ Memory access patterns are coalesced
4. ✓ Thread block sizes are multiples of 32
5. ✓ No unnecessary `__syncthreads()` calls
6. ✓ Appropriate memory types used (shared, constant, texture)
7. ✓ Streams used for asynchronous operations
8. ✓ Pinned memory used for frequent transfers
9. ✓ Device memory is properly freed
10. ✓ Compute capability requirements documented

## Template Structures

### Basic CUDA Program Structure
```cuda
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Device function
__device__ float deviceFunction(float x) {
    return x * x;
}

// Kernel
__global__ void kernel(float* d_out, float* d_in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_out[idx] = deviceFunction(d_in[idx]);
    }
}

int main() {
    // 1. Allocate host memory
    float *h_in = (float*)malloc(size);
    float *h_out = (float*)malloc(size);
    
    // 2. Allocate device memory
    float *d_in, *d_out;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);
    
    // 3. Transfer data to device
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);
    
    // 4. Launch kernel
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    kernel<<<gridSize, blockSize>>>(d_out, d_in, n);
    
    // 5. Transfer results back
    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);
    
    // 6. Clean up
    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);
    
    return 0;
}
```

These rules should be applied by copilot agents when generating, reviewing, or optimizing CUDA code to ensure high-performance, correct, and maintainable GPU applications.