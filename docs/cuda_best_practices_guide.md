# CUDA Programming Best Practices Guide

## Critical Areas to Review in CUDA Code

### 1. Memory Management (High Priority)

#### Common Issues
❌ **Using CPU memory allocation in GPU code**
```cuda
// Incorrect - using new/delete
int *data = new int[size];
delete[] data;
```

#### Best Practices
✅ **Use proper CUDA memory management**
```cuda
// Correct approaches
int *data_d;
cudaMalloc((void**)&data_d, size * sizeof(int));
cudaFree(data_d);

// Or use stream-ordered allocation (CUDA 11.2+)
cudaMallocAsync(&data_d, size * sizeof(int), stream);
cudaFreeAsync(data_d, stream);
```

**Key Rules**:
- Always use CUDA memory allocation functions for device memory
- Consider cudaMallocManaged for unified memory when appropriate
- Use stream-ordered allocators for better performance in modern CUDA

### 2. Error Checking (High Priority)

#### Common Issues
❌ **Missing error checks on CUDA API calls**
```cuda
cudaMalloc((void**)&ptr, size);
kernel<<<grid, block>>>();
cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
```

#### Best Practices
✅ **Implement systematic error checking**
```cuda
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

// Usage
CUDA_CHECK(cudaMalloc((void**)&ptr, size));
kernel<<<grid, block>>>();
CUDA_CHECK(cudaGetLastError());
CUDA_CHECK(cudaDeviceSynchronize());
```

### 3. Thread Block Configuration (Medium Priority)

#### Common Issues
⚠️ **Suboptimal thread block sizes**
```cuda
dim3 blockSize(32, 32);  // 1024 threads may limit occupancy
```

#### Best Practices
✅ **Optimize thread block dimensions**
```cuda
// Better occupancy with smaller blocks
constexpr int BLOCK_X = 16;
constexpr int BLOCK_Y = 16;  // 256 threads total
dim3 blockSize(BLOCK_X, BLOCK_Y);

// Ensure multiple of warp size (32)
constexpr int THREADS_PER_BLOCK = 256;  // Common sweet spot
```

**Guidelines**:
- Keep thread blocks between 128-512 threads
- Ensure thread count is multiple of 32 (warp size)
- Profile different configurations for your specific kernels

### 4. Memory Access Patterns (Medium Priority)

#### Common Issues
⚠️ **Uncoalesced memory accesses**
```cuda
// Strided access pattern - poor performance
data[threadIdx.x * stride]
```

#### Best Practices
✅ **Ensure coalesced memory access**
```cuda
// Sequential threads access sequential memory
data[blockIdx.x * blockDim.x + threadIdx.x]

// Use shared memory for irregular patterns
__shared__ float tile[TILE_SIZE];
tile[threadIdx.x] = global_data[computed_index];
__syncthreads();
```

### 5. Kernel Optimization (Medium Priority)

#### Common Issues
❌ **Missing occupancy hints**
```cuda
__global__ void myKernel(...) {
    // No launch bounds specified
}
```

#### Best Practices
✅ **Use launch bounds for better optimization**
```cuda
__launch_bounds__(256, 4)  // max 256 threads, min 4 blocks per SM
__global__ void myKernel(...) {
    // Kernel code
}
```

## Performance Optimization Checklist

### Immediate Optimizations (Do First)

1. **Add Error Checking**
   - Wrap all CUDA API calls
   - Check kernel launches with cudaGetLastError()
   - Add debug builds with extensive checking

2. **Optimize Memory Management**
   - Use appropriate memory types (global, shared, constant, texture)
   - Implement memory pooling for frequent allocations
   - Consider unified memory for simplicity

3. **Configure Thread Blocks**
   - Start with 256 threads per block
   - Profile and adjust based on occupancy
   - Use occupancy calculator API

### Advanced Optimizations

4. **Memory Access Optimization**
   - Ensure coalesced global memory access
   - Use shared memory for data reuse
   - Leverage constant memory for read-only data
   - Consider texture memory for spatial locality

5. **Concurrent Execution**
   ```cuda
   // Use streams for overlap
   cudaStream_t stream1, stream2;
   cudaStreamCreate(&stream1);
   cudaStreamCreate(&stream2);
   
   kernel1<<<grid, block, 0, stream1>>>();
   kernel2<<<grid, block, 0, stream2>>>();
   ```

6. **Math Optimizations**
   - Use fast math compiler flag: `-use_fast_math`
   - Prefer single precision when possible
   - Use intrinsic functions: `__sinf()`, `__expf()`

## Architecture Considerations

### Compute Capability Checking
```cuda
int device;
cudaGetDevice(&device);
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, device);

if (prop.major < 7) {
    // Use fallback for older GPUs
}
```

### Conditional Compilation
```cuda
#if __CUDA_ARCH__ >= 700
    // Volta and newer optimizations
#else
    // Legacy code path
#endif
```

## Common Pitfalls to Avoid

1. **Memory Issues**
   - Mixing host and device pointers
   - Forgetting to free allocated memory
   - Race conditions in shared memory

2. **Performance Issues**
   - Bank conflicts in shared memory
   - Divergent warps from branching
   - Low occupancy from excessive register use

3. **Correctness Issues**
   - Missing synchronization barriers
   - Incorrect grid/block dimensions
   - Overflow in index calculations

## Profiling and Debugging Tools

### Essential Tools
- **Nsight Compute**: Kernel profiling
- **Nsight Systems**: System-wide profiling
- **cuda-memcheck**: Memory error detection
- **compute-sanitizer**: Race condition detection

### Key Metrics to Monitor
- Occupancy (aim for >50%)
- Memory bandwidth utilization
- Warp efficiency
- Cache hit rates
- SM utilization

## Best Practice Summary

### Always Do
✅ Check CUDA errors systematically  
✅ Profile before optimizing  
✅ Use appropriate memory types  
✅ Ensure coalesced memory access  
✅ Test on target GPU architectures  

### Never Do
❌ Assume kernels succeed without checking  
❌ Use CPU memory allocation for GPU data  
❌ Ignore compiler warnings  
❌ Optimize without profiling first  
❌ Hard-code for single GPU architecture  

## Quick Reference

### Memory Hierarchy (Fastest to Slowest)
1. Registers (per thread)
2. Shared Memory (per block)
3. L1 Cache
4. L2 Cache
5. Global Memory

### Typical Optimization Workflow
1. **Functional correctness** - Get it working
2. **Profile** - Identify bottlenecks
3. **Memory optimization** - Fix access patterns
4. **Occupancy tuning** - Adjust block sizes
5. **Algorithm optimization** - Consider different approaches
6. **Architecture-specific tuning** - Target specific GPUs

## Conclusion

Effective CUDA programming requires attention to memory management, error handling, and performance optimization. Start with the critical issues (error checking and memory management), then progressively optimize based on profiling results. Remember that different kernels may require different optimization strategies, so always measure performance impacts of changes.