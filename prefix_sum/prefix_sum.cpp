
// nvcc -x cu -arch=compute_75 -code=sm_75 -std=c++17 -o prefix_scan prefix_scan.cpp
#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

#define PRINT

// Function to print the array
void print(float* input, int N, const std::string& message = "") {
    if (!message.empty()) {
        std::cout << message << ": ";
    }
    for (int i = 0; i < N; i++) {
        std::cout << input[i] << " ";
    }
    std::cout << std::endl;
}

// CPU inclusive prefix scan
void prefix_scan_cpu(float* input, float* output, int N) {
    output[0] = input[0];
    for (int i = 1; i < N; i++) {
        output[i] = output[i - 1] + input[i];
    }
}

// Kernel 1: Per-block inclusive scan (Hillis-Steele)
__global__ void block_inclusive_scan(const float *input, float *output, float *block_sums, int n, int block_size) {
    extern __shared__ float temp[];
    int tid = threadIdx.x;
    int block_offset = blockIdx.x * block_size;

    // Load input chunk into shared memory
    if (tid < block_size && block_offset + tid < n) {
        temp[tid] = input[block_offset + tid];
    } else {
        temp[tid] = 0; // Pad with identity
    }
    __syncthreads();

    // Hillis-Steele scan
    for (int offset = 1; offset < n; offset *= 2) {
        float t = 0;
        if (tid >= offset) {
            t = temp[tid - offset];
        }
        __syncthreads();
        if (tid < n && tid >= offset) {
            temp[tid] += t;
        }
        __syncthreads();
    }

    // Store block sum for multi-block case
    if (tid == 0 && block_offset + block_size - 1 < n) {
        block_sums[blockIdx.x] = temp[block_size - 1];
    }

    // Write to output
    if (tid < block_size && block_offset + tid < n) {
        output[block_offset + tid] = temp[tid];
    }
}

// Kernel 2: Scan block sums (Hillis-Steele)
__global__ void scan_block_sums(float *block_sums, int num_blocks) {
    extern __shared__ float temp[];
    int tid = threadIdx.x;

    if (tid < num_blocks) {
        temp[tid] = block_sums[tid];
    } else {
        temp[tid] = 0;
    }
    __syncthreads();

    for (int offset = 1; offset < num_blocks; offset *= 2) {
        float t = 0;
        if (tid >= offset) {
            t = temp[tid - offset];
        }
        __syncthreads();
        if (tid < num_blocks && tid >= offset) {
            temp[tid] += t;
        }
        __syncthreads();
    }

    if (tid < num_blocks) {
        block_sums[tid] = temp[tid];
    }
}

// Kernel 3: Add block sums to per-block scans
__global__ void add_block_sums(float *output, float *block_sums, int n, int block_size) {
    int tid = threadIdx.x;
    int block_offset = blockIdx.x * block_size;

    if (blockIdx.x > 0 && tid < block_size && block_offset + tid < n) {
        output[block_offset + tid] += block_sums[blockIdx.x - 1];
    }
}

int main() {
    int N = 4; // Test case size
    float* input = (float*)malloc(sizeof(float) * N);
    float* output = (float*)malloc(sizeof(float) * N);
    if (!input || !output) {
        std::cerr << "Host memory allocation failed" << std::endl;
        return 1;
    }

    // Initialize array
    float test_input[] = {1.0f, 2.0f, 3.0f, 4.0f};
    for (int i = 0; i < N; i++) {
        input[i] = test_input[i];
        output[i] = 0.0f;
    }
#ifdef PRINT
    print(input, N, "Starting list");
#endif

    // CPU prefix scan
    auto start = std::chrono::steady_clock::now();
    prefix_scan_cpu(input, output, N);
    auto end = std::chrono::steady_clock::now();
    
#ifdef PRINT
    print(output, N, "CPU prefix scan");
#endif
    std::cout << "CPU time: " << std::chrono::duration<double, std::milli>(end - start).count() << " ms" << std::endl;

    // GPU prefix scan
    float *d_input, *d_output, *d_block_sums = nullptr;
    cudaError_t err = cudaMalloc(&d_input, sizeof(float) * N);
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed (input): " << cudaGetErrorString(err) << std::endl;
        free(input);
        free(output);
        return 1;
    }
    err = cudaMalloc(&d_output, sizeof(float) * N);
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed (output): " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input);
        free(input);
        free(output);
        return 1;
    }

    start = std::chrono::steady_clock::now();
    err = cudaMemcpy(d_input, input, sizeof(float) * N, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy H2D failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input);
        cudaFree(d_output);
        free(input);
        free(output);
        return 1;
    }

    int threadsPerBlock = 256;
    int numberOfBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    if (N <= threadsPerBlock) {
        // Single-block case
        block_inclusive_scan<<<1, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(d_input, d_output, nullptr, N, threadsPerBlock);
    } else {
        // Multi-block case
        err = cudaMalloc(&d_block_sums, numberOfBlocks * sizeof(float));
        if (err != cudaSuccess) {
            std::cerr << "CUDA malloc failed (block_sums): " << cudaGetErrorString(err) << std::endl;
            cudaFree(d_input);
            cudaFree(d_output);
            free(input);
            free(output);
            return 1;
        }
        block_inclusive_scan<<<numberOfBlocks, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(d_input, d_output, d_block_sums, N, threadsPerBlock);
        scan_block_sums<<<1, numberOfBlocks, numberOfBlocks * sizeof(float)>>>(d_block_sums, numberOfBlocks);
        add_block_sums<<<numberOfBlocks, threadsPerBlock>>>(d_output, d_block_sums, N, threadsPerBlock);
    }

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input);
        cudaFree(d_output);
        if (d_block_sums) cudaFree(d_block_sums);
        free(input);
        free(output);
        return 1;
    }

    err = cudaMemcpy(output, d_output, sizeof(float) * N, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy D2H failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input);
        cudaFree(d_output);
        if (d_block_sums) cudaFree(d_block_sums);
        free(input);
        free(output);
        return 1;
    }

    end = std::chrono::steady_clock::now();
#ifdef PRINT
    print(output, N, "GPU prefix scan");
#endif
    std::cout << "GPU time: " << std::chrono::duration<double, std::milli>(end - start).count() << " ms" << std::endl;

    // Verify output
    float expected[] = {1.0f, 3.0f, 6.0f, 10.0f};
    bool passed = true;
    float max_diff = 0.0f;
    for (int i = 0; i < N; i++) {
        float diff = fabs(output[i] - expected[i]);
        if (diff > 1e-5) {
            passed = false;
            max_diff = fmax(max_diff, diff);
        }
    }
    if (passed) {
        std::cout << "Test passed!" << std::endl;
    } else {
        std::cout << "Test failed! Max difference: " << max_diff << std::endl;
    }

    cudaFree(d_input);
    cudaFree(d_output);
    if (d_block_sums) cudaFree(d_block_sums);
    free(input);
    free(output);
    return 0;
}
