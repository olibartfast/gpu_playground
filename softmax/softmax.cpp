// nvcc -x cu -arch=compute_70 -code=sm_70,compute_70 -std=c++17 -o softmax softmax.cpp
#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <cmath>
#include <iomanip>
#include <vector>
#include <float.h> // For FLT_MAX

#define PRINT // Comment this out for large N to avoid printing thousands of numbers

// Define the number of threads per block. 256 is a good general-purpose choice.
#define BSIZE 256

// Forward declaration of the new GPU solver function
void softmax_gpu_efficient(const float* d_input, float* d_output, int N);

// ===================================================================================
// HELPER AND CPU FUNCTIONS (Unchanged)
// ===================================================================================

// Function to print the array
void print(const float* input, int N, const std::string& message = "") {
    if (!message.empty()) {
        std::cout << message << ": ";
    }
    std::cout << std::fixed << std::setprecision(4);
    for (int i = 0; i < N; i++) {
        std::cout << input[i] << " ";
    }
    std::cout << std::endl;
}

void softmax_cpu(const float* input, float* output, int N) {
    // Step 1: Find maximum value for numerical stability
    float max_val = -FLT_MAX;
    for (int i = 0; i < N; i++) {
        max_val = fmaxf(max_val, input[i]);
    }
    
    // Step 2: Compute sum of exponentials with max subtracted
    float sum_exp = 0.0f;
    for (int i = 0; i < N; i++) {
        sum_exp += expf(input[i] - max_val);
    }
    
    // Step 3: Compute softmax for each element
    for (int i = 0; i < N; i++) {
        output[i] = expf(input[i] - max_val) / sum_exp;
    }
}

// ===================================================================================
// EFFICIENT MULTI-KERNEL GPU IMPLEMENTATION
// ===================================================================================

__global__ void find_max_partial_kernel(const float* input, float* partial_maxs, int N) {
    __shared__ float sdata[BSIZE];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + tid;

    float my_max = -FLT_MAX;
    while (i < N) {
        my_max = fmaxf(my_max, input[i]);
        i += gridDim.x * blockDim.x;
    }
    sdata[tid] = my_max;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }

    if (tid == 0) partial_maxs[blockIdx.x] = sdata[0];
}

__global__ void sum_exp_partial_kernel(const float* input, float* partial_sums, const float* max_val, int N) {
    __shared__ float sdata[BSIZE];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + tid;

    float my_sum = 0.0f;
    while (i < N) {
        my_sum += expf(input[i] - (*max_val));
        i += gridDim.x * blockDim.x;
    }
    sdata[tid] = my_sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) partial_sums[blockIdx.x] = sdata[0];
}

__global__ void reduce_final_kernel(float* partials, float* result, int N, bool is_max_reduction) {
    __shared__ float sdata[BSIZE];
    unsigned int tid = threadIdx.x;
    unsigned int i = tid;

    float my_val = is_max_reduction ? -FLT_MAX : 0.0f;
    while (i < N) {
        if (is_max_reduction) my_val = fmaxf(my_val, partials[i]);
        else my_val += partials[i];
        i += blockDim.x;
    }
    sdata[tid] = my_val;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (is_max_reduction) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
            else sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) result[0] = sdata[0];
}

__global__ void softmax_elementwise_kernel(const float* input, float* output, const float* max_val, const float* sum_exp, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        output[i] = expf(input[i] - (*max_val)) / (*sum_exp);
    }
}

// Host-side orchestrator for the efficient GPU softmax
void softmax_gpu_efficient(const float* d_input, float* d_output, int N) {
    int threadsPerBlock = BSIZE;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    float *d_partials, *d_max_val, *d_sum_exp;
    cudaMalloc(&d_partials, blocksPerGrid * sizeof(float));
    cudaMalloc(&d_max_val, sizeof(float));
    cudaMalloc(&d_sum_exp, sizeof(float));

    // Stage 1: Find max value
    find_max_partial_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_partials, N);
    reduce_final_kernel<<<1, threadsPerBlock>>>(d_partials, d_max_val, blocksPerGrid, true);

    // Stage 2: Sum exponentials
    sum_exp_partial_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_partials, d_max_val, N);
    reduce_final_kernel<<<1, threadsPerBlock>>>(d_partials, d_sum_exp, blocksPerGrid, false);

    // Stage 3: Final element-wise calculation
    softmax_elementwise_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, d_max_val, d_sum_exp, N);

    cudaFree(d_partials);
    cudaFree(d_max_val);
    cudaFree(d_sum_exp);
}

// ===================================================================================
// MAIN FUNCTION
// ===================================================================================

int main() {
    // Use N=8 for printing and correctness check.
    // For performance, use a much larger N and comment out the #define PRINT.
    int N = 8;
    // int N = 1 << 20; // 1,048,576 elements for performance test

    std::vector<float> input(N);
    std::vector<float> output_cpu(N);
    std::vector<float> output_gpu(N);

    // Initialize array with some sample values
    if (N == 8) {
        float sample_values[] = {1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f};
        for (int i = 0; i < N; i++) input[i] = sample_values[i];
    } else {
        // For large N, fill with random-like data
        for (int i = 0; i < N; i++) input[i] = (float)(i % 100);
    }
    
    #ifdef PRINT
    print(input.data(), N, "Input");
    #endif

    // --- CPU softmax ---
    auto start_cpu = std::chrono::steady_clock::now();
    softmax_cpu(input.data(), output_cpu.data(), N);
    auto end_cpu = std::chrono::steady_clock::now();
    
    #ifdef PRINT
    print(output_cpu.data(), N, "CPU softmax");
    #endif
    
    float cpu_sum = 0.0f;
    for (int i = 0; i < N; i++) cpu_sum += output_cpu[i];
    std::cout << "CPU softmax sum: " << std::fixed << std::setprecision(6) << cpu_sum << std::endl;
    std::cout << "CPU time: " << std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count() << " ms" << std::endl << std::endl;

    // --- GPU softmax ---
    float *d_input, *d_output;
    cudaMalloc(&d_input, sizeof(float) * N);
    cudaMalloc(&d_output, sizeof(float) * N);

    auto start_gpu = std::chrono::steady_clock::now();
    
    cudaMemcpy(d_input, input.data(), sizeof(float) * N, cudaMemcpyHostToDevice);
    
    // *** CALL THE NEW EFFICIENT GPU SOLVER ***
    softmax_gpu_efficient(d_input, d_output, N);
    
    // Ensure all GPU work is done before stopping the timer
    cudaDeviceSynchronize();

    cudaMemcpy(output_gpu.data(), d_output, sizeof(float) * N, cudaMemcpyDeviceToHost);
    
    auto end_gpu = std::chrono::steady_clock::now();

    #ifdef PRINT
    print(output_gpu.data(), N, "GPU softmax");
    #endif
    
    float gpu_sum = 0.0f;
    for (int i = 0; i < N; i++) gpu_sum += output_gpu[i];
    std::cout << "GPU softmax sum: " << std::fixed << std::setprecision(6) << gpu_sum << std::endl;
    std::cout << "GPU time: " << std::chrono::duration<double, std::milli>(end_gpu - start_gpu).count() << " ms" << std::endl << std::endl;

    // --- Verification ---
    bool results_match = true;
    const float tolerance = 1e-5f; // Slightly larger tolerance for multi-stage reduction
    for (int i = 0; i < N; i++) {
        if (fabsf(output_cpu[i] - output_gpu[i]) > tolerance) {
            results_match = false;
            std::cout << "Mismatch at index " << i << ": CPU=" << output_cpu[i] 
                      << " GPU=" << output_gpu[i] << std::endl;
            break;
        }
    }
    
    if (results_match) {
        std::cout << "Results match!" << std::endl;
    } else {
        std::cout << "Results do not match!" << std::endl;
    }

    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}
