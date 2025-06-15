// nvcc -x cu -o softmax softmax.cpp -std=c++17
// on tesla T4 GPU
// nvcc -x cu -arch=compute_70 -code=sm_70,compute_70 -std=c++17 -o softmax softmax.cpp
#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <cmath>
#include <iomanip>

#define PRINT

// Function to print the array
void print(float* input, int N, const std::string& message = "") {
    if (!message.empty()) {
        std::cout << message << ": ";
    }
    std::cout << std::fixed << std::setprecision(4);
    for (int i = 0; i < N; i++) {
        std::cout << input[i] << " ";
    }
    std::cout << std::endl;
}

__global__ void softmax_kernel(const float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        // Step 1: Find maximum value for numerical stability (naive - each thread finds max)
        float max_val = input[0];
        for (int i = 1; i < N; i++) {
            max_val = fmaxf(max_val, input[i]);
        }
        
        // Step 2: Compute sum of exponentials with max subtracted
        float sum_exp = 0.0f;
        for (int i = 0; i < N; i++) {
            sum_exp += expf(input[i] - max_val);
        }
        
        // Step 3: Compute softmax
        output[idx] = expf(input[idx] - max_val) / sum_exp;
    }
}

void softmax_cpu(const float* input, float* output, int N) {
    // Step 1: Find maximum value for numerical stability
    float max_val = input[0];
    for (int i = 1; i < N; i++) {
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

int main() {
    int N = 8;
    float* input = (float*)malloc(sizeof(float) * N);
    float* output_cpu = (float*)malloc(sizeof(float) * N);
    float* output_gpu = (float*)malloc(sizeof(float) * N);
    
    if (!input || !output_cpu || !output_gpu) {
        std::cerr << "Host memory allocation failed" << std::endl;
        return 1;
    }

    // Initialize array with some sample values
    float sample_values[] = {1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f};
    for (int i = 0; i < N; i++) {
        input[i] = sample_values[i];
    }
    
    #ifdef PRINT
    print(input, N, "Input");
    #endif

    // CPU softmax
    auto start = std::chrono::steady_clock::now();
    softmax_cpu(input, output_cpu, N);
    auto end = std::chrono::steady_clock::now();
    
    #ifdef PRINT
    print(output_cpu, N, "CPU softmax");
    #endif
    
    // Verify CPU softmax sums to 1
    float cpu_sum = 0.0f;
    for (int i = 0; i < N; i++) {
        cpu_sum += output_cpu[i];
    }
    std::cout << "CPU softmax sum: " << std::fixed << std::setprecision(6) << cpu_sum << std::endl;
    std::cout << "CPU time: " << std::chrono::duration<double, std::milli>(end - start).count() << " ms" << std::endl;

    // GPU softmax
    float* d_input;
    float* d_output;
    cudaError_t err = cudaMalloc(&d_input, sizeof(float) * N);
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc input failed: " << cudaGetErrorString(err) << std::endl;
        free(input);
        free(output_cpu);
        free(output_gpu);
        return 1;
    }
    
    err = cudaMalloc(&d_output, sizeof(float) * N);
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc output failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input);
        free(input);
        free(output_cpu);
        free(output_gpu);
        return 1;
    }

    start = std::chrono::steady_clock::now();
    err = cudaMemcpy(d_input, input, sizeof(float) * N, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy H2D failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input);
        cudaFree(d_output);
        free(input);
        free(output_cpu);
        free(output_gpu);
        return 1;
    }

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    softmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input);
        cudaFree(d_output);
        free(input);
        free(output_cpu);
        free(output_gpu);
        return 1;
    }

    err = cudaMemcpy(output_gpu, d_output, sizeof(float) * N, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy D2H failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input);
        cudaFree(d_output);
        free(input);
        free(output_cpu);
        free(output_gpu);
        return 1;
    }

    end = std::chrono::steady_clock::now();
    #ifdef PRINT
    print(output_gpu, N, "GPU softmax");
    #endif
    
    // Verify GPU softmax sums to 1
    float gpu_sum = 0.0f;
    for (int i = 0; i < N; i++) {
        gpu_sum += output_gpu[i];
    }
    std::cout << "GPU softmax sum: " << std::fixed << std::setprecision(6) << gpu_sum << std::endl;
    std::cout << "GPU time: " << std::chrono::duration<double, std::milli>(end - start).count() << " ms" << std::endl;

    // Verify results match (with small tolerance for floating point differences)
    bool results_match = true;
    const float tolerance = 1e-6f;
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
    free(input);
    free(output_cpu);
    free(output_gpu);
    return 0;
}