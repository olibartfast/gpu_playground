// nvcc -x cu -o reverse reverse.cpp -std=c++17
// on tesla T4 GPU
// nvcc -x cu -arch=compute_70 -code=sm_70,compute_70 -std=c++17 -o reverse reverse.cpp
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

__global__ void reverse_array(float* input, int N) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int j = N - tid - 1;
    if (tid < N/2) {
        float tmp = input[tid];
        input[tid] = input[j];
        input[j] = tmp;
    }
}

void reverse_array_cpu(float* input, int N) {
    int i = 0, j = N - 1;
    while (i < j) {
        float tmp = input[i];
        input[i] = input[j];
        input[j] = tmp;
        i++;
        j--;
    }
}


int main() {
    int N = 10;
    float* input = (float*)malloc(sizeof(float) * N);
    if (!input) {
        std::cerr << "Host memory allocation failed" << std::endl;
        return 1;
    }

    // Initialize array
    for (int i = 0; i < N; i++) {
        input[i] = i;
    }
    #ifdef PRINT
    print(input, N, "Starting list");
    #endif

    // CPU reversal
    auto start = std::chrono::steady_clock::now();
    reverse_array_cpu(input, N);
    auto end = std::chrono::steady_clock::now();
    
    #ifdef PRINT
    print(input, N, "CPU reversed");
    #endif
    std::cout << "CPU time: " << std::chrono::duration<double, std::milli>(end - start).count() << " ms" << std::endl;

    // GPU reversal
    float* d_input;
    cudaError_t err = cudaMalloc(&d_input, sizeof(float) * N);
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed: " << cudaGetErrorString(err) << std::endl;
        free(input);
        return 1;
    }

    start = std::chrono::steady_clock::now();
    err = cudaMemcpy(d_input, input, sizeof(float) * N, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy H2D failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input);
        free(input);
        return 1;
    }

    int threadsPerBlock = 256;
    int numberOfBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    reverse_array<<<numberOfBlocks, threadsPerBlock>>>(d_input, N);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input);
        free(input);
        return 1;
    }

    err = cudaMemcpy(input, d_input, sizeof(float) * N, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy D2H failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input);
        free(input);
        return 1;
    }

    end = std::chrono::steady_clock::now();
    #ifdef PRINT
    print(input, N, "GPU reversed");
    #endif
    std::cout << "GPU time: " << std::chrono::duration<double, std::milli>(end - start).count() << " ms" << std::endl;

    cudaFree(d_input);
    free(input);
    return 0;
}
