#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <cmath>
#include <iomanip>
#include <vector>

#define BSIZE 256

void silu(const float* d_input, float* d_output, int N);

__global__ void silu_kernel(const float* input, float* output, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i >= N )
        return;
    float sigma = 1.0f / (1 + exp(-input[i]));    
    output[i] = input[i]*sigma;    
}


void silu_cpu(const float* input, float* output, int N) {
    for(int i=0; i<N; i++) {
        float sigma = 1.0f / (1 + exp(-input[i]));
        output[i] = input[i]*sigma;
    }
}


void silu(const float* d_input, float* d_output, int N) {
    int threadsPerBlock = BSIZE;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    silu_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
    
}


int main(int argc, char const *argv[])
{
    int N = 1024;
    std::vector<float> input(N);
    std::vector<float> output_cpu(N);
    std::vector<float> output_gpu(N);
    for(int i=0; i<N; i++) {
        input[i] = (float)(i % 100);
    }

    auto start_cpu = std::chrono::steady_clock::now();
    silu_cpu(input.data(), output_cpu.data(), N);
    auto end_cpu = std::chrono::steady_clock::now();

    std::chrono::duration<double, std::milli> duration_cpu = end_cpu - start_cpu;
    std::cout << "CPU duration: " << duration_cpu.count() << " ms" << std::endl;

    // GPU computation
    auto start_gpu = std::chrono::steady_clock::now();
    float *d_input, *d_output;
    cudaMalloc(&d_input, sizeof(float) * N);
    cudaMalloc(&d_output, sizeof(float) * N);
    cudaMemcpy(d_input, input.data(), sizeof(float) * N, cudaMemcpyHostToDevice);
    silu(d_input, d_output, N);
    cudaMemcpy(output_gpu.data(), d_output, sizeof(float) * N, cudaMemcpyDeviceToHost);
    auto end_gpu = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> duration_gpu = end_gpu - start_gpu;
    std::cout << "GPU duration: " << duration_gpu.count() << " ms" << std::endl;
    cudaFree(d_input);
    cudaFree(d_output); 
    for(int i=0; i<N; i++) {
        if(fabs(output_cpu[i] - output_gpu[i]) > 1e-5) {
            std::cout << "Mismatch at index " << i << ": CPU " << output_cpu[i] << " vs GPU " << output_gpu[i] << std::endl;
            return -1;
        }
    }
    std::cout << "Results match!" << std::endl;
    return 0;
}


