// nvcc -x cu -o transpose transpose.cpp -std=c++17
// on tesla T4 GPU
// nvcc -x cu -arch=compute_70 -code=sm_70,compute_70 -std=c++17 -o transpose transpose.cpp
#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

#define PRINT
#define BLOCK_SIZE 16

// Function to print the matrix
void print_matrix(float* matrix, int rows, int cols, const std::string& message = "") {
    if (!message.empty()) {
        std::cout << message << ":\n";
    }
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << matrix[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if(r >= rows || c >= cols)
        return;
    output[c*rows + r] = input[r*cols + c];    
}

void matrix_transpose_cpu(const float* input, float* output, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            output[c*rows + r] = input[r*cols + c];
        }
    }
}

int main() {
    int rows = 4;
    int cols = 6;
    int input_size = rows * cols;
    int output_size = cols * rows;
    
    float* input = (float*)malloc(sizeof(float) * input_size);
    float* output_cpu = (float*)malloc(sizeof(float) * output_size);
    float* output_gpu = (float*)malloc(sizeof(float) * output_size);
    
    if (!input || !output_cpu || !output_gpu) {
        std::cerr << "Host memory allocation failed" << std::endl;
        return 1;
    }

    // Initialize matrix
    for (int i = 0; i < input_size; i++) {
        input[i] = i + 1;
    }
    
    #ifdef PRINT
    print_matrix(input, rows, cols, "Original matrix");
    #endif

    // CPU transpose
    auto start = std::chrono::steady_clock::now();
    matrix_transpose_cpu(input, output_cpu, rows, cols);
    auto end = std::chrono::steady_clock::now();
    
    #ifdef PRINT
    print_matrix(output_cpu, cols, rows, "CPU transposed");
    #endif
    std::cout << "CPU time: " << std::chrono::duration<double, std::milli>(end - start).count() << " ms" << std::endl;

    // GPU transpose
    float* d_input;
    float* d_output;
    cudaError_t err = cudaMalloc(&d_input, sizeof(float) * input_size);
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc input failed: " << cudaGetErrorString(err) << std::endl;
        free(input);
        free(output_cpu);
        free(output_gpu);
        return 1;
    }
    
    err = cudaMalloc(&d_output, sizeof(float) * output_size);
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc output failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input);
        free(input);
        free(output_cpu);
        free(output_gpu);
        return 1;
    }

    start = std::chrono::steady_clock::now();
    err = cudaMemcpy(d_input, input, sizeof(float) * input_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy H2D failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input);
        cudaFree(d_output);
        free(input);
        free(output_cpu);
        free(output_gpu);
        return 1;
    }

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numberOfBlocks((cols + BLOCK_SIZE - 1) / BLOCK_SIZE,
                        (rows + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    matrix_transpose_kernel<<<numberOfBlocks, threadsPerBlock>>>(d_input, d_output, rows, cols);
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

    err = cudaMemcpy(output_gpu, d_output, sizeof(float) * output_size, cudaMemcpyDeviceToHost);
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
    print_matrix(output_gpu, cols, rows, "GPU transposed");
    #endif
    std::cout << "GPU time: " << std::chrono::duration<double, std::milli>(end - start).count() << " ms" << std::endl;

    // Verify results match
    bool results_match = true;
    for (int i = 0; i < output_size; i++) {
        if (output_cpu[i] != output_gpu[i]) {
            results_match = false;
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