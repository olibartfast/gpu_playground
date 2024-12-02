#pragma once

#include <cuda_runtime.h>
#include <stdexcept>

class GpuTimer {
private:
    cudaEvent_t start_;
    cudaEvent_t stop_;

public:
    GpuTimer() noexcept(false) {
        if (cudaEventCreate(&start_) != cudaSuccess ||
            cudaEventCreate(&stop_) != cudaSuccess) {
            throw std::runtime_error("Failed to create CUDA events");
        }
    }

    ~GpuTimer() noexcept {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    // Delete copy constructor and assignment operator
    GpuTimer(const GpuTimer&) = delete;
    GpuTimer& operator=(const GpuTimer&) = delete;

    void start() noexcept(false) {
        if (cudaEventRecord(start_, 0) != cudaSuccess) {
            throw std::runtime_error("Failed to record start event");
        }
    }

    void stop() noexcept(false) {
        if (cudaEventRecord(stop_, 0) != cudaSuccess) {
            throw std::runtime_error("Failed to record stop event");
        }
    }

    float elapsed() const noexcept(false) {
        float elapsed;
        if (cudaEventSynchronize(stop_) != cudaSuccess ||
            cudaEventElapsedTime(&elapsed, start_, stop_) != cudaSuccess) {
            throw std::runtime_error("Failed to calculate elapsed time");
        }
        return elapsed;
    }
};