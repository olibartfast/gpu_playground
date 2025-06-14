# GPU Testing with Python, C++, and Mojo

This repository contains examples and utilities for GPU programming using Python, C++, and Mojo. The examples are inspired by resources from the GPU Mode Group.

## Prerequisites

Ensure the following tools are installed:

- CUDA Toolkit (for Python, C++, and Mojo development)
- PyTorch or TensorFlow (for Python-based GPU testing)
- A CUDA-capable GPU and appropriate drivers
- CMake and `gcc`/`g++` (for C++ development)
- Mojo SDK (for Mojo development, available from Modular)

## Installing Required Libraries

### For Python:

```bash
pip install torch  # or tensorflow-gpu
pip install tinygrad  # for TinyGrad
pip install triton  # for OpenAI Triton
```

### For C++ (ensure CUDA is set up):

```bash
sudo apt-get install nvidia-cuda-toolkit
```

### For Mojo:

* Install the Mojo SDK by following instructions from Modular's official documentation.
* Ensure CUDA support is configured for GPU acceleration.

## Project Structure
- `reverse_array/` - Array reversal examples
- other examples incoming...

## Examples

For detailed code examples and benchmarks using different frameworks, see [EXAMPLES.md](EXAMPLES.md).

## Further Resources and References

* [GPU Mode GitHub](https://github.com/gpu-mode)
* [LeetGPU](https://leetgpu.com)
* [TinyGrad GitHub](https://github.com/geohot/tinygrad)
* [Triton GitHub](https://github.com/openai/triton)
* [NVIDIA CUDA Samples](https://github.com/NVIDIA/cuda-samples)
* [Mojo Documentation](https://docs.modular.com/mojo)
* [Modular CUDA Setup Guide](https://www.modular.com/mojo)



