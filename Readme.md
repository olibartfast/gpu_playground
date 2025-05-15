# GPU Testing with Python and C++  
GPU programming and testing using Python and C++. The examples are inspired by resources from the [GPU Mode Group](https://www.youtube.com/channel/UCJgIbYl6C5no72a0NUAPcTA)
## Prerequisites  
Before diving into the code snippets, ensure the following tools are installed:
- **CUDA Toolkit** (for both Python and C++ development)
- **PyTorch or TensorFlow** (for Python-based GPU testing)
- **A CUDA-capable GPU** and appropriate drivers
- **CMake** and **gcc/g++** (for C++ development)
If you don't have a local GPU available, you can easily run the Python examples on **Google Colab**, which offers free access to GPUs.
### Installing Required Libraries
For Python:
```bash
pip install torch  # or tensorflow-gpu
pip install tinygrad  # for TinyGrad
pip install triton  # for OpenAI Triton
```
For C++ (ensure CUDA is set up in your environment):
```bash
sudo apt-get install nvidia-cuda-toolkit
```
## Running on Google Colab  
Google Colab offers free access to a GPU, and the Python code can be executed directly there with minimal setup. Follow these steps:
1. Go to [Google Colab](https://colab.research.google.com/).
2. Open a new notebook.
3. Enable the GPU by navigating to **Runtime > Change runtime type** and selecting **GPU**.
4. Run the following code snippets directly in Colab.
## Example 1: GPU Testing with Python (PyTorch)  
We begin with a simple GPU-accelerated tensor operation using **PyTorch**:
```python
import torch
# Check if GPU is available
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("GPU available: ", torch.cuda.get_device_name(0))
else:
    device = torch.device('cpu')
    print("Running on CPU")
# Create a tensor on the GPU
tensor = torch.rand(1000, 1000, device=device)
# Perform a simple operation
result = torch.matmul(tensor, tensor)
print("Result on GPU: ", result)
```
### Profiling CUDA Kernels in Python  
You can profile CUDA kernel operations to measure GPU performance using PyTorch's profiler:
```python
import torch
from torch.profiler import profile, ProfilerActivity
# Enable CUDA profiling
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    tensor = torch.rand(1000, 1000, device='cuda')
    result = torch.matmul(tensor, tensor)
print(prof.key_averages().table(sort_by="cuda_time_total"))
```

## Example 2: TinyGrad - Lightweight GPU Acceleration

TinyGrad is a minimalist neural network library designed with simplicity and educational purposes in mind. It provides GPU acceleration with a clean and readable codebase.

```python
from tinygrad.tensor import Tensor
import numpy as np

# Check if GPU is available
print(f"Using GPU: {Tensor.gpu}")

# Create tensors
a = Tensor.rand(1024, 1024)
b = Tensor.rand(1024, 1024)

# Perform operations
result = a.matmul(b)
print(result.shape)

# Basic neural network operations
x = Tensor.rand(128, 10)
W = Tensor.rand(10, 20)
b = Tensor.rand(20)

# Forward pass
out = x.matmul(W).add(b)
print(out.shape)
```

### TinyGrad with CUDA Performance Testing

```python
import time
from tinygrad.tensor import Tensor
from tinygrad.helpers import GlobalCounters

# Enable GPU mode
Tensor.no_gpu = False

# Create large tensors
size = 4096
a = Tensor.rand(size, size)
b = Tensor.rand(size, size)

# Benchmark matrix multiplication
GlobalCounters.reset()
start_time = time.time()
c = a.matmul(b)
c.realize()  # Force computation
end_time = time.time()

print(f"Matrix multiplication {size}x{size} took {end_time - start_time:.4f} seconds")
print(f"FLOPS: {2 * size**3 / (end_time - start_time):.2e}")
print(GlobalCounters.print_hot())
```

## Example 3: OpenAI Triton - GPU Programming Made Accessible

Triton is a language and compiler designed to make GPU programming more accessible to AI researchers and engineers. It simplifies writing efficient GPU kernels.

```python
import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # Block sizes
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    # Matrix multiplication kernel implementation
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    
    # Calculate the group ID and the start indices for this block
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    
    # Calculate the start indices for this block
    m_start = pid_m * BLOCK_SIZE_M
    n_start = pid_n * BLOCK_SIZE_N
    
    # Create pointers to the input and output tensors
    a_ptrs = a_ptr + m_start * K + tl.arange(0, BLOCK_SIZE_M)[:, None] * K + tl.arange(0, BLOCK_SIZE_K)[None, :]
    b_ptrs = b_ptr + tl.arange(0, BLOCK_SIZE_K)[:, None] * N + n_start + tl.arange(0, BLOCK_SIZE_N)[None, :]
    c_ptrs = c_ptr + m_start * N + n_start + tl.arange(0, BLOCK_SIZE_M)[:, None] * N + tl.arange(0, BLOCK_SIZE_N)[None, :]
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Iterate through k dimension
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K * N
    
    # Store the result
    tl.store(c_ptrs, acc)

# Function to use the kernel
def matmul(a, b):
    # Get tensor dimensions
    M, K = a.shape
    K, N = b.shape
    
    # Allocate output tensor
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    # Define grid and block sizes
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 16
    
    # Launch kernel
    grid = lambda META: (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )
    
    return c

# Test the custom kernel
a = torch.randn((1024, 1024), device='cuda', dtype=torch.float32)
b = torch.randn((1024, 1024), device='cuda', dtype=torch.float32)

# Run both PyTorch and Triton implementations
c_torch = torch.matmul(a, b)
c_triton = matmul(a, b)

# Verify the results
print(f"Max difference: {(c_torch - c_triton).abs().max().item()}")
```

### Benchmarking Triton vs PyTorch

```python
import torch
import triton
import time

# Previous matmul function and kernel definitions...

# Benchmarking
sizes = [512, 1024, 2048, 4096]
torch_times = []
triton_times = []

for size in sizes:
    a = torch.randn((size, size), device='cuda', dtype=torch.float32)
    b = torch.randn((size, size), device='cuda', dtype=torch.float32)
    
    # Warmup
    torch.matmul(a, b)
    matmul(a, b)
    torch.cuda.synchronize()
    
    # Benchmark PyTorch
    start = time.time()
    for _ in range(10):
        torch.matmul(a, b)
    torch.cuda.synchronize()
    torch_time = (time.time() - start) / 10
    torch_times.append(torch_time)
    
    # Benchmark Triton
    start = time.time()
    for _ in range(10):
        matmul(a, b)
    torch.cuda.synchronize()
    triton_time = (time.time() - start) / 10
    triton_times.append(triton_time)
    
    print(f"Size: {size}x{size}")
    print(f"PyTorch: {torch_time*1000:.2f} ms")
    print(f"Triton: {triton_time*1000:.2f} ms")
    print(f"Speedup: {torch_time/triton_time:.2f}x")
    print("-" * 30)
```

### Running C++ Code  
The C++ examples must be run locally, as Google Colab does not support C++/CUDA out of the box. If you need to test C++ code, it's recommended to set up a local development environment with **CUDA Toolkit**. For Python, you can run the code directly in your Google Colab notebook as described above and make sure to enable the GPU in  **Runtime > Change runtime type > GPU**.

## Further resources and references
* https://github.com/gpu-mode
* https://leetgpu.com/
* https://github.com/tinygrad/tinygrad
* https://github.com/triton-lang/triton
* https://github.com/NVIDIA/cuda-samples
