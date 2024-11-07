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

### Running C++ Code  
The C++ examples must be run locally, as Google Colab does not support C++/CUDA out of the box. If you need to test C++ code, it's recommended to set up a local development environment with **CUDA Toolkit**. For Python, you can run the code directly in your Google Colab notebook as described above and make sure to enable the GPU in  **Runtime > Change runtime type > GPU**.
