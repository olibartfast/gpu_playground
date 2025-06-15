cmake_minimum_required(VERSION 3.20)

# Enable CUDA for this directory
enable_language(CUDA)

# Create the softmax array executable
add_executable(softmax softmax.cpp)

# Set required CUDA properties and flags
set_target_properties(softmax PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 70)  # For Tesla T4 GPU

# Set C++ standard
set_property(TARGET softmax PROPERTY CXX_STANDARD 17)