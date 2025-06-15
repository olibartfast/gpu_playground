cmake_minimum_required(VERSION 3.20)

# Enable CUDA for this directory
enable_language(CUDA)

# Create the reverse array executable
add_executable(matrix_transpose matrix_transpose.cpp)

# Set required CUDA properties and flags
set_target_properties(matrix_transpose PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 70)  # For Tesla T4 GPU

# Set C++ standard
set_property(TARGET matrix_transpose PROPERTY CXX_STANDARD 17)