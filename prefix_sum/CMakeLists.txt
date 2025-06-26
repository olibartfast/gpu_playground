
cmake_minimum_required(VERSION 3.20)

# Enable CUDA for this directory
enable_language(CUDA)

# Set CUDA language property for source files
set_source_files_properties(prefix_sum.cpp PROPERTIES LANGUAGE CUDA)

# Create the reverse array executable
add_executable(prefix_sum prefix_sum.cpp)

# Set required CUDA properties and flags
set_target_properties(prefix_sum PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 70)  # For Tesla T4 GPU

# Set C++ standard
set_property(TARGET prefix_sum PROPERTY CXX_STANDARD 17)
