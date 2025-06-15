cmake_minimum_required(VERSION 3.20)
project(gpu_playground 
    VERSION 0.1.0
    DESCRIPTION "A playground for GPU programming"
    LANGUAGES CXX CUDA)
add_subdirectory(reverse_array)
add_subdirectory(matrix_transpose)
