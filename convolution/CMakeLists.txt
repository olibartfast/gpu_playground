cmake_minimum_required(VERSION 3.18)  # Specify minimum CMake version

project(convolution LANGUAGES CXX CUDA)

find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Explicitly list your source files
set(CPP_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/reference_calc.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/compare.cpp
   # ${CMAKE_CURRENT_SOURCE_DIR}/convolution.cpp
)

set(CUDA_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/convolution.cu
)

# Optional: If you really need to use GLOB
# file(GLOB HEADER_FILES "*.hpp" "*.h")

message("CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
message("CPP files: ${CPP_FILES}")
message("CUDA files: ${CUDA_FILES}")



add_executable(${PROJECT_NAME} ${CPP_FILES} ${CUDA_FILES})

# Link against CUDA libraries
target_link_libraries(${PROJECT_NAME} PRIVATE 
    ${OpenCV_LIBS}
    CUDA::cudart
)
target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS})

# Set C++ standard (optional, but recommended)
set_target_properties(${PROJECT_NAME} PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON
)

# Optionally, specify CUDA architectures
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_ARCHITECTURES "50;60;70;75;80"
)