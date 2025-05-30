cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(torch_square_operations)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(torch_square_operations main.cpp)
target_link_libraries(torch_square_operations "${TORCH_LIBRARIES}")
set_property(TARGET torch_square_operations PROPERTY CXX_STANDARD 14)

# The following code is needed for Windows MSVC
if(MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET torch_square_operations
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:torch_square_operations>)
endif(MSVC)