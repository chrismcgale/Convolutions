cmake_minimum_required(VERSION 3.10)
project(MyProgram CUDA CXX)

# Add all source files to the project
file(GLOB_RECURSE SOURCES
    "*.cpp"
    "*.cu"
)

# Add an executable target
add_executable(my_program ${SOURCES})

# Set C++ standard
set_property(TARGET my_program PROPERTY CUDA_STANDARD 11)
set_property(TARGET my_program PROPERTY CXX_STANDARD 11)

# Include CUDA libraries
find_package(CUDAToolkit REQUIRED)
target_link_libraries(my_program PRIVATE CUDA::cudart)

# Set optimization flags
if (CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(my_program PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
    target_compile_options(my_program PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3>)
endif()
