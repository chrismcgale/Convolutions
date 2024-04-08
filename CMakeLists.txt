cmake_minimum_required(VERSION 3.10)
project(Friede-The-Convolution CUDA CXX)

# Add all source files to the project
file(GLOB_RECURSE SOURCES
    "*.cpp"
    "*.cu"
)

# Add an executable target
add_executable(2dConvolution ${SOURCES})

# Set C++ standard
set_property(TARGET 2dConvolution PROPERTY CUDA_STANDARD 11)
set_property(TARGET 2dConvolution PROPERTY CXX_STANDARD 11)

# Include CUDA libraries
find_package(CUDAToolkit REQUIRED)
target_link_libraries(2dConvolution PRIVATE CUDA::cudart)

# Set optimization flags
if (CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(2dConvolution PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
    target_compile_options(2dConvolution PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3>)
endif()
