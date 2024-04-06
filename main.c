#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>

#include "convolution_kernel.h"


int main() {
    int width = 1024;  // Width of the input and output arrays
    int length = 1024; // Length of the input and output arrays

    // Allocate memory for input and output arrays on the host
    float *h_N = new float[width * length];
    float *h_P = new float[width * length];

    // Generate random input array
    for (int i = 0; i < width * length; ++i) {
        h_N[i] = static_cast<float>(rand()) / RAND_MAX; // Random float between 0 and 1
    }

    // Allocate memory on the GPU
    float *d_N, *d_P;
    cudaMalloc((void **)&d_N, width * length * sizeof(float));
    cudaMalloc((void **)&d_P, width * length * sizeof(float));

    // Copy input array from host to device
    cudaMemcpy(d_N, h_N, width * length * sizeof(float), cudaMemcpyHostToDevice);

    // Define grid and block dimensions for CUDA kernel launch
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (length + blockDim.y - 1) / blockDim.y);

    // Launch the CUDA kernel
    convolution_2D_cached_const_mem_tiling_kernel<<<gridDim, blockDim>>>(d_N, d_P, width, length);

    // Copy output array from device to host
    cudaMemcpy(h_P, d_P, width * length * sizeof(float), cudaMemcpyDeviceToHost);

    // Output the result
    std::cout << "Output array P:\n";
    for (int i = 0; i < length; ++i) {
        for (int j = 0; j < width; ++j) {
            std::cout << h_P[i * width + j] << " ";
        }
        std::cout << "\n";
    }

    // Free memory on the GPU
    cudaFree(d_N);
    cudaFree(d_P);

    // Free memory on the host
    delete[] h_N;
    delete[] h_P;

    return 0;
}
