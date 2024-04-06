// convolution_kernel.h
#ifndef CONVOLUTION_KERNEL_H
#define CONVOLUTION_KERNEL_H

__global__ void convolution_2D_cached_const_mem_tiling_kernel(float *N, float *P, int width, int length);

#endif