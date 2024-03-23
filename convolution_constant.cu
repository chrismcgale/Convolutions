#define FILTER_RADIUS 2

__constant__ float F_2D[2*FILTER_RADIUS+1][2*FILTER_RADIUS+1];
__constant__ float F_3D[2*FILTER_RADIUS+1][2*FILTER_RADIUS+1][2*FILTER_RADIUS+1];
__global__ void convolution_2D_const_mem_kernel(float *N, float *P, int width, int height) {
    int outCol = blockIdx.x*blockDim.x + threadIdx.x;
    int outRow = blockIdx.y*blockDim.y + threadIdx.y;
    float pValue = 0.0f;
    for (int fRow = 0; fRow < 2*FILTER_RADIUS + 1; fRow++) {
        for (int fCol; fCol < 2*FILTER_RADIUS + 1; fCol++) {
            int inRow = outRow - FILTER_RADIUS + fRow;
            int inCol = outCol - FILTER_RADIUS + fCol;
            if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
                pValue += F_2D[fRow][fCol] * N[inRow*width + inCol];
            }
        }
    }
    P[outRow][outCol] = pValue;
}

__global__ void convolution_3D_const_mem_kernel(float *N, float *P, int width, int length, int height) {
    int outCol = blockIdx.x*blockDim.x + threadIdx.x;
    int outRow = blockIdx.y*blockDim.y + threadIdx.y;
    int outSlice = blockIdx.z*blockDim.y + threadIdx.z;
    float pValue = 0.0f;
    for (int fRow = 0; fRow < 2*FILTER_RADIUS + 1; fRow++) {
        for (int fCol; fCol < 2*FILTER_RADIUS + 1; fCol++) {
            for (int fSlice; fSlice < 2*FILTER_RADIUS + 1; fSlice++) {
                int inRow = outRow - FILTER_RADIUS + fRow;
                int inCol = outCol - FILTER_RADIUS + fCol;
                int inSlice = outSlice - FILTER_RADIUS + fSlice;
                if (inRow >= 0 && inRow < length && inCol >= 0 && inCol < width && inSlice >= 0 && inSlice < height) {
                    pValue += F_3D[fRow][fCol][fSlice] * N[inRow*length*width + inCol*width + slice];
                }
            }
        }
    }
    P[outRow][outCol][outSlice] = pValue;
}