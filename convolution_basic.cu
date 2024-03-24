
// Very basic, doesn't take advantage of constant memory caching of the filter or tiling N
__global__ void convolution_2D_basic_kernel(float *N, float *F, float *P, int r, int width, int height) {
    int outCol = blockIdx.x*blockDim.x + threadIdx.x;
    int outRow = blockIdx.y*blockDim.y + threadIdx.y;
    float pValue = 0.0f;
    for (int fRow = 0; fRow < 2*r + 1; fRow++) {
        for (int fCol; fCol < 2*r + 1; fCol++) {
            int inRow = outRow - r + fRow;
            int inCol = outCol - r + fCol;
            if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
                pValue += F[fRow][fCol] * N[inRow*width + inCol];
            }
        }
    }
    P[outRow][outCol] = pValue;
}

__global__ void convolution_3D_basic_kernel(float *N, float *F, float *P, int r, int width, int length, int height) {
    int outCol = blockIdx.x*blockDim.x + threadIdx.x;
    int outRow = blockIdx.y*blockDim.y + threadIdx.y;
    int outSlice = blockIdx.z*blockDim.y + threadIdx.z;
    float pValue = 0.0f;
    for (int fRow = 0; fRow < 2*r + 1; fRow++) {
        for (int fCol; fCol < 2*r + 1; fCol++) {
            for (int fSlice; fSlice < 2*r + 1; fSlice++) {
                int inRow = outRow - r + fRow;
                int inCol = outCol - r + fCol;
                int inSlice = outSlice - r + fSlice;
                if (inRow >= 0 && inRow < length && inCol >= 0 && inCol < width && inSlice >= 0 && inSlice < height) {
                    pValue += F[fSlce][fRow][fCol] * N[slice*length*width + inRow*width + inCol];
                }
            }
        }
    }
    P[outSlice][outRow][outCol] = pValue;
}