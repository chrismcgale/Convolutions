#define FILTER_RADIUS 2
#define IN_TILE_DIM 32
#define OUT_TILE_DIM (IN_TILE_DIM - 2 * FILTER_RADIUS)

__constant__ float F_2D[2*FILTER_RADIUS+1][2*FILTER_RADIUS+1];
__constant__ float F_3D[2*FILTER_RADIUS+1][2*FILTER_RADIUS+1][2*FILTER_RADIUS+1];
__global__ void convolution_2D_const_mem_tiling_kernel(float *N, float *P, int width, int length) {
    int col = blockIdx.x*OUT_TILE_DIM + threadIdx.x - FILTER_RADIUS;
    int row = blockIdx.y*OUT_TILE_DIM + threadIdx.y - FILTER_RADIUS;

    __shared__ N_s[IN_TILE_DIM][IN_TILE_DIM];

    if (row >= 0 && row < length && col >= 0 && col < width) {
        N_s[threadIdx.y][threadIdx.x] = N[row*width + col];
    } else {
        N_s[threadIdx.y][threadIdx.x] = 0.0;
    }

    __syncthreads();

    int tileCol = threadIdx.x - FILTER_RADIUS;
    int tileRow = threadIdx.y - FILTER_RADIUS;

    if (row >= 0 && row < length && col >= 0 && col < width) {
        if (tileRow >= 0 && tileRow < OUT_TILE_DIM && tileCol >= 0 && tileCol < OUT_TILE_DIM) {
        float pValue = 0.0f;
        for (int fRow = 0; fRow < 2*FILTER_RADIUS + 1; fRow++) {
            for (int fCol; fCol < 2*FILTER_RADIUS + 1; fCol++) {
                pValue += F_2D[fRow][fCol] * N_s[tileRow+fRow][tileCol+fCol];
            }
        }
        P[row*width+col] = pValue;
        }
    }
}

__global__ void convolution_3D_const_mem_tiling_kernel(float *N, float *P, int width, int length, int height) {
    int col = blockIdx.x*OUT_TILE_DIM + threadIdx.x - FILTER_RADIUS;
    int row = blockIdx.y*OUT_TILE_DIM + threadIdx.y - FILTER_RADIUS;
    int slice = blockIdx.z*OUT_TILE_DIM + threadIdx.z - FILTER_RADIUS;

    __shared__ N_s[IN_TILE_DIM][IN_TILE_DIM][IN_TILE_DIM];

    if (row >= 0 && row < length && col >= 0 && col < width && slice >= 0 && slice < height) {
        N_s[threadIdx.y][threadIdx.x][threadIdx.z] = N[row*width*length + col*length + slice];
    } else {
        N_s[threadIdx.y][threadIdx.x][threadIdx.z] = 0.0;
    }

    __syncthreads();

    int tileCol = threadIdx.x - FILTER_RADIUS;
    int tileRow = threadIdx.y - FILTER_RADIUS;
    int tileSlice = threadIdx.z - FILTER_RADIUS;

    if (row >= 0 && row < length && col >= 0 && col < width && slice >= 0 && slice < height) {
        if (tileRow >= 0 && tileRow < OUT_TILE_DIM && tileCol >= 0 && tileCol < OUT_TILE_DIM  && tileSlice >= 0 && tileSlice < OUT_TILE_DIM) {
        float pValue = 0.0f;
        for (int fRow = 0; fRow < 2*FILTER_RADIUS + 1; fRow++) {
            for (int fCol; fCol < 2*FILTER_RADIUS + 1; fCol++) {
                for (int fSlice; fSlice < 2*FILTER_RADIUS + 1; fSlice++) {
                    pValue += F_3D[fSlce][fRow][fCol] * N_S[slice*length*width + inRow*width + inCol];
                }
            }
        }
        P[slice*width*length+row*width+col] = pValue;
        }
    }
}