#define FILTER_RADIUS 2
#define TILE_DIM 32

__constant__ float F_2D[2*FILTER_RADIUS+1][2*FILTER_RADIUS+1];
__constant__ float F_3D[2*FILTER_RADIUS+1][2*FILTER_RADIUS+1][2*FILTER_RADIUS+1];

// Takes advantage of halo cells likely being in L1/L2 cache and so we (hopefully) only need to fetch from DRAM once rather than once per time cell is in halo
// Allows us to have input tiles = output tiles
__global__ void convolution_2D_cached_const_mem_tiling_kernel(float *N, float *P, int width, int length) {
    int col = blockIdx.x*TILE_DIM + threadIdx.x;
    int row = blockIdx.y*TILE_DIM + threadIdx.y;

    __shared__ N_s[TILE_DIM][TILE_DIM];

    // Load tile into shared memory
    if (row < length && col < width) {
        N_s[threadIdx.y][threadIdx.x] = N[row*width + col];
    } else {
        N_s[threadIdx.y][threadIdx.x] = 0.0;
    }

    __syncthreads();

    if (row < length && col < width) {

        int tileCol = threadIdx.x - FILTER_RADIUS;
        int tileRow = threadIdx.y - FILTER_RADIUS;
        
        float pValue = 0.0f;
        for (int fRow = 0; fRow < 2*FILTER_RADIUS + 1; fRow++) {
            for (int fCol; fCol < 2*FILTER_RADIUS + 1; fCol++) {
                // If in tile check shared
                if (tileRow+fRow >= 0 && tileRow+fRow < TILE_DIM && tileCol+fCol >= 0 && tileCol+fCol < TILE_DIM) {
                    pValue += F_2D[fRow][fCol] * N_s[threadIdx.y+fRow][threadIdx.x+fCol];
                } else {
                    // Check cache / Main main 
                    if (row-FILTER_RADIUS+fRow >= 0 && row-FILTER_RADIUS+fRow < length && col-FILTER_RADIUS+fCol >= 0 && col-FILTER_RADIUS+fCol < width) {
                        pValue += F_2D[fRow][fCol] * N[(row-FILTER_RADIUS+fRow)*width+col-FILTER_RADIUS+fCol];
                    }
                }
            }
        }
        P[row*width+col] = pValue;
    }
}