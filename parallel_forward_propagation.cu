#DEFINE TILE_WIDTH 16

int convLayer_parallel_forward_propagation(int N, int M, int C, int H, int W, int K, float* X, float* Y, float* Z) {
    int H_out = H - K + 1;
    int W_out = W - K + 1;

    for (int n = 0; n < N; n++) {                       // for each sample in the mini-batch
        for (int m = 0; m < M; m++) {                   // for each output feature map
            for (int h = 0; h < H_out; h++) {           // for each output element
                int W_grid = W_out/TILE_WIDTH;
                int H_grid = H_out/TILE_WIDTH;
                T = H_grid * W_grid;
                dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
                dim3 gridDim(M, T, N);

                float* X_d, * Y_d, * Z_d;

                unsigned int x_size = sizeof(float) * C * H * W;
                unsigned int y_size = sizeof(float) * M * C * K * K;
                unsigned int z_size = sizeof(float) * M * H_out * W_out;

                cudaMalloc((void**)&X_d, x_size);
                cudaMalloc((void**)&Y_d, y_size);
                cudaMalloc((void**)&Z_d, z_size);

                cudaMemcpy(X_d, X, x_size, cudaMemcpyHostToDevice);
                cudaMemcpy(Y_d, Y, y_size, cudaMemcpyHostToDevice);

                ConvLayerForward_Kernel<<< gridDim, blockDim >>>(C, W_grid, K, X_d, W_d, Y_d);

                cudaMemcpy(Z, Z_d, z_size, cudaMemcpyDeviceToHost);

                cudaFree(X_d);
                cudaFree(Y_d);
                cudaFree(Z_d);
            }
        }
    }
}

// TO DO: CONSUMES TOO MUCH GMB
// ADD CONSTANT MEM CACHING AND TILING
__global__ void ConvLayer_forward_Kernel(int C, int W_grid, int K, float* X, float* Y, float* Z) {
    int m = blockIdx.x;
    int h = (blockIdx.y / W_grid)*TILE_WIDTH + threadIdx.y;
    int w = (blockIdx.y % W_grid)*TILE_WIDTH + threadIdx.x;
    int n = blockIdx.x;

    float acc = 0.0f;
    for (int c = 0; c < C; c++) {
        for (int p = 0; p < K; p++) {
            for (int q = 0; q < K; q++) {
                acc += X[n * C * H * W + c * H * W + (h + p) * W + (w + q)] * Y[m * C * K * K + c * K * K + p * K + q];
            }
        }
    }
    Z[n * M * H_out * W_out + m * H_out * W_out + h * W_out + w] = acc;
}