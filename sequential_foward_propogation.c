#include <math.h>

float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}


// Foward propogation path of a convolutional layer

// M is the number of output feature maps
// C is the number of input feature maps
// H is the height of each input feature map
// W is the width of each input feature map
// K is the height and width of each 2D filter
// X stores the input feature map and is a [C x H x W] array
// Y stores the filter banks and is a [M x C x K x K] array
// Z stores the output feature maps and is a [M x (H - K + 1) x (W - K + 1)] array
void convlayer_forward_sequential(int M, int C, int H, int W, int K, float* X, float* Y, float* Z) {
    int H_out = H - K + 1;
    int W_out = W - K + 1;

    for (int m = 0; m < M; m++) {                   // for each output feature map
        for (int h = 0; h < H_out; h++) {           // for each output element
            for (int w = 0; w < W_out; w++) {
                Z[m * H_out * W + h * W + w] = 0;       // Y[m, h, w]
                for (int c = 0; c < C; c++) {       // Sum over all input feature maps
                    for (int p = 0; p < K; p++) {   // KxK filter
                        for (int q = 0; q < K; q++) {
                            Z[m * H * W + h * W + w] += X[c * H * W + (h + p) * W + (w + q)] * Y[m * C * K * K + c * K * K + p * K + q];
                        }
                    }
                }
            }
        }
    }
}

// Foward propogation of a subsampling layer

// M is the number of output feature maps
// H is the height of each input feature map, divisible by K
// W is the width of each input feature map, divisible by K
// K is the height and width of each 2D sample
// B is a [M] array of biases
// Y is a [M * H * W] array of input samples
// S is a [M * H / K * W / K] array of output samples
void subsamplingLayer_forward(int M, int H, int W, int K, int* B, float* Y, float* S) {
    for (int m = 0; m < M; m++) {                   // for each output feature map
        for (int h = 0; h < H / K; h++) {           // for each output element
            for (int w = 0; w < W / K; w++) {
                S[m * H * W + h * W + w] = 0.0f;
                for (int p = 0; p < K; p++) {       // KxK input samples
                    for (int q = 0; q < K; q++) {
                        S[m * H * W + h * W + w] += Y[m * (W - K + 1) * (H - K + 1) + (K * h + p) * (W - K + 1) + K*w + q] / (K*K);
                    }
                }
                // Add bias and apply non-linear activation
                S[m * H * W + h * W + w] = sigmoid(S[m * H * W + h * W + w] + B[m]);
            }
        }
    }
}