// Back propogation of dx_dW derivates

// M is the number of output feature maps
// C is the number of input feature maps
// H_in is the height of each input feature map, divisible by K
// W_in is the width of each input feature map, divisible by K
// K is the height and width of each 2D sample
// dE_dY is a [C * H_in * W_in] array of derivates WRT Y
// dE_dx is a [C * H_in * W_in] array of derivates WRT Y
// W is a [M * H / K * W / K] array of weights
void convlayer_back_x_grad(int M, int C, int H_in, int W_in, int K, float* dE_dY, float* dE_dX, float* W) {
    int H_out = H_in - K + 1;
    int W_out = W_in - K + 1;

    for (int c = 0; c < C; c++) {  
        for (int h = 0; h < H_in; h++) {  
            for (int w = 0; w < W_in; w++) {
                dE_dX[c * H_in * W_in + h * W_in + w] = 0; // dE_dX[c, h, w]
            }
        }
    }

    for (int m = 0; m < M; m++) {   
        for (int h = 0; h < H_in - 1; h++) { 
            for (int w = 0; w < W_in - 1; w++) {
                for (int c = 0; c < C; c++) {  
                    for (int p = 0; p < K; p++) {   
                        for (int q = 0; q < K; q++) {
                            if (h - p >= 0 && w - p >= 0 && h - p < H_out && w - p < W_out) {
                                dE_dX[c * H_in * W_in + h * W_in + w] += dE_dY[c * H_in * W_in + (h - p) *  W_in + (w - q)]     // dE_dY[m, h-p, w-p]                    
                                                                    * W[m * C * K * K + c * K * K + (K - p) * K + (K - q)];     // W[m, c, k-p, k-q]
                            }
                        }
                    }
                }
            }
        }
    }
}

// Back propogation of dy_dW derivates

// M is the number of output feature maps
// C is the number of input feature maps
// H_in is the height of each input feature map, divisible by K
// W_in is the width of each input feature map, divisible by K
// K is the height and width of each 2D sample
// dE_dY is a [C * H_in * W_in] array of derivates WRT Y
// dE_dW is a [M * C * K * K] array of derivates WRT W
// W is a [M * H / K * W / K] array of weights
void convlayer_back_w_grad(int M, int C, int H_in, int W_in, int K, float* dE_dY, float* dE_dW, float* W) {
    int H_out = H_in - K + 1;
    int W_out = W_in - K + 1;

    for (int m = 0; m < M; m++) {  
        for (int c = 0; c < C; c++) {  
            for (int p = 0; p < K; p++) {   
                    for (int q = 0; q < K; q++) {
                        dE_dW[m * C * K * K + c * K * K + p * K + q] = 0; // dE_dW[m, c, p, q]
                    }
            }
        }
    }

    for (int m = 0; m < M; m++) {   
        for (int h = 0; h < H_out; h++) { 
            for (int w = 0; w < W_out; w++) {
                for (int c = 0; c < C; c++) {  
                    for (int p = 0; p < K; p++) {   
                        for (int q = 0; q < K; q++) {
                            dE_dW[m * C * K * K + c * K * K + p * K + q] += dE_dY[c * H_in * W_in + (h + p) *  W_in + (w + q)]     // dE_dY[m, h+p, w+p]                    
                                                                        * W[m * C * K * K + c * K * K + (K - p) * K + (K - q)];    // W[m, c, h, w]
                        }
                    }
                }
            }
        }
    }
}