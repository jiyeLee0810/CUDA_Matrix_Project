#include <stdio.h>
#include <cuda.h>

#define KERNEL_SIZE 3
#define TILE_WIDTH 16
#define BLOCK_SIZE (TILE_WIDTH + (KERNEL_SIZE - 1))

__constant__ float Mc[KERNEL_SIZE][KERNEL_SIZE];

__global__ void convolutionKernel(float *N, float *P, int width, int height) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row_o = blockIdx.y * TILE_WIDTH + ty;
    int col_o = blockIdx.x * TILE_WIDTH + tx;

    int row_i = row_o - (KERNEL_SIZE)/2;
    int col_i = col_o - (KERNEL_SIZE)/2;  

    __shared__ float N_ds[BLOCK_SIZE][BLOCK_SIZE];

    if (row_i >= 0 && row_i < height && col_i >= 0 && col_i < width) {
        N_ds[ty][tx] = N[row_i * width + col_i];
    } else {
        N_ds[ty][tx] = 0.0f;
    }
    
    __syncthreads();

    if (row_o < height && col_o < width && tx < TILE_WIDTH && ty < TILE_WIDTH) {
        float output = 0.0f;
        for (int i = 0; i < KERNEL_SIZE; i++) {
            for (int j = 0; j < KERNEL_SIZE; j++) {
                output += Mc[i][j] * N_ds[ty + i][tx + j];
            }
        }
        P[row_o * width + col_o] = output;
    }
}

void verification(const float *N, const float *M, const float *P, int Rows, int Columns) {
    int r, c, h, w;
    int row_i, col_i;
    bool equal;
    float* results;

    results = (float*)malloc(Rows * Columns * sizeof(float));
    memset(results, 0, Rows * Columns * sizeof(float));

    for (r = 0; r < Rows; r++) {
        for (c = 0; c < Columns; c++) {
            for (h = 0; h < KERNEL_SIZE; h++) {
                for (w = 0; w < KERNEL_SIZE; w++) {
                    row_i = r - ((KERNEL_SIZE - 1) / 2) + h;
                    col_i = c - ((KERNEL_SIZE - 1) / 2) + w;
                    if ((row_i >= 0) && (row_i < Rows) && (col_i >= 0) && (col_i < Columns)) {
                        results[r * Columns + c] += (M[h * KERNEL_SIZE + w] * N[row_i * Columns + col_i]);
                    }
                }
            }
        }
    }

    equal = true;
    for (int i = 0; i < Rows * Columns && equal; i++) {
        if (abs(results[i] - P[i]) >= 0.001f) {
            equal = false;
            printf("NOT EQUAL!\n");
            printf("Mismatch at index %d: CPU = %f, GPU = %f\n", i, results[i], P[i]);
        }
    }

    if (equal) {
        printf("Results are equal!\n");
    } else {
        printf("Results are NOT equal!\n");
    }

    free(results);
}

int main() {
    int width = 256;
    int height = 256;

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    float *h_N = (float*)malloc(width * height * sizeof(float));
    float *h_M = (float*)malloc(KERNEL_SIZE * KERNEL_SIZE * sizeof(float));
    float *h_P = (float*)malloc(width * height * sizeof(float));

    for(int i = 0; i < width*height; i++) {
        h_N[i] = 1.0f; 
    }
    for(int i = 0; i < KERNEL_SIZE*KERNEL_SIZE; i++) {
        h_M[i] = 1.0f;
    }
    memset(h_P, 0, width * height * sizeof(float));

    float *d_N, *d_P;
    cudaMalloc((void**)&d_N, width*height*sizeof(float));
    cudaMalloc((void**)&d_P, width*height*sizeof(float));

    cudaMemcpy(d_N, h_N, width*height*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(Mc, h_M, KERNEL_SIZE*KERNEL_SIZE*sizeof(float));

    int gridX = (width  + TILE_WIDTH - 1) / TILE_WIDTH;
    int gridY = (height + TILE_WIDTH - 1) / TILE_WIDTH;  
    dim3 dimGrid(gridX, gridY);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    cudaEventRecord(start, 0);
    convolutionKernel<<<dimGrid, dimBlock>>>(d_N, d_P, width, height);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(error));
        return 1;
    }

    float time_ms = 0;
    cudaEventElapsedTime(&time_ms, start, end);
    printf("Execution time for kernel: %.2f ms\n", time_ms);

    cudaMemcpy(h_P, d_P, width*height*sizeof(float), cudaMemcpyDeviceToHost);

    verification(h_N, h_M, h_P, width, height);

    free(h_N);
    free(h_M);
    free(h_P);
    cudaFree(d_N);
    cudaFree(d_P);

    return 0;
}
