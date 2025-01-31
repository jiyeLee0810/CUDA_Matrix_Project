#include <stdio.h>
#include <cuda.h>

#define TILE_WIDTH 16

__global__ void matrixMulKernel (float* d_N, float* d_M, float* d_P, int width) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if ((row < width) && (col < width)) {
        float Pvalue = 0;
        for (int k = 0; k < width; ++k) {
            Pvalue += d_N[row*width+k]*d_M[k*width+col];
            d_P[row*width+col] = Pvalue;
        }
    }
}

int main() {
    int width = 32;
    size_t size = width * width * sizeof(float);

    cudaEvent_t start, end; 
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start,0);

    float *h_M = (float*)malloc(size);
    float *h_N = (float*)malloc(size);
    float *h_P = (float*)malloc(size);

    float *d_M, *d_N, *d_P;
    cudaMalloc((void**)&d_M, size);
    cudaMalloc((void**)&d_N, size);
    cudaMalloc((void**)&d_P, size);

    cudaMemcpy(d_M, h_M, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, size, cudaMemcpyHostToDevice);

    dim3 dimGrid(ceil(width/(TILE_WIDTH*1.0)), ceil(width/(TILE_WIDTH*1.0)), 1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

    matrixMulKernel<<<dimGrid, dimBlock>>>(d_M, d_N, d_P, width);
    
    cudaEventRecord(end,0);
    cudaEventSynchronize(end);

    cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(error));
		return 1;
	}
    float time_ms = 0;
    cudaEventElapsedTime(&time_ms, start, end);

    cudaMemcpy(h_P, d_P, size, cudaMemcpyDeviceToHost);
    printf("Execution time for kernel: %.2f ms\n", time_ms);
    printf("results: %d\n", h_P);

    free(h_M);
    free(h_N);
    free(h_P);

    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);

    return 0;
}