#include <cuda.h>
#include <vector>
#include <fstream>

#include "matrix_data.h"

using namespace std;

__global__ void convolutionKernel(float *N, float *P, float *M) {
    int kernelSize = M.row;
    int blockSize = TILE_WIDTH + (kernelSize - 1);

    int x = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int y = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int z = blockIdx.z * TILE_WIDTH + threadIdx.z;

    float output = 0.0f;

    __shared__ float N_ds[blockSize][blockSize][blockSize];
    extern __shared__ float deviceKernel[];

    if (x < N.col && y < N.row && z < N.depth) {
        N_ds[threadIdx.y][threadIdx.x][threadIdx.z] = N[(z*N.row*N.col)+(y*N.col)+x];
    } else {
        N_ds[threadIdx.y][threadIdx.x][threadIdx.z] = 0.0f;
    }

    __syncthreads();

    if (threadIdx.y < TILE_WIDTH && threadIdx.x < TILE_WIDTH && threadIdx.z < TILE_WIDTH) {
        for (int i = 0; i < kernelSize; i++) {
            for (int j = 0; j < kernelSize; j++) {
                for (int k = 0; k < kernelSize; k++) {
                    output += N_ds[i+threadIdx.y][j+threadIdx.x][k+threadIdx.z] 
                                + deviceKernel[(i*kernelSize*kernelSize)+(j*kernelSize)+k];
                }
            }
        }
        if (x < P.col && y < P.row && x < P.depth) {
            P.hostMemoryP[(z*P.row*P.col)+(y*P.col)+x] = output;
        }
    }
}

int main() {
    matrixData N, M, P;
    int width, height, depth, kernelSize;
    string testDirectory = "test3";

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);

    N.hostMemoryP = nullptr;
    M.hostMemoryP = nullptr;

    float* inputMatrix = nullptr;
    float* kernelMatrix = nullptr;

    read3DInputFile(testDirectory, width, height, depth, kernelSize, inputMatrix, kernelMatrix);

    N.row = width;
    N.col = height;
    N.depth = depth;
    N.size = width*height*depth*sizeof(float);
    N.hostMemoryP = inputMatrix;

    M.row = kernelSize;
    M.col = kernelSize;
    M.depth = kernelSize;
    M.size = kernelSize*kernelSize*kernelSize*sizeof(float);
    M.hostMemoryP = kernelMatrix;

    P.row = width - kernelSize + 1;
    P.col = height - kernelSize + 1;
    P.depth = depth - kernelSize + 1;
    P.size = (width - kernelSize + 1)*(height - kernelSize + 1)*(depth - kernelSize + 1)*sizeof(float);
    P.hostMemoryP = (float*)malloc(P.size);

    int blockSize = TILE_WIDTH + (kernelSize - 1);
    cudaError_t resultN = cudaMalloc((void**)&N.deviceMemoryP, N.size);
    cudaError_t resultP = cudaMalloc((void**)&P.deviceMemoryP, P.size);

    cudaMemcpy(N.deviceMemoryP, N.hostMemoryP, N.size, cudaMemcpyHostToDevice);

    dim3 dimBlock(blockSize, blockSize, blockSize);
    dim3 dimGrid(ceil(P.row/(float)TILE_WIDTH), ceil(P.col/(float)TILE_WIDTH), ceil(P.col/(float)TILE_WIDTH));
    
    return 0;
}
