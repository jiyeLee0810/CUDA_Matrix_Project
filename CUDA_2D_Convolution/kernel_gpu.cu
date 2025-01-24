#include <stdio.h>
#include <cuda.h>

#define KERNEL_SIZE 3
#define TILE_WIDTH 16
#define BLOCK_SIZE (TILE_WIDTH + (KERNEL_SIZE - 1))

__constant__ float Mc[KERNEL_SIZE][KERNEL_SIZE];

__global__ void convolutionKernel(float *N, float *P, int width, int height) {
    int row_o = blockIdx.y*TILE_WIDTH+threadIdx.y;
    int col_o = blockIdx.x*TILE_WIDTH+threadIdx.x;
	
    int row_i = row_o - KERNEL_SIZE/2;
    int col_i = col_o - KERNEL_SIZE/2;

	float output = 0.0f;
    __shared__ float N_ds[BLOCK_SIZE][BLOCK_SIZE];

	if ((row_i < height) && (col_i < width)) {
		N_ds[threadIdx.y][threadIdx.x] = N[row_i * width + col_i];
    } else {
		// padding
		N_ds[threadIdx.y][threadIdx.x] = 0.0f;
	}

	__syncthreads();

	// convolution
	if (threadIdx.y < TILE_WIDTH && threadIdx.x < TILE_WIDTH) {
		for (int i = 0; i < KERNEL_SIZE; i++) {
			for (int j = 0; j < KERNEL_SIZE; j++) {
				output += Mc[i][j] * N_ds[i+threadIdx.y][j+threadIdx.x];
			}
		}
		if (row_o < height-KERNEL_SIZE+1 && col_o < width-KERNEL_SIZE+1) {
			P[row_o * (width-KERNEL_SIZE+1) + col_o] = output;
		}
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
						results[r*Columns + c] += (M[h*KERNEL_SIZE + w] * N[row_i*Columns + col_i]);
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
		}
	}

	if (equal) {
		printf("Results are equal!\n");
	}
	else {
		printf("Results are NOT equal!\n");
	}

	free(results);
	return;
}


int main() {
	int width = 256;

	cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	cudaEventRecord(start,0);

	float *h_M = (float*)malloc(width*width*sizeof(float));
	float *h_N = (float*)malloc(KERNEL_SIZE*KERNEL_SIZE*sizeof(float));
	float *h_P = (float*)malloc((width-KERNEL_SIZE+1)*(width-KERNEL_SIZE+1)*sizeof(float));

	float *d_N, *d_P;
	cudaMalloc((void**)&d_N, width*width*sizeof(float));
	// cudaMalloc((void**)&d_M, KERNEL_SIZE*KERNEL_SIZE*sizeof(float));
	cudaMalloc((void**)&d_P, (width-KERNEL_SIZE+1)*(width-KERNEL_SIZE+1)*sizeof(float));

	cudaMemcpy(d_N, h_N, width*width*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(Mc, h_M, KERNEL_SIZE*KERNEL_SIZE*sizeof(float));

	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(ceil((width - KERNEL_SIZE + 1) / (float)TILE_WIDTH), ceil((width - KERNEL_SIZE + 1) / (float)TILE_WIDTH));


	conv2D<<<dimGrid, dimBlock>>>(d_N, d_P, width, width);

	cudaEventRecord(end, 0);
	cudaEventSynchronize(end);

	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(error));
		return 1;
	}
	float time_ms = 0;
	cudaEventElapsedTime(&time_ms, start, end);
	
	cudaMemcpy(d_P, h_P, (width-KERNEL_SIZE+1)*(width-KERNEL_SIZE+1)*sizeof(float), cudaMemcpyDeviceToHost);
	printf("Execution time for kernel: %.2f ms\n", time_ms);

	verification(h_N, h_M, h_P, width - KERNEL_SIZE + 1, width - KERNEL_SIZE + 1);

	free(h_M);
    free(h_N);
    free(h_P);


	return 0;
}