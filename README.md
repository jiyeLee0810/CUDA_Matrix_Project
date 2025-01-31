# CUDA Matrix

## CUDA Parallel Reduction
This project implements **Parallel Reduction** using CUDA to efficiently sum large arrays.
The reduction process leverages shared memory, thread synchronization, and multiple optimization techniques to achieve high performance.

### Features
- Implements multiple versions of reduction kernels.
- Optimizes memory access and thread usage.
- Uses **kernel decomposition** to avoid global synchronization.
- Achieves significant performance improvements using **loop unrolling**.

## CUDA Matrix Multiplication
This project implements **tiled dense matrix multiplication** using **shared memory** in CUDA.
This implementation optimizes memory access patterns and improves performance compared to naive global memory access.

### Features
- **Tiling Strategy:** Uses shared memory to optimize memory access.
- **Thread-Level Parallelism:** Each thread computes a portion of the matrix product.

## CUDA 2D Convolution
This project implements **2D convolution** using CUDA, a fundamental operation in image and signal processing. The implementation applies a **3x3 convolution kernel** to a 2D matrix in parallel, optimizing memory access with **constant memory and tiling**.

### Features
- **Parallel Convolution Computation** – Each thread computes an output pixel.
- **Tiled Memory Optimization** – Uses shared memory for efficient data access.
- **Constant Memory for Kernel Mask** – Reduces redundant global memory reads.
- **Boundary Handling** – Ensures correct computation near matrix edges.

## CUDA 3D Convolution