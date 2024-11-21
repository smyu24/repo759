#include "matmul.cuh"
#include <stdio.h>

// Computes the matrix product of A and B, storing the result in C.
// Each thread should compute _one_ element of output.
// Does not use shared memory for this problem.
//
// A, B, and C are row major representations of nxn matrices in device memory.
//
// Assumptions:
// - 1D kernel configuration
__global__ void matmul_kernel(const float *A, const float *B, float *C, size_t n)
{
    //global idx
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    size_t cNum = idx % n;
    size_t rNum = idx / n;
    
    if (rNum < n && cNum < n) {
        float sum = 0.0;
        for (size_t i = 0; i < n; i++) {
            sum += A[rNum * n + i] * B[i * n + cNum];
        }
        C[rNum * n + cNum] = sum;
    }
}


// Makes one call to matmul_kernel with threads_per_block threads per block.
// You can consider following the kernel call with cudaDeviceSynchronize (but if you use 
// cudaEventSynchronize to time it, that call serves the same purpose as cudaDeviceSynchronize).
void matmul(const float* A, const float* B, float* C, size_t n, unsigned int threads_per_block)
{
    //calc blockspergrid here.also set threads
    int blocksPerGrid = ((n*n) + threads_per_block - 1) / threads_per_block; // this can be improved

    dim3 threads(threads_per_block, threads_per_block);
    dim3 blocks(blocksPerGrid, blocksPerGrid);

    matmul_kernel<<<blocksPerGrid, threads_per_block>>>(A, B, C, n);

    cudaDeviceSynchronize();
}