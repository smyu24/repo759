#include <cuda_runtime.h>
#include "matmul.cuh"
#include <iostream>

template <typename T>
__global__ void matmul_kernel(const T *A, const T *B, T *C, unsigned int n, unsigned int block_dim)
{
    extern __shared__ char sh_mem[];
    T *tile_A = reinterpret_cast<T *>(sh_mem);
    T *tile_B = reinterpret_cast<T *>(sh_mem) + block_dim * block_dim;

    // calc thread indx
    unsigned int row = threadIdx.y + blockIdx.y * block_dim;
    unsigned int col = threadIdx.x + blockIdx.x * block_dim;

    T value = 0;

    // load ino shared mem
    for (unsigned int i = 0; i < (n + block_dim - 1) / block_dim; i++)
    {
        if (row < n && (i * block_dim + threadIdx.x) < n)
        {
            tile_A[threadIdx.y * block_dim + threadIdx.x] = A[row * n + i * block_dim + threadIdx.x];
        }
        else
        {
            tile_A[threadIdx.y * block_dim + threadIdx.x] = 0;
        }

        if (col < n && (i * block_dim + threadIdx.y) < n)
        {
            tile_B[threadIdx.y * block_dim + threadIdx.x] = B[(i * block_dim + threadIdx.y) * n + col];
        }
        else
        {
            tile_B[threadIdx.y * block_dim + threadIdx.x] = 0;
        }

        // sync point
        __syncthreads();

        for (unsigned int i = 0; i < block_dim; i++)
        {
            value += tile_A[threadIdx.y * block_dim + i] * tile_B[i * block_dim + threadIdx.x];
        }
        __syncthreads();
    }
    // glob mem
    if (row < n && col < n)
    {
        C[row * n + col] = value;
    }
}

template <typename T>
void matmul(const T *A, const T *B, T *C, unsigned int n, unsigned int block_dim)
{
    // block and grid dimensions
    dim3 threads(block_dim, block_dim);
    dim3 blocks((n + block_dim - 1) / block_dim, (n + block_dim - 1) / block_dim);
    size_t sh_mem_size = 2 * (block_dim * block_dim) * sizeof(T);

    matmul_kernel<T><<<blocks, threads, sh_mem_size>>>(A, B, C, n, block_dim);
    cudaDeviceSynchronize();
}

// Only difference is the type
__host__ void matmul_1(const int *A, const int *B, int *C, unsigned int n, unsigned int block_dim)
{
    matmul(A, B, C, n, block_dim);
}

__host__ void matmul_2(const float *A, const float *B, float *C, unsigned int n, unsigned int block_dim)
{
    matmul(A, B, C, n, block_dim);
}

__host__ void matmul_3(const double *A, const double *B, double *C, unsigned int n, unsigned int block_dim)
{
    matmul(A, B, C, n, block_dim);
}
