#include "matmul.cuh"
#include <stdio.h>

// Computes the convolution of image and mask, storing the result in output.
// Each thread should compute _one_ element of the output matrix.
// Shared memory should be allocated _dynamically_ only.
//
// image is an array of length n.
// mask is an array of length (2 * R + 1).
// output is an array of length n.
// All of them are in device memory
//
// Assumptions:
// - 1D configuration
// - blockDim.x >= 2 * R + 1
//
// The following should be stored/computed in shared memory:
// - The entire mask
// - The elements of image that are needed to compute the elements of output corresponding to the threads in the given block
// - The output image elements corresponding to the given block before it is written back to global memory
__global__ void stencil_kernel(const float *image, const float *mask, float *output, unsigned int n, unsigned int R)
{
    extern __shared__ float sh_mem[]; // only dynamic ; i tihnk

    float *sh_image = sh_mem;
    float *sh_mask = &sh_mem[blockDim.x + 2 * R];
    int g_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // loading in block
    if (threadIdx.x < 2 * R + 1)
    {
        sh_mask[threadIdx.x] = mask[threadIdx.x];
    }

    int sh_idx = threadIdx.x + R;
    // i=0 til n-1
    if (g_idx < n)
    {
        sh_image[sh_idx] = image[g_idx];
    }
    else
    {
        sh_image[sh_idx] = 1.0;
    }

    // edgecase and assumption1
    if (threadIdx.x < R)
    {
        int left_idx = g_idx - R;
        int right_idx = blockDim.x + g_idx;

        //0 til n-1
        if (left_idx >= 0)
        {
            sh_image[threadIdx.x] = image[left_idx];
        }
        else
        {
            sh_image[threadIdx.x] = 1.0;
        }
        if (right_idx < n)
        {
            sh_image[sh_idx + blockDim.x] = image[right_idx];
        }
        else
        {
            sh_image[sh_idx + blockDim.x] = 1.0;
        }
    }
    // - The entire mask
    // - The elements of image that are needed to compute the elements of output corresponding to the threads in the given block
    // - The output image elements corresponding to the given block before it is written back to global memory
    __syncthreads();

    if (g_idx < n)
    {
        //sigma notation loop
        float temp = 0;
        for (int i = -R; i <= R; i++)
        {
            result += sh_image[sh_idx + i] * sh_mask[i + R];
        }

//s

        output[g_idx] = temp;
    }
}

// Makes one call to stencil_kernel with threads_per_block threads per block.
// You can consider following the kernel call with cudaDeviceSynchronize (but if you use
// cudaEventSynchronize to time it, that call serves the same purpose as cudaDeviceSynchronize).
//
// Assumptions:
// - threads_per_block >= 2 * R + 1
__host__ void stencil(const float *image,
                      const float *mask,
                      float *output,
                      unsigned int n,
                      unsigned int R,
                      unsigned int threads_per_block)
{

    // assumption #1
    if (threads_per_block < 2 * R + 1)
    {
        return;
    }

    int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;
    size_t sh_size = (threads_per_block + 2 * R) * sizeof(float) + (2 * R + 1) * sizeof(float);

    stencil_kernel<<<blocks_per_grid, threads_per_block, sh_size>>>(image, mask, output, n, R);
    cudaDeviceSynchronize();
}