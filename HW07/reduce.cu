#include <cuda_runtime.h>
#include "reduce.cuh"

#include <iostream>

__global__ void reduce_kernel(float *g_idata, float *g_odata, unsigned int n)
{
    extern __shared__ float sh_data[];
    unsigned int t_id = threadIdx.x;
    unsigned int index = (blockIdx.x * blockDim.x) * 2 + threadIdx.x;

    // add first when load

    // load to shared
    float sum = 0.0f;
    if (index < n)
    {
        sum += g_idata[index];
    }

    if (index + blockDim.x < n)
    {
        sum += g_idata[index + blockDim.x];
    }

    sh_data[t_id] = sum;
    __syncthreads();

    // reduction from clas
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (t_id < stride)
        {
            sh_data[t_id] += sh_data[t_id + stride];
        }
        __syncthreads();
    }

    //--> glob mem
    if (t_id == 0)
    {
        g_odata[blockIdx.x] = sh_data[0];
    }
}

__host__ void reduce(float **input, float **output, unsigned int N, unsigned int threads_per_block)
{
    unsigned int blocks = (N + threads_per_block * 2 - 1) / (threads_per_block * 2);
    unsigned int sh_mem_size = threads_per_block * sizeof(float);

    //looped to be until final sum of the entire array is obtained
    while (N > 1)
    {
        reduce_kernel<<<blocks, threads_per_block, sh_mem_size>>>(*input, *output, N);
        cudaDeviceSynchronize();

        // for next iter
        float *temp = *input;
        *input = *output;
        *output = temp;
        N = blocks;
        blocks = (N + threads_per_block * 2 - 1) / (threads_per_block * 2);
    }
}
