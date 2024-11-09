#include "vscale.cuh"
#include <stdio.h>

__global__ void vscale(const float *a, float *b, unsigned int n)
{
    int globalIndx = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalIndx < n)
    {
        b[globalIndx] *= a[globalIndx];
    }
}