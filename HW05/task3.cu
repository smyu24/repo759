#include <cstdio>
#include <cstdlib>
#include <random>
#include <cuda_runtime.h>
#include "vscale.cuh"

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        return 1;
    }

    int n = std::atoi(argv[1]);

    float *hA = new float[n];
    float *hB = new float[n];

    //rand
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> disA(-10.0, 10.0);
    std::uniform_real_distribution<float> disB(0.0, 1.0);

    for (int i = 0; i < n; ++i)
    {
        hA[i] = disA(gen);
        hB[i] = disB(gen);
    }

    //memalloc/set
    float *dA, *dB;
    cudaMalloc(&dA, n * sizeof(float));
    cudaMalloc(&dB, n * sizeof(float));
    cudaMemcpy(dA, hA, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, n * sizeof(float), cudaMemcpyHostToDevice);

    //timing
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int threadNumInBlock = 512;
    int blocksPerGrid = (n + threadNumInBlock - 1) / threadNumInBlock;

    //kernel call rec
    cudaEventRecord(start);
    vscale<<<blocksPerGrid, threadNumInBlock>>>(dA, dB, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float duration_sec = 0;
    cudaEventElapsedTime(&duration_sec, start, stop);
    cudaMemcpy(hB, dB, n * sizeof(float), cudaMemcpyDeviceToHost);

    printf("%f\n", duration_sec);
    printf("%f\n", hB[0]);
    printf("%f\n", hB[n - 1]);

    delete[] hA;
    delete[] hB;
    cudaFree(dA);
    cudaFree(dB);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
