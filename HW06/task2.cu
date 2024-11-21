#include <cstdio>
#include <cstdlib>
#include <random>
#include <cuda_runtime.h>
#include "stencil.cuh"

int main(int argc, char **argv)
{
    if (argc != 4)
    {
        return 1;
    }

    int n = std::atoi(argv[1]);
    int R = std::atoi(argv[2]);
    int threads_per_block = std::atoi(argv[3]);

    // lengths
    float *image = new float[n];
    float *mask = new float[2 * R + 1];
    float *output = new float[n];

    // rand
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0, 1.0);

    for (int i = 0; i < n; i++)
    {
        image[i] = dis(gen);
    }
    for (int i = 0; i < 2 * R + 1; i++)
    {
        mask[i] = dis(gen);
    }

    // memalloc/set
    float *dImage, *dMask, *dOutput;
    cudaMalloc(&dImage, n * sizeof(float));
    cudaMalloc(&dMask, (2 * R + 1) * sizeof(float));
    cudaMalloc(&dOutput, n * sizeof(float));

    cudaMemcpy(dImage, image, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dMask, mask, (2 * R + 1) * sizeof(float), cudaMemcpyHostToDevice);

    // timing
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // kernel call rec
    cudaEventRecord(start);
    stencil(dImage, dMask, dOutput, n, R, threads_per_block);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float duration_sec = 0;
    cudaEventElapsedTime(&duration_sec, start, stop);
    cudaMemcpy(output, dOutput, n * sizeof(float), cudaMemcpyDeviceToHost);

    printf("%f\n", output[n - 1]);
    printf("%f\n", duration_sec);

    delete[] image;
    delete[] output;
    delete[] mask;
    cudaFree(dImage);
    cudaFree(dMask);
    cudaFree(dOutput);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}