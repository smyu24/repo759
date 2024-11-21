#include <cstdio>
#include <cstdlib>
#include <random>
#include <cuda_runtime.h>
#include "matmul.cuh"

int main(int argc, char **argv)
{

    if (argc != 3)
    {
        return 1;
    }

    int n = std::atoi(argv[1]);
    int threads_per_block = std::atoi(argv[2]);

    float *A = new float[n * n];
    float *B = new float[n * n];
    float *C = new float[n * n]; // check if zero init

    // rand
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0, 1.0);

    for (int i = 0; i < n * n; ++i)
    {
        A[i] = dis(gen);
        B[i] = dis(gen);
    }

    // memalloc/set
    float *dA, *dB, *dC;
    cudaMalloc(&dA, n * n * sizeof(float));
    cudaMalloc(&dB, n * n * sizeof(float));
    cudaMalloc(&dC, n * n * sizeof(float));

    cudaMemcpy(dA, A, n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, n * n * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(dC, C, n*n * sizeof(float), cudaMemcpyHostToDevice);

    // timing
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // kernel call rec
    cudaEventRecord(start);
    matmul(dA, dB, dC, n * n, threads_per_block); // const float* A, const float* B, float* C, size_t n, unsigned int threads_per_block
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float duration_sec = 0;
    cudaEventElapsedTime(&duration_sec, start, stop);
    cudaMemcpy(C, dC, n * n * sizeof(float), cudaMemcpyDeviceToHost);

    printf("%f\n", C[n * n - 1]);
    printf("%f\n", duration_sec);

    delete[] A;
    delete[] B;
    delete[] C;
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
