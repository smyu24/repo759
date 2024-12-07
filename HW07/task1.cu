#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include "matmul.cuh"
#include <random>

int main(int argc, char **argv) {
    if (argc != 3) {
        return -1;
    }

    unsigned int n = std::atoi(argv[1]);
    unsigned int block_dim = std::atoi(argv[2]);

    int *A = new int[n * n];
    int *B = new int[n * n];
    int *C = new int[n * n]; // check if zero init

    // rand
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(-10, 10);

    for (unsigned int i = 0; i < n * n; ++i)
    {
        A[i] = dis(gen);
        B[i] = dis(gen);
    }

    // memalloc/set
    int *dA, *dB, *dC;
    cudaMalloc(&dA, n * n * sizeof(int));
    cudaMalloc(&dB, n * n * sizeof(int));
    cudaMalloc(&dC, n * n * sizeof(int));

    cudaMemcpy(dA, A, n * n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, n * n * sizeof(int), cudaMemcpyHostToDevice);
    // cudaMemcpy(dC, C, n*n * sizeof(float), cudaMemcpyHostToDevice);

    // timing
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // kernel call rec
    cudaEventRecord(start);
    matmul_1(dA, dB, dC, n, block_dim);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float duration_sec = 0;
    cudaEventElapsedTime(&duration_sec, start, stop);
    cudaMemcpy(C, dC, n * n * sizeof(int), cudaMemcpyDeviceToHost);

    printf("%i\n", C[0]);
    printf("%i\n", C[n * n - 1]);
    printf("%f\n", duration_sec);

    delete[] A;
    delete[] B;
    delete[] C;
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);




    float *A2 = new float[n * n];
    float *B2 = new float[n * n];
    float *C2 = new float[n * n]; // check if zero init

    // rand
    std::random_device rd2;
    std::mt19937 gen2(rd2());
    std::uniform_real_distribution<float> dis2(-10.0, 10.0);

    for (unsigned int i = 0; i < n * n; ++i)
    {
        A2[i] = dis2(gen2);
        B2[i] = dis2(gen2);
    }

    // memalloc/set
    float *dA2, *dB2, *dC2;
    cudaMalloc(&dA2, n * n * sizeof(float));
    cudaMalloc(&dB2, n * n * sizeof(float));
    cudaMalloc(&dC2, n * n * sizeof(float));

    cudaMemcpy(dA2, A2, n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB2, B2, n * n * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(dC, C, n*n * sizeof(float), cudaMemcpyHostToDevice);

    // timing
    cudaEvent_t start2;
    cudaEvent_t stop2;
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);

    // kernel call rec
    cudaEventRecord(start2);
    matmul_2(dA2, dB2, dC2, n, block_dim);
    cudaEventRecord(stop2);
    cudaEventSynchronize(stop2);

    duration_sec = 0;
    cudaEventElapsedTime(&duration_sec, start2, stop2);
    cudaMemcpy(C2, dC2, n * n * sizeof(float), cudaMemcpyDeviceToHost);

    printf("%f\n", C2[0]);
    printf("%f\n", C2[n * n - 1]);
    printf("%f\n", duration_sec);

    delete[] A2;
    delete[] B2;
    delete[] C2;
    cudaFree(dA2);
    cudaFree(dB2);
    cudaFree(dC2);
    cudaEventDestroy(start2);
    cudaEventDestroy(stop2);


    double *A3 = new double[n * n];
    double *B3 = new double[n * n];
    double *C3 = new double[n * n]; // check if zero init

    // rand
    std::random_device rd3;
    std::mt19937 gen3(rd3());
    std::uniform_real_distribution<double> dis3(-10.0, 10.0);

    for (unsigned int i = 0; i < n * n; ++i)
    {
        A3[i] = dis3(gen3);
        B3[i] = dis3(gen3);
    }

    // memalloc/set
    double *dA3, *dB3, *dC3;
    cudaMalloc(&dA3, n * n * sizeof(double));
    cudaMalloc(&dB3, n * n * sizeof(double));
    cudaMalloc(&dC3, n * n * sizeof(double));

    cudaMemcpy(dA3, A3, n * n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dB3, B3, n * n * sizeof(double), cudaMemcpyHostToDevice);
    // cudaMemcpy(dC, C, n*n * sizeof(float), cudaMemcpyHostToDevice);

    // timing
    cudaEvent_t start3;
    cudaEvent_t stop3;
    cudaEventCreate(&start3);
    cudaEventCreate(&stop3);

    // kernel call rec
    cudaEventRecord(start3);
    matmul_3(dA3, dB3, dC3, n, block_dim);
    cudaEventRecord(stop3);
    cudaEventSynchronize(stop3);

    duration_sec = 0;
    cudaEventElapsedTime(&duration_sec, start3, stop3);
    cudaMemcpy(C3, dC3, n * n * sizeof(double), cudaMemcpyDeviceToHost);

    printf("%f\n", C3[0]);
    printf("%f\n", C3[n * n - 1]);
    printf("%f\n", duration_sec);

    delete[] A3;
    delete[] B3;
    delete[] C3;
    cudaFree(dA3);
    cudaFree(dB3);
    cudaFree(dC3);
    cudaEventDestroy(start3);
    cudaEventDestroy(stop3);

    return 0;
}
