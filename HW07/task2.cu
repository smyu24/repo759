#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include "reduce.cuh"
#include <chrono>
#include <random>


int main(int argc, char **argv) {
    if (argc != 3) {
        return -1;
    }

    unsigned int N = std::atoi(argv[1]);
    unsigned int threads_per_block = std::atoi(argv[2]); 

    float *h_input = new float[N];\
    // rand
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0, 1.0);

    for (unsigned int i = 0; i < N; i++)
    {
        h_input[i] = dis(gen);
    }

    float *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    unsigned int blocks = (N + threads_per_block * 2 - 1) / (threads_per_block * 2);
    cudaMalloc(&d_output, blocks * sizeof(float));

    float *input = d_input;
    float *output = d_output;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    reduce(&input, &output, N, threads_per_block);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    float result;
    cudaMemcpy(&result, input, sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << result << "\n";
    std::cout << milliseconds << "\n";

    delete[] h_input;
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
