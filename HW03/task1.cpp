// In HW02 task3, you have implemented several different ways to carry out matrix multiplication
// in sequential computing. In this task, you have to write a function called mmul, which takes the
// mmul2 function in HW02 and parallelize it with OpenMP.

// b) Write a program task1.cpp that will accomplish the following:
// • Create and fill with float type numbers the square matrices A and B (with the data
// range and format specified in the description of HW02 task3; if the range is not explic-
// itly given then you should populate the matrices however you like). The dimension of
// A and B should be n×n where n is the first command line argument, see below.
// • Compute the matrix multiplication C = AB using your parallel implementation with
// t threads, where t is the second command line argument, see below.
// • Print the first element of the resulting C array.
// • Print the last element of the resulting C array.
// • Print the time taken to run the mmul function in milliseconds.

// c) On Euler , via Slurm do the following:
// • Run task1 for value n = 1024, and value t = 1, 2, · · · , 20. Generate a plot called
// task1.pdf which plots time taken by your mmul function vs. t in linear-linear scale.

#include <iostream>
#include <cstdlib>
#include "matmul.h"
#include <vector>

#include <chrono>
#include <ratio>
#include <random>

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        return 1;
    }

    int n = std::atoi(argv[1]);
    int t = std::atoi(argv[2]);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis1(-10.0, 10.0);

    float *A = (float *)malloc(sizeof(float) * (n * n));
    float *B = (float *)malloc(sizeof(float) * (n * n));
    float *C = (float *)malloc(sizeof(float) * (n * n));

    // beyween [-10,10]
    for (int i = 0; i < n * n; ++i)
    {
        // generate random float domains: [-10,10]
        A[i] = dis1(gen);
        B[i] = dis1(gen);
    }

    // chronos to get time taken to run scan function
    std::chrono::high_resolution_clock::time_point start;
    std::chrono::high_resolution_clock::time_point end;
    std::chrono::duration<double, std::milli> duration_sec;

    volatile double warmup = 0.0;
    // warmup cache
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            warmup += A[i * n + j];
            warmup += B[j * n + i];
        }
    }

    // Initialize C to zero
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            C[i * n + j] = 0;
        }
    }

    for (int i = 1; i <= t; i++)
    {
        // thread nums to iterate from 1 to 20
        omp_set_num_threads(i);

        start = std::chrono::high_resolution_clock::now();
        mmul(A, B, C, n);
        end = std::chrono::high_resolution_clock::now();
        duration_sec = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start);

        std::cout << C[0] << std::endl
                  << C[n * n - 1] << std::endl
                  << duration_sec.count() << std::endl;
    }

    //     for (int i = 1; i <= t; i++)
    //     {
    // // fork join start
    // #pragma omp parallel num_threads(t)
    //         {
    //             // master thread - may have to ustilize atomic
    // #pragma omp master
    //             {
    //                 start = std::chrono::high_resolution_clock::now();
    //                 mmul(A, B, C, n);
    //                 end = std::chrono::high_resolution_clock::now();
    //                 duration_sec = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start);
    //                 std::cout << C[0] << std::endl
    //                           << C[n * n - 1] << std::endl
    //                           << duration_sec.count() << std::endl;
    //             }
    //         }
    //     }

    // free all remainging heap allocs
    free(A);
    free(B);
    free(C);
}