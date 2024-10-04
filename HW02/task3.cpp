// • generates square matrices A and B of dimension at least 1000×1000 stored in row-major
// order.
// • computes the matrix product C = AB using each of your functions (note that you may
// have to prepare A and B in different data types so they comply with the function argument
// types). Your result stored in matrix C should be the same no matter which function defined
// at a) through d) above you call .
// • prints the number of rows of your input matrices, and for each mmul function in ascending
// order, prints the amount of time taken in milliseconds and the last element of the resulting
// C. There should be nine values printed, one per line

#include <iostream>
#include <cstdlib>
#include "matmul.h"
#include <vector>

#include <chrono>
#include <ratio>
#include <random>

int main(int argc, char *argv[])
{
    if (argc != 1)
    {
        return 1;
    }

    const int n = 1024;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis1(-10.0, 10.0);

    double *A = (double *)malloc(sizeof(double) * (n * n));
    double *B = (double *)malloc(sizeof(double) * (n * n));
    double *C = (double *)malloc(sizeof(double) * (n * n));

    std::vector<double> vectorA(n * n, 0.0f);
    std::vector<double> vectorB(n * n, 0.0f);

    // beyween [-10,10]
    for (int i = 0; i < n * n; ++i)
    {
        // generate random float domains: [-10,10]
        A[i] = dis1(gen);
        vectorA[i] = A[i];
        B[i] = dis1(gen);
        vectorB[i] = B[i];
    }

    // chronos to get time taken to run scan function
    std::chrono::high_resolution_clock::time_point start;
    std::chrono::high_resolution_clock::time_point end;
    std::chrono::duration<double, std::milli> duration_sec1;
    std::chrono::duration<double, std::milli> duration_sec2;
    std::chrono::duration<double, std::milli> duration_sec3;
    std::chrono::duration<double, std::milli> duration_sec4;


    volatile double warmup = 0.0;
    // warmup cache 
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            warmup += A[i * n + j];
            warmup += B[j * n + i];
        }
    }

    start = std::chrono::high_resolution_clock::now();
    mmul1(A, B, C, n);
    end = std::chrono::high_resolution_clock::now();
    duration_sec1 = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start);
    std::cout << n << std::endl
              << duration_sec1.count() << std::endl
              << C[n * n - 1] << std::endl;
    free(C);
    C = (double *)malloc(sizeof(double) * (n * n));

    start = std::chrono::high_resolution_clock::now();
    mmul2(A, B, C, n);
    end = std::chrono::high_resolution_clock::now();
    duration_sec2 = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start);
    std::cout << duration_sec2.count() << std::endl
              << C[n * n - 1] << std::endl;
    free(C);
    C = (double *)malloc(sizeof(double) * (n * n));

    start = std::chrono::high_resolution_clock::now();
    mmul3(A, B, C, n);
    end = std::chrono::high_resolution_clock::now();
    duration_sec3 = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start);
    std::cout << duration_sec3.count() << std::endl
              << C[n * n - 1] << std::endl;
    free(C);
    C = (double *)malloc(sizeof(double) * (n * n));

    start = std::chrono::high_resolution_clock::now();
    mmul4(vectorA, vectorB, C, n);
    end = std::chrono::high_resolution_clock::now();
    duration_sec4 = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start);
    std::cout << duration_sec4.count() << std::endl
              << C[n * n - 1] << std::endl;

    // free all remainging heap allocs
    free(A);
    free(B);
    free(C);

}
