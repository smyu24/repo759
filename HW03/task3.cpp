

// There is no relationship between the array length and the threshold.

// ts = 2^1, 2^2, … 2^10

#include <iostream>
#include <cstdlib>
#include "msort.h"
#include <vector>

#include <chrono>
#include <ratio>
#include <random>

int main(int argc, char *argv[])
{
    if (argc != 4)
    {
        return 1;
    }

    int n = std::atoi(argv[1]);
    int t = std::atoi(argv[2]);
    int ts = std::atoi(argv[3]);

    // where n is a positive integer, t is an integer in the range [1, 20], ts is the
    //  threshold as the lower limit to make recursive calls
    //  n and m must be a positive integer
    if (n <= 0 || t < 1 || t > 20)
    {
        return 1;
    }


    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis1(-1000, 1000);

    int *arr = (int *)calloc(n, sizeof(int));

    // beyween [-1000,1000]
    for (int i = 0; i < n; ++i)
    {
        // generate random float domains: [-1000,1000]
        arr[i] = dis1(gen);
        // std::cout << imageMatrix[i] << " " << i << std::endl;
    }

    // chronos to get time taken to run scan function
    std::chrono::high_resolution_clock::time_point start;
    std::chrono::high_resolution_clock::time_point end;
    std::chrono::duration<double, std::milli> duration_sec;

    omp_set_num_threads(t);

    // printf("executing...%i\n", t);
    start = std::chrono::high_resolution_clock::now();
    msort(arr, n, ts); //int *arr, const std::size_t n, const std::size_t threshold
    end = std::chrono::high_resolution_clock::now();
    duration_sec = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start);

    std::cout << arr[0] << std::endl
              << arr[n - 1] << std::endl
              << duration_sec.count() << std::endl;

    free(arr);
}