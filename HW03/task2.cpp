
#include <iostream>
#include <cstdlib>
#include "convolution.h"
#include <vector>

#include <chrono>
#include <ratio>
#include <random>

int main(int argc, char *argv[])
{
    // argc == 3 for n and t
    if (argc != 3)
    {
        return 1;
    }

    int n = std::atoi(argv[1]);
    int t = std::atoi(argv[2]);

    int m = 3;

    // n and m must be a positive integer
    if (n <= 0 && t <= 0)
    {
        return 1;
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis1(-10.0, 10.0);
    std::uniform_real_distribution<float> dis2(-1.0, 1.0);

    std::vector<float> imageMatrix(n * n, 0.0f); // zero init values

    // SAA
    float mask[m * m];
    float output[n * n];

    // beyween [-10,10]
    for (int i = 0; i < n * n; ++i)
    {
        // generate random float domains: [-10,10]
        imageMatrix[i] = dis1(gen);
        // std::cout << imageMatrix[i] << " " << i << std::endl;
    }

    // between [-1,1]
    for (int i = 0; i < m * m; ++i)
    {
        // generate random float domains: [-1,1]
        mask[i] = dis2(gen);
        // std::cout << mask[i] << " " << i << std::endl;
    }

    // chronos to get time taken to run scan function
    std::chrono::high_resolution_clock::time_point start;
    std::chrono::high_resolution_clock::time_point end;
    std::chrono::duration<double, std::milli> duration_sec;

    for (int i = 1; i <= t; i++)
    {
        // thread nums to iterate from 1 to 20
        omp_set_num_threads(i);

        start = std::chrono::high_resolution_clock::now();
        convolve(imageMatrix.data(), output, n, mask, m); // parallelized function
        end = std::chrono::high_resolution_clock::now();
        duration_sec = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start);

        std::cout << duration_sec.count() << std::endl
              << output[0] << std::endl
              << output[n * n - 1] << std::endl;
    }
}