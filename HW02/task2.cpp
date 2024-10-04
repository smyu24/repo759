// b) Write a file task2.cpp with a main function which (in this order)
// i) Creates an n×n image matrix (stored in 1D in row-major order) of random float numbers
// between -10.0 and 10.0. The value of n should be read as the first command line argument.
// ii) Creates an m×m mask matrix (stored in 1D in row-major order) of random float numbers
// between -1.0 and 1.0. The value of m should be read as the second command line argument.
// iii) Applies the mask to image using your convolve function.
// iv) Prints out the time taken by your convolve function in milliseconds.
// v) Prints the first element of the resulting convolved array.
// vi) Prints the last element of the resulting convolved array.
// vii) Deallocates memory when necessary via the delete function.

#include <iostream>
#include <cstdlib>
#include "convolution.h"
#include <vector>

#include <chrono>
#include <ratio>
#include <random>

int main(int argc, char *argv[])
{
    // argc == 3 for n and m
    if (argc != 3)
    {
        return 1;
    }

    int n = std::atoi(argv[1]);
    int m = std::atoi(argv[2]);

    // n and m must be a positive integer
    if (n <= 0 && m <= 0)
    {
        return 1;
    }

    // m must be odd number
    if (m % 2 == 0)
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
    for (int i = 0; i < n; ++i)
    {
        // generate random float domains: [-10,10]
        imageMatrix[i] = dis1(gen);
        // std::cout << imageMatrix[i] << " " << i << std::endl;
    }

    // between [-1,1]
    for (int i = 0; i < m; ++i)
    {
        // generate random float domains: [-1,1]
        mask[i] = dis2(gen);
        // std::cout << mask[i] << " " << i << std::endl;
    }
    // chronos to get time taken to run scan function
    std::chrono::high_resolution_clock::time_point start;
    std::chrono::high_resolution_clock::time_point end;
    std::chrono::duration<double, std::milli> duration_sec;

    start = std::chrono::high_resolution_clock::now();
    convolve(imageMatrix.data(), output, n, mask, m);
    end = std::chrono::high_resolution_clock::now();

    duration_sec = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start);

    std::cout << duration_sec.count() << std::endl
              << output[0] << std::endl
              << output[n*n-1] << std::endl;
}