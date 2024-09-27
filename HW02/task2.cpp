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
// TEST OUT PREINCREMENT FOR CONVOL

#include <iostream>
#include <cstdlib>
#include <ctime>
#include "convolution.cpp"
#include <vector>

#include <chrono>
#include <ratio>

int main(int argc, char *argv[])
{
    // argc == 3 for n and m
    if (argc < 3)
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
    if ((m - 1) / 2 % 1)
    {
        return 1;
    }

    std::srand(static_cast<unsigned>(time(0)));

    // HAA
    std::vector<float> imageMatrix(n*n, 0.0f);

    float mask[m * m];
    float output[n * n];

    // beyween [-10,10]
    for (int i = 0; i < n; ++i)
    {
        // generate random float domains: [0,1] + [0,1] then - 1 and save (randomization code inspired from https://www.geeksforgeeks.org/generate-a-random-float-number-in-cpp/#)
        imageMatrix[i] = (static_cast<float>(rand()) / static_cast<float>(RAND_MAX) + static_cast<float>(rand()) / static_cast<float>(RAND_MAX)) - 1;
        std::cout << imageMatrix[i] << " " << i << std::endl;
    }

    // between [-1,1]
    for (int i = 0; i < m; ++i)
    {
        // generate random float domains: [0,1] + [0,1] then - 1 and save (randomization code inspired from https://www.geeksforgeeks.org/generate-a-random-float-number-in-cpp/#)
        mask[i] = ((rand() % (20)) + static_cast<float>(rand()) / static_cast<float>(RAND_MAX)) - 10;
        std::cout << mask[i] << " " << i << std::endl;
    }
    // chronos to get time taken to run scan function
    std::chrono::high_resolution_clock::time_point start;
    std::chrono::high_resolution_clock::time_point end;
    std::chrono::duration<double, std::milli> duration_sec;

    start = std::chrono::high_resolution_clock::now();
    // convolve(imageMatrix, output, n, mask, m);
    end = std::chrono::high_resolution_clock::now();

    duration_sec = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start);

    // output delimited by comma
    // std::cout << duration_sec.count() << ","
    //           << output[0] << ","
    //           << output[n] << "," << n << std::endl;
}

//     i) Creates an array of n random float numbers between -1.0 and 1.0. n should be read as the first command line argument as below.
//     ii) Scans the array using your scan function.
//     iii) Prints out the time taken by your scan function in milliseconds2.
//     iv) Prints the first element of the output scanned array.
//     v) Prints the last element of the output scanned array.
//     vi) Deallocates memory when necessary.