#include "scan.h"
#include <iostream>

// 1. a) Implement the scan function in a file called scan.cpp with signature defined as in scan.h.
// You should write an inclusive scan1 on your own and not use any library scan functions.

void scan(const float *arr, float *output, std::size_t n)
{
    if (n == 0)
    {
        return;
    }

    output[0] = arr[0];

    for (std::size_t i = 1; i < n; i++)
    {
        output[i] = arr[i] + output[i - 1];
        // std::cout << "output[] " << output[i] << " = " << arr[i] << " + " << output[i-1] << std::endl;
    }
    return;
}

// Performs an inclusive scan on input array arr and stores
// the result in the output array
// arr and output are arrays of n elements