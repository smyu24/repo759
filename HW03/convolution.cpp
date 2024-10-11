#include "convolution.h"


// This function does a parallel version of the convolution process in HW02 task2
// using OpenMP. You may recycle your code from HW02.

// "image" is an n by n grid stored in row-major order.
// "mask" is an m by m grid stored in row-major order.
// "output" stores the result as an n by n grid in row-major order.
void convolve(const float *image, float *output, std::size_t n, const float *mask, std::size_t m)
{
// no need to check for emptyness
    // take advantage of contiguous in memory to access row order array with 1D array
    /* original image is nxn dimension */
    #pragma omp parallel for collapse(2) // no data dependency so collapse
    for (size_t x = 0; x < n; x++)
    {
        for (size_t y = 0; y < n; y++)
        {
            float f = 0.0; // original img
            float w = 0.0; // mask
            float convolutionResult = 0.0;
            /* nested summation */
            for (size_t i = 0; i <= m - 1; i++)
            {
                for (size_t j = 0; j <= m - 1; j++)
                {
                    int iValue = static_cast<int>(x) + static_cast<int>(i) - ((m - 1) / 2); // x+i-(m-1)/2
                    int jValue = static_cast<int>(y) + static_cast<int>(j) - ((m - 1) / 2); // y+j-(m-1)/2

                    // if there exists a condition that is not satisfied
                    if (iValue < 0 || iValue >= static_cast<int>(n) || jValue < 0 || jValue >= static_cast<int>(n))
                    {
                        if ((iValue < 0 || iValue >= static_cast<int>(n)) && (jValue < 0 || jValue >= static_cast<int>(n)))
                        {
                            // corner case (both condition failed)
                            f = 0.0;
                        }
                        else
                        {
                            // edge case (one condition failed)
                            f = 1.0;
                        }
                    }
                    else
                    {
                        // f[x+i-(m-1)/2,y+i-(m-1)/2]
                        f = image[iValue * n + jValue];
                    }
                    w = mask[i * m + j]; // w[i, j]
                    convolutionResult += f * w;
                }
            }
            // nxn g[x,y] = summed
            output[x * n + y] = convolutionResult;
        }
    }
}