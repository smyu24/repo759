#include "convolution.h"

// Computes the result of applying a mask to an image as in the convolution process described in HW02.pdf.
// image is an nxn grid stored in row-major order.
// mask is an mxm grid stored in row-major order.
// Stores the result in output, which is an nxn grid stored in row-major order.
void convolve(const float *image, float *output, std::size_t n, const float *mask, std::size_t m)
{
    // no need to check for emptyness

    // take advantage of contiguous in memory to access row order array with 1D array
    /* original image is nxn dimension */
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
                    int xValue = static_cast<int>(x) + static_cast<int>(i) - ((m - 1) / 2); // x+i-(m-1)/2
                    int yValue = static_cast<int>(y) + static_cast<int>(j) - ((m - 1) / 2); // y+j-(m-1)/2

                    // if there exists a condition that is not satisfied
                    if (xValue < 0 || xValue >= static_cast<int>(n) || yValue < 0 || yValue >= static_cast<int>(n))
                    {
                        if ((xValue < 0 || xValue >= static_cast<int>(n)) && (yValue < 0 || yValue >= static_cast<int>(n)))
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
                        f = image[xValue * n + yValue];
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