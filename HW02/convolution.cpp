#include "convolution.h"

// Computes the result of applying a mask to an image as in the convolution process described in HW02.pdf.
// image is an nxn grid stored in row-major order.
// mask is an mxm grid stored in row-major order.
// Stores the result in output, which is an nxn grid stored in row-major order.
void convolve(const float *image, float *output, std::size_t n, const float *mask, std::size_t m)
{
    // check for emptyness
    //output


    // take advantage of contiguous in memory to access row order array with 1D array
    /* original image is nxn dimension */
    for (size_t x = 0; x < n; x++)
    {
        for (size_t y = 0; y < n; y++)
        {
            float f = 0.0; // original img
            float w = 0.0; // mask
            /* nested summation */
            for (size_t i = 0; i <= m - 1; i++)
            {
                for (size_t j = 0; j <= m - 1; j++)
                {
                    // check for 0 <= i/j < n (both condition failed) check later
                    if ((0 <= i && i < n) && (0 <= j && j < n))
                    {
                        f = 0.0;
                    }
                    // check for either (one condition failed)
                    else if ((0 <= i && i < n) || (0 <= j && j < n))
                    {
                        f = 1.0;
                    }
                    else
                    { 
                        f = image[(x + i - (m - 1) / 2)*n + (y + j - (m - 1) / 2)];
                    }
                    w = mask[i * m + j];
                }
            }
            // nxn
            output[x * n + y] = w * f;
        }
    }
}
