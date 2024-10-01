#include "matmul.h"

// Each function produces a row-major representation of the matrix C = A B.
// Details on the expected representation and order of operations within the
// function are given in the task1 description. The matrices A, B, and C are n
// by n and represented as 1D arrays or vectors.
void mmul1(const double *A, const double *B, double *C, const unsigned int n)
{
    // sweeps i through rows of C
    for (size_t i = 0; i < n; i++)
    {
        // sweeps j through columns of C
        for (size_t j = 0; j < n; j++)
        {
            // stride size n
            *(C + i * n + j) = 0;
            // sweeps index k through the dot product of the ith rowA w/ jth column B
            // carry out the dot product of the ith row, jth column
            for (size_t k = 0; k < n; k++)
            {
                // pointer arith
                *(C + i * n + j) += (*(A + i * n + k)) * (*(B + k * n + j));
                // C += A X B
                // increments C_ij
            }
        }
    }
    /*
    a) mmul1 should have three for loops: the outer loop sweeps index i through the rows of C, the
middle loop sweeps index j through the columns of C, and the innermost loop sweeps index k
through; i.e., to carry out, the dot product of the ith row A with the jth column of B. Inside
the innermost loop, you should have a single line of code which increments Cij . Assume that
A and B are 1D arrays storing the matrices in row-major order.
*/
}
void mmul2(const double *A, const double *B, double *C, const unsigned int n)
{
    for (size_t i = 0; i < n; i++)
    {
        for (size_t k = 0; k < n; k++)
        {
            // stride size n
            // *(C + i * n + j) = 0;
            for (size_t j = 0; j < n; j++)
            {
                // pointer arith
                *(C + i * n + j) += (*(A + i * n + k)) * (*(B + k * n + j));
                // C += A X B
                // increments C_ij
            }
        }
    }
    /*
    b) mmul2 should also have three for loops, but the two innermost loops should be swapped
relative to mmul1 (such that, if your original iterators are from outer to inner (i,j,k), then
they now become (i,k,j)). That is the only difference between mmul1 and mmul2.
*/
}
void mmul3(const double *A, const double *B, double *C, const unsigned int n)
{
    for (size_t j = 0; j < n; j++)
    {
        for (size_t k = 0; k < n; k++)
        {
            for (size_t i = 0; i < n; i++)
            {
                // pointer arith
                *(C + i * n + j) += (*(A + i * n + k)) * (*(B + k * n + j));
                // C += A X B
                // increments C_ij
            }
        }
    }
    /*
    c) mmul3 should also have three for loops, but the outermost loop in mmul1 should become the
innermost loop in mmul3, and the other 2 loops do not change their relative positions (such that,
if your original iterators are from outer to inner (i,j,k), then they now become (j,k,i)).
That is the only difference between mmul1 and mmul3.
*/
}
void mmul4(const std::vector<double> &A, const std::vector<double> &B, double *C, const unsigned int n)
{
    // sweeps i through rows of C
    for (size_t i = 0; i < n; i++)
    {
        // sweeps j through columns of C
        for (size_t j = 0; j < n; j++)
        {
            // sweeps index k through the dot product of the ith rowA w/ jth column B
            // carry out the dot product of the ith row, jth column,
            for (size_t k = 0; k < n; k++)
            {
                *(C + i * n + j) += A[i * n + k] * B[k * n + j]; // (*(A + i * n + k)) * (*(B + k * n + j));
                // C += A X B
                // increments C_ij
            }
        }
    }
    /*
    d) mmul4 should have the for loops ordered as in mmul1, but this time around A and B are stored
as std::vector<double>. That is the only difference between mmul1 and mmul4.
*/
}
