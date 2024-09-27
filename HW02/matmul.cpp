#include "matmul.h"

// Each function produces a row-major representation of the matrix C = A B.
// Details on the expected representation and order of operations within the
// function are given in the task1 description. The matrices A, B, and C are n
// by n and represented as 1D arrays or vectors.
void mmul1(const double *A, const double *B, double *C, const unsigned int n)
{
    // row
    for (size_t i = 0; i < n; i++)
    {
        // column
        for (size_t j = 0; j < n; j++)
        {
            // carry out the dot product of the ith row, jth column,
            for (size_t k = 0; k < n; k++)
            {

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
    /*
    b) mmul2 should also have three for loops, but the two innermost loops should be swapped
relative to mmul1 (such that, if your original iterators are from outer to inner (i,j,k), then
they now become (i,k,j)). That is the only difference between mmul1 and mmul2.
*/
}
void mmul3(const double *A, const double *B, double *C, const unsigned int n)
{
    /*
    c) mmul3 should also have three for loops, but the outermost loop in mmul1 should become the
innermost loop in mmul3, and the other 2 loops do not change their relative positions (such that,
if your original iterators are from outer to inner (i,j,k), then they now become (j,k,i)).
That is the only difference between mmul1 and mmul3.
*/
}
void mmul4(const std::vector<double> &A, const std::vector<double> &B, double *C, const unsigned int n)
{
    /*
    d) mmul4 should have the for loops ordered as in mmul1, but this time around A and B are stored
as std::vector<double>. That is the only difference between mmul1 and mmul4.
*/
}
