// This function does a merge sort on the input array "arr" of length n.
// You can add more functions as needed to complete the merge sort,
// but do not change this file. Declare and define your addtional
// functions in the msort.cpp file, but the calls to your addtional functions
// should be wrapped in the "msort" function.

// "threshold" is the lower limit of array size where your function would
// start making parallel recursive calls. If the size of array goes below
// the threshold, a serial sort algorithm will be used to avoid overhead
// of task scheduling

#include <iostream>
#include <cstdlib>
#include "msort.h"
#include <ratio>
#include <random>
#include <algorithm>
void merge(int *arr, int *leftArr, int sizeOfLeft, int *rightArr, int sizeOfRight)
{
    int i = 0, j = 0, k = 0;

    // Merge sub array
    while (i < sizeOfLeft && j < sizeOfRight)
    {
        if (leftArr[i] <= rightArr[j])
        {
            arr[k++] = leftArr[i++];
        }
        else
        {
            arr[k++] = rightArr[j++];
        }
    }

    // copy in elemts 
    while (i < sizeOfLeft)
    {
        arr[k++] = leftArr[i++];
    }
    while (j < sizeOfRight)
    {
        arr[k++] = rightArr[j++];
    }
}

void msort(int *arr, const std::size_t n, const std::size_t threshold)
{
    // threshold is the lower limit of arraysize where function would start making parallel recursive calls
    //  if size of array goes below threshold, serial sort algorithm will be used to avoid overhead of task sched

    if (n < threshold)
    {
        // basecase: if input array size is under designated threshold
        std::sort(arr, arr + n);
        return;
    }
    else
    {
        // recursive case: perform split
        std::size_t middle = n / 2;
        int *leftSide = new int[middle];
        int *rightSide = new int[n - middle];

        std::copy(arr, arr + middle, leftSide);
        std::copy(arr + middle, arr + n, rightSide);

        // parallel region
        #pragma omp parallel
        {
            #pragma omp single
            {
                #pragma omp task shared(leftSide)
                msort(leftSide, middle, threshold);

                #pragma omp task shared(rightSide)// maybe dont need this???
                msort(rightSide, n - middle, threshold);

                #pragma omp taskwait
            }
        }
        merge(arr, leftSide, middle, rightSide, n - middle);

        delete[] leftSide;
        delete[] rightSide;
    }
}
