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
#include <vector>

#include <chrono>
#include <ratio>
#include <random>

void msort(int *arr, const std::size_t n, const std::size_t threshold)
{
    //threshold is the lower limit of arraysize where function would start making parallel recursive calls
        // if size of array goes below threshold, serial sort algorithm will be used to avoid overhead of task sched
    if(n >= threshold)
    {
        // good to start parallel implementation and recursive calls
        for (int i = 0; i < n; i++)
        {
            // keeps track of array size and swaps to serial implementation when needed
            if(n >= threshold)
            {break;}

            //base case 


            //recursive case
        }

    }

    // serial implementation of msort

}


// Implement in a file called msort.cpp the parallel merge sort algorithm with the prototype
// specified in msort.h. You may add other functions in your msort.cpp program, but you
// should not change the msort.h file. 
// Your msort function should take an array of integers
// and return the sorted results in place in ascending order. For instance, after calling msort
// function, the original arr = [3, 7, 10, 2, 1, 3] would become arr = [1, 2, 3, 3, 7, 10].