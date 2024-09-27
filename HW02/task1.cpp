#include <iostream>
#include <cstdlib>
#include <ctime>
#include "scan.h"

#include <chrono>
#include <ratio>
#include <random>

int main(int argc, char *argv[])
{
    // argc == 2 for n
    if (argc < 2)
    {
        return 1;
    }

    int n = std::atoi(argv[1]);

    // n must be a positive integer
    if (n <= 0)
    {
        return 1;
    }

    std::random_device rd;  
    std::mt19937 gen(rd()); 
    std::uniform_real_distribution<float> dis(-1.0, 1.0); 
    
    // SAA
    float arrayOfNums[n];
    float output[n];
    for (int i = 0; i < n; ++i)
    {
        // generate random float domains: [-1,1]
        arrayOfNums[i] = dis(gen);
        //std::cout << arrayOfNums[i] << " " << i << std::endl;
    }

    // chronos to get time taken to run scan function
    std::chrono::high_resolution_clock::time_point start;
    std::chrono::high_resolution_clock::time_point end;
    std::chrono::duration<double, std::milli> duration_sec;

    start = std::chrono::high_resolution_clock::now();
    // use chronos to get scan function runtime to accuracy of millisecond
    scan(arrayOfNums, output, n);
    end = std::chrono::high_resolution_clock::now();

    duration_sec = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start);
    // output delimited by comma
    std::cout << duration_sec.count() << ","
              << output[0] << ","
              << output[n-1] << std::endl;

    // no mem to dealloc
}

//     i) Creates an array of n random float numbers between -1.0 and 1.0. n should be read as the first command line argument as below.
//     ii) Scans the array using your scan function.
//     iii) Prints out the time taken by your scan function in milliseconds2.
//     iv) Prints the first element of the output scanned array.
//     v) Prints the last element of the output scanned array.
//     vi) Deallocates memory when necessary.