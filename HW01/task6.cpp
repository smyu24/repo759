#include <cstdio>
#include <iostream>

//command line argument code from https://github.com/tsung-wei-huang/repo759/blob/main/FAQ/BestPractices/command_line_arguments.md
int main(int argc, char *argv[]) {
        
    int N = std::atoi(argv[1]);
    // int N = 6;

    for(int i = 0; i < N + 1; i++){
        printf("%i ", i);
    }

    std::cout << std::endl;

    //print backwards
    for(int i = N; i > -1; i--){
        std::cout << i << " ";
    }

    std::cout << std::endl;

    return 0;   
}


// a) Takes a command line argument N. (If you are confused about command line arguments, it may be helpful for you to read this) 
// b) Prints out each integer from 0 to N (including 0 and N) in ascending order with the printf function. 
// c) Prints out each integer from N to 0 (including N and 0) in descending order with std::cout. 


// integers separated by spaces ; lines ending in a newline

//g++ task6.cpp -Wall -O3 -std=c++17 -o task6 
// ./task6 N 

// • Expected output (followed by a newline): 
// 0 1 2 3 · · · N 
// N · · · 3 2 1 0 
// • Example expected output for N = 6 (followed by a newline): 
// 0 1 2 3 4 5 6 
// 6 5 4 3 2 1 0 
