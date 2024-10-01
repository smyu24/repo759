#! /usr/bin/env zsh
#SBATCH --job-name=task2
#SBATCH --output=task2_output_%j.txt
#SBATCH --error=task2_error_%j.txt
#SBATCH --time=00:15:00          
#SBATCH --ntasks=1               
#SBATCH --cpus-per-task=1
#SBATCH --partition=instruction

module load gcc

g++ convolution.cpp task2.cpp -Wall -O3 -std=c++17 -o task2

./task2 5 5 
for i in {10..30}; do
    echo "n=2^$i" >> timing_results.txt  # Log the value of n
    n=$((2**i))
    ./task1 $n >> timing_results.txt
done