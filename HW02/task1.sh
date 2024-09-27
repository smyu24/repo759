#! /usr/bin/env zsh
#SBATCH --job-name=task1
#SBATCH --output=task1_output_%j.txt
#SBATCH --error=task1_error_%j.txt
#SBATCH --time=01:00:00          
#SBATCH --ntasks=1               
#SBATCH --cpus-per-task=1
#SBATCH --partition=instruction

module load gcc

g++ scan.cpp task1.cpp -Wall -O3 -std=c++17 -o task1

echo "n time(s)" > timing_results.txt

for i in {10..30}; do
    echo "n=2^$i" >> timing_results.txt  # Log the value of n
    n=$((2**i))
    ./task1 $n >> timing_results.txt
done