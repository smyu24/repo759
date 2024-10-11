#! /usr/bin/env zsh
#SBATCH --job-name=task3
#SBATCH --output=task3_output_%j.txt
#SBATCH --error=task3_error_%j.txt
#SBATCH --time=00:15:00          
#SBATCH --ntasks=1               
#SBATCH --cpus-per-task=20
#SBATCH --partition=instruction

module load gcc

g++ task3.cpp msort.cpp -Wall -O3 -std=c++17 -o task3 -fopenmp


for {1, 2...10} do:
./task3 $10**6 8 20 

