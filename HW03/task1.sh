#! /usr/bin/env zsh
#SBATCH --job-name=task1
#SBATCH --output=task1_output_%j.txt
#SBATCH --error=task1_error_%j.txt
#SBATCH --time=00:15:00          
#SBATCH --ntasks=1               
#SBATCH --cpus-per-task=20
#SBATCH --partition=instruction

module load gcc

g++ task1.cpp matmul.cpp -Wall -O3 -std=c++17 -o task1 -fopenmp

./task1 1024 20