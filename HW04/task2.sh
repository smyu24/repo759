#! /usr/bin/env zsh
#SBATCH --job-name=task2
#SBATCH --output=task2_output_%j.txt
#SBATCH --error=task2_error_%j.txt
#SBATCH --time=00:15:00          
#SBATCH --ntasks=1               
#SBATCH --cpus-per-task=8
#SBATCH --partition=instruction

module load gcc

g++ task2.cpp -Wall -O3 -std=c++17 -o task2
./task2 100 10