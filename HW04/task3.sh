#! /usr/bin/env zsh
#SBATCH --job-name=task3
#SBATCH --output=task3_output_%j.txt
#SBATCH --error=task3_error_%j.txt
#SBATCH --time=00:15:00          
#SBATCH --ntasks=1               
#SBATCH --cpus-per-task=8
#SBATCH --partition=instruction

module load gcc

g++ task3.cpp -Wall -O3 -std=c++17 -o task3 -fopenmp

for (( t = 1; t <= 8; t++ )); do
    ./task3 150 150 ${t} 1 >> timing_results{static}.txt
done    


for (( t = 1; t <= 8; t++ )); do
    ./task3 150 150 ${t} 2 >> timing_results{dynamic}.txt
done    


for (( t = 1; t <= 8; t++ )); do
    ./task3 150 150 ${t} 3 >> timing_results{guided}.txt
done    