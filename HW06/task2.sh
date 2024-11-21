#! /usr/bin/env zsh
#SBATCH --job-name=task2
#SBATCH --output=task2_output_%j.txt
#SBATCH --error=task2_error_%j.txt
#SBATCH --time=00:15:00          
#SBATCH --ntasks=1               
#SBATCH --gpus-per-task=1
#SBATCH --partition=instruction

module load nvidia/cuda/11.8.0
module load gcc/11.3.0

nvcc task2.cu stencil.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o task2

for (( t = 10; t <= 29; t++ )); do
    ts=$((2**t))
    ./task2 ${ts} 128 1024 >> timing_results.txt
done 

for (( t = 10; t <= 29; t++ )); do
    ts=$((2**t))
    ./task2 ${ts} 128 512 >> timing_results2.txt
done 
