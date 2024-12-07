#! /usr/bin/env zsh
#SBATCH --job-name=task1
#SBATCH --output=task1_output_%j.txt
#SBATCH --error=task1_error_%j.txt
#SBATCH --time=00:15:00          
#SBATCH --ntasks=1               
#SBATCH --gpus-per-task=1
#SBATCH --partition=instruction

module load nvidia/cuda/11.8.0
module load gcc/11.3.0

nvcc task1.cu matmul.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o task1
# ./task1 n blk dim

for (( t = 5; t <= 14; t++ )); do
    ts=$((2**t))
    ./task1 ${ts} 32 >> timing_results.txt
done 

for (( t = 5; t <= 14; t++ )); do
    ts=$((2**t))
    ./task1 ${ts} 16 >> timing_results2.txt
done 