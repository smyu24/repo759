#! /usr/bin/env zsh
#SBATCH --job-name=task3
#SBATCH --output=task3_output_%j.txt
#SBATCH --error=task3_error_%j.txt
#SBATCH --time=00:15:00          
#SBATCH --ntasks=1               
#SBATCH --gpus-per-task=1
#SBATCH --partition=instruction

module load nvidia/cuda/11.8.0
module load gcc/11.3.0

nvcc task3.cu vscale.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task3

./task3 5

for (( t = 10; t <= 29; t++ )); do
    ./task3 ${t} >> timing_results.txt
done 