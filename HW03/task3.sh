#! /usr/bin/env zsh
#SBATCH --job-name=task3
#SBATCH --output=task3_output_%j.txt
#SBATCH --error=task3_error_%j.txt
#SBATCH --time=00:15:00          
#SBATCH --ntasks=1               
#SBATCH --cpus-per-task=20
#SBATCH --partition=instruction
#SBATCH --exclusive

module load gcc

g++ task3.cpp msort.cpp -Wall -O3 -std=c++17 -o task3 -fopenmp

for (( i = 1; i <= 10; i++ )); do
    ts=$((2**i))
    ./task3 1000000 8 ${ts} >> timing_results.txt
done

# Second case: varying t (t = 1, 2, ..., 20)
# echo "Running task3 with t values..."
for (( t = 1; t <= 20; t++ )); do
    ./task3 1000000 ${t} 128 >> timing_results.txt
done    


# task3 ts.pdf
# value n = 10^6, value t = 8, and value ts = 2^1, 2^2, ... , 2^10

# task3 t.pdf
# value n = 10^6, value t = 1, 2, · · · , 20, and ts equals to the value that
# yields the best performance as you found in the plot of time vs. ts
