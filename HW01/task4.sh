#! /usr/bin/env zsh

#SBATCH --job-name=FirstSlurm
#SBATCH --partition=instruction
#SBATCH --ntasks=1 --cpus-per-task=2
#SBATCH -t 0-00:30:00
#SBATCH -o FirstSlurm.out -e FirstSlurm.err

cd $SLURM_SUBMIT_DIR

echo $SLURM_SUBMIT_HOST