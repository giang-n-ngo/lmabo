#!/bin/bash
#SBATCH --output=log/report.out
#SBATCH --error=log/report.err
#SBATCH --nodes=1
#SBATCH --partition=cpu
#SBATCH --time=00:10:00 
#SBATCH --cpus-per-task=1
#SBATCH --qos=batch-short

# Load necessary modules
module load Anaconda3
source activate
conda activate lmabo

python report.py --setting full
# python report.py --setting synthetic
# python report.py --setting real