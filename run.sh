#!/bin/bash
#SBATCH --output=log/%x.out
#SBATCH --error=log/%x.err
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=gpu-v100
#SBATCH --time=12:00:00 
#SBATCH --mem=6G
#SBATCH --cpus-per-gpu=4
#SBATCH --qos=batch-short
#SBATCH --mail-type=END
#SBATCH --mail-user=s222509501@deakin.edu.au

# Load necessary modules
module load Anaconda3
source activate
conda activate lmabo

CUDA_LAUNCH_BLOCKING=1

# Run the Python script
python run.py --problem $problem --method $method