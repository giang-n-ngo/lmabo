#!/bin/bash
#SBATCH --job-name=phi4r
#SBATCH --output=log/phi4r.out
#SBATCH --error=log/phi4r.err
#SBATCH --nodes=1
#SBATCH --partition=gpu-large
#SBATCH --gpus=h100:1
#SBATCH --time=48:00:00 
#SBATCH --mem=128GB
#SBATCH --cpus-per-gpu=16
#SBATCH --qos=batch-short
#SBATCH --mail-type=END
#SBATCH --mail-user=s222509501@deakin.edu.au

# Load necessary modules
module load Anaconda3
source activate
conda activate vllm_server

CUDA_LAUNCH_BLOCKING=1

vllm serve microsoft/Phi-4-reasoning --enable-reasoning --reasoning-parser deepseek_r1
