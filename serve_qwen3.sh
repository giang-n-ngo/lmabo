#!/bin/bash
#SBATCH --output=log/%x.out
#SBATCH --error=log/%x.err
#SBATCH --nodes=1
#SBATCH --partition=gpu-large
#SBATCH --gres=gpu:h100:1
#SBATCH --time=24:00:00 
#SBATCH --mem=32GB
#SBATCH --cpus-per-gpu=4
#SBATCH --qos=batch-short
#SBATCH --mail-type=END
#SBATCH --mail-user=s222509501@deakin.edu.au

# Load necessary modules
module load Anaconda3
source activate
conda activate lmabo-ops

CUDA_LAUNCH_BLOCKING=1

vllm serve Qwen/Qwen3-8B \
    --enable-reasoning \
    --reasoning-parser deepseek_r1 \