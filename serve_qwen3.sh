#!/bin/bash
#SBATCH --output=log/qwen3.out
#SBATCH --error=log/qwen3.err
#SBATCH --nodes=1
#SBATCH --partition=gpu-large
#SBATCH --gres=gpu:a100:1
#SBATCH --time=48:00:00 
#SBATCH --mem=48GB
#SBATCH --cpus-per-gpu=16
#SBATCH --qos=batch-short
#SBATCH --mail-type=END
#SBATCH --mail-user=s222509501@deakin.edu.au

# Load necessary modules
module load Anaconda3
source activate
conda activate vllm_server

CUDA_LAUNCH_BLOCKING=1

vllm serve Qwen/Qwen3-8B \
    --enable-reasoning \
    --reasoning-parser deepseek_r1 \