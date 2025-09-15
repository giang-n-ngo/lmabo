#!/bin/bash
#SBATCH --job-name=qwen3-14b-2
#SBATCH --output=log/qwen3_14b-2.out
#SBATCH --error=log/qwen3_14b-2.err
#SBATCH --nodes=1
#SBATCH --partition=gpu-large
#SBATCH --gpus=h100:1
#SBATCH --time=48:00:00 
#SBATCH --mem=120GB
#SBATCH --cpus-per-gpu=16
#SBATCH --qos=batch-short
#SBATCH --mail-type=END
#SBATCH --mail-user=s222509501@deakin.edu.au

# Load necessary modules
module load Anaconda3
source activate
conda activate vllm_server

CUDA_LAUNCH_BLOCKING=1

vllm serve Qwen/Qwen3-14B \
    --enable-reasoning \
    --reasoning-parser deepseek_r1 \