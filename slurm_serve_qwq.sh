#!/bin/bash
#SBATCH --job-name=qwq-32B
#SBATCH --output=log/qwq-32B.out
#SBATCH --error=log/qwq-32B.err
#SBATCH --nodes=1
#SBATCH --partition=gpu-large
#SBATCH --gpus=h100:2
#SBATCH --time=24:00:00 
#SBATCH --mem=120GB
#SBATCH --cpus-per-gpu=32
#SBATCH --qos=batch-short
#SBATCH --mail-type=BEGIN
#SBATCH --mail-user=s222509501@deakin.edu.au

# Load necessary modules
module load Anaconda3
source activate
conda activate vllm_server

vllm serve Qwen/QwQ-32B \
    --max-model-len 262144 \
    --reasoning-parser deepseek_r1 \
    --gpu-memory-utilization 0.9 \
    --dtype auto \
    --tensor-parallel-size 2 \