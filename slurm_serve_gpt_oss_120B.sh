#!/bin/bash
#SBATCH --job-name=gptoss-120b
#SBATCH --output=log/gptoss-120b.out
#SBATCH --error=log/gptoss-120b.err
#SBATCH --nodes=1
#SBATCH --partition=gpu-large
#SBATCH --gpus=h200:1
#SBATCH --time=48:00:00 
#SBATCH --mem=128GB
#SBATCH --cpus-per-gpu=16
#SBATCH --qos=batch-short
#SBATCH --mail-type=BEGIN
#SBATCH --mail-user=s222509501@deakin.edu.au

# Load necessary modules
module load Anaconda3
source activate
conda activate vllm_server2

# uv pip install --pre vllm==0.10.1+gptoss \
#     --extra-index-url https://wheels.vllm.ai/gpt-oss/ \
#     --extra-index-url https://download.pytorch.org/whl/nightly/cu124 \
#     --index-strategy unsafe-best-match

vllm serve openai/gpt-oss-120b \
  --dtype auto \
  --gpu-memory-utilization 0.95 \