#!/bin/bash
#SBATCH --output=log/%x.out
#SBATCH --error=log/%x.err
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=24:00:00 
#SBATCH --mem=12G
#SBATCH --cpus-per-gpu=8
#SBATCH --qos=batch-short
#SBATCH --mail-type=END
#SBATCH --mail-user=s222509501@deakin.edu.au

# Load necessary modules
module load Anaconda3
source activate
conda activate lmabo

CUDA_LAUNCH_BLOCKING=1

# Run the Python script
python run_moo.py --problem $problem --acq_type $acq_type --server_node $server_node --starting_exp_idx $starting_exp_idx