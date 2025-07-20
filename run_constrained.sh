#!/bin/bash
#SBATCH --output=log/constrained_lmabo.out
#SBATCH --error=log/constrained_lmabo.err
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1  # Request 2 GPUs
#SBATCH --time=24:00:00
#SBATCH --mem=12GB  # Increased memory for more simultaneous processes
#SBATCH --cpus-per-gpu=4
#SBATCH --qos=batch-short
#SBATCH --mail-type=END
#SBATCH --mail-user=s222509501@deakin.edu.au

# Load necessary modules
module load Anaconda3
source activate
conda activate lmabo

CUDA_LAUNCH_BLOCKING=1

problems=(
    # "ConstrainedGramacy"
    # "ConstrainedHartmann"
    # "WeldedBeamSO"
    "KeaneBumpFunction"
    # "PressureVessel"
    # "TensionCompressionString"
)

# Function to run problems in batches, assigning GPUs
run_batch() {
    local start=$1       # Starting index
    local count=$2       # Number of problems to run
    local gpu_id=$3      # GPU ID to use

    for ((i=0; i<count; i++)); do
        local index=$((start + i))
        local problem="${problems[$index]}"
        echo "Starting problem: $problem on GPU $gpu_id (Index: $index)"
        CUDA_VISIBLE_DEVICES=$gpu_id python run.py --problem "$problem" --method lmabo --constrained &
    done
    wait # Wait for the batch to complete
}

# Run 3 problems simultaneously per GPU
echo "Running 3 problems simultaneously per GPU..."

# GPU 0: Problems 0, 1, 2
run_batch 0 1 0 &

# GPU 1: Problems 3, 4, 5
# run_batch 3 3 1 &

wait # Wait for all batches to complete

echo "All problems completed!"

# an alternative script to run just one problem from scratch
