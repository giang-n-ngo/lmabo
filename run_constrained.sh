#!/bin/bash
#SBATCH --output=log/constrained.out
#SBATCH --error=log/constrained.err
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=48:00:00 
#SBATCH --mem=24GB                    
#SBATCH --cpus-per-gpu=16             
#SBATCH --qos=batch-short
#SBATCH --mail-type=END
#SBATCH --mail-user=s222509501@deakin.edu.au

# Load necessary modules
module load Anaconda3
source activate
conda activate lmabo

CUDA_LAUNCH_BLOCKING=1

problems=(
    "ConstrainedGramacy"
    "KeaneBumpFunction"
    "PressureVessel"
    "ConstrainedHartmann"
    "TensionCompressionString"
    "WeldedBeamSO"
)

# Function to run problems in batches
run_batch() {
    local batch=("$@")
    for problem in "${batch[@]}"; do
        echo "Starting problem: $problem"
        CUDA_VISIBLE_DEVICES=0 python run.py --problem $problem --method lmabo --constrained &
    done
    wait  # Wait for this batch to complete
}

# Split into batches of 4 problems
echo "Running problems in batches of 4..."
run_batch "${problems[@]:0:3}"    # Problems 0-2
run_batch "${problems[@]:3:3}"    # Problems 3-5

echo "All problems completed!"