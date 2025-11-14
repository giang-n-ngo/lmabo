#!/bin/bash
#SBATCH --job-name=lmabo-gpt
#SBATCH --output=log/lmabo-gpt.out
#SBATCH --error=log/lmabo-gpt.err
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=24:00:00 
#SBATCH --mem=32GB                    
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
    "Ackley" 
    "Beale" 
    "Bukin" 
    "Cosine8" 
    "DixonPrice" 
    "DropWave" 
    "EggHolder" 
    "Griewank" 
    "Hartmann" 
    "HolderTable" 
    "Levy" 
    "Michalewicz" 
    "Shekel" 
    "SixHumpCamel" 
    "StyblinskiTang" 
    "BucheRastrigin" 
    "LinearSlope" 
    "AttractiveSector" 
    "StepEllipsoid" 
    "Discus" 
    "BentCigar" 
    "SharpRidge" 
    "DifferentPowers" 
    "Weierstrass" 
    "SchaffersIllCond" 
    "CompositeGriewankRosenbrock" 
    "Gallagher21" 
    "Gallagher101" 
    "Katsuura" 
    "LunacekBiRastrigin" 
    "hpt_breast_RandomForest"
    "hpt_breast_DecisionTree"
    "hpt_breast_SVM"
    "hpt_breast_AdaBoost"
    "hpt_breast_MLPSGD"
    "hpt_digits_RandomForest"
    "hpt_digits_DecisionTree"
    "hpt_digits_SVM"
    "hpt_digits_AdaBoost"
    "hpt_digits_MLPSGD"
    "hpt_wine_RandomForest"
    "hpt_wine_DecisionTree"
    "hpt_wine_SVM"
    "hpt_wine_AdaBoost"
    "hpt_wine_MLPSGD"
    "hpt_diabetes_RandomForest"
    "hpt_diabetes_DecisionTree"
    "hpt_diabetes_SVM"
    "hpt_diabetes_AdaBoost"
    "hpt_diabetes_MLPSGD"
)

# Function to run problems in batches
run_batch() {
    local batch=("$@")
    for problem in "${batch[@]}"; do
        echo "Starting problem: $problem"
        CUDA_VISIBLE_DEVICES=0 python run.py --problem $problem --method lmabo-ops6 --server_node h200l-m-04 &
    done
    wait  # Wait for this batch to complete
}

# Split into batches of 8 problems for 50 problems
echo "Running problems in batches of 8..."
run_batch "${problems[@]:0:8}"    # Problems 0-7
run_batch "${problems[@]:8:8}"   # Problems 8-15
run_batch "${problems[@]:16:8}"   # Problems 16-23
run_batch "${problems[@]:24:8}"   # Problems 24-31
run_batch "${problems[@]:32:8}"   # Problems 32-39
run_batch "${problems[@]:40:8}"   # Problems 40-47
run_batch "${problems[@]:48:3}"   # Problems 48-50

echo "All problems completed!"