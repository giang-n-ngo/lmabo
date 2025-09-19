#!/bin/bash
#SBATCH --job-name=lmabo-ops3-2
#SBATCH --output=log/lmabo-ops3-2.out
#SBATCH --error=log/lmabo-ops3-2.err
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --time=18:00:00 
#SBATCH --mem=24GB                    
#SBATCH --cpus-per-gpu=24             
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
    # "Griewank" 
    # "Hartmann" 
    # "HolderTable" 
    # "Levy" 
    # "Michalewicz" 
    # "Shekel" 
    # "SixHumpCamel" 
    # "StyblinskiTang" 
    # "BucheRastrigin" 
    # "LinearSlope" 
    # "AttractiveSector" 
    # "StepEllipsoid" 
    # "Discus" 
    # "BentCigar" 
    # "SharpRidge" 
    # "DifferentPowers" 
    # "Weierstrass" 
    # "SchaffersIllCond" 
    # "CompositeGriewankRosenbrock" 
    # "Gallagher21" 
    # "Gallagher101" 
    # "Katsuura" 
    # "LunacekBiRastrigin" 
    # "hpt_breast_RandomForest"
    # "hpt_breast_DecisionTree"
    # "hpt_breast_SVM"
    # "hpt_breast_AdaBoost"
    # "hpt_breast_MLPSGD"
    # "hpt_digits_RandomForest"
    # "hpt_digits_DecisionTree"
    # "hpt_digits_SVM"
    # "hpt_digits_AdaBoost"
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
        CUDA_VISIBLE_DEVICES=0 python run.py --problem $problem --method lmabo-ops3 --server_node h100-m-12 &
    done
    wait  # Wait for this batch to complete
}

# Split into batches of 6 problems for 18 problems
echo "Running problems in batches of 6..."
run_batch "${problems[@]:0:6}"    # Problems 0-5
run_batch "${problems[@]:6:6}"    # Problems 6-11
run_batch "${problems[@]:12:6}"   # Problems 12-17
# run_batch "${problems[@]:18:6}"   # Problems 18-23
# run_batch "${problems[@]:24:6}"   # Problems 24-29
# run_batch "${problems[@]:30:3}"   # Problems 30-32
# run_batch "${problems[@]:48:3}"   # Problems 48-50
# run_batch "${problems[@]:51:3}"   # Problems 51-53
# run_batch "${problems[@]:54:3}"   # Problems 54-56
# run_batch "${problems[@]:57:3}"   # Problems 57-59

echo "All problems completed!"