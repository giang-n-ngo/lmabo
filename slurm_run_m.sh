#!/bin/bash
#SBATCH --job-name=lmabo-ops3-1
#SBATCH --output=log/lmabo-ops3-1.out
#SBATCH --error=log/lmabo-ops3-1.err
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=6:00:00 
#SBATCH --mem=20GB                    
#SBATCH --cpus-per-gpu=16             
#SBATCH --qos=batch-short
#SBATCH --mail-type=END
#SBATCH --mail-user=s222509501@deakin.edu.au

# Load necessary modules
module load Anaconda3
source activate
conda activate lmabo

CUDA_LAUNCH_BLOCKING=1

# problems=(
    # "Ackley" 
    # "Beale" 
    # "Bukin" 
    # "Cosine8" 
    # "DixonPrice" 
    # "DropWave" 
    # "EggHolder" 
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
    # "hpt_digits_MLPSGD"
    # "hpt_wine_RandomForest"
    # "hpt_wine_DecisionTree"
    # "hpt_wine_SVM"
    # "hpt_wine_AdaBoost"
    # "hpt_wine_MLPSGD"
    # "hpt_diabetes_RandomForest"
    # "hpt_diabetes_DecisionTree"
    # "hpt_diabetes_SVM"
    # "hpt_diabetes_AdaBoost"
    # "hpt_diabetes_MLPSGD"
# )
problems=(
    "DixonPrice"
    # "Levy"
    # "SixHumpCamel"
    # "AttractiveSector"
    # "hpt_digits_AdaBoost"
    # "hpt_digits_MLPSGD"
)

# Function to run problems in batches
run_batch() {
    local batch=("$@")
    for problem in "${batch[@]}"; do
        echo "Starting problem: $problem"
        CUDA_VISIBLE_DEVICES=0 python run.py --problem $problem --method lmabo-ops3 --server_node a100-m-02 &
    done
    wait  # Wait for this batch to complete
}

# Split into batches of 12 problems for 20 problems
echo "Running problems in batches of 12..."
run_batch "${problems[@]:0:1}"    # Problems 0-5
# run_batch "${problems[@]:6:7}"   # Problems 6-12
# run_batch "${problems[@]:13:12}"   # Problems 13-24
# run_batch "${problems[@]:30:3}"   # Problems 30-32
# run_batch "${problems[@]:48:3}"   # Problems 48-50
# run_batch "${problems[@]:51:3}"   # Problems 51-53
# run_batch "${problems[@]:54:3}"   # Problems 54-56
# run_batch "${problems[@]:57:3}"   # Problems 57-59

echo "All problems completed!"