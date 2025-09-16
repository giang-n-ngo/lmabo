#!/bin/bash
#SBATCH --job-name=lmabo-ops2-1
#SBATCH --output=log/lmabo-ops2-1.out
#SBATCH --error=log/lmabo-ops2-1.err
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=48:00:00 
#SBATCH --mem=20GB                    
#SBATCH --cpus-per-gpu=20             
#SBATCH --qos=batch-short
#SBATCH --mail-type=END
#SBATCH --mail-user=s222509501@deakin.edu.au

# Load necessary modules
module load Anaconda3
source activate
conda activate lmabo

CUDA_LAUNCH_BLOCKING=1

problems=(
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
    "Levy" 
    # "Michalewicz" 
    # "Shekel" 
    # "SixHumpCamel" 
    # "StyblinskiTang" 
    # "BucheRastrigin" 
    # "LinearSlope" 
    # "AttractiveSector" 
    # "StepEllipsoid" 
    # "Discus" 
    "BentCigar" 
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
    "hpt_wine_MLPSGD"
    # "hpt_diabetes_RandomForest"
    # "hpt_diabetes_DecisionTree"
    # "hpt_diabetes_SVM"
    "hpt_diabetes_AdaBoost"
    # "hpt_diabetes_MLPSGD"
)

# Function to run problems in batches
run_batch() {
    local batch=("$@")
    for problem in "${batch[@]}"; do
        echo "Starting problem: $problem"
        # if method is bo_alternating, we need to try different values for k from [1,3,5]
        if [ "$method" == "bo_alternating_k1" ]; then
            CUDA_VISIBLE_DEVICES=0 python run.py --problem $problem --method bo_alternating --k 1 &
        elif [ "$method" == "bo_alternating_k3" ]; then
            CUDA_VISIBLE_DEVICES=0 python run.py --problem $problem --method bo_alternating --k 3 &
        elif [ "$method" == "bo_alternating_k5" ]; then
            CUDA_VISIBLE_DEVICES=0 python run.py --problem $problem --method bo_alternating --k 5 &
        else
            CUDA_VISIBLE_DEVICES=0 python run.py --problem $problem --method lmabo-ops2 --server_node h100-m-01 &
        fi
    done
    wait  # Wait for this batch to complete
}

# Split into batches of 5 problems for 14 problems
echo "Running problems in batches of 5..."
run_batch "${problems[@]:0:4}"    # Problems 0-4
# run_batch "${problems[@]:5:5}"    # Problems 5-9
# run_batch "${problems[@]:10:4}"   # Problems 10-13
# run_batch "${problems[@]:39:3}"   # Problems 39-41
# run_batch "${problems[@]:42:3}"   # Problems 42-44
# run_batch "${problems[@]:45:3}"   # Problems 45-47
# run_batch "${problems[@]:48:3}"   # Problems 48-50
# run_batch "${problems[@]:51:3}"   # Problems 51-53
# run_batch "${problems[@]:54:3}"   # Problems 54-56
# run_batch "${problems[@]:57:3}"   # Problems 57-59

echo "All problems completed!"