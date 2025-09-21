#!/bin/bash
#SBATCH --output=log/%x.out
#SBATCH --error=log/%x.err
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=24:00:00 
#SBATCH --mem=40G                    
#SBATCH --cpus-per-gpu=40             
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
    "Beale" 
    # "Bukin" 
    # "Cosine8" 
    "DixonPrice" 
    # "DropWave" 
    # "EggHolder" 
    # "Griewank" 
    # "Hartmann" 
    # "HolderTable" 
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
        # if method is bo_alternating, we need to try different values for k from [1,3,5]
        if [ "$method" == "bo_alternating_k1" ]; then
            CUDA_VISIBLE_DEVICES=0 python run.py --problem $problem --method bo_alternating --k 1 &
        elif [ "$method" == "bo_alternating_k3" ]; then
            CUDA_VISIBLE_DEVICES=0 python run.py --problem $problem --method bo_alternating --k 3 &
        elif [ "$method" == "bo_alternating_k5" ]; then
            CUDA_VISIBLE_DEVICES=0 python run.py --problem $problem --method bo_alternating --k 5 &
        else
            CUDA_VISIBLE_DEVICES=0 python run.py --problem $problem --method $method &
        fi
    done
    wait  # Wait for this batch to complete
}

# Split into batches of 10 problems for 50 problems
echo "Running problems in batches of 10..."
run_batch "${problems[@]:0:10}"    # Problems 0-9
run_batch "${problems[@]:10:10}"   # Problems 10-19
run_batch "${problems[@]:20:10}"    # Problems 20-29
run_batch "${problems[@]:30:10}"    # Problems 30-39
run_batch "${problems[@]:40:10}"    # Problems 40-49
# run_batch "${problems[@]:10:2}"   # Problems 10-11
# run_batch "${problems[@]:12:2}"   # Problems 12-13
# run_batch "${problems[@]:14:2}"   # Problems 14-15
# run_batch "${problems[@]:15:3}"  # Problems 15-17
# run_batch "${problems[@]:18:2}"  # Problems 18-19

echo "All problems completed!"