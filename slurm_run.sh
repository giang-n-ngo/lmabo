#!/bin/bash
#SBATCH --output=log/%x.out
#SBATCH --error=log/%x.err
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=8:00:00 
#SBATCH --mem=8G                    
#SBATCH --cpus-per-gpu=8             
#SBATCH --qos=batch-short
#SBATCH --mail-type=END
#SBATCH --mail-user=s222509501@deakin.edu.au

# Load necessary modules
module load Anaconda3
source activate
conda activate lmabo

CUDA_LAUNCH_BLOCKING=1

problems=(
    # # "Ackley" 
    # # "Beale" 
    # "Bukin" 
    # # "Cosine8" 
    # # "DixonPrice" 
    # "DropWave" 
    # "EggHolder" 
    # "Griewank" 
    # # "Hartmann" 
    # "HolderTable" 
    # # "Levy" 
    # # "Michalewicz" 
    # # "Shekel" 
    # # "SixHumpCamel" 
    # # "StyblinskiTang" 
    # "BucheRastrigin" 
    # # "LinearSlope" 
    # # "AttractiveSector" 
    # # "StepEllipsoid" 
    # # "Discus" 
    # # "BentCigar" 
    # "SharpRidge" 
    # "DifferentPowers" 
    # "Weierstrass" 
    # "SchaffersIllCond" 
    # "CompositeGriewankRosenbrock" 
    # # "Gallagher21" 
    # "Gallagher101" 
    "Katsuura" 
    "LunacekBiRastrigin" 
    # "hpt_breast_RandomForest"
    "hpt_breast_DecisionTree"
    "hpt_breast_SVM"
    # "hpt_breast_AdaBoost"
    "hpt_breast_MLPSGD"
    "hpt_digits_RandomForest"
    "hpt_digits_DecisionTree"
    # "hpt_digits_SVM"
    # "hpt_digits_AdaBoost"
    "hpt_digits_MLPSGD"
    # "hpt_wine_RandomForest"
    "hpt_wine_DecisionTree"
    # "hpt_wine_SVM"
    # "hpt_wine_AdaBoost"
    # "hpt_wine_MLPSGD"
    # "hpt_diabetes_RandomForest"
    "hpt_diabetes_DecisionTree"
    # "hpt_diabetes_SVM"
    # "hpt_diabetes_AdaBoost"
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

# Split into batches of 6 problems for 12 problems
echo "Running problems in batches of 6..."
run_batch "${problems[@]:0:6}"    # Problems 0-5
run_batch "${problems[@]:6:6}"   # Problems 6-11
# run_batch "${problems[@]:12:6}"    # Problems 12-17
# run_batch "${problems[@]:18:6}"    # Problems 18-23
# run_batch "${problems[@]:16:4}"   # Problems 16-19
# run_batch "${problems[@]:20:4}"   # Problems 20-23
# run_batch "${problems[@]:24:4}"   # Problems 24-27
# run_batch "${problems[@]:28:4}"   # Problems 28-31
# run_batch "${problems[@]:32:4}"   # Problems 32-35
# run_batch "${problems[@]:36:4}"   # Problems 36-39
# run_batch "${problems[@]:40:4}"   # Problems 40-43
# run_batch "${problems[@]:44:4}"   # Problems 44-47
# run_batch "${problems[@]:48:2}"   # Problems 48-49

echo "All problems completed!"