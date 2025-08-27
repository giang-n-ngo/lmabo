#!/bin/bash
#SBATCH --output=log/%x.out
#SBATCH --error=log/%x.err
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=8:00:00 
#SBATCH --mem=8GB                    
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
    # BoTorch problems
    # "Ackley" 
    # "Beale" "Branin" "Bukin" 
    # "Cosine8" "DixonPrice" "DropWave" 
    # "EggHolder" "Griewank" "Hartmann" 
    # "HolderTable" "Levy" "Michalewicz" # ablation
    # "Powell" "Rastrigin" "Rosenbrock" # ablation
    # "Shekel" "SixHumpCamel" "StyblinskiTang" "Easom" # ablation
    # COCO problems
    # "Sphere" "BucheRastrigin" "LinearSlope" # ablation
    # "AttractiveSector" "StepEllipsoid" "RosenbrockRotated" # ablation
    # "Ellipsoid2" "Discus" "BentCigar" "SharpRidge" # ablation
    # "DifferentPowers" "Weierstrass" "Schaffers" 
    # "SchaffersIllCond" "CompositeGriewankRosenbrock" 
    # "Schwefel" "Gallagher21" "Gallagher101" "Katsuura" "LunacekBiRastrigin" 
    # HPT problems
    ## breast
    "hpt_breast_RandomForest"
    "hpt_breast_DecisionTree"
    "hpt_breast_SVM"
    "hpt_breast_AdaBoost"
    "hpt_breast_MLPSGD"
    ## digits
    "hpt_digits_RandomForest"
    "hpt_digits_DecisionTree"
    "hpt_digits_SVM"
    "hpt_digits_AdaBoost"
    "hpt_digits_MLPSGD"
    ## wine
    "hpt_wine_RandomForest"
    "hpt_wine_DecisionTree"
    "hpt_wine_SVM"
    "hpt_wine_AdaBoost"
    "hpt_wine_MLPSGD"
    ## diabetes
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

# Split into batches of 3 problems for 40 problems
echo "Running problems in batches of 3..."
run_batch "${problems[@]:0:3}"    # Problems 0-2
run_batch "${problems[@]:3:3}"    # Problems 3-5
run_batch "${problems[@]:6:3}"    # Problems 6-8
run_batch "${problems[@]:9:3}"    # Problems 9-11
run_batch "${problems[@]:12:3}"   # Problems 12-14
run_batch "${problems[@]:15:3}"   # Problems 15-17
run_batch "${problems[@]:18:3}"   # Problems 18-20
run_batch "${problems[@]:21:3}"   # Problems 21-23
run_batch "${problems[@]:24:3}"   # Problems 24-26
run_batch "${problems[@]:27:3}"   # Problems 27-29
run_batch "${problems[@]:30:3}"   # Problems 30-32
run_batch "${problems[@]:33:3}"   # Problems 33-35
run_batch "${problems[@]:36:3}"   # Problems 36-38
run_batch "${problems[@]:39:1}"   # Problems 39-39

echo "All problems completed!"