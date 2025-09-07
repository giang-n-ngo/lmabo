#!/bin/bash
#SBATCH --output=log/%x.out
#SBATCH --error=log/%x.err
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=4:00:00 
#SBATCH --mem=24G                    
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
    # "Ackley" 
    # "Beale" 
    # "Branin" 
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
    # "Powell" 
    # "Rastrigin" 
    "Rosenbrock" 
    # "Shekel" 
    # "SixHumpCamel" 
    # "StyblinskiTang" 
    # "Easom" 
    "Sphere" 
    # "BucheRastrigin" 
    # "LinearSlope" 
    # "AttractiveSector" 
    # "StepEllipsoid" 
    # "RosenbrockRotated" 
    # "Ellipsoid2" 
    # "Discus" 
    # "BentCigar" 
    # "SharpRidge" 
    "DifferentPowers" 
    # "Weierstrass" 
    # "Schaffers" 
    # "SchaffersIllCond" 
    "CompositeGriewankRosenbrock" 
    "Schwefel" 
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

# Split into batches of 12 problems for 60 problems
echo "Running problems in batches of 12..."
run_batch "${problems[@]:0:12}"    # Problems 0-11
# run_batch "${problems[@]:12:12}"   # Problems 12-23
# run_batch "${problems[@]:24:12}"   # Problems 24-35
# run_batch "${problems[@]:36:12}"   # Problems 36-47
# run_batch "${problems[@]:48:12}"   # Problems 48-59

echo "All problems completed!"