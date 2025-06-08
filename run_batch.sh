#!/bin/bash

# Define the problems and algorithms
# problems=("Ackley" "Branin" "DixonPrice" "Easom" "Griewank" "Levy" "Powell" "Rastrigin" "Rosenbrock" "StyblinskiTang" "Beale" "Bukin" "Cosine8" "DropWave" "EggHolder" "Hartmann" "HolderTable" "Michalewicz" "Shekel" "SixHumpCamel")
problems=("Bukin" "Cosine8" "DropWave" "Hartmann")
method="lmabo"

# Loop over each problem and algorithm
for problem in "${problems[@]}"; do
    echo "Submitting job: ${problem}_${method}"
    sbatch --job-name="${problem}_${method}" --export=problem=$problem,method=$method run.sh
done