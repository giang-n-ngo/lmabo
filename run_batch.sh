#!/bin/bash

# Define the problems and algorithms
problems=("Ackley" "Branin" "DixonPrice" "Easom" "Griewank" "Levy" "Powell" "Rastrigin" "Rosenbrock" "StyblinskiTang" "Beale" "Bukin" "Cosine8" "DropWave" "EggHolder" "Hartmann" "HolderTable" "Michalewicz" "Shekel" "SixHumpCamel")

# Loop over each problem and algorithm
for problem in "${problems[@]}"; do
    echo "Submitting job: ${problem}"
    sbatch --job-name="${problem}" --export=problem=$problem run.sh
done