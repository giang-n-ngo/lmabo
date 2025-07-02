#!/bin/bash

# Define the problems and algorithms
problems=(
    "Ackley" "Beale" "Branin" "Bukin" "Cosine8" 
    "DixonPrice" "DropWave" "EggHolder" "Griewank" 
    "Hartmann" "HolderTable" "Levy" "Michalewicz" "Powell" 
    "Rastrigin" "Rosenbrock" "Shekel" "SixHumpCamel" "StyblinskiTang" 
    "Sphere" "Ellipsoid" "BucheRastrigin" "LinearSlope" "AttractiveSector" 
    "StepEllipsoid" "RosenbrockRotated" "Ellipsoid2" "Discus" "BentCigar" 
    "SharpRidge" "DifferentPowers" "Weierstrass" "Schaffers" 
    "SchaffersIllCond" "CompositeGriewankRosenbrock" "Schwefel" "Gallagher21" 
    "Gallagher101" "Katsuura" "LunacekBiRastrigin" "Easom"
)
# problems=("Ackley" "DixonPrice" "Griewank" "Powell" "Rosenbrock" "StyblinskiTang" "AttractiveSector" "RosenbrockRotated" "Discus" "DifferentPowers" "SchaffersIllCond" "CompositeGriewankRosenbrock")
# method="lmabo"
method="lmabo-ops"
# method="bo"
# method="gphedge"

# Loop over each problem and algorithm
for problem in "${problems[@]}"; do
    echo "Submitting job: ${problem}_${method}"
    sbatch --job-name="${problem}_${method}" --export=problem=$problem,method=$method run.sh
done