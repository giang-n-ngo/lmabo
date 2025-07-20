#!/bin/bash

# Define the problems and algorithms
## Main problems
# problems=(
#     "Ackley" "Beale" "Branin" "Bukin" "Cosine8" 
#     "DixonPrice" "DropWave" "EggHolder" "Griewank" 
#     "Hartmann" "HolderTable" "Levy" "Michalewicz" "Powell" 
#     "Rastrigin" "Rosenbrock" "Shekel" "SixHumpCamel" "StyblinskiTang" 
#     "Sphere" "BucheRastrigin" "LinearSlope" "AttractiveSector" 
#     "StepEllipsoid" "RosenbrockRotated" "Ellipsoid2" "Discus" "BentCigar" 
#     "SharpRidge" "DifferentPowers" "Weierstrass" "Schaffers" 
#     "SchaffersIllCond" "CompositeGriewankRosenbrock" "Schwefel" "Gallagher21" 
#     "Gallagher101" "Katsuura" "LunacekBiRastrigin" "Easom"
# )
## OPS problems
# problems=(
    # "Beale" "Branin" "Bukin" "Cosine8" "DropWave" 
    # "Griewank" "Michalewicz" "Powell" "Shekel" "SixHumpCamel"
    # "Sphere" "BucheRastrigin" "LinearSlope" "AttractiveSector" "StepEllipsoid"
    # "SchaffersIllCond" "CompositeGriewankRosenbrock" "Schwefel" "Gallagher21" "Gallagher101"
# )
## Constrained problems
problems=(
    # "ConstrainedGramacy"
    "ConstrainedHartmann"
    # "PressureVessel"
    # "WeldedBeamSO"
    # "TensionCompressionString"
    # "KeaneBumpFunction"
)
# problems=("Ackley" "DixonPrice" "Griewank" "Powell" "Rosenbrock" "StyblinskiTang" "AttractiveSector" "RosenbrockRotated" "Discus" "DifferentPowers" "SchaffersIllCond" "CompositeGriewankRosenbrock")
# method="lmabo"
# method="lmabo-ops"
method="bo"
# method="gphedge"

# constrained_flag=""
constrained_flag="--constrained"

# Loop over each problem and algorithm
for problem in "${problems[@]}"; do
    echo "Submitting job: ${problem}_${method}"
    sbatch --job-name="${problem}_${method}" --export=problem=$problem,method=$method,constrained_flag="$constrained_flag" run.sh
done