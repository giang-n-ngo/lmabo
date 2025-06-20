#!/bin/bash

# Define the problems and algorithms
# problems=(
#     "ZDT1" "ZDT2" "ZDT3"
#     "DTLZ1" "DTLZ2" "BraninCurrin" 
#     "Penicillin" "VehicleSafety" "CarSideImpact"
# )
problems=("DTLZ1" "DTLZ2" "Penicillin" "CarSideImpact")
method="moo"

# Loop over each problem and algorithm
for problem in "${problems[@]}"; do
    echo "Submitting job: ${problem}_${method}"
    sbatch --job-name="${problem}_${method}" --export=problem=$problem,method=$method run_moo.sh
done