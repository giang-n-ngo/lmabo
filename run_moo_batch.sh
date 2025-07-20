#!/bin/bash

# Define the problems and algorithms
# problems=(
#     "ZDT1" "ZDT2" "ZDT3"
#     "DTLZ1" "DTLZ2" "BraninCurrin" 
#     "Penicillin" "VehicleSafety" "CarSideImpact"
# )
problems=(
    # "ZDT1" 
    # "ZDT2" 
    # "ZDT3"
    # "DTLZ1"
    # "DTLZ2" 
    # "BraninCurrin"
    # "Penicillin" 
    "VehicleSafety" 
    # "CarSideImpact"
)
# acq_types=(
#     "qNEHVI"
#     "qLogNEHVI"
#     "qHVKG" 
#     "qLBMOJES" 
#     "qLBMOMES" 
#     "qMOPES" 
#     "qParEGO"
# )
acq_types=("lmamoo")
server_node="localhost"
starting_exp_indices=(
    0 
    0 
    0 
    0 
    0 
    0 
    0 
    0 
    0
) # Starting indices for each problem

for i in "${!problems[@]}"; do  # Loop through array indices
    problem="${problems[i]}"
    starting_exp_idx="${starting_exp_indices[i]}"
    
    for acq_type in "${acq_types[@]}"; do
        echo "Submitting job: ${problem}_${acq_type} (starting from exp ${starting_exp_idx})"
        sbatch --job-name="${problem}_${acq_type}" \
               --export=problem=$problem,acq_type=$acq_type,server_node=$server_node,starting_exp_idx=$starting_exp_idx \
               run_moo.sh
    done
done