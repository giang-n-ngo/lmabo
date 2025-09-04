#!/bin/bash

methods=(
    "lmabo"
    # "lmabo-ab1"
    # "lmabo-ab2"
    # "lmabo-ab3"
    # "bo" 
    # "gphedge"
    # "esp"
    # "no_past_bo"
    # "setup_bo"
    # "bo_alternating_k1"
    # "bo_alternating_k3"
    # "bo_alternating_k5"
    # "bo_explore_exploit" 
)

# Loop over each problem and algorithm
for method in "${methods[@]}"; do
    echo "Submitting job: ${method}"
    sbatch --job-name="${method}" --export=method=$method slurm_run.sh
done