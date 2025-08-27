#!/bin/bash

methods=(
    "lmabo"
    # "lmabo-ab1"
    # "lmabo-ab2"
    # "lmabo-ab3"
    # "lmabo2" 
    # "bo" 
    # "bo_alternating_k1"
    # "bo_alternating_k3"
    # "bo_alternating_k5"
    # "bo_explore_exploit" 
    # "bo_explore_exploit_with_probability" 
    # "gphedge"
)

# Loop over each problem and algorithm
for method in "${methods[@]}"; do
    echo "Submitting job: ${method}"
    sbatch --job-name="${method}" --export=method=$method run.sh
done