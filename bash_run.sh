#!/bin/bash

methods=(
    # "lmabo"
    "lmabo-gpt"
    # "lmabo-ab1"
    # "lmabo-ab2"
    # "lmabo-ab3"
    # "lmabo-ab4"
    # "lmabo-ab5"
    # "bo" 
    # "gphedge"
    # "gphedge-curated"
    # "esp"
    # "esp-curated"
    # "no_past_bo"
    # "no_past_bo-curated"
    # "setup_bo"
    # "setup_bo-curated"
    # "random_acq"
    # "bo_alternating_k1"
    # "bo_alternating_k3"
    # "bo_alternating_k5"
    # "bo_explore_exploit" 
)

# Loop over each problem and algorithm
for method in "${methods[@]}"; do
    echo "Submitting job: ${method}"
    sbatch --job-name="${method}_1" --export=method=$method slurm_run.sh
done