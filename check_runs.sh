#!/bin/bash

# Script to check the number of runs for lmabo-ops in each problem directory
# Each run generates 6 files, so number of runs = total files / 6

RESULTS_DIR="/home/s222509501/LMABO/lmabo/numerical_results"
METHOD="lmabo-ops2"

echo "Checking number of runs for $METHOD in each problem..."
echo "=================================================="
printf "%-25s %s\n" "Problem" "Runs"
echo "=================================================="

total_runs=0
total_problems=0

# Loop through all problem directories
for problem_dir in "$RESULTS_DIR"/*/ ; do
    if [ -d "$problem_dir" ]; then
        problem_name=$(basename "$problem_dir")
        method_dir="$problem_dir$METHOD"
        
        if [ -d "$method_dir" ]; then
            # Count files in the lmabo-ops directory
            file_count=$(find "$method_dir" -maxdepth 1 -type f | wc -l)
            
            # Calculate number of runs (6 files per run)
            runs=$((file_count / 6))
            
            printf "%-25s %d\n" "$problem_name" "$runs"
            
            total_runs=$((total_runs + runs))
            total_problems=$((total_problems + 1))
        else
            printf "%-25s %s\n" "$problem_name" "No data"
        fi
    fi
done

echo "=================================================="
printf "%-25s %d\n" "Total problems:" "$total_problems"
printf "%-25s %d\n" "Total runs:" "$total_runs"

if [ $total_problems -gt 0 ]; then
    avg_runs=$((total_runs / total_problems))
    printf "%-25s %d\n" "Average runs/problem:" "$avg_runs"
fi

echo ""
echo "Note: Each run generates 6 files:"
echo "  - N_acq_types.txt"
echo "  - N_cum_regret.npy"
echo "  - N_messages.txt"
echo "  - N_simple_regret.npy"
echo "  - N_train_X.npy"
echo "  - N_train_Y.npy"
