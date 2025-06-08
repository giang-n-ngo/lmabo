from bo import bo_full_loop, acq_type_mapping, prepare_objective_func
from lmabo import lm_assisted_adaptive_bo
from constants import DIMS, EXP_RUNS, NUMERICAL_RESULTS_DIR, FIG_DIR, OBJECTIVE_FUNCTIONS

import argparse
from botorch.test_functions import *
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import torch
from torch.quasirandom import SobolEngine

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double  # Use double precision for GP models

matplotlib_colors = [
    '#1f77b4',  # Muted blue (from default 'tab10' palette)
    '#ff7f0e',  # Orange
    '#2ca02c',  # Green
    '#d62728',  # Red
    '#9467bd',  # Purple
    '#8c564b',  # Brown
    '#e377c2',  # Pink
    '#7f7f7f',  # Gray
    '#bcbd22',  # Olive/Lime Green
    '#17becf',  # Cyan/Teal
    '#a6cee3',  # Light Blue (from 'Paired' palette)
    '#b2df8a',  # Light Green
    '#fb9a99',  # Light Red
    '#fdbf6f',  # Light Orange
    '#cab2d6'   # Light Purple
]

def plot_results(problem_name):
    legends = list(acq_type_mapping.keys())
    legends.append("lmabo")
    plt.figure(figsize=(16, 10))
    for i, acq_type in enumerate(legends):
        acq_type_values = []
        for exp_idx in range(EXP_RUNS):
            try:
                file_name = f"{NUMERICAL_RESULTS_DIR}/{problem_name}/{acq_type}/{exp_idx}.npy"
                best_values = np.load(file_name)
                acq_type_values.append(best_values)
            except:
                print(file_name + " does not exist")
        if len(acq_type_values) > 0:
            acq_type_values = np.stack(acq_type_values)
            mean_acq_type_values = acq_type_values.mean(axis=0)
            std_acq_type_values = acq_type_values.std(axis=0)
            n_iter = np.arange(1, mean_acq_type_values.shape[0]+1)
            plt.plot(
                n_iter, 
                mean_acq_type_values, 
                color=matplotlib_colors[i],
                label=acq_type
            )
            plt.fill_between(
                n_iter, 
                mean_acq_type_values - 0.5*std_acq_type_values, 
                mean_acq_type_values + 0.5*std_acq_type_values, 
                color=matplotlib_colors[i],
                alpha=0.2
            )
    plt.xlabel("Iteration")
    plt.ylabel("Best Function Value Found")
    plt.title(f"BO on {problem_name}")
    plt.legend()
    plt.grid(True)
    os.makedirs(f"{FIG_DIR}", exist_ok=True)
    plt.savefig(f"{FIG_DIR}/{problem_name}.pdf", dpi=300)  # Save plot as PDF

def check_result_complete(folder_path):
    if not os.path.exists(folder_path):
        count = 0
    else:
        count = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
    if count < EXP_RUNS:
        return False
    else:
        return True

if __name__=="__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--problem",
        type=str,
        default="Ackley",
        help="Function name to run the optimization on",
    )
    argparser.add_argument(
        "--bo_flag",
        action="store_true",
        help="Use Bayesian Optimization"
    )
    argparser.add_argument(
        "--lmabo_flag",
        action="store_true",
        help="Use LM-assisted Adaptive Bayesian Optimization"
    )
    argparser.add_argument(
        "--plot_flag",
        action="store_true",
        help="Plot the results after running the optimization"
    )
    args = argparser.parse_args()
    # prepare function
    objective_func, dim, bounds = prepare_objective_func(args.problem)
    # Experiment settings
    print(device, bounds.device)
    
    num_initial_points = 2*dim + 1
    num_iterations = 50 if dim <= 10 else 100
    for exp_idx in range(EXP_RUNS):
        print(f"RUN {exp_idx}")
        # Generate initial training data
        sobol = SobolEngine(dimension=dim, scramble=True, seed=exp_idx)
        fixed_train_X  = bounds[0] + (bounds[1] - bounds[0]) * sobol.draw(num_initial_points).to(dtype=dtype, device=device)
        fixed_train_Y  = objective_func(fixed_train_X).unsqueeze(-1) # Evaluate function and reshape
        if args.bo_flag:
            # run fixed acq_type
            for acq_type, acq_name in acq_type_mapping.items():
                print(f"Running {acq_name} ({acq_type})")
                folder_path = f"{NUMERICAL_RESULTS_DIR}/{args.problem}/{acq_type}"
                if os.path.exists(f"{folder_path}/{exp_idx}.npy"):
                    print("Completed!")
                    continue
                best_values = bo_full_loop(
                    objective_func, 
                    acq_type, 
                    fixed_train_X, fixed_train_Y, 
                    bounds,
                    num_iterations
                )
                os.makedirs(folder_path, exist_ok=True)
                np.save(f"{folder_path}/{exp_idx}.npy", best_values)
        if args.lmabo_flag:
            # run LMABO
            print("Run LMABO")
            folder_path = f"{NUMERICAL_RESULTS_DIR}/{args.problem}/lmabo"
            if os.path.exists(f"{folder_path}/{exp_idx}.npy"):
                continue
            best_values, acq_type_list = lm_assisted_adaptive_bo(
                objective_func, 
                fixed_train_X, 
                fixed_train_Y, 
                bounds, 
                num_iterations
            )
            os.makedirs(folder_path, exist_ok=True)
            np.save(f"{folder_path}/{exp_idx}.npy", best_values)
            # save acq_type_list as text in one line for analysis
            with open(f"{folder_path}/{exp_idx}_acq_types.txt", "w") as f:
                f.write(" ".join(acq_type_list))
    if args.plot_flag:
        plot_results(args.problem)