from bo import bo_full_loop, acq_type_mapping, prepare_objective_func
from constants import EXP_RUNS, NUMERICAL_RESULTS_DIR, FIG_DIR
from lmabo import lm_assisted_adaptive_bo
from utils import plot_results

import argparse
from botorch.test_functions import *
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torch.quasirandom import SobolEngine

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double  # Use double precision for GP models

def check_result_complete(folder_path):
    if not os.path.exists(folder_path):
        count = 0
    else:
        count = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
    if count < EXP_RUNS:
        return False
    else:
        return True

def run_problem(problem, acq_type=None):
    print(f"Running {acq_type}")
    # prepare function
    objective_func, dim, bounds = prepare_objective_func(problem)
    bounds = torch.tensor(objective_func.bounds, dtype=dtype, device=device)  # Search space bounds
    # Experiment settings
    print(device, bounds.device)
    num_initial_points = 2*dim + 1
    num_iterations = 50 if dim <= 10 else 100
    folder_path = f"{NUMERICAL_RESULTS_DIR}/{problem}/{acq_type}"
    os.makedirs(folder_path, exist_ok=True)
    for exp_idx in range(EXP_RUNS):
        print(f"RUN {exp_idx}")
        if os.path.exists(f"{folder_path}/{exp_idx}.npy"):
            print("Completed!")
            continue
        # Generate initial training data
        sobol = SobolEngine(dimension=dim, scramble=True, seed=exp_idx)
        fixed_train_X  = bounds[0] + (bounds[1] - bounds[0]) * sobol.draw(num_initial_points).to(dtype=dtype, device=device)
        fixed_train_Y  = objective_func(fixed_train_X).unsqueeze(-1) # Evaluate function and reshape
        if acq_type != "lmabo":
            # run fixed acq_type
            best_values = bo_full_loop(
                objective_func, 
                acq_type, 
                fixed_train_X, fixed_train_Y, 
                bounds,
                num_iterations
            )
            np.save(f"{folder_path}/{exp_idx}.npy", best_values)  
        else:
            # run LMABO
            best_values, acq_type_list = lm_assisted_adaptive_bo(
                objective_func, 
                fixed_train_X, fixed_train_Y, 
                bounds, 
                num_iterations
            )
            np.save(f"{folder_path}/{exp_idx}.npy", best_values)
            # save acq_type_list as text in one line for analysis
            with open(f"{folder_path}/{exp_idx}_acq_types.txt", "w") as f:
                f.write(" ".join(acq_type_list))

if __name__=="__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--problem",
        type=str,
        default="Ackley",
        help="Function name to run the optimization on",
    )
    argparser.add_argument(
        "--method",
        type=str,
        default="bo",
        help="Whether to run BO or LMABO",
    )
    argparser.add_argument(
        "--plot_flag",
        action="store_true",
        help="Plot the results after running the optimization"
    )
    args = argparser.parse_args()
    if args.method == "bo":
        for acq_type in acq_type_mapping.keys():
            run_problem(args.problem, acq_type)
    elif args.method == "lmabo":
        run_problem(args.problem, "lmabo")
    if args.plot_flag:
        plot_results(args.problem)