from bo import bo_full_loop, acq_type_mapping, prepare_objective_func
from constants import EXP_RUNS, NUMERICAL_RESULTS_DIR
from lmabo import lm_assisted_adaptive_bo
from utils import plot_results_best_value

import argparse
from botorch.test_functions import *
import numpy as np
import os
import torch
from torch.quasirandom import SobolEngine

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double  # Use double precision for GP models

def save_results(folder_path, exp_idx, simple_regret, cum_regret, train_X, train_Y):
    """
    Save the results of the optimization run.
    
    Args:
        folder_path: str, path to the folder where results will be saved
        exp_idx: int, index of the experiment run
        simple_regret: numpy array, simple regret values
        cum_regret: numpy array, cumulative regret values
        train_X: torch tensor, training input points
        train_Y: torch tensor, training output values
    """
    np.save(f"{folder_path}/{exp_idx}_simple_regret.npy", simple_regret)
    np.save(f"{folder_path}/{exp_idx}_cum_regret.npy", cum_regret)
    np.save(f"{folder_path}/{exp_idx}_train_X.npy", train_X)
    np.save(f"{folder_path}/{exp_idx}_train_Y.npy", train_Y)

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
        if os.path.exists(f"{folder_path}/{exp_idx}_train_X.npy"):
            print("Completed!")
            continue
        # Generate initial training data
        sobol = SobolEngine(dimension=dim, scramble=True, seed=exp_idx)
        fixed_train_X  = bounds[0] + (bounds[1] - bounds[0]) * sobol.draw(num_initial_points).to(dtype=dtype, device=device)
        fixed_train_Y  = objective_func(fixed_train_X).unsqueeze(-1) # Evaluate function and reshape
        if acq_type != "lmabo":
            # run fixed acq_type
            simple_regret, cum_regret, train_X, train_Y = bo_full_loop(
                objective_func, 
                acq_type, 
                fixed_train_X, fixed_train_Y, 
                bounds,
                num_iterations
            )
            save_results(
                folder_path, 
                exp_idx, 
                simple_regret, 
                cum_regret, 
                train_X, 
                train_Y
            ) 
        else:
            # run LMABO
            simple_regret, cum_regret, train_X, train_Y, acq_type_list, messages = lm_assisted_adaptive_bo(
                objective_func, 
                fixed_train_X, fixed_train_Y, 
                bounds, 
                num_iterations
            )
            save_results(
                folder_path, 
                exp_idx, 
                simple_regret, 
                cum_regret, 
                train_X, 
                train_Y
            ) 
            # save both acq_type_list and messages
            with open(f"{folder_path}/{exp_idx}_acq_types.txt", "w") as f1, \
                open(f"{folder_path}/{exp_idx}_messages.txt", "w") as f2:
                f1.write("\n".join(acq_type_list))
                f2.write("\n".join(messages))

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
        plot_results_best_value(args.problem)