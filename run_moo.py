from moo import (
    mobo_full_loop, 
    moo_acq_type_mapping, 
    prepare_objective_func_moo, 
    MOO_NUM_ITERATIONS
)
from constants import EXP_RUNS, NUMERICAL_RESULTS_DIR
from lmamoo import lm_assisted_adaptive_moo

import argparse
import numpy as np
import os
import torch
from torch.quasirandom import SobolEngine

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double  # Use double precision for GP models

def save_results(folder_path, exp_idx, hv_list, log_hv_diff_list, train_X, train_Y):
    """
    Save the results of the optimization run.
    
    Args:
        folder_path: str, path to the folder where results will be saved
        exp_idx: int, index of the experiment run
        hv_list: numpy array, hypervolume values
        log_hv_diff_list: numpy array, log hypervolume differences
        train_X: torch tensor, training input points
        train_Y: torch tensor, training output values
    """
    np.save(f"{folder_path}/{exp_idx}_hv.npy", hv_list)
    np.save(f"{folder_path}/{exp_idx}_log_hv_diff.npy", log_hv_diff_list)
    np.save(f"{folder_path}/{exp_idx}_train_X.npy", train_X)
    np.save(f"{folder_path}/{exp_idx}_train_Y.npy", train_Y)

def run_problem(problem, acq_type=None):
    print(f"Running {acq_type}")
    # Prepare function
    objective_func, dim, bounds = prepare_objective_func_moo(problem)
    bounds = torch.tensor(objective_func.bounds, dtype=dtype, device=device)  # Search space bounds
    # Experiment settings
    num_initial_points = 2 * dim + 1
    num_iterations = MOO_NUM_ITERATIONS[problem]
    folder_path = f"{NUMERICAL_RESULTS_DIR}/{problem}/{acq_type}"
    os.makedirs(folder_path, exist_ok=True)
    
    for exp_idx in range(EXP_RUNS):
        print(f"RUN {exp_idx} on {device}")
        if os.path.exists(f"{folder_path}/{exp_idx}_hv.npy"):
            print("Completed!")
            continue
        
        # Generate initial training data
        sobol = SobolEngine(dimension=dim, scramble=True, seed=exp_idx)
        fixed_train_X = bounds[0] + (bounds[1] - bounds[0]) * sobol.draw(num_initial_points).to(dtype=dtype, device=device)
        fixed_train_Y = objective_func(fixed_train_X)  # Evaluate function and reshape
        
        if acq_type != "lmamoo":
            # Run multi-objective optimization loop
            try:
                hv_list, log_hv_diff_list, train_X, train_Y = mobo_full_loop(
                    objective_func,
                    acq_type,
                    fixed_train_X,
                    fixed_train_Y,
                    bounds,
                    num_iterations
                )
            except Exception as e:
                print(f"Error during optimization: {e}")
                continue
        else:
            # Run LMAMOO
            try:
                hv_list, log_hv_diff_list, train_X, train_Y, acq_type_list, messages = lm_assisted_adaptive_moo(
                    objective_func,
                    fixed_train_X,
                    fixed_train_Y,
                    bounds,
                    num_iterations
                )
                # save both acq_type_list and messages
                with open(f"{folder_path}/{exp_idx}_acq_types.txt", "w") as f1, \
                    open(f"{folder_path}/{exp_idx}_messages.txt", "w") as f2:
                    f1.write("\n".join(acq_type_list))
                    f2.write("\n".join(messages))
            except Exception as e:
                print(f"Error during LMAMOO: {e}")
                continue
        save_results(
            folder_path,
            exp_idx,
            hv_list,
            log_hv_diff_list,
            train_X,
            train_Y
        )
        del fixed_train_X, fixed_train_Y, hv_list, log_hv_diff_list, train_X, train_Y

if __name__=="__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--problem",
        type=str,
        default="BraninCurrin",
        help="Function name to run the optimization on",
    )
    argparser.add_argument(
        "--method",
        type=str,
        default="bo",
        help="Whether to run BO or LMABO",
    )
    args = argparser.parse_args()
    if args.method == "moo":
        for acq_type in moo_acq_type_mapping.keys():
            run_problem(args.problem, acq_type)
    elif args.method == "lmamoo":
        run_problem(args.problem, "lmamoo")