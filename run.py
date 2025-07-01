from bo import (
    bo_full_loop, 
    acq_type_mapping, 
    prepare_objective_func,
    gp_hedge_full_loop
)
from constants import EXP_RUNS, NUMERICAL_RESULTS_DIR
from lmabo import LanguageModelAssistedAdaptiveBO

import argparse
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

def run_problem(
    problem,
    acq_type=None, 
    starting_exp_idx=0,
):
    print(f"Running {acq_type}")
    # prepare function
    objective_func, dim, bounds = prepare_objective_func(problem)
    bounds = torch.tensor(objective_func.bounds, dtype=dtype, device=device)  # Search space bounds
    # Experiment settings
    print("Running on ", device)
    num_initial_points = 2*dim + 1
    num_iterations = 50 if dim <= 10 else 100
    folder_path = f"{NUMERICAL_RESULTS_DIR}/{problem}/{acq_type}"
    os.makedirs(folder_path, exist_ok=True)
    for exp_idx in range(starting_exp_idx, EXP_RUNS):
        print(f"RUN {exp_idx}")
        if os.path.exists(f"{folder_path}/{exp_idx}_train_X.npy"):
            print("Completed!")
            continue
        # Generate initial training data
        sobol = SobolEngine(dimension=dim, scramble=True, seed=exp_idx)
        fixed_train_X  = bounds[0] + (bounds[1] - bounds[0]) * sobol.draw(num_initial_points).to(dtype=dtype, device=device)
        fixed_train_Y  = objective_func(fixed_train_X).unsqueeze(-1) # Evaluate function and reshape
        if "lmabo" in acq_type:
            if acq_type == "lmabo":
                llm = "api"
            elif acq_type == "lmabo-ops":
                llm = "ops"
            # run LMABO
            LMABO = LanguageModelAssistedAdaptiveBO(
                objective_func, 
                fixed_train_X, 
                fixed_train_Y, 
                bounds, 
                num_iterations,
                llm
            )
            # optimize and get results
            simple_regret, cum_regret, train_X, train_Y, acq_type_list, messages = LMABO.optimize()
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
        elif acq_type == "gphedge":
            # run GP-Hedge
            simple_regret, cum_regret, train_X, train_Y, weights, acq_type_list = gp_hedge_full_loop(
                objective_func,
                list(acq_type_mapping.keys()),
                fixed_train_X, fixed_train_Y,
                bounds,
                num_iterations,
            )        
            save_results(
                folder_path, 
                exp_idx, 
                simple_regret, 
                cum_regret, 
                train_X, 
                train_Y
            ) 
            np.save(f"{folder_path}/{exp_idx}_weights.npy", weights)
            # save both acq_type_list and messages
            with open(f"{folder_path}/{exp_idx}_acq_types.txt", "w") as f:
                f.write("\n".join(acq_type_list))
        else:
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
        "--starting_exp_idx",
        type=int,
        default=-1,
        help="Which experiment to run",        
    )
    args = argparser.parse_args()
    starting_exp_idx = max(0, args.starting_exp_idx)
    if args.method == "bo":
        for acq_type in acq_type_mapping.keys():
            run_problem(args.problem, acq_type, starting_exp_idx)
    else:
        run_problem(args.problem, args.method, starting_exp_idx) 