from moo import (
    mobo_full_loop, 
    prepare_objective_func_moo, 
    MOO_NUM_ITERATIONS
)
from constants import EXP_RUNS, NUMERICAL_RESULTS_DIR
from lmamoo import LanguageModelAssistedAdaptiveMOO

import argparse
import numpy as np
import os
import torch
from torch.quasirandom import SobolEngine

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double  # Use double precision for GP models

def save_results(
    folder_path, 
    exp_idx, 
    hv_list, 
    log_hv_diff_list, 
    train_X, 
    train_Y,
    acq_type_list=None,
    messages=None
):
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
    np.save(f"{folder_path}/{exp_idx}_train_X.npy", train_X)
    np.save(f"{folder_path}/{exp_idx}_train_Y.npy", train_Y)
    np.save(f"{folder_path}/{exp_idx}_hv.npy", hv_list)
    np.save(f"{folder_path}/{exp_idx}_log_hv_diff.npy", log_hv_diff_list)
    if acq_type_list is not None:
        np.save(f"{folder_path}/{exp_idx}_acq_type_list.npy", acq_type_list)
    if messages is not None:
        with open(f"{folder_path}/{exp_idx}_messages.txt", "w") as f:
            f.write("\n".join(messages))

def setup_experiment(problem):
    objective_func, dim, bounds = prepare_objective_func_moo(problem)
    bounds = torch.tensor(objective_func.bounds, dtype=dtype, device=device)  # Search space bounds
    num_initial_points = 2 * dim + 1
    num_iterations = MOO_NUM_ITERATIONS[problem]
    return objective_func, bounds, num_initial_points, num_iterations

def generate_candidates(bounds, num_candidates, exp_idx):
    """
    Generate a set of candidate points uniformly distributed within the bounds.
    
    Args:
        bounds: torch tensor, shape [2, dim], lower and upper bounds for each dimension
        num_candidates: int, number of candidate points to generate
    
    Returns:
        candidates: torch tensor, shape [num_candidates, dim], generated candidate points
    """
    sobol = SobolEngine(dimension=bounds.shape[1], scramble=True, seed=exp_idx)
    candidates = bounds[0] + (bounds[1] - bounds[0]) * sobol.draw(num_candidates).to(dtype=dtype, device=device)
    return candidates

def generate_initial_data(objective_func, bounds, num_initial_points, exp_idx):
    """
    Generate initial training data for the optimization.
    
    Args:
        objective_func: callable, the objective function to evaluate
        bounds: torch tensor, shape [2, dim], lower and upper bounds for each dimension
        num_initial_points: int, number of initial points to generate
        exp_idx: int, experiment index for reproducibility
    
    Returns:
        fixed_train_X: torch tensor, shape [num_initial_points, dim], initial training inputs
        fixed_train_Y: torch tensor, shape [num_initial_points, num_objectives], initial training outputs
    """
    fixed_train_X = generate_candidates(bounds, num_initial_points, exp_idx).to(dtype=dtype, device=device)
    fixed_train_Y = objective_func(fixed_train_X)  # Evaluate function and reshape
    fixed_train_Y = fixed_train_Y.to(dtype=dtype, device=device)
    return fixed_train_X, fixed_train_Y

def run_problem(
    problem, 
    acq_type,
    starting_exp_idx=0,
    server_node="localhost"
):
    print(f"Running {acq_type} on {device}")
    # Experiment setup
    objective_func, bounds, num_initial_points, num_iterations = setup_experiment(problem)
    folder_path = f"{NUMERICAL_RESULTS_DIR}/{problem}/{acq_type}"
    os.makedirs(folder_path, exist_ok=True)
    for exp_idx in range(starting_exp_idx, EXP_RUNS):
        print(f"RUN {exp_idx} on {device}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if os.path.exists(f"{folder_path}/{exp_idx}_hv.npy"):
            print("Completed!")
            continue
        # Generate initial training data
        fixed_train_X, fixed_train_Y = generate_initial_data(
            objective_func, 
            bounds, 
            num_initial_points, 
            exp_idx
        )
        acq_type_list, messages = None, None
        
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
                llm = "api" if "ops" not in acq_type else "ops"
                LMAMOO = LanguageModelAssistedAdaptiveMOO(
                    objective_func,
                    fixed_train_X,
                    fixed_train_Y,
                    bounds,
                    num_iterations,
                    llm=llm,
                    server_node=server_node
                )
                hv_list, log_hv_diff_list, train_X, train_Y, acq_type_list, messages = LMAMOO.optimize()
                del LMAMOO
            except Exception as e:
                print(f"Error during LMAMOO: {e}")
                continue
        save_results(
            folder_path,
            exp_idx,
            hv_list,
            log_hv_diff_list,
            train_X,
            train_Y,
            acq_type_list,
            messages
        )
        del fixed_train_X, fixed_train_Y, hv_list, log_hv_diff_list, train_X, train_Y

def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        args: Namespace, parsed arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--problem",
        type=str,
        default="BraninCurrin",
        help="Function name to run the optimization on",
    )
    parser.add_argument(
        "--acq_type",
        type=str,
        default="qNEHVI",
        help="Which acquisition type to use for MOO",
    )
    parser.add_argument(
        "--server_node", 
        type=str, 
        default="localhost",
        help="Server node for vLLM serving (if applicable)"
    )
    parser.add_argument(
        "--starting_exp_idx", 
        type=int, 
        default=0,
        help="Starting experiment index"
    )
    return parser.parse_args()

if __name__=="__main__":
    args = parse_arguments()
    starting_exp_idx = max(0, args.starting_exp_idx)
    run_problem(args.problem, args.acq_type, starting_exp_idx, args.server_node)