from bo import (
    bo_full_loop, 
    gp_hedge_full_loop,
    bo_constrained_full_loop,
    acq_type_mapping, 
    prepare_objective_func,
    prepare_objective_func_constrained
)
from constants import EXP_RUNS, NUMERICAL_RESULTS_DIR
from lmabo import LanguageModelAssistedAdaptiveBO

import argparse
import numpy as np
import os
import torch
from dataclasses import dataclass
from torch.quasirandom import SobolEngine

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double  # Use double precision for GP models

SMOKE_TEST = False  # Set to True for quick testing with fewer runs

def save_results(
    folder_path, 
    exp_idx, 
    train_X, 
    train_Y, 
    simple_regret=None, 
    cum_regret=None, 
    acq_type_list=None, 
    messages=None, 
    weights=None,
    train_constraints=None
):
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
    np.save(f"{folder_path}/{exp_idx}_train_X.npy", train_X)
    np.save(f"{folder_path}/{exp_idx}_train_Y.npy", train_Y)
    if simple_regret is not None:
        np.save(f"{folder_path}/{exp_idx}_simple_regret.npy", simple_regret)
    if cum_regret is not None:
        np.save(f"{folder_path}/{exp_idx}_cum_regret.npy", cum_regret)
    if acq_type_list is not None:
        with open(f"{folder_path}/{exp_idx}_acq_types.txt", "w") as f:
            f.write("\n".join(acq_type_list))
    if messages is not None:
        with open(f"{folder_path}/{exp_idx}_messages.txt", "w") as f:
            f.write("\n".join(messages))
    if weights is not None:
        np.save(f"{folder_path}/{exp_idx}_weights.npy", weights)
    if train_constraints is not None:
        np.save(f"{folder_path}/{exp_idx}_train_constraints.npy", train_constraints)

@dataclass
class ExperimentConfig:
    num_initial_points_multiplier: int = 2
    num_initial_points_offset: int = 1
    num_iterations_low_dim: int = 50
    num_iterations_high_dim: int = 100
    dim_threshold: int = 10
    dtype: torch.dtype = torch.double

def setup_experiment(problem, constrained=False):
    if not constrained:
        """Common setup for main experiments."""
        objective_func, dim, bounds = prepare_objective_func(problem)
        bounds = torch.tensor(objective_func.bounds, dtype=dtype, device=device)
        config = ExperimentConfig()
        num_initial_points = config.num_initial_points_multiplier * dim + config.num_initial_points_offset
        if not SMOKE_TEST:
            num_iterations = config.num_iterations_low_dim if dim <= config.dim_threshold else config.num_iterations_high_dim
        else:
            num_iterations = 5
        return objective_func, bounds, num_initial_points, num_iterations
    elif constrained:
        """Setup for constrained experiments."""
        objective_func, constraint_func, dim, bounds = prepare_objective_func_constrained(problem)
        bounds = torch.tensor(objective_func.bounds, dtype=dtype, device=device)
        config = ExperimentConfig()
        num_initial_points = config.num_initial_points_multiplier * dim + config.num_initial_points_offset
        if not SMOKE_TEST:
            num_iterations = config.num_iterations_low_dim if dim <= config.dim_threshold else config.num_iterations_high_dim
        else:
            num_iterations = 5
        return objective_func, constraint_func, bounds, num_initial_points, num_iterations

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

def generate_initial_data(bounds, num_initial_points, exp_idx, objective_func, constrain_fuc=None):
    """Generate initial training data."""
    train_X = generate_candidates(bounds, num_initial_points, exp_idx)
    if constrain_fuc is None:
        train_Y = objective_func(train_X).unsqueeze(-1)
        return train_X, train_Y
    else:
        train_Y = objective_func(train_X).unsqueeze(-1)
        train_constraints = constrain_fuc(train_X).unsqueeze(-1)
        return train_X, train_Y, train_constraints

def run_problem(
    problem,
    acq_type=None, 
    starting_exp_idx=0,
    server_node="localhost"  # Default to localhost if not specified
):
    print(f"Running {acq_type} on {device}")
    # Experiment setup
    objective_func, bounds, num_initial_points, num_iterations = setup_experiment(problem)
    folder_path = f"{NUMERICAL_RESULTS_DIR}/{problem}/{acq_type}"
    os.makedirs(folder_path, exist_ok=True)
    for exp_idx in range(starting_exp_idx, EXP_RUNS):
        print(f"RUN {exp_idx}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if os.path.exists(f"{folder_path}/{exp_idx}_train_X.npy"):
            print("Completed!")
            continue
        # Generate initial training data
        fixed_train_X, fixed_train_Y  = generate_initial_data(
            bounds, 
            num_initial_points, 
            exp_idx, 
            objective_func
        )
        acq_type_list, messages, weights = None, None, None
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
                llm,
                server_node,
            )
            # optimize and get results
            simple_regret, cum_regret, train_X, train_Y, acq_type_list, messages = LMABO.optimize()
            del LMABO  # Free memory
        elif acq_type == "gphedge":
            # run GP-Hedge
            simple_regret, cum_regret, train_X, train_Y, weights, acq_type_list = gp_hedge_full_loop(
                objective_func,
                list(acq_type_mapping.keys()),
                fixed_train_X, fixed_train_Y,
                bounds,
                num_iterations,
            )
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
            train_X, 
            train_Y,
            simple_regret=simple_regret, 
            cum_regret=cum_regret, 
            acq_type_list=acq_type_list,
            messages=messages,
            weights=weights
        )
        del fixed_train_X, fixed_train_Y, train_X, train_Y  # Free memory

def run_problem_constrained(
    problem,
    acq_type=None,
    starting_exp_idx=0,
):
    print(f"Running {acq_type} on {device}")
    # Experiment setup
    objective_func, constraint_func, bounds, num_initial_points, num_iterations = setup_experiment(problem, constrained=True)
    folder_path = f"{NUMERICAL_RESULTS_DIR}/{problem}/{acq_type}"
    os.makedirs(folder_path, exist_ok=True)
    for exp_idx in range(starting_exp_idx, EXP_RUNS):
        print(f"RUN {exp_idx}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if os.path.exists(f"{folder_path}/{exp_idx}_train_X.npy"):
            print("Completed!")
            continue
        # Generate initial training data
        fixed_train_X, fixed_train_Y, fixed_train_constraints = generate_initial_data(
            bounds, 
            num_initial_points, 
            exp_idx, 
            objective_func,
            constrain_fuc=constraint_func
        )
        all_candidates = generate_candidates(
            bounds, 
            min(100000, 10000*bounds.shape[1]),  # Limit candidates to a reasonable number
            exp_idx
        )
        acq_type_list, messages = None, None
        # run constrained BO
        if "lmabo" in acq_type:
            pass
        else:
            train_X, train_Y, train_constraints = bo_constrained_full_loop(
                objective_func,
                constraint_func,
                acq_type,
                bounds,
                fixed_train_X,
                fixed_train_Y,
                fixed_train_constraints,
                num_iterations,
                all_candidates
            )
        save_results(
            folder_path,
            exp_idx,
            train_X,
            train_Y,
            simple_regret=None,  # No simple regret for constrained BO
            cum_regret=None,  # No cumulative regret for constrained BO
            acq_type_list=acq_type_list,
            messages=messages,
            weights=None,  # No weights for constrained BO
            train_constraints=train_constraints,
        )
        del fixed_train_X, fixed_train_Y, train_X, train_Y, train_constraints  # Free memory

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run optimization experiments")
    parser.add_argument("--problem", type=str, default="Ackley", 
                       help="Function name to run optimization on")
    parser.add_argument("--method", type=str, default="bo",
                       choices=["bo", "lmabo", "lmabo-ops", "gphedge"],
                       help="Optimization method to use")
    parser.add_argument("--constrained", action="store_true",
                       help="Run constrained optimization")
    parser.add_argument("--server_node", type=str, default="localhost",
                       help="Server node for vLLM serving (if applicable)")
    parser.add_argument("--starting_exp_idx", type=int, default=0,
                       help="Starting experiment index")
    return parser.parse_args()

if __name__=="__main__":
    args = parse_arguments()
    starting_exp_idx = max(0, args.starting_exp_idx)
    if args.constrained:
        if args.method == "bo":
            for acq_type in acq_type_mapping.keys():
                run_problem_constrained(args.problem, acq_type, starting_exp_idx)
        else:
            run_problem_constrained(args.problem, args.method, starting_exp_idx)
    else:
        if args.method == "bo":
            for acq_type in acq_type_mapping.keys():
                run_problem(args.problem, acq_type, starting_exp_idx)
        else:
            run_problem(args.problem, args.method, starting_exp_idx) 
