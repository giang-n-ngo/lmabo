from constants import (
    EXP_RUNS, 
    NUMERICAL_RESULTS_DIR, 
    BOTORCH_FUNCTIONS_NAMES,
    COCO_FUNCTIONS_NAMES,
    HPT_FUNCTIONS_NAMES,
    ACQ_TYPE_MAPPING,
    OPS_MODEL_MAPPING
)   
from test_functions.test_function_loader import load_objective_func

import argparse
import json
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
    choice_list=None,
    messages=None, 
    weights=None,
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
    if choice_list is not None:
        with open(f"{folder_path}/{exp_idx}_choices.txt", "w") as f:
            f.write("\n".join(choice_list))
    if messages is not None:
        with open(f"{folder_path}/{exp_idx}_messages.txt", "w") as f:
            f.write("\n".join(messages))
    if weights is not None:
        np.save(f"{folder_path}/{exp_idx}_weights.npy", weights)

@dataclass
class ExperimentConfig:
    num_initial_points_multiplier: int = 2
    num_initial_points_offset: int = 1
    num_iterations_low_dim: int = 50
    num_iterations_high_dim: int = 100
    dim_threshold: int = 10

@dataclass
class ExperimentConfigHPT:
    # Special configs for hyperparameter tuning
    num_initial_points: int = 10
    num_iterations: int = 50

def prepare_objective_func(problem):
    if problem in BOTORCH_FUNCTIONS_NAMES:
        problem_type = "botorch"
    elif problem in COCO_FUNCTIONS_NAMES:
        problem_type = "coco"
    elif problem in HPT_FUNCTIONS_NAMES:
        problem_type = "hpt"
    objective_func, dim, bounds = load_objective_func(problem, problem_type)
    if problem_type != "hpt":
        objective_func = objective_func.to(dtype=dtype, device=device)
    bounds = objective_func.bounds.clone().detach().to(dtype=dtype, device=device)
    return objective_func, dim, bounds

def setup_experiment(problem):
    """Common setup for main experiments."""
    objective_func, dim, bounds = prepare_objective_func(problem)
    if problem[:4] == "hpt_":
        config = ExperimentConfigHPT()
        num_initial_points = config.num_initial_points
        num_iterations = config.num_iterations
    else:
        config = ExperimentConfig()
        num_initial_points = config.num_initial_points_multiplier * dim + config.num_initial_points_offset
        if not SMOKE_TEST:
            num_iterations = config.num_iterations_low_dim if dim <= config.dim_threshold else config.num_iterations_high_dim
        else:
            num_iterations = 5
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

def generate_initial_data(
        bounds, 
        num_initial_points, 
        exp_idx, 
        objective_func, 
    ):
    """Generate initial training data."""
    if objective_func.name[:4] == "hpt_":
        initial_configs = objective_func.generate_initialization(
            num_initial_points,
            exp_idx
        )
        train_X = torch.tensor(
            [objective_func.map_configs(config) for config in initial_configs], 
            dtype=dtype, 
            device=device
        )
        train_Y = objective_func(train_X).unsqueeze(-1)
    else:
        train_X = generate_candidates(bounds, num_initial_points, exp_idx)
        train_Y = objective_func(train_X).unsqueeze(-1)
    return train_X, train_Y

def context_loader(problem):
    """Load context message for some specific problems."""
    with open("test_functions/contexts.json", "r") as f:
        contexts = json.load(f)
    return contexts.get(problem, "")

def run_problem(
    problem,
    acq_type=None, 
    starting_exp_idx=0,
    server_node="localhost",  # Default to localhost if not specified
    k=None  # Default value for k
):
    print(f"Running {acq_type} on {device}")
    # Experiment setup
    objective_func, bounds, num_initial_points, num_iterations = setup_experiment(problem)
    if acq_type == "bo_alternating":
        folder_path = f"{NUMERICAL_RESULTS_DIR}/{problem}/{acq_type}_k{k}"
    else:
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
        acq_type_list, choice_list, messages, weights = None, None, None, None
        if "curated" in acq_type:
            af_portfolio = ["EI", "LogEI", "TS"]
        else:
            af_portfolio = list(ACQ_TYPE_MAPPING.keys())
        if "lmabo" in acq_type:
            from lmabo import (
                LanguageModelAssistedAdaptiveBO, 
                LanguageModelAssistedAdaptiveBOAblation
            )
            llm = "api" if "ops" not in acq_type else "ops"
            api_type = "gpt" if "gpt" in acq_type else "gemini"
            context_component = context_loader(problem) if "context" in acq_type else ""
            kwargs = {
                "objective_func": objective_func,
                "X_init": fixed_train_X,
                "Y_init": fixed_train_Y,
                "bounds": bounds,
                "num_iterations": num_iterations,
                "llm": llm,
                "server_node": server_node,
                "api_type": api_type,
                "context_component": context_component,
            }
            # run LMABO
            if "-ab" in acq_type:
                ablation_id = int(acq_type[-1])
                LMABO = LanguageModelAssistedAdaptiveBOAblation(
                    ablation_id=ablation_id,
                    **kwargs
                )
            else:
                LMABO = LanguageModelAssistedAdaptiveBO(
                    ops_model_name=OPS_MODEL_MAPPING[acq_type] if llm == "ops" else None,
                    **kwargs
                )
            # optimize and get results
            simple_regret, cum_regret, train_X, train_Y, acq_type_list, messages = LMABO.optimize()
            del LMABO  # Free memory
        elif acq_type in ["gphedge", "gphedge-curated"]:
            from baselines.gp_hedge import gp_hedge_full_loop
            simple_regret, cum_regret, train_X, train_Y, weights, acq_type_list = gp_hedge_full_loop(
                objective_func,
                af_portfolio,
                fixed_train_X, fixed_train_Y,
                bounds,
                num_iterations,
            )   
        elif acq_type in ["esp", "esp-curated"]:
            from baselines.esp import esp_full_loop
            simple_regret, cum_regret, train_X, train_Y, acq_type_list = esp_full_loop(
                objective_func,
                af_portfolio,
                fixed_train_X, fixed_train_Y,
                bounds,
                num_iterations,
            )
        elif acq_type in ["no_past_bo", "no_past_bo-curated"]:
            from baselines.no_past_bo import no_past_bo_full_loop
            simple_regret, cum_regret, train_X, train_Y, weights, acq_type_list = no_past_bo_full_loop(
                objective_func,
                af_portfolio,
                fixed_train_X, fixed_train_Y,
                bounds,
                num_iterations,
            )
        elif acq_type in ["setup_bo", "setup_bo-curated"]:
            from baselines.setup_bo import setup_bo_full_loop
            simple_regret, cum_regret, train_X, train_Y, weights, acq_type_list = setup_bo_full_loop(
                objective_func,
                af_portfolio,
                fixed_train_X, fixed_train_Y,
                bounds,
                num_iterations,
            )
        elif acq_type == "bo_alternating":
            from baselines.bo import bo_alternating_full_loop
            # run alternating BO
            simple_regret, cum_regret, train_X, train_Y = bo_alternating_full_loop(
                objective_func,
                fixed_train_X,
                fixed_train_Y,
                bounds,
                num_iterations,
                k
            )
        elif acq_type == "bo_explore_exploit":
            from baselines.bo import bo_explore_exploit
            # run explore-exploit BO
            simple_regret, cum_regret, train_X, train_Y = bo_explore_exploit(
                objective_func,
                fixed_train_X,
                fixed_train_Y,
                bounds,
                num_iterations,
            )
        elif acq_type == "bo_explore_exploit_with_probability":
            from baselines.bo import bo_explore_exploit_with_probability
            # run explore-exploit BO with probability
            simple_regret, cum_regret, train_X, train_Y = bo_explore_exploit_with_probability(
                objective_func,
                fixed_train_X,
                fixed_train_Y,
                bounds,
                num_iterations,
            )
        elif acq_type == "random_acq":
            from baselines.random_acq import random_acq_full_loop
            simple_regret, cum_regret, train_X, train_Y, acq_type_list = random_acq_full_loop(
                objective_func,
                af_portfolio,
                fixed_train_X, fixed_train_Y,
                bounds,
                num_iterations,
            )
        elif acq_type == "random_acq_curated1":
            from baselines.random_acq import random_acq_full_loop
            simple_regret, cum_regret, train_X, train_Y, acq_type_list = random_acq_full_loop(
                objective_func,
                ["EI", "LogEI", "TS"],
                fixed_train_X, fixed_train_Y,
                bounds,
                num_iterations,
            )    
        elif acq_type == "random_acq_curated2":
            from baselines.random_acq import random_acq_full_loop
            simple_regret, cum_regret, train_X, train_Y, acq_type_list = random_acq_full_loop(
                objective_func,
                ["EI", "TS", "UCB", "PosMean"],
                fixed_train_X, fixed_train_Y,
                bounds,
                num_iterations,
            )            
        else:
            from baselines.bo import bo_full_loop
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
            choice_list=choice_list,
            messages=messages,
            weights=weights
        )
        del fixed_train_X, fixed_train_Y, train_X, train_Y  # Free memory

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run optimization experiments")
    parser.add_argument("--problem", type=str, default="Ackley", 
                       help="Function name to run optimization on")
    parser.add_argument("--method", type=str, default="bo",
                       help="Optimization method to use")
    parser.add_argument("--k", type=int, default=5,
                       help="Number of iterations to run each acquisition function before switching")
    parser.add_argument("--server_node", type=str, default="localhost",
                       help="Server node for vLLM serving (if applicable)")
    parser.add_argument("--starting_exp_idx", type=int, default=0,
                       help="Starting experiment index")
    return parser.parse_args()

if __name__=="__main__":
    args = parse_arguments()
    starting_exp_idx = max(0, args.starting_exp_idx)
    if args.method == "bo":
        for acq_type in ACQ_TYPE_MAPPING.keys():
            run_problem(args.problem, acq_type, starting_exp_idx)
    else:
        run_problem(
            args.problem,
            acq_type=args.method, 
            starting_exp_idx=starting_exp_idx,
            server_node=args.server_node,
            k=args.k
        )
