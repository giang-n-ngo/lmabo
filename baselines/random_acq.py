import torch
import math
import numpy as np

from baselines.bo_helpers import (
    bo_single_iteration,
    calculate_cumulative_regret
)

dtype = torch.double
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def random_acq_full_loop(
    objective_func,
    portfolio_acq_types, # List of strings, e.g., ["EI", "UCB", "PI"]
    X_init,
    Y_init,
    bounds,
    num_iterations,
):
    """
    Implements the GP-Hedge Bayesian Optimization loop.
    Manages a portfolio of acquisition functions using the Hedge algorithm.
    """
    train_X = X_init.clone()
    train_Y = Y_init.clone()

    N = len(portfolio_acq_types)

    best_values = [train_Y.min().item()] # Simple regret values

    # Track probabilities for analysis
    acq_type_list = []

    for iteration_idx in range(num_iterations):
        # Randomly select an acquisition function
        selected_index = np.random.randint(0, N)
        acq_type = portfolio_acq_types[selected_index]
        acq_type_list.append(acq_type)
        # Perform a single BO iteration with the selected acquisition function
        train_X, train_Y, _ = bo_single_iteration(train_X, train_Y, acq_type, objective_func, bounds)

        # Store best observed value
        best_values.append(train_Y.min().item())
        print(f"Iter {iteration_idx+1} | Selected Acq: {portfolio_acq_types[selected_index]} | Current best value: {train_Y.min().item()}")

    return (
        np.array(best_values) - objective_func._optimal_value, # simple regret
        calculate_cumulative_regret(
            train_Y.detach().cpu().numpy(),
            objective_func._optimal_value
        ), # cumulative regret
        np.array(train_X.detach().cpu().numpy()),
        np.array(train_Y.detach().cpu().numpy()).flatten(),
        acq_type_list
    )