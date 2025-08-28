import numpy as np
from baselines.bo_helpers import (
    bo_single_iteration,
    calculate_cumulative_regret
)

def bo_full_loop(objective_func, acq_type, X_init, Y_init, bounds, num_iterations):
    """
    Runs the full Bayesian Optimization loop for a single acquisition function.
    """
    # Generate initial training data
    train_X  = X_init.clone()
    train_Y  = Y_init.clone()
    best_values = [train_Y.min().item()]
    for iteration_idx in range(num_iterations):
        train_X, train_Y, _ = bo_single_iteration(train_X, train_Y, acq_type, objective_func, bounds)
        # Store best observed value
        best_values.append(train_Y.min().item())
        print(f"Iter {iteration_idx} | Current best value: {train_Y.min().item()}")
    return (
        np.array(best_values) - objective_func._optimal_value, # simple regret
        calculate_cumulative_regret(
            train_Y.detach().cpu().numpy(), 
            objective_func._optimal_value
        ), # cumulative regret
        np.array(train_X.detach().cpu().numpy()), 
        np.array(train_Y.detach().cpu().numpy()).flatten()
    )

def bo_alternating_full_loop(objective_func, X_init, Y_init, bounds, num_iterations, k):
    """
    Run the full BO loop but alternate between EI and TS every k iterations
    """
    # Generate initial training data
    train_X  = X_init.clone()
    train_Y  = Y_init.clone()
    best_values = [train_Y.min().item()]
    acq_type = "TS"  # Start with TS
    for iteration_idx in range(num_iterations):
        # after running the current acq_type for k iterations, switch to the other one
        if iteration_idx % k == 0 and iteration_idx > 0:
            acq_type = "EI" if acq_type == "TS" else "TS"
        train_X, train_Y, _ = bo_single_iteration(train_X, train_Y, acq_type, objective_func, bounds)
        # Store best observed value
        best_values.append(train_Y.min().item())
        print(f"Iter {iteration_idx} | Current best value: {train_Y.min().item()}")
    return (
        np.array(best_values) - objective_func._optimal_value, # simple regret
        calculate_cumulative_regret(
            train_Y.detach().cpu().numpy(), 
            objective_func._optimal_value
        ), # cumulative regret
        np.array(train_X.detach().cpu().numpy()), 
        np.array(train_Y.detach().cpu().numpy()).flatten()
    )  

def bo_explore_exploit(objective_func, X_init, Y_init, bounds, num_iterations):
    """
    Run the full BO loop but explore in the first half then exploit in the second half
    """
    # Generate initial training data
    train_X = X_init.clone()
    train_Y = Y_init.clone()
    best_values = [train_Y.min().item()]
    for iteration_idx in range(num_iterations):
        if iteration_idx > num_iterations // 2:
            acq_type = "EI"
        else:
            acq_type = "TS"
        train_X, train_Y, _ = bo_single_iteration(train_X, train_Y, acq_type, objective_func, bounds)
        # Store best observed value
        best_values.append(train_Y.min().item())
        print(f"Iter {iteration_idx} | Current best value: {train_Y.min().item()}")
    return (
        np.array(best_values) - objective_func._optimal_value, # simple regret
        calculate_cumulative_regret(
            train_Y.detach().cpu().numpy(),
            objective_func._optimal_value
        ), # cumulative regret
        np.array(train_X.detach().cpu().numpy()),
        np.array(train_Y.detach().cpu().numpy()).flatten()
    )

def bo_explore_exploit_with_probability(objective_func, X_init, Y_init, bounds, num_iterations):
    """
    Run the full BO loop but prefer exploration in the beginning and exploitation in the end.
    Preference is determined by a probability that changes linearly.
    """
    # Generate initial training data
    train_X = X_init.clone()
    train_Y = Y_init.clone()
    best_values = [train_Y.min().item()]
    for iteration_idx in range(num_iterations):
        exploration_prob = 1 - ((iteration_idx + 1) / num_iterations)
        acq_type = "TS" if np.random.rand() < exploration_prob else "EI"
        train_X, train_Y, _ = bo_single_iteration(train_X, train_Y, acq_type, objective_func, bounds)
        # Store best observed value
        best_values.append(train_Y.min().item())
        print(f"Iter {iteration_idx} | Current best value: {train_Y.min().item()}")
    return (
        np.array(best_values) - objective_func._optimal_value, # simple regret
        calculate_cumulative_regret(
            train_Y.detach().cpu().numpy(),
            objective_func._optimal_value
        ), # cumulative regret
        np.array(train_X.detach().cpu().numpy()),
        np.array(train_Y.detach().cpu().numpy()).flatten()
    )