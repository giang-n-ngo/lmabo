import torch
import math
import numpy as np

from baselines.bo_helpers import (
    calculate_cumulative_regret,
    fit_gp,
)

from baselines.gp_hedge import _get_nominated_point_and_posterior_mean

dtype = torch.double
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def no_past_bo_full_loop(
    objective_func,
    portfolio_acq_types, # List of strings, e.g., ["EI", "UCB", "PI"]
    X_init,
    Y_init,
    bounds,
    num_iterations,
):
    """
    Implements the No-PASt-BO Bayesian Optimization loop.
    Manages a portfolio of acquisition functions using the Hedge algorithm.
    """
    train_X = X_init.clone()
    train_Y = Y_init.clone()
    eta = 4
    m = 0.8

    N = len(portfolio_acq_types)
    gains = torch.zeros(N, dtype=dtype, device=device) # Initialize cumulative gains for each arm

    best_values = [train_Y.min().item()] # Simple regret values

    # Track probabilities for analysis
    acquisition_function_weights_history = []
    acq_type_list = []
    gp = fit_gp(train_X, train_Y)

    for iteration_idx in range(num_iterations):
        # 1. Build or update the Gaussian Process (GP) model on the *current* data
        nominated_points = []
        rewards_for_gains = [] # Expected GP means at nominated points for Hedge update

        # 2. Nominate points from each acquisition function in the portfolio
        for i, acq_type in enumerate(portfolio_acq_types):
            nominated_x_i, _ = _get_nominated_point_and_posterior_mean(
                gp=gp,
                acq_type=acq_type,
                bounds=bounds,
                best_f=best_values[-1],
            )
            nominated_points.append(nominated_x_i)
            r_min = gains.min().detach().cpu().item()
            r_max = gains.max().detach().cpu().item()
            normalized_r = (gains[i] - r_min) / (r_max - r_min + 1e-8)
            rewards_for_gains.append(normalized_r)

        # 3. Select nominee x_t with probability p_t(j)
        # Calculate probabilities (weights) using the Hedge formula
        exp_gains = torch.exp(torch.abs(eta * gains))
        probabilities = exp_gains / torch.sum(exp_gains)
        acquisition_function_weights_history.append(probabilities.cpu().numpy())

        # Randomly select one acquisition function's nominee based on these probabilities
        # Check for invalid values and fix them
        probabilities = torch.clamp(probabilities, min=1e-8)  # Ensure positive values
        probabilities = torch.nan_to_num(probabilities, nan=1e-8, posinf=1e8, neginf=1e-8)  # Handle NaN/inf

        # Normalize to ensure they sum to 1
        probabilities = probabilities / probabilities.sum()

        # Add final safety check
        if torch.any(torch.isnan(probabilities)) or torch.any(torch.isinf(probabilities)) or torch.any(probabilities < 0):
            # Fallback to uniform distribution if still invalid
            probabilities = torch.ones_like(probabilities) / len(probabilities)

        selected_index = torch.multinomial(probabilities, 1).item()
        x_t = nominated_points[selected_index]
        acq_type_list.append(portfolio_acq_types[selected_index])
        acq_type = portfolio_acq_types[selected_index]

        # 4. Sample the objective function at the selected point
        new_Y_val = objective_func(x_t).unsqueeze(-1)

        # 5. Augment the data
        train_X = torch.cat([train_X, x_t])
        train_Y = torch.cat([train_Y, new_Y_val], dim=0)

        # 6. Fit GP
        gp = fit_gp(train_X, train_Y)

        # 6. Update gains for each acquisition function
        for i, point in enumerate(nominated_points):
            gains[i] = m * gains[i] - gp.posterior(point).mean.detach().squeeze().item()

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
        np.array(acquisition_function_weights_history), # Return weights history
        acq_type_list
    )