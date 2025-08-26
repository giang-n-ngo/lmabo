import numpy as np
import torch
import math
from botorch.test_functions import *
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.analytic import (
    ProbabilityOfImprovement, 
    LogProbabilityOfImprovement,
    ExpectedImprovement, 
    LogExpectedImprovement, 
    UpperConfidenceBound,
    PosteriorMean,
    PosteriorStandardDeviation
)
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
from botorch.acquisition.predictive_entropy_search import qPredictiveEntropySearch
from botorch.acquisition.joint_entropy_search import qJointEntropySearch
from botorch.acquisition.max_value_entropy_search import qLowerBoundMaxValueEntropy
from botorch.generation import MaxPosteriorSampling
from botorch.acquisition.utils import get_optimal_samples
from botorch.optim import (
    optimize_acqf,
    optimize_acqf_discrete
)
from botorch.models.transforms import Normalize, Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
from torch.quasirandom import SobolEngine

# Set device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double  # Use double precision for GP models

import warnings
warnings.filterwarnings("ignore")

acq_type_mapping = {
    "PI": "Probability of Improvement",
    "LogPI": "Log Probability of Improvement",
    "EI": "Expected Improvement",
    "LogEI": "Log Expected Improvement",
    "UCB": "Upper Confidence Bound",
    "PosMean": "Posterior Mean",
    "PosSTD": "Posterior Standard Deviation",
    "TS": "Thompson Sampling",
    "qKG": "Knowledge Gradient",
    "qPES": "Predictive Entropy Search",
    "qMES": "Max-value Entropy Search",
    "qJES": "Joint Entropy Search"
}
MC_SAMPLES = 128

def calculate_auc_simple_regret(best_values, true_minimum):
    """
    Calculate the AUC of the simple regret curve for a minimization problem.
    
    Args:
        best_values: numpy array of best observed values at each iteration
        true_minimum: float, true global minimum of the objective function
        
    Returns:
        AUC value as a float
    """
    # Calculate simple regret at each iteration
    simple_regret = best_values - true_minimum
    
    # Calculate cumulative regret
    auc_regret = np.cumsum(simple_regret)[-1]
    
    return auc_regret

def calculate_cumulative_regret(observations, true_minimum):
    """
    Calculate the cumulative regret over the iterations.
    
    Args:
        observations: numpy array of observed values at each iteration
        true_minimum: float, true global minimum of the objective function
        
    Returns:
        Cumulative regret as a numpy array
    """
    # Calculate cumulative regret
    cumulative_regret = np.cumsum(observations - true_minimum)
    
    return cumulative_regret

def fit_gp(X, Y):
    nrows = X.shape[0]
    ncols = X.shape[1]
    X = X.reshape((nrows, ncols))
    Y = Y.reshape((nrows, 1))  # Ensure Y is a column vector
    gp = SingleTaskGP(
        X, 
        Y, 
        covar_module=ScaleKernel(
            MaternKernel(
                ard_num_dims=X.shape[-1]
            )
        ), 
        input_transform=Normalize(d=X.shape[-1]),
        outcome_transform=Standardize(m=1)
    )
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    return gp

def fit_gp_list(X, Y_list):
    """
    Fit a list of GPs for each output in Y_list.
    This is useful for multi-output problems.
    """
    gplist = []
    for i, Y in enumerate(Y_list):
        gp = SingleTaskGP(
            X, 
            Y, 
            covar_module=ScaleKernel(
                MaternKernel(
                    ard_num_dims=X.shape[-1]
                )
            ), 
            input_transform=Normalize(d=X.shape[-1]),
            outcome_transform=Standardize(m=1)
        )
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        gplist.append(gp)
    return gplist

def _prepare_acquisition_function(acq_type, bounds, train_X, train_Y, gp, flip):
    if acq_type == "PI":
        acq_func = ProbabilityOfImprovement(model=gp, best_f=train_Y.min(), maximize=False)
    elif acq_type == "LogPI":
        acq_func = LogProbabilityOfImprovement(model=gp, best_f=train_Y.min(), maximize=False)
    elif acq_type == "EI":
        acq_func = ExpectedImprovement(model=gp, best_f=train_Y.min(), maximize=False)
    elif acq_type == "LogEI":
        acq_func = LogExpectedImprovement(model=gp, best_f=train_Y.min(), maximize=False)
    elif acq_type == "UCB":
        acq_func = UpperConfidenceBound(model=gp, beta=10*math.log(train_Y.shape[0]), maximize=False)
    elif acq_type == "PosMean":
        acq_func = PosteriorMean(model=gp, maximize=False)
    elif acq_type == "PosSTD":
        acq_func = PosteriorStandardDeviation(model=gp, maximize=False)
    elif acq_type == "qKG":
        acq_func = qKnowledgeGradient(model=gp, num_fantasies=4)
    elif acq_type in ["qPES", "qJES"]:
        gp_cpu = fit_gp(
            train_X.cpu(),
            train_Y.cpu() * flip
        )
        optimal_inputs, optimal_outputs = get_optimal_samples(
            model=gp_cpu, 
            bounds=bounds.cpu(), 
            num_optima=4
        )
        del gp_cpu  # Free memory
        if acq_type == "qPES":
            acq_func = qPredictiveEntropySearch(
                model=gp, 
                optimal_inputs=optimal_inputs.to(train_X), 
            )
        elif acq_type == "qJES":
            acq_func = qJointEntropySearch(
                model=gp,
                optimal_inputs=optimal_inputs.to(device).to(dtype),
                optimal_outputs=optimal_outputs.to(device).to(dtype),
                estimation_type="LB",
            )
    elif acq_type == "qMES":
        acq_func = qLowerBoundMaxValueEntropy(
            model=gp,
            candidate_set=torch.rand(1000, bounds.size(1)).to(train_X.dtype).to(train_X.device),
        )
    elif acq_type == "TS":
        acq_func = MaxPosteriorSampling(gp)
    else:
        print(f"Invalid acquisition function type: {acq_type}")
        raise ValueError("Invalid acquisition function type")    
    return acq_func

def _optimize_acqf(acq_type, acq_func, bounds):
    if acq_type == "TS":
        n_candidates = min(5000, max(2000, 200 * bounds.size(1)))
        sobol = SobolEngine(bounds.size(1), scramble=True)
        X_cand = bounds[0] + (bounds[1] - bounds[0]) * sobol.draw(n_candidates).to(dtype=dtype, device=device)
        candidate = acq_func(X_cand, num_samples=1)
    elif acq_type == "qPES":
        n_candidates = min(1000, max(1000, 200 * bounds.size(1)))
        sobol = SobolEngine(bounds.size(1), scramble=True)
        X_cand = bounds[0] + (bounds[1] - bounds[0]) * sobol.draw(n_candidates).to(dtype=dtype, device=device)
        acq_val = acq_func(X_cand)
        candidate = X_cand[torch.argmax(acq_val)].unsqueeze(0)
    else:
        optimize_acqf_kwargs = {
            "acq_function": acq_func,
            "bounds": bounds,
            "q": 1,
            "num_restarts": 10,
            "raw_samples": 512
        }
        if acq_type == "qPES":
            candidate, _ = optimize_acqf(
                options={"with_grad": False},
                **optimize_acqf_kwargs
            )
        else:
            candidate, _ = optimize_acqf(**optimize_acqf_kwargs)
    return candidate

def bo_single_iteration(
    train_X,
    train_Y,
    acq_type,
    objective_func,
    bounds
):
    """
    Performs a single Bayesian Optimization iteration for a given acquisition function type.
    This function is primarily used by individual (non-portfolio) BO strategies.
    For GP-Hedge, a modified version `bo_single_iteration_gph` is used to get nominated points.
    """
    if acq_type in ["qKG", "TS", "qPES", "qMES", "qJES"]:
        flip = -1
    else:
        flip = 1
    gp = fit_gp(train_X, train_Y*flip)
    acq_func = _prepare_acquisition_function(acq_type, bounds, train_X, train_Y, gp, flip)
    # Optimize the acquisition function to find the next query point
    candidate = _optimize_acqf(acq_type, acq_func, bounds)
    # Evaluate the function at the new point
    new_Y = objective_func(candidate).unsqueeze(-1)
    # Update the dataset
    train_X = torch.cat([train_X, candidate])
    train_Y = torch.cat([train_Y, new_Y])
    return train_X, train_Y, gp

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
        
def _get_nominated_point_and_posterior_mean(
    train_X, # Used for getting current best_f
    train_Y, # Used for getting current best_f
    acq_type,
    bounds,
):
    """
    Helper function to get a nominated point and its posterior mean for a single acquisition function.
    This is designed to be called by GP-Hedge for each 'arm'.
    """
    if acq_type in ["qKG", "TS", "qPES", "qMES", "qJES"]:
        flip = -1
    else:
        flip = 1
    gp = fit_gp(train_X, train_Y*flip)
    acq_func = _prepare_acquisition_function(acq_type, bounds, train_X, train_Y, gp, flip)
    # Optimize the acquisition function
    candidate = _optimize_acqf(acq_type, acq_func, bounds)

    # Get the posterior mean at the nominated point
    posterior = gp.posterior(candidate)
    posterior_mean = posterior.mean.item()*flip
    return candidate, posterior_mean

def gp_hedge_full_loop(
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
    eta = 10**(-2 - max(int(math.floor(math.log10(torch.abs(train_Y).max().cpu().item()))), 0))

    N = len(portfolio_acq_types)
    gains = torch.zeros(N, dtype=dtype, device=device) # Initialize cumulative gains for each arm

    best_values = [train_Y.min().item()] # Simple regret values

    # Track probabilities for analysis
    acquisition_function_weights_history = []
    acq_type_list = []

    for iteration_idx in range(num_iterations):
        # 1. Build or update the Gaussian Process (GP) model on the *current* data
        # Note: GP-Hedge generally implies minimizing objective, so no Y flipping here directly.
        # Flipping for specific ACQ functions (like qKG) happens inside _get_nominated_point_and_posterior_mean.
        nominated_points = []
        rewards_for_gains = [] # Expected GP means at nominated points for Hedge update

        # 2. Nominate points from each acquisition function in the portfolio
        for i, acq_type in enumerate(portfolio_acq_types):
            nominated_x_i, posterior_mean_i = _get_nominated_point_and_posterior_mean(
                train_X=train_X,
                train_Y=train_Y,
                acq_type=acq_type,
                bounds=bounds,
            )
            nominated_points.append(nominated_x_i)
            rewards_for_gains.append(posterior_mean_i) 

        # 3. Select nominee x_t with probability p_t(j)
        # Calculate probabilities (weights) using the Hedge formula
        exp_gains = torch.exp(torch.abs(eta * gains))
        probabilities = exp_gains / torch.sum(exp_gains)
        acquisition_function_weights_history.append(probabilities.cpu().numpy())

        # Randomly select one acquisition function's nominee based on these probabilities
        selected_index = torch.multinomial(probabilities, 1).item()
        x_t = nominated_points[selected_index]
        acq_type_list.append(portfolio_acq_types[selected_index])

        # 4. Sample the objective function at the selected point
        new_Y_val = objective_func(x_t).unsqueeze(-1)

        # 5. Augment the data
        train_X = torch.cat([train_X, x_t])
        train_Y = torch.cat([train_Y, new_Y_val], dim=0)

        # 6. Update gains for each acquisition function
        gains = gains - torch.tensor(rewards_for_gains, dtype=dtype, device=device)

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

def optimize_acqf_discrete_qKG(
    bounds,
    acq_function,
    choices=None,
    batch_size=256
):
    """
    Optimize an acquisition function over a discrete set of choices.
    This is a wrapper around the BoTorch optimize_acqf_discrete function.
    """
    if choices is None:
        raise ValueError("choices must be provided for discrete optimization.")
   
    # Get acq value for each candidate in batch using for loop with a maximum batch_size
    acq_values = torch.cat([acq_function.evaluate(X=X_, bounds=bounds) for X_ in choices.split(batch_size)])
    # Find the candidate with the maximum acquisition value
    max_idx = torch.argmax(acq_values)
    candidate = choices[max_idx].unsqueeze(0)  # Get the candidate corresponding to
    
    return candidate

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