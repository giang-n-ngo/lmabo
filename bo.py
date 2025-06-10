import numpy as np
import torch
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
from botorch.acquisition.utils import get_optimal_samples
from botorch.generation import MaxPosteriorSampling
from botorch.optim import optimize_acqf
from botorch.models.transforms import Normalize, Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
from torch.quasirandom import SobolEngine

from test_functions import *
from constants import *

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

def prepare_objective_func(problem):
    for item in OBJECTIVE_FUNCTIONS:
        if item.__name__ == problem:
            f = item
            break
        elif hasattr(item, "name") and item.name == problem:
            f = item
            break
    dim = 0
    if f in DIMS:  
        dim = DIMS[f]
        objective_func = f(dim=dim).to(dtype=dtype, device=device)
    else:
        dim = f.dim
        objective_func = f().to(dtype=dtype, device=device)
    bounds = torch.tensor(objective_func.bounds, dtype=dtype, device=device)
    if problem == "Cosine8":
        # Fix incorrect optimal value for Cosine8
        objective_func._optimal_value = -8.8
    return objective_func, dim, bounds

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

def bo_single_iteration(
    train_X,
    train_Y,
    acq_type,
    objective_func,
    bounds
):
    if acq_type in ["qKG", "TS", "qPES", "qMES", "qJES"]:
        flip = -1
    else:
        flip = 1
    dim = bounds.size(1)
    gp = fit_gp(train_X, train_Y*flip)
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
            num_optima=12
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
            candidate_set=torch.rand(1000, dim).to(train_X.dtype).to(train_X.device),
        )
    elif acq_type == "TS":
        acq_func = MaxPosteriorSampling(model=gp, replacement=False)
    else:
        raise ValueError("Invalid acquisition function type")    
    # Optimize the acquisition function to find the next query point
    if acq_type == "TS":
        n_candidates = min(5000, max(2000, 200 * dim))
        sobol = SobolEngine(dim, scramble=True)
        X_cand = bounds[0] + (bounds[1] - bounds[0]) * sobol.draw(n_candidates).to(dtype=dtype, device=device)
        with torch.no_grad():  # We don't need gradients when using TS
            candidate = acq_func(X_cand, num_samples=1)
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
    # Evaluate the function at the new point
    new_Y = objective_func(candidate).unsqueeze(-1)
    # Update the dataset
    train_X = torch.cat([train_X, candidate])
    train_Y = torch.cat([train_Y, new_Y])
    return train_X, train_Y, gp

def bo_full_loop(objective_func, acq_type, X_init, Y_init, bounds, num_iterations):
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
        