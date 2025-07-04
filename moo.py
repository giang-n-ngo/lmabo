"""
Operation code for multi-objective optimization using Bayesian Optimization (BO) with Gaussian Processes (GP).
"""

from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms import Standardize
from botorch.utils.transforms import unnormalize, normalize
from botorch.utils.sampling import sample_simplex
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch import fit_gpytorch_mll
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.acquisition.objective import GenericMCObjective
from botorch.acquisition.multi_objective.monte_carlo import (
    qNoisyExpectedHypervolumeImprovement
)
from botorch.acquisition.multi_objective.logei import (
    qLogNoisyExpectedHypervolumeImprovement
)
from botorch.acquisition.multi_objective.hypervolume_knowledge_gradient import (
    qHypervolumeKnowledgeGradient
)
from botorch.acquisition.multi_objective.joint_entropy_search import (
    qLowerBoundMultiObjectiveJointEntropySearch
)
from botorch.acquisition.multi_objective.max_value_entropy_search import (
    qLowerBoundMultiObjectiveMaxValueEntropySearch
)
from botorch.acquisition.multi_objective.predictive_entropy_search import (
    qMultiObjectivePredictiveEntropySearch
)
from botorch.optim.optimize import optimize_acqf, optimize_acqf_discrete
from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
)
from botorch.acquisition.multi_objective.utils import (
    compute_sample_box_decomposition,
    random_search_optimizer,
    sample_optimal_points,
)
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.exceptions.warnings import (
    BadInitialCandidatesWarning
)
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
from torch import Tensor
from torch.quasirandom import SobolEngine
import torch
import numpy as np
import gpytorch
import warnings

from botorch.test_functions.multi_objective import (
    ZDT1,
    ZDT2,
    ZDT3,
    DTLZ1,
    DTLZ2,
    BraninCurrin,
    Penicillin,
    VehicleSafety,
    CarSideImpact
)

warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double  # Use double precision for GP models
tkwargs = {
    "dtype": dtype,
    "device": device
}

NUM_RESTARTS = 10
RAW_SAMPLES = 512
MC_SAMPLES = 128
MAX_CHOLESKY_SIZE = float('inf')

moo_acq_type_mapping = {
    "qNEHVI": "qNoisyExpectedHypervolumeImprovement",
    "qLogNEHVI": "qLogNoisyExpectedHypervolumeImprovement",
    "qHVKG": "qHypervolumeKnowledgeGradient",
    "qLBMOJES": "qLowerBoundMultiObjectiveJointEntropySearch",
    "qLBMOMES": "qLowerBoundMultiObjectiveMaxValueEntropySearch",
    "qMOPES": "qMultiObjectivePredictiveEntropySearch",
    "qParEGO": "qParEGO",
}

MOO_OBJECTIVE_FUNCTIONS = [
    ZDT1,
    ZDT2,
    ZDT3,
    DTLZ1,
    DTLZ2,
    BraninCurrin,
    Penicillin,
    VehicleSafety,
    CarSideImpact
]

MOO_TEST_CONFIGS = {
    # ZDT test suite (2 objectives)
    "ZDT1": {"dim": 5},
    "ZDT2": {"dim": 5},
    "ZDT3": {"dim": 5},
    
    # DTLZ test suite 
    "DTLZ1": {"dim": 10, "num_objectives": 4},  
    "DTLZ2": {"dim": 10, "num_objectives": 4},  
    "DTLZ5": {"dim": 10, "num_objectives": 4},  
}

MOO_NUM_ITERATIONS = {
    "ZDT1": 20,
    "ZDT2": 20,
    "ZDT3": 50,
    "DTLZ1": 50,
    "DTLZ2": 200,
    "BraninCurrin": 50,
    "Penicillin": 200,
    "VehicleSafety": 200,
    "CarSideImpact": 200
}

def prepare_objective_func_moo(problem):
    for item in MOO_OBJECTIVE_FUNCTIONS:
        if item.__name__ == problem:
            f = item
            break
    f_kwargs = MOO_TEST_CONFIGS.get(problem, {})
    objective_func = f(negate=True, **f_kwargs).to(dtype=dtype, device=device)
    dim = objective_func.dim
    bounds = objective_func.bounds.to(dtype=dtype, device=device)
    return objective_func, dim, bounds

def fit_moo_gp(X, train_obj, bounds):
    # define models for objective and constraint
    normed_X = normalize(X, bounds)
    models = []
    for i in range(train_obj.shape[-1]):
        train_y = train_obj[:, i : i + 1]
        models.append(
            SingleTaskGP(
                normed_X, 
                train_y, 
                covar_module=ScaleKernel(
                    MaternKernel(
                        ard_num_dims=normed_X.shape[-1]
                    )
                ), 
                outcome_transform=Standardize(m=1)
            )
        )
    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    with gpytorch.settings.max_cholesky_size(MAX_CHOLESKY_SIZE):
        fit_gpytorch_mll(mll)
    model.eval()
    return model

def get_moo_results(obj_values, ref_point, max_hv):
    """
    Calculate hypervolume and log hypervolume difference.

    Args:
        obj_values: Objective values from the optimization.
        ref_point: Reference point for hypervolume calculation.
        max_hv: Maximum hypervolume value.

    Returns:
        Tuple containing hypervolume, log hypervolume difference, and objective values.
    """
    pf_bd = DominatedPartitioning(ref_point=ref_point.cpu(), Y=obj_values.cpu())
    hv = pf_bd.compute_hypervolume().item()
    log_hv_difference = np.log10(max_hv - hv)
    return hv, log_hv_difference

def robust_sample_optimal_points(model, bounds, num_samples, num_points):
    """Sample Pareto points with fallback mechanisms."""
    try:
        ps, pf = sample_optimal_points(
            model=model,
            bounds=bounds,
            num_samples=num_samples,
            num_points=num_points,
            optimizer=random_search_optimizer,
            optimizer_kwargs={"pop_size": 500, "max_tries": 10},
        )
        
        # Check if Pareto front is degenerate
        if pf.std(dim=0).min() < 1e-6:
            print("Warning: Degenerate Pareto front detected, using random points")
            # Generate diverse random points as fallback
            ps = torch.rand(num_samples, num_points, bounds.shape[1], **tkwargs)
            pf = model.posterior(ps.view(-1, bounds.shape[1])).mean.view(num_samples, num_points, -1)
        
        return ps, pf
        
    except Exception as e:
        print(f"Pareto sampling failed: {e}, using random fallback")
        # Fallback to random sampling
        ps = torch.rand(num_samples, num_points, bounds.shape[1], **tkwargs)
        pf = model.posterior(ps.view(-1, bounds.shape[1])).mean.view(num_samples, num_points, -1)
        return ps, pf

def mobo_single_iteration(
    train_X: Tensor,
    train_Y: Tensor,
    acq_type: str,
    objective_func,
    bounds: Tensor,
):
    """
    Perform a single iteration of multi-objective Bayesian optimization.

    Args:
        train_X: Training input data.
        train_Y: Training output data.
        acq_type: Type of acquisition function to use ('qNEI' or 'qEI').
        objective_func: The objective function to optimize.
        bounds: Bounds for the optimization.

    Returns:
        New training points and their corresponding objective values.
    """
    # Fit the model
    model = fit_moo_gp(train_X, train_Y, bounds)

    standard_bounds = torch.zeros(2, train_X.shape[-1], **tkwargs)
    standard_bounds[1] = 1

    # Define the acquisition function
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))
    if acq_type == 'qNEHVI':
        acqf = qNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=objective_func.ref_point.tolist(),
            X_baseline=normalize(train_X, bounds),
            prune_baseline=True,
            sampler=sampler,
        )
    elif acq_type == 'qLogNEHVI':
        acqf = qLogNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=objective_func.ref_point.tolist(),
            X_baseline=normalize(train_X, bounds),
            prune_baseline=True,
            sampler=sampler,
        )
    elif acq_type == 'qHVKG':
        acqf = qHypervolumeKnowledgeGradient(
            model=model,
            ref_point=objective_func.ref_point,
            num_fantasies=16,
            num_pareto=10
        )
    elif "ES" in acq_type:
        num_pareto_samples = 16
        num_pareto_points = 16
        ps, pf = robust_sample_optimal_points(
            model=model,
            bounds=standard_bounds,
            num_samples=num_pareto_samples,
            num_points=num_pareto_points,
        )
        hypercell_bounds = compute_sample_box_decomposition(pf)
        if acq_type == 'qLBMOJES':
            acqf = qLowerBoundMultiObjectiveJointEntropySearch(
                model=model,
                pareto_sets=ps,
                pareto_fronts=pf,
                hypercell_bounds=hypercell_bounds,
                estimation_type="LB",
            )
        elif acq_type == 'qLBMOMES':
            acqf = qLowerBoundMultiObjectiveMaxValueEntropySearch(
                model=model,
                hypercell_bounds=hypercell_bounds,
                estimation_type="LB",
            )
        elif acq_type == 'qMOPES':
            acqf = qMultiObjectivePredictiveEntropySearch(model=model, pareto_sets=ps)
    elif acq_type == 'qParEGO':
        normed_train_X = normalize(train_X, bounds)
        with torch.no_grad():
            pred = model.posterior(normed_train_X).mean
        weights = sample_simplex(train_Y.shape[-1], **tkwargs).squeeze()
        objective = GenericMCObjective(
            get_chebyshev_scalarization(weights=weights, Y=pred)
        )
        acqf = qNoisyExpectedImprovement(  # pyre-ignore: [28]
            model=model,
            objective=objective,
            X_baseline=normed_train_X,
            sampler=sampler,
            prune_baseline=True,
        )
    else:
        raise ValueError(f"Unknown acquisition type: {acq_type}")

    # optimize
    if acq_type in ["qNEHVI", "qLogNEHVI", "qHVKG", "qParEGO"]:
        new_x, _ = optimize_acqf(
            acq_function=acqf,
            bounds=standard_bounds,
            q=1,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,  # used for intialization heuristic
            options={"batch_limit": 5, "maxiter": 200},
            sequential=True,
        )
    elif "ES" in acq_type:
        optimize_acqf_kwargs = {
            "acq_function": acqf,
            "bounds": standard_bounds,
            "q": 1,
            "num_restarts": 4,
        }
        if acq_type == "qMOPES":
            new_x, _ = optimize_acqf(
                raw_samples=128,
                options={"with_grad": False},
                **optimize_acqf_kwargs
            )
        else:
            try:
                new_x, _ = optimize_acqf(
                    sequential=True, 
                    raw_samples=512,
                    **optimize_acqf_kwargs
                )
            except:
                print("Sequential optimization failed, falling back to grid search")
                sobol = SobolEngine(dimension=bounds.shape[1], scramble=True, seed=0)
                candidates = sobol.draw(2048).to(**tkwargs)
                new_x, _ = optimize_acqf_discrete(
                    acq_function=acqf,
                    q=1,
                    choices=candidates
                )
    new_x = unnormalize(new_x, bounds)
    # Get new observations
    new_y = objective_func(new_x)

    train_X = torch.cat([train_X, new_x])
    train_Y = torch.cat([train_Y, new_y])

    return train_X, train_Y, model

def mobo_full_loop(
    objective_func, 
    acq_type, 
    X_init, 
    Y_init, 
    bounds, 
    num_iterations
):
    max_hv = objective_func._max_hv
    # Generate initial training data
    train_X  = X_init.clone()
    train_Y  = Y_init.clone()
    hv_list, log_hv_difference_list = [], []
    for iteration_idx in range(num_iterations):
        train_X, train_Y, _ = mobo_single_iteration(train_X, train_Y, acq_type, objective_func, bounds)
        hv, log_hv_difference = get_moo_results(
            train_Y, 
            objective_func.ref_point, 
            max_hv
        )
        hv_list.append(hv)
        log_hv_difference_list.append(log_hv_difference)
        print(f"Iter {iteration_idx} | HV: {hv:.4f} | Log HV Diff: {log_hv_difference:.4f}")
    return (
        np.array(hv_list), 
        np.array(log_hv_difference_list),
        np.array(train_X.detach().cpu().numpy()), 
        np.array(train_Y.detach().cpu().numpy())
    )
        