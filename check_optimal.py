"""
This script double checks the optimal value of the objective functions (100 repetitions, 10,000 points each).
"""

import torch
from bo import prepare_objective_func
from torch.quasirandom import SobolEngine
import random
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double  # Use double precision for GP models
num_initial_points = 10000
def check_problem(problem):
    # prepare function
    objective_func, dim, bounds = prepare_objective_func(problem)
    bounds = torch.tensor(objective_func.bounds, dtype=dtype, device=device)  # Search space bounds
    current_min = float("inf")
    sample_bounds = torch.tensor([[float("-inf")] * dim, [float("inf")] * dim], dtype=dtype, device=device)
    for i in range(100):
        random_seed = random.randint(0, 1000000)
        sobol = SobolEngine(dimension=dim, scramble=True, seed=random_seed)
        fixed_train_X  = bounds[0] + (bounds[1] - bounds[0]) * sobol.draw(num_initial_points).to(dtype=dtype, device=device)
        fixed_train_Y  = objective_func(fixed_train_X).unsqueeze(-1) # Evaluate function and reshape
        current_min = min(current_min, fixed_train_Y.min().detach().cpu().item())
        # Update bounds based on the current minimum
        sample_bounds[0] = torch.max(sample_bounds[0], fixed_train_X[fixed_train_Y.argmin()])
        sample_bounds[1] = torch.min(sample_bounds[1], fixed_train_X[fixed_train_Y.argmin()])
    print("Sample bounds:", sample_bounds)
    print(problem, current_min)

if __name__ == "__main__":
    check_problem("Cosine8")
    check_problem("LinearSlope")