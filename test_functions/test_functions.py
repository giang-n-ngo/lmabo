import cocoex
import math
import torch
from botorch.test_functions import SyntheticTestFunction
from torch import Tensor
from typing import Optional

class Alpine1(SyntheticTestFunction):
    """Alpine 1 test function.
    
    f(x) = sum |x_i * sin(x_i) + 0.1 * x_i|
    
    The function is typically evaluated on [-10, 10]^d.
    The global minimum is at x* = (0, ..., 0) with f(x*) = 0.
    """
    _optimal_value = 0.0
    
    def __init__(self, dim=2, noise_std=None, negate=False, dtype=torch.double):
        self.dim = dim
        bounds = [(-10.0, 10.0) for _ in range(self.dim)]
        self._optimizers = [tuple(0.0 for _ in range(self.dim))]
        super().__init__(noise_std=noise_std, negate=negate, bounds=bounds, dtype=dtype)
    
    def evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        return torch.sum(torch.abs(X * torch.sin(X) + 0.1 * X), dim=-1)

class Alpine2(SyntheticTestFunction):
    """Alpine 2 test function.
    
    f(x) = prod sqrt(x_i) * sin(x_i)
    
    The function is typically evaluated on [0, 10]^d.
    The global minimum is at x* = (7.917, ..., 7.917) with f(x*) = 2.808^d.
    """
    def __init__(self, dim=2, noise_std=None, negate=False, dtype=torch.double):
        self.dim = dim
        bounds = [(0.0, 10.0) for _ in range(self.dim)]
        self._optimizers = [tuple(7.917 for _ in range(self.dim))]
        super().__init__(noise_std=noise_std, negate=negate, bounds=bounds, dtype=dtype)
    
    def evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        return torch.prod(torch.sqrt(X) * torch.sin(X), dim=-1)
    
class Easom(SyntheticTestFunction):
    r"""Easom Test Function.

    Two-dimensional test function with a single, sharp global minimum. It is
    characterized by a largely flat surface with a narrow valley.
    The function is typically evaluated on the square x_i \in [-10, 10] or
    x_i \in [-100, 100].

    f(x, y) = -cos(x) * cos(y) * exp(-((x - \pi)^2 + (y - \pi)^2))

    The global minimum is f(\pi, \pi) = -1.
    """

    dim = 2
    continuous_inds = list(range(dim))
    _bounds = [(-10.0, 10.0), (-10.0, 10.0)]  # Typical bounds
    _optimal_value = -1.0
    # Use math.pi for class-level definitions of optimizers, as it's a Python float.
    # BoTorch's base class will convert this to a tensor.
    _optimizers = [(math.pi, math.pi)]

    def __init__(
        self,
        noise_std: Optional[float] = None,
        negate: bool = False,
    ) -> None:
        super().__init__(noise_std=noise_std, negate=negate)
        # Ensure optimizers is a tensor, which super().__init__ should handle
        # from _optimizers. If not using the full BoTorch hierarchy, ensure it here.
        # For BoTorch, this is typically handled by the base SyntheticTestFunction.
        # self.optimizers = torch.tensor(self._optimizers, dtype=torch.float64)

    def _evaluate_true(self, X: Tensor) -> Tensor:
        r"""Evaluate the Easom function on a batch of points.

        Args:
            X: A `(batch_shape) x 2` tensor of points.

        Returns:
            A `(batch_shape)` tensor of function values.
        """
        if X.ndim < 1: # Should not happen if forward method pre-processes
            raise ValueError("Input X must have at least one dimension.")
        
        # Ensure X has the correct dimension
        if X.shape[-1] != self.dim:
            raise ValueError(
                f"Input X must be of dimension {self.dim} in the last axis, "
                f"but got {X.shape[-1]}."
            )

        x1 = X[..., 0]
        x2 = X[..., 1]

        # Use torch.pi for calculations involving tensors
        pi_val = torch.pi

        term1 = -torch.cos(x1) * torch.cos(x2)
        term2 = torch.exp(-((x1 - pi_val) ** 2 + (x2 - pi_val) ** 2))
        
        # The result should have dimensions corresponding to batch_shape
        return term1 * term2
    
# optimal values for COCO functions estimated with 1,000,000 random samples    
COCO_OPTIMAL_VALUE = {
    "Sphere": 79.5188, 
    "Ellipsoid": 9099.1274, 
    "BucheRastrigin": -445.4024,
    "LinearSlope": -9.21, 
    "AttractiveSector": 38.2819, 
    "StepEllipsoid": 93.6699,
    "RosenbrockRotated": 804.1678, 
    "Ellipsoid2": 12993.3869,
    "Discus": 81.0588, 
    "BentCigar": 44836.2297, 
    "SharpRidge": 86.2341, 
    "DifferentPowers": -52.1712,
    "Weierstrass": 71.6925, 
    "Schaffers": -16.1823,
    "SchaffersIllCond": -15.0724, 
    "CompositeGriewankRosenbrock": -98.6764,
    "Schwefel": -545.0874, 
    "Gallagher21": 40.7951, 
    "Gallagher101": -999.9830, 
    "Katsuura": 7.3876,
    "LunacekBiRastrigin": 112.8333
}

def create_coco_class(function_id, dimension, problem_name):
    """Create a COCO function class with specific ID and dimension."""
    class COCOProblem(SyntheticTestFunction):
        dim = dimension
        name = problem_name
        continuous_inds = list(range(dim))
        _optimal_value = COCO_OPTIMAL_VALUE.get(problem_name, None)
        
        def __init__(self, noise_std=None, negate=False):
            self.suite = cocoex.Suite("bbob", "", f"function_indices:{function_id} dimensions:{dimension}")
            self.problem = self.suite[0]
            self._bounds = [
                (
                    self.problem.lower_bounds[i],
                    self.problem.upper_bounds[i]
                )
                for i in range(dimension)
            ]
            super().__init__(noise_std=noise_std, negate=negate)
        
        def _evaluate_true(self, X):
            if X.ndim == 1:
                X = X.unsqueeze(0)
            result = torch.zeros(X.shape[0], device=X.device, dtype=X.dtype)
            for i in range(X.shape[0]):
                result[i] = self.problem(X[i].cpu().numpy())
            return result
        
        def __del__(self):
            if hasattr(self, 'suite'):
                self.suite.free()
                
    return COCOProblem