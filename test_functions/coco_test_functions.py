import cocoex
import math
import torch
from botorch.test_functions import SyntheticTestFunction
from torch import Tensor
from typing import Optional
    
# optimal values for COCO functions estimated with 1,000,000 random samples    
COCO_OPTIMAL_VALUE = {
    "BucheRastrigin": -445.4024,
    "LinearSlope": -9.21, 
    "AttractiveSector": 38.2819, 
    "StepEllipsoid": 93.6699,
    "Discus": 81.0588, 
    "BentCigar": 44836.2297, 
    "SharpRidge": 86.2341, 
    "DifferentPowers": -52.1712,
    "Weierstrass": 71.6925, 
    "SchaffersIllCond": -15.0724, 
    "CompositeGriewankRosenbrock": -98.6764,
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
        
        def evaluate_true(self, X):
            return self._evaluate_true(X)
        
        def __del__(self):
            if hasattr(self, 'suite'):
                self.suite.free()
                
    return COCOProblem