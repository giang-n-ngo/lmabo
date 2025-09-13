from botorch.test_functions import *
from test_functions.coco_test_functions import Easom, create_coco_class
from test_functions.hpt_test_functions import create_bayesmark_class

COCO_DICT = {
    "Sphere": {"function_id": 1, "dimension": 5},
    "Ellipsoid": {"function_id": 2, "dimension": 5},
    "BucheRastrigin": {"function_id": 4, "dimension": 5},
    "LinearSlope": {"function_id": 5, "dimension": 5},
    "AttractiveSector": {"function_id": 6, "dimension": 5},
    "StepEllipsoid": {"function_id": 7, "dimension": 5},
    "RosenbrockRotated": {"function_id": 9, "dimension": 10},
    "Ellipsoid2": {"function_id": 10, "dimension": 10},
    "Discus": {"function_id": 11, "dimension": 5},
    "BentCigar": {"function_id": 12, "dimension": 5},
    "SharpRidge": {"function_id": 13, "dimension": 5},
    "DifferentPowers": {"function_id": 14, "dimension": 5},
    "Weierstrass": {"function_id": 16, "dimension": 5},
    "Schaffers": {"function_id": 17, "dimension": 5},
    "SchaffersIllCond": {"function_id": 18, "dimension": 5},
    "CompositeGriewankRosenbrock": {"function_id": 19, "dimension": 10},
    "Schwefel": {"function_id": 20, "dimension": 5},
    "Gallagher21": {"function_id": 21, "dimension": 5},
    "Gallagher101": {"function_id": 22, "dimension": 5},
    "Katsuura": {"function_id": 23, "dimension": 5},
    "LunacekBiRastrigin": {"function_id": 24, "dimension": 5},
}

DIMS = {
    "Ackley": 50,
    "DixonPrice": 15,
    "Griewank": 9,
    "Hartmann": 6,
    "Levy": 13,
    "Michalewicz": 10,
    "Powell": 18,
    "Rosenbrock": 24,
    "Rastrigin": 23,
    "StyblinskiTang": 21,
}

def load_objective_func(problem, problem_type="botorch"):
    # load f
    if problem_type == "coco":
        f = create_coco_class(
            function_id=COCO_DICT[problem]["function_id"], 
            dimension=COCO_DICT[problem]["dimension"],
            problem_name=problem
        )
    elif problem_type == "botorch":
        f = eval(problem)
    elif problem_type == "hpt":
        _, dataset, model = problem.split("_")
        f = create_bayesmark_class(model, dataset)
    # get dimensions
    dim = 0
    if problem in DIMS:  
        dim = DIMS[problem]
        objective_func = f(dim=dim)
    else:
        dim = f.dim
        objective_func = f()
    bounds = objective_func.bounds
    assert len(bounds) == 2
    assert len(bounds[0]) == dim
    if problem == "Cosine8":
        # Fix incorrect optimal value for Cosine8
        objective_func._optimal_value = -8.8
    if problem_type == "botorch":
        objective_func.name = problem
    return objective_func, dim, bounds
    
