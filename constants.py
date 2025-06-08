from botorch.test_functions import *
from test_functions.test_functions import *
from test_functions.lunar_lander import LunarLander
FIG_DIR = "figures" 
NUMERICAL_RESULTS_DIR= "numerical_results" 
EXP_RUNS  = 10  # Number of runs per acquisition function
OBJECTIVE_FUNCTIONS = [
    Ackley,
    Beale,
    Branin,
    Bukin,
    Cosine8,
    DixonPrice,
    DropWave,
    Easom,
    EggHolder,
    Griewank,
    Hartmann,
    HolderTable,
    Levy,
    Michalewicz,
    Powell,
    Rastrigin,
    Rosenbrock,
    Shekel,
    SixHumpCamel,
    StyblinskiTang
]
DIMS = {
    Ackley: 50,
    DixonPrice: 15,
    Easom: 2,
    Griewank: 9,
    Hartmann: 6,
    Levy: 13,
    Michalewicz: 10,
    Powell: 18,
    Rastrigin: 23,
    Rosenbrock: 24,
    StyblinskiTang: 21,
    Shekel: 10
}
