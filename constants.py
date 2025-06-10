from botorch.test_functions import *
from test_functions.test_functions import *
from test_functions.lunar_lander import LunarLander
FIG_DIR = "figures" 
NUMERICAL_RESULTS_DIR= "numerical_results" 
EXP_RUNS  = 10  # Number of runs per acquisition function

# Add COCO test problems
Sphere = create_coco_class(function_id=1, dimension=5, problem_name="Sphere")  # F1: Sphere
Ellipsoid = create_coco_class(function_id=2, dimension=10, problem_name="Ellipsoid")  # F2: Separable Ellipsoid
BucheRastrigin = create_coco_class(function_id=4, dimension=5, problem_name="BucheRastrigin")  # F4: Büche-Rastrigin
LinearSlope = create_coco_class(function_id=5, dimension=5, problem_name="LinearSlope")  # F5: Linear Slope
AttractiveSector = create_coco_class(function_id=6, dimension=5, problem_name="AttractiveSector")  # F6: Attractive Sector
StepEllipsoid = create_coco_class(function_id=7, dimension=5, problem_name="StepEllipsoid")  # F7: Step Ellipsoid
RosenbrockRotated = create_coco_class(function_id=9, dimension=10, problem_name="RosenbrockRotated")  # F9: Rotated Rosenbrock
Ellipsoid2 = create_coco_class(function_id=10, dimension=10, problem_name="Ellipsoid2")  # F10: Ellipsoid
Discus = create_coco_class(function_id=11, dimension=5, problem_name="Discus")  # F11: Discus
BentCigar = create_coco_class(function_id=12, dimension=5, problem_name="BentCigar")  # F12: Bent Cigar
SharpRidge = create_coco_class(function_id=13, dimension=5, problem_name="SharpRidge")  # F13: Sharp Ridge
DifferentPowers = create_coco_class(function_id=14, dimension=5, problem_name="DifferentPowers")  # F14: Different Powers
Weierstrass = create_coco_class(function_id=16, dimension=5, problem_name="Weierstrass")  # F16: Weierstrass
Schaffers = create_coco_class(function_id=17, dimension=5, problem_name="Schaffers")  # F17: Schaffers F7
SchaffersIllCond = create_coco_class(function_id=18, dimension=5, problem_name="SchaffersIllCond")  # F18: Schaffers F7 Ill-conditioned
CompositeGriewankRosenbrock = create_coco_class(function_id=19, dimension=10, problem_name="CompositeGriewankRosenbrock")  # F19: Composite G-R
Schwefel = create_coco_class(function_id=20, dimension=5, problem_name="Schwefel")  # F20: Schwefel
Gallagher21 = create_coco_class(function_id=21, dimension=5, problem_name="Gallagher21")  # F21: Gallagher 101 Peaks
Gallagher101 = create_coco_class(function_id=22, dimension=5, problem_name="Gallagher101")  # F22: Gallagher 21 Peaks
Katsuura = create_coco_class(function_id=23, dimension=5, problem_name="Katsuura")  # F23: Katsuura
LunacekBiRastrigin = create_coco_class(function_id=24, dimension=5, problem_name="LunacekBiRastrigin")  # F24: Lunacek bi-Rastrigin

OBJECTIVE_FUNCTIONS = [
    Ackley,
    Beale,
    Branin,
    Bukin,
    Cosine8,
    DixonPrice,
    DropWave,
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
    StyblinskiTang,
    Sphere, 
    Ellipsoid, 
    BucheRastrigin,
    LinearSlope, 
    AttractiveSector, 
    StepEllipsoid,
    RosenbrockRotated, 
    Ellipsoid2,
    Discus, 
    BentCigar, 
    SharpRidge, 
    DifferentPowers,
    Weierstrass, 
    Schaffers,
    SchaffersIllCond, 
    CompositeGriewankRosenbrock,
    Schwefel, 
    Gallagher21, 
    Gallagher101, 
    Katsuura,
    LunacekBiRastrigin
]

DIMS = {
    Ackley: 50,
    DixonPrice: 15,
    Griewank: 9,
    Hartmann: 6,
    Levy: 13,
    Michalewicz: 10,
    Powell: 18,
    Rastrigin: 23,
    Rosenbrock: 24,
    StyblinskiTang: 21,
}
