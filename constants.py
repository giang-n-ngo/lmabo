FIG_DIR = "figures" 
NUMERICAL_RESULTS_DIR = "numerical_results" 
EXP_RUNS  = 10  # Number of runs per acquisition function
LLMGP_NUMERICAL_RESULTS_DIR = "llm_processes/output/black_box"

BOTORCH_FUNCTIONS_NAMES = [
    "Ackley",
    "Beale",
    "Branin",
    "Bukin",
    "Cosine8",
    "DixonPrice",
    "DropWave",
    "EggHolder",
    "Griewank",
    "Hartmann",
    "HolderTable",
    "Levy",
    "Michalewicz",
    "Powell",
    "Rastrigin",
    "Rosenbrock",
    "Shekel",
    "SixHumpCamel",
    "StyblinskiTang",
    "Easom"
]

COCO_FUNCTIONS_NAMES = [
    "Sphere",
    "BucheRastrigin",
    "LinearSlope",
    "AttractiveSector",
    "StepEllipsoid",
    "RosenbrockRotated",
    "Ellipsoid2",
    "Discus",
    "BentCigar",
    "SharpRidge",
    "DifferentPowers",
    "Weierstrass",
    "Schaffers",
    "SchaffersIllCond",
    "CompositeGriewankRosenbrock",
    "Schwefel",
    "Gallagher21",
    "Gallagher101",
    "Katsuura",
    "LunacekBiRastrigin"
]

HPT_FUNCTIONS_NAMES = [
    "hpt_breast_RandomForest",
    "hpt_breast_DecisionTree",
    "hpt_breast_SVM",
    "hpt_breast_AdaBoost",
    "hpt_breast_MLPSGD",
    "hpt_digits_RandomForest",
    "hpt_digits_DecisionTree",
    "hpt_digits_SVM",
    "hpt_digits_AdaBoost",
    "hpt_digits_MLPSGD",
    "hpt_wine_RandomForest",
    "hpt_wine_DecisionTree",
    "hpt_wine_SVM",
    "hpt_wine_AdaBoost",
    "hpt_wine_MLPSGD",
    "hpt_diabetes_RandomForest",
    "hpt_diabetes_DecisionTree",
    "hpt_diabetes_SVM",
    "hpt_diabetes_AdaBoost",
    "hpt_diabetes_MLPSGD",
]

OBJECTIVE_FUNCTIONS_NAMES = \
    BOTORCH_FUNCTIONS_NAMES \
     + COCO_FUNCTIONS_NAMES \
     + HPT_FUNCTIONS_NAMES
