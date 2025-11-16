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

ACQ_TYPE_MAPPING = {
    "PI": "PI",
    "LogPI": "LogPI",
    "EI": "EI",
    "LogEI": "LogEI",
    "UCB": "UCB",
    "PosMean": "PosMean",
    "PosSTD": "PosSTD",
    "TS": "TS",
    "qKG": "KG",
    "qPES": "PES",
    "qMES": "MES",
    "qJES": "JES"
}

ALGO_FILE_COUNT = {
    "lmabo": 6,
    "lmabo-gpt": 6,
    "lmabo-context": 6,
    "lmabo-ops": 6,
    # "lmabo-ops2": 6,
    "lmabo-ops3": 6,
    # "lmabo-ops4": 6,
    # "lmabo-ops5": 6,
    "lmabo-ops6": 6,
    "lmabo-ab1": 6,
    "lmabo-ab2": 6,
    "lmabo-ab3": 6,
    "lmabo-ab4": 6,
    # "lmabo-ab5": 6,
    "gphedge": 6,
    "gphedge-curated": 6,
    "no_past_bo": 6,
    "no_past_bo-curated": 6,
    "setup_bo": 6,
    "setup_bo-curated": 6,
    "esp": 5,
    "esp-curated": 5,
    "random_acq": 5,
    "random_acq_curated1": 5,
    "random_acq_curated2": 5,
    "llmgp": 3,
    "llambo": 3,
    "bo_alternating_k1": 4,
    "bo_alternating_k3": 4,
    "bo_alternating_k5": 4,
    "bo_explore_exploit": 4
}

OPS_MODEL_MAPPING = {
    "lmabo-ops": "Qwen/Qwen-3-8B",
    "lmabo-ops2": "Qwen/Qwen3-14B",
    "lmabo-ops3": "Qwen/Qwen3-30B-A3B-Thinking-2507",
    "lmabo-ops4": "Qwen/QwQ-32B",
    "lmabo-ops5": "microsoft/Phi-4-reasoning",
    "lmabo-ops6": "openai/gpt-oss-120b",
}