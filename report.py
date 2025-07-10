import argparse
import numpy as np
import sys

from utils import (
    report_completion,
    report_ranking_summary,
    plot_results,
    report_relative_reg,
    print_sorted_dict,
    get_problem_result
)

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--setting",
    type=str,
    default="bo",
    help="bo, moo, or constrained",
)
args = argparser.parse_args()

sys.stdout = open(f"report_{args.setting}.txt", 'w')
if args.setting == "bo":
    from bo import acq_type_mapping
    from constants import OBJECTIVE_FUNCTIONS

    active_acq_type_list = list(acq_type_mapping.keys())
    active_acq_type_list += [
        "lmabo",
        "gphedge",
        "llmgp",
        "llambo",
    ]
    excepted_acq_type_list = [
        "lmabo-ops"
    ]
    full_acq_type_list = active_acq_type_list + excepted_acq_type_list

    # Redirect output to file
    problems = []
    for item in OBJECTIVE_FUNCTIONS:
        if hasattr(item, "name"):
            problems.append(item.name)
        else:
            problems.append(item.__name__)

    completed_problems = report_completion(problems, active_acq_type_list, excepted_acq_type_list)
    print(f"Completed {len(completed_problems)} problems out of {len(problems)}")
    problem_result = get_problem_result(completed_problems, active_acq_type_list, "simple_regret")
    print("Total simple regret ranking (lower better):")
    report_ranking_summary(problem_result, "simple_regret", active_acq_type_list)
    all_relative_simple_reg = {acq_type: [] for acq_type in active_acq_type_list}
    for problem in completed_problems:
        relative_simple_reg = report_relative_reg(problem, "simple_regret", problem_result[problem], True, True)
        for acq_type in relative_simple_reg.keys():
            all_relative_simple_reg[acq_type].append(relative_simple_reg[acq_type].item())
        print("=="*35)
    print("Summary relative simple regret (lower better):")
    mean_relative_simple_reg = {}
    std_relative_simple_reg = {}
    for acq_type, val in all_relative_simple_reg.items():
        mean_relative_simple_reg[acq_type] = np.mean(val)
        std_relative_simple_reg[acq_type] = np.std(val)
    print("Mean")
    print_sorted_dict(mean_relative_simple_reg, reverse=False)
    print("Std")
    print_sorted_dict(std_relative_simple_reg, reverse=False)

    for problem in completed_problems:
        plot_results(problem, "simple_regret", full_acq_type_list)
        plot_results(problem, "best_val", full_acq_type_list)
elif args.setting == "moo":
    from moo import moo_acq_type_mapping, MOO_OBJECTIVE_FUNCTIONS
    
    active_acq_type_list = list(moo_acq_type_mapping.keys())
    active_acq_type_list += [
    ]
    excepted_acq_type_list = [
        # Add any methods to exclude from reporting
        "lmamoo",
    ]
    full_acq_type_list = active_acq_type_list + excepted_acq_type_list

    # Get MOO problems
    problems = []
    for item in MOO_OBJECTIVE_FUNCTIONS:
        if hasattr(item, "name"):
            problems.append(item.name)
        else:
            problems.append(item.__name__)

    completed_problems = report_completion(problems, active_acq_type_list, excepted_acq_type_list)
    print(f"Completed {len(completed_problems)} MOO problems out of {len(problems)}")
    
    # Report hypervolume (primary MOO metric)
    problem_result = get_problem_result(completed_problems, active_acq_type_list, "hv")
    print("Total hypervolume ranking (lower better):")
    report_ranking_summary(problem_result, "hv", active_acq_type_list)
    
    all_relative_hv = {acq_type: [] for acq_type in active_acq_type_list}
    for problem in completed_problems:
        relative_hv = report_relative_reg(problem, "hv", problem_result[problem], True, False)
        for acq_type in relative_hv.keys():
            all_relative_hv[acq_type].append(relative_hv[acq_type].item())
        print("=="*35)
    
    print("Summary relative hypervolume (lower better):")
    mean_relative_hv = {}
    std_relative_hv = {}
    for acq_type, val in all_relative_hv.items():
        mean_relative_hv[acq_type] = np.mean(val)
        std_relative_hv[acq_type] = np.std(val)
    print("Mean")
    print_sorted_dict(mean_relative_hv, reverse=True)
    print("Std")
    print_sorted_dict(std_relative_hv, reverse=True)

    # Generate plots for MOO problems
    for problem in completed_problems:
        plot_results(problem, "log_hv_diff", full_acq_type_list)
        plot_results(problem, "hv", full_acq_type_list)
elif args.setting == "constrained":
    from bo import acq_type_mapping, CONSTRAINED_OBJECTIVE_FUNCTIONS
    
    active_acq_type_list = list(acq_type_mapping.keys())
    active_acq_type_list += [
    ]
    excepted_acq_type_list = [
        # Add any methods to exclude from reporting
        "lmabo"
    ]
    full_acq_type_list = active_acq_type_list + excepted_acq_type_list

    # Get constrained problems
    problems = []
    for item in CONSTRAINED_OBJECTIVE_FUNCTIONS:
        if hasattr(item, "name"):
            problems.append(item.name)
        else:
            problems.append(item.__name__)

    completed_problems = report_completion(
        problems, 
        active_acq_type_list, 
        excepted_acq_type_list,
        constrained=True
    )
    print(f"Completed {len(completed_problems)} constrained problems out of {len(problems)}")

# Don't forget to close the file
sys.stdout.close()
# Restore standard output
sys.stdout = sys.__stdout__