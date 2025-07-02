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
        "llambo"
    ]
    excepted_acq_type_list = [
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
    pass
elif args.setting == "constrained":
    pass
# Don't forget to close the file
sys.stdout.close()
# Restore standard output
sys.stdout = sys.__stdout__