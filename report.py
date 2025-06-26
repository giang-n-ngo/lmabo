import argparse
import numpy as np
import sys

from utils import (
    report_completion,
    report_ranking,
    plot_results,
    report_relative_reg,
    print_sorted_dict
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
    from utils import (
        plot_results_best_value,
    )
    from bo import acq_type_mapping
    from constants import OBJECTIVE_FUNCTIONS

    active_acq_type_list = list(acq_type_mapping.keys())
    active_acq_type_list += [
        "lmabo",
        "gphedge",
    ]
    excepted_acq_type_list = [
        "llmgp"
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
    print("Total simple regret ranking (lower better):")
    report_ranking(completed_problems, "simple_regret", True, active_acq_type_list)
    all_relative_simple_reg = {acq_type: [] for acq_type in active_acq_type_list}
    for problem in completed_problems:
        relative_simple_reg = report_relative_reg(problem, "simple_regret", True, True, active_acq_type_list)
        for acq_type in relative_simple_reg.keys():
            all_relative_simple_reg[acq_type].append(relative_simple_reg[acq_type])
        print("=="*20)
    print("Summary relative simple regret (lower better):")
    mean_relative_simple_reg = {acq_type: np.mean(val) for acq_type, val in all_relative_simple_reg.items()}
    std_relative_simple_reg = {acq_type: np.std(val) for acq_type, val in all_relative_simple_reg.items()}
    print("Mean")
    print_sorted_dict(mean_relative_simple_reg, reverse=True)
    print("Std")
    print_sorted_dict(std_relative_simple_reg, reverse=True)

    for problem in problems:
        plot_results(problem, "simple_regret", full_acq_type_list)
        plot_results_best_value(problem, full_acq_type_list)
elif args.setting == "moo":
    from moo import MOO_OBJECTIVE_FUNCTIONS, moo_acq_type_mapping
    moo_acq_type_list = list(moo_acq_type_mapping.keys())
    problems = []
    for item in MOO_OBJECTIVE_FUNCTIONS:
        if hasattr(item, "name"):
            problems.append(item.name)
        else:
            problems.append(item.__name__)
    completed_problems = report_completion(problems, moo_acq_type_list)
    print(f"Completed {len(completed_problems)} problems out of {len(problems)}")
    print("Total HV ranking (higher better):")
    report_ranking(problems, "hv", reverse=False, acq_type_list=moo_acq_type_list)
    all_relative_hv = {acq_type: 0 for acq_type in moo_acq_type_list+['lmabo']}
    for problem in problems:
        relative_hv = report_relative_reg(problem, "hv", True, False, moo_acq_type_list)
        for acq_type in relative_hv.keys():
            all_relative_hv[acq_type] = all_relative_hv[acq_type] + relative_hv[acq_type]
        print("=="*20)
    print("Total relative HV (higher better):")
    print_sorted_dict(all_relative_hv, reverse=False)
    print("Total log HV difference ranking (lower better):")
    report_ranking(problems, "log_hv_diff", reverse=True, acq_type_list=moo_acq_type_list)
    all_relative_log_hv_diff = {acq_type: 0 for acq_type in moo_acq_type_list+['lmabo']}
    for problem in problems:
        relative_log_hv_diff = report_relative_reg(problem, "log_hv_diff", True, True, moo_acq_type_list)
        for acq_type in relative_log_hv_diff.keys():
            all_relative_log_hv_diff[acq_type] = all_relative_log_hv_diff[acq_type] + relative_log_hv_diff[acq_type]
        print("=="*20)
    print("Total relative log HV difference (lower better):")
    print_sorted_dict(all_relative_log_hv_diff, reverse=False)

elif args.setting == "constrained":
    pass
# Don't forget to close the file
sys.stdout.close()
# Restore standard output
sys.stdout = sys.__stdout__