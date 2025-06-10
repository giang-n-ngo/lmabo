from utils import (
    report_completion, 
    report_ranking, 
    plot_results, 
    plot_results_best_value,
    report_relative_reg,
    print_sorted_dict
)
from bo import acq_type_mapping
from constants import OBJECTIVE_FUNCTIONS
import sys

# Redirect output to file
sys.stdout = open('report.txt', 'w')
problems = []
for item in OBJECTIVE_FUNCTIONS:
    if hasattr(item, "name"):
        problems.append(item.name)
    else:
        problems.append(item.__name__)

completed_problems = report_completion()
print(f"Completed {len(completed_problems)} problems out of {len(problems)}")
print("Total simple regret ranking (lower better):")
report_ranking(completed_problems, "simple_regret")
print("Total cumulative regret ranking (lower better):")
report_ranking(completed_problems, "cum_regret")
all_relative_simple_reg = {acq_type: 0 for acq_type in list(acq_type_mapping.keys())+['lmabo']}
for problem in completed_problems:
    relative_simple_reg = report_relative_reg(problem, "simple_regret", True)
    for acq_type in relative_simple_reg.keys():
        all_relative_simple_reg[acq_type] = all_relative_simple_reg[acq_type] + relative_simple_reg[acq_type]
    print("=="*20)
print("Total relative simple regret (lower better):")
print_sorted_dict(all_relative_simple_reg, reverse=True)


all_relative_cum_reg = {acq_type: 0 for acq_type in list(acq_type_mapping.keys())+['lmabo']}
for problem in completed_problems:
    relative_cum_reg = report_relative_reg(problem, "cum_regret", True)
    for acq_type in relative_simple_reg.keys():
        all_relative_cum_reg[acq_type] = all_relative_cum_reg[acq_type] + relative_cum_reg[acq_type]
    print("=="*20)
print("Total relative cumulative regret (lower better):")
print_sorted_dict(all_relative_cum_reg, reverse=True)

for problem in problems:
    plot_results(problem, "cum_regret")
    plot_results(problem, "simple_regret")
    plot_results_best_value(problem)
# Don't forget to close the file
sys.stdout.close()
# Restore standard output
sys.stdout = sys.__stdout__