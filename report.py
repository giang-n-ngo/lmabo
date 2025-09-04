import numpy as np
import pandas as pd
import os
import sys

from utils import (
    report_completion,
)

from constants import (
    OBJECTIVE_FUNCTIONS_NAMES,
    ACQ_TYPE_MAPPING,
    ALGO_FILE_COUNT,
    LLMGP_NUMERICAL_RESULTS_DIR,
    NUMERICAL_RESULTS_DIR,
    EXP_RUNS,
)

def read_raw_result(problem, acq_type, result_type):
    raw_result = []
    for exp_idx in range(EXP_RUNS):
        try:
            if acq_type == "llmgp":
                file_name = f"{LLMGP_NUMERICAL_RESULTS_DIR}/{problem}/llmgp/{exp_idx}_{result_type}.npy"
            else:
                file_name = f"{NUMERICAL_RESULTS_DIR}/{problem}/{acq_type}/{exp_idx}_{result_type}.npy"
            result_sequence = np.load(file_name)
            if np.isnan(result_sequence).any():
                print(f"Found nan value in {file_name}")
                os.remove(file_name)  # Remove the file if it contains NaN values
                continue
            # if result_type is "simple_regret", check if any number is negative and print file name:
            if result_type == "simple_regret" and any(result < 0 for result in result_sequence):
                print(f"Found negative value in {file_name}")
                return []
            raw_result.append(result_sequence)
        except FileNotFoundError:
            continue 
    return raw_result

def get_agg_result(raw_result, agg):
    if agg == "auc":
        agg_result = np.trapezoid(raw_result.squeeze(), dx=1.0).item()
    elif agg == "mean":
        agg_result = np.mean(raw_result)
    elif agg == "last":
        agg_result = raw_result[-1]
    return agg_result

def get_all_problem_raw_result(problem_list, method_list, result_type):
    problem_result = {}
    for problem in problem_list:
        problem_result[problem] = {}
        for method in method_list:
            problem_result[problem][method] = read_raw_result(problem, method, result_type)
    return problem_result

def find_best_result_per_problem(result_by_all_methods):
    # set best to infty
    best_result = float("inf")
    for _, results in result_by_all_methods.items():
        for result in results:
            if result < best_result:
                best_result = result
    return best_result

def load_results_and_empirical_performance(problem_list, method_list):
    all_raw_results = get_all_problem_raw_result(problem_list, method_list, "train_Y")
    empirical_optimum = {}
    for problem, problem_raw_results in all_raw_results.items():
        minimum_value = float("inf")
        for method_raw_results in problem_raw_results.values():
            method_minimum_value = [min(result_sequence).item() for result_sequence in method_raw_results]
            if len(method_minimum_value)==0:
                continue
            elif minimum_value > min(method_minimum_value):
                minimum_value = min(method_minimum_value)
        empirical_optimum[problem] = minimum_value
    return all_raw_results, empirical_optimum

def cal_simple_regret(all_raw_results, empirical_optimum):
    all_simple_regrets = {}

    for problem, methods in all_raw_results.items():
        all_simple_regrets[problem] = {}
        optimum = empirical_optimum[problem]
        for method, runs in methods.items():
            all_simple_regrets[problem][method] = []
            for i, run in enumerate(runs):
                # run is a numpy array of values for all iterations
                # first filter the actual run
                if run.shape[0] > 100:
                    run = run[-101:]
                else:
                    run = run[-51:]
                # then get the current best value at each iteration
                best_values = np.minimum.accumulate(run)
                # get the simple regret
                simple_regret = best_values - optimum
                all_simple_regrets[problem][method].append(simple_regret)
                assert len(simple_regret) == 101 or len(simple_regret) == 51
                assert np.all(simple_regret >= 0), f"Negative simple regret found at {problem}-{method}{i}"
    return all_simple_regrets

def aggregate_and_to_df(all_simple_regrets, agg):
    # Aggregate by AUC for each run
    agg_simple_regrets = {}
    for problem, methods in all_simple_regrets.items():
        agg_simple_regrets[problem] = {}
        for method, runs in methods.items():
            agg_simple_regrets[problem][method] = []
            for run in runs:
                agg_simple_regrets[problem][method].append(get_agg_result(run, agg))
    # Create a DataFrame from agg_simple_regrets
    agg_simple_regrets_df = pd.DataFrame([
        {"problem": problem, "method": method, **{f"run_{i+1}": run_mean for i, run_mean in enumerate(run_means)}}
        for problem, methods in agg_simple_regrets.items()
        for method, run_means in methods.items()
    ])
    return agg_simple_regrets_df

def list_completed_problems(agg_simple_regrets_df):
    # List all completed problems, which are the ones that has no NaN or inf for all runs by all methods
    completed_problems = []
    for problem in agg_simple_regrets_df["problem"].unique():
        if not agg_simple_regrets_df[agg_simple_regrets_df["problem"] == problem].isnull().values.any():
            completed_problems.append(problem)
    return completed_problems

def get_relative_performance_and_rank(agg_simple_regrets_df, problem_list):
    # get the sum across all runs for problems in problem_list
    rel_performance_df = agg_simple_regrets_df.copy()
    rel_performance_df = rel_performance_df[rel_performance_df["problem"].isin(problem_list)]
    rel_performance_df["sum"] = rel_performance_df.iloc[:, 2:].sum(axis=1)
    # for each problem, divide the sum by the best sum
    for problem in rel_performance_df["problem"].unique():
        best_sum = rel_performance_df.loc[rel_performance_df["problem"] == problem, "sum"].min()
        rel_performance_df.loc[rel_performance_df["problem"] == problem, "relative_performance"] = rel_performance_df["sum"] / best_sum
        # get problem-wise ranking
        rel_performance_df.loc[rel_performance_df["problem"] == problem, "problem_rank"] = rel_performance_df.loc[rel_performance_df["problem"] == problem, "relative_performance"].rank(method="min")
    return rel_performance_df

def summary_by_method(rel_performance_df):
    # Summarize the relative performance by method
    summary_df_1 = rel_performance_df.groupby("method")["relative_performance"].agg(["mean", "min", "max"]).reset_index()
    summary_df_1["range"] = summary_df_1["max"] - summary_df_1["min"]
    # For each problem, compute the average rank of each method
    summary_df_2 = rel_performance_df.groupby("method")["problem_rank"].mean().reset_index()
    # Merge two df
    summary_df = pd.merge(summary_df_1, summary_df_2, on="method", suffixes=("_performance", "_rank"))
    # sort by mean
    summary_df = summary_df.sort_values(by="mean")
    return summary_df

def rank_methods_by_problem(rel_performance_df, problem):
    ranked_df = rel_performance_df[rel_performance_df["problem"] == problem].copy()
    ranked_df["rank"] = ranked_df["relative_performance"].rank(method="min")
    ranked_df = ranked_df.sort_values(by="rank")
    # print full df without new line
    print(ranked_df.to_string(index=False))

if __name__=="__main__":
    sys.stdout = open(f"report_bo.txt", 'w')

    all_methods = list(ACQ_TYPE_MAPPING.keys())
    all_methods.extend(list(ALGO_FILE_COUNT.keys()))
    all_problems = OBJECTIVE_FUNCTIONS_NAMES
    # exclude some methods during the main report
    excluded_methods = [
        # "no_past_bo",
        # "setup_bo",
        # "esp",
        "lmabo-ops", # ablation method
        "bo_alternating_k1", # ablation method
        "bo_alternating_k3", # ablation method
        "bo_alternating_k5", # ablation method
        "bo_explore_exploit", # ablation method
        "lmabo-ab1", # ablation method
        "lmabo-ab2", # ablation method
        "lmabo-ab3", # ablation method
    ]
    all_methods = [method for method in all_methods if method not in excluded_methods]

    # Redirect output to file
    problems = []
    for item in OBJECTIVE_FUNCTIONS_NAMES:
        problems.append(item)

    completed_problems = report_completion(problems, all_methods, excluded_methods)
    print(f"Completed {len(completed_problems)} problems out of {len(problems)}")

    all_raw_results, empirical_optimum = load_results_and_empirical_performance(
        all_problems, 
        all_methods
    )
    all_simple_regrets = cal_simple_regret(all_raw_results, empirical_optimum)
    agg_simple_regrets_df = aggregate_and_to_df(all_simple_regrets, "auc")
    rel_performance_df = get_relative_performance_and_rank(agg_simple_regrets_df, completed_problems)
    summary_df = summary_by_method(rel_performance_df)
    print(summary_df.to_string(index=False))
    print("="*200)
    for problem in completed_problems:
        rank_methods_by_problem(rel_performance_df, problem)
        print("="*200)

    # Don't forget to close the file
    sys.stdout.close()
    # Restore standard output
    sys.stdout = sys.__stdout__