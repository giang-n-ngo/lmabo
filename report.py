import numpy as np
import pandas as pd
import json
import os
import sys
from scipy import stats
from statsmodels.stats.multitest import multipletests

from utils import (
    report_completion,
)

from constants import (
    ACQ_TYPE_MAPPING,
    ALGO_FILE_COUNT,
    LLMGP_NUMERICAL_RESULTS_DIR,
    NUMERICAL_RESULTS_DIR,
    EXP_RUNS,
)

final_problems = [
    # botorch
    "Ackley", # must have
    "Beale",
    "Bukin",
    "Cosine8", # must have
    "DixonPrice",
    "DropWave",
    "EggHolder",
    "Griewank",
    "Hartmann",
    "HolderTable",
    "Levy",
    "Michalewicz",
    "StyblinskiTang",
    "Shekel",
    "SixHumpCamel",
    # coco
    "BucheRastrigin",
    "LinearSlope",
    "AttractiveSector",
    "StepEllipsoid",
    "Discus",
    "BentCigar",
    "SharpRidge",
    "DifferentPowers",
    "Weierstrass",
    "SchaffersIllCond",
    "CompositeGriewankRosenbrock",
    "Gallagher21",
    "Gallagher101", # must have
    "Katsuura",
    "LunacekBiRastrigin",
    # hpSV
    "hpt_breast_RandomForest",
    "hpt_breast_DecisionTree",
    "hpt_breast_SVM",
    "hpt_breast_AdaBoost",
    "hpt_breast_MLPSGD",
    "hpt_digits_RandomForest", # must have
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

methods_order_curated = [
    "gphedge",
    "gphedge-curated",
    "no_past_bo",
    "no_past_bo-curated",
    "setup_bo",
    "setup_bo-curated",
    "esp",
    "esp-curated",
]

methods_order = [
    "PI", 
    "LogPI",
    "EI",
    "LogEI",
    "PosMean",
    "PosSTD",
    "UCB",
    "TS",
    "qKG",
    "qPES",
    "qMES",
    "qJES",
    "random_acq",
    "random_acq_curated2",
    "random_acq_curated1",
    "bo_alternating_k1",
    "bo_alternating_k3",
    "bo_alternating_k5",
    "bo_explore_exploit",
    "llambo",
    "llmgp",
    "gphedge",
    "no_past_bo",
    "setup_bo",
    "esp",
    "lmabo",
]

methods_order_ablation = [
    "lmabo-ab1",
    "lmabo-ab2",
    "lmabo-ab3",
    "lmabo-ab4",
    "lmabo-ops",
    "lmabo-ops3",
    "lmabo-ops6",
    "lmabo-gpt",
    "lmabo",
]

method_name_mapping = {
    "PosSTD": "PosSTD",
    "PosMean": "PosMean",
    "PI": "PI", 
    "LogPI": "LogPI",
    "EI": "EI",
    "LogEI": "LogEI",
    "UCB": "UCB",
    "TS": "TS",
    "qKG": "KG",
    "qPES": "PES",
    "qMES": "MES",
    "qJES": "JES",
    "llambo": "LLAMBO",
    "llmgp": "LLMP",
    "random_acq": "Random",
    "random_acq_curated1": "Random (EI, LogEI, TS)",
    "random_acq_curated2": "Random (EI, TS, UCB, PosMean)",
    "bo_alternating_k1": "Alt-EI-TS-1",
    "bo_alternating_k3": "Alt-EI-TS-3",
    "bo_alternating_k5": "Alt-EI-TS-5",
    "bo_explore_exploit": "TwoPhase-TS-EI",
    "gphedge": "GP-Hedge",
    "gphedge-curated": "GP-Hedge-Curated",
    "no_past_bo": "No-PASt-BO",
    "no_past_bo-curated": "No-PASt-BO-Curated",
    "setup_bo": "SETUP-BO",
    "setup_bo-curated": "SETUP-BO-Curated",
    "esp": "ESP",
    "esp-curated": "ESP-Curated",
    "lmabo": "LMABO",
    "lmabo-context": "LMABO-Context",
    "lmabo-gpt": "LMABO (GPT-4o mini)",
    "lmabo-ab1": "LMABO-AB1",
    "lmabo-ab2": "LMABO-AB2",
    "lmabo-ab3": "LMABO-AB3",
    "lmabo-ab4": "LMABO-AB4",
    "lmabo-ops": "LMABO-8B",
    "lmabo-ops3": "LMABO-30B",
    "lmabo-ops6": "LMABO-120B",
}

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

def get_mean_iqr_summary(rel_performance_df):
    # rel_performance_df: columns: method, problem, relative_performance, problem_rank

    gp = rel_performance_df.groupby("method")
    mean = gp["relative_performance"].mean()
    q1 = gp["relative_performance"].quantile(0.25)
    q3 = gp["relative_performance"].quantile(0.75)
    mean_rank = gp["problem_rank"].mean()
    min_rank = gp["problem_rank"].min()
    max_rank = gp["problem_rank"].max()
    best_count = rel_performance_df[rel_performance_df["problem_rank"] == 1].groupby("method").size()
    n = gp.size()

    summary_df = pd.DataFrame({
        "method": mean.index,
        "mean": mean.values,
        "Q1": q1.values,
        "Q3": q3.values,
        "mean_rank": mean_rank.values,
        "min_rank": min_rank.values,
        "max_rank": max_rank.values,
        "best_count": best_count.reindex(mean.index).fillna(0).astype(int).values,
        "n": n.values,
    }).set_index("method")

    return summary_df

def calculate_coefficient_of_variation(agg_simple_regrets_df):
    # Calculate the coefficient of variation (CV) for each method across all problems
    cv_df = agg_simple_regrets_df.copy()
    cv_df["mean"] = cv_df.iloc[:, 2:].mean(axis=1)
    cv_df["std"] = cv_df.iloc[:, 2:].std(axis=1)
    cv_df["cv"] = cv_df["std"] / cv_df["mean"]
    # Average CV per method and return the dictionary of method -> avg CV
    avg_cv = cv_df.groupby("method")["cv"].mean().to_dict()
    return avg_cv

def summary_to_latex(
        summary_df, 
        avg_cv, 
        filename="summary.tex", 
        pairwise_p_rel=None, 
        pairwise_p_rank=None,
        friedman_res=None
    ):
    # header/footer unchanged
    header = r"""
\begin{table}
\caption{
\textbf{Overall performance comparison of LMABO against all baselines across 50 optimization problems}. 
P-values from Friedman tests in the last row indicate statistically significant differences among methods for both RP and rank.
The third and fifth columns show p-values of post-hoc pairwise comparisons between LMABO and each method, which confirm that the differences in performance between LMABO and all methods are significant.
Exploitative AFs are marked in \textcolor{blue}{blue} and explorative AFs are marked in \textcolor{magenta}{magenta} (see Appendix \ref{appendix:acquisition_functions} for details).
}
\label{tab:aggregated}
\centering
\renewcommand{\arraystretch}{1.2}
\begin{tabular}{@{}lccccc@{}}
\toprule
\textbf{Method} & \begin{tabular}[c]{@{}c@{}}\textbf{Mean RP} \\ \textbf{(Interquartile Range)} \end{tabular} & \begin{tabular}[c]{@{}c@{}}\textbf{P-value} \\ \textbf{(RP)}\end{tabular} & \begin{tabular}[c]{@{}c@{}}\textbf{Mean Rank} \\ \textbf{(Min - Max)}\end{tabular} & \begin{tabular}[c]{@{}c@{}}\textbf{P-value} \\ \textbf{(Rank)}\end{tabular} & \begin{tabular}[c]{@{}c@{}}\textbf{CV of} \\ \textbf{(AUC)}\end{tabular}\\
\midrule
\multicolumn{5}{l}{\textit{Static Acquisition Functions}} \\
    """
    footer = r"""\bottomrule
    \end{tabular}
    \end{table}
    """
    rows = []
    for m in methods_order:
        display = method_name_mapping[m]
        if m not in summary_df.index:
            rows.append(f"{display} & -- & -- & -- & -- \\\\")
            continue

        row = summary_df.loc[m]
        mean = row["mean"]
        q1 = row["Q1"]
        q3 = row["Q3"]
        mean_r = row["mean_rank"]
        min_r = int(row["min_rank"])
        max_r = int(row["max_rank"])

        perf_str = f"{mean:.3f} ({q1:.3f}--{q3:.3f})"
        rank_str = f"{mean_r:.2f} ({min_r}--{max_r})"

        p_rel = (pairwise_p_rel.get(m) if pairwise_p_rel is not None else None)
        p_rank = (pairwise_p_rank.get(m) if pairwise_p_rank is not None else None)
        p_rel_str = f"{p_rel:.3e}" if p_rel is not None else "--"
        p_rank_str = f"{p_rank:.3e}" if p_rank is not None else "--"
        cv = avg_cv.get(m, None)
        cv_str = f"{cv:.3f}" if cv is not None else "--"

        rows.append(f"{display} & {perf_str} & {p_rel_str} & {rank_str} & {p_rank_str} & {cv_str} \\\\")
        if m == "qJES":
            rows.append(r"\midrule")
            rows.append(r"\multicolumn{5}{l}{\textit{Simple Meta-strategies}} \\")
        elif m == "bo_explore_exploit":
            rows.append(r"\midrule")
            rows.append(r"\multicolumn{5}{l}{\textit{LLM-based Methods}} \\")
        elif m == "llmgp":
            rows.append(r"\midrule")
            rows.append(r"\multicolumn{5}{l}{\textit{Adaptive Portfolio Methods}} \\")
        elif m == "esp":
            rows.append(r"\midrule")
    # finally add a row for Friedman test p-values
    rows.append(f"\multicolumn{{2}}{{l}}{{\textit{{P-values of Friedman Tests}}}} & {friedman_res['rel']['p']:.3e} & & {friedman_res['rank']['p']:.3e} & \\")

    table = header + "\n".join(rows) + "\n" + footer
    with open(filename, "w") as f:
        f.write(table)
    print(f"Wrote LaTeX summary to {filename}")

def ablation_summary_to_latex(
        summary_df, 
        avg_cv, 
        filename,
        pairwise_p_rel=None, 
        pairwise_p_rank=None
    ):
    # header/footer unchanged
    header = r"""
\begin{table}
\caption{
    \textbf{Ablation study on the components of LMABO}. 
    We analyze the contribution of LMABO's key components by comparing the full model to four ablated versions: LMABO-AB1, LMABO-AB2, and LMABO-AB3, which remove the remaining budget, GP model characteristics, and shortest distance information, respectively; and LMABO-OPS, which uses a smaller language model (Qwen3-8B). 
    The Mean RP and Mean Rank are calculated using the same global ranking of all baseline and ablation methods as in Table \ref{tab:aggregated}. 
    The p-values from a one-sided Wilcoxon signed-rank test with Holm-Bonferroni correction compare each ablated model against the full LMABO.
}
\label{tab:ablation}
\centering
\renewcommand{\arraystretch}{1.2}
\begin{tabular}{@{}lccrr@{}}
\toprule
\textbf{Method} & \begin{tabular}[c]{@{}c@{}}\textbf{Mean RP} \\ \textbf{(Interquartile Range)} \end{tabular} & \begin{tabular}[c]{@{}c@{}}\textbf{P-value} \\ \textbf{(RP)}\end{tabular} & \begin{tabular}[c]{@{}c@{}}\textbf{Mean Rank} \\ \textbf{(Min - Max)}\end{tabular} & \begin{tabular}[c]{@{}c@{}}\textbf{P-value} \\ \textbf{(Rank)}\end{tabular} & \begin{tabular}[c]{@{}c@{}}\textbf{CV of} \\ \textbf{(AUC)}\end{tabular}\\
\midrule
\multicolumn{5}{l}{\textit{LMABO without}} \\
    """
    footer = r"""
    \bottomrule
    \end{tabular}
    \end{table}
    """
    rows = []
    for m in methods_order_ablation:
        display = method_name_mapping[m]
        if m not in summary_df.index:
            rows.append(f"{display} & -- & -- & -- & -- \\\\")
            continue

        row = summary_df.loc[m]
        mean = row["mean"]
        q1 = row["Q1"]
        q3 = row["Q3"]
        mean_r = row["mean_rank"]
        min_r = int(row["min_rank"])
        max_r = int(row["max_rank"])

        perf_str = f"{mean:.3f} ({q1:.3f}--{q3:.3f})"
        rank_str = f"{mean_r:.2f} ({min_r}--{max_r})"

        p_rel = (pairwise_p_rel.get(m) if pairwise_p_rel is not None else None)
        p_rank = (pairwise_p_rank.get(m) if pairwise_p_rank is not None else None)
        p_rel_str = f"{p_rel:.3e}" if p_rel is not None else "--"
        p_rank_str = f"{p_rank:.3e}" if p_rank is not None else "--"
        cv = avg_cv.get(m, None)
        cv_str = f"{cv:.3f}" if cv is not None else "--"

        rows.append(f"{display} & {perf_str} & {p_rel_str} & {rank_str} & {p_rank_str} & {cv_str} \\\\")
        if m == "lmabo-ab4":
            rows.append(r"\midrule")
            rows.append(r"\multicolumn{5}{l}{\textit{LMABO using other LLMs}} \\")
        if m == "lmabo-gpt":
            rows.append(r"\midrule")

    table = header + "\n".join(rows) + "\n" + footer
    with open(filename, "w") as f:
        f.write(table)
    print(f"Wrote LaTeX ablation summary to {filename}")

def curated_summary_to_latex(        
    summary_df, 
    avg_cv, 
    filename,
    pairwise_p_rel=None, 
    pairwise_p_rank=None
):
        # header/footer unchanged
    header = r"""
\begin{table}
\caption{
    \textbf{Comparing adaptive portfolio methods between using a large portfolio and a curated portfolio}. 
}
\label{tab:curated}
\centering
\renewcommand{\arraystretch}{1.2}
\begin{tabular}{@{}lccrr@{}}
\toprule
\textbf{Method} & \begin{tabular}[c]{@{}c@{}}\textbf{Mean RP} \\ \textbf{(Interquartile Range)} \end{tabular} & \begin{tabular}[c]{@{}c@{}}\textbf{P-value} \\ \textbf{(RP)}\end{tabular} & \begin{tabular}[c]{@{}c@{}}\textbf{Mean Rank} \\ \textbf{(Min - Max)}\end{tabular} & \begin{tabular}[c]{@{}c@{}}\textbf{P-value} \\ \textbf{(Rank)}\end{tabular} & \begin{tabular}[c]{@{}c@{}}\textbf{CV of} \\ \textbf{(AUC)}\end{tabular}\\
\midrule
    """
    footer = r"""
    \bottomrule
    \end{tabular}
    \end{table}
    """
    rows = []
    for m in methods_order_curated:
        display = method_name_mapping[m]
        if m not in summary_df.index:
            rows.append(f"{display} & -- & -- & -- & -- \\\\")
            continue

        row = summary_df.loc[m]
        mean = row["mean"]
        q1 = row["Q1"]
        q3 = row["Q3"]
        mean_r = row["mean_rank"]
        min_r = int(row["min_rank"])
        max_r = int(row["max_rank"])

        perf_str = f"{mean:.3f} ({q1:.3f}--{q3:.3f})"
        rank_str = f"{mean_r:.2f} ({min_r}--{max_r})"

        p_rel = (pairwise_p_rel.get(m) if pairwise_p_rel is not None else None)
        p_rank = (pairwise_p_rank.get(m) if pairwise_p_rank is not None else None)
        p_rel_str = f"{p_rel:.3e}" if p_rel is not None else "--"
        p_rank_str = f"{p_rank:.3e}" if p_rank is not None else "--"
        cv = avg_cv.get(m, None)
        cv_str = f"{cv:.3f}" if cv is not None else "--"

        rows.append(f"{display} & {perf_str} & {p_rel_str} & {rank_str} & {p_rank_str} & {cv_str} \\\\")

    table = header + "\n".join(rows) + "\n" + footer
    with open(filename, "w") as f:
        f.write(table)
    print(f"Wrote LaTeX ablation summary to {filename}")

def run_stats_on_rel_perf_and_ranks(rel_performance_df, completed_problems, control_method="lmabo"):
    """
    Perform Friedman omnibus and pairwise Wilcoxon tests on:
      - relative_performance (numeric)
      - problem_rank (ranks)
    Returns dictionaries of Holm-corrected p-values for pairwise comparisons (control vs others),
    indexed by internal method key.
    """
    # filter to completed problems and drop any NaNs
    relp = rel_performance_df[rel_performance_df["problem"].isin(completed_problems)].copy()
    relp = relp.dropna(subset=["relative_performance", "problem_rank"])
    # pivot to matrices
    pivot_rel = relp.pivot(index="problem", columns="method", values="relative_performance")
    pivot_rank = relp.pivot(index="problem", columns="method", values="problem_rank")

    # keep problems that have all methods present
    pivot_rel = pivot_rel.dropna(axis=0, how="any")
    pivot_rank = pivot_rank.loc[pivot_rel.index]  # align

    available_methods = list(pivot_rel.columns)
    methods = [m for m in methods_order if m in available_methods]
    methods += [m for m in available_methods if m not in methods]

    pivot_rel = pivot_rel[methods]
    pivot_rank = pivot_rank[methods]

    # Friedman on relative performance: prepare arrays per method
    args_rel = [pivot_rel[method].values for method in methods]
    friedman_stat_rel, friedman_p_rel = stats.friedmanchisquare(*args_rel)

    # Friedman on ranks (though ranks are already ranks, still valid)
    args_rank = [pivot_rank[method].values for method in methods]
    friedman_stat_rank, friedman_p_rank = stats.friedmanchisquare(*args_rank)

    # Ensure control method exists (case-insensitive fallback)
    if control_method not in methods:
        cands = [m for m in methods if m.lower() == control_method.lower()]
        if cands:
            control_method = cands[0]
    if control_method not in methods:
        raise ValueError(f"Control method '{control_method}' not found among available methods: {methods}")

    control_rel = pivot_rel[control_method].values
    control_rank = pivot_rank[control_method].values

    # Pairwise Wilcoxon (control vs each other) on relative performance and on ranks
    pairwise = []
    for m in methods:
        if m == control_method:
            continue
        other_rel = pivot_rel[m].values
        other_rank = pivot_rank[m].values
        try:
            _, p_rel = stats.wilcoxon(control_rel, other_rel, alternative="two-sided", zero_method="wilcox", mode="approx")
        except Exception:
            p_rel = 1.0
        try:
            _, p_rank = stats.wilcoxon(control_rank, other_rank, alternative="two-sided", zero_method="wilcox", mode="approx")
        except Exception:
            p_rank = 1.0
        pairwise.append((m, float(p_rel), float(p_rank)))

    pairwise_df = pd.DataFrame(pairwise, columns=["method", "p_rel_unc", "p_rank_unc"]).set_index("method")

    # Holm correction separately
    pvals_rel = pairwise_df["p_rel_unc"].values
    rej_rel, pvals_rel_holm, _, _ = multipletests(pvals_rel, alpha=0.05, method="holm")
    pairwise_df["p_rel_holm"] = pvals_rel_holm
    pairwise_df["reject_rel"] = rej_rel

    pvals_rank = pairwise_df["p_rank_unc"].values
    rej_rank, pvals_rank_holm, _, _ = multipletests(pvals_rank, alpha=0.05, method="holm")
    pairwise_df["p_rank_holm"] = pvals_rank_holm
    pairwise_df["reject_rank"] = rej_rank

    # return friedman results and pairwise dataframe
    friedman_res = {
        "rel": {"stat": float(friedman_stat_rel), "p": float(friedman_p_rel)},
        "rank": {"stat": float(friedman_stat_rank), "p": float(friedman_p_rank)},
    }

    return {"friedman": friedman_res, "pairwise": pairwise_df, "methods": methods}

if __name__=="__main__":
    sys.stdout = open(f"report_bo.txt", 'w')

    all_methods = list(ACQ_TYPE_MAPPING.keys())
    all_methods.extend(list(ALGO_FILE_COUNT.keys()))
    all_problems = final_problems
    # exclude some methods during the main report
    excluded_methods = [
        "lmabo-context",
    ]
    all_methods = [method for method in all_methods if method not in excluded_methods]
    # Redirect output to file
    problems = []
    for item in all_problems:
        problems.append(item)

    completed_problems = report_completion(problems, all_methods, excluded_methods)
    print(f"Completed {len(completed_problems)} problems out of {len(problems)}")

    all_raw_results, empirical_optimum = load_results_and_empirical_performance(
        all_problems, 
        all_methods
    )
    # save empirical_optimum to a json file for future reference
    with open(f"{NUMERICAL_RESULTS_DIR}/empirical_optimum_full.json", "w") as f:
        json.dump(empirical_optimum, f, indent=4)
    all_simple_regrets = cal_simple_regret(all_raw_results, empirical_optimum)
    agg_simple_regrets_df = aggregate_and_to_df(all_simple_regrets, "auc")
    # save best AUC per problem to a json file for future reference
    best_auc = {}
    for problem in agg_simple_regrets_df["problem"].unique():
        best_auc[problem] = agg_simple_regrets_df[agg_simple_regrets_df["problem"] == problem].iloc[:, 2:-1].min().min()
    with open(f"{NUMERICAL_RESULTS_DIR}/best_auc_full.json", "w") as f:
        json.dump(best_auc, f, indent=4)
    avg_cv = calculate_coefficient_of_variation(agg_simple_regrets_df)
    rel_performance_df = get_relative_performance_and_rank(agg_simple_regrets_df, completed_problems)
    temp_summary_df = summary_by_method(rel_performance_df)
    print(temp_summary_df.to_string(index=False))
    print("="*200)
    for problem in completed_problems:
        rank_methods_by_problem(rel_performance_df, problem)
        print("="*200)
    summary_df = get_mean_iqr_summary(rel_performance_df)

    # Run statistical tests on relative performance and ranks using rel_performance_df.
    completed_for_stats = list_completed_problems(agg_simple_regrets_df)
    print(f"Using {len(completed_for_stats)} fully-completed problems for statistical tests")

    try:
        if len(completed_for_stats) == 0:
            raise RuntimeError("No fully-completed problems available for statistical testing.")

        # run tests on rel_performance_df filtered to completed problems
        stats_res = run_stats_on_rel_perf_and_ranks(
            rel_performance_df,
            completed_problems=completed_for_stats,
            control_method="lmabo",
        )

        # print Friedman omnibus results
        print("Friedman tests:")
        print(f"  Relative performance: chi2 = {stats_res['friedman']['rel']['stat']:.3f}, p = {stats_res['friedman']['rel']['p']:.3e}")
        print(f"  Ranks:                chi2 = {stats_res['friedman']['rank']['stat']:.3f}, p = {stats_res['friedman']['rank']['p']:.3e}")

        pairwise_df = stats_res["pairwise"]
        # pairwise_df indexed by method and contains p_rel_holm and p_rank_holm
        print("\nPairwise (control=lmabo) Holm-corrected p-values:")
        print(pairwise_df[["p_rel_holm", "p_rank_holm"]].to_string(float_format=lambda x: f"{x:.3e}"))

        # prepare dicts for latex table (map internal method -> corrected p)
        pairwise_p_rel = pairwise_df["p_rel_holm"].to_dict()
        pairwise_p_rank = pairwise_df["p_rank_holm"].to_dict()

        # regenerate LaTeX table including p-values
        summary_root = "summary"
        summary_to_latex(
            summary_df,
            avg_cv,
            filename=f"{summary_root}/full.tex",
            pairwise_p_rel=pairwise_p_rel,
            pairwise_p_rank=pairwise_p_rank,
            friedman_res=stats_res["friedman"],
        )
        # also generate ablation table
        ablation_summary_to_latex(
            summary_df,
            avg_cv,
            filename=f"{summary_root}/ablation.tex",
            pairwise_p_rel=pairwise_p_rel,
            pairwise_p_rank=pairwise_p_rank,
        )
        # also generate curated table
        curated_summary_to_latex(
            summary_df,
            avg_cv,
            filename=f"{summary_root}/curated.tex",
            pairwise_p_rel=pairwise_p_rel,
            pairwise_p_rank=pairwise_p_rank,
        )
    except Exception as e:
        raise e

    # Don't forget to close the file
    sys.stdout.close()
    # Restore standard output
    sys.stdout = sys.__stdout__