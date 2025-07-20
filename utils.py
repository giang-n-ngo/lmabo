import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from scipy.stats import kstest

from bo import acq_type_mapping
from constants import (
    NUMERICAL_RESULTS_DIR, 
    EXP_RUNS, 
    FIG_DIR, 
    LLMGP_NUMERICAL_RESULTS_DIR,
    CONSTRAINED_OBJECTIVE_BEST_VALUES,
    CONSTRAINED_OBJECTIVE_MAX_VALUES
)

matplotlib_colors = [
    '#1f77b4',  # Muted blue (from default 'tab10' palette)
    '#ff7f0e',  # Orange
    '#2ca02c',  # Green
    '#d62728',  # Red
    '#9467bd',  # Purple
    '#8c564b',  # Brown
    '#e377c2',  # Pink
    '#7f7f7f',  # Gray
    '#bcbd22',  # Olive/Lime Green
    '#17becf',  # Cyan/Teal
    '#a6cee3',  # Light Blue (from 'Paired' palette)
    '#b2df8a',  # Light Green
    '#fb9a99',  # Light Red
    '#fdbf6f',  # Light Orange
    '#cab2d6',   # Light Purple
    '#ff1493',  # Deep Pink
    '#32cd32',  # Lime Green
    '#ff6347',  # Tomato Red
    '#4169e1',  # Royal Blue
    '#daa520'   # Goldenrod
]

def get_shortest_distance_from_last_point(X, bounds):
    """
    Calculates the shortest Euclidean distance between the last point
    and all other points in a PyTorch tensor, after normalizing the points
    to a [0,1]^D hypercube based on the provided bounds.

    Args:
        points_tensor: A PyTorch tensor of shape (N, D), where N is the number
                       of points and D is the number of dimensions.
                       Assumes points_tensor values are within the given bounds.
        bounds_tensor: A PyTorch tensor of shape (2, D), where the first row
                       contains the lower bounds for each dimension and the second row
                       contains the upper bounds for each dimension.

    Returns:
        The shortest Euclidean distance in the normalized [0,1]^D space as a float.

    Raises:
        ValueError: If the input tensor has fewer than 2 points, or if bounds are invalid.
    """
    if X.shape[0] < 2:
        raise ValueError("Points tensor must contain at least 2 points to calculate distances.")
    if bounds.shape != (2, X.shape[1]):
        raise ValueError(f"Bounds tensor must have shape (2, D) where D is {X.shape[1]}. "
                         f"Received shape: {bounds.shape}")
    if torch.any(bounds[0] >= bounds[1]):
        raise ValueError("Lower bounds must be strictly less than upper bounds in all dimensions.")

    # Extract lower and upper bounds
    lower_bounds = bounds[0, :]
    upper_bounds = bounds[1, :]

    # Calculate the range (width) of each dimension
    ranges = upper_bounds - lower_bounds

    # Normalize the points to the [0,1]^D hypercube
    # This ensures distances are comparable across dimensions of different scales
    # Add a small epsilon to ranges to prevent division by zero for fixed dimensions if any
    epsilon = 1e-9
    normalized_points = (X - lower_bounds) / (ranges + epsilon)

    # The last normalized point
    normalized_last_point = normalized_points[-1:, :]

    # All other normalized points
    normalized_other_points = normalized_points[:-1, :]

    # Calculate Euclidean distance between the normalized last point and each of the other normalized points
    # torch.cdist is efficient for batch distances
    # The output 'distances' will be a (N-1, 1) tensor
    distances = torch.cdist(normalized_other_points, normalized_last_point, p=2)

    # Find the minimum distance among them and convert to a Python float
    shortest_dist = torch.min(distances).item()

    return shortest_dist
        
def get_rank_dict(dictionary, reverse=True):
    """
    Create a dictionary mapping items to their rank positions based on values.
    
    Args:
        dictionary: Input dictionary to rank
        reverse: If True, highest value gets rank 1 (default). If False, lowest value gets rank 1.
    
    Returns:
        dict: A dictionary where keys are the original keys and values are their ranks
    """
    # Sort items by value and enumerate to get positions
    sorted_items = sorted(dictionary.items(), key=lambda x: x[1], reverse=reverse)
    # Create a new dictionary with original keys and their rank positions (1-based)
    return {key: rank + 1 for rank, (key, _) in enumerate(sorted_items)}

def get_auc(curve):
    """
    Calculate the Area Under the Curve (AUC) for a given curve.
    
    Args:
        curve: A 1D numpy array representing the curve values.
        
    Returns:
        float: The AUC value.
    """
    assert len(curve.shape)==1, "Wrong result shape"
    assert curve.shape[0] > 2, "Not enough elements to get AUC"
    return np.trapezoid(curve.squeeze(), dx=1.0).item()  # Assuming uniform spacing of 1.0 between points

def read_raw_result(problem, acq_type, result_type):
    raw_result = []
    for exp_idx in range(EXP_RUNS):
        try:
            if acq_type == "llmgp":
                file_name = f"{LLMGP_NUMERICAL_RESULTS_DIR}/{problem}/llmgp/{exp_idx}_{result_type}.npy"
            else:
                file_name = f"{NUMERICAL_RESULTS_DIR}/{problem}/{acq_type}/{exp_idx}_{result_type}.npy"
            simple_regret = np.load(file_name)
            if np.isnan(simple_regret).any():
                print(f"Found nan value in {file_name}")
                os.remove(file_name)  # Remove the file if it contains NaN values
                continue
            raw_result.append(simple_regret)
        except FileNotFoundError:
            continue 
    return raw_result

def get_result_auc(problem, acq_type, result_type):
    raw_result = read_raw_result(problem, acq_type, result_type)
    result_auc = [get_auc(result) for result in raw_result]
    return result_auc

def get_problem_result(problem_list, acq_type_list, result_type):
    problem_result = {}
    for problem in problem_list:
        problem_result[problem] = {}
        for acq_type in acq_type_list:
            problem_result[problem][acq_type] = get_result_auc(problem, acq_type, result_type)
    return problem_result

def get_ranking(result, result_type):
    auc_acq_type = {}
    for acq_type in result.keys():
        acq_type_values = result[acq_type]
        if len(acq_type_values) > 0:
            auc_acq_type[acq_type] = np.mean(acq_type_values).item()
        else:
            auc_acq_type[acq_type] = 1e50 if result_type in ["simple_regret", "cum_regret", "log_hv_diff"] else 0.0
    if result_type in ["simple_regret", "cum_regret", "log_hv_diff"]:
        return get_rank_dict(auc_acq_type, False)
    else:
        return get_rank_dict(auc_acq_type, True)

def print_sorted_dict(dictionary, reverse=True, indent=2):
    """
    Pretty print a dictionary sorted by values.
    
    Args:
        dictionary: Input dictionary to sort and print
        reverse: If True, sort in descending order (default). If False, ascending order
        indent: Number of spaces for indentation (default=2)
    """
    # Sort the dictionary by values
    sorted_items = sorted(dictionary.items(), key=lambda x: x[1], reverse=reverse)
    
    # Get the maximum length of keys for alignment
    max_key_length = max(len(str(key)) for key in dictionary.keys())
    
    print("{")
    for key, value in sorted_items:
        # Format each line with proper indentation and alignment
        print(f"{' ' * indent}{str(key):<{max_key_length}} : {value}")
    print("}")
            
def report_relative_reg(
    problem, 
    result_type, 
    auc_acq_type_dict, 
    verbose=False, 
    reverse=False, 
):
    """
    Analyze relative performance of acquisition functions for a given problem.
    Returns a sorted dictionary of relative cumulative regrets compared to the best performer.
    
    Args:
        problem_name: Name of the test problem
        true_minimum: True minimum value of the objective function
    """
    # Dictionary to store mean cumulative regret for each acquisition type
    mean_auc_regrets = {}
    std_auc_regrets = {}
    ks_test_scores = {}
    p_values = {}
    
    # Process results for each acquisition type
    for acq_type in auc_acq_type_dict.keys():
        acq_type_values = np.stack(auc_acq_type_dict[acq_type])
        if len(acq_type_values) > 0:
            mean_auc_regrets[acq_type] = np.mean(acq_type_values, axis=0)
            std_auc_regrets[acq_type] = np.std(acq_type_values, axis=0)
            # Perform KS test for normality
            ks_test_score, p_value = check_normality(acq_type_values)
            ks_test_scores[acq_type] = ks_test_score
            p_values[acq_type] = p_value
    
    # Find the best performing method
    if reverse: # for lower better metrics
        best_method = min(mean_auc_regrets.items(), key=lambda x: x[1])
    else: # for higher better metrics
        best_method = max(mean_auc_regrets.items(), key=lambda x: x[1])
    best_method_name = best_method[0]
    best_method_regret = best_method[1]    
    
    # Calculate relative performance
    relative_performance = {
        acq_type: regret / best_method_regret 
        for acq_type, regret in mean_auc_regrets.items()
    }
    
    # Sort by performance (ascending)
    sorted_performance = dict(sorted(relative_performance.items(), key=lambda x: x[1]))
    
    if verbose:
        # Print results in a nice format
        print(f"Performance Analysis for {problem}")
        print(f"Best method: {best_method_name} (baseline)")
        print(f"Relative {result_type} AUC (compared to best):")
        print("-" * 50)
        print(f"{'Method':<15} | {'Relative':>33} | {'vs. Best':>10} | {'KS Score':>7} | {'p-value':>7}")
        print("-" * 50)
        for method, rel_regret in sorted_performance.items():
            if reverse:
                print(f"{method:<15} | {mean_auc_regrets[method]:>15.3f}(\u00B1{std_auc_regrets[method]:>15.3f}) | {'+':>2}{(rel_regret-1)*100:>7.1f}% | {ks_test_scores[method]:>7.3f} | {p_values[method]:>7.3f}")
            else:
                print(f"{method:<15} | {mean_auc_regrets[method]:>15.3f}(\u00B1{std_auc_regrets[method]:>15.3f}) | {'-':>2}{(1-rel_regret)*100:>7.1f}% | {ks_test_scores[method]:>7.3f} | {p_values[method]:>7.3f}")
    return sorted_performance

def report_ranking_summary(problem_result, result_type, acq_type_list=list(acq_type_mapping.keys())):
    problem_ranking = {}
    for problem in problem_result.keys():
        problem_ranking[problem] = get_ranking(problem_result[problem], result_type)
    acq_type_ranking_mean = {}
    acq_type_ranking_std = {}
    for acq_type in acq_type_list:
        acq_type_ranking_list = []
        for problem in problem_result.keys():
            rank = problem_ranking[problem][acq_type]
            acq_type_ranking_list.append(rank)
        acq_type_ranking_mean[acq_type] = np.mean(acq_type_ranking_list)
        acq_type_ranking_std[acq_type] = np.std(acq_type_ranking_list)
    reverse = False if result_type in ["simple_regret", "cum_regret", "log_hv_diff"] else True
    print("Mean ranking")
    print_sorted_dict(acq_type_ranking_mean, reverse=reverse)
    print("Std ranking")
    print_sorted_dict(acq_type_ranking_std, reverse=reverse)
    
def report_completion(
    problems, 
    active_acq_type_list=list(acq_type_mapping.keys()), 
    excepted_acq_type_list=[],
    constrained=False
):
    """
    Print a table showing number of completed runs for each problem and acquisition type.
    """
    print("Checking number of completed runs for each problem and acquisition type...")
    completed_problems = []
    
    # Calculate padding for pretty printing
    problem_width = max(len(str(p)) for p in problems)
    acq_width = max(len(str(a)) for a in active_acq_type_list + excepted_acq_type_list)
    
    # Print header
    header = f"{'Problem':<{problem_width}}|"
    header += "".join(f"{acq:^{acq_width}}|" for acq in active_acq_type_list + excepted_acq_type_list)
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    
    # Print each problem's row and check completion
    for problem in problems:
        row = f"{problem:<{problem_width}}|"
        problem_complete = True
        
        for acq in active_acq_type_list + excepted_acq_type_list:
            if acq == "llmgp":
                folder_path = f"{LLMGP_NUMERICAL_RESULTS_DIR}/{problem}/llmgp"
            else:
                folder_path = f"{NUMERICAL_RESULTS_DIR}/{problem}/{acq}"
            if not os.path.exists(folder_path):
                count = 0
            elif acq in [
                "lmabo", 
                "lmabo-ops",
                "lmamoo", 
                "gphedge"
            ]:
                count = len([f for f in os.listdir(folder_path) 
                            if f.endswith('.npy') or f.endswith('.txt')])
                if constrained:
                    count = int(count//5)
                else:
                    count = int(count//6)
            elif acq == "llmgp" or acq == "llambo":
                count = len([f for f in os.listdir(folder_path) if f.endswith('.npy')])
                count = int(count//3)
            else:
                count = len([f for f in os.listdir(folder_path)])
                if constrained:
                    count = int(count//3)
                else:
                   count = int(count//4)
            row += f"{count:^{acq_width}}|"
            if count < EXP_RUNS and acq not in excepted_acq_type_list:
                problem_complete = False
                
        print(row)
        if problem_complete:
            completed_problems.append(problem)
            
    print("-" * len(header))
    print("Completed: ", completed_problems)
    return completed_problems

def plot_results(problem_name, result_type, acq_type_list=list(acq_type_mapping.keys())):
    plt.figure(figsize=(16, 10))
    for i, acq_type in enumerate(acq_type_list):
        if result_type == "best_val":
            raw_result = read_raw_result(problem_name, acq_type, "train_Y")
            if raw_result[0].shape[0] > 100:
                raw_result = [result[-101:] for result in raw_result]
            else:
                raw_result = [result[-51:] for result in raw_result]
            acq_type_values = [np.minimum.accumulate(result) for result in raw_result]
        else:
            raw_result = read_raw_result(problem_name, acq_type, result_type)
            acq_type_values = raw_result
        if len(acq_type_values) > 0:
            acq_type_values = np.stack(acq_type_values)
            mean_acq_type_values = acq_type_values.mean(axis=0)
            std_acq_type_values = acq_type_values.std(axis=0)
            n_iter = np.arange(1, mean_acq_type_values.shape[0]+1)
            plt.plot(
                n_iter, 
                mean_acq_type_values, 
                color=matplotlib_colors[i],
                label=acq_type
            )
            plt.fill_between(
                n_iter, 
                mean_acq_type_values - 0.5*std_acq_type_values, 
                mean_acq_type_values + 0.5*std_acq_type_values, 
                color=matplotlib_colors[i],
                alpha=0.2
            )
    plt.xlabel("Iteration")
    plt.ylabel(result_type.capitalize())
    plt.title(f"{problem_name}")
    plt.legend()
    plt.grid(True)
    os.makedirs(f"{FIG_DIR}", exist_ok=True)
    plt.savefig(f"{FIG_DIR}/{problem_name}_{result_type}.pdf", dpi=300)  # Save plot as PDF
    plt.close()  # Close the plot to free memory

def check_normality(data):
    """
    Performs the Kolmogorov-Smirnov test for normality.

    Args:
        data (np.ndarray): The data to test for normality.

    Returns:
        tuple: A tuple containing the KS statistic and p-value.
    """
    # Standardize the data
    mean = np.mean(data)
    std = np.std(data)
    standardized_data = (data - mean) / std

    # Perform the Kolmogorov-Smirnov test
    ks_statistic, p_value = kstest(standardized_data, 'norm')

    return ks_statistic, p_value

import numpy as np
from scipy.stats import kstest

def get_best_feasible(problem, X, Y, constraints):
    """
    Find the best feasible at each iteration
    """
    best_feasible_values = np.zeros((X.shape[0]))
    for i in range(X.shape[0]):
        feasible_indices = np.where((constraints[:(i+1), :] >= 0).all(axis=1))[0]
        if feasible_indices.size > 0:
            best_feasible_values[i] = np.min(Y[:(i+1)][feasible_indices])
        else:
            best_feasible_values[i] = CONSTRAINED_OBJECTIVE_MAX_VALUES[problem]  # No feasible solution
    return best_feasible_values

def get_best_feasible_normalized_auc(best_feasible_values, best_known_value):
    """
    Calculate the area under the curve for the best feasible values from the first feasible point.
    """
    assert len(best_feasible_values.shape)==1, "Wrong result shape"
    assert best_feasible_values.shape[0] > 2, "Not enough elements to get AUC"
    if np.all(best_feasible_values == np.inf):
        return float("inf")  # No feasible solutions found
    if np.all(best_feasible_values[:-1] == np.inf): # all except last point are inf
        return best_feasible_values[-1] - best_known_value # only one point, no AUC
    first_feasible_index = np.where(best_feasible_values != np.inf)[0][0]
    feasible_regrets = best_feasible_values[first_feasible_index:] - best_known_value
    auc = np.trapezoid(feasible_regrets.squeeze(), dx=1.0).item()
    normalized_auc = auc / np.sum(best_feasible_values != np.inf)
    return normalized_auc.item()

def get_time_to_first_feasible(constraints, dim):
    """
    Calculate the time to the first feasible solution.
    """
    n_starting_points = 2*dim+1
    feasible_indices = np.where((constraints[n_starting_points:, :] >= 0).all(axis=1))[0]
    if feasible_indices.size > 0:
        time_to_first_feasible = feasible_indices[0].item() + 1
    else:
        time_to_first_feasible = 50 if dim <= 10 else 100  # No feasible solution found
    return time_to_first_feasible

def get_constrained_result(problem, acq_type):
    """
    Get the constrained results for a given problem and acquisition type.
    Returns a dictionary with time to first feasible and AUC of best feasible curve.
    """
    folder_path = f"numerical_results/{problem}/{acq_type}/"
    result_dict = {
        "time_to_first_feasible": {
            "values": [],
        },
        "best_feasible_auc": {
            "values": [],
        },
    }
    for i in range(10):
        train_X = np.load(f"{folder_path}{i}_train_X.npy")
        train_Y = np.load(f"{folder_path}{i}_train_Y.npy")
        constraints = np.load(f"{folder_path}{i}_train_constraints.npy").squeeze(-1)
        if constraints.ndim == 1:
            constraints = constraints.reshape(-1, 1)
        best_feasible_values = get_best_feasible(problem, train_X, train_Y, constraints)
        normalized_auc = get_best_feasible_normalized_auc(
            best_feasible_values, 
            CONSTRAINED_OBJECTIVE_BEST_VALUES[problem]
        )
        time_to_first_feasible = get_time_to_first_feasible(constraints, train_X.shape[1])
        result_dict["time_to_first_feasible"]["values"].append(time_to_first_feasible)
        result_dict["best_feasible_auc"]["values"].append(normalized_auc)
    result_dict["time_to_first_feasible"]["mean"] = np.mean(result_dict["time_to_first_feasible"]["values"]).item()
    result_dict["time_to_first_feasible"]["std"] = np.std(result_dict["time_to_first_feasible"]["values"]).item()
    result_dict["time_to_first_feasible"]["kstest"] = kstest(result_dict["time_to_first_feasible"]["values"], 'norm')
    result_dict["best_feasible_auc"]["mean"] = np.mean(result_dict["best_feasible_auc"]["values"]).item()
    result_dict["best_feasible_auc"]["std"] = np.std(result_dict["best_feasible_auc"]["values"]).item()
    result_dict["best_feasible_auc"]["kstest"] = kstest(result_dict["best_feasible_auc"]["values"], 'norm')
    return result_dict

def fetch_all_constrained_results(problem_list, acq_type_list):
    """
    Fetch constrained results for all acquisition types for a given problem list.
    """
    all_result_dict = {}
    for problem in problem_list:
        all_result_dict[problem] = {}
        for acq_type in acq_type_list:
            all_result_dict[problem][acq_type] = get_constrained_result(problem, acq_type)
    return all_result_dict
    
def report_constrained_metrics(
    problem, 
    result_dicts
):
    """
    Analyze relative performance of acquisition functions for a given problem.
    Returns a sorted dictionary of relative time-to-feasible and AUC of best feasible curve compared to the best performer.
    
    Args:
        problem: Name of the test problem
        result_dicts: Performance results for all acquisition functions. 
        Keys are acquisition function names, values are dictionaries with 'time_to_first_feasible' and 'best_feasible_auc'.
    """
    # process results
    mean_time_to_first_feasible = {}
    mean_best_feasible_auc = {}
    for acq_type, result_dict in result_dicts.items():
        mean_time_to_first_feasible[acq_type] = result_dict["time_to_first_feasible"]["mean"]
        mean_best_feasible_auc[acq_type] = result_dict["best_feasible_auc"]["mean"]
    # find the best performer
    best_method_time_to_first_feasible = min(mean_time_to_first_feasible.items(), key=lambda x: x[1])
    best_method_best_feasible_auc = min(mean_best_feasible_auc.items(), key=lambda x: x[1])
    best_method_time_to_first_feasible_name = best_method_time_to_first_feasible[0]
    best_method_time_to_first_feasible_value = best_method_time_to_first_feasible[1]
    best_method_best_feasible_auc_name = best_method_best_feasible_auc[0]
    best_method_best_feasible_auc_value = best_method_best_feasible_auc[1]
    print(problem, best_method_time_to_first_feasible_name, best_method_best_feasible_auc_name)
    # calculate relative performance
    relative_time_to_first_feasible = {
        acq_type: (value / best_method_time_to_first_feasible_value) 
        for acq_type, value in mean_time_to_first_feasible.items()
    }
    relative_best_feasible_auc = {
        acq_type: (value / best_method_best_feasible_auc_value) 
        for acq_type, value in mean_best_feasible_auc.items()
    }
    # sort by relative performance
    sorted_relative_time_to_first_feasible = dict(sorted(relative_time_to_first_feasible.items(), key=lambda item: item[1]))
    sorted_relative_best_feasible_auc = dict(sorted(relative_best_feasible_auc.items(), key=lambda item: item[1]))
    # print the results
    print(f"Performance report for {problem}:")
    print(f"Best method by normalized AUC: {best_method_best_feasible_auc_name}")
    print(f"Best method by time to first feasible: {best_method_time_to_first_feasible_name}")
    print("-" * 100)
    print(f"{'Method':<8} | {'Normalized AUC':>20} | {'vs. Best':>10} | {'KS Score':>7} | {'p-value':>7}| {'Time to first feasible':>20} | {'vs. Best':>10} | {'KS Score':>7} | {'p-value':>7}")
    print("-" * 100)
    for acq_type in sorted_relative_best_feasible_auc.keys():
        auc_value = mean_best_feasible_auc[acq_type]
        auc_kstest = result_dicts[acq_type]["best_feasible_auc"]["kstest"]
        time_to_first_feasible_value = mean_time_to_first_feasible[acq_type]
        time_to_first_feasible_kstest = result_dicts[acq_type]["time_to_first_feasible"]["kstest"]
        print(f"{acq_type:<8} | {auc_value:>20.4f} | {sorted_relative_best_feasible_auc[acq_type]:>10.4f} | {auc_kstest.statistic:>7.4f} | {auc_kstest.pvalue:>7.4f} | {time_to_first_feasible_value:>20.4f} | {sorted_relative_time_to_first_feasible[acq_type]:>10.4f} | {time_to_first_feasible_kstest.statistic:>7.4f} | {time_to_first_feasible_kstest.pvalue:>7.4f}")
    print("-" * 100)