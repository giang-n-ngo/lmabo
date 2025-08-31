import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from scipy.stats import kstest

from constants import (
    NUMERICAL_RESULTS_DIR, 
    EXP_RUNS, 
    FIG_DIR, 
    LLMGP_NUMERICAL_RESULTS_DIR,
    ACQ_TYPE_MAPPING,
    ALGO_FILE_COUNT
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

    # Calculate Euclidean distance between the normalized last point 
    # and each of the other normalized points
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
    return np.trapezoid(curve.squeeze(), dx=1.0).item()  

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

def get_problem_result(problem_list, acq_type_list, result_type, agg="mean"):
    problem_result = {}
    for problem in problem_list:
        problem_result[problem] = {}
        for acq_type in acq_type_list:
            if agg == "auc":
                problem_result[problem][acq_type] = get_result_auc(problem, acq_type, result_type)
            elif agg == "mean":
                problem_result[problem][acq_type] = get_result_mean(problem, acq_type, result_type)
    return problem_result

def get_ranking(result, result_type):
    auc_acq_type = {}
    for acq_type in result.keys():
        acq_type_values = result[acq_type]
        if len(acq_type_values) > 0:
            auc_acq_type[acq_type] = np.mean(acq_type_values).item()
        elif result_type in ["simple_regret", "cum_regret", "log_hv_diff"]:
            auc_acq_type[acq_type] = 1e50 
        else:
            auc_acq_type[acq_type] = 0.0
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
        print(f"{'Method':<15} | {'Relative':>33} | {'vs. Best':>10}")
        print("-" * 50)
        for method, rel_regret in sorted_performance.items():
            if reverse:
                print(f"{method:<15} | {mean_auc_regrets[method]:>15.3f}(\u00B1{std_auc_regrets[method]:>15.3f}) | {'+':>2}{(rel_regret-1)*100:>7.1f}%")
            else:
                print(f"{method:<15} | {mean_auc_regrets[method]:>15.3f}(\u00B1{std_auc_regrets[method]:>15.3f}) | {'-':>2}{(1-rel_regret)*100:>7.1f}%")
    return sorted_performance

def report_ranking_summary(problem_result, result_type, acq_type_list=list(ACQ_TYPE_MAPPING.keys())):
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
    active_acq_type_list=list(ACQ_TYPE_MAPPING.keys()), 
    excepted_acq_type_list=[]
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
            else:
                count = len([f for f in os.listdir(folder_path)])
                if acq in ALGO_FILE_COUNT.keys():
                    count = int(count//ALGO_FILE_COUNT[acq])
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

def plot_results(problem_name, result_type, acq_type_list=list(ACQ_TYPE_MAPPING.keys())):
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
