import google.generativeai as genai
import matplotlib.pyplot as plt
import numpy as np
import os
import torch

from bo import prepare_objective_func, calculate_cumulative_regret, acq_type_mapping
from constants import NUMERICAL_RESULTS_DIR, EXP_RUNS, OBJECTIVE_FUNCTIONS, FIG_DIR

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
    '#cab2d6'   # Light Purple
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
 
def check_available_model():
    # List all available models
    print("Listing available models and their supported methods:")
    for m in genai.list_models():
        # Check if the model supports the 'generateContent' method
        if 'generateContent' in m.supported_generation_methods:
            print(f"  Model Name: {m.name}, Supported Methods: {m.supported_generation_methods}")
        else:
            print(f"  Model Name: {m.name}, (Does NOT support generateContent)")

def get_problem_cum_reg(problem, acq_type):
    objective_func, dim, bounds = prepare_objective_func(problem)
    true_minimum = objective_func._optimal_value
    total_cum_reg = 0
    # Load all runs for this acquisition type
    for exp_idx in range(EXP_RUNS):
        file_name = f"{NUMERICAL_RESULTS_DIR}/{problem}/{acq_type}/{exp_idx}.npy"
        best_values = np.load(file_name)
        cum_regret = calculate_cumulative_regret(best_values, true_minimum)[-1]
        print(f"Run {exp_idx}: {cum_regret}")
        total_cum_reg += cum_regret
    print(total_cum_reg/EXP_RUNS)
        
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

def get_ranking(problem):
    objective_func, dim, bounds = prepare_objective_func(problem)
    true_minimum = objective_func._optimal_value
    legends = list(acq_type_mapping.keys())
    legends.append("lmabo")
    cum_reg_acq_type = {}
    for i, acq_type in enumerate(legends):
        acq_type_values = []
        for exp_idx in range(EXP_RUNS):
            try:
                file_name = f"{NUMERICAL_RESULTS_DIR}/{problem}/{acq_type}/{exp_idx}.npy"
                best_values = np.load(file_name)
                acq_type_values.append(best_values)
            except:
                pass
        if len(acq_type_values) > 0:
            all_cum_reg = []
            for row in acq_type_values:
                cum_reg = calculate_cumulative_regret(row, true_minimum)[-1]
                all_cum_reg.append(cum_reg)
            cum_reg_acq_type[acq_type] = np.mean(all_cum_reg)
        else:
            cum_reg_acq_type[acq_type] = 1e50
    return get_rank_dict(cum_reg_acq_type, False)

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
        
def report_ranking(problem_list):
    problem_ranking = {}
    for problem in problem_list:
        problem_ranking[problem] = get_ranking(problem)
    legends = list(acq_type_mapping.keys())
    legends.append("lmabo")
    acq_type_ranking = {}
    for i, acq_type in enumerate(legends):
        acq_type_ranking[acq_type] = 0
        for problem in problem_list:
            rank = problem_ranking[problem][acq_type]
            acq_type_ranking[acq_type] = rank + acq_type_ranking[acq_type]
    print_sorted_dict(acq_type_ranking)
    
def report_completion():
    """
    Print a table showing number of completed runs for each problem and acquisition type.
    """
    problems = [item.__name__ for item in OBJECTIVE_FUNCTIONS]
    # Add LMABO to acquisition types for complete view
    all_acq_types = list(acq_type_mapping.keys()) + ["lmabo"]
    completed_problems = []
    
    # Calculate padding for pretty printing
    problem_width = max(len(str(p)) for p in problems)
    acq_width = max(len(str(a)) for a in all_acq_types)
    
    # Print header
    header = f"{'Problem':<{problem_width}}|"
    header += "".join(f"{acq:^{acq_width}}|" for acq in all_acq_types)
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    
    # Print each problem's row and check completion
    for problem in problems:
        row = f"{problem:<{problem_width}}|"
        problem_complete = True
        
        for acq in all_acq_types:
            folder_path = f"{NUMERICAL_RESULTS_DIR}/{problem}/{acq}"
            if not os.path.exists(folder_path):
                count = 0
            else:
                count = len([f for f in os.listdir(folder_path) 
                           if f.endswith('.npy') and not f.endswith('_acq_types.npy')])
            row += f"{count:^{acq_width}}|"
            
            # Check if this acquisition type has all runs
            if count < EXP_RUNS:
                problem_complete = False
                
        print(row)
        if problem_complete:
            completed_problems.append(problem)
            
    print("-" * len(header))
    print("Completed: ", completed_problems)
    return completed_problems
            
def report_relative_cumreg(problem):
    """
    Analyze relative performance of acquisition functions for a given problem.
    Returns a sorted dictionary of relative cumulative regrets compared to the best performer.
    
    Args:
        problem_name: Name of the test problem
        true_minimum: True minimum value of the objective function
    """
    objective_func, dim, bounds = prepare_objective_func(problem)
    true_minimum = objective_func._optimal_value
    # Dictionary to store mean cumulative regret for each acquisition type
    mean_cum_regrets = {}
    
    # Process results for each acquisition type
    for acq_type in list(acq_type_mapping.keys()) + ["lmabo"]:
        acq_type_values = []
        
        # Load all runs for this acquisition type
        for exp_idx in range(EXP_RUNS):
            try:
                file_name = f"{NUMERICAL_RESULTS_DIR}/{problem}/{acq_type}/{exp_idx}.npy"
                best_values = np.load(file_name)
                cum_regret = calculate_cumulative_regret(best_values, true_minimum)
                acq_type_values.append(cum_regret)
            except:
                continue
                
        if len(acq_type_values) > 0:
            # Calculate mean cumulative regret across all runs
            mean_cum_regrets[acq_type] = np.mean(np.stack(acq_type_values), axis=0)[-1]
    
    # Find the best performing method (lowest cumulative regret)
    best_method = min(mean_cum_regrets.items(), key=lambda x: x[1])
    best_method_name = best_method[0]
    best_method_regret = best_method[1]
    
    # Calculate relative performance
    relative_performance = {
        acq_type: regret / best_method_regret 
        for acq_type, regret in mean_cum_regrets.items()
    }
    
    # Sort by performance (ascending)
    sorted_performance = dict(sorted(relative_performance.items(), key=lambda x: x[1]))
    
    # Print results in a nice format
    print(f"\nPerformance Analysis for {problem}")
    print(f"Best method: {best_method_name} (baseline)")
    print("\nRelative Cumulative Regret (compared to best):")
    print("-" * 50)
    print(f"{'Method':<15} | {'Relative Regret':>15} | {'vs. Best':>10}")
    print("-" * 50)
    for method, rel_regret in sorted_performance.items():
        print(f"{method:<15} | {rel_regret:>15.3f} | {'+':>2}{(rel_regret-1)*100:>7.1f}%")
    
    return sorted_performance

def plot_results(problem_name):
    legends = list(acq_type_mapping.keys())
    legends.append("lmabo")
    plt.figure(figsize=(16, 10))
    for i, acq_type in enumerate(legends):
        acq_type_values = []
        for exp_idx in range(EXP_RUNS):
            try:
                file_name = f"{NUMERICAL_RESULTS_DIR}/{problem_name}/{acq_type}/{exp_idx}.npy"
                best_values = np.load(file_name)
                acq_type_values.append(best_values)
            except:
                print(file_name + " does not exist")
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
    plt.ylabel("Best Function Value Found")
    plt.title(f"BO on {problem_name}")
    plt.legend()
    plt.grid(True)
    os.makedirs(f"{FIG_DIR}", exist_ok=True)
    plt.savefig(f"{FIG_DIR}/{problem_name}.pdf", dpi=300)  # Save plot as PDF