import google.generativeai as genai
import numpy as np
import time
import random
import re

from google.api_core.exceptions import ResourceExhausted

from bo import *

genai.configure(api_key="YourAPIKey")  # Replace with your actual API key
MAX_RETRIES = 10
MAX_DELAY_SECONDS = 120
INITIAL_DELAY_SECONDS = 1

INITIAL_PROMPT_CONTENT = """
You are an expert in Bayesian Optimization, specifically tasked with recommending the most suitable acquisition function for the next iteration. 
Your goal is to advise on the optimal strategy to efficiently find the global minimum of a black-box function.

We use a Gaussian Process with a Matern 5/2 kernel with ARD. Prefer exploration in the early iterations and exploitation in the later ones, but always consider the current state of the optimization process.

I will provide you with a summary of the Bayesian Optimization process at each step. This summary will include the following information:
- **N:** The total number of points evaluated so far.
- **Remaining iterations:** The number of iterations left in the optimization process.
- **D:** The dimensionality of the search space (number of input parameters).
- **f_min:** The current best (lowest) observed objective value.
- **Shortest distance**: The shortest distance from the last point to any other point, indicating whether it is exploiting too much.
- **Model lengthscales:** These are crucial hyperparameters of the Gaussian Process model's kernel. They describe how the model perceives the smoothness and relevance of each input dimension to the objective function. 
You will receive their range (min/max), mean, and standard deviation, along with qualitative descriptions of their variability and overall scale (e.g., if they are generally small, implying a complex function, or large, implying a smooth one).
- **Model outputscale: ** It defines the overall magnitude or amplitude of the function's variation.

Available acquisition functions you can choose from, with brief descriptions of their primary uses:
1.  **PI (Probability of Improvement):** 
2.  **LogPI (Log Probability of Improvement):** 
3.  **EI (Expected Improvement):** 
4.  **LogEI (Log Expected Improvement):** 
5.  **UCB (Upper Confidence Bound):** 
6.  **PosMean (Posterior Mean):** 
7.  **PosSTD (Posterior Standard Deviation):** 
8.  **TS (Thompson Sampling):**
9.  **qKG (Knowledge Gradient):** 
10. **qPES (Predictive Entropy Search):** 
11. **qMES (Max-value Entropy Search):**
12. **qJES (Joint Entropy Search):** 

At each step:
- **Review the provided summary of the optimization process and consider the current state of the optimization.**
- **Select the acquisition function that you believe will be best for the optimization process.**
- **Avoid reusing acquisition functions that failed to improve the objective function in previous iterations.**

**Please respond with ONLY the abbreviation of the selected acquisition function, followed by a colon and then a brief justification. Do not include any other text, greetings, or additional formatting.**
"""

FOLLOW_UP_PROMPT_TEMPLATE = """
Current optimization state:
- N: {N} 
- Remaining iterations: {remaining}
- D: {D}
- f_min: {f_min:.3f}
- Shortest distance: {shortest_dist}
- Lengthscales: Range [{min_ls:.3f}, {max_ls:.3f}], Mean {mean_ls:.3f} (Std Dev {std_ls:.3f})
- Outputscale: {outputscale}
"""

FINAL_GUESS = """
Now that you have finished the optimization process, can you guess which function is this?
"""

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

def suggest_acq_type(chat, train_X, train_Y, acq_type_list, bounds, lengthscales, outputscale, remaining_iterations):
    # --- NEW: Calculate shortest distance of the last point relative to bounds ---
    shortest_dist = get_shortest_distance_from_last_point(train_X, bounds)
    if lengthscales is not None:
        # --- Calculate descriptive statistics ---
        min_ls = np.min(lengthscales)
        max_ls = np.max(lengthscales)
        mean_ls = np.mean(lengthscales)
        std_ls = np.std(lengthscales)

    prompt = FOLLOW_UP_PROMPT_TEMPLATE.format(
        N=train_Y.shape[0],
        remaining=remaining_iterations,
        D=train_X.shape[1],
        f_min=torch.round(train_Y.min().detach().cpu(), decimals=3).item(),
        # last_y_values=train_Y[-5:].detach().cpu().squeeze(-1).tolist(),
        # acquisition_history=acq_type_list,
        shortest_dist=shortest_dist,
        min_ls=min_ls,
        max_ls=max_ls,
        mean_ls=mean_ls,
        std_ls=std_ls,
        # ls_variation_description=ls_variation_description,
        # general_ls_scale_description=general_ls_scale_description,
        outputscale=outputscale
    )
    print(f"Iter {len(acq_type_list)}|", prompt)
    retries = 0
    current_delay = INITIAL_DELAY_SECONDS
    
    llm_suggested_af = "PosSTD"
    while retries < MAX_RETRIES:
        try:
            # Send the updated summary to the active chat
            response = chat.send_message(prompt)

            if response.text:
                llm_suggested_af_raw = response.text.strip()
                # Assuming the LLM responds with "AF_ABBREVIATION" or "AF_ABBREVIATION: Justification"
                if ":" in llm_suggested_af_raw:
                    llm_suggested_af = llm_suggested_af_raw.split(":")[0].strip()
                    justification = llm_suggested_af_raw.split(":")[1:]
                    llm_suggested_af = llm_suggested_af.strip()
                else:
                    llm_suggested_af = llm_suggested_af_raw.strip()
                    justification = "Nothing"

                print(f"LLM suggested AF: {llm_suggested_af} justified by: {justification}")
                break # Success, exit retry loop

            else:
                print("LLM returned no text content in response.")
                llm_suggested_af = "UCB" # Or handle as an error
                break

        except ResourceExhausted as e:
            error_message = str(e) # Get the full string representation of the error
            suggested_delay_seconds = current_delay # Default to current backoff delay

            # Use regex to find the retry_delay from the error string
            match = re.search(r"retry_delay \{[\s\n]+seconds: (\d+)[\s\n]+\}", error_message)
            if match:
                try:
                    suggested_delay_seconds = int(match.group(1))
                    print(f"API suggested waiting {suggested_delay_seconds} seconds (parsed from error message).")
                except ValueError:
                    print("Could not parse suggested retry delay from error message. Using exponential backoff.")
            else:
                print("No specific retry_delay found in error message. Using exponential backoff.")


            print(f"Rate limit hit (Retry {retries+1}/{MAX_RETRIES}).")
            
            # Use the parsed suggested delay, or our exponential backoff
            wait_time = suggested_delay_seconds + random.uniform(0, suggested_delay_seconds * 0.1) # Add jitter
            wait_time = min(wait_time, MAX_DELAY_SECONDS) # Cap the wait time

            print(f"Waiting for {wait_time:.2f} seconds...")
            time.sleep(wait_time)

            retries += 1
            current_delay = min(current_delay * 2, MAX_DELAY_SECONDS) # Double delay for next retry

        except Exception as e:
            print(f"An unexpected error occurred during API call: {e}")
            break # Exit retry loop for other errors
    else:
        print(f"Failed to get LLM response for iteration {len(acq_type_list)} after {MAX_RETRIES} retries. Using default AF.")
        return "UCB", chat
    return llm_suggested_af, chat
    
def check_available_model():
    # List all available models
    print("Listing available models and their supported methods:")
    for m in genai.list_models():
        # Check if the model supports the 'generateContent' method
        if 'generateContent' in m.supported_generation_methods:
            print(f"  Model Name: {m.name}, Supported Methods: {m.supported_generation_methods}")
        else:
            print(f"  Model Name: {m.name}, (Does NOT support generateContent)")

def last_guess(chat):
    try:
        response = chat.send_message(FINAL_GUESS)
        if response.text:
            print(response.text)
        else:
            print("No text guesses")
    except ResourceExhausted as e:
        print("No more resources - no guessing")

def lm_assisted_adaptive_bo(objective_func, X_init, Y_init, bounds, num_iterations):
    # Generate initial training data
    train_X  = X_init.clone()
    train_Y  = Y_init.clone()
    best_values = [train_Y.min().item()]
    acq_type_list = []
    # init LLM
    # check_available_model()
    model = genai.GenerativeModel(
        'gemini-2.5-flash-preview-05-20', 
    )
    # --- START THE CHAT SESSION ---
    print("Starting Gemini chat session with initial context...")
    try:
        chat = model.start_chat(history=[
            {"role": "user", "parts": [INITIAL_PROMPT_CONTENT]}
        ])
        # The first response from the model just confirms it understands the context
        # You might want to print/log this response, or just ignore it
        initial_response = chat.send_message("Do you understand the context?")
        print(f"Gemini's initial acknowledgement: {initial_response.text.strip()}")
    except Exception as e:
        print(f"Error starting chat or initial acknowledgement: {e}")
        print("Please check your API key, model availability, and network connection.")
        exit() # Exit if we can't even start the chat
    # optimization loop
    gp = fit_gp(train_X, train_Y)
    lengthscales = gp.covar_module.base_kernel.lengthscale.detach().cpu().numpy()
    outputscale = gp.covar_module.outputscale.detach().cpu().numpy()
    remaining_iterations = num_iterations
    for i in range(num_iterations):
        # use LLM to suggest the best acq_type
        acq_type, chat = suggest_acq_type(
            chat, 
            train_X,
            train_Y, 
            acq_type_list, 
            bounds,
            lengthscales,
            outputscale,
            remaining_iterations
        )
        acq_type_list.append(acq_type)
        # run one BO iter with the acq_type suggested by LLM
        train_X, train_Y, gp = bo_single_iteration(train_X, train_Y, acq_type, objective_func, bounds)
        lengthscales = gp.covar_module.base_kernel.lengthscale.detach().cpu().numpy()
        outputscale = gp.covar_module.outputscale.detach().cpu().numpy()
        # Store best observed value
        best_values.append(train_Y.min().item())
        print(f"Current best value: {train_Y.min().item()}")
        remaining_iterations -= 1
    last_guess(chat)
    return np.array(best_values), acq_type_list