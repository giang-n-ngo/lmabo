import google.generativeai as genai
import numpy as np
import time
import random
import re

from google.api_core.exceptions import ResourceExhausted

from moo import *
from key import API_KEYS
from utils import get_shortest_distance_from_last_point

genai.configure(api_key=API_KEYS[0])  # Replace with your actual API key
MAX_RETRIES = 10
MAX_DELAY_SECONDS = 120
INITIAL_DELAY_SECONDS = 1

INITIAL_PROMPT_CONTENT = """
You are an expert in Bayesian Optimization, specifically tasked with recommending the most suitable acquisition function for the next iteration. 
Your goal is to advise on the optimal strategy to efficiently push the Pareto frontier of a black-box multi-objective function.

We use Gaussian Processes with a Matern 5/2 kernel with ARD. Prefer exploration in the early iterations and exploitation in the later ones, but always consider the current state of the optimization process.

I will provide you with a summary of the Bayesian Optimization process at each step. This summary will include the following information:
- **N:** The total number of points evaluated so far.
- **Remaining iterations:** The number of iterations left in the optimization process.
- **D:** The dimensionality of the search space (number of input parameters).
- **J:** The number of objectives being optimized.
- **HV:** The current hypervolume.
- **Shortest distance**: The shortest distance from the last point to any other point, indicating whether it is exploiting too much.
- **Model lengthscales:** These are crucial hyperparameters of the Gaussian Process model's kernel. They describe how the model perceives the smoothness and relevance of each input dimension to the objective function. 
You will receive their range (min/max), mean, and standard deviation, along with qualitative descriptions of their variability and overall scale (e.g., if they are generally small, implying a complex function, or large, implying a smooth one).
- **Model outputscale: ** It defines the overall magnitude or amplitude of the function's variation.

Available acquisition functions you can choose from, with brief descriptions of their primary uses:
1. **qNEHVI**: q-Noisy Expected Hypervolume Improvement
2. **qLogNEHVI**: q-Log Noisy Expected Hypervolume Improvement
3. **qHVKG**: Batch Hypervolume Knowledge Gradient using one-shot optimization
4. **qLBMOJES**: Multi-objective joint entropy search
5. **qLBMOMES**: The acquisition function for the multi-objective Max-value Entropy Search
6. **qMOPES**: The acquisition function for Predictive Entropy Search on multi-objective problems
7. **qParEGO**: ParEGO with Chebyshev scalarization on top of qNoisyExpectedImprovement

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
- J: {J}
- HV: {hv:.3f}
- Shortest distance: {shortest_dist}
- Lengthscales: Range [{min_ls:.3f}, {max_ls:.3f}], Mean {mean_ls:.3f} (Std Dev {std_ls:.3f})
- Outputscale: {outputscale}
"""

FINAL_GUESS = """
Now that you have finished the optimization process, can you guess which function is this?
"""

def suggest_acq_type_moo(
        chat, 
        train_X, train_Y, 
        acq_type_list, 
        bounds, 
        lengthscales, 
        outputscale, 
        remaining_iterations,
        hv
    ):
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
        J=train_Y.shape[-1],
        hv=hv,
        shortest_dist=shortest_dist,
        min_ls=min_ls,
        max_ls=max_ls,
        mean_ls=mean_ls,
        std_ls=std_ls,
        outputscale=outputscale
    )
    print(f"Iter {len(acq_type_list)}|", prompt)
    retries = 0
    current_delay = INITIAL_DELAY_SECONDS
    
    llm_suggested_af = "qNEHVI"
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
                llm_suggested_af = "qNEHVI" # Or handle as an error
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
        return "Intentional Incorrect AF", chat, "Failed to get LLM response after retries"
    return llm_suggested_af, chat, response.text.strip()  # Return the chat object and the response text for logging


def last_guess(chat):
    try:
        response = chat.send_message(FINAL_GUESS)
        if response.text:
            print(response.text)
            return response.text.strip()
        else:
            print("No text guesses")
            return "No guesses"
    except ResourceExhausted as e:
        print("No more resources - no guessing")
        return "No more resources - no guessing"

def get_moo_lengthscales(model_list):
    """
    Extracts the lengthscales from a ModelList Gaussian Process model's covariance module.
    Returns them as a numpy array.
    """
    all_lengthscales = []
    all_outputscales = []
    for model in model_list.models:
        lengthscales = model.covar_module.base_kernel.lengthscale.detach().cpu().numpy()
        outputscale = model.covar_module.outputscale.detach().cpu().numpy()
        all_lengthscales.append(lengthscales)
        all_outputscales.append(outputscale)
    lengthscales = np.stack(all_lengthscales)
    outputscales = np.stack(all_outputscales)
    return lengthscales, outputscales

def lm_assisted_adaptive_moo(
    objective_func, 
    X_init, Y_init,
    bounds, 
    num_iterations
):
    max_hv = objective_func._max_hv
    # Generate initial training data
    train_X  = X_init.clone()
    train_Y  = Y_init.clone()
    hv_list, log_hv_diff_list = [], []
    hv, log_hv_difference = get_moo_results(
        train_Y, 
        objective_func.ref_point, 
        max_hv
    )
    hv_list.append(hv)
    log_hv_diff_list.append(log_hv_difference)
    acq_type_list = []
    # init LLM
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
    model_list = fit_moo_gp(train_X, train_Y, bounds)
    lengthscales, outputscale = get_moo_lengthscales(model_list)
    remaining_iterations = num_iterations
    messages = []  # Store messages for later use
    messages.append(initial_response.text.strip())  # Store the initial prompt content
    for iteration_idx in range(num_iterations):
        # use LLM to suggest the best acq_type
        acq_type, chat, message = suggest_acq_type_moo(
            chat, 
            train_X,
            train_Y, 
            acq_type_list, 
            bounds,
            lengthscales,
            outputscale,
            remaining_iterations,
            hv
        )
        acq_type_list.append(acq_type)
        # run one BO iter with the acq_type suggested by LLM
        train_X, train_Y, model_list = mobo_single_iteration(train_X, train_Y, acq_type, objective_func, bounds)
        lengthscales, outputscale = get_moo_lengthscales(model_list)
        hv, log_hv_difference = get_moo_results(
            train_Y, 
            objective_func.ref_point, 
            max_hv
        )
        hv_list.append(hv)
        log_hv_diff_list.append(log_hv_difference)
        print(f"Iter {iteration_idx} | HV: {hv:.4f} | Log HV Diff: {log_hv_difference:.4f}")
        remaining_iterations -= 1
        messages.append(message)  # Store the LLM's response message
    messages.append(last_guess(chat))
    return (
        np.array(hv_list), 
        np.array(log_hv_diff_list),
        np.array(train_X.detach().cpu().numpy()), 
        np.array(train_Y.detach().cpu().numpy()),
        acq_type_list,
        messages
    )