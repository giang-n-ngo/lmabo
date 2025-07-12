import google.generativeai as genai
import numpy as np
import time
import random
import re

from google.api_core.exceptions import ResourceExhausted

from llm_helper import ConversationHolder
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

For context, we use a Gaussian Process as the surrogate model with a Matern 5/2 kernel with ARD for each objective.

I will provide you with a summary of the Bayesian Optimization process at each step. This summary will include the following information:
- **N:** The total number of points evaluated so far.
- **Remaining iterations:** The number of iterations left in the optimization process.
- **D:** The dimensionality of the search space (number of input parameters).
- **J:** The number of objectives being optimized.
- **HV:** The current hypervolume.
- **Shortest distance**: The shortest distance from the last point to any other point, indicating whether it is exploiting too much.
For each objective, you will also receive information about its function value range and its GP model characteristics:
- **f_range:** The range of the objective function values observed so far.
- **Model lengthscales:** These are crucial hyperparameters of the Gaussian Process model's kernel. 
They describe how the model perceives the smoothness and relevance of each input dimension to the objective function. 
You will receive their range (min/max), mean, and standard deviation.
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

When responding, select the acquisition function you deem most appropriate. 
Your justification should briefly explain why that function is suitable given the provided optimization summary, referencing relevant aspects like exploration/exploitation balance, remaining iterations, or model characteristics. 
The response should be in the format "Acquisition abbreviation: justification", similar to these examples:
- 'qLBMOMES: This is a good choice because ...'
- 'qParEGO: This is chosen given the current state of the optimization since ...'
Firstly, just give a brief confirmation that you understand the task and the available acquisition functions.
"""

FOLLOW_UP_PROMPT_TEMPLATE = """
Current optimization state:
- N: {N} 
- Remaining iterations: {remaining}
- D: {D}
- J: {J}
- HV: {hv:.3f}
- Shortest distance: {shortest_dist}
{objective_info}
"""

MOO_OBJ_STATS_TEMPLATE = """
Objective {i}:
- f_range: [{f_min:.3f}, {f_max:.3f}], Mean {f_mean:.3f} (Std Dev {f_std:.3f})
- Lengthscales: Range [{min_ls:.3f}, {max_ls:.3f}], Mean {mean_ls:.3f} (Std Dev {std_ls:.3f})
- Outputscale: {outputscale:.3f}"""

FINAL_GUESS = """
Now that you have finished the optimization process, can you guess which function is this?
"""

class LanguageModelAssistedAdaptiveMOO:
    def __init__(
        self, 
        objective_func, 
        X_init, 
        Y_init, 
        bounds, 
        num_iterations,
        llm="api",
        server_node="localhost"
    ):
        self.objective_func = objective_func
        self.train_X  = X_init.clone()
        self.train_Y  = Y_init.clone()
        self.bounds = bounds
        self.num_iterations = num_iterations
        self.llm = llm
        self.best_values = [self.train_Y.min().item()]
        self.acq_type_list = []
        # optimization loop
        self.gps = fit_moo_gp(self.train_X, self.train_Y, self.bounds)
        self._extract_scales()  # Extract lengthscales and outputscale from the GP models
        self.remaining_iterations = self.num_iterations
        self.convo = ConversationHolder(
            llm, 
            first_prompt=INITIAL_PROMPT_CONTENT, 
            full_acq_type_list=list(moo_acq_type_mapping.keys()),
            server_node=server_node
        )
        self.hv_list = []
        self.log_hv_diff_list = []

    def _get_objective_stats_prompt(self):
        """
        Constructs a prompt summarizing the statistics of the objectives.
        """
        objective_stats = []
        for i in range(self.bounds.shape[-1]):
            f_min = self.train_Y[:, i].min().item()
            f_max = self.train_Y[:, i].max().item()
            f_mean = self.train_Y[:, i].mean().item()
            f_std = self.train_Y[:, i].std().item()
            min_ls = self.lengthscales[:, i].min().item()
            max_ls = self.lengthscales[:, i].max().item()
            mean_ls = self.lengthscales[:, i].mean().item()
            std_ls = self.lengthscales[:, i].std().item()
            outputscale = self.outputscale[i].item()

            objective_stats.append(
                MOO_OBJ_STATS_TEMPLATE.format(
                    i=i + 1, 
                    f_min=f_min, 
                    f_max=f_max, 
                    f_mean=f_mean, 
                    f_std=f_std,
                    min_ls=min_ls, 
                    max_ls=max_ls, 
                    mean_ls=mean_ls, 
                    std_ls=std_ls,
                    outputscale=outputscale
                )
            )
        
        return "\n".join(objective_stats)

    def _construct_prompt(self):
        """
        Constructs the prompt for the LLM based on the current state of the optimization.
        """
        shortest_dist = get_shortest_distance_from_last_point(self.train_X, self.bounds)
        objective_stats_prompt = self._get_objective_stats_prompt()

        return FOLLOW_UP_PROMPT_TEMPLATE.format(
            N=self.train_Y.shape[0],
            remaining=self.remaining_iterations,
            D=self.train_X.shape[1],
            J=self.train_Y.shape[-1],
            hv=self.best_values[-1],
            shortest_dist=shortest_dist,
            objective_info=objective_stats_prompt
        )
    
    def _extract_scales(self):
        """Extracts lengthscales and outputscale from all Gaussian Process models."""
        self.lengthscales = []
        self.outputscale = []
        for gp in self.gps.models:
            ls = gp.covar_module.base_kernel.lengthscale.detach().cpu().numpy()
            os = gp.covar_module.outputscale.detach().cpu().numpy()
            self.lengthscales.append(ls)
            self.outputscale.append(os)
        self.lengthscales = np.array(self.lengthscales)
        self.outputscale = np.array(self.outputscale)

    def _get_moo_results(self):
        """
        Computes the hypervolume and log hypervolume difference from the current training data.
        """
        hv, log_hv_difference = get_moo_results(
            self.train_Y, 
            self.objective_func.ref_point, 
            self.objective_func._max_hv
        )
        self.hv_list.append(hv)
        self.log_hv_diff_list.append(log_hv_difference)
        return hv, log_hv_difference

    def optimize(self):
        _, _ = self._get_moo_results()
        for iteration_idx in range(self.num_iterations):
            # use LLM to suggest the best acq_type
            acq_type = acq_type = self.convo.suggest_acq_type(self._construct_prompt())
            if acq_type == "Intentional Incorrect AF":
                exit()
            self.acq_type_list.append(acq_type)
            # run one BO iter with the acq_type suggested by LLM
            self.train_X, self.train_Y, self.gps = mobo_single_iteration(
                self.train_X, 
                self.train_Y, 
                acq_type, 
                self.objective_func, 
                self.bounds
            )
            # update GP model and lengthscales
            self._extract_scales()
            hv, log_hv_difference = self._get_moo_results()
            print(f"Iter {iteration_idx} | HV: {hv:.4f} | Log HV Diff: {log_hv_difference:.4f}")
            # update remaining iterations
            self.remaining_iterations -= 1
        self.convo.last_guess(FINAL_GUESS)  
        messages = self.convo.messages
        del self.convo
        return (
            np.array(self.hv_list), 
            np.array(self.log_hv_diff_list),
            np.array(self.train_X.detach().cpu().numpy()), 
            np.array(self.train_Y.detach().cpu().numpy()),
            np.array(self.acq_type_list),
            messages
        )