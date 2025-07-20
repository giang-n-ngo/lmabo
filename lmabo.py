import math
import numpy as np

from bo import (
    bo_single_iteration, 
    bo_constrained_single_iteration,
    fit_gp, 
    calculate_cumulative_regret, 
    bo_constrained_single_iteration,
    get_feasible_candidates,
    acq_type_mapping
)
from llm_helper import ConversationHolder
from utils import get_shortest_distance_from_last_point

INITIAL_PROMPT_CONTENT = """
You are an expert in Bayesian Optimization, specifically tasked with recommending the most suitable acquisition function for the next iteration to minimize an objective function.

For context, we use a Gaussian Process as the surrogate model with a Matern 5/2 kernel with ARD.

I will provide you with a summary of the Bayesian Optimization process at each step. This summary will include the following information:
- **N:** The total number of points evaluated so far.
- **Remaining iterations:** The number of iterations left in the optimization process.
- **D:** The dimensionality of the search space (number of input parameters).
- **f_range:** The range of the objective function values observed so far.
- **f_min:** The current best (lowest) observed objective value.
- **Shortest distance**: The shortest distance from the last point to any other point, indicating whether it is exploiting too much.
- **Model lengthscales:** These are crucial hyperparameters of the Gaussian Process model's kernel. 
They describe how the model perceives the smoothness and relevance of each input dimension to the objective function. 
You will receive their range (min/max), mean, and standard deviation.
- **Model outputscale: ** It defines the overall magnitude or amplitude of the function's variation.

Available acquisition functions you can choose from, with brief descriptions of their primary uses:
1.  PI (Probability of Improvement)
2.  LogPI (Log Probability of Improvement)
3.  EI (Expected Improvement) 
4.  LogEI (Log Expected Improvement) 
5.  UCB (Upper Confidence Bound) 
6.  PosMean (Posterior Mean):** 
7.  PosSTD (Posterior Standard Deviation) 
8.  TS (Thompson Sampling)
9.  qKG (Knowledge Gradient) 
10. qPES (Predictive Entropy Search) 
11. qMES (Max-value Entropy Search)
12. qJES (Joint Entropy Search) 

At each step:
- **Review the provided summary of the optimization process and consider the current state of the optimization.**
- **Select the acquisition function that you believe will be best for the optimization process.**
- **Avoid reusing acquisition functions that failed to improve the objective function in previous iterations.**

When responding, select the acquisition function you deem most appropriate. 
Your justification should briefly explain why that function is suitable given the provided optimization summary, referencing relevant aspects like exploration/exploitation balance, remaining iterations, or model characteristics. 
The response must strictly follow the format "Acquisition abbreviation: justification", similar to these examples:
- 'qKG: This is a good choice because ...'
- 'EI: This is chosen given the current state of the optimization since ...'
Firstly, just give a brief confirmation that you understand the task and the available acquisition functions.
"""

INITIAL_PROMPT_CONTENT_CONSTRAINED = """
You are an expert in Constrained Bayesian Optimization, specifically tasked with recommending the most suitable acquisition function for the next iteration to minimize an objective function while satisfying certain constraints.

For context, we use a Gaussian Process as the surrogate model with a Matern 5/2 kernel with ARD. 
We consider the problem on a discrete set of candidate points in a hypercube in the search space. 
The constraints are also modeled using Gaussian Processes.
At each iteration, we filter the candidate points based on the LCB (Lower Confidence Bound) of the constraints, and only consider feasible points for the objective function evaluation.

I will provide you with a summary of the Constrained Bayesian Optimization process at each step. This summary will include the following information:
- **N:** The total number of points evaluated so far.
- **Remaining iterations:** The number of iterations left in the optimization process.
- **D:** The dimensionality of the search space (number of input parameters).
- **C**: The number of constraints.
- **S**: The number of candidate points in the search space.
- **R**: The number of candidate points that are feasible (i.e., satisfy all constraints).
- **f_range:** The range of the objective function values observed so far.
- **f_min:** The current best (lowest) feasible observed objective value.
- **Shortest distance**: The shortest distance from the last point to any other point, indicating whether it is exploiting too much.
For the main GP and the constraint GPs, you will receive:
- **Model lengthscales:** These are crucial hyperparameters of the Gaussian Process model's kernel. 
They describe how the model perceives the smoothness and relevance of each input dimension to the objective function. 
You will receive their range (min/max), mean, and standard deviation.
- **Model outputscale: ** It defines the overall magnitude or amplitude of the function's variation

Available acquisition functions you can choose from, with brief descriptions of their primary uses:
1.  PI (Probability of Improvement)
2.  LogPI (Log Probability of Improvement)
3.  EI (Expected Improvement) 
4.  LogEI (Log Expected Improvement) 
5.  UCB (Upper Confidence Bound) 
6.  PosMean (Posterior Mean):** 
7.  PosSTD (Posterior Standard Deviation) 
8.  TS (Thompson Sampling)
9.  qKG (Knowledge Gradient) 
10. qPES (Predictive Entropy Search) 
11. qMES (Max-value Entropy Search)
12. qJES (Joint Entropy Search) 

At each step:
- **Review the provided summary of the optimization process and consider the current state of the optimization.**
- **Select the acquisition function that you believe will be best for the optimization process.**
- **Avoid reusing acquisition functions that failed to improve the objective function in previous iterations.**

When responding, select the acquisition function you deem most appropriate. 
Your justification should briefly explain why that function is suitable given the provided optimization summary, referencing relevant aspects like exploration/exploitation balance, remaining iterations, or model characteristics. 
The response must strictly follow the format "Acquisition abbreviation: justification", similar to these examples:
- 'qKG: This is a good choice because ...'
- 'EI: This is chosen given the current state of the optimization since ...'
Firstly, just give a brief confirmation that you understand the task and the available acquisition functions.
"""

FOLLOW_UP_PROMPT_TEMPLATE = """
Current optimization state:
- N: {N} 
- Remaining iterations: {remaining}
- D: {D}
- f_range: Range [{f_min:.3f}, {f_max:.3f}], Mean {f_mean:.3f} (Std Dev {f_std:.3f})
- f_min: {f_min:.3f}
- Shortest distance: {shortest_dist}
- Lengthscales: Range [{min_ls:.3f}, {max_ls:.3f}], Mean {mean_ls:.3f} (Std Dev {std_ls:.3f})
- Outputscale: {outputscale}
"""

FOLLOW_UP_PROMPT_TEMPLATE_CONSTRAINED = """
Current optimization state:
- N: {N}
- Remaining iterations: {remaining}
- D: {D}
- C: {C}
- S: {S}
- R: {R}
- f_range: Range [{f_min:.3f}, {f_max:.3f}], Mean {f_mean:.3f} (Std Dev {f_std:.3f})
- f_min: {f_min:.3f}
- Shortest distance: {shortest_dist}
Main GP statistics:
- Lengthscales: Range [{min_ls:.3f}, {max_ls:.3f}], Mean {mean_ls:.3f} (Std Dev {std_ls:.3f})
- Outputscale: {outputscale}
{constraint_gps_stats_prompt}
"""

CONSTRAINT_GPS_STATS_TEMPLATE = """Constraint GP {i} statistics:
- Lengthscales: Range [{min_ls:.3f}, {max_ls:.3f}], Mean {mean_ls:.3f} (Std Dev {std_ls:.3f})
- Outputscale: {outputscale:.3f}"""

FINAL_GUESS = """
Now that you have finished the optimization process, can you guess which function is this?
"""

class LanguageModelAssistedAdaptiveBO:
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
        self.gp = fit_gp(self.train_X, self.train_Y)
        self.lengthscales = self.gp.covar_module.base_kernel.lengthscale.detach().cpu().numpy()
        self.outputscale = self.gp.covar_module.outputscale.detach().cpu().numpy()
        self.remaining_iterations = self.num_iterations
        self.convo = ConversationHolder(
            llm, 
            first_prompt=INITIAL_PROMPT_CONTENT, 
            full_acq_type_list=list(acq_type_mapping.keys()),
            server_node=server_node,
            default_af="UCB"  # Default acquisition function
        )

    def _construct_prompt(self):
        # --- NEW: Calculate shortest distance of the last point relative to bounds ---
        shortest_dist = get_shortest_distance_from_last_point(self.train_X, self.bounds)
        # --- Calculate descriptive statistics ---
        min_ls = np.min(self.lengthscales)
        max_ls = np.max(self.lengthscales)
        mean_ls = np.mean(self.lengthscales)
        std_ls = np.std(self.lengthscales)

        prompt = FOLLOW_UP_PROMPT_TEMPLATE.format(
            N=self.train_Y.shape[0],
            remaining=self.remaining_iterations,
            D=self.train_X.shape[1],
            f_max=np.round(self.train_Y.max().detach().cpu().numpy(), decimals=3).item(),
            f_mean=np.round(self.train_Y.mean().detach().cpu().numpy(), decimals=3).item(),
            f_std=np.round(self.train_Y.std().detach().cpu().numpy(), decimals=3).item(),
            f_min=np.round(self.train_Y.min().detach().cpu().numpy(), decimals=3).item(),
            shortest_dist=shortest_dist,
            min_ls=min_ls,
            max_ls=max_ls,
            mean_ls=mean_ls,
            std_ls=std_ls,
            outputscale=self.outputscale
        )
        print(f"Iter {len(self.acq_type_list)}|", prompt)
        return prompt

    def optimize(self):
        # Generate initial training data
        for _ in range(self.num_iterations):
            # use LLM to suggest the best acq_type
            acq_type = self.convo.suggest_acq_type(self._construct_prompt())
            if acq_type == "Intentional Incorrect AF":
                exit()
            self.acq_type_list.append(acq_type)
            # run one BO iter with the acq_type suggested by LLM
            self.train_X, self.train_Y, self.gp = bo_single_iteration(
                self.train_X, 
                self.train_Y, 
                acq_type, 
                self.objective_func, 
                self.bounds
            )
            self.lengthscales = self.gp.covar_module.base_kernel.lengthscale.detach().cpu().numpy()
            self.outputscale = self.gp.covar_module.outputscale.detach().cpu().numpy()
            # Store best observed value
            self.best_values.append(self.train_Y.min().item())
            print(f"Current best value: {self.train_Y.min().item()}")
            self.remaining_iterations -= 1
        self.convo.last_guess(FINAL_GUESS)  
        messages = self.convo.messages
        del self.convo # free memory
        return (
            np.array(self.best_values) - self.objective_func._optimal_value, # simple regret
            calculate_cumulative_regret(
                self.train_Y.detach().cpu().numpy(), 
                self.objective_func._optimal_value
            ), # cumulative regret
            np.array(self.train_X.detach().cpu().numpy()), 
            np.array(self.train_Y.detach().cpu().numpy()).flatten(),
            self.acq_type_list,
            messages
        )
    
class LanguageModelAssistedAdaptiveConstrainedBO:
    def __init__(
        self,
        objective_func,
        constraint_func,
        X_init,
        Y_init,
        constraint_init,
        all_candidates,
        bounds,
        num_iterations,
    ):
        self.objective_func = objective_func
        self.constraint_func = constraint_func
        self.train_X  = X_init.clone()
        self.train_Y  = Y_init.clone()
        self.train_constraints = constraint_init.clone()
        self.all_candidates = all_candidates.clone()
        self.bounds = bounds
        self.num_iterations = num_iterations
        self.best_values = [self.train_Y.min().item()]
        self.acq_type_list = []
        # optimization loop
        self.gp = fit_gp(self.train_X, self.train_Y)
        self.constraint_gps = [fit_gp(self.train_X, self.train_constraints[:, i])
            for i in range(self.train_constraints.shape[1])]
        self._extract_scales()  # Extract lengthscales and output scales from the GP models
        self.remaining_iterations = self.num_iterations
        self.convo = ConversationHolder(
            "api", 
            first_prompt=INITIAL_PROMPT_CONTENT, 
            full_acq_type_list=list(acq_type_mapping.keys()),
            server_node=""
        )

    def _get_gp_stats_prompt(self):
        """Constructs a string with the statistics of the constraint GPs."""
        stats = []
        for i, gp in enumerate(self.constraint_gps):
            lengthscales = gp.covar_module.base_kernel.lengthscale.detach().cpu().numpy()
            outputscale = gp.covar_module.outputscale.detach().cpu().numpy()
            min_ls = np.min(lengthscales)
            max_ls = np.max(lengthscales)
            mean_ls = np.mean(lengthscales)
            std_ls = np.std(lengthscales)
            stats.append(CONSTRAINT_GPS_STATS_TEMPLATE.format(
                i=i+1, 
                min_ls=min_ls, 
                max_ls=max_ls, 
                mean_ls=mean_ls, 
                std_ls=std_ls, 
                outputscale=outputscale
            ))
        return "\n".join(stats)

    def _construct_prompt(self):
        # --- NEW: Calculate shortest distance of the last point relative to bounds ---
        shortest_dist = get_shortest_distance_from_last_point(self.train_X, self.bounds)
        # --- Calculate descriptive statistics ---
        min_ls = np.min(self.main_lengthscales)
        max_ls = np.max(self.main_lengthscales)
        mean_ls = np.mean(self.main_lengthscales)
        std_ls = np.std(self.main_lengthscales)

        constraint_gps_stats_prompt = self._get_gp_stats_prompt()

        prompt = FOLLOW_UP_PROMPT_TEMPLATE_CONSTRAINED.format(
            N=self.train_Y.shape[0],
            remaining=self.remaining_iterations,
            D=self.train_X.shape[1],
            C=self.train_constraints.shape[1],
            S=self.all_candidates.shape[0],
            R=self.R,
            f_max=np.round(self.train_Y.max().detach().cpu().numpy(), decimals=3).item(),
            f_mean=np.round(self.train_Y.mean().detach().cpu().numpy(), decimals=3).item(),
            f_std=np.round(self.train_Y.std().detach().cpu().numpy(), decimals=3).item(),
            f_min=np.round(self.train_Y.min().detach().cpu().numpy(), decimals=3).item(),
            shortest_dist=shortest_dist,
            min_ls=min_ls,
            max_ls=max_ls,
            mean_ls=mean_ls,
            std_ls=std_ls,
            outputscale=self.main_outputscale,
            constraint_gps_stats_prompt=constraint_gps_stats_prompt
        )
        print(f"Iter {len(self.acq_type_list)}|", prompt)
        return prompt

    def _extract_scales(self):
        """Extracts lengthscales from the Gaussian Process models."""
        self.main_lengthscales = self.gp.covar_module.base_kernel.lengthscale.detach().cpu().numpy()
        self.main_outputscale = self.gp.covar_module.outputscale.detach().cpu().numpy()
        self.constraint_lengthscales = [
            gp.covar_module.base_kernel.lengthscale.detach().cpu().numpy()
            for gp in self.constraint_gps
        ]
        self.constraint_outputscales = [
            gp.covar_module.outputscale.detach().cpu().numpy()
            for gp in self.constraint_gps
        ]
    
    def optimize(self):
        feasible_candidates, sampling_flag = get_feasible_candidates(
            constraint_gps=self.constraint_gps,  # No initial constraints GP
            all_candidates=self.all_candidates,
            beta_t_sqrt=(10*math.log(self.train_Y.shape[0]))**0.5,
            return_sampling_flag=True
        )
        if sampling_flag:
            self.R = 0
        else:
            self.R = feasible_candidates.shape[0]
        for iteration_idx in range(self.num_iterations):
            # use LLM to suggest the best acq_type
            acq_type = self.convo.suggest_acq_type(self._construct_prompt())
            if acq_type == "Intentional Incorrect AF":
                exit()
            self.acq_type_list.append(acq_type)
            # run one BO iter with the acq_type suggested by LLM
            self.train_X, self.train_Y, self.train_constraints, self.gp, self.constraint_gps = \
                bo_constrained_single_iteration(
                    self.objective_func,
                    self.constraint_func,
                    acq_type,
                    self.bounds,
                    self.train_X, 
                    self.train_Y, 
                    self.train_constraints,
                    feasible_candidates,
                    self.gp
                )
            self._extract_scales()
            feasible_candidates, sampling_flag = get_feasible_candidates(
                constraint_gps=self.constraint_gps,  # No initial constraints GP
                all_candidates=self.all_candidates,
                beta_t_sqrt=(10*math.log(self.train_Y.shape[0]))**0.5,
                return_sampling_flag=True
            )
            if sampling_flag:
                self.R = 0
            else:
                self.R = feasible_candidates.shape[0]
            # Store best observed value
            if (self.train_constraints >= 0).all(dim=1).squeeze(-1).any():
                best_feasible_value = self.train_Y[(self.train_constraints >= 0).all(dim=1).squeeze(-1)].min().item()
            else:
                best_feasible_value = "Not found"
            print(f"Iter {iteration_idx} | Current best feasible value: {best_feasible_value}")
            self.remaining_iterations -= 1
        self.convo.last_guess(FINAL_GUESS)  
        messages = self.convo.messages
        del self.convo
        return (
            np.array(self.train_X.detach().cpu().numpy()), 
            np.array(self.train_Y.detach().cpu().numpy()).flatten(),
            np.array(self.train_constraints.detach().cpu().numpy()),
            self.acq_type_list,
            messages
        )