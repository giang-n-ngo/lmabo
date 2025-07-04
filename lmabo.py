import numpy as np

from bo import bo_single_iteration, fit_gp, calculate_cumulative_regret, acq_type_mapping
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

When responding, select the acquisition function you deem most appropriate. 
Your justification should briefly explain why that function is suitable given the provided optimization summary, referencing relevant aspects like exploration/exploitation balance, remaining iterations, or model characteristics. 
The response should be in the format "Acquisition abbreviation: justification", similar to these examples:
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
            server_node=server_node
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